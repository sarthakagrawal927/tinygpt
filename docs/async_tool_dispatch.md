---
title: Async tool-call dispatch — investigated, skipped
description: Why async tool dispatch in AgentLoop doesn't justify the work at current bottleneck shape. Logged so the decision doesn't get re-litigated.
---

# Async tool-call dispatch — investigated, skipped

**Date**: 2026-05-31
**Verdict**: SKIP at current bottleneck shape. Revisit when a real
multi-tool specialist ships.

## What was investigated

AgentLoop today (`native-mac/Sources/TinyGPT/AgentLoop.swift:300`)
dispatches tools synchronously inline:

```swift
let toolResult = runToolCall(name: toolName, arguments: args)  // BLOCKS
let resultJSON = encodeToolResult(toolName, result: toolResult)
feedText(toolResultPreface + resultJSON + toolResultSuffix)     // continues
continue   // → next LM turn
```

The original Wave 2.6 idea was "start subprocess while LM is still
streaming tokens, overlap the wait." Concretely two flavors:

1. **Streaming overlap**: dispatch subprocess immediately when JSON
   args are complete, before the LM finishes the trailing `<|im_end|>`.
2. **Inter-tool overlap**: when the model emits multiple tool calls
   in sequence (or in parallel via a future tool_calls array), run
   them concurrently.

## Why neither pays off right now

**Decode latency is the bottleneck, not subprocess.** From
`docs/research/mac_decode_baseline_m5pro.md`:

- ITL on M5 Pro is **1.3–4.9 ms** across model sizes
- A 128-token turn = ~200–700 ms of LM time
- Average tool subprocess (read_file, run_test) on local SSD: **5–100 ms**

The LM dominates by 5–100×. Overlapping a 50ms subprocess with a
500ms generation saves at most 10% wall time. Not a perceptible
UX win for a single-user CLI agent.

**Flavor #1 (streaming overlap) saves only the trailing tokens
between args-complete and `<|im_end|>`** — typically 2-5 tokens =
**~10ms**. Not worth the synchronization complexity.

**Flavor #2 (inter-tool overlap)** would matter for a specialist
that emits parallel tool calls (`tool_calls: [{...}, {...}]` —
Apple's `Tool` protocol shape supports this). tinygpt's current
prompt template asks for ONE tool per JSON object, looping
serially. No specialist exists yet that emits parallel calls.

## When this becomes worth doing

Three triggers that flip the verdict:

1. **A specialist ships that emits parallel `tool_calls`** — at that
   point flavor #2 saves real wall-time. The Apple-style
   `tool_calls: [...]` array is the shape to support.
2. **Tool execution dominates a meaningful fraction of turns** —
   e.g., a debugger specialist whose `run_test` calls take 5–30s.
   Then async dispatch + a UI progress hook ("running tests…")
   matters for UX even if not for raw throughput.
3. **The tool-call extractor (mini-router) lands** and we want to
   start dispatch the moment the router predicts a tool, BEFORE the
   LM has finished emitting the JSON. Saves the full LM turn cost
   (200–700 ms). High ROI but blocked on the router being trained
   and the LM being aware of the router's prediction.

## What would the minimum implementation look like

When the time comes:

```swift
// AgentLoop additions:
private let toolQueue = DispatchQueue(label: "tinygpt.agent.tool",
                                       attributes: .concurrent)

// In runTurn(), when JSON parsed:
if let toolName = obj["tool"] as? String {
    let args = (obj["arguments"] as? [String: Any]) ?? [:]
    let resultPromise = DispatchSemaphore(value: 0)
    var result: ToolExecutor.Result?
    toolQueue.async {
        result = self.runToolCall(name: toolName, arguments: args)
        resultPromise.signal()
    }
    // Stream "calling X..." event immediately for UX
    emitJSONEvent(["type": "tool_dispatch", "tool": toolName])
    // Block when we actually need the result for next LM turn
    resultPromise.wait()
    // ... rest as before
}
```

That's it for flavor #1. Flavor #2 requires the LM to emit
`tool_calls: [{...}, {...}]` arrays + a `runToolBatch(...)`
that dispatches all simultaneously.

## Why this gets logged instead of built

Per the project's ROI-driven scope rule: a 5–10% wall-time win
that's contingent on infrastructure (parallel tool emission,
trained router) that hasn't shipped yet isn't worth multi-day work
right now. Logging the decision so it doesn't get re-investigated.

The actual measured bottleneck (LM ITL) means **the way to make
the agent faster is faster decode, not async dispatch**. Speculative
decoding (Medusa/EAGLE-2 — already shipped, gated on trained heads)
is the right next lever when the time comes.

## Related research

- Tool-call extractor scaffold (Wave 2.6, b5bbdd9) — when the
  router predicts a tool with confidence > threshold, we could
  dispatch BEFORE the LM finishes its JSON. The router's
  prediction latency is < 20ms warm (mini-router design).
- Apple `Tool` protocol shape (`docs/research/wave_4_landscape.md`
  §2) — parallel/serial tool-call graph support, the right
  abstraction when we add it.
