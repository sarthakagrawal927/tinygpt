# North Star — refined (2026-05-31)

This doc supersedes the rough Tier 5 sketch with a sharper product
vision. The project is now:

> **An on-device specialist SLM (1-3B) that reads the Mac screen, calls
> local tools, and gracefully escalates to a cloud model when it
> doesn't know. Local model is the always-on layer; cloud is the
> safety net. Multiple specialists are loaded via a tiny router. Works
> across languages so it serves India + global users.**

That's a product. Every piece below is grounded in user-stated vision.

## Architecture

```
                    ┌──────────────────────────┐
                    │  User input (text, screen) │
                    └──────────────┬─────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      Tiny ROUTER (~100M)     │
                    │  classifies task; chooses    │
                    │  specialist or cloud         │
                    └──────────────┬──────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
   ┌────▼─────┐              ┌─────▼─────┐              ┌─────▼─────┐
   │ debugger │              │  SQL/code │              │ defer to  │
   │ 1-3B     │              │   1-3B    │              │  cloud    │
   │ specialist│              │ specialist│              │ (Claude   │
   └────┬─────┘              └─────┬─────┘              │  / GPT)   │
        │                          │                    └────┬──────┘
        │  reads screen via macOS  │                         │
        │  ScreenCaptureKit + AX   │                         │
        │  tree                    │                         │
        └────────────┬─────────────┘                         │
                     │                                       │
        ┌────────────▼────────────┐                          │
        │  Tools: file I/O, exec, │                          │
        │  test runner, search    │                          │
        │  (subprocess sandbox)   │                          │
        └─────────────────────────┘                          │
                                                             │
        ┌────────────────────────────────────────────────────┘
        │
        ▼ when complex / out-of-distribution / requires world knowledge
   ┌──────────────────┐
   │ Cloud model      │
   │ Claude / GPT API │
   └──────────────────┘
```

## Why this architecture is right

**vs pure on-device** (Apple Intelligence, Phi-4-mini chat):
- Fails on hard tasks (limited capacity at 1-3B)
- We escalate when uncertain → users get correct answers always

**vs pure cloud** (Claude, GPT, Cursor):
- Latency (every call is 200-500ms round-trip)
- Cost ($0.50-$2/M tokens × frequent agent loops = expensive)
- Privacy (screen + files sent to cloud every turn)
- We handle 80%+ locally → faster, cheaper, more private

**vs single big general model**:
- A 70B can do everything but uses ~40 GB at int4 + 100W
- N small specialists at 1-3B each, only one loaded at a time
- Mixture-of-Specialists with router → smaller resident footprint
- Each specialist tuned for its task → outperforms generalist on that task

## Concrete capabilities the agent needs

| Capability | Status | Effort |
|---|---|---|
| **Tool calling** (structured JSON output) | ✅ JSON-mode constrained gen + agent runtime |  |
| **Multi-turn conversation** | ✅ AgentLoop with persistent KV |  |
| **Tool execution sandbox** | ✅ ToolExecutor subprocess + timeout |  |
| **Local model fine-tuning** | ✅ SFT/DPO/SimPO + LoRA/DoRA |  |
| **Specialist registry** | 🟡 Have HF + GitHub data; need per-specialist training pipeline | 1 week |
| **Screen reading — capture** | ⬜ ScreenCaptureKit (macOS Sonoma+) integration | 3-5 days |
| **Screen reading — UI tree** | ⬜ macOS Accessibility (AX) APIs | 3-5 days |
| **Screen reading — vision model** | ⬜ Small ViT encoder → tinygpt decoder | 2 weeks |
| **Cloud-escalation training** | ⬜ Train model to emit `{"defer_to_cloud": ...}` when uncertain | 1 week |
| **Cloud API client** | ✅ curl-shellout for Anthropic + OpenAI (commit ef0e5e3) |  |
| **SSE streaming on serve** | ✅ OpenAI-compat per-token streaming + cancellation (e754d6c, c11265b) |  |
| **Tool-call extractor (mini-router)** | ⬜ Tiny encoder (~30-100M, BERT-class) → classification head over the tool catalog. Pre-step before the full LM forward pass: given the user query + active tool catalog, picks which tool (and rough arg slots). Way cheaper than letting the full LM hallucinate the choice — runs in <5ms even on CPU, no KV cache needed. A focused special case of the broader Router model. | 1 week |
| **Router model** | ⬜ Tiny (~100M) classifier-style router for specialist selection (debugger vs shell vs SQL vs cloud). Sibling to the tool-call extractor, but routes between models instead of tools. | 1-2 weeks |
| **Multilingual** | 🟡 Tokenizer-side OK (smollm2 / Qwen3 vocabs); need eval | 1 week |

## Critical research dives needed

### Thinking Machines Lab (Mira Murati, et al.)

User specifically called out — TML is positioning around "interaction
models." Known public artifacts as of cutoff:
- **Tinker** (2025): fine-tuning API for open-weight models
- Focus on human-in-the-loop systems + multimodal interaction
- Founded by Mira Murati after leaving OpenAI

Action: spawn Explore agent to summarize TML's published technical
direction + identify what we should learn from / not duplicate.

### Apple Foundation Models

WWDC 2024+ announcements about on-device 3B model + cloud
extension via "Private Cloud Compute." This is the SAME architecture
we're proposing. Need to:
- Understand how Apple's local model handles tool calling
- Identify whether ours can hook into their ecosystem (App Intents?)
- Or whether we're a parallel/competitive offering

### Cursor / Continue.dev / Cline / Codeium

These are existing code agents (cloud-bound). Our differentiation:
on-device specialist that handles 80% locally + escalates. What does
the architecture look like — separate eval needed.

## Demonstration specialists to ship

In order of probable demo value:

1. **Code debugger** — issue → diagnosis → fix proposal
   - Training data: GitHub issue→PR pairs (we have `tinygpt fetch-github`)
   - Tools: read_file, run_test, search_code, edit_file
   - Eval: SWE-bench scaled-down + custom corpus

2. **Shell command writer** — natural language → bash command
   - Training data: history dumps + man pages + StackOverflow Q&A
   - Tools: minimal (it's mostly text-out)
   - Eval: custom corpus + manual review

3. **SQL writer** — schema + question → query
   - Training data: spider, BIRD, gretelai/synthetic_text_to_sql
   - Tools: schema inspector + query runner
   - Eval: spider/BIRD exact-match

4. **Screen reader / UI explainer** — "what is this app showing me?"
   - Training data: SYNTHETIC — screenshot + caption pairs generated
     by Claude/GPT on a corpus of macOS app screenshots
   - Tools: capture_screen, click, type
   - Eval: custom screenshot corpus

5. **Indian-context assistant** (multilingual)
   - Hindi + English specialist
   - Training data: desi-max base + curated Hindi instruction sets
   - Tools: same as general assistant
   - Eval: Hindi-language LLM benchmarks (where they exist)

## Realtime / interaction model (NEW — user-stated)

Beyond turn-based agent, the user wants a **realtime interaction model**
— continuous, low-latency, streaming, interruptible. Think GPT-4o
realtime API or Anthropic's streaming voice-to-voice, but on-device.

What "realtime" means concretely:

| Property | Target | Current state |
|---|---|---|
| TTFT (time-to-first-token) | < 100ms cold, < 50ms warm | Cold-start work landed (1.80s→0.10s on demo; mmap + lazy embed + async + Metal warmup); warm probably 50-100ms on huge preset |
| Per-token latency (decode jitter) | < 30ms p99 | 470 tok/s = 2ms median; spec decode + KIVI keep this; jitter measurement unmeasured |
| Streaming output | token-by-token to client | Already streams via `tinygpt sample` stdout; `tinygpt serve` /v1/chat/completions is non-streaming today — needs SSE wrapper |
| Interrupt mid-stream | user cancels generation, model stops within 1 step | Not wired; needs cancellation token through AgentLoop |
| Continuous conversation | no clear turn boundaries; new input can arrive while model is generating | Not wired; persistent-KV cache makes this possible |
| Audio in/out (eventually) | speech-to-text + text-to-speech | Not in scope; Apple's Speech.framework + AVSpeechSynthesizer are the natural local choices |

### What needs to be built

| Item | Effort |
|---|---|
| **SSE streaming on serve endpoint** — `/v1/chat/completions` with `stream: true` honored | 2 days |
| **Cancellation token** through Sample.swift's decode loop | 1 day |
| **Interrupt handling in AgentLoop** — user input mid-generation cancels current turn | 2 days |
| **Decode jitter benchmark** — measure p50/p95/p99 ITL, identify spikes | 1 day |
| **Cold-start audit follow-up** — squeeze warm TTFT to < 50ms | 3 days |
| **Async tool-call dispatch** — start the tool execution while still streaming the call argument tokens | 3 days |
| **Audio I/O bridge** (optional, eventually) | 1 week |

The realtime work is **complementary to the agent factory** — same
infrastructure, different surface (streaming + interrupt instead of
turn-based + transcript). Pairs naturally with screen reading
(realtime "watch what the user does, react when asked") for the
"interaction model" framing.

## Updated wave plan

```
Wave 1 (DONE):          agent runtime + JSON mode + HF data
Wave 2 (DONE/partial):  GitHub data + eval harness (eval deferred —
                        subprocess refactor needed)
Wave 2.5 (in progress): CPU + GPU + ANE + browser optimizations
                        - LOW risk: CF R2 + Pausable training + GPU lock (DONE)
                        - HIGH risk: Metal kernels, ANE routing
                          (agent pool degraded — re-attempt later)
Wave 2.6 (NEW):         screen reading + cloud escalation + router
                        infrastructure
                        - ScreenCaptureKit + AX integration
                        - Cloud API client (Claude/OpenAI HTTP)
                        - Router model architecture + training pipeline
                        - Vision encoder (ViT) → text decoder bridge
Wave 3 (DEFERRED until 2.5 + 2.6 mostly done):
                        - First specialist (debugger) — train + ship
                        - Second specialist (shell or SQL)
                        - Demo on the web playground
Wave 4 (research):      - TML / Apple / Cursor research dives
                        - Mixture-of-Specialists router actually wired
                        - Multilingual specialist (Hindi)
                        - Public model card on HF Hub
                        - Writeup / blog / HN post
```

## Roadmap items by user-stated motivation

| User said | Maps to |
|---|---|
| "best at tool calling inside the Mac" | Tier 5.1 reasoning training + JSON mode + agent runtime |
| "reading from the screen" | Wave 2.6 screen reading bundle |
| "calling a larger model when needed" | Wave 2.6 cloud escalation + router |
| "GVT to real time" | Streaming sample + cold-start (already shipped) |
| "very efficient... build for India" | Wave 4 multilingual specialist + Mixture-of-Specialists |
| "smaller models working together... router on top" | Mixture-of-Specialists architecture |
| "fine-tune models like desi-max" | Already supported by SFT/DPO + HF loader |

Nothing in this vision conflicts with what's been built. Everything
lands cleanly on the existing infrastructure. The remaining work is
roughly:
- 2-3 weeks of Wave 2.6 (screen reading + cloud escalation + router)
- 4-6 weeks of Wave 3 (first 1-2 specialists)
- Then Wave 4 polish / publication

## Honest reality checks

1. **Wave 2.5 Metal kernels are harder than agent prompts can solve.**
   We just had 4 parallel agent spawns stall at the "I'll start by..."
   stage. Custom Metal kernel work probably needs more focused +
   step-by-step prompting, possibly done over multiple short sessions
   with explicit handoff between them.

2. **The screen reader is the biggest scope risk.** ScreenCaptureKit
   is well-documented; ViT integration is well-trodden; but training
   a 1-3B model to actually USE screenshots for tool calling is
   research-grade work. May land as a toy first.

3. **The cloud escalation training signal is non-trivial.** You need
   a corpus where the same prompts have BOTH local-resolvable AND
   cloud-only answers, and the model learns to predict "defer". Could
   bootstrap via Claude/GPT-as-judge.

4. **The router is a small model itself.** ~100M classifier-style. We
   have the infrastructure but no training pipeline tuned for the
   "which specialist handles this?" classification task. The
   **tool-call extractor** is a smaller sibling — encoder-only,
   classification over a fixed tool catalog. It's easier (no
   generation, no KV cache, fits comfortably in CPU) and a good
   warm-up project before the bigger router. Possible
   training signal: scrape tool-using agent traces (from Anthropic /
   OpenAI tool-use datasets, BFCL, τ-bench), pair (query, tool_name)
   as supervised examples, train a tiny encoder + softmax head. Could
   plausibly be a sub-30M model that beats LM-emitted tool calls on
   latency by 50-100x.

5. **Compute budget reality.** Training a 1.5B specialist properly is
   24-48 GPU hours. On a Mac with the existing infra, that's
   feasible but uses cloud (CF R2 just shipped — unblocks this).

## What I'd ship next if doing this solo

1. Land remaining Wave 2.5 Metal kernels (or get honest negative
   results) — re-spawn agents in shorter sessions
2. Wave 2.6 ScreenCaptureKit + Cloud API client (both ~3-5 days each,
   manual implementation; agent pool not reliable enough)
3. Train first specialist end-to-end (debugger) — Wave 3
4. Wire it into a Mac app demo
5. Public release + writeup

Everything above is buildable. The hard part is sequencing patience
+ honesty about which agent jobs will actually land cleanly.
