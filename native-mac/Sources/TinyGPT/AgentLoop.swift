import Foundation
import MLX
import MLXNN
import MLXRandom
import TinyGPTModel
import TinyGPTServe

/// The agent conversation loop. Owns:
///   - the loaded model
///   - a tokenizer (HF BPE if the model pins one, otherwise byte-level)
///   - a KV cache that grows across turns within a single session
///   - a tool schema + executor for dispatching tool calls
///
/// # Lifecycle
///
///  1. `prefillSystemPrompt(...)` runs once at startup. It feeds the
///     system prompt + tool descriptions through `forwardCached` so the
///     cache holds the prefix at the entry to the first user turn. If a
///     persistent prompt cache was passed in, this hits the fast path
///     and skips prefill.
///
///  2. For each user turn the loop:
///       a. Appends the user message text to the cache via `forwardCached`.
///       b. Generates up to `maxTokens` tokens, decoding as it goes.
///       c. As tokens stream out we look for a complete JSON object in
///          the decoded text. The model is told via the system prompt to
///          emit one of two shapes:
///                { "tool": <name>, "arguments": {...} }
///                { "answer": <text> }
///          so a balanced top-level JSON object is the natural turn
///          boundary. We bail out of generation as soon as we have one.
///       d. If the JSON is a tool call, execute it (subprocess), format
///          the result as `<tool_result>{...}</tool_result>`, append THAT
///          to the cache, and loop back to (b). If it's an answer (or
///          we couldn't parse JSON after maxTokens), return to the caller.
///
/// # Limits we know about
///
///  - Context cap: when the cache hits `cfg.contextLength` we stop. No
///    eviction. A long agent run with many turns will eventually overflow;
///    we report it honestly and the next call has to start fresh.
///  - Sampling: temperature is configurable but defaults to 0 (greedy).
///    Agents want determinism; chat models want creativity. This is the
///    default for tool-calling.
///  - JSON robustness: an unspecialized base model will emit malformed
///    JSON often. We log + degrade to "treat the raw text as a final
///    answer" instead of crashing the loop.
public final class AgentLoop {

    // Owned state.
    private let model: AnyModel
    private let cfg: ModelConfig
    private let tokenizer: HFTokenizer?
    private let schema: ToolSchema
    private var cache: KVCache
    private let temperature: Float
    private let maxTokensPerTurn: Int
    private let toolTimeoutSec: Double
    private let transcriptURL: URL?
    private let jsonOut: Bool
    private let maxAgentSteps: Int

    // Cloud-escalation: when set, an "escalate" tool is added to the
    // schema and dispatched against the configured remote provider.
    // The cloud's response comes back into the conversation as a
    // tool_result, so the agent can decide what to do with it (often
    // just relay it as the final answer).
    private let cloudEscalateProvider: CloudEscalate.Provider?
    private let cloudEscalateModel: String?

    // Tool-call extractor (mini-router) — Wave 2.6 scaffold.
    //
    // Carrying state for the optional pre-step that runs BEFORE every
    // LM tool-call generation. When set, `runTurn` queries the router
    // with the user's text; if the top-class softmax exceeds the
    // threshold the router's pick is logged + can be fed downstream as
    // a constrained-decode hint. The full constraint-injection path
    // (steering the JSON schema FSM to one tool name) is left as a
    // Follow-up: see docs/tool_call_extractor.md §"Integration with
    // ConstrainedGen" — because the FSM doesn't currently expose a
    // "pin one tool" entrypoint. Today the loop simply records the
    // router's prediction in the transcript so downstream consumers
    // (a future MCP-style supervisor) can decide what to do.
    public struct RouterHook {
        public let router: ToolRouterModel
        public let labels: [String]
        public let threshold: Float
        public init(router: ToolRouterModel, labels: [String], threshold: Float) {
            self.router = router
            self.labels = labels
            self.threshold = threshold
        }
    }
    private let routerHook: RouterHook?

    /// Token boundary the agent expects between turns. We use a ChatML-ish
    /// scheme so models trained with that template have a fighting chance
    /// of producing the right shape. For byte-level models this is just
    /// plain ASCII that we look for in the decoded stream.
    private let userPreface = "<|im_start|>user\n"
    private let userSuffix = "<|im_end|>\n<|im_start|>assistant\n"
    private let toolResultPreface = "<|im_start|>tool\n"
    private let toolResultSuffix = "<|im_end|>\n<|im_start|>assistant\n"
    private let assistantTerminator = "<|im_end|>"

    public init(model: AnyModel, cfg: ModelConfig, tokenizer: HFTokenizer?,
                schema: ToolSchema, cache: KVCache,
                temperature: Float = 0.0, maxTokensPerTurn: Int = 256,
                toolTimeoutSec: Double = 30.0,
                transcriptURL: URL? = nil, jsonOut: Bool = false,
                maxAgentSteps: Int = 8,
                cloudEscalateProvider: CloudEscalate.Provider? = nil,
                cloudEscalateModel: String? = nil,
                routerHook: RouterHook? = nil)
    {
        self.routerHook = routerHook
        self.model = model
        self.cfg = cfg
        self.tokenizer = tokenizer
        // When escalation is on, add a synthetic "escalate" tool. The
        // model only sees it via the systemPromptDescription; the
        // executor route is intercepted in `runToolCall` so the tool
        // never goes through subprocess.
        self.schema = cloudEscalateProvider != nil
            ? AgentLoop.augmentWithEscalateTool(base: schema)
            : schema
        self.cache = cache
        self.temperature = temperature
        self.maxTokensPerTurn = maxTokensPerTurn
        self.toolTimeoutSec = toolTimeoutSec
        self.transcriptURL = transcriptURL
        self.jsonOut = jsonOut
        self.maxAgentSteps = maxAgentSteps
        self.cloudEscalateProvider = cloudEscalateProvider
        self.cloudEscalateModel = cloudEscalateModel
    }

    /// Build a synthetic `escalate` tool definition and append it to the
    /// user's schema. The tool has no `_exec` — it's dispatched in
    /// `runToolCall` directly into `CloudEscalate.complete(...)`.
    private static func augmentWithEscalateTool(base: ToolSchema) -> ToolSchema {
        // Skip if the user already declared one — don't shadow their
        // version (they might have a custom escalation policy).
        if base.tools.contains(where: { $0.name == "escalate" }) { return base }
        let escalate = ToolSchema.Tool(
            name: "escalate",
            description: "Call this when you don't know the answer or the question requires a stronger general-purpose model. Pass the user's question verbatim as `question`. The response is the cloud model's full answer — relay it back to the user.",
            parameters: ToolSchema.ParameterSpec(
                type: "object",
                properties: [
                    "question": ToolSchema.PropertySpec(
                        type: "string",
                        description: "The question to forward to the cloud model.")
                ],
                required: ["question"],
                raw: [
                    "type": "object",
                    "properties": [
                        "question": [
                            "type": "string",
                            "description": "The question to forward to the cloud model."
                        ] as [String: Any]
                    ] as [String: Any],
                    "required": ["question"]
                ] as [String: Any]
            ),
            exec: nil,
            execArgs: nil,
            handler: "cloud_escalate"
        )
        return ToolSchema(tools: base.tools + [escalate])
    }

    // MARK: - Prefill

    /// Build the system prompt as a single string. The exact wording
    /// matters less than the structural commitments: tool descriptions
    /// up top, then the two JSON shapes the model can emit.
    public static func systemPrompt(userSystem: String, schema: ToolSchema) -> String {
        let toolList = schema.systemPromptDescription()
        let header = userSystem.isEmpty
            ? "You are a specialized on-device agent."
            : userSystem
        return """
        <|im_start|>system
        \(header)

        You have access to these tools:
        <tools>
        \(toolList)
        </tools>

        When you need to use a tool, respond with a SINGLE valid JSON object on its own line:
        { "tool": "<name>", "arguments": { ... } }

        After the tool runs you will receive its output between <tool_result>...</tool_result>.
        Use as many tool calls as you need.

        When you have the final answer, respond with a SINGLE valid JSON object:
        { "answer": "..." }

        Do not include any other text outside the JSON object.<|im_end|>
        """
    }

    /// Run the prompt through the cache so subsequent turns can start
    /// from `cache.currentLength == prefixLen`. Returns the token count.
    @discardableResult
    public func prefillSystemPrompt(_ text: String,
                                     alreadyPrefilled: Bool = false) -> Int
    {
        // If the cache was loaded from disk for this prompt, currentLength
        // is already populated. Caller signals via `alreadyPrefilled`.
        if alreadyPrefilled {
            // Track last token so the rewind/recover path in
            // `generateUntilJSONOrLimit` has a valid id to feed back.
            // We don't know the actual id from a disk-loaded cache, so
            // we encode the prompt just to recover the tail.
            let ids = encode(text)
            if let last = ids.last { lastFedToken = last }
            return cache.currentLength
        }
        var ids = encode(text)
        if ids.isEmpty { return 0 }
        // Don't try to push more tokens than the model can attend to.
        // We leave a small budget for the first user turn — without it
        // every byte-level model with ctx=256 would crash on the first
        // forward. Real agent models (ctx ≥ 2048) won't hit this.
        let prefillCap = max(8, cfg.contextLength - 64)
        if ids.count > prefillCap {
            fputs("warning: system prompt (\(ids.count) tokens) > context (\(cfg.contextLength)) — truncating head. The model will likely produce gibberish at this context size.\n", stderr)
            ids = Array(ids.suffix(prefillCap))
        }
        lastFedToken = ids.last!
        let arr = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        _ = model.forwardCached(arr, cache: cache)
        return ids.count
    }

    // MARK: - Turn

    /// One user turn. May fire multiple model generations + tool calls
    /// internally. Returns the model's final-answer text (or the raw
    /// text we fell back to when JSON parsing failed).
    public func runTurn(userText: String) -> String {
        recordEvent(event: ["type": "user", "text": userText])
        // Tool-call extractor pre-step. Runs BEFORE the LM forward
        // pass — predicts the most likely tool from the user query
        // alone (no system prompt, no tool catalog). High-confidence
        // picks are logged + made visible to downstream consumers via
        // the transcript. Constrained-decode injection (steering the
        // JSON FSM to a single tool name) is a follow-up.
        if let hook = routerHook {
            let pred = predictWithRouter(hook: hook, query: userText)
            recordEvent(event: [
                "type": "router_prediction",
                "tool": pred.tool,
                "prob": pred.prob,
                "fired": pred.prob >= hook.threshold,
            ])
            if jsonOut {
                emitJSONEvent([
                    "type": "router_prediction",
                    "tool": pred.tool,
                    "prob": pred.prob,
                    "fired": pred.prob >= hook.threshold,
                ])
            }
        }
        // Append the user message to the cache.
        feedText(userPreface + userText + userSuffix)
        var lastAnswer: String? = nil

        for step in 0..<maxAgentSteps {
            if TrainSupport.stopRequested.isSet { break }
            let result = generateUntilJSONOrLimit()
            // Trim any closing ChatML tag the model emitted around the
            // JSON so the parser doesn't have to deal with it.
            let trimmed = stripChatML(result)

            if jsonOut {
                emitJSONEvent(["type": "assistant", "step": step, "raw": trimmed])
            } else {
                print(trimmed)
            }
            recordEvent(event: ["type": "assistant_raw", "step": step, "text": trimmed])

            // Parse JSON. On failure we treat the raw text as a final
            // answer — the unspecialized-base-model degradation path.
            if let obj = extractJSONObject(trimmed) {
                if let answer = obj["answer"] as? String {
                    recordEvent(event: ["type": "answer", "text": answer])
                    if jsonOut {
                        emitJSONEvent(["type": "answer", "text": answer])
                    }
                    return answer
                }
                if let toolName = obj["tool"] as? String {
                    let args = (obj["arguments"] as? [String: Any]) ?? [:]
                    let toolResult = runToolCall(name: toolName, arguments: args)
                    if jsonOut {
                        emitJSONEvent([
                            "type": "tool_call",
                            "tool": toolName,
                            "arguments": args,
                            "stdout": toolResult.stdout,
                            "stderr": toolResult.stderr,
                            "exit_code": Int(toolResult.exitCode),
                            "duration_sec": toolResult.durationSec,
                        ])
                    }
                    recordEvent(event: [
                        "type": "tool_result",
                        "tool": toolName,
                        "stdout": toolResult.stdout,
                        "stderr": toolResult.stderr,
                        "exit_code": Int(toolResult.exitCode),
                    ])
                    let resultJSON = encodeToolResult(toolName, result: toolResult)
                    feedText(toolResultPreface + resultJSON + toolResultSuffix)
                    continue
                }
                // JSON but neither shape we expect. Treat as freeform.
                lastAnswer = trimmed
                break
            } else {
                // No parseable JSON — degrade.
                lastAnswer = trimmed
                break
            }
        }
        let final = lastAnswer ?? "(no answer)"
        recordEvent(event: ["type": "answer", "text": final, "fallback": true])
        if jsonOut {
            emitJSONEvent(["type": "answer", "text": final, "fallback": true])
        }
        return final
    }

    private func runToolCall(name: String, arguments: [String: Any]) -> ToolExecutor.Result {
        // Cloud-escalation hook — intercept BEFORE we hand off to the
        // subprocess executor. The "escalate" tool name is reserved when
        // `--cloud-escalate` is on. We resolve the named provider and
        // call CloudEscalate.complete with the model-provided question.
        if let provider = cloudEscalateProvider, name == "escalate" {
            let t0 = Date()
            let question = (arguments["question"] as? String) ?? ""
            if question.isEmpty {
                return ToolExecutor.Result(
                    stdout: "",
                    stderr: "escalate: missing 'question' argument",
                    exitCode: 2,
                    durationSec: -t0.timeIntervalSinceNow)
            }
            do {
                let text = try CloudEscalate.complete(
                    provider: provider,
                    model: cloudEscalateModel,
                    messages: [CloudEscalate.Message(role: "user", content: question)],
                    maxTokens: 1024)
                return ToolExecutor.Result(
                    stdout: text, stderr: "",
                    exitCode: 0,
                    durationSec: -t0.timeIntervalSinceNow)
            } catch {
                return ToolExecutor.Result(
                    stdout: "",
                    stderr: "escalate (\(provider.rawValue)) failed: \(error)",
                    exitCode: 1,
                    durationSec: -t0.timeIntervalSinceNow)
            }
        }

        do {
            return try ToolExecutor.execute(toolName: name, arguments: arguments,
                                              schema: schema,
                                              timeoutSec: toolTimeoutSec)
        } catch {
            // Synthesise a Result so the model gets *something* back.
            return ToolExecutor.Result(
                stdout: "",
                stderr: "tool error: \(error)",
                exitCode: 2,
                durationSec: 0)
        }
    }

    private func encodeToolResult(_ name: String, result: ToolExecutor.Result) -> String {
        // Compact JSON; the model just needs to see the contents.
        let obj: [String: Any] = [
            "tool": name,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": Int(result.exitCode),
        ]
        if let data = try? JSONSerialization.data(withJSONObject: obj),
           let s = String(data: data, encoding: .utf8) { return s }
        return "{\"tool\":\"\(name)\",\"stdout\":\"\(result.stdout)\"}"
    }

    // MARK: - Generation

    /// Stream tokens through `forwardCached`, decode incrementally, and
    /// stop as soon as we've accumulated a balanced top-level JSON
    /// object OR we hit `maxTokensPerTurn` / cache cap / assistant
    /// terminator. Returns the decoded assistant text.
    private func generateUntilJSONOrLimit() -> String {
        var generated: [Int] = []
        var lastLogits: MLXArray
        // Seed with the cache's last logits — we just appended text via
        // `forwardCached`, so the *next* token's logits live in the tail
        // of that pass's output. The cache doesn't store logits, so we
        // do one tiny forward on a single token equal to the last input
        // byte to recover them. Simpler approach: feed a no-op "<assistant
        // marker has already been fed>" placeholder. Since `feedText`
        // discards its logits, we re-feed the LAST byte to get them back
        // here. To avoid double-counting, we rewind first.
        cache.rewind(by: 1)
        // Reconstruct the last token id we appended. We track it via
        // `lastFedToken`, set inside `feedText`.
        let lastTok = lastFedToken
        let oneTok = MLXArray([Int32(lastTok)], [1, 1])
        let l0 = model.forwardCached(oneTok, cache: cache)
        lastLogits = l0[0..., l0.shape[1] - 1, 0...]

        for _ in 0..<maxTokensPerTurn {
            if TrainSupport.stopRequested.isSet { break }
            let nextId: MLXArray
            if temperature <= 0 {
                nextId = argMax(lastLogits, axis: -1).reshaped([1, 1])
            } else {
                let scaled = lastLogits / MLXArray(temperature)
                nextId = MLXRandom.categorical(scaled).reshaped([1, 1])
            }
            eval(nextId)
            let id = Int(nextId.item(Int32.self))
            generated.append(id)
            // Decoded-so-far view for stop & JSON detection.
            let decoded = decode(generated)
            if isCompleteJSONObject(decoded) { break }
            if decoded.contains(assistantTerminator) { break }
            if cache.currentLength >= cfg.contextLength { break }
            let logits = model.forwardCached(nextId.asType(.int32), cache: cache)
            lastLogits = logits[0..., 0, 0...]
        }
        return decode(generated)
    }

    // MARK: - Cache feeding

    private var lastFedToken: Int = 0

    /// Encode + forward through the cache. We track the last token id so
    /// `generateUntilJSONOrLimit` can rewind + recover the next-token
    /// logits without buffering them at every call site.
    private func feedText(_ text: String) {
        var ids = encode(text)
        if ids.isEmpty { return }
        // Cap any single chunk at remaining context. Without this a
        // chatty tool result on a small-context model would crash the
        // forward pass with a positional-encoding out-of-range error.
        let remaining = cfg.contextLength - cache.currentLength
        if ids.count > remaining {
            if remaining <= 0 {
                fputs("warning: KV cache full (\(cache.currentLength)/\(cfg.contextLength)); dropping chunk.\n", stderr)
                return
            }
            fputs("warning: chunk (\(ids.count) tokens) exceeds remaining context (\(remaining)); truncating tail.\n", stderr)
            ids = Array(ids.prefix(remaining))
        }
        lastFedToken = ids.last!
        let arr = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        _ = model.forwardCached(arr, cache: cache)
    }

    // MARK: - Router pre-step

    /// Predict the most-likely tool for `query` using the loaded
    /// router checkpoint. Returns the top-1 tool name + softmax
    /// probability. Byte-level encode + pad to the router's context
    /// length — matches `TrainExtractor.encode` exactly so the
    /// distribution shift between train and inference is zero.
    private func predictWithRouter(hook: RouterHook, query: String)
        -> (tool: String, prob: Float)
    {
        let cfg = hook.router.config
        var ids = [UInt8](query.utf8)
            .prefix(cfg.contextLength)
            .map { Int32($0) }
        // Clamp + pad.
        for i in 0..<ids.count {
            if ids[i] < 0 || ids[i] >= Int32(cfg.vocabSize) { ids[i] = 0 }
        }
        while ids.count < cfg.contextLength { ids.append(0) }
        let x = MLXArray(ids, [1, cfg.contextLength])
        let top = hook.router.topK(idx: x, k: 1)
        guard let first = top.first else { return ("?", 0) }
        let name = (first.classIdx >= 0 && first.classIdx < hook.labels.count)
            ? hook.labels[first.classIdx] : "?"
        return (name, first.prob)
    }

    // MARK: - Tokenization

    private func encode(_ s: String) -> [Int] {
        if let tok = tokenizer {
            do { return try tok.encode(s) }
            catch {
                fputs("tokenizer encode failed (\(error)), falling back to bytes\n", stderr)
                return [UInt8](s.utf8).map { Int($0) }
            }
        }
        return [UInt8](s.utf8).map { Int($0) }
    }

    private func decode(_ ids: [Int]) -> String {
        if let tok = tokenizer { return tok.decode(ids) }
        let bytes = ids.compactMap { (id: Int) -> UInt8? in
            (id >= 0 && id < 256) ? UInt8(id) : nil
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    // MARK: - JSON detection / extraction

    /// True iff `text` contains a top-level balanced `{ ... }` object.
    /// We scan from the first `{` and track brace depth, respecting
    /// strings and escapes. Whitespace-only between objects is fine.
    private func isCompleteJSONObject(_ text: String) -> Bool {
        return findFirstBalancedObject(in: text) != nil
    }

    private func extractJSONObject(_ text: String) -> [String: Any]? {
        guard let s = findFirstBalancedObject(in: text) else { return nil }
        guard let data = s.data(using: .utf8) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private func findFirstBalancedObject(in text: String) -> String? {
        var depth = 0
        var inString = false
        var escape = false
        var start: String.Index? = nil
        for i in text.indices {
            let c = text[i]
            if escape { escape = false; continue }
            if inString {
                if c == "\\" { escape = true }
                else if c == "\"" { inString = false }
                continue
            }
            switch c {
            case "\"":
                inString = true
            case "{":
                if depth == 0 { start = i }
                depth += 1
            case "}":
                depth -= 1
                if depth == 0, let s = start {
                    let end = text.index(after: i)
                    return String(text[s..<end])
                }
            default:
                continue
            }
        }
        return nil
    }

    private func stripChatML(_ text: String) -> String {
        // Drop a leading <|im_start|>assistant\n if the model echoed one.
        var t = text
        let leading = "<|im_start|>assistant\n"
        if t.hasPrefix(leading) { t = String(t.dropFirst(leading.count)) }
        if let r = t.range(of: assistantTerminator) {
            t = String(t[..<r.lowerBound])
        }
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Transcript / JSON output

    private func recordEvent(event: [String: Any]) {
        guard let url = transcriptURL else { return }
        guard let data = try? JSONSerialization.data(withJSONObject: event,
                                                      options: []) else { return }
        if let fh = try? FileHandle(forWritingTo: url) {
            fh.seekToEndOfFile()
            fh.write(data)
            fh.write(Data([0x0A]))
            try? fh.close()
        } else {
            // First write — create the file.
            var combined = data
            combined.append(0x0A)
            try? combined.write(to: url, options: .atomic)
        }
    }

    private func emitJSONEvent(_ obj: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: obj),
              let s = String(data: data, encoding: .utf8) else { return }
        print(s)
        fflush(stdout)
    }
}
