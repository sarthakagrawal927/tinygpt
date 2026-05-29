import Foundation
import MLX
import MLXNN
import MLXRandom
import TinyGPTIO
import TinyGPTModel

/// Common surface every benchmark target must satisfy.
///
/// Design intent: an "engine" is a black-box LLM inference runtime. The
/// harness feeds it a tokenised prompt + generation budget; it returns
/// per-token timings. Foreign engines (MLX-LM, llama.cpp, MLC-LLM,
/// Ollama) will be subprocess wrappers that satisfy the same protocol,
/// so the WorkloadController and MetricsCollector don't have to know
/// which engine they're driving.
///
/// This matches the Bench360 "backend abstraction" component
/// (arXiv 2511.16682) — see docs/benchmark_harness_design.md §3.1.
public protocol EngineAdapter {
    /// Human-readable engine name reported in the output JSON.
    var name: String { get }

    /// Engine commit hash if available. Foreign engines fill this in by
    /// shelling out to their binary's `--version`; the in-process
    /// TinyGPT engine returns this repo's HEAD.
    var commitHash: String? { get }

    /// Load weights + tokenizer from disk. May be slow; not timed.
    /// Subsequent `prefill`/`decode` calls must reuse the loaded model.
    mutating func load(modelPath: String) throws

    /// Number of model parameters. Reported in the metrics block.
    func parameterCount() -> Int

    /// Process the prompt and return the time to first generated token.
    /// Implementations must:
    ///   - reset any per-request KV cache state (or call `reset()`),
    ///   - run the prompt through the model,
    ///   - sample one token,
    ///   - return the wall-clock interval from call-start to that
    ///     first-token-available point.
    ///
    /// The first sampled token is *not* included in the decode loop; the
    /// caller hands the engine a fresh `decode(maxTokens-1, ...)` after.
    mutating func prefill(tokens: [Int32]) throws -> PrefillResult

    /// Stream `maxTokens` tokens via the engine's per-step decode path.
    /// Each token's inter-arrival time is recorded in
    /// `DecodeResult.interTokenLatenciesMs`. Implementations that
    /// support a batch dimension > 1 use `batchSize`; engines that
    /// don't may ignore it (the WorkloadController will catch that
    /// case and not over-claim throughput).
    mutating func decode(maxTokens: Int, batchSize: Int) throws -> DecodeResult

    /// Peak resident bytes the engine has observed since `load()`.
    /// Implementations sample `task_info(TASK_VM_INFO)` themselves and
    /// keep a running max — the collector aggregates across engines.
    func peakResidentBytes() -> Int

    /// Reset per-request state (KV cache, RNG seed, etc.) so the next
    /// `prefill` starts fresh. Does NOT unload weights.
    mutating func reset()
}

/// Returned by `EngineAdapter.prefill`.
public struct PrefillResult {
    /// Wall-clock from prefill-start to first-token-available, in ms.
    public let ttftMs: Double
    /// The first sampled token id. Caller appends to its transcript.
    public let firstToken: Int32

    public init(ttftMs: Double, firstToken: Int32) {
        self.ttftMs = ttftMs
        self.firstToken = firstToken
    }
}

/// Returned by `EngineAdapter.decode`.
public struct DecodeResult {
    /// Token ids emitted (length ≤ requested maxTokens; may be shorter
    /// if the engine hit a context-length cap or an EOS).
    public let tokens: [Int32]
    /// Per-token inter-arrival latencies, in ms. `tokens.count` entries.
    /// `interTokenLatenciesMs[0]` is the time between prefill's last
    /// token and the first decoded token; subsequent entries are
    /// per-step inter-token latencies. The harness drops the first
    /// entry from ITL distributions because it includes a small amount
    /// of warm-up beyond a pure decode step.
    public let interTokenLatenciesMs: [Double]
    /// Total decode wall-clock, in ms (sum of `interTokenLatenciesMs`).
    public let totalDecodeMs: Double

    public init(tokens: [Int32], interTokenLatenciesMs: [Double], totalDecodeMs: Double) {
        self.tokens = tokens
        self.interTokenLatenciesMs = interTokenLatenciesMs
        self.totalDecodeMs = totalDecodeMs
    }
}

/// Errors common to all adapters.
public enum EngineError: Error, CustomStringConvertible {
    case notImplemented(String)
    case modelNotLoaded
    case loadFailed(String)
    case decodeFailed(String)

    public var description: String {
        switch self {
        case .notImplemented(let what): return "engine: not implemented — \(what)"
        case .modelNotLoaded: return "engine: model not loaded; call load() first"
        case .loadFailed(let why): return "engine: load failed — \(why)"
        case .decodeFailed(let why): return "engine: decode failed — \(why)"
        }
    }
}

// =============================================================================
// TinyGPTEngine — in-process MLX adapter.
// =============================================================================

/// The reference adapter — wraps the existing `AnyModel.forwardCached`
/// path so we have real numbers from day one. Stays in-process for
/// minimum overhead; the harness does NOT subprocess itself.
public final class TinyGPTEngine: EngineAdapter {
    public let name = "tinygpt"
    public let commitHash: String?

    private var model: AnyModel?
    private var config: ModelConfig?
    private var cache: KVCache?
    private var lastToken: Int32?
    private var peakBytes: Int = 0

    /// Sampling parameters — fixed across runs so timings are
    /// comparable. Defaults are: temperature 0 (greedy), no top-p,
    /// no top-k. We sample greedily by default so a benchmark re-run
    /// produces identical tokens, which makes regression detection
    /// trivial.
    public var temperature: Float = 0.0
    public var seed: UInt64 = 42

    public init() {
        self.commitHash = TinyGPTEngine.gitHeadShort()
    }

    public func parameterCount() -> Int {
        return model?.numParameters() ?? 0
    }

    public func load(modelPath: String) throws {
        do {
            let load = try ModelLoader.load(modelPath)
            self.model = load.model
            self.config = load.config
            MLXRandom.seed(seed)
        } catch {
            throw EngineError.loadFailed("\(error)")
        }
    }

    public func reset() {
        // Drop the KV cache and the last-token cursor; weights stay
        // resident so the next prefill is hot.
        self.cache = nil
        self.lastToken = nil
        MLXRandom.seed(seed)
    }

    public func prefill(tokens: [Int32]) throws -> PrefillResult {
        guard let model = self.model, let cfg = self.config else {
            throw EngineError.modelNotLoaded
        }
        // Fresh KV cache per request — the WorkloadController calls
        // reset() between runs explicitly but we re-init here too.
        cache = KVCache(nLayers: cfg.nLayers, kvDtype: nil, sink: nil, window: nil)

        let input = MLXArray(tokens, [1, tokens.count])
        let t0 = Date()
        let logits = model.forwardCached(input, cache: cache!)
        let last = logits[0..., logits.shape[1] - 1, 0...]
        let nextId: MLXArray
        if temperature <= 0 {
            nextId = argMax(last, axis: -1).reshaped([1, 1])
        } else {
            let scaled = last / MLXArray(temperature)
            nextId = MLXRandom.categorical(scaled).reshaped([1, 1])
        }
        eval(nextId)
        let elapsedMs = -t0.timeIntervalSinceNow * 1000.0
        let tokId = nextId.item(Int32.self)
        self.lastToken = tokId
        sampleRSS()
        return PrefillResult(ttftMs: elapsedMs, firstToken: tokId)
    }

    public func decode(maxTokens: Int, batchSize: Int) throws -> DecodeResult {
        guard let model = self.model, let cfg = self.config, let cache = self.cache else {
            throw EngineError.modelNotLoaded
        }
        // The TinyGPT KV path is B=1 today; the harness reports the
        // discrepancy rather than over-claiming.
        if batchSize > 1 {
            // No-op; we still run B=1 and report effective batch=1.
        }
        guard var lastTok = self.lastToken else {
            throw EngineError.decodeFailed("prefill did not produce a starting token")
        }

        var tokens: [Int32] = []
        var itls: [Double] = []
        let overallStart = Date()

        for _ in 0..<maxTokens {
            if cache.currentLength >= cfg.contextLength { break }
            let tStep = Date()
            let input = MLXArray([lastTok], [1, 1])
            let logits = model.forwardCached(input, cache: cache)
            let last = logits[0..., 0, 0...]
            let nextId: MLXArray
            if temperature <= 0 {
                nextId = argMax(last, axis: -1).reshaped([1, 1])
            } else {
                let scaled = last / MLXArray(temperature)
                nextId = MLXRandom.categorical(scaled).reshaped([1, 1])
            }
            eval(nextId)
            let stepMs = -tStep.timeIntervalSinceNow * 1000.0
            let tokId = nextId.item(Int32.self)
            tokens.append(tokId)
            itls.append(stepMs)
            lastTok = tokId
            self.lastToken = tokId
        }
        sampleRSS()
        let totalMs = -overallStart.timeIntervalSinceNow * 1000.0
        return DecodeResult(tokens: tokens, interTokenLatenciesMs: itls, totalDecodeMs: totalMs)
    }

    public func peakResidentBytes() -> Int {
        sampleRSS()
        return peakBytes
    }

    /// Update the running peak resident bytes.
    private func sampleRSS() {
        let rss = ProcessMemory.residentBytes()
        if rss > peakBytes { peakBytes = rss }
    }

    /// `git rev-parse --short HEAD` against the repo containing this
    /// binary's source tree. Best-effort; nil if not in a git repo.
    static func gitHeadShort() -> String? {
        let p = Process()
        p.launchPath = "/usr/bin/env"
        p.arguments = ["git", "rev-parse", "--short", "HEAD"]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = Pipe()
        do {
            try p.run()
            p.waitUntilExit()
            if p.terminationStatus == 0 {
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                let s = String(data: data, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                return s?.isEmpty == false ? s : nil
            }
        } catch {
            return nil
        }
        return nil
    }
}

// =============================================================================
// Stubs for foreign engines. Protocol shape only — implementing these
// is the next milestone (see docs/benchmark_harness_design.md §7).
// =============================================================================

/// `mlx_lm.generate` subprocess wrapper. Stub.
public final class MLXLMEngine: EngineAdapter {
    public let name = "mlx-lm"
    public var commitHash: String? { nil }
    public init() {}
    public func parameterCount() -> Int { 0 }
    public func load(modelPath: String) throws {
        throw EngineError.notImplemented("MLXLMEngine.load — subprocess wrapper pending")
    }
    public func reset() {}
    public func prefill(tokens: [Int32]) throws -> PrefillResult {
        throw EngineError.notImplemented("MLXLMEngine.prefill")
    }
    public func decode(maxTokens: Int, batchSize: Int) throws -> DecodeResult {
        throw EngineError.notImplemented("MLXLMEngine.decode")
    }
    public func peakResidentBytes() -> Int { 0 }
}

/// `llama-cli` (llama.cpp Metal) subprocess wrapper. Stub.
public final class LlamaCppEngine: EngineAdapter {
    public let name = "llama.cpp"
    public var commitHash: String? { nil }
    public init() {}
    public func parameterCount() -> Int { 0 }
    public func load(modelPath: String) throws {
        throw EngineError.notImplemented("LlamaCppEngine.load — subprocess wrapper pending")
    }
    public func reset() {}
    public func prefill(tokens: [Int32]) throws -> PrefillResult {
        throw EngineError.notImplemented("LlamaCppEngine.prefill")
    }
    public func decode(maxTokens: Int, batchSize: Int) throws -> DecodeResult {
        throw EngineError.notImplemented("LlamaCppEngine.decode")
    }
    public func peakResidentBytes() -> Int { 0 }
}

/// Ollama HTTP-API wrapper. Stub.
public final class OllamaEngine: EngineAdapter {
    public let name = "ollama"
    public var commitHash: String? { nil }
    public init() {}
    public func parameterCount() -> Int { 0 }
    public func load(modelPath: String) throws {
        throw EngineError.notImplemented("OllamaEngine.load — HTTP wrapper pending")
    }
    public func reset() {}
    public func prefill(tokens: [Int32]) throws -> PrefillResult {
        throw EngineError.notImplemented("OllamaEngine.prefill")
    }
    public func decode(maxTokens: Int, batchSize: Int) throws -> DecodeResult {
        throw EngineError.notImplemented("OllamaEngine.decode")
    }
    public func peakResidentBytes() -> Int { 0 }
}
