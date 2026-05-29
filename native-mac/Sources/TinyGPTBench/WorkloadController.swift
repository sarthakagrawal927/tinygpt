import Foundation

/// Orchestrates the n_runs × workload mode × (engine, model) experiment.
///
/// Modes (Bench360 §3.2):
///   - single-stream — one prompt at a time. The default.
///   - batch         — `--batch-size N` prompts processed simultaneously.
///   - server        — concurrent requests with arrival distribution.
///                     **Placeholder**: currently falls back to batch and
///                     emits a warning. Real implementation needs a
///                     scheduler (next milestone).
///   - sustained     — continuous decode for N seconds; thermal regression.
///                     **Placeholder** — also pending.
///
/// See docs/benchmark_harness_design.md §3.2.
public final class WorkloadController {
    public enum Mode: String { case single, batch, server, sustained }

    public struct Config {
        public let mode: Mode
        public let promptTokens: Int
        public let genTokens: Int
        public let batchSize: Int
        public let nRuns: Int
        public let warmRuns: Int
        public let enableEnergy: Bool

        public init(mode: Mode, promptTokens: Int, genTokens: Int,
                    batchSize: Int, nRuns: Int, warmRuns: Int,
                    enableEnergy: Bool) {
            self.mode = mode
            self.promptTokens = promptTokens
            self.genTokens = genTokens
            self.batchSize = batchSize
            self.nRuns = nRuns
            self.warmRuns = warmRuns
            self.enableEnergy = enableEnergy
        }
    }

    public struct RunResult {
        public let runIndex: Int
        public let warm: Bool
        public let metrics: MetricsCollector.RunMetrics
    }

    public let config: Config
    public var engine: EngineAdapter

    public init(config: Config, engine: EngineAdapter) {
        self.config = config
        self.engine = engine
    }

    /// Execute the requested workload. Returns one `RunResult` per
    /// timed run; warm-up runs are *not* included in the returned
    /// array (they only matter for cache warming).
    public func execute(prompt: [Int32]) throws -> [RunResult] {
        switch config.mode {
        case .single:
            return try runSingle(prompt: prompt)
        case .batch:
            return try runBatch(prompt: prompt)
        case .server:
            fputs("warning: server mode is a placeholder — falling back to batch with batchSize=\(config.batchSize)\n", stderr)
            return try runBatch(prompt: prompt)
        case .sustained:
            fputs("warning: sustained mode is a placeholder — falling back to single-stream\n", stderr)
            return try runSingle(prompt: prompt)
        }
    }

    private func runSingle(prompt: [Int32]) throws -> [RunResult] {
        var out: [RunResult] = []
        // Warm-up runs — discarded, but pay the JIT-compile and KV-init
        // cost so the timed runs are steady-state.
        for _ in 0..<config.warmRuns {
            engine.reset()
            _ = try engine.prefill(tokens: prompt)
            _ = try engine.decode(maxTokens: max(1, config.genTokens / 4), batchSize: 1)
        }
        for i in 0..<config.nRuns {
            engine.reset()
            let collector = MetricsCollector()
            collector.start(enableEnergy: config.enableEnergy)
            let pre = try engine.prefill(tokens: prompt)
            collector.recordPrefill(tokenCount: prompt.count, ttftMs: pre.ttftMs)
            collector.markDecodeStart()
            let dec = try engine.decode(maxTokens: config.genTokens - 1, batchSize: 1)
            collector.recordDecode(tokenCount: dec.tokens.count,
                                    totalMs: dec.totalDecodeMs,
                                    itlsMs: dec.interTokenLatenciesMs)
            let m = collector.stop(peakResidentBytes: engine.peakResidentBytes())
            out.append(RunResult(runIndex: i, warm: false, metrics: m))
        }
        return out
    }

    /// Batch mode — we issue `batchSize` copies of the prompt at once.
    /// The TinyGPTEngine adapter currently caps effective batch at 1
    /// (it'll log a one-line note about that on the first batched run);
    /// foreign engines that DO batch will fill it in properly.
    private func runBatch(prompt: [Int32]) throws -> [RunResult] {
        if config.batchSize < 1 {
            throw EngineError.decodeFailed("batchSize must be ≥ 1, got \(config.batchSize)")
        }
        // Warm-up.
        for _ in 0..<config.warmRuns {
            engine.reset()
            _ = try engine.prefill(tokens: prompt)
            _ = try engine.decode(maxTokens: max(1, config.genTokens / 4),
                                  batchSize: config.batchSize)
        }
        var out: [RunResult] = []
        for i in 0..<config.nRuns {
            engine.reset()
            let collector = MetricsCollector()
            collector.start(enableEnergy: config.enableEnergy)
            let pre = try engine.prefill(tokens: prompt)
            collector.recordPrefill(tokenCount: prompt.count * config.batchSize,
                                     ttftMs: pre.ttftMs)
            collector.markDecodeStart()
            let dec = try engine.decode(maxTokens: config.genTokens - 1,
                                         batchSize: config.batchSize)
            // For honesty: report tokens × effective_batch in
            // tokens/sec only when the engine actually scaled. For the
            // TinyGPT in-process adapter, effective batch is 1.
            let effectiveTokens = dec.tokens.count  // engine returns per-batch row 0 only
            collector.recordDecode(tokenCount: effectiveTokens,
                                    totalMs: dec.totalDecodeMs,
                                    itlsMs: dec.interTokenLatenciesMs)
            let m = collector.stop(peakResidentBytes: engine.peakResidentBytes())
            out.append(RunResult(runIndex: i, warm: false, metrics: m))
        }
        return out
    }
}
