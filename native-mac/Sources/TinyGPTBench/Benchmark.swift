import Foundation
import TinyGPTModel

/// `tinygpt bench` — Mac LLM-inference benchmark harness.
///
/// CLI entry point. Parses args, builds the engine + workload, runs
/// it, dumps JSON + markdown.
///
/// Design doc: docs/benchmark_harness_design.md
public enum Benchmark {
    public static func run(args: [String]) {
        var engineName = "tinygpt"
        var modelPath: String? = nil
        var mode = "single"
        var batchSize = 1
        var promptTokens = 64
        var genTokens = 64
        var nRuns = 5
        var warmRuns = 1
        var outputPath: String? = nil
        var enableEnergy = true
        var promptOverride: String? = nil

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--engine":
                guard i + 1 < args.count else { exitUsage() }
                engineName = args[i + 1]; i += 2
            case "--model":
                guard i + 1 < args.count else { exitUsage() }
                modelPath = args[i + 1]; i += 2
            case "--mode":
                guard i + 1 < args.count else { exitUsage() }
                mode = args[i + 1]; i += 2
            case "--batch-size":
                guard i + 1 < args.count else { exitUsage() }
                batchSize = Int(args[i + 1]) ?? batchSize; i += 2
            case "--prompt-tokens":
                guard i + 1 < args.count else { exitUsage() }
                promptTokens = Int(args[i + 1]) ?? promptTokens; i += 2
            case "--gen-tokens":
                guard i + 1 < args.count else { exitUsage() }
                genTokens = Int(args[i + 1]) ?? genTokens; i += 2
            case "--n-runs":
                guard i + 1 < args.count else { exitUsage() }
                nRuns = Int(args[i + 1]) ?? nRuns; i += 2
            case "--warm-runs":
                guard i + 1 < args.count else { exitUsage() }
                warmRuns = Int(args[i + 1]) ?? warmRuns; i += 2
            case "--output":
                guard i + 1 < args.count else { exitUsage() }
                outputPath = args[i + 1]; i += 2
            case "--no-energy":
                enableEnergy = false; i += 1
            case "--prompt":
                guard i + 1 < args.count else { exitUsage() }
                promptOverride = args[i + 1]; i += 2
            case "-h", "--help":
                exitUsage()
            default:
                fputs("unknown bench flag: \(args[i])\n", stderr)
                exitUsage()
            }
        }

        guard let modelPath = modelPath else {
            fputs("--model <path> required\n", stderr)
            exitUsage()
        }
        guard let workloadMode = WorkloadController.Mode(rawValue: mode) else {
            fputs("unknown --mode \(mode). Choose single|batch|server|sustained\n", stderr)
            exit(2)
        }

        // Build the engine.
        var engine: EngineAdapter
        switch engineName {
        case "tinygpt":
            let e = TinyGPTEngine()
            engine = e
        case "mlx-lm":
            engine = MLXLMEngine()
        case "llama.cpp", "llamacpp":
            engine = LlamaCppEngine()
        case "ollama":
            engine = OllamaEngine()
        default:
            fputs("unknown engine \(engineName). Choose tinygpt|mlx-lm|llama.cpp|ollama\n", stderr)
            exit(2)
        }

        print("""

        tinygpt bench — \(engineName)
        ---------------------------------
        model:           \(modelPath)
        workload:        \(mode), prompt=\(promptTokens) tok, gen=\(genTokens) tok, batch=\(batchSize)
        runs:            \(nRuns) (+\(warmRuns) warm)
        energy metrics:  \(enableEnergy ? "on (powermetrics)" : "off")
        """)

        do {
            try engine.load(modelPath: modelPath)
        } catch {
            fputs("engine load failed: \(error)\n", stderr)
            exit(1)
        }
        print("loaded — \(formatInt(engine.parameterCount())) params")

        // Build the prompt. If the user passes --prompt, byte-encode
        // it; otherwise generate a deterministic byte sequence of the
        // requested length. We do byte-level encoding so the harness
        // doesn't depend on a tokenizer being present — every model
        // either has byte-level vocab=256 or a BPE tokenizer that
        // happens to map ASCII to single tokens. For paper-quality
        // numbers the user should pass a real prompt that's been
        // tokenized externally; this is the dev-loop fast path.
        let prompt: [Int32]
        if let text = promptOverride {
            prompt = [UInt8](text.utf8).map { Int32($0) }
        } else {
            // Deterministic ASCII filler: cycle "The quick brown fox… "
            // padded to promptTokens length. Same content every run so
            // tokenizers map it the same way.
            let filler = "The quick brown fox jumps over the lazy dog. "
            var s = ""
            while s.count < promptTokens { s += filler }
            s = String(s.prefix(promptTokens))
            prompt = [UInt8](s.utf8).map { Int32($0) }
        }
        print("prompt: \(prompt.count) bytes/tokens (byte-level)")

        let workload = WorkloadController.Config(
            mode: workloadMode,
            promptTokens: prompt.count,
            genTokens: genTokens,
            batchSize: batchSize,
            nRuns: nRuns,
            warmRuns: warmRuns,
            enableEnergy: enableEnergy
        )
        let controller = WorkloadController(config: workload, engine: engine)

        print("\nrunning…")
        let t0 = Date()
        let results: [WorkloadController.RunResult]
        do {
            results = try controller.execute(prompt: prompt)
        } catch {
            fputs("bench run failed: \(error)\n", stderr)
            exit(1)
        }
        let elapsed = -t0.timeIntervalSinceNow
        print(String(format: "done in %.1fs across %d runs\n", elapsed, results.count))

        let report = Reporter.Report(
            engineName: engine.name,
            engineCommit: engine.commitHash,
            modelPath: modelPath,
            modelParams: engine.parameterCount(),
            workload: workload,
            provenance: Reporter.Provenance(),
            runs: results
        )

        // Markdown to stdout, JSON to --output (or stdout under
        // RESULTS_JSON section).
        print(Reporter.toMarkdown(report))

        do {
            let json = try Reporter.toJSON(report)
            if let path = outputPath {
                try json.write(toFile: path, atomically: true, encoding: .utf8)
                print("\nJSON → \(path)")
            } else {
                print("\n--- RESULTS_JSON ---")
                print(json)
            }
        } catch {
            fputs("JSON emit failed: \(error)\n", stderr)
            exit(1)
        }
    }

    static func formatInt(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    static func exitUsage() -> Never {
        print("""
        usage: tinygpt bench --model <path> [options]

        --engine tinygpt|mlx-lm|llama.cpp|ollama
                                    Inference engine to benchmark (default: tinygpt).
                                    Only `tinygpt` is implemented in this scaffold.
        --model <path>              Path to a .tinygpt file or an HF model dir. Required.
        --mode single|batch|server|sustained
                                    Workload pattern (default: single).
                                    server/sustained are placeholders.
        --batch-size N              Batch size for batch mode (default 1).
        --prompt-tokens N           Synthetic prompt length when --prompt not given (default 64).
        --gen-tokens N              Tokens to generate per run (default 64).
        --n-runs N                  Timed runs (default 5; ≥20 recommended for p95/p99).
        --warm-runs N               Discarded warm-up runs (default 1).
        --output <file.json>        Write JSON results to file (default: stdout).
        --no-energy                 Skip powermetrics (and the sudo it needs).
        --prompt "..."              Use this text as the prompt instead of synthetic.

        Design doc: docs/benchmark_harness_design.md.
        """)
        exit(2)
    }
}
