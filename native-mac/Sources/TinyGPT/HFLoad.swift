import Foundation
import MLX
import MLXRandom
import TinyGPTIO
import TinyGPTModel
@preconcurrency import Tokenizers

/// `tinygpt hf-load <dir> [--sample] [--prompt "..."]` — instantiate a
/// TinyGPTModelHF from a downloaded HuggingFace model directory, load
/// the safetensors weights, optionally generate a sample to verify
/// everything wires up.
///
/// USAGE
///   huggingface-cli download meta-llama/Llama-3.2-1B --local-dir ~/Models/llama-3.2-1b
///   tinygpt hf-load ~/Models/llama-3.2-1b --sample --prompt "The capital of France is"
///
/// The dir must contain:
///   config.json                                — architecture description
///   tokenizer.json (+ tokenizer_config.json)   — BPE / SentencePiece vocab
///   model.safetensors (or sharded variants)    — weights
enum HFLoad {
    static func run(args: [String]) {
        var dirPath: String?
        var doSample = false
        var prompt = "The capital of France is"
        var maxTokens = 60
        var temperature: Float = 0.7
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--sample":      doSample = true; i += 1
            case "--prompt":      prompt = args[i+1]; i += 2
            case "--tokens":      maxTokens = Int(args[i+1]) ?? maxTokens; i += 2
            case "--temperature": temperature = Float(args[i+1]) ?? temperature; i += 2
            case "-h", "--help":  exitUsage()
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                dirPath = args[i]; i += 1
            }
        }
        guard let dirPath = dirPath else {
            fputs("missing <dir>\n", stderr); exitUsage()
        }
        let dir = URL(fileURLWithPath: dirPath)

        // Load the model
        print("loading HF model from \(dir.path)…")
        let result: HFModelLoader.LoadResult
        do { result = try HFModelLoader.load(from: dir) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        let model = result.model
        let cfg = result.config

        print("""

        ✓ loaded \(result.hfConfig.architectures.first ?? "unknown")
          params:   \(formatLargeInt(model.numParameters()))
          layers:   \(cfg.nLayers) · d=\(cfg.dModel) · ctx=\(cfg.contextLength)
          heads:    \(cfg.nHeads) Q / \(cfg.nKvHeads) KV (GQA: \(cfg.nHeads != cfg.nKvHeads))
          rope:     base=\(cfg.ropeBase) (extrapolation-friendly)
          vocab:    \(cfg.vocabSize) (needs BPE tokenizer for real text)
          device:   \(Device.defaultDevice())
        """)

        if doSample {
            sampleWithTokenizer(model: model, cfg: cfg, dir: dir,
                                 prompt: prompt, maxTokens: maxTokens,
                                 temperature: temperature)
        }

        print("\nNext: `tinygpt finetune \(dirPath) --corpus my.txt --out my.lora`")
    }

    /// Sample using the HF tokenizer attached to the model directory.
    /// Uses swift-transformers' AutoTokenizer to pick BPE / SentencePiece
    /// automatically based on the tokenizer.json file.
    private static func sampleWithTokenizer(
        model: TinyGPTModelHF, cfg: ModelConfig, dir: URL,
        prompt: String, maxTokens: Int, temperature: Float
    ) {
        // Load the tokenizer synchronously (we're a CLI; no async UI to
        // block). The async call returns Sendable text input/output, so
        // we can park it on a task and await.
        // Bridge async tokenizer load to our sync CLI via a thread-safe box.
        // Swift 6 strict concurrency flags Result<Tokenizer, Error> as
        // non-Sendable through a Task boundary; an actor wrapper resolves it.
        let tokenizer: Tokenizer
        do {
            tokenizer = try TokenizerBox.loadBlocking(from: dir)
        } catch {
            fputs("tokenizer load failed: \(error). Falling back to byte-level.\n", stderr)
            return
        }
        print("\n✓ tokenizer loaded\n")

        // Encode prompt
        let promptIds = tokenizer.encode(text: prompt)
        print(prompt, terminator: "")
        fflush(stdout)
        var idx = MLXArray(promptIds.map { Int32($0) }, [1, promptIds.count])

        // Greedy or temperature sample, decoding each new token via the
        // tokenizer (so multi-byte BPE tokens decode correctly).
        var generated: [Int] = []
        let t0 = Date()
        for _ in 0..<maxTokens {
            let T = idx.shape.last!
            let lo = max(0, T - cfg.contextLength)
            let cond = idx[0..., lo..<T]
            let logits = model(cond)
            let last = logits[0..., logits.shape[1] - 1, 0...]
            let next: MLXArray
            if temperature <= 0 {
                next = argMax(last, axis: -1).reshaped([1, 1])
            } else {
                next = MLXRandom.categorical(last / MLXArray(temperature))
                    .reshaped([1, 1])
            }
            eval(next)
            let id = Int(next.item(Int32.self))
            generated.append(id)
            // Decode incrementally: re-decode the whole generated tail
            // every step (BPE tokens for things like " word" only render
            // correctly when neighbours are known). Print the diff from
            // the previous render.
            let renderedSoFar = tokenizer.decode(tokens: generated)
            let priorLen = max(0, generated.count - 1)
            let prior = tokenizer.decode(tokens: Array(generated.prefix(priorLen)))
            let newPiece = String(renderedSoFar.dropFirst(prior.count))
            print(newPiece, terminator: "")
            fflush(stdout)
            idx = concatenated([idx, next.asType(idx.dtype)], axis: 1)
        }
        let elapsed = -t0.timeIntervalSinceNow
        print()
        print("\n(\(maxTokens) tokens in \(String(format: "%.2f", elapsed))s — \(String(format: "%.0f", Double(maxTokens) / elapsed)) tok/s)")
    }

    private static func formatLargeInt(_ n: Int) -> String {
        if n >= 1_000_000_000 { return String(format: "%.1f B", Double(n) / 1_000_000_000) }
        if n >= 1_000_000 { return String(format: "%.1f M", Double(n) / 1_000_000) }
        let f = NumberFormatter(); f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    /// Tiny actor that owns a single loaded Tokenizer and bridges the
    /// async load to the CLI's sync execution model. Necessary because
    /// swift-transformers' AutoTokenizer is async, and Swift 6 strict
    /// concurrency doesn't let us shuttle the result across a raw Task
    /// boundary without an actor-isolated container.
    private actor TokenizerBox {
        static func loadBlocking(from dir: URL) throws -> Tokenizer {
            let sem = DispatchSemaphore(value: 0)
            nonisolated(unsafe) var boxed: Tokenizer? = nil
            nonisolated(unsafe) var error: Error? = nil
            Task.detached {
                do {
                    boxed = try await AutoTokenizer.from(modelFolder: dir)
                } catch let e {
                    error = e
                }
                sem.signal()
            }
            sem.wait()
            if let e = error { throw e }
            guard let t = boxed else {
                throw NSError(domain: "TinyGPT", code: 99,
                              userInfo: [NSLocalizedDescriptionKey: "tokenizer load returned nothing"])
            }
            return t
        }
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt hf-load <hf-model-dir> [options]

        --sample                Run a quick sample after loading (smoke test)
        --prompt "..."          Sampling prompt (default: "The capital of France is")
        --tokens N              Max new tokens (default 60)
        --temperature F         Sampling temperature (default 0.7)
        """)
        exit(2)
    }
}
