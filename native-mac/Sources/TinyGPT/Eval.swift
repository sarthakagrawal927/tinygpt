import Foundation
import MLX
import TinyGPTIO
import TinyGPTModel

/// `tinygpt eval` — measure how well a checkpoint predicts a held-out text.
/// Reports:
///   - cross-entropy loss (lower = better; ln(256) ≈ 5.55 is uniform random)
///   - bits per byte (loss / ln(2))
///   - perplexity (exp(loss))
///   - a few generated samples
///
/// Why bits-per-byte: it's the metric byte-level language models are scored
/// on in the literature, and lets you compare TinyGPT against published
/// numbers (e.g., Shakespeare BPB ≈ 1.0 for character-level LSTM, ~0.9 for
/// well-trained transformers).
///
/// USAGE
///
///   tinygpt eval path/to/model.tinygpt --corpus held-out.txt
///   tinygpt eval shakespeare.bin --corpus shakespeare-complete.txt --batches 100
enum Eval {
    static func run(args: [String]) {
        var path: String?
        var corpusPath: String?
        var loraPath: String? = nil
        var nBatches = 50
        var batchSize: Int? = nil
        var seed: UInt32 = 0
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--corpus": corpusPath = args[i+1]; i += 2
            case "--lora":   loraPath = args[i+1]; i += 2
            case "--batches": nBatches = Int(args[i+1]) ?? nBatches; i += 2
            case "--batch": batchSize = Int(args[i+1]); i += 2
            case "--seed": seed = UInt32(args[i+1]) ?? 0; i += 2
            case "-h", "--help": exitUsage()
            default:
                if args[i].hasPrefix("-") {
                    fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
                }
                path = args[i]; i += 1
            }
        }
        guard let path = path else {
            fputs("eval: missing <model.tinygpt>\n", stderr); exitUsage()
        }
        guard let corpusPath = corpusPath else {
            fputs("eval: --corpus is required\n", stderr); exitUsage()
        }
        let url = URL(fileURLWithPath: path)
        let corpusURL = URL(fileURLWithPath: corpusPath)

        // Load model
        let file: TinyGPTFile
        do { file = try TinyGPTFileReader.read(url) }
        catch { fputs("error reading \(path): \(error)\n", stderr); exit(1) }
        let h = file.header.config
        let cfg = ModelConfig(
            vocabSize: 256,
            contextLength: h.ctx ?? 256,
            nLayers: h.layers ?? 12,
            nHeads: h.heads ?? 8,
            dModel: h.dModel ?? 256,
            dMlp: h.dMlp ?? 1024
        )
        let model = TinyGPTModel(cfg)
        do { try TinyGPTWeightLoader.load(file, into: model) }
        catch { fputs("error loading weights: \(error)\n", stderr); exit(1) }

        // Apply LoRA adapter on top if provided.
        if let loraPath = loraPath {
            do {
                let adapter = try LoraAdapterReader.read(URL(fileURLWithPath: loraPath))
                try LoraAdapterReader.apply(adapter, to: model)
                print("• with LoRA adapter: rank=\(adapter.header.rank) alpha=\(adapter.header.alpha) targets=\(adapter.header.targetSuffixes.joined(separator: ","))")
            } catch {
                fputs("error loading LoRA: \(error)\n", stderr); exit(1)
            }
        }

        // Load corpus
        let corpus: ByteCorpus
        do { corpus = try ByteCorpus(contentsOf: corpusURL) }
        catch { fputs("error reading corpus: \(error)\n", stderr); exit(1) }

        let B = batchSize ?? 8
        print("""

        TinyGPT — eval
        --------------
        model:    \(path)
        corpus:   \(corpusPath) (\(formatBytes(corpus.bytes.count)))
        config:   \(cfg.nLayers)L · d=\(cfg.dModel) · ctx=\(cfg.contextLength)
        batches:  \(nBatches) × batch \(B) × ctx \(cfg.contextLength)
                  = \(formatLargeInt(nBatches * B * cfg.contextLength)) tokens scored

        """)

        // Score the corpus across N random windows. Each batch:
        //   - inputs:  [B, T] int32 token ids
        //   - targets: shifted by 1 — predict next byte
        //   - loss:    mean cross-entropy
        var lossSum: Float = 0
        var count = 0
        for k in 0..<nBatches {
            let (x, y) = corpus.sampleBatch(batchSize: B, contextLength: cfg.contextLength)
            let loss = model.loss(x, y)
            eval(loss)
            let lv = loss.item(Float.self)
            lossSum += lv
            count += 1
            if k < 3 || k % 10 == 0 || k == nBatches - 1 {
                fputs(String(format: "  batch %3d  loss %.3f  running avg %.3f\n",
                             k + 1, lv, lossSum / Float(count)), stderr)
            }
        }
        let avgLoss = lossSum / Float(count)
        let bpb = avgLoss / log(Float(2))  // ln → log2
        let ppl = exp(avgLoss)

        print("""

        RESULTS
        -------
        cross-entropy loss:    \(String(format: "%.4f", avgLoss))   (uniform baseline: \(String(format: "%.2f", log(Float(cfg.vocabSize)))))
        bits per byte (BPB):   \(String(format: "%.4f", bpb))
        perplexity:            \(String(format: "%.2f", ppl))

        """)
        // Reference points the user can sanity-check against:
        if avgLoss < 1.0 {
            print("✓ very strong — well below typical byte-level transformer scores")
        } else if avgLoss < 1.3 {
            print("✓ strong — comparable to a well-trained byte-level transformer")
        } else if avgLoss < 1.8 {
            print("· OK — grammar mostly emerges in samples; more training would help")
        } else if avgLoss < 3.0 {
            print("· weak — words form but sentences won't")
        } else {
            print("⚠ near random — the model isn't doing useful work")
        }

        // A few quick samples to anchor the numbers in observed output.
        print("\nSAMPLES")
        for prompt in ["The ", "He said, \"", "Once "] {
            print("  prompt: \(prompt.debugDescription)")
            let promptBytes = [UInt8](prompt.utf8)
            var idx = MLXArray(promptBytes.map { Int32($0) }, [1, promptBytes.count])
            var generated = prompt
            for _ in 0..<80 {
                let T = idx.shape.last!
                let lo = max(0, T - cfg.contextLength)
                let cond = idx[0..., lo..<T]
                let logits = model(cond)
                let last = logits[0..., logits.shape[1] - 1, 0...]
                let scaled = last / MLXArray(Float(0.7))
                let next = MLX.argMax(scaled, axis: -1).reshaped([1, 1])
                eval(next)
                let id = Int(next.item(Int32.self))
                if let scalar = UnicodeScalar(id), id >= 9 {
                    generated.append(Character(scalar))
                }
                idx = concatenated([idx, next.asType(idx.dtype)], axis: 1)
            }
            let clipped = generated.prefix(150).replacingOccurrences(of: "\n", with: "\\n")
            print("    \(clipped)")
        }
    }

    private static func formatBytes(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1f MB", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0f KB", Double(n) / 1_000) }
        return "\(n) B"
    }

    private static func formatLargeInt(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt eval <model.tinygpt> --corpus path.txt [options]

        --corpus path.txt    Held-out UTF-8 text to score (required)
        --batches N          Number of random windows to score (default: 50)
        --batch N            Tokens per window batch (default: 8)
        --seed N             Random seed (default: 0)

        Reports cross-entropy loss, bits-per-byte (the standard
        byte-level LM metric), and perplexity. Plus a few quick samples.
        """)
        exit(2)
    }
}
