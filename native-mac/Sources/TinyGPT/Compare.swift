import Foundation
import MLX
import TinyGPTIO
import TinyGPTModel

/// `tinygpt compare` — the headline workflow: pit a base model against
/// the same base + a LoRA adapter on a held-out eval set, report the
/// delta. Lets the user see "did my fine-tune actually help, and by
/// how much?" in one command.
///
/// USAGE
///   tinygpt compare base.tinygpt --lora my.lora --corpus held-out.txt
///   tinygpt compare base.tinygpt --lora a.lora --lora b.lora \
///       --corpus held-out.txt   # composes both adapters
///
/// Prints a side-by-side table of cross-entropy loss / BPB / perplexity
/// for base alone vs base + adapter(s), with delta and verdict.
enum Compare {
    static func run(args: [String]) {
        var basePath: String?
        var corpusPath: String?
        var loraPaths: [String] = []
        var nBatches = 30
        var batchSize: Int? = nil
        var sampleAfter = true
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--corpus":  corpusPath = args[i+1]; i += 2
            case "--lora":    loraPaths.append(args[i+1]); i += 2
            case "--batches": nBatches = Int(args[i+1]) ?? nBatches; i += 2
            case "--batch":   batchSize = Int(args[i+1]); i += 2
            case "--no-sample": sampleAfter = false; i += 1
            case "-h", "--help": exitUsage()
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                basePath = args[i]; i += 1
            }
        }
        guard let basePath = basePath else { fputs("missing base.tinygpt\n", stderr); exitUsage() }
        guard let corpusPath = corpusPath else { fputs("--corpus required\n", stderr); exitUsage() }
        guard !loraPaths.isEmpty else { fputs("at least one --lora required\n", stderr); exitUsage() }

        // Read base + corpus + adapters once.
        let file: TinyGPTFile
        do { file = try TinyGPTFileReader.read(URL(fileURLWithPath: basePath)) }
        catch { fputs("base read failed: \(error)\n", stderr); exit(1) }
        let h = file.header.config
        let cfg = ModelConfig(
            vocabSize: 256,
            contextLength: h.ctx ?? 256, nLayers: h.layers ?? 12,
            nHeads: h.heads ?? 8, dModel: h.dModel ?? 256, dMlp: h.dMlp ?? 1024
        )
        let corpus: ByteCorpus
        do { corpus = try ByteCorpus(contentsOf: URL(fileURLWithPath: corpusPath)) }
        catch { fputs("corpus read failed: \(error)\n", stderr); exit(1) }
        let B = batchSize ?? 8

        // For fair comparison, score BOTH models on the same windows.
        // Pre-generate the batches.
        var windows: [(x: MLXArray, y: MLXArray)] = []
        for _ in 0..<nBatches {
            windows.append(corpus.sampleBatch(batchSize: B, contextLength: cfg.contextLength))
        }

        print("""

        TinyGPT — base vs LoRA comparison
        ---------------------------------
        base:     \(URL(fileURLWithPath: basePath).lastPathComponent) (\(cfg.nLayers)L · d=\(cfg.dModel))
        adapters: \(loraPaths.count) — \(loraPaths.map { URL(fileURLWithPath: $0).lastPathComponent }.joined(separator: " + "))
        corpus:   \(URL(fileURLWithPath: corpusPath).lastPathComponent) (\(formatBytes(corpus.bytes.count)))
        windows:  \(nBatches) × batch \(B) × ctx \(cfg.contextLength) = \(formatNum(nBatches * B * cfg.contextLength)) tokens scored


        """)

        // 1) Score the BASE.
        print("scoring base…")
        let baseModel = TinyGPTModel(cfg)
        do { try TinyGPTWeightLoader.load(file, into: baseModel) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        let baseLoss = scoreOver(model: baseModel, windows: windows)

        // 2) Score base + LoRA(s) — fresh model so the base side isn't
        //    contaminated by LoRA-injected modules.
        print("scoring base + adapter(s)…")
        let loraModel = TinyGPTModel(cfg)
        do { try TinyGPTWeightLoader.load(file, into: loraModel) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        do {
            let adapters = try loraPaths.map { try LoraAdapterReader.read(URL(fileURLWithPath: $0)) }
            if adapters.count == 1 {
                try LoraAdapterReader.apply(adapters[0], to: loraModel)
            } else {
                let weights = [Float](repeating: 1.0, count: adapters.count)
                try LoraStackInjection.apply(adapters, weights: weights, to: loraModel)
            }
        } catch {
            fputs("adapter load failed: \(error)\n", stderr); exit(1)
        }
        let loraLoss = scoreOver(model: loraModel, windows: windows)

        // Report (cast Float → Double for CVarArg compatibility)
        let baseBPB = Double(baseLoss) / log(2.0)
        let baseP = exp(Double(baseLoss))
        let loraBPB = Double(loraLoss) / log(2.0)
        let loraP = exp(Double(loraLoss))
        let delta = Double(baseLoss) - Double(loraLoss)
        let pplDelta = baseP - loraP

        print("""

        RESULTS
        -------
                       loss      BPB     perplexity
        base           \(String(format: "%.3f", Double(baseLoss)))   \(String(format: "%.3f", baseBPB))     \(String(format: "%.2f", baseP))
        +adapter(s)    \(String(format: "%.3f", Double(loraLoss)))   \(String(format: "%.3f", loraBPB))     \(String(format: "%.2f", loraP))
        Δ              \(String(format: "%+.3f", -delta))   \(String(format: "%+.3f", -(baseBPB - loraBPB)))   \(String(format: "%+.2f", -pplDelta))
        """)

        if delta > 0.05 {
            let pplPctDrop = pplDelta / baseP * 100
            print("\n✓ adapter helped — \(String(format: "%.0f%%", pplPctDrop)) perplexity reduction on this corpus")
        } else if delta > 0 {
            print("\n· adapter helped slightly — \(String(format: "%.3f", delta)) lower loss")
        } else {
            print("\n⚠ adapter HURT performance — loss is \(String(format: "%.3f", -delta)) higher")
        }

        if sampleAfter {
            print("\nQUICK SAMPLE FROM EACH (greedy, prompt 'The ')")
            print("  base:")
            print("    " + sample(model: baseModel, cfg: cfg, prompt: "The ", n: 80))
            print("  +adapter:")
            print("    " + sample(model: loraModel, cfg: cfg, prompt: "The ", n: 80))
        }
    }

    private static func scoreOver(model: TinyGPTModel,
                                   windows: [(x: MLXArray, y: MLXArray)]) -> Float {
        var sum: Float = 0
        for (i, w) in windows.enumerated() {
            let l = model.loss(w.x, w.y)
            eval(l)
            let v = l.item(Float.self)
            sum += v
            if i < 3 || i == windows.count - 1 {
                fputs(String(format: "  win %3d  loss %.3f\n", i + 1, v), stderr)
            }
        }
        return sum / Float(windows.count)
    }

    private static func sample(model: TinyGPTModel, cfg: ModelConfig, prompt: String, n: Int) -> String {
        var bytes = [UInt8](prompt.utf8)
        var idx = MLXArray(bytes.map { Int32($0) }, [1, bytes.count])
        for _ in 0..<n {
            let T = idx.shape.last!
            let lo = max(0, T - cfg.contextLength)
            let cond = idx[0..., lo..<T]
            let logits = model(cond)
            let last = logits[0..., logits.shape[1] - 1, 0...]
            let next = argMax(last / MLXArray(Float(0.8)), axis: -1).reshaped([1, 1])
            eval(next)
            bytes.append(UInt8(Int(next.item(Int32.self)) & 0xff))
            idx = concatenated([idx, next.asType(idx.dtype)], axis: 1)
        }
        return (String(bytes: bytes, encoding: .utf8) ?? "<non-utf8>")
            .prefix(150).replacingOccurrences(of: "\n", with: "\\n").description
    }

    private static func formatNum(_ n: Int) -> String {
        let f = NumberFormatter(); f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }
    private static func formatBytes(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1f MB", Double(n)/1_000_000) }
        if n >= 1_000 { return String(format: "%.0f KB", Double(n)/1_000) }
        return "\(n) B"
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt compare <base.tinygpt> [options]

        --corpus path.txt    UTF-8 held-out text (required)
        --lora path.lora     Adapter to test (required; can repeat for composition)
        --batches N          Eval windows (default 30)
        --batch N            Batch size (default 8)
        --no-sample          Skip the side-by-side sample at the end
        """)
        exit(2)
    }
}
