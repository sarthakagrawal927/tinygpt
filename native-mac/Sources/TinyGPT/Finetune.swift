import Foundation
import MLX
import TinyGPTIO
import TinyGPTModel

/// `tinygpt finetune` — LoRA-fine-tune a checkpoint on a small text
/// corpus. The base weights stay frozen; only the rank-r adapter
/// matrices are trained. Adapter files are tiny (~100KB-1MB) and
/// portable: load a base + multiple adapters to switch "voices"
/// without reloading the base.
///
/// USAGE
///   tinygpt finetune base.tinygpt --corpus my-text.txt --out mine.lora
///   tinygpt finetune shakespeare.bin --corpus my-blog.txt --rank 8 --steps 200 --out blog.lora
enum Finetune {
    static func run(args: [String]) {
        var basePath: String?
        var corpusPath: String?
        var outPath: String?
        var rank = 4
        var alpha: Float = 8.0
        var steps = 200
        var lr: Float = 1e-3  // higher than full-finetune since adapter params are few
        var targetSuffixesArg = "q_proj,v_proj"
        var batchSize: Int? = nil
        var sampleEvery = 100
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--corpus":  corpusPath = args[i+1]; i += 2
            case "--out":     outPath = args[i+1]; i += 2
            case "--rank":    rank = Int(args[i+1]) ?? rank; i += 2
            case "--alpha":   alpha = Float(args[i+1]) ?? alpha; i += 2
            case "--steps":   steps = Int(args[i+1]) ?? steps; i += 2
            case "--lr":      lr = Float(args[i+1]) ?? lr; i += 2
            case "--targets": targetSuffixesArg = args[i+1]; i += 2
            case "--batch":   batchSize = Int(args[i+1]); i += 2
            case "--sample-every": sampleEvery = Int(args[i+1]) ?? sampleEvery; i += 2
            case "-h", "--help": exitUsage()
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                basePath = args[i]; i += 1
            }
        }
        guard let basePath = basePath else { fputs("missing base.tinygpt\n", stderr); exitUsage() }
        guard let corpusPath = corpusPath else { fputs("--corpus required\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out required\n", stderr); exitUsage() }

        let targetSuffixes = targetSuffixesArg.split(separator: ",").map(String.init)
        let loraCfg = LoraConfig(rank: rank, alpha: alpha, targetSuffixes: targetSuffixes)

        // Load the base model
        let baseURL = URL(fileURLWithPath: basePath)
        let file: TinyGPTFile
        do { file = try TinyGPTFileReader.read(baseURL) }
        catch { fputs("error reading base: \(error)\n", stderr); exit(1) }
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

        // Inject LoRA + freeze base
        LoraInjection.inject(model, config: loraCfg)
        LoraInjection.freezeBase(model)
        let nTrainable = LoraInjection.trainableParamCount(in: model)
        let nTotal = model.numParameters()

        // Load corpus
        let corpusURL = URL(fileURLWithPath: corpusPath)
        let corpus: ByteCorpus
        do { corpus = try ByteCorpus(contentsOf: corpusURL) }
        catch { fputs("error reading corpus: \(error)\n", stderr); exit(1) }

        let B = batchSize ?? defaultBatch(cfg)
        print("""

        TinyGPT — LoRA fine-tune
        ------------------------
        base:           \(basePath)
        corpus:         \(corpusPath) (\(formatBytes(corpus.bytes.count)))
        config:         \(cfg.nLayers)L · d=\(cfg.dModel) · ctx=\(cfg.contextLength)
        LoRA:           rank=\(rank) alpha=\(alpha) targets=\(targetSuffixes.joined(separator: ","))
        trainable:      \(formatNum(nTrainable))  /  total \(formatNum(nTotal))  (\(String(format: "%.2f%%", 100 * Float(nTrainable) / Float(nTotal))))
        steps:          \(steps)
        batch / lr:     \(B) / \(lr)
        device:         \(Device.defaultDevice())

        """)
        fflush(stdout)

        let trainer = Trainer(model: model, learningRate: lr, weightDecay: 0.0, compileStep: false)
        let t0 = Date()
        var lastLoss: Float = 0
        for step in 0..<steps {
            let (x, y) = corpus.sampleBatch(batchSize: B, contextLength: cfg.contextLength)
            lastLoss = trainer.step(inputs: x, targets: y)
            if step == 0 || (step + 1) % 25 == 0 || step == steps - 1 {
                let elapsed = -t0.timeIntervalSinceNow
                let sps = Double(step + 1) / elapsed
                fputs(String(format: "  step %4d/%4d  loss %.3f  · %.1f step/s · eta %.0fs\n",
                             step + 1, steps, lastLoss, sps,
                             Double(steps - step - 1) / sps), stderr)
            }
            if (step + 1) % sampleEvery == 0 || step == steps - 1 {
                printSample(model: model, cfg: cfg, tag: "step \(step + 1)")
            }
        }
        let elapsed = -t0.timeIntervalSinceNow
        print(String(format: "\ndone — %d steps in %.1fs (%.1f step/s) · final loss %.3f",
                     steps, elapsed, Double(steps) / elapsed, lastLoss))

        // Save the adapter (NOT the full model — only the small A, B matrices)
        do {
            try LoraAdapterWriter.write(model: model, baseConfig: cfg,
                                         loraConfig: loraCfg, finalLoss: lastLoss,
                                         to: URL(fileURLWithPath: outPath))
            let attrs = try FileManager.default.attributesOfItem(atPath: outPath)
            let sz = attrs[.size] as? Int ?? 0
            print("✓ wrote \(outPath)  (\(formatBytes(sz)))")
        } catch {
            fputs("save failed: \(error)\n", stderr); exit(1)
        }
    }

    private static func printSample(model: TinyGPTModel, cfg: ModelConfig, tag: String) {
        let promptBytes: [UInt8] = [UInt8]("The ".utf8)
        var idx = MLXArray(promptBytes.map { Int32($0) }, [1, promptBytes.count])
        var bytes = promptBytes
        for _ in 0..<60 {
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
        let s = (String(bytes: bytes, encoding: .utf8) ?? "<non-utf8>")
            .prefix(120).replacingOccurrences(of: "\n", with: "\\n")
        fputs("    [\(tag)] \(s)\n", stderr)
    }

    private static func defaultBatch(_ cfg: ModelConfig) -> Int {
        if cfg.dModel >= 512 { return 4 }
        if cfg.dModel >= 256 { return 8 }
        return 16
    }
    private static func formatNum(_ n: Int) -> String {
        let f = NumberFormatter(); f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }
    private static func formatBytes(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1f MB", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0f KB", Double(n) / 1_000) }
        return "\(n) B"
    }
    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt finetune <base.tinygpt> [options]

        --corpus path.txt        UTF-8 text to fine-tune on (required)
        --out path.lora          Where to save the adapter (required)
        --rank N                 LoRA rank (default 4; try 8 for more capacity)
        --alpha F                LoRA scale (default 8.0; usually 2× rank)
        --steps N                Training steps (default 200)
        --lr F                   Learning rate (default 1e-3; higher than full-finetune)
        --targets q,v[,k,o,...]  Which Linear modules to wrap (default: q_proj,v_proj)
        --batch N                Batch size (default by preset)
        --sample-every N         Print sample every N steps (default 100)
        """)
        exit(2)
    }
}
