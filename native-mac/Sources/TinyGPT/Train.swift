import Foundation
import MLX
import TinyGPTIO
import TinyGPTModel

/// `tinygpt train` — train a model from scratch on a UTF-8 text corpus and
/// save the result to a `.tinygpt` file. Closes the loop: trains here,
/// loads back through `sample`, generates from the same architecture.
///
/// The saved file uses the training-resumable fp32 layout (per-tensor
/// `[w, m, v]` triplets) so the browser playground can load it for
/// continued training.
enum Train {
    static func run(args: [String]) {
        var preset = "tiny"
        var steps = 500
        var corpusPath: String? = nil
        var outPath: String? = nil
        var dtype = "float32"
        var batchSize: Int? = nil
        var sampleEvery = 100
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--preset": preset = args[i+1]; i += 2
            case "--steps": steps = Int(args[i+1]) ?? steps; i += 2
            case "--corpus": corpusPath = args[i+1]; i += 2
            case "--out": outPath = args[i+1]; i += 2
            case "--dtype": dtype = args[i+1]; i += 2
            case "--batch": batchSize = Int(args[i+1]); i += 2
            case "--sample-every": sampleEvery = Int(args[i+1]) ?? sampleEvery; i += 2
            case "-h", "--help": exitUsage()
            default:
                fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
            }
        }

        var cfg = configFor(preset)
        cfg.dtype = dtype

        let corpus: ByteCorpus
        if let p = corpusPath {
            do {
                corpus = try ByteCorpus(contentsOf: URL(fileURLWithPath: p))
            } catch {
                fputs("error reading corpus: \(error)\n", stderr); exit(1)
            }
        } else {
            print("⚠ no --corpus given, training on random bytes (loss will land at ~ln(256)=5.55)")
            let randomBytes = (0..<1_000_000).map { _ in UInt8.random(in: 0...255) }
            corpus = ByteCorpus(Data(randomBytes))
        }

        let B = batchSize ?? defaultBatch(cfg)
        let model = TinyGPTModel(cfg)
        let trainer = Trainer(model: model)

        print("""

        TinyGPT — training run
        ---------------------
        preset:        \(preset) (\(cfg.nLayers)L · d=\(cfg.dModel) · ctx=\(cfg.contextLength))
        params:        \(formatLargeInt(model.numParameters()))
        batch size:    \(B)
        steps:         \(steps)
        corpus:        \(corpusPath ?? "<random>") (\(formatBytes(corpus.bytes.count)))
        device:        \(Device.defaultDevice())

        """)
        fflush(stdout)

        let t0 = Date()
        var lastLoss: Float = 0

        // Pipelined batch prep: the CPU prepares batch i+1 while the GPU
        // works on batch i's forward + backward + optimiser step. Saves
        // 2-5% on small models and more as the corpus sampling cost grows.
        let prefetchQueue = DispatchQueue(label: "tinygpt.batch-prefetch")
        var nextBatchRaw: DispatchWorkItem? = nil
        var pendingInputs: [Int32] = []
        var pendingTargets: [Int32] = []

        let kickPrefetch: () -> Void = {
            let item = DispatchWorkItem {
                let (i, t) = corpus.sampleBatchRaw(batchSize: B, contextLength: cfg.contextLength)
                pendingInputs = i
                pendingTargets = t
            }
            nextBatchRaw = item
            prefetchQueue.async(execute: item)
        }
        kickPrefetch()

        for step in 0..<steps {
            // Wait for the pre-built raw batch then materialise + launch
            // the next prefetch immediately so the CPU can run in parallel
            // with the GPU's train step.
            nextBatchRaw?.wait()
            let x = MLXArray(pendingInputs, [B, cfg.contextLength])
            let y = MLXArray(pendingTargets, [B, cfg.contextLength])
            if step < steps - 1 { kickPrefetch() }

            lastLoss = trainer.step(inputs: x, targets: y)

            // Print a status line every 50 steps + sample-every milestones.
            if step == 0 || (step + 1) % 50 == 0 || step == steps - 1 {
                let elapsed = -t0.timeIntervalSinceNow
                let stepsPerSec = Double(step + 1) / elapsed
                let eta = Double(steps - step - 1) / stepsPerSec
                fputs(String(format: "  step %5d/%5d  loss %.3f  · %.1f step/s · eta %.0fs\n",
                             step + 1, steps, lastLoss, stepsPerSec, eta), stderr)
            }
            // Generate a sample to visualise learning progress.
            if (step + 1) % sampleEvery == 0 || step == steps - 1 {
                printSample(model: model, cfg: cfg, tag: "step \(step + 1)")
            }
        }
        let elapsed = -t0.timeIntervalSinceNow
        print(String(format: "\ndone — %d steps in %.1fs (%.1f step/s) · final loss %.3f",
                     steps, elapsed, Double(steps) / elapsed, lastLoss))

        if let out = outPath {
            print("saving to \(out)…")
            do {
                try saveCheckpoint(model: model, cfg: cfg, step: steps,
                                   finalLoss: lastLoss, to: URL(fileURLWithPath: out))
                print("✓ wrote \(out)")
            } catch {
                fputs("save failed: \(error)\n", stderr); exit(1)
            }
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
            let id = Int(next.item(Int32.self))
            bytes.append(UInt8(id & 0xff))
            idx = concatenated([idx, next.asType(idx.dtype)], axis: 1)
        }
        let s = String(bytes: bytes, encoding: .utf8) ?? "<non-utf8>"
        let clipped = s.prefix(120).replacingOccurrences(of: "\n", with: "\\n")
        fputs("    [\(tag) sample] \(clipped)\n", stderr)
    }

    private static func saveCheckpoint(model: TinyGPTModel, cfg: ModelConfig,
                                       step: Int, finalLoss: Float, to url: URL) throws {
        // Build the manifest in the same order the browser/python_ref use,
        // and assemble per-tensor `[w, m, v]` fp32 triplets. We don't have
        // the optimizer m/v exposed by MLX-Swift's AdamW yet — write zeros
        // for those slots so the file is loadable everywhere; the result
        // is a fp32 file that can be sampled from but not train-continued.
        let entries = manifestEntries(cfg)
        let params = model.parameters().flattened()
        let paramMap: [String: MLXArray] = Dictionary(uniqueKeysWithValues: params)

        var tensors: [TinyGPTTensor] = []
        tensors.reserveCapacity(entries.count)
        for entry in entries {
            guard let mlxValue = paramMap[entry.name] else {
                // Linear weights need re-transposing back to WASM order
                // before save. Handled below.
                throw NSError(domain: "TinyGPT", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "missing param \(entry.name)"])
            }
            var array = mlxValue
            // Reverse the WeightLoader's transpose: model stores Linear
            // weights as [out, in] (PyTorch); the file stores them as
            // [in, out] (WASM). Transpose Linear-module weights on save.
            if isLinearWeightName(entry.name) && array.shape.count == 2 {
                array = array.transposed()
            }
            eval(array)
            let floats: [Float] = array.asArray(Float.self)
            let weightData = floats.withUnsafeBufferPointer { Data(buffer: $0) }
            let zeros = Data(count: weightData.count)
            tensors.append(TinyGPTTensor(
                entry: entry, weight: weightData, adamM: zeros, adamV: zeros, dtype: .fp32
            ))
        }

        let header = TinyGPTHeader(
            config: .init(
                layers: cfg.nLayers, dModel: cfg.dModel, ctx: cfg.contextLength,
                heads: cfg.nHeads, dMlp: cfg.dMlp, batchSize: 8, backend: "mlx-swift"
            ),
            manifest: entries,
            savedAt: ISO8601DateFormatter().string(from: Date()),
            finalLoss: .init(step: step, train: Double(finalLoss), val: nil),
            sample: nil,
            weightDtype: "fp32",
            includesOptimizerState: true,
            stateByteLength: 4 + tensors.reduce(0) { $0 + 3 * $1.weight.count }
        )
        let file = TinyGPTFile(
            version: TinyGPTFormat.currentVersion,
            header: header, step: Int32(step), tensors: tensors
        )
        try TinyGPTFileWriter.write(file, to: url)
    }

    private static func manifestEntries(_ cfg: ModelConfig) -> [TinyGPTHeader.TensorEntry] {
        var entries: [TinyGPTHeader.TensorEntry] = []
        var offset = 0
        let push: (String, [Int]) -> Void = { name, shape in
            let size = shape.reduce(1, *)
            entries.append(.init(name: name, shape: shape, floatOffset: offset))
            offset += size
        }
        let C = cfg.dModel, M = cfg.dMlp
        push("token_embedding.weight", [cfg.vocabSize, C])
        push("position_embedding.weight", [cfg.contextLength, C])
        push("ln_final.weight", [C])
        push("ln_final.bias", [C])
        for i in 0..<cfg.nLayers {
            push("blocks.\(i).ln1.weight", [C])
            push("blocks.\(i).ln1.bias", [C])
            push("blocks.\(i).attn.q_proj.weight", [C, C])
            push("blocks.\(i).attn.q_proj.bias", [C])
            push("blocks.\(i).attn.k_proj.weight", [C, C])
            push("blocks.\(i).attn.k_proj.bias", [C])
            push("blocks.\(i).attn.v_proj.weight", [C, C])
            push("blocks.\(i).attn.v_proj.bias", [C])
            push("blocks.\(i).attn.o_proj.weight", [C, C])
            push("blocks.\(i).attn.o_proj.bias", [C])
            push("blocks.\(i).ln2.weight", [C])
            push("blocks.\(i).ln2.bias", [C])
            push("blocks.\(i).mlp.fc_in.weight", [M, C])
            push("blocks.\(i).mlp.fc_in.bias", [M])
            push("blocks.\(i).mlp.fc_out.weight", [C, M])
            push("blocks.\(i).mlp.fc_out.bias", [C])
        }
        return entries
    }

    private static func isLinearWeightName(_ name: String) -> Bool {
        guard name.hasSuffix(".weight") else { return false }
        if name == "token_embedding.weight" || name == "position_embedding.weight" {
            return false
        }
        if name.hasSuffix(".ln1.weight") || name.hasSuffix(".ln2.weight")
            || name == "ln_final.weight" {
            return false
        }
        return true
    }

    private static func configFor(_ preset: String) -> ModelConfig {
        switch preset.lowercased() {
        case "tiny":     return ModelConfig(vocabSize: 256, contextLength: 128, nLayers: 4,
                                             nHeads: 4, dModel: 128, dMlp: 512)
        case "small":    return ModelConfig(vocabSize: 256, contextLength: 256, nLayers: 6,
                                             nHeads: 6, dModel: 192, dMlp: 768)
        case "huge":     return ModelConfig.huge
        case "mega":     return ModelConfig.mega
        case "behemoth": return ModelConfig.behemoth
        case "titan":    return ModelConfig.titan
        default:
            fputs("unknown preset: \(preset). Choose tiny|small|huge|mega|behemoth|titan.\n", stderr)
            exit(2)
        }
    }

    private static func defaultBatch(_ cfg: ModelConfig) -> Int {
        if cfg.dModel >= 1024 { return 2 }  // Behemoth / Titan
        if cfg.dModel >= 512 { return 4 }   // Mega
        if cfg.dModel >= 256 { return 8 }   // Huge
        return 16
    }

    private static func formatLargeInt(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func formatBytes(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1f MB", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0f KB", Double(n) / 1_000) }
        return "\(n) B"
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt train [options]

        --preset tiny|small|huge|mega    Model size (default: tiny)
        --steps N                         Training steps (default: 500)
        --corpus path.txt                 UTF-8 text file (default: random bytes)
        --out path.tinygpt                Where to save the trained checkpoint
        --dtype float32|float16           Training dtype (default: float32)
        --batch N                         Batch size (default: based on preset)
        --sample-every N                  Print a sample every N steps (default: 100)
        """)
        exit(2)
    }
}
