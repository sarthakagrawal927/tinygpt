import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import TinyGPTIO
import TinyGPTModel

/// `tinygpt train-extractor` — train a tool-call extractor (mini-router).
///
/// Reads a JSONL of `{"query": "...", "tool": "name"}` pairs (built by
/// `tinygpt extractor-data`), builds a `ToolRouterModel`, runs a
/// standard AdamW training loop, and saves a `.tinygpt` checkpoint +
/// sidecar `.labels.json` file.
///
/// The checkpoint reuses the existing `.tinygpt` binary format — same
/// readers/writers, same manifest shape, with the extra `router_head`
/// tensor at the end of the manifest list. The kind ("router" vs LM)
/// is implied by the manifest: a router checkpoint has a `router_head.
/// weight` tensor and NO `lm_head.weight` (the LM head is dropped
/// outright). Inference / inspect tools can detect this by scanning
/// the manifest.
///
/// FLAGS
///   <data>                  Training corpus JSONL (positional)
///   --preset tiny|small     Architecture preset (default: tiny)
///   --vocab-size N          Vocab size override (default: 256, byte-level)
///   --context N             Sequence length cap (default: 128)
///   --steps N               Training steps (default: 500)
///   --batch B               Batch size (default: 32)
///   --lr F                  Max learning rate (default: 3e-4)
///   --warmup N              Linear warmup steps (default: 50)
///   --val-split F           Fraction held out for val loss (default: 0.05)
///   --val-every N           Val eval cadence (default: 100)
///   --out <path>            Output checkpoint path (default: router.tinygpt)
///   --seed S                Random seed (default: 0)
///   --dry-run               Build the model + report sizes; don't train
///
/// EXAMPLES
///   tinygpt train-extractor router_data.jsonl --out router.tinygpt
///   tinygpt train-extractor router_data.jsonl --preset small --steps 1000
enum TrainExtractor {

    static func run(args: [String]) {
        if ProcessInfo.processInfo.environment["TINYGPT_DISABLE_QOS"] != "1" {
            TrainSupport.bumpQoSToUserInteractive()
        }
        var dataPath: String? = nil
        var preset = "tiny"
        var vocabSize = 256
        var contextLength = 128
        var steps = 500
        var batchSize = 32
        var maxLR: Float = 3e-4
        var minLR: Float = 3e-5
        var warmup = 50
        var valSplit: Double = 0.05
        var valEvery = 100
        var outPath = "router.tinygpt"
        var seed: UInt64 = 0
        var dryRun = false

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--preset":
                guard i + 1 < args.count else { exitUsage() }
                preset = args[i + 1]; i += 2
            case "--vocab-size":
                guard i + 1 < args.count else { exitUsage() }
                vocabSize = Int(args[i + 1]) ?? vocabSize; i += 2
            case "--context":
                guard i + 1 < args.count else { exitUsage() }
                contextLength = Int(args[i + 1]) ?? contextLength; i += 2
            case "--steps":
                guard i + 1 < args.count else { exitUsage() }
                steps = Int(args[i + 1]) ?? steps; i += 2
            case "--batch":
                guard i + 1 < args.count else { exitUsage() }
                batchSize = Int(args[i + 1]) ?? batchSize; i += 2
            case "--lr":
                guard i + 1 < args.count else { exitUsage() }
                maxLR = Float(args[i + 1]) ?? maxLR; i += 2
            case "--min-lr":
                guard i + 1 < args.count else { exitUsage() }
                minLR = Float(args[i + 1]) ?? minLR; i += 2
            case "--warmup":
                guard i + 1 < args.count else { exitUsage() }
                warmup = Int(args[i + 1]) ?? warmup; i += 2
            case "--val-split":
                guard i + 1 < args.count else { exitUsage() }
                valSplit = Double(args[i + 1]) ?? valSplit; i += 2
            case "--val-every":
                guard i + 1 < args.count else { exitUsage() }
                valEvery = Int(args[i + 1]) ?? valEvery; i += 2
            case "--out":
                guard i + 1 < args.count else { exitUsage() }
                outPath = args[i + 1]; i += 2
            case "--seed":
                guard i + 1 < args.count else { exitUsage() }
                seed = UInt64(args[i + 1]) ?? seed; i += 2
            case "--dry-run":
                dryRun = true; i += 1
            case "-h", "--help":
                exitUsage()
            default:
                if args[i].hasPrefix("-") {
                    fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
                }
                dataPath = args[i]; i += 1
            }
        }

        guard let dataPath = dataPath else {
            fputs("train-extractor: missing <data.jsonl>\n", stderr); exitUsage()
        }

        // Load data.
        let dataURL = URL(fileURLWithPath: dataPath)
        let examples: [RouterExample]
        do {
            examples = try loadJSONL(url: dataURL)
        } catch {
            fputs("train-extractor: load failed: \(error)\n", stderr); exit(1)
        }
        guard !examples.isEmpty else {
            fputs("train-extractor: corpus is empty\n", stderr); exit(1)
        }
        // Build the label table from the union of tool names.
        let labelNames = Array(Set(examples.map { $0.tool })).sorted()
        let labelIndex: [String: Int] = Dictionary(
            uniqueKeysWithValues: labelNames.enumerated().map { ($0.element, $0.offset) }
        )
        fputs("train-extractor: \(examples.count) examples across \(labelNames.count) classes\n", stderr)
        if labelNames.count < 2 {
            fputs("train-extractor: need at least 2 distinct tool names to train a classifier\n", stderr)
            exit(1)
        }

        // Build the model.
        let (cfg, numClasses): (ModelConfig, Int) = {
            switch preset.lowercased() {
            case "small":
                return ToolRouterModel.smallPreset(
                    vocabSize: vocabSize, contextLength: contextLength,
                    numClasses: labelNames.count)
            default:
                return ToolRouterModel.tinyPreset(
                    vocabSize: vocabSize, contextLength: contextLength,
                    numClasses: labelNames.count)
            }
        }()
        _ = numClasses
        MLXRandom.seed(seed)
        let model = ToolRouterModel(cfg, numClasses: labelNames.count, pooling: .mean)
        eval(model)
        let params = model.numParameters()
        fputs("train-extractor: model \(preset)  ·  \(formatLargeInt(params)) params  ·  ctx \(cfg.contextLength)\n", stderr)

        if dryRun {
            fputs("--dry-run: skipping training\n", stderr)
            return
        }

        // Train/val split.
        var shuffled = examples
        var rng = SystemRandomNumberGenerator()
        shuffled.shuffle(using: &rng)
        let valCount = max(0, Int(Double(shuffled.count) * valSplit))
        let val = Array(shuffled.prefix(valCount))
        let train = Array(shuffled.dropFirst(valCount))
        fputs("train-extractor: split \(train.count) train / \(val.count) val\n", stderr)

        // Training loop — closure-captured AdamW + valueAndGrad on the
        // router. Same pattern as `TrainHeads.swift`.
        let lossFn = { (m: ToolRouterModel, x: MLXArray, y: MLXArray) -> MLXArray in
            return m.loss(x, y)
        }
        let gradFn = valueAndGrad(model: model, lossFn)
        let opt = AdamW(learningRate: maxLR, weightDecay: 0.01)

        TrainSupport.installSigintHandler()
        TrainSupport.stopRequested.reset()
        let t0 = Date()
        var lastLoss: Float = 0
        var lastValLoss: Float? = nil
        var lossCurve: [(Int, Float)] = []
        var lastStep = 0
        var stoppedEarly = false

        for step in 0..<steps {
            if TrainSupport.stopRequested.isSet { stoppedEarly = true; break }
            // LR schedule.
            let lr = TrainSupport.lrAt(
                step: step, total: steps, warmup: warmup,
                maxLR: maxLR, minLR: minLR)
            opt.learningRate = lr

            let (x, y) = sampleBatch(
                examples: train, batchSize: batchSize,
                contextLength: cfg.contextLength,
                vocabSize: vocabSize, labelIndex: labelIndex)
            let (loss, grads) = gradFn(model, x, y)
            opt.update(model: model, gradients: grads)
            MLX.eval(loss, model, opt)
            lastLoss = loss.item(Float.self)
            lastStep = step + 1
            if step == 0 || (step + 1) % 25 == 0 || step == steps - 1 {
                lossCurve.append((step + 1, lastLoss))
                let elapsed = -t0.timeIntervalSinceNow
                let sps = Double(step + 1) / elapsed
                let eta = Double(steps - step - 1) / max(sps, 1e-6)
                let valTag = lastValLoss.map { String(format: "  val %.3f", $0) } ?? ""
                fputs(String(format: "  step %4d/%4d  loss %.3f  lr %.2e%@  · %.1f step/s · eta %.0fs\n",
                              step + 1, steps, lastLoss, lr, valTag, sps, eta), stderr)
            }
            if !val.isEmpty, (step + 1) % valEvery == 0 {
                let (vx, vy) = sampleBatch(
                    examples: val, batchSize: min(batchSize, val.count),
                    contextLength: cfg.contextLength,
                    vocabSize: vocabSize, labelIndex: labelIndex)
                let vloss = model.loss(vx, vy)
                MLX.eval(vloss)
                lastValLoss = vloss.item(Float.self)
            }
        }

        let elapsed = -t0.timeIntervalSinceNow
        let firstLoss = lossCurve.first?.1 ?? lastLoss
        fputs("\ntrain-extractor: \(stoppedEarly ? "interrupted" : "done") — \(lastStep) steps in \(String(format: "%.1f", elapsed))s\n", stderr)
        fputs(String(format: "  loss %.3f → %.3f (Δ=%.3f)%@\n",
                      firstLoss, lastLoss, firstLoss - lastLoss,
                      lastValLoss.map { String(format: "  · val %.3f", $0) } ?? ""), stderr)

        // Save weights + labels sidecar.
        let outURL = URL(fileURLWithPath: outPath)
        do {
            try saveCheckpoint(model: model, cfg: cfg, step: lastStep,
                                finalLoss: lastLoss, to: outURL)
            let labels = ToolRouterLabels(labels: labelNames)
            let labelsURL = ToolRouterLabels.sidecarURL(forCheckpoint: outURL)
            try labels.save(to: labelsURL)
            fputs("train-extractor: wrote \(outURL.path)\n", stderr)
            fputs("train-extractor: wrote \(labelsURL.path)\n", stderr)
        } catch {
            fputs("train-extractor: save failed: \(error)\n", stderr)
            exit(1)
        }
        if stoppedEarly { exit(130) }
    }

    // MARK: - Data types + loader

    struct RouterExample {
        let query: String
        let tool: String
    }

    static func loadJSONL(url: URL) throws -> [RouterExample] {
        let data = try Data(contentsOf: url)
        var out: [RouterExample] = []
        var line = Data()
        for byte in data {
            if byte == 0x0A {
                if !line.isEmpty,
                   let obj = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
                   let q = obj["query"] as? String,
                   let t = obj["tool"] as? String,
                   !q.isEmpty, !t.isEmpty {
                    out.append(RouterExample(query: q, tool: t))
                }
                line.removeAll(keepingCapacity: true)
            } else {
                line.append(byte)
            }
        }
        if !line.isEmpty,
           let obj = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
           let q = obj["query"] as? String,
           let t = obj["tool"] as? String,
           !q.isEmpty, !t.isEmpty {
            out.append(RouterExample(query: q, tool: t))
        }
        return out
    }

    // MARK: - Batching

    /// Encode a query → token-id vector (byte-level by default). Truncates
    /// or zero-pads to `contextLength`. Returns ids clamped to
    /// `[0, vocabSize-1]` so a BPE-mismatched vocab doesn't index out of
    /// range during early experiments.
    static func encode(_ s: String, contextLength: Int, vocabSize: Int) -> [Int32] {
        var ids = [UInt8](s.utf8).prefix(contextLength).map { Int32($0) }
        // Clamp to vocab range.
        for i in 0..<ids.count {
            if ids[i] < 0 || ids[i] >= Int32(vocabSize) {
                ids[i] = 0
            }
        }
        while ids.count < contextLength { ids.append(0) }
        return ids
    }

    static func sampleBatch(examples: [RouterExample], batchSize: Int,
                             contextLength: Int, vocabSize: Int,
                             labelIndex: [String: Int]) -> (MLXArray, MLXArray) {
        var inputs = [Int32](repeating: 0, count: batchSize * contextLength)
        var labels = [Int32](repeating: 0, count: batchSize)
        for i in 0..<batchSize {
            let ex = examples[Int.random(in: 0..<examples.count)]
            let ids = encode(ex.query, contextLength: contextLength, vocabSize: vocabSize)
            for j in 0..<contextLength {
                inputs[i * contextLength + j] = ids[j]
            }
            labels[i] = Int32(labelIndex[ex.tool] ?? 0)
        }
        let x = MLXArray(inputs, [batchSize, contextLength])
        let y = MLXArray(labels)
        return (x, y)
    }

    // MARK: - Checkpoint save

    /// Walk the model's parameter tree and write a `.tinygpt` file with
    /// a manifest covering every Module parameter. Reuses the file
    /// format unchanged — only difference is the presence of
    /// `router_head.weight` / `router_head.bias` and the absence of
    /// `lm_head`. Resumability is best-effort: AdamW state is zeroed
    /// the same way `Train.swift::writeCheckpoint` does.
    static func saveCheckpoint(model: ToolRouterModel, cfg: ModelConfig,
                                step: Int, finalLoss: Float, to url: URL) throws {
        let params = model.parameters().flattened()
        let paramMap: [String: MLXArray] = Dictionary(uniqueKeysWithValues: params)

        var entries: [TinyGPTHeader.TensorEntry] = []
        var tensors: [TinyGPTTensor] = []
        // Sort for deterministic on-disk order.
        let sortedKeys = paramMap.keys.sorted()
        for name in sortedKeys {
            let arr = paramMap[name]!
            eval(arr)
            let shape = arr.shape
            let elementCount = shape.reduce(1, *)
            let floats: [Float] = arr.asArray(Float.self)
            let weightData = floats.withUnsafeBufferPointer { Data(buffer: $0) }
            let zeros = Data(count: weightData.count)
            _ = elementCount
            let entry = TinyGPTHeader.TensorEntry(
                name: name,
                shape: shape
            )
            entries.append(entry)
            tensors.append(TinyGPTTensor(
                entry: entry, weight: weightData,
                adamM: zeros, adamV: zeros, dtype: .fp32
            ))
        }

        let header = TinyGPTHeader(
            config: .init(
                layers: cfg.nLayers, dModel: cfg.dModel,
                ctx: cfg.contextLength, heads: cfg.nHeads,
                dMlp: cfg.dMlp, batchSize: 0, backend: "mlx-swift",
                vocabSize: cfg.vocabSize == 256 ? nil : cfg.vocabSize,
                tokenizerSource: cfg.tokenizerSource
            ),
            manifest: entries,
            savedAt: ISO8601DateFormatter().string(from: Date()),
            finalLoss: .init(step: step, train: Double(finalLoss), val: nil),
            sample: "tool-router",
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

    // MARK: - Misc

    private static func formatLargeInt(_ n: Int) -> String {
        let f = NumberFormatter(); f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    static func exitUsage() -> Never {
        print("""
        usage: tinygpt train-extractor <data.jsonl> [flags]

          --preset tiny|small     model size (default: tiny)
          --vocab-size N          vocab size (default: 256, byte-level)
          --context N             sequence length (default: 128)
          --steps N               training steps (default: 500)
          --batch B               batch size (default: 32)
          --lr F                  max LR (default: 3e-4)
          --warmup N              linear warmup steps (default: 50)
          --val-split F           val fraction (default: 0.05)
          --val-every N           val cadence (default: 100)
          --out <path>            output (default: router.tinygpt)
          --seed S                random seed (default: 0)
          --dry-run               build + report sizes, don't train

        Output is a pair of files:
          <out>             — .tinygpt checkpoint (router weights)
          <out>.labels.json — sidecar mapping class index → tool name
        """)
        exit(2)
    }
}
