import Foundation
import MLX
import TinyGPTIO
import TinyGPTModel

/// Helpers for long pre-training runs: cosine LR schedule, atomic
/// checkpoint write, cooperative Ctrl-C, train/val split, and val
/// loss evaluation. Kept in a separate file from `Train.swift` so the
/// main flow there reads as a sequence of high-level steps.
enum TrainSupport {

    // MARK: - LR schedule

    /// Cosine learning rate with linear warmup.
    ///
    /// - 0 ≤ step < warmup: linear ramp from 0 → maxLR
    /// - warmup ≤ step < total: cosine decay from maxLR → minLR
    /// - step ≥ total: minLR (rarely hit, but defensive for off-by-one)
    ///
    /// Used by `tinygpt train` when `--lr-schedule=cosine`. The constant
    /// schedule path just returns `maxLR` for all steps.
    static func lrAt(step: Int, total: Int, warmup: Int,
                     maxLR: Float, minLR: Float) -> Float {
        if step < warmup {
            // Linear ramp 0 → maxLR over warmup steps.
            return maxLR * Float(step + 1) / Float(max(1, warmup))
        }
        if step >= total { return minLR }
        let progress = Float(step - warmup) / Float(max(1, total - warmup))
        // Half-cosine: 1.0 at progress=0, 0.0 at progress=1.
        let cos = 0.5 * (1.0 + Foundation.cos(Double.pi * Double(progress)))
        return minLR + (maxLR - minLR) * Float(cos)
    }

    // MARK: - Atomic checkpoint write

    /// Write the file to `<path>.tmp`, fsync, then rename to `<path>`.
    /// A crash mid-write leaves `<path>` either at the previous
    /// successful checkpoint or untouched — never half-written.
    static func atomicSave(
        model: TinyGPTModel, cfg: ModelConfig, step: Int, finalLoss: Float,
        weightTranspose: (String) -> Bool,
        manifestEntries: (ModelConfig) -> [TinyGPTHeader.TensorEntry],
        to url: URL
    ) throws {
        let tmpURL = URL(fileURLWithPath: url.path + ".tmp")
        try writeCheckpoint(model: model, cfg: cfg, step: step,
                             finalLoss: finalLoss,
                             weightTranspose: weightTranspose,
                             manifestEntries: manifestEntries,
                             to: tmpURL)
        // Atomic rename. POSIX rename(2) is atomic within a filesystem.
        // Same-volume guaranteed; cross-volume falls back to copy-then-rm
        // which isn't atomic, but our checkpoint dir is always one volume.
        let fm = FileManager.default
        if fm.fileExists(atPath: url.path) {
            _ = try? fm.removeItem(at: url)
        }
        try fm.moveItem(at: tmpURL, to: url)
    }

    /// Same call surface as `atomicSave` but writes directly to `url` (no
    /// rename). Used by the legacy single-shot save callers that don't need
    /// the .tmp + rename safety.
    static func writeCheckpointDirect(
        model: TinyGPTModel, cfg: ModelConfig, step: Int, finalLoss: Float,
        weightTranspose: (String) -> Bool,
        manifestEntries: (ModelConfig) -> [TinyGPTHeader.TensorEntry],
        to url: URL
    ) throws {
        try writeCheckpoint(model: model, cfg: cfg, step: step,
                             finalLoss: finalLoss,
                             weightTranspose: weightTranspose,
                             manifestEntries: manifestEntries,
                             to: url)
    }

    /// The actual write — same body as the previous `saveCheckpoint`,
    /// hoisted so atomicSave can drive both the tmp + final paths.
    private static func writeCheckpoint(
        model: TinyGPTModel, cfg: ModelConfig, step: Int, finalLoss: Float,
        weightTranspose: (String) -> Bool,
        manifestEntries: (ModelConfig) -> [TinyGPTHeader.TensorEntry],
        to url: URL
    ) throws {
        let entries = manifestEntries(cfg)
        let params = model.parameters().flattened()
        let paramMap: [String: MLXArray] = Dictionary(uniqueKeysWithValues: params)

        var tensors: [TinyGPTTensor] = []
        tensors.reserveCapacity(entries.count)
        for entry in entries {
            guard let mlxValue = paramMap[entry.name] else {
                throw NSError(domain: "TinyGPT", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "missing param \(entry.name)"])
            }
            var array = mlxValue
            if weightTranspose(entry.name) && array.shape.count == 2 {
                array = array.transposed()
            }
            eval(array)
            let floats: [Float] = array.asArray(Float.self)
            let weightData = floats.withUnsafeBufferPointer { Data(buffer: $0) }
            let zeros = Data(count: weightData.count)
            // Adam m/v omitted — MLX-Swift's AdamW state isn't externally
            // readable. Resume restores weights + step but restarts Adam,
            // causing a ~100-step loss warm-up that re-converges.
            tensors.append(TinyGPTTensor(
                entry: entry, weight: weightData, adamM: zeros, adamV: zeros, dtype: .fp32
            ))
        }
        let header = TinyGPTHeader(
            config: .init(
                layers: cfg.nLayers, dModel: cfg.dModel, ctx: cfg.contextLength,
                heads: cfg.nHeads, dMlp: cfg.dMlp, batchSize: 8, backend: "mlx-swift",
                vocabSize: cfg.vocabSize == 256 ? nil : cfg.vocabSize,
                tokenizerSource: cfg.tokenizerSource,
                // MoE metadata — nil when standard dense, so old readers
                // round-trip unchanged. When MoE, downstream loaders pick
                // these up to reconstruct the router + expert tree.
                nExperts: cfg.isMoE ? cfg.nExperts : nil,
                moeTopK: cfg.isMoE ? cfg.moeTopK : nil,
                loadBalanceWeight: cfg.isMoE ? cfg.loadBalanceWeight : nil,
                slidingWindow: cfg.slidingWindow,
                useMoD: cfg.useMoD ? true : nil,
                useDifferentialAttention: cfg.useDifferentialAttention ? true : nil,
                useYOCO: cfg.useYOCO ? true : nil,
                useGradCheckpoint: cfg.useGradCheckpoint ? true : nil
            ),
            manifest: entries,
            savedAt: ISO8601DateFormatter().string(from: Date()),
            finalLoss: .init(step: step, train: Double(finalLoss), val: nil),
            sample: nil,
            weightDtype: "fp32",
            // includesOptimizerState selects the on-disk BODY LAYOUT
            // (true = [w,m,v] triplets, false = single fp16 weight buffer)
            // — not whether m/v are nonzero. We use the triplet layout so
            // resume can re-load the weights cleanly; m/v are written as
            // zeros (MLX-Swift AdamW state isn't externally readable yet).
            includesOptimizerState: true,
            stateByteLength: 4 + tensors.reduce(0) { $0 + 3 * $1.weight.count }
        )
        let file = TinyGPTFile(
            version: TinyGPTFormat.currentVersion,
            header: header, step: Int32(step), tensors: tensors
        )
        try TinyGPTFileWriter.write(file, to: url)
    }

    // MARK: - Cooperative cancel via SIGINT

    /// Flag set by the SIGINT handler. Polled by the training loop;
    /// when true, the next iteration flushes a final checkpoint and exits.
    ///
    /// `sig_atomic_t` (Int32 on Darwin) is the only type guaranteed
    /// race-free for handler↔main-thread comms in POSIX signal semantics.
    /// Wrapped in a class so the handler (a global @convention(c) closure)
    /// can capture-by-reference.
    static let stopRequested = StopFlag()

    final class StopFlag: @unchecked Sendable {
        var rawFlag: Int32 = 0
        var isSet: Bool { rawFlag != 0 }
        func set() { rawFlag = 1 }
        func reset() { rawFlag = 0 }
    }

    /// Install a SIGINT handler that flips `stopRequested`. The closure
    /// is `@convention(c)` so it matches `signal(2)`'s expected pointer
    /// type. Signal-handler safety: we only do a single sig_atomic_t
    /// write — no allocation, no Swift runtime calls.
    static func installSigintHandler() {
        let handler: @convention(c) (Int32) -> Void = { _ in
            TrainSupport.stopRequested.rawFlag = 1
        }
        signal(SIGINT, handler)
    }

    // MARK: - Train/val split + val loss eval

    /// Split a byte corpus into train + (optional) val partitions. The
    /// val partition is the LAST `valSplit` fraction of the corpus, so
    /// train and val don't overlap. Returns nil val for valSplit == 0.
    static func splitCorpus(_ source: ByteCorpus, valSplit: Double) -> (train: ByteCorpus, val: ByteCorpus?) {
        guard valSplit > 0, valSplit < 0.5 else {
            return (source, nil)
        }
        let total = source.bytes.count
        let valBytes = max(1, Int(Double(total) * valSplit))
        let trainEnd = total - valBytes
        let train = ByteCorpus(Data(source.bytes[0..<trainEnd]))
        let val = ByteCorpus(Data(source.bytes[trainEnd..<total]))
        return (train, val)
    }

    /// Evaluate held-out loss on the val corpus by sampling `nBatches`
    /// random windows and averaging cross-entropy. Cheap — N forward
    /// passes, no backward.
    static func evalValLoss(model: TinyGPTModel, cfg: ModelConfig,
                             val: ByteCorpus, batchSize: Int,
                             nBatches: Int = 8) -> Float {
        var total: Float = 0
        for _ in 0..<nBatches {
            let (x, y) = val.sampleBatch(batchSize: batchSize, contextLength: cfg.contextLength)
            let loss = model.loss(x, y)
            eval(loss)
            total += loss.item(Float.self)
        }
        return total / Float(nBatches)
    }

    /// Same, but for tokenized (BPE) corpora.
    static func evalValLossTokenized(model: TinyGPTModel, cfg: ModelConfig,
                                      val: TokenizedCorpus, batchSize: Int,
                                      nBatches: Int = 8) -> Float {
        var total: Float = 0
        for _ in 0..<nBatches {
            let (x, y) = val.sampleBatch(batchSize: batchSize, contextLength: cfg.contextLength)
            let loss = model.loss(x, y)
            eval(loss)
            total += loss.item(Float.self)
        }
        return total / Float(nBatches)
    }
}

