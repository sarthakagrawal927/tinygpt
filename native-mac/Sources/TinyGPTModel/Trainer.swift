import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// Byte-level corpus loader. Materialises the whole corpus into memory as
/// a single `[UInt8]` and serves random `(B, T+1)` windows so each step
/// trains on a different chunk. Matches the browser's CPU-side sampler.
public final class ByteCorpus: Sendable {
    public let bytes: [UInt8]

    public init(_ data: Data) {
        self.bytes = Array(data)
    }

    public convenience init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        self.init(data)
    }

    /// Sample a batch: `(input [B, T] int32, target [B, T] int32)`.
    /// `target = input shifted by 1` so the model predicts next-byte.
    public func sampleBatch(batchSize B: Int, contextLength T: Int) -> (MLXArray, MLXArray) {
        let (inputs, targets) = sampleBatchRaw(batchSize: B, contextLength: T)
        return (MLXArray(inputs, [B, T]), MLXArray(targets, [B, T]))
    }

    /// Generate the raw Int32 windows without materialising MLXArrays. Used
    /// by the prefetcher so the CPU-side sampling runs concurrently with
    /// the GPU's previous-step compute.
    public func sampleBatchRaw(batchSize B: Int, contextLength T: Int) -> ([Int32], [Int32]) {
        precondition(bytes.count > T + 1, "corpus too small for context \(T)")
        var inputs = [Int32](repeating: 0, count: B * T)
        var targets = [Int32](repeating: 0, count: B * T)
        for i in 0..<B {
            let start = Int.random(in: 0..<(bytes.count - T - 1))
            for j in 0..<T {
                inputs[i * T + j] = Int32(bytes[start + j])
                targets[i * T + j] = Int32(bytes[start + j + 1])
            }
        }
        return (inputs, targets)
    }
}

/// Token-id corpus loader. Same `sampleBatch` interface as `ByteCorpus`,
/// but the underlying buffer is already-tokenised `Int32` ids — used when
/// fine-tuning an HF model whose embedding table is BPE-indexed (vocab in
/// the tens of thousands), so feeding raw bytes would index into a tiny
/// slice of the vocab and train the LoRA against a wrong distribution.
///
/// Callers build it from any tokenizer (typically `HFTokenizer.encode`)
/// — the corpus doesn't care which scheme produced the ids.
public final class TokenizedCorpus: Sendable {
    public let tokens: [Int32]
    public let vocabSize: Int

    public init(tokens: [Int32], vocabSize: Int) {
        self.tokens = tokens
        self.vocabSize = vocabSize
    }

    /// Sample a batch: `(input [B, T] int32, target [B, T] int32)`.
    public func sampleBatch(batchSize B: Int, contextLength T: Int) -> (MLXArray, MLXArray) {
        let (inputs, targets) = sampleBatchRaw(batchSize: B, contextLength: T)
        return (MLXArray(inputs, [B, T]), MLXArray(targets, [B, T]))
    }

    public func sampleBatchRaw(batchSize B: Int, contextLength T: Int) -> ([Int32], [Int32]) {
        precondition(tokens.count > T + 1, "tokenized corpus too small for context \(T) (got \(tokens.count) tokens)")
        var inputs = [Int32](repeating: 0, count: B * T)
        var targets = [Int32](repeating: 0, count: B * T)
        for i in 0..<B {
            let start = Int.random(in: 0..<(tokens.count - T - 1))
            for j in 0..<T {
                inputs[i * T + j] = tokens[start + j]
                targets[i * T + j] = tokens[start + j + 1]
            }
        }
        return (inputs, targets)
    }

    /// Hold out the last `valSplit` fraction as a validation set. Same
    /// semantics as `TrainSupport.splitCorpus` for byte corpora.
    public func split(valSplit: Double) -> (train: TokenizedCorpus, val: TokenizedCorpus?) {
        guard valSplit > 0, valSplit < 0.5 else { return (self, nil) }
        let total = tokens.count
        let valCount = max(1, Int(Double(total) * valSplit))
        let trainEnd = total - valCount
        let train = TokenizedCorpus(tokens: Array(tokens[0..<trainEnd]), vocabSize: vocabSize)
        let val = TokenizedCorpus(tokens: Array(tokens[trainEnd..<total]), vocabSize: vocabSize)
        return (train, val)
    }
}

/// Async batch prefetcher — pipelines CPU-side batch construction with the
/// previous step's GPU compute. Maintains one pre-built batch ahead.
public actor BatchPrefetcher {
    private let corpus: ByteCorpus
    private let batchSize: Int
    private let contextLength: Int

    public init(corpus: ByteCorpus, batchSize: Int, contextLength: Int) {
        self.corpus = corpus
        self.batchSize = batchSize
        self.contextLength = contextLength
    }

    public func next() -> ([Int32], [Int32]) {
        corpus.sampleBatchRaw(batchSize: batchSize, contextLength: contextLength)
    }
}

/// Global L2-norm gradient clipping. Computes `‖g‖₂` across every
/// parameter, then uniformly scales each leaf by `min(1, maxNorm / ‖g‖₂)`.
/// Standard transformer-LM training stability lever — without it, the
/// occasional spike (early steps, rare token in a long sequence) can
/// blow up bf16 weights past the point the optimiser recovers from.
///
/// All ops are MLX ops, so this composes cleanly inside `compile`.
public func clipGradNorm(_ grads: ModuleParameters, maxNorm: Float) -> ModuleParameters {
    var sumSq = MLXArray(Float(0))
    for (_, g) in grads.flattened() {
        sumSq = sumSq + (g * g).sum()
    }
    let norm = MLX.sqrt(sumSq)
    // scale ≤ 1 ALWAYS (we never amplify) — `minimum(1, ratio)`.
    let scale = MLX.minimum(MLXArray(Float(1)),
                            MLXArray(maxNorm) / (norm + MLXArray(Float(1e-6))))
    return grads.mapValues { g in g * scale }
}

/// AdamW + value-and-grad train loop. One `step()` call does a full
/// forward + backward + optimiser update and returns the scalar loss.
///
/// Supports gradient accumulation via `accumulatedStep(microBatches:)` —
/// runs N micro-batches, sums gradients element-wise, divides by N, and
/// applies one optimizer update. Useful when the effective batch you
/// want exceeds memory: ctx=1024 × B=8 might OOM, but ctx=1024 × B=2
/// repeated 4× gives the same effective batch with ¼ the memory cost.
public final class Trainer {
    public let model: TinyGPTModel
    /// Generic optimiser handle — any of AdamW, Lion, Sophia, Muon, or
    /// Adafactor (the latter wrapped in `AdafactorAdapter`). Exposed
    /// through the `Optimizer & LearningRateMutable` composition so
    /// the schedule code can both step it and adjust its learning rate.
    public let optimizer: any Optimizer & LearningRateMutable
    /// Which optimiser kind backs `optimizer`, for diagnostics + the
    /// per-step LR-scheduler bookkeeping. Defaults to .adamw when the
    /// older AdamW-only initializer path is used.
    public let optimizerKind: OptimizerKind
    public private(set) var stepCount: Int = 0
    /// L2 norm cap for gradient clipping. `nil` = off; `1.0` is the
    /// transformer-LM default.
    public let gradClipNorm: Float?

    /// GaLore manager — `nil` when GaLore is disabled (the common case
    /// today). When non-nil, every step projects 2-D weight gradients
    /// through a rank-R basis before the optimiser sees them.
    /// See `GaLore.swift` for the details.
    public let galore: GaLoreManager?

    /// Layer-wise LR decay factor (default 1.0 = no decay). When < 1,
    /// each block's gradient is multiplied by `factor^(L - 1 - i)` so
    /// deeper layers get the full LR. Cheap — one MLX scalar multiply
    /// per leaf.
    public let lrLayerDecay: Float

    /// Compiled (graph-traced) train step. MLX-Swift's `compile` traces the
    /// step the first time it's called and reuses the kernel-launch sequence
    /// thereafter — the single biggest win over an interpreted train loop.
    private let trainStepFn: (MLXArray, MLXArray) -> MLXArray
    private let gradFn: (TinyGPTModel, MLXArray, MLXArray) -> (MLXArray, ModuleParameters)
    private let useCompile: Bool

    public init(
        model: TinyGPTModel,
        learningRate: Float = 3e-4,
        weightDecay: Float = 0.1,
        betas: (Float, Float) = (0.9, 0.95),
        eps: Float = 1e-8,
        compileStep: Bool = true,
        gradClipNorm: Float? = nil,
        optimizer optimizerKind: OptimizerKind = .adamw,
        galore: GaLoreManager? = nil,
        lrLayerDecay: Float = 1.0
    ) {
        self.model = model
        self.useCompile = compileStep
        self.gradClipNorm = gradClipNorm
        self.optimizerKind = optimizerKind
        self.galore = galore
        self.lrLayerDecay = lrLayerDecay
        self.optimizer = makeOptimizer(
            kind: optimizerKind,
            learningRate: learningRate,
            weightDecay: weightDecay,
            betas: betas,
            eps: eps
        )
        // value_and_grad of the loss function, captured to apply via optimizer.
        // The closure captures `model` by reference — MLX's autograd
        // discovers parameters through `Module.trainableParameters()`.
        let lossFn = { (m: TinyGPTModel, x: MLXArray, y: MLXArray) -> MLXArray in
            m.loss(x, y)
        }
        let gradFn = valueAndGrad(model: model, lossFn)
        self.gradFn = gradFn
        let optimizer = self.optimizer
        let m = model
        let clip = gradClipNorm
        let layerDecay = lrLayerDecay
        let nLayers = model.config.nLayers
        let galoreMgr = galore

        // GaLore mutates projector state out-of-graph, so it MUST live on
        // the uncompiled path. Layer-wise LR decay is graph-pure (just a
        // scalar multiply per leaf) and stays compile-safe.
        let canCompile = compileStep && galoreMgr == nil

        if canCompile {
            // Compile the full train step so MLX traces it once and reuses
            // the kernel-launch sequence thereafter. `inputs:` and `outputs:`
            // are model and optimizer so the compile knows to handle their
            // updated state across re-invocations. Clip + layer-LR scaling
            // happen INSIDE the traced graph, so they cost ~nothing per
            // step after the first.
            let compiled = compile(
                inputs: [m, optimizer],
                outputs: [m, optimizer]
            ) { (x: MLXArray, y: MLXArray) -> MLXArray in
                let (loss, grads) = gradFn(m, x, y)
                var processed = grads
                processed = clip.map { clipGradNorm(processed, maxNorm: $0) } ?? processed
                if layerDecay < 0.9999 {
                    processed = scaleLayerwiseLR(processed, decay: layerDecay, nLayers: nLayers)
                }
                optimizer.update(model: m, gradients: processed)
                return loss
            }
            self.trainStepFn = compiled
        } else {
            self.trainStepFn = { (x: MLXArray, y: MLXArray) -> MLXArray in
                let (loss, grads) = gradFn(m, x, y)
                var processed = grads
                processed = clip.map { clipGradNorm(processed, maxNorm: $0) } ?? processed
                // GaLore projection happens AFTER clipping so the norm cap
                // sees the raw gradient (the rank-R version is by
                // definition a contraction — clipping it twice is fine
                // but unnecessary).
                if let g = galoreMgr {
                    processed = g.processGradients(processed)
                }
                if layerDecay < 0.9999 {
                    processed = scaleLayerwiseLR(processed, decay: layerDecay, nLayers: nLayers)
                }
                optimizer.update(model: m, gradients: processed)
                return loss
            }
        }
    }

    /// One training step. Returns the scalar batch loss.
    public func step(inputs: MLXArray, targets: MLXArray) -> Float {
        let loss = trainStepFn(inputs, targets)
        // Force eager evaluation so the lazy graph doesn't grow across steps.
        // `model` and `optimizer` both conform to `Updatable`; eval walks
        // their parameters / state.
        eval(loss, model, optimizer)
        stepCount += 1
        return loss.item(Float.self)
    }

    /// Gradient-accumulated step. Runs every micro-batch through the loss
    /// + gradient function, sums the gradients element-wise across
    /// micro-batches, then divides by N and applies a single optimizer
    /// update. The returned scalar is the mean loss across micro-batches.
    ///
    /// Compile is unused on this path — the size of the micro-batch list
    /// changes the trace shape; uncompiled fallback ensures correctness.
    /// At reasonable accumulation counts (2-16) the per-step overhead
    /// from skipping compile is well-amortized by the much bigger
    /// effective batch.
    public func accumulatedStep(microBatches: [(MLXArray, MLXArray)]) -> Float {
        precondition(!microBatches.isEmpty, "accumulatedStep needs ≥1 micro-batch")
        var accumGrads: ModuleParameters? = nil
        var lossSum: Float = 0
        let n = microBatches.count
        for (x, y) in microBatches {
            let (loss, grads) = gradFn(model, x, y)
            eval(loss)
            lossSum += loss.item(Float.self)
            if let accum = accumGrads {
                // Sum element-wise. mapValues with two dicts visits matching
                // leaves; we add the corresponding MLXArrays. Each call
                // returns a NEW ModuleParameters; the old one becomes garbage.
                accumGrads = accum.mapValues(grads) { a, b in a + (b ?? a) }
            } else {
                accumGrads = grads
            }
        }
        // Mean: divide accumulated sum by micro-batch count, then update.
        let scale = MLXArray(1.0 / Float(n))
        var avg = accumGrads!.mapValues { (g: MLXArray) -> MLXArray in g * scale }
        if let cn = gradClipNorm {
            avg = clipGradNorm(avg, maxNorm: cn)
        }
        // GaLore projection — runs once per *optimiser update*, not once
        // per micro-batch (the projection is linear, so projecting the
        // mean is exactly the mean of the projections — same answer,
        // cheaper).
        if let g = galore {
            avg = g.processGradients(avg)
        }
        if lrLayerDecay < 0.9999 {
            avg = scaleLayerwiseLR(avg, decay: lrLayerDecay, nLayers: model.config.nLayers)
        }
        optimizer.update(model: model, gradients: avg)
        eval(model, optimizer)
        stepCount += 1
        return lossSum / Float(n)
    }
}
