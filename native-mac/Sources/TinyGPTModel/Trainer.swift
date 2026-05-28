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

/// AdamW + value-and-grad train loop. One `step()` call does a full
/// forward + backward + optimiser update and returns the scalar loss.
public final class Trainer {
    public let model: TinyGPTModel
    public let optimizer: AdamW
    public private(set) var stepCount: Int = 0

    /// Compiled (graph-traced) train step. MLX-Swift's `compile` traces the
    /// step the first time it's called and reuses the kernel-launch sequence
    /// thereafter — the single biggest win over an interpreted train loop.
    private let trainStepFn: (MLXArray, MLXArray) -> MLXArray
    private let useCompile: Bool

    public init(
        model: TinyGPTModel,
        learningRate: Float = 3e-4,
        weightDecay: Float = 0.1,
        betas: (Float, Float) = (0.9, 0.95),
        eps: Float = 1e-8,
        compileStep: Bool = true
    ) {
        self.model = model
        self.useCompile = compileStep
        self.optimizer = AdamW(
            learningRate: learningRate,
            betas: betas,
            eps: eps,
            weightDecay: weightDecay
        )
        // value_and_grad of the loss function, captured to apply via optimizer.
        // The closure captures `model` by reference — MLX's autograd
        // discovers parameters through `Module.trainableParameters()`.
        let lossFn = { (m: TinyGPTModel, x: MLXArray, y: MLXArray) -> MLXArray in
            m.loss(x, y)
        }
        let gradFn = valueAndGrad(model: model, lossFn)
        let optimizer = self.optimizer
        let m = model

        if compileStep {
            // Compile the full train step so MLX traces it once and reuses
            // the kernel-launch sequence thereafter. `inputs:` and `outputs:`
            // are model and optimizer so the compile knows to handle their
            // updated state across re-invocations.
            let compiled = compile(
                inputs: [m, optimizer],
                outputs: [m, optimizer]
            ) { (x: MLXArray, y: MLXArray) -> MLXArray in
                let (loss, grads) = gradFn(m, x, y)
                optimizer.update(model: m, gradients: grads)
                return loss
            }
            self.trainStepFn = compiled
        } else {
            self.trainStepFn = { (x: MLXArray, y: MLXArray) -> MLXArray in
                let (loss, grads) = gradFn(m, x, y)
                optimizer.update(model: m, gradients: grads)
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
}
