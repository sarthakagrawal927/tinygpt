import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// AdamW trainer for TinyGPTModelHF. Parallel to Trainer (which is
/// specialised on TinyGPTModel). Same step semantics, same compile
/// option — just bound to the HF model class because Swift generics
/// don't compose nicely with the MLX value-and-grad signature today.
public final class TrainerHF {
    public let model: TinyGPTModelHF
    /// Generic optimiser handle — see `Trainer.optimizer`.
    public let optimizer: any Optimizer & LearningRateMutable
    public let optimizerKind: OptimizerKind
    public private(set) var stepCount: Int = 0
    public let gradClipNorm: Float?
    /// See `Trainer.galore` for the rationale. Forces uncompiled path.
    public let galore: GaLoreManager?
    /// See `Trainer.lrLayerDecay`.
    public let lrLayerDecay: Float
    private let trainStepFn: (MLXArray, MLXArray) -> MLXArray

    public init(model: TinyGPTModelHF,
                learningRate: Float = 3e-4,
                weightDecay: Float = 0.0,
                betas: (Float, Float) = (0.9, 0.95),
                eps: Float = 1e-8,
                compileStep: Bool = true,
                gradClipNorm: Float? = nil,
                optimizer optimizerKind: OptimizerKind = .adamw,
                galore: GaLoreManager? = nil,
                lrLayerDecay: Float = 1.0) {
        self.model = model
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
        let m = model
        let opt = self.optimizer
        let clip = gradClipNorm
        let lossFn = { (mod: TinyGPTModelHF, x: MLXArray, y: MLXArray) -> MLXArray in
            mod.loss(x, y)
        }
        let gradFn = valueAndGrad(model: model, lossFn)
        let layerDecay = lrLayerDecay
        let nLayers = model.config.nLayers
        let galoreMgr = galore
        // Same constraint as Trainer: GaLore mutates projector state out-of-graph.
        let canCompile = compileStep && galoreMgr == nil
        if canCompile {
            self.trainStepFn = compile(
                inputs: [m, opt], outputs: [m, opt]
            ) { x, y in
                let (loss, grads) = gradFn(m, x, y)
                var processed = grads
                processed = clip.map { clipGradNorm(processed, maxNorm: $0) } ?? processed
                if layerDecay < 0.9999 {
                    processed = scaleLayerwiseLR(processed, decay: layerDecay, nLayers: nLayers)
                }
                opt.update(model: m, gradients: processed)
                return loss
            }
        } else {
            self.trainStepFn = { x, y in
                let (loss, grads) = gradFn(m, x, y)
                var processed = grads
                processed = clip.map { clipGradNorm(processed, maxNorm: $0) } ?? processed
                if let g = galoreMgr {
                    processed = g.processGradients(processed)
                }
                if layerDecay < 0.9999 {
                    processed = scaleLayerwiseLR(processed, decay: layerDecay, nLayers: nLayers)
                }
                opt.update(model: m, gradients: processed)
                return loss
            }
        }
    }

    public func step(inputs: MLXArray, targets: MLXArray) -> Float {
        let loss = trainStepFn(inputs, targets)
        eval(loss, model, optimizer)
        stepCount += 1
        return loss.item(Float.self)
    }
}
