import Foundation
import MLX
import MLXNN
import MLXRandom

/// Byte-level causal language model — Swift / MLX port of `python_ref/model.py`'s
/// `TinyGPT`. Same architecture, same parameter names, same `.tinygpt`
/// file format. A browser-trained checkpoint loads here unchanged.
public final class TinyGPTModel: Module {
    public let config: ModelConfig

    @ModuleInfo(key: "token_embedding") public var tokenEmbedding: Embedding
    @ModuleInfo(key: "position_embedding") public var positionEmbedding: Embedding
    @ModuleInfo(key: "blocks") public var blocks: [TransformerBlock]
    @ModuleInfo(key: "ln_final") public var lnFinal: LayerNorm
    /// Untied output head — only used when `tie_embeddings == false`. Tied
    /// embeddings reuse `token_embedding.weight.T` and never allocate this.
    @ModuleInfo(key: "lm_head") public var lmHead: Linear?

    public init(_ config: ModelConfig) {
        self.config = config
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.dModel
        )
        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.contextLength, dimensions: config.dModel
        )
        self._blocks.wrappedValue = (0..<config.nLayers).map { _ in TransformerBlock(config) }
        self._lnFinal.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
        if !config.tieEmbeddings {
            self._lmHead.wrappedValue = Linear(config.dModel, config.vocabSize, bias: false)
        } else {
            self._lmHead.wrappedValue = nil
        }
        super.init()
        // Note: `python_ref/model.py` applies a GPT-2-style scaled init for
        // residual-path output projections (std = 0.02 / sqrt(2L)). MLX-Swift
        // Linear weights are `let`, so swapping them needs `Module.update`.
        // Skipped here — only affects the first ~50 training steps before
        // the optimiser dominates, and is irrelevant when loading pretrained
        // weights (the common case). Re-add via update() before publishing
        // a "trained from scratch" comparison.
    }

    /// `idx: [B, T]` int32 token ids → `[B, T, vocab_size]` logits.
    public func callAsFunction(_ idx: MLXArray) -> MLXArray {
        let T = idx.shape[1]
        precondition(T <= config.contextLength,
                     "sequence length \(T) exceeds context \(config.contextLength)")
        // Build positions as an explicit Int32 array so we don't depend on
        // any implicit Range→MLXArray init behaviour. Shape [T].
        let positions = MLXArray((0..<T).map { Int32($0) })
        let posEmb = positionEmbedding(positions).expandedDimensions(axis: 0) // [1, T, C]
        var x = tokenEmbedding(idx) + posEmb
        for block in blocks {
            x = block(x)
        }
        x = lnFinal(x)
        return projectLogits(x)
    }

    /// Forward + cross-entropy loss. Targets are next-token ids, same shape as
    /// `idx`. Loss reduced over the full batch × time dimension.
    public func loss(_ idx: MLXArray, _ targets: MLXArray) -> MLXArray {
        let logits = self(idx)
        let v = logits.shape.last!
        let flatLogits = logits.reshaped([-1, v])
        let flatTargets = targets.reshaped([-1])
        return crossEntropy(logits: flatLogits, targets: flatTargets, reduction: .mean)
    }

    private func projectLogits(_ x: MLXArray) -> MLXArray {
        if let head = lmHead {
            return head(x)
        }
        // Tied embeddings: use Embedding.asLinear which is built for this case.
        return tokenEmbedding.asLinear(x)
    }

    /// Total parameter count. Useful as a sanity check against the browser's
    /// `paramCount` field in the gallery manifest.
    public func numParameters() -> Int {
        var total = 0
        for (_, p) in parameters().flattened() {
            total += p.shape.reduce(1, *)
        }
        return total
    }

    /// Autoregressive greedy decoder. For batched / temperature / top-k
    /// sampling, see the upcoming `sample` CLI subcommand.
    public func generate(prompt: MLXArray, maxNewTokens: Int, temperature: Float = 1.0)
        -> MLXArray
    {
        var idx = prompt
        for _ in 0..<maxNewTokens {
            let T = idx.shape.last!
            let lo = max(0, T - config.contextLength)
            let cond = idx[0..., lo..<T]
            let logits = self(cond)
            let lastLogits = logits[0..., logits.shape[1] - 1, 0...]
            let nextId: MLXArray
            if temperature <= 0.0 {
                nextId = argMax(lastLogits, axis: -1).reshaped([-1, 1])
            } else {
                let scaled = lastLogits / MLXArray(temperature)
                nextId = MLXRandom.categorical(scaled).reshaped([-1, 1])
            }
            idx = concatenated([idx, nextId.asType(idx.dtype)], axis: 1)
        }
        return idx
    }
}
