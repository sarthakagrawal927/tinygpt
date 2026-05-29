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

    /// Extra output heads for Multi-Token Prediction (Gloeckle et al.,
    /// 2024; DeepSeek-V3 popularised). One Linear per horizon beyond 1.
    /// Training-time only: NOT in the .tinygpt manifest, so saved files
    /// stay drop-in compatible. nil when `config.mtpHorizons == 1`.
    @ModuleInfo(key: "mtp_heads") public var mtpHeads: [Linear]?

    /// Tuned-lens probes (Belrose et al., 2023). One Linear(d_model →
    /// vocab) per layer, trained on a frozen base to produce a
    /// LAYER-CALIBRATED projection instead of the noisy "reuse the
    /// final LN" lens. Loaded from a sidecar `.lenses` file via
    /// `attachTunedLens`. Inference: forwardTunedLens returns per-
    /// layer logits via these probes; training: tinygpt tuned-lens
    /// freezes the base and SGDs the probes on a corpus.
    @ModuleInfo(key: "tuned_lens") public var tunedLens: [Linear]?

    /// NEFTune (Jain et al., 2024) — scale of uniform noise added to the
    /// token-embedding output during forward. 0 (default) = off. SFT/DPO
    /// flips this to ~5 for the policy; the DPO reference stays at 0.
    /// Not a Parameter — never serialised, never differentiated.
    public var nefTuneAlpha: Float = 0

    public init(_ config: ModelConfig) {
        self.config = config
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.dModel
        )
        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.contextLength, dimensions: config.dModel
        )
        self._blocks.wrappedValue = (0..<config.nLayers).map { _ in
            let b = TransformerBlock(config)
            b.useGradCheckpoint = config.useGradCheckpoint
            return b
        }
        self._lnFinal.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
        if !config.tieEmbeddings {
            self._lmHead.wrappedValue = Linear(config.dModel, config.vocabSize, bias: false)
        } else {
            self._lmHead.wrappedValue = nil
        }
        // MTP extra heads: one Linear(d_model → vocab) per horizon beyond 1.
        // Bias-free to mirror the primary lm_head when untied.
        if config.mtpHorizons > 1 {
            let extras = (0..<(config.mtpHorizons - 1)).map { _ in
                Linear(config.dModel, config.vocabSize, bias: false)
            }
            self._mtpHeads.wrappedValue = extras
        } else {
            self._mtpHeads.wrappedValue = nil
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
    ///
    /// At inference time only the primary head is consulted; MTP extras
    /// are training-time auxiliaries (see `forwardMTP`).
    public func callAsFunction(_ idx: MLXArray) -> MLXArray {
        return projectLogits(forwardToHidden(idx))
    }

    /// Forward all the way through the blocks + final norm, returning
    /// the `[B, T, C]` hidden state. Factored out so MTP's multi-head
    /// path doesn't have to duplicate the block loop.
    public func forwardToHidden(_ idx: MLXArray) -> MLXArray {
        let T = idx.shape[1]
        precondition(T <= config.contextLength,
                     "sequence length \(T) exceeds context \(config.contextLength)")
        // Build positions as an explicit Int32 array so we don't depend on
        // any implicit Range→MLXArray init behaviour. Shape [T].
        let positions = MLXArray((0..<T).map { Int32($0) })
        let posEmb = positionEmbedding(positions).expandedDimensions(axis: 0) // [1, T, C]
        var tokEmb = tokenEmbedding(idx)
        // NEFTune noise — applied to TOKEN embedding only (not positional),
        // matching the paper's intent of regularising the input-token signal.
        // scale s = alpha / sqrt(seq_len · embed_dim), so per-element noise
        // stays small relative to the embedding norm regardless of T or d.
        if nefTuneAlpha > 0 {
            let s = nefTuneAlpha / sqrt(Float(T * config.dModel))
            let noise = MLXRandom.uniform(
                low: -s, high: s, tokEmb.shape
            ).asType(tokEmb.dtype)
            tokEmb = tokEmb + noise
        }
        var x = tokEmb + posEmb
        if config.useYOCO {
            // YOCO orchestration: first half runs standard self-attn;
            // the LAST first-half layer captures (K, V) as an "anchor";
            // second-half layers do cross-attention onto that anchor
            // instead of computing their own K, V. Halves the KV cache
            // memory at long-context decode.
            let anchorIdx = max(0, (blocks.count / 2) - 1)
            var yocoK: MLXArray? = nil
            var yocoV: MLXArray? = nil
            for (i, block) in blocks.enumerated() {
                if i < anchorIdx {
                    x = block(x)
                } else if i == anchorIdx {
                    let result = block.callCapturingKV(x)
                    x = result.out; yocoK = result.k; yocoV = result.v
                } else {
                    // anchor MUST be set by now — preconditionFailure if
                    // not (means we entered an "after anchor" iteration
                    // without ever capturing).
                    guard let k = yocoK, let v = yocoV else {
                        preconditionFailure("YOCO anchor missing at layer \(i)")
                    }
                    x = block.callWithExternalKV(x, k: k, v: v)
                }
            }
        } else {
            for block in blocks {
                x = block(x)
            }
        }
        return lnFinal(x)
    }

    /// Multi-Token-Prediction forward. Returns one logits tensor per
    /// horizon, all sharing the same final hidden state — the only
    /// difference between horizons is the output head's projection.
    /// Index 0 is the primary head (predicts t+1), 1..H-1 are the
    /// MTP extras (predict t+2, t+3, ...).
    public func forwardMTP(_ idx: MLXArray) -> [MLXArray] {
        let h = forwardToHidden(idx)
        var out: [MLXArray] = [projectLogits(h)]
        if let heads = mtpHeads {
            for head in heads { out.append(head(h)) }
        }
        return out
    }

    /// Per-LAYER hidden states captured during forward. Used by tuned-
    /// lens training to feed each layer's residual stream through its
    /// own learned projection probe. Returns `[blocks.count]` of
    /// `[B, T, C]` tensors — the post-block residual at each depth.
    /// Includes the EMBEDDING output at index 0? No — we start from
    /// the FIRST block's output (depth 1). Index k = output of block k.
    public func forwardLayerwise(_ idx: MLXArray) -> [MLXArray] {
        let T = idx.shape[1]
        precondition(T <= config.contextLength,
                     "sequence length \(T) exceeds context \(config.contextLength)")
        let positions = MLXArray((0..<T).map { Int32($0) })
        let posEmb = positionEmbedding(positions).expandedDimensions(axis: 0)
        var x = tokenEmbedding(idx) + posEmb
        var states: [MLXArray] = []
        states.reserveCapacity(blocks.count)
        for block in blocks {
            x = block(x)
            states.append(x)
        }
        return states
    }

    /// Tuned-lens forward: returns one logits `[B, T, vocab]` per
    /// layer, computed via the trained per-layer probes. Caller must
    /// have called `attachTunedLens(from:)` (or pre-populated
    /// `tunedLens`) — fails with a clean error otherwise.
    public func forwardTunedLens(_ idx: MLXArray) -> [MLXArray] {
        guard let lenses = tunedLens else {
            preconditionFailure("forwardTunedLens called without trained lenses — load via attachTunedLens(from:)")
        }
        precondition(lenses.count == blocks.count,
                     "tuned-lens probe count (\(lenses.count)) ≠ layer count (\(blocks.count))")
        let states = forwardLayerwise(idx)
        return zip(states, lenses).map { (state, lens) in lens(state) }
    }

    /// Initialise the tuned-lens probes (one Linear per block). Used by
    /// `tinygpt tuned-lens` BEFORE training. After training, persist via
    /// `saveTunedLens` and re-attach next session via `attachTunedLens`.
    public func initTunedLens() {
        let probes = (0..<blocks.count).map { _ in
            Linear(config.dModel, config.vocabSize, bias: true)
        }
        _tunedLens.wrappedValue = probes
    }

    /// Forward + cross-entropy loss. Targets are next-token ids, same shape as
    /// `idx`. Loss reduced over the full batch × time dimension.
    ///
    /// When the model is MoE (`config.isMoE`), the auxiliary load-balance
    /// loss accumulated by every MoEMLP during this forward is added in,
    /// scaled by `config.loadBalanceWeight`. The MoE-aware loss MUST be
    /// computed in the same call as the forward — the aux side-channel is
    /// populated by the forward and read here while it's still fresh.
    ///
    /// When `config.mtpHorizons > 1`, the loss is the MEAN of per-horizon
    /// cross-entropies: horizon 1 against `targets` directly, horizon h
    /// against `targets` shifted left by `h-1` positions (the last h-1
    /// positions are unscored — we run out of look-ahead). MoE aux still
    /// fires when both are active.
    public func loss(_ idx: MLXArray, _ targets: MLXArray) -> MLXArray {
        let ce: MLXArray
        if config.mtpHorizons > 1 {
            ce = mtpCrossEntropy(idx: idx, targets: targets)
        } else {
            let logits = self(idx)
            let v = logits.shape.last!
            ce = crossEntropy(
                logits: logits.reshaped([-1, v]),
                targets: targets.reshaped([-1]),
                reduction: .mean
            )
        }
        if config.isMoE {
            let aux = sumMoEAuxLosses(blocks)
            return ce + MLXArray(config.loadBalanceWeight) * aux
        }
        return ce
    }

    /// Mean per-horizon CE. Horizon `h` (1-indexed) predicts the token
    /// `h` positions ahead — its target is `targets` shifted left by
    /// `h-1` (because `targets[t]` already represents the `t+1` ground
    /// truth, so horizon 2's target at position t is `targets[t+1]`).
    /// The last `h-1` positions can't be scored at horizon h; we slice
    /// them off symmetrically from logits and targets.
    private func mtpCrossEntropy(idx: MLXArray, targets: MLXArray) -> MLXArray {
        let allLogits = forwardMTP(idx)
        let H = allLogits.count
        let T = targets.shape[1]
        var total = MLXArray(Float(0))
        var horizonsScored = 0
        for h in 0..<H {
            // Shift = h ; valid window length = T - h.
            let valid = T - h
            if valid <= 0 { continue }
            let logitsH = allLogits[h][0..., 0..<valid, 0...]
            let targetsH = targets[0..., h..<T]
            let v = logitsH.shape.last!
            let ce = crossEntropy(
                logits: logitsH.reshaped([-1, v]),
                targets: targetsH.reshaped([-1]),
                reduction: .mean
            )
            total = total + ce
            horizonsScored += 1
        }
        return total / MLXArray(Float(max(1, horizonsScored)))
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
