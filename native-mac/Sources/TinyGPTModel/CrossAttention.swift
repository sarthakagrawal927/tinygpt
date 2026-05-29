import Foundation
import MLX
import MLXNN
import MLXFast

/// CrossAttention — YOCO second-half attention block. Q is computed from
/// the local hidden state; K, V are supplied externally (from the YOCO
/// anchor — the LAST first-half layer's K, V). This module deliberately
/// does NOT allocate `k_proj` / `v_proj` Linears — that's the central
/// architectural commitment of YOCO ("You Only Cache Once", Lin et al.,
/// 2024): KV memory at long context drops by ~½ because second-half
/// layers neither compute nor cache their own K, V.
///
/// Why a dedicated module (instead of just reusing CausalSelfAttention's
/// `forwardWithExternalKV` extension)?
///
///   1. **Param count**: the existing CausalSelfAttention always allocates
///      `k_proj` + `v_proj`. For YOCO second-half layers those are dead
///      weight at forward time but still consume parameter budget /
///      training memory. A separate CrossAttention class trims that.
///
///   2. **RoPE offset for cached decode**: cross-attention with a cached
///      decode needs Q rotated at the absolute position (basePos + t), not
///      at 0. The previous inline `forwardWithExternalKV` always used
///      offset=0 which silently produces wrong rotations after the prompt
///      prefill. This module takes `posOffset` explicitly.
///
///   3. **Manifest cleanliness**: with this module, YOCO-on .tinygpt files
///      have a smaller manifest (no q_proj/k_proj/v_proj/o_proj duplication
///      for the second half), but at the cost of breaking interop with the
///      plain-attention `attn` field. To keep that interop working we keep
///      `CausalSelfAttention attn` ALSO allocated on every block; the
///      CrossAttention sibling is a parallel module that the block routes
///      to when YOCO is on and the layer is second-half. The dead `attn`
///      weights are accepted as the cost of LoRA / manifest stability —
///      see the rationale on `TransformerBlock.diffAttn`.
public final class CrossAttention: Module {
    public let nHeads: Int
    public let nKvHeads: Int
    public let headDim: Int
    public let scale: Float
    public let ropeBase: Float
    public let useRoPE: Bool

    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear

    public init(_ cfg: ModelConfig) {
        self.nHeads = cfg.nHeads
        self.nKvHeads = cfg.nKvHeads
        self.headDim = cfg.headDim
        self.scale = 1.0 / sqrt(Float(cfg.headDim))
        self.ropeBase = cfg.ropeBase
        self.useRoPE = cfg.useRoPE
        self._qProj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: cfg.attnBias)
        self._oProj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: cfg.attnBias)
        super.init()
    }

    /// Training / prefill forward. Q is from local x, K/V are supplied
    /// from the anchor. `posOffset` is the absolute-position shift used
    /// by RoPE — 0 during training (Q positions 0..T-1 match the anchor's
    /// own positions 0..T-1) and `basePos` (cache length) during decode.
    public func callAsFunction(_ x: MLXArray, externalK k: MLXArray, externalV v: MLXArray,
                                posOffset: Int = 0) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        // Q: full nHeads. K, V come in already shaped [B, nKvHeads, T_anchor, D]
        // from the anchor's forward (which used `nKvHeads` for GQA).
        var q = qProj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        if useRoPE {
            // Rotate Q at the absolute decode position so its relative
            // angle vs the anchor's K matches "I'm at position posOffset+t
            // attending to anchor positions 0..(T_anchor-1)".
            q = MLXFast.RoPE(q, dimensions: headDim, traditional: false,
                              base: ropeBase, scale: 1.0, offset: posOffset)
        }
        // SDPA. For prefill (T == T_anchor and posOffset == 0) we want
        // causal masking among the new tokens. For per-token decode
        // (T == 1) the single query attends to every anchor position
        // without masking. Match the convention used in KVCache.swift.
        let useCausal = (T == k.shape[2] && posOffset == 0)
        let out = useCausal
            ? MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .causal)
            : MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .none)
        let merged = out.transposed(0, 2, 1, 3).reshaped([B, T, nHeads * headDim])
        return oProj(merged)
    }
}
