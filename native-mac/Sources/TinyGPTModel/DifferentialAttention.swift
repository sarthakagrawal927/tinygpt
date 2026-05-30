import Foundation
import MLX
import MLXNN
import MLXFast

// AUDIT FLAG: Differential Attention (Ye et al., 2024).
//
// Tested: smoke train at 22M params, vanilla data.
// Saw: no measured benefit. Doubled Q/K projections (cost) for no
//   measured quality lift at our scale.
// When this would help: paper demonstrates gains at >100M params on
//   long-context reasoning. Probably real at frontier scale; invisible
//   at small.
// Default recipe uses vanilla attention. Available via --diff-attn.

/// Differential Attention (Ye et al., 2024 — "DIFF Transformer:
/// Differential Transformer", arXiv:2410.05258).
///
/// Each attention head computes TWO independent softmax attention
/// maps and subtracts them, weighted by a learnable scalar λ:
///
///     A = softmax(Q1 K1ᵀ / √d) − λ · softmax(Q2 K2ᵀ / √d)
///     out = A · V
///
/// The subtraction cancels correlated noise across the two heads —
/// the model learns to keep the "useful" attention pattern and zero
/// out the spurious correlations both branches share. Net effect:
/// less attention noise, fewer hallucinations, better long-context
/// reasoning. Compute per head is roughly 1.5-2× a standard MHA
/// head (two QK, two softmax, one V), but the reduced noise improves
/// per-token efficiency enough to be net-positive on most benchmarks.
///
/// The paper's λ has a per-head reparam:
///     λ = exp(λ_q1 · λ_k1) − exp(λ_q2 · λ_k2) + λ_init
/// where λ_init is a fixed scalar that decreases with layer depth.
/// For simplicity this implementation uses a learnable scalar λ
/// directly (no reparam, no depth-dependent init) — the paper's
/// extra precision is bounded follow-up.
public final class DifferentialAttention: Module {
    public let nHeads: Int
    public let headDim: Int
    public let scale: Float
    /// RoPE-compatible. Same conventions as CausalSelfAttention.
    public let ropeBase: Float
    public let useRoPE: Bool

    // Two Q projections, two K projections, one V, one O. The "1" and
    // "2" suffixes match the paper's notation for the two branches.
    @ModuleInfo(key: "q1_proj") public var q1Proj: Linear
    @ModuleInfo(key: "k1_proj") public var k1Proj: Linear
    @ModuleInfo(key: "q2_proj") public var q2Proj: Linear
    @ModuleInfo(key: "k2_proj") public var k2Proj: Linear
    @ModuleInfo(key: "v_proj")  public var vProj: Linear
    @ModuleInfo(key: "o_proj")  public var oProj: Linear

    /// Learnable subtraction weight. Initialised to 0.5 (the paper's
    /// approximate mid-depth λ_init). Scalar — applies uniformly across
    /// heads and positions. The paper's per-head + reparam version is
    /// a bounded follow-up.
    @ParameterInfo(key: "lambda") public var lambda: MLXArray

    public init(_ cfg: ModelConfig) {
        self.nHeads = cfg.nHeads
        self.headDim = cfg.headDim
        self.scale = 1.0 / sqrt(Float(cfg.headDim))
        self.ropeBase = cfg.ropeBase
        self.useRoPE = cfg.useRoPE
        self._q1Proj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: cfg.attnBias)
        self._k1Proj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: cfg.attnBias)
        self._q2Proj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: cfg.attnBias)
        self._k2Proj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: cfg.attnBias)
        self._vProj.wrappedValue  = Linear(cfg.dModel, cfg.dModel, bias: cfg.attnBias)
        self._oProj.wrappedValue  = Linear(cfg.dModel, cfg.dModel, bias: cfg.attnBias)
        // λ as a single learnable scalar.
        self._lambda.wrappedValue = MLXArray(Float(0.5))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        // Two query/key projections; shared value.
        var q1 = q1Proj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        var k1 = k1Proj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        var q2 = q2Proj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        var k2 = k2Proj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        let  v  = vProj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)

        if useRoPE {
            q1 = MLXFast.RoPE(q1, dimensions: headDim, traditional: false,
                               base: ropeBase, scale: 1.0, offset: 0)
            k1 = MLXFast.RoPE(k1, dimensions: headDim, traditional: false,
                               base: ropeBase, scale: 1.0, offset: 0)
            q2 = MLXFast.RoPE(q2, dimensions: headDim, traditional: false,
                               base: ropeBase, scale: 1.0, offset: 0)
            k2 = MLXFast.RoPE(k2, dimensions: headDim, traditional: false,
                               base: ropeBase, scale: 1.0, offset: 0)
        }

        // Two attention outputs against the SAME V — SDPA fuses the
        // softmax + V matmul, so we get out1 = softmax(Q1K1ᵀ/√d)·V and
        // out2 = softmax(Q2K2ᵀ/√d)·V as fast kernels.
        let out1 = MLXFast.scaledDotProductAttention(
            queries: q1, keys: k1, values: v, scale: scale, mask: .causal
        )
        let out2 = MLXFast.scaledDotProductAttention(
            queries: q2, keys: k2, values: v, scale: scale, mask: .causal
        )
        // Differential combination: out1 − λ · out2. Both are
        // [B, H, T, D]; λ is a scalar that broadcasts.
        let combined = out1 - lambda * out2
        let merged = combined.transposed(0, 2, 1, 3).reshaped([B, T, nHeads * headDim])
        return oProj(merged)
    }
}
