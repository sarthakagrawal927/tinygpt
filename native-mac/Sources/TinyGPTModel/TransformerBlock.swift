import Foundation
import MLX
import MLXNN
import MLXFast

/// Multi-head causal self-attention, mirroring `python_ref/model.py`'s
/// `CausalSelfAttention`. The four projections (`q_proj`, `k_proj`,
/// `v_proj`, `o_proj`) are kept separate so LoRA targeting works name-wise.
///
/// Uses `MLXFast.scaledDotProductAttention` — Apple's fused Flash-Attention
/// equivalent, the single highest-impact perf primitive on the Mac side.
/// Memory bandwidth for the attention matrix is the matmul-bound term; the
/// fused kernel cuts it dramatically vs. naive `qk^T → softmax → v`.
public final class CausalSelfAttention: Module {
    public let nHeads: Int
    public let headDim: Int
    public let scale: Float

    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear

    public init(_ cfg: ModelConfig) {
        self.nHeads = cfg.nHeads
        self.headDim = cfg.headDim
        self.scale = 1.0 / sqrt(Float(cfg.headDim))
        self._qProj.wrappedValue = Linear(cfg.dModel, cfg.dModel)
        self._kProj.wrappedValue = Linear(cfg.dModel, cfg.dModel)
        self._vProj.wrappedValue = Linear(cfg.dModel, cfg.dModel)
        self._oProj.wrappedValue = Linear(cfg.dModel, cfg.dModel)
        super.init()
    }

    /// `x: [B, T, C]` → `[B, T, C]`. Causal mask is created on-the-fly by
    /// `scaledDotProductAttention` from a string ("causal") so it can be
    /// fused into the kernel.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        // q, k, v: [B, T, C] → [B, n_heads, T, head_dim]
        let q = qProj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)

        // Fused fast attention with built-in causal masking — the highest-
        // impact perf primitive on the Mac side. In practice (verified
        // empirically against a known-good model) the function returns
        // output in [B, H, T, head_dim] layout, despite docs suggesting
        // otherwise; transpose to [B, T, H, head_dim] before reshape.
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .causal
        )
        let merged = out.transposed(0, 2, 1, 3).reshaped([B, T, nHeads * headDim])
        return oProj(merged)
    }
}

/// Position-wise feed-forward — Linear → GELU → Linear. Matches the
/// `python_ref/model.py` MLP exactly (no SwiGLU; the python reference uses
/// plain GELU and the browser was trained with that, so changing here would
/// break weight loading).
public final class MLP: Module {
    @ModuleInfo(key: "fc_in") public var fcIn: Linear
    @ModuleInfo(key: "fc_out") public var fcOut: Linear

    public init(_ cfg: ModelConfig) {
        self._fcIn.wrappedValue = Linear(cfg.dModel, cfg.dMlp)
        self._fcOut.wrappedValue = Linear(cfg.dMlp, cfg.dModel)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return fcOut(gelu(fcIn(x)))
    }
}

/// Pre-LayerNorm transformer block: `x = x + attn(ln1(x)); x = x + mlp(ln2(x))`.
public final class TransformerBlock: Module {
    @ModuleInfo(key: "ln1") public var ln1: LayerNorm
    @ModuleInfo(key: "attn") public var attn: CausalSelfAttention
    @ModuleInfo(key: "ln2") public var ln2: LayerNorm
    @ModuleInfo(key: "mlp") public var mlp: MLP

    public init(_ cfg: ModelConfig) {
        self._ln1.wrappedValue = LayerNorm(dimensions: cfg.dModel, eps: 1e-5)
        self._attn.wrappedValue = CausalSelfAttention(cfg)
        self._ln2.wrappedValue = LayerNorm(dimensions: cfg.dModel, eps: 1e-5)
        self._mlp.wrappedValue = MLP(cfg)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        x = x + attn(ln1(x))
        x = x + mlp(ln2(x))
        return x
    }
}
