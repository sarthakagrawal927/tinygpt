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
    /// Number of K/V heads. Equal to nHeads in standard multi-head
    /// attention; less than nHeads in Grouped Query Attention (Llama-3,
    /// Mistral, modern HF models). When nKvHeads < nHeads, each KV head
    /// is broadcast across `nHeads / nKvHeads` query heads.
    public let nKvHeads: Int
    public let headDim: Int
    public let scale: Float
    /// RoPE base frequency. Standard transformer uses learned absolute
    /// position embeddings (RoPE off). HF models use RoPE — they rotate
    /// Q and K by an angle proportional to position before the attention
    /// matmul. When > 0, RoPE is applied with this base (typically 10000
    /// or 500000).
    public let ropeBase: Float
    /// `true` means RoPE is applied. When false (the default for our
    /// from-scratch models), absolute learned position embeddings are
    /// used (added to the input embedding upstream).
    public let useRoPE: Bool

    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear

    public init(_ cfg: ModelConfig) {
        self.nHeads = cfg.nHeads
        self.nKvHeads = cfg.nKvHeads
        self.headDim = cfg.headDim
        self.scale = 1.0 / sqrt(Float(cfg.headDim))
        self.ropeBase = cfg.ropeBase
        self.useRoPE = cfg.useRoPE
        // Q goes from dModel to (nHeads * headDim) = dModel — unchanged
        // K, V go to (nKvHeads * headDim) which is smaller for GQA models
        let kvDim = cfg.nKvHeads * cfg.headDim
        self._qProj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: cfg.attnBias)
        self._kProj.wrappedValue = Linear(cfg.dModel, kvDim, bias: cfg.attnBias)
        self._vProj.wrappedValue = Linear(cfg.dModel, kvDim, bias: cfg.attnBias)
        self._oProj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: cfg.attnBias)
        super.init()
    }

    /// `x: [B, T, C]` → `[B, T, C]`. Causal mask is created on-the-fly by
    /// `scaledDotProductAttention` from a string ("causal") so it can be
    /// fused into the kernel.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        // Q: full nHeads heads. K, V: nKvHeads heads (= nHeads in standard
        // attention; less for Grouped Query Attention).
        var q = qProj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped([B, T, nKvHeads, headDim]).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([B, T, nKvHeads, headDim]).transposed(0, 2, 1, 3)

        // RoPE: rotate Q and K (not V) by position-dependent angles.
        // Standard transformers add learned absolute position embeddings
        // upstream; HF-style models skip that and apply RoPE here instead.
        if useRoPE {
            q = MLXFast.RoPE(q, dimensions: headDim, traditional: false,
                              base: ropeBase, scale: 1.0, offset: 0)
            k = MLXFast.RoPE(k, dimensions: headDim, traditional: false,
                              base: ropeBase, scale: 1.0, offset: 0)
        }

        // Fused fast attention with built-in causal masking. MLX-Fast's
        // SDPA natively supports unequal Q vs KV head counts — when
        // nKvHeads < nHeads, the kernel broadcasts each KV head across
        // nHeads/nKvHeads Q heads internally (zero-copy).
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

/// SwiGLU MLP — gated variant used by Llama 2+, Mistral, Phi-3, Qwen,
/// Gemma, LFM. Three linears instead of two:
///
///     y = down( silu(up(x)) * gate(x) )
///
/// `gate(x)` produces an element-wise "soft on/off" signal that the
/// network can learn to use to suppress unwanted features, instead of
/// only being able to add more on top (which is all a plain MLP can do).
///
/// In HuggingFace param names:
///   up_proj   = `fcUp`
///   gate_proj = `fcGate`
///   down_proj = `fcDown`
public final class SwiGLU: Module {
    @ModuleInfo(key: "up_proj")   public var fcUp: Linear
    @ModuleInfo(key: "gate_proj") public var fcGate: Linear
    @ModuleInfo(key: "down_proj") public var fcDown: Linear

    public init(dModel: Int, dMlp: Int, bias: Bool = false) {
        // SwiGLU MLPs in modern HF models are bias-free (Llama/Mistral
        // family). Plain MLP keeps biases; pass `bias: true` if needed.
        self._fcUp.wrappedValue   = Linear(dModel, dMlp, bias: bias)
        self._fcGate.wrappedValue = Linear(dModel, dMlp, bias: bias)
        self._fcDown.wrappedValue = Linear(dMlp, dModel, bias: bias)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Llama-style SwiGLU: silu is applied to GATE, then element-wise
        // multiplied by UP. NOT silu(up) * gate — that's a different (and
        // worse) gating function which compiles fine but produces garbage.
        // Reference: HF transformers' modeling_llama.py LlamaMLP.forward.
        return fcDown(silu(fcGate(x)) * fcUp(x))
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
