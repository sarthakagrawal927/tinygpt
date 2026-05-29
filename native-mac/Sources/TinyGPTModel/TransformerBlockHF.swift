import Foundation
import MLX
import MLXNN
import MLXFast

/// HF-style transformer block — uses RMSNorm + SwiGLU MLP. Parallel to
/// TransformerBlock (which keeps LayerNorm + plain GELU MLP for our
/// from-scratch teaching path); the two share `CausalSelfAttention`
/// which is already config-aware (RoPE + GQA toggled via ModelConfig).
///
/// Why a separate class instead of polymorphic switches inside one
/// Block class: MLX-Swift's `@ModuleInfo<T>` wraps a CONCRETE Module
/// subtype. Holding "either LayerNorm or RMSNorm" in one slot
/// requires a common parent type the framework can serialise, which
/// MLX-Swift's existing Module surface doesn't cleanly expose. Two
/// concrete blocks keep the framework-side serialisation clean at
/// the cost of ~30 lines of duplication.
///
/// Architecture matches Llama 2+ / Mistral / Phi-3 / Qwen / Gemma / LFM:
///
///     x = x + attn(rms_norm_1(x))    -- attention is RoPE + GQA inside
///     x = x + swiglu(rms_norm_2(x))  -- gated feedforward
public final class TransformerBlockHF: Module {
    @ModuleInfo(key: "input_layernorm")          public var ln1: RMSNorm
    @ModuleInfo(key: "self_attn")                public var attn: CausalSelfAttention
    @ModuleInfo(key: "post_attention_layernorm") public var ln2: RMSNorm
    @ModuleInfo(key: "mlp")                      public var mlp: SwiGLU

    /// HF-block MoE support is not wired yet — kept here so the
    /// `sumMoEAuxLossesHF` walker compiles. Future: parallel to the
    /// from-scratch path, swap SwiGLU for an MoE-of-SwiGLU when
    /// `cfg.isMoE` and the model is HF.
    public var mlpUnit: Module { mlp }

    /// Gradient checkpointing toggle. See the matching field on
    /// `TransformerBlock` and `GradCheckpoint.swift` for the mechanism.
    public var useGradCheckpoint: Bool = false

    public init(_ cfg: ModelConfig) {
        // The HF naming convention is `input_layernorm` for ln1 and
        // `post_attention_layernorm` for ln2. We use those keys so
        // safetensors weight names match without re-mapping.
        self._ln1.wrappedValue = RMSNorm(dimensions: cfg.dModel, eps: 1e-5)
        self._attn.wrappedValue = CausalSelfAttention(cfg)
        self._ln2.wrappedValue = RMSNorm(dimensions: cfg.dModel, eps: 1e-5)
        self._mlp.wrappedValue = SwiGLU(dModel: cfg.dModel, dMlp: cfg.dMlp, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if useGradCheckpoint {
            return GradCheckpoint.wrap(block: self, x: x) { b, xt in
                b.rawForward(xt)
            }
        }
        return rawForward(x)
    }

    /// Raw block forward — also used as the recompute payload inside
    /// `GradCheckpoint.wrap`. Does NOT consult `useGradCheckpoint` so
    /// the wrapper's VJP doesn't recurse.
    public func rawForward(_ x: MLXArray) -> MLXArray {
        var x = x
        x = x + attn(ln1(x))
        x = x + mlp(ln2(x))
        return x
    }
}
