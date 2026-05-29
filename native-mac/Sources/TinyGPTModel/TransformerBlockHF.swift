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
    /// YOCO CrossAttention sibling — same role as
    /// `TransformerBlock.crossAttn`. Installed by the model on
    /// second-half layers when `cfg.useYOCO` is set. The `self_attn`
    /// field stays allocated for HF-weight-mapping stability — its
    /// weights are dead on second-half forwards but still load cleanly
    /// from a non-YOCO HF checkpoint when YOCO is toggled on at init.
    @ModuleInfo(key: "cross_attn") public var crossAttn: CrossAttention?

    /// HF-block MoE support is not wired yet — kept here so the
    /// `sumMoEAuxLossesHF` walker compiles. Future: parallel to the
    /// from-scratch path, swap SwiGLU for an MoE-of-SwiGLU when
    /// `cfg.isMoE` and the model is HF.
    public var mlpUnit: Module { mlp }

    /// Gradient checkpointing toggle. See the matching field on
    /// `TransformerBlock` and `GradCheckpoint.swift` for the mechanism.
    public var useGradCheckpoint: Bool = false

    public init(_ cfg: ModelConfig, yocoSecondHalf: Bool = false) {
        // The HF naming convention is `input_layernorm` for ln1 and
        // `post_attention_layernorm` for ln2. We use those keys so
        // safetensors weight names match without re-mapping.
        self._ln1.wrappedValue = RMSNorm(dimensions: cfg.dModel, eps: 1e-5)
        self._attn.wrappedValue = CausalSelfAttention(cfg)
        self._ln2.wrappedValue = RMSNorm(dimensions: cfg.dModel, eps: 1e-5)
        self._mlp.wrappedValue = SwiGLU(dModel: cfg.dModel, dMlp: cfg.dMlp, bias: false)
        if cfg.useYOCO && yocoSecondHalf {
            self._crossAttn.wrappedValue = CrossAttention(cfg)
        } else {
            self._crossAttn.wrappedValue = nil
        }
        super.init()
    }

    /// YOCO anchor forward — standard self-attn, ALSO returns K, V so
    /// the downstream cross-attn layers can reuse them.
    public func callCapturingKV(_ x: MLXArray) -> (out: MLXArray, k: MLXArray, v: MLXArray) {
        let (attnOut, k, v) = attn.forwardCapturingKV(ln1(x))
        var y = x + attnOut
        y = y + mlp(ln2(y))
        return (y, k, v)
    }

    /// YOCO cross-attn forward — Q from current x, K/V from the anchor.
    public func callWithExternalKV(_ x: MLXArray, k: MLXArray, v: MLXArray,
                                    posOffset: Int = 0) -> MLXArray {
        let attnOut: MLXArray
        if let ca = crossAttn {
            attnOut = ca(ln1(x), externalK: k, externalV: v, posOffset: posOffset)
        } else {
            attnOut = attn.forwardWithExternalKV(ln1(x), k: k, v: v)
        }
        var y = x + attnOut
        y = y + mlp(ln2(y))
        return y
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
