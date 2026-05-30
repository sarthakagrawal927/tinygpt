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

    /// DeepNorm α — see the matching field on `TransformerBlock`.
    public let deepNormAlpha: Float

    public init(_ cfg: ModelConfig, yocoSecondHalf: Bool = false) {
        self.deepNormAlpha = cfg.deepNormAlpha
        // The HF naming convention is `input_layernorm` for ln1 and
        // `post_attention_layernorm` for ln2. We use those keys so
        // safetensors weight names match without re-mapping.
        self._ln1.wrappedValue = RMSNorm(dimensions: cfg.dModel, eps: 1e-5)
        self._attn.wrappedValue = CausalSelfAttention(cfg)
        self._ln2.wrappedValue = RMSNorm(dimensions: cfg.dModel, eps: 1e-5)
        let swiglu = SwiGLU(dModel: cfg.dModel, dMlp: cfg.dMlp, bias: false)
        swiglu.qatBits = cfg.qatBits
        self._mlp.wrappedValue = swiglu
        if cfg.useYOCO && yocoSecondHalf {
            self._crossAttn.wrappedValue = CrossAttention(cfg)
        } else {
            self._crossAttn.wrappedValue = nil
        }
        super.init()

        // DeepNorm β init — see TransformerBlock for the rationale.
        // For SwiGLU we scale `down_proj` (the output projection) and,
        // following the GLM-130B variant, also the v_proj / o_proj of
        // attention.
        if cfg.useDeepNorm {
            let beta = MLXArray(cfg.deepNormBeta)
            var attnUpd = NestedDictionary<String, Module>()
            attnUpd["v_proj"] = .value(applyBetaInit(attn.vProj, beta: beta))
            attnUpd["o_proj"] = .value(applyBetaInit(attn.oProj, beta: beta))
            attn.update(modules: attnUpd)
            var mlpUpd = NestedDictionary<String, Module>()
            mlpUpd["down_proj"] = .value(applyBetaInit(mlp.fcDown, beta: beta))
            mlp.update(modules: mlpUpd)
        }
    }

    /// YOCO anchor forward — standard self-attn, ALSO returns K, V so
    /// the downstream cross-attn layers can reuse them.
    public func callCapturingKV(_ x: MLXArray) -> (out: MLXArray, k: MLXArray, v: MLXArray) {
        let (attnOut, k, v) = attn.forwardCapturingKV(ln1(x))
        let alphaArr = MLXArray(deepNormAlpha)
        var y = x * alphaArr + attnOut
        y = y * alphaArr + mlp(ln2(y))
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
        let alphaArr = MLXArray(deepNormAlpha)
        var y = x * alphaArr + attnOut
        y = y * alphaArr + mlp(ln2(y))
        return y
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // LayerDrop — see TransformerBlock.callAsFunction for the
        // rationale. Toggled globally via `LayerDropState.probability`
        // so we don't have to thread a per-block field through every
        // existing call site.
        if LayerDropState.shouldDrop() {
            return x
        }
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
        let alphaArr = MLXArray(deepNormAlpha)
        x = x * alphaArr + attn(ln1(x))
        x = x * alphaArr + mlp(ln2(x))
        return x
    }
}
