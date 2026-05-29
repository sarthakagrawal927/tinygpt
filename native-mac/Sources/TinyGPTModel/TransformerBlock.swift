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
    /// Sliding-window size. `nil` = full causal. When set, attention is
    /// restricted to the last W positions — Mistral / GPT-OSS recipe.
    /// We build the mask on-demand inside `callAsFunction` because the
    /// query length T isn't known at construction time.
    public let slidingWindow: Int?
    /// ALiBi (Press et al., 2021): when true, add per-head linear-
    /// distance penalties to attention scores in lieu of positional
    /// embeddings. Slopes are deterministic from `nHeads`.
    public let useALiBi: Bool

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
        self.slidingWindow = cfg.slidingWindow
        self.useALiBi = cfg.useALiBi
        // Q goes from dModel to (nHeads * headDim) = dModel — unchanged
        // K, V go to (nKvHeads * headDim) which is smaller for GQA models
        let kvDim = cfg.nKvHeads * cfg.headDim
        self._qProj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: cfg.attnBias)
        self._kProj.wrappedValue = Linear(cfg.dModel, kvDim, bias: cfg.attnBias)
        self._vProj.wrappedValue = Linear(cfg.dModel, kvDim, bias: cfg.attnBias)
        self._oProj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: cfg.attnBias)
        super.init()
    }

    /// Per-head ALiBi geometric slopes (Press et al., 2021).
    /// `slope[h] = 2^(-8(h+1)/H)` for the closest non-power-of-2
    /// generalisation. Returns Float array of length `nHeads`.
    private func aliBiSlopes() -> [Float] {
        // Standard recipe assumes H is a power of two — handles
        // arbitrary H by interleaving two geometric sequences. Here we
        // use the simpler "base = 2^(-8/H), slope_h = base^(h+1)" which
        // is the recipe most reimplementations ship. Quality difference
        // from the paper's split sequence is small at our scale.
        let base = pow(2.0, -8.0 / Float(nHeads))
        return (0..<nHeads).map { h in pow(base, Float(h + 1)) }
    }

    /// Build a [1, H, T_q, T_kv] ALiBi + causal mask in additive form.
    /// bias[h, i, j] = -slope[h] · (i - j)   if j ≤ i  (causal positions)
    /// bias[h, i, j] = -∞                    if j > i  (future)
    private func aliBiMask(Tq: Int, Tkv: Int, dtype: DType) -> MLXArray {
        let slopes = aliBiSlopes()
        let slopesArr = MLXArray(slopes, [nHeads])
            .expandedDimensions(axis: 1).expandedDimensions(axis: 2)        // [H, 1, 1]
        let rows = MLXArray((0..<Tq).map { Int32($0) }).expandedDimensions(axis: 1)  // [Tq, 1]
        let cols = MLXArray((0..<Tkv).map { Int32($0) }).expandedDimensions(axis: 0) // [1, Tkv]
        let dist = (rows - cols).asType(dtype)                              // [Tq, Tkv] — positive for past
        // -slope[h] * dist per head. Broadcasts over the batch axis later.
        let aliBi = (-slopesArr) * dist.expandedDimensions(axis: 0)         // [H, Tq, Tkv]
        // Causal: positions where j > i are forbidden — add -1e9 there.
        let future = (cols .> rows).asType(dtype)
        let causalNeg = future * MLXArray(Float(-1e9)).asType(dtype)        // [Tq, Tkv]
        let combined = aliBi + causalNeg.expandedDimensions(axis: 0)        // [H, Tq, Tkv]
        // Add the leading batch axis so SDPA's broadcasting works:
        // expected mask shape is [B, H, Tq, Tkv] or broadcastable.
        return combined.expandedDimensions(axis: 0)                         // [1, H, Tq, Tkv]
    }

    /// Build a [T_q, T_kv] sliding-window causal mask in additive form
    /// (0 inside the window, large negative outside — added to scores
    /// before softmax). Built once per forward and reused across heads.
    private func slidingMask(Tq: Int, Tkv: Int, window: Int, dtype: DType) -> MLXArray {
        // rows = query positions [0..Tq), cols = key positions [0..Tkv).
        // For the prefill (Tq == Tkv) case, "outside window" means
        // j > i (future, causal) OR j < i - window + 1 (too far back).
        // For incremental decode (Tq == 1, basePos > 0) we'd need to add
        // basePos to the row index — that lives in the cached-forward
        // path, not this one. Here Tq == Tkv always.
        let rows = MLXArray((0..<Tq).map { Int32($0) }).expandedDimensions(axis: 1)   // [Tq, 1]
        let cols = MLXArray((0..<Tkv).map { Int32($0) }).expandedDimensions(axis: 0)  // [1, Tkv]
        // Two block conditions: future (j > i) and too-old (i - j ≥ W).
        // Cast each Bool tensor to the target dtype (1.0 where blocked,
        // 0.0 otherwise), then take max: produces 1.0 if EITHER condition
        // holds, else 0.0 — exactly the boolean OR we need.
        let futureF = (cols .> rows).asType(dtype)
        let tooOldF = ((rows - cols) .>= MLXArray(Int32(window))).asType(dtype)
        let blocked = MLX.maximum(futureF, tooOldF)
        // Convert blocked-flag → -1e9 (effectively -inf under softmax).
        let negInf = MLXArray(Float(-1e9)).asType(dtype)
        return blocked * negInf
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

        let out = computeSDPA(q: q, k: k, v: v, T: T)
        let merged = out.transposed(0, 2, 1, 3).reshaped([B, T, nHeads * headDim])
        return oProj(merged)
    }

    /// YOCO anchor variant — standard self-attention, ALSO returns
    /// the layer's K and V so downstream cross-attention layers can
    /// reuse them without recomputing. Same math as `callAsFunction`;
    /// just makes the intermediates public.
    public func forwardCapturingKV(_ x: MLXArray) -> (out: MLXArray, k: MLXArray, v: MLXArray) {
        let B = x.shape[0]
        let T = x.shape[1]
        var q = qProj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped([B, T, nKvHeads, headDim]).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([B, T, nKvHeads, headDim]).transposed(0, 2, 1, 3)
        if useRoPE {
            q = MLXFast.RoPE(q, dimensions: headDim, traditional: false,
                              base: ropeBase, scale: 1.0, offset: 0)
            k = MLXFast.RoPE(k, dimensions: headDim, traditional: false,
                              base: ropeBase, scale: 1.0, offset: 0)
        }
        let out = computeSDPA(q: q, k: k, v: v, T: T)
        let merged = out.transposed(0, 2, 1, 3).reshaped([B, T, nHeads * headDim])
        return (oProj(merged), k, v)
    }

    /// YOCO cross-attention variant — Q from current x, (K, V) supplied
    /// externally (the anchor's saved tensors). kProj / vProj are NOT
    /// called: that's the KV-cache memory saving. Q gets the same
    /// RoPE rotation as the anchor's K had — preserving relative
    /// position information.
    public func forwardWithExternalKV(_ x: MLXArray, k: MLXArray, v: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        var q = qProj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        if useRoPE {
            q = MLXFast.RoPE(q, dimensions: headDim, traditional: false,
                              base: ropeBase, scale: 1.0, offset: 0)
        }
        let out = computeSDPA(q: q, k: k, v: v, T: T)
        let merged = out.transposed(0, 2, 1, 3).reshaped([B, T, nHeads * headDim])
        return oProj(merged)
    }

    /// Shared SDPA dispatch — ALiBi → sliding window → plain causal in
    /// priority order. Factored out so the three callers (standard
    /// forward, YOCO anchor, YOCO cross-attn) stay short.
    private func computeSDPA(q: MLXArray, k: MLXArray, v: MLXArray, T: Int) -> MLXArray {
        if useALiBi {
            let mask = aliBiMask(Tq: T, Tkv: T, dtype: q.dtype)
            return MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .array(mask)
            )
        } else if let window = slidingWindow {
            let mask = slidingMask(Tq: T, Tkv: T, window: window, dtype: q.dtype)
            return MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .array(mask)
            )
        } else {
            return MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .causal
            )
        }
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
///
/// MoE option: when `cfg.isMoE`, the block constructs a `MoEMLP` instead
/// of the dense `MLP`. The two are mutually exclusive — exactly one of
/// `mlp` and `moe` is populated. Forward routes to whichever is non-nil;
/// LoRA + save paths gate on `mlp != nil` (MoE blocks aren't LoRA-
/// targetable or .tinygpt-serialisable in this first cut).
public final class TransformerBlock: Module {
    @ModuleInfo(key: "ln1") public var ln1: LayerNorm
    @ModuleInfo(key: "attn") public var attn: CausalSelfAttention
    @ModuleInfo(key: "ln2") public var ln2: LayerNorm
    @ModuleInfo(key: "mlp") public var mlp: MLP?
    @ModuleInfo(key: "moe") public var moe: MoEMLP?
    /// Differential attention (Ye et al., 2024) — when cfg
    /// .useDifferentialAttention is set, this Optional sibling is
    /// populated and the forward routes through it instead of `attn`.
    /// The standard `attn` stays constructed (and its params land in
    /// the manifest as usual) — a small ~constant cost in exchange
    /// for keeping every existing call site that touches `block.attn`
    /// unchanged.
    @ModuleInfo(key: "diff_attn") public var diffAttn: DifferentialAttention?
    /// YOCO cross-attention sibling — populated on SECOND-half layers when
    /// `cfg.useYOCO` is set. When non-nil, the block's forward routes
    /// through `crossAttn` (Q-only projections) instead of `attn`. The
    /// caller (TinyGPTModel.forwardToHidden) is responsible for threading
    /// the anchor's K, V into `callWithCrossAttention`. The existing
    /// `attn` stays allocated so the manifest layout / LoRA targeting
    /// stays stable; its weights are dead at forward time on second-half
    /// layers but are still trained as part of the param tree until the
    /// first save filters them out (future work). See CrossAttention.swift
    /// for the design rationale of a dedicated module vs. extension-on-
    /// `CausalSelfAttention`.
    @ModuleInfo(key: "cross_attn") public var crossAttn: CrossAttention?
    /// Mixture-of-Depths per-token router (Raposo et al., 2024).
    /// Linear(d_model → 1), populated when cfg.useMoD. When present,
    /// the block's contribution is gated by sigmoid(router(x)) per
    /// token — tokens the router scores low pass through unchanged.
    /// Soft routing here; hard top-K + scatter is queued behind the
    /// same scatter_add upstream gap as MoE sparse dispatch.
    @ModuleInfo(key: "mod_router") public var modRouter: Linear?

    /// Type-erased pointer to whichever MLP-shaped unit owns this block.
    /// Used by introspection helpers (e.g. `sumMoEAuxLosses`) that need
    /// to walk the model without caring which flavour is active.
    public var mlpUnit: Module { (mlp as Module?) ?? moe! }

    /// Gradient checkpointing toggle — when true, `callAsFunction` wraps
    /// the raw block forward in a `GradCheckpoint.wrap(...)` so the
    /// block's intermediate activations don't persist across the outer
    /// backward. Not a `Module` parameter / not serialised on the block
    /// itself; the model sets it after construction based on the config.
    public var useGradCheckpoint: Bool = false

    public init(_ cfg: ModelConfig, yocoSecondHalf: Bool = false) {
        self._ln1.wrappedValue = LayerNorm(dimensions: cfg.dModel, eps: 1e-5)
        self._attn.wrappedValue = CausalSelfAttention(cfg)
        self._ln2.wrappedValue = LayerNorm(dimensions: cfg.dModel, eps: 1e-5)
        if cfg.isMoE {
            self._mlp.wrappedValue = nil
            self._moe.wrappedValue = MoEMLP(cfg)
        } else {
            self._mlp.wrappedValue = MLP(cfg)
            self._moe.wrappedValue = nil
        }
        if cfg.useMoD {
            // bias init defaults to 0 — at init the sigmoid gate ≈ 0.5,
            // i.e. the block applies ~half of its delta on every token.
            // Training pushes the gate towards 0 or 1 per token.
            self._modRouter.wrappedValue = Linear(cfg.dModel, 1, bias: true)
        } else {
            self._modRouter.wrappedValue = nil
        }
        if cfg.useDifferentialAttention {
            self._diffAttn.wrappedValue = DifferentialAttention(cfg)
        } else {
            self._diffAttn.wrappedValue = nil
        }
        // YOCO crossAttn — set HERE at init time (the only legal point
        // to set an Optional @ModuleInfo) when this layer is in the
        // second half. The block needs the layer index to decide; init
        // takes `yocoSecondHalf` from the model. Defaults to false to
        // keep all existing TransformerBlock(cfg) call sites working.
        if cfg.useYOCO && yocoSecondHalf {
            self._crossAttn.wrappedValue = CrossAttention(cfg)
        } else {
            self._crossAttn.wrappedValue = nil
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if useGradCheckpoint {
            // Wrap the raw block forward in a CustomFunction whose VJP
            // re-runs the same forward at backward time. Block params
            // are threaded through as CustomFunction inputs so MLX's
            // autodiff still routes gradients to them. See
            // GradCheckpoint.swift for the full mechanism.
            return GradCheckpoint.wrap(block: self, x: x) { b, xt in
                b.rawForward(xt)
            }
        }
        return rawForward(x)
    }

    /// Raw block forward — used both as the standard non-checkpointed
    /// path AND as the "recompute" payload inside `GradCheckpoint.wrap`.
    /// Importantly, this method does NOT consult `useGradCheckpoint`,
    /// so the checkpoint wrapper's VJP doesn't recurse into itself.
    public func rawForward(_ x: MLXArray) -> MLXArray {
        return blockAfterAttn(x: x, attnOut: attentionFor(x: x))
    }

    /// Standard attention path — picks differential variant when
    /// configured, else the regular self-attention.
    private func attentionFor(x: MLXArray) -> MLXArray {
        (diffAttn != nil) ? diffAttn!(ln1(x)) : attn(ln1(x))
    }

    /// YOCO anchor variant: returns (block output, K, V) so the
    /// downstream cross-attention layers can reuse the K, V without
    /// recomputing them. Differential attention has no K, V to capture
    /// — anchoring at a diff-attn layer falls back to standard
    /// self-attention for that layer.
    public func callCapturingKV(_ x: MLXArray) -> (out: MLXArray, k: MLXArray, v: MLXArray) {
        let (attnOut, k, v) = attn.forwardCapturingKV(ln1(x))
        let y = blockAfterAttn(x: x, attnOut: attnOut)
        return (y, k, v)
    }

    /// YOCO cross-attention variant: Q is fresh from current x; K, V
    /// come from the anchor. kProj / vProj are NOT invoked — that's
    /// the KV-cache memory saving downstream. Prefers the dedicated
    /// `crossAttn` sibling when installed (no k_proj/v_proj allocated
    /// at all on that path); falls back to the `attn.forwardWithExternalKV`
    /// extension for backwards compatibility with blocks built before
    /// `installCrossAttention` was wired.
    public func callWithExternalKV(_ x: MLXArray, k: MLXArray, v: MLXArray,
                                    posOffset: Int = 0) -> MLXArray {
        let attnOut: MLXArray
        if let ca = crossAttn {
            attnOut = ca(ln1(x), externalK: k, externalV: v, posOffset: posOffset)
        } else {
            // Legacy path — only correct when posOffset == 0 because
            // `forwardWithExternalKV` doesn't take an offset.
            attnOut = attn.forwardWithExternalKV(ln1(x), k: k, v: v)
        }
        return blockAfterAttn(x: x, attnOut: attnOut)
    }

    /// Shared post-attention path (MLP / MoE / MoD gate) for the three
    /// flavours of attention — standard, anchor-capturing, cross-attn.
    private func blockAfterAttn(x: MLXArray, attnOut: MLXArray) -> MLXArray {
        let blockIn = x
        var y = blockIn + attnOut
        if let moe = moe {
            y = y + moe(ln2(y))
        } else if let mlp = mlp {
            y = y + mlp(ln2(y))
        }
        // MoD soft routing: per-token sigmoid gate scales the block's
        // total delta. gate ≈ 1 → block fires as normal; gate ≈ 0 →
        // token bypasses the block entirely.
        if let router = modRouter {
            // router(x): [B, T, 1]. sigmoid via 1 / (1 + exp(-x)).
            let logits = router(blockIn)
            let gate = MLXArray(Float(1)) / (MLXArray(Float(1)) + MLX.exp(-logits))
            // y = blockIn + gate * (y - blockIn). Broadcast over C via the
            // trailing dim of size 1 on `gate`.
            return blockIn + gate * (y - blockIn)
        }
        return y
    }
}
