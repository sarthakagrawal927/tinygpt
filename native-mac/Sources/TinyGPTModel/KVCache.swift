import Foundation
import MLX
import MLXFast
import MLXNN

/// KV cache for autoregressive sampling.
///
/// Without a cache, every generated token re-runs attention over the full
/// context: O(T²) work per token × T tokens = O(T³) total. With a cache,
/// each step we only compute Q for the new position and use the stored
/// K and V from past positions. Per-token work drops to O(T), making
/// total generation O(T²) — and the practical speedup is 10-50× depending
/// on context length and model depth.
///
/// Per-layer state: stacked tensors of shape `[B, H, T_so_far, D]` for
/// keys and values. Grown by `T_new` (usually 1) each step.
///
/// Three storage modes coexist behind one façade:
///   - **fp32/fp16/bf16** dtype storage (the default and the `--kv-quantize fp16/bf16`
///     paths). Stored verbatim in `Entry.keys` / `Entry.values`; reads cast
///     to whatever Q's dtype is so SDPA sees a consistent precision.
///   - **KIVI int8 / int4** (Liu et al., 2023). Keys are quantised per-channel
///     (one min/max scale per (B, H, channel), shared across time);
///     values are quantised per-token (one scale per (B, H, t), shared
///     across channels). Stored in `Entry.keysQ` / `Entry.valuesQ` (int8
///     dtype, range ±127 for int8 mode and ±7 for int4 mode); scales /
///     zero-points live in `Entry.kScales|kZeros|vScales|vZeros` as fp16.
///     On read, we dequantise: `x_dequant = (q - zero) * scale`.
public final class KVCache {
    public struct Entry {
        // High-precision storage path (fp32/fp16/bf16). When KIVI is on,
        // these are 0-shape placeholders.
        public var keys: MLXArray   // [B, H, T, D]
        public var values: MLXArray // [B, H, T, D]

        // KIVI-quantised storage. `.int8` dtype either way (int4 uses the
        // same int8 storage with a tighter ±7 range — the precision loss
        // matches true int4, the storage cost matches int8; we report
        // both byte counts in the doc so the tradeoff is honest).
        public var keysQ: MLXArray?   // [B, H, T, D] int8
        public var kScales: MLXArray? // [B, H, D]    fp16 — per-channel scale
        public var kZeros: MLXArray?  // [B, H, D]    fp16 — per-channel zero point

        public var valuesQ: MLXArray?   // [B, H, T, D] int8
        public var vScales: MLXArray? // [B, H, T]    fp16 — per-token scale
        public var vZeros: MLXArray?  // [B, H, T]    fp16 — per-token zero point
    }

    public var entries: [Entry]
    public let nLayers: Int
    public var currentLength: Int = 0

    /// Pre-allocated maximum tokens for the in-place storage path. `nil`
    /// (default) = grow-by-concat: each `append` allocates a new MLXArray
    /// at the next length (the historic behaviour). When set to a positive
    /// value, the FIRST append allocates a full `[B, H, capacity, D]`
    /// buffer per layer and subsequent appends write into the existing
    /// rows via slice assignment — no per-step allocation, no growing
    /// concat, peak memory roughly constant across decode steps.
    ///
    /// Reads honour `validLengths[layer]` so SDPA only sees the populated
    /// prefix of each layer's buffer, not the trailing zeros.
    ///
    /// `var` (not `let`) so a disk-loaded cache can be promoted into
    /// in-place mode via `migrateToPreAlloc(capacity:)` — the persistent
    /// cache path saves at the live length and we want the in-place fast
    /// path to keep working through the remaining decode loop.
    public var preAllocCapacity: Int?

    /// Per-layer count of populated time slots in the pre-allocated
    /// buffer. Equal to `currentLength` for layers that have been written
    /// the same number of times — divergence is reserved for future
    /// per-layer eviction (e.g., per-layer YOCO drops). Stays at 0 for
    /// layers that never received a write. Unused in grow-by-concat mode.
    public var validLengths: [Int] = []

    /// Storage dtype for cached K/V. `nil` (default) = match whatever the
    /// attention computes natively. Set to `.float16` to halve KV memory
    /// when running an fp32 model — attention output is dequantised on
    /// read, so accuracy stays within fp16 noise.
    ///
    /// Ignored when `kivi` is set (int8/int4 path takes over).
    public let kvDtype: DType?

    /// KIVI quantisation config (Liu et al., 2023). `nil` (default) = the
    /// fp16/bf16/fp32 dtype path. Set to `.int8` or `.int4` and the cache
    /// switches to per-channel-K / per-token-V affine quantisation. The
    /// dequantise happens on every read (cheap — just elementwise mul +
    /// add) so SDPA sees fp16/fp32 Q, K, V.
    public struct KIVIConfig: Sendable, Equatable {
        public var bits: Int          // 4 or 8
        public init(bits: Int) {
            precondition(bits == 4 || bits == 8, "KIVI bits must be 4 or 8 (got \(bits))")
            self.bits = bits
        }
        /// Quantisation range. int8 uses [-128, 127], int4 uses [-8, 7]
        /// (same int8 storage, but precision rounded to 16 levels — that's
        /// the "fake int4" trade we make so cache slicing under
        /// StreamingLLM stays simple).
        public var qMin: Float { bits == 4 ? -8  : -128 }
        public var qMax: Float { bits == 4 ?  7  :  127 }
        public var levels: Float { qMax - qMin }
    }
    public let kivi: KIVIConfig?

    /// StreamingLLM (Xiao et al., 2024) — keep the FIRST `sink` tokens
    /// always, and the LAST `window` tokens, dropping everything in
    /// between when the cache exceeds `sink + window`. Sliding-window
    /// attention with the leading anchor that makes the model survive
    /// arbitrarily-long generations. `nil` (default) = no pruning,
    /// pure causal accumulation.
    ///
    /// RoPE caveat: keys are stored POST-rotation at their ORIGINAL
    /// absolute positions. After eviction, sink keys retain RoPE phases
    /// from positions 0..sink-1, window keys from their actual generation
    /// positions. New queries are rotated at `cache.currentLength`
    /// (the true generated count, NOT the cache length). This is the
    /// "vanilla" StreamingLLM scheme — coherent because the RoPE relative
    /// distances Q↔window-K stay small (local context) and Q↔sink-K
    /// stretches into out-of-distribution-large distances but the sink
    /// tokens' role is to anchor attention's softmax norm, not to be
    /// content-attended-to, so the OOD distance is forgiving.
    public let sink: Int?
    public let window: Int?

    public init(
        nLayers: Int,
        kvDtype: DType? = nil,
        kivi: KIVIConfig? = nil,
        sink: Int? = nil, window: Int? = nil,
        preAllocCapacity: Int? = nil
    ) {
        self.nLayers = nLayers
        self.entries = []
        self.entries.reserveCapacity(nLayers)
        self.kvDtype = kvDtype
        self.kivi = kivi
        self.sink = sink
        self.window = window
        // In-place is mutually exclusive with KIVI / StreamingLLM (eviction
        // would require fragmenting the contiguous buffer; we punt to a
        // future patch). Silently fall back to grow-by-concat when those
        // features are active.
        if kivi != nil || sink != nil || window != nil {
            self.preAllocCapacity = nil
        } else {
            self.preAllocCapacity = preAllocCapacity
        }
        self.validLengths = Array(repeating: 0, count: nLayers)
    }

    public func append(layer: Int, keys: MLXArray, values: MLXArray) {
        // KIVI takes precedence over kvDtype storage — they're mutually
        // exclusive on disk (you don't store quantised AND fp16 of the
        // same K, V). When neither is set we get raw whatever-dtype the
        // attention computed in.
        if kivi != nil {
            appendKIVI(layer: layer, kNew: keys, vNew: values)
        } else {
            appendDense(layer: layer, kNew: keys, vNew: values)
        }
        // StreamingLLM pruning: drop the middle of the cache once it
        // exceeds sink + window. Run AFTER concatenation so the just-
        // -written tail is the part we keep. Per-layer because each
        // layer's cache grows in lockstep.
        if let s = sink, let w = window {
            evictMiddleIfNeeded(layer: layer, sink: s, window: w)
        }
    }

    private func appendDense(layer: Int, kNew: MLXArray, vNew: MLXArray) {
        let kIn = kvDtype.map { kNew.asType($0) } ?? kNew
        let vIn = kvDtype.map { vNew.asType($0) } ?? vNew
        // Make sure per-layer slots exist (entries + validLengths).
        if entries.count <= layer {
            while entries.count <= layer {
                entries.append(Entry(
                    keys: kIn[0..., 0..., 0..<0, 0...],
                    values: vIn[0..., 0..., 0..<0, 0...]
                ))
            }
        }
        while validLengths.count <= layer { validLengths.append(0) }

        if let cap = preAllocCapacity {
            appendInPlace(layer: layer, kIn: kIn, vIn: vIn, capacity: cap)
        } else {
            appendByConcat(layer: layer, kIn: kIn, vIn: vIn)
        }
    }

    /// Historic grow-by-concat path. Each call allocates a new MLXArray
    /// sized `currentLength + T_new` along axis=2. Simple, correct, but
    /// peak memory scales with cache length. Used when
    /// `preAllocCapacity == nil` and when KIVI / StreamingLLM are on.
    private func appendByConcat(layer: Int, kIn: MLXArray, vIn: MLXArray) {
        let cur = entries[layer].keys
        if cur.shape[2] == 0 {
            entries[layer].keys = kIn
            entries[layer].values = vIn
        } else {
            entries[layer].keys = concatenated([cur, kIn], axis: 2)
            entries[layer].values = concatenated([entries[layer].values, vIn], axis: 2)
        }
        validLengths[layer] = entries[layer].keys.shape[2]
    }

    /// In-place path. On the first write to this layer we materialise the
    /// full `[B, H, capacity, D]` buffer once (`MLXArray.zeros`), then for
    /// every subsequent step we slice-assign `[:, :, valid:valid+T_new, :]`.
    /// MLX-Swift's slice-set goes through a scatter op which still produces
    /// a new MLXArray (MLX-C arrays are functional by design), but the
    /// allocation cost is bounded by the SLICE size (T_new × D × bytes),
    /// not by the cache length — so peak memory across a long decode
    /// stays bounded at ~capacity once.
    ///
    /// Why we still get a measurable win even though MLX scatter copies:
    /// the OUTPUT of the scatter is itself the full buffer, so we keep one
    /// MLXArray alive across all steps. The concat path on the other hand
    /// keeps creating ever-larger arrays whose lifetimes overlap with the
    /// previous step's keys / values until MLX's reference counting frees
    /// them — and Metal's allocator hasn't always released memory promptly
    /// under pressure. Pre-allocation pins the working set at the start.
    private func appendInPlace(layer: Int, kIn: MLXArray, vIn: MLXArray, capacity: Int) {
        let valid = validLengths[layer]
        let tNew = kIn.shape[2]
        precondition(valid + tNew <= capacity,
                     "KV cache overflow: valid \(valid) + new \(tNew) > capacity \(capacity)")
        let keys: MLXArray
        let values: MLXArray
        if entries[layer].keys.shape[2] != capacity {
            // First write to this layer — materialise the full buffer.
            // Shape matches the new write's leading axes; we just stretch
            // axis=2 to `capacity` and zero-fill the unused tail.
            let B = kIn.shape[0]
            let H = kIn.shape[1]
            let D = kIn.shape[3]
            keys = MLXArray.zeros([B, H, capacity, D]).asType(kIn.dtype)
            let Hv = vIn.shape[1]
            let Dv = vIn.shape[3]
            values = MLXArray.zeros([B, Hv, capacity, Dv]).asType(vIn.dtype)
        } else {
            keys = entries[layer].keys
            values = entries[layer].values
        }
        // Slice-assign the new K, V rows into the persistent buffer.
        // The right-hand-side broadcasts along the time slice we name.
        // MLXArray is a reference type so the subscript setter mutates
        // the buffer in place — the binding stays `let`.
        keys[0..., 0..., valid..<(valid + tNew), 0...] = kIn
        values[0..., 0..., valid..<(valid + tNew), 0...] = vIn
        entries[layer].keys = keys
        entries[layer].values = values
        validLengths[layer] = valid + tNew
    }

    /// KIVI quantisation path. `kNew` / `vNew` are the fresh K/V projected
    /// at this step (fp16/fp32). We do:
    ///
    ///   - V: quantise NEW token(s) per-token (one scale per (B, H, t)
    ///     computed over D channels). Append to `valuesQ` + scale arrays.
    ///   - K: append NEW token(s) fp16 to a temporary buffer, then RE-
    ///     COMPUTE per-channel min/max over ALL cached K (old + new) and
    ///     re-quantise the whole K block. Per-channel scales are global
    ///     across the time axis, so any new token can shift the channel
    ///     range — must recompute. Cost is O(T·D) per step; T is bounded
    ///     by context length, fine for sample-time.
    ///
    /// **Bug the previous attempt hit (silent NaN → 0 tokens)**:
    ///   Common KIVI failure modes are (a) computing `scale = range /
    ///   levels` when range == 0 (constant channel/token) → div-by-eps
    ///   blow-up or scale == 0 → dequant gives NaN; (b) storing scales
    ///   in fp16 but reading back as fp32 and forgetting the cast; (c)
    ///   off-by-one in clip range (clamp to [qMin, qMax] not [qMin,
    ///   qMax-1]). We guard all three: scale floored at 1e-5, dtype
    ///   matched on read, qMin/qMax explicit. Plus: stash a high-prec
    ///   "residual" of new K/V for the next-step recompute (so we don't
    ///   dequant-then-requant in a quality-losing loop).
    private func appendKIVI(layer: Int, kNew: MLXArray, vNew: MLXArray) {
        let cfg = kivi!
        // Ensure layer slot exists.
        if entries.count <= layer {
            while entries.count <= layer {
                entries.append(Entry(
                    keys: kNew.asType(.float32)[0..., 0..., 0..<0, 0...],
                    values: vNew.asType(.float32)[0..., 0..., 0..<0, 0...]
                ))
            }
        }
        // High-precision residual for K (last appended tokens, fp16). Reused
        // each step to re-quantise. We grow this exactly like the dense
        // path's `keys`. It's the cheapest correct way to handle the
        // "per-channel scale changes when new tokens arrive" case without
        // dequant→requant loss.
        let kF = kNew.asType(.float16)
        let vF = vNew.asType(.float16)
        if entries[layer].keys.shape[2] == 0 {
            entries[layer].keys = kF
        } else {
            // Cast the existing residual to fp16 if it was fp32 from the
            // 0-shape placeholder (shouldn't happen after first append but
            // guard anyway).
            let existing = entries[layer].keys.dtype == .float16
                ? entries[layer].keys : entries[layer].keys.asType(.float16)
            entries[layer].keys = concatenated([existing, kF], axis: 2)
        }
        // ============ K QUANTISATION (per-channel, over all T) ============
        // Shape [B, H, T, D]. We want min/max over axis=2 (time).
        let kBuf = entries[layer].keys
        let kMin = kBuf.min(axis: 2, keepDims: false).asType(.float32) // [B, H, D]
        let kMax = kBuf.max(axis: 2, keepDims: false).asType(.float32)
        let (kQ, kSc, kZp) = Self.quantiseAffine(
            x: kBuf.asType(.float32), xMin: kMin, xMax: kMax,
            broadcastAxis: 2, // expand scales along T to broadcast over kBuf
            qMin: cfg.qMin, qMax: cfg.qMax
        )
        entries[layer].keysQ = kQ
        entries[layer].kScales = kSc.asType(.float16)
        entries[layer].kZeros = kZp.asType(.float16)

        // ============ V QUANTISATION (per-token, over all D) ==============
        // For V we only quantise the NEW tokens — their scales never
        // change once written (per-token scale is fixed at append). Then
        // concat with the existing valuesQ.
        let vMin = vF.asType(.float32).min(axis: 3, keepDims: false) // [B, H, T_new]
        let vMax = vF.asType(.float32).max(axis: 3, keepDims: false)
        let (vQ, vSc, vZp) = Self.quantiseAffine(
            x: vF.asType(.float32), xMin: vMin, xMax: vMax,
            broadcastAxis: 3, // expand scales along D
            qMin: cfg.qMin, qMax: cfg.qMax
        )
        if entries[layer].valuesQ == nil || entries[layer].valuesQ!.shape[2] == 0 {
            entries[layer].valuesQ = vQ
            entries[layer].vScales = vSc.asType(.float16)
            entries[layer].vZeros = vZp.asType(.float16)
        } else {
            entries[layer].valuesQ = concatenated([entries[layer].valuesQ!, vQ], axis: 2)
            entries[layer].vScales = concatenated(
                [entries[layer].vScales!, vSc.asType(.float16)], axis: 2)
            entries[layer].vZeros = concatenated(
                [entries[layer].vZeros!, vZp.asType(.float16)], axis: 2)
        }
        // values residual is unused (V is final once quantised) — keep at
        // 0-shape to save memory.
        entries[layer].values = vF[0..., 0..., 0..<0, 0...]
    }

    /// Affine quantisation: `q = round((x - zero) / scale)` then clipped
    /// to [qMin, qMax]; `dequant = q * scale + zero`. Returns (q, scale,
    /// zero) with scale and zero broadcastable to x along `broadcastAxis`.
    ///
    /// `xMin` and `xMax` are the per-channel/per-token extremes (must
    /// already have the broadcast axis squeezed out). We expand them with
    /// `expandedDimensions(axis: broadcastAxis)` so multiply/subtract
    /// against x works element-wise.
    ///
    /// **NaN guard**: when `xMax == xMin` (constant), `scale` would
    /// collapse to 0 and dequant → NaN. We floor scale at 1e-5. The cost
    /// is at-worst-tiny quantisation error in pathological cases.
    private static func quantiseAffine(
        x: MLXArray, xMin: MLXArray, xMax: MLXArray,
        broadcastAxis: Int, qMin: Float, qMax: Float
    ) -> (q: MLXArray, scale: MLXArray, zero: MLXArray) {
        let levels = qMax - qMin
        // scale per slice. Floored at 1e-5 to avoid NaN on constant data.
        let scaleRaw = (xMax - xMin) / MLXArray(levels)
        let scale = MLX.maximum(scaleRaw, MLXArray(Float(1e-5)))
        // zero point = lower extreme — so dequant of qMin recovers xMin.
        let zero = xMin
        // Broadcast helpers along the axis we squeezed.
        let scaleB = scale.expandedDimensions(axis: broadcastAxis)
        let zeroB = zero.expandedDimensions(axis: broadcastAxis)
        // Quantise. `round` returns the same dtype as x; cast to int8 for
        // storage. Clip BEFORE the round to avoid round-then-clip
        // pathologies on boundary values.
        let qFloat = clip(
            MLX.round((x - zeroB) / scaleB + MLXArray(qMin)),
            min: MLXArray(qMin), max: MLXArray(qMax))
        return (qFloat.asType(.int8), scale, zero)
    }

    /// Dequantise the K block. `cache.keysQ` is int8 [B,H,T,D]; scales/
    /// zeros are fp16 [B,H,D]. We compute `(q - qMin) * scale + zero`
    /// (the inverse of `quantiseAffine`) and cast to the requested dtype.
    private static func dequantiseK(
        q: MLXArray, scale: MLXArray, zero: MLXArray,
        qMin: Float, asDType dt: DType
    ) -> MLXArray {
        let qF = q.asType(.float32) - MLXArray(qMin)
        let scaleB = scale.asType(.float32).expandedDimensions(axis: 2) // [B,H,1,D]
        let zeroB = zero.asType(.float32).expandedDimensions(axis: 2)
        let result = qF * scaleB + zeroB
        return result.asType(dt)
    }

    /// Dequantise the V block. `valuesQ` is int8 [B,H,T,D]; scales/zeros
    /// are fp16 [B,H,T]. Per-token scales broadcast along D.
    private static func dequantiseV(
        q: MLXArray, scale: MLXArray, zero: MLXArray,
        qMin: Float, asDType dt: DType
    ) -> MLXArray {
        let qF = q.asType(.float32) - MLXArray(qMin)
        let scaleB = scale.asType(.float32).expandedDimensions(axis: 3) // [B,H,T,1]
        let zeroB = zero.asType(.float32).expandedDimensions(axis: 3)
        let result = qF * scaleB + zeroB
        return result.asType(dt)
    }

    /// StreamingLLM eviction: drop indices [sink, len-window) along T.
    /// Operates on whichever storage is live (dense or KIVI). For KIVI,
    /// we slice the quantised tensor AND the per-token V scales. K's
    /// per-channel scales are independent of T so they stay as-is, but
    /// we ALSO need to re-quantise after eviction because the per-channel
    /// min/max are now computed over fewer tokens — skip that re-quant
    /// for simplicity, accept the slight imprecision (the dropped tokens
    /// were unlikely to be the per-channel extremes anyway; if they were,
    /// the next append's re-quant absorbs it).
    private func evictMiddleIfNeeded(layer: Int, sink s: Int, window w: Int) {
        // Source-of-truth for the cache length is the active storage path.
        let len: Int
        if kivi != nil {
            len = entries[layer].keysQ?.shape[2] ?? 0
        } else {
            len = entries[layer].keys.shape[2]
        }
        guard len > s + w else { return }
        let dropStart = s
        let dropEnd = len - w
        if kivi != nil {
            // Slice quantised K and the residual fp16 K.
            let kQ = entries[layer].keysQ!
            let kQHead = kQ[0..., 0..., 0..<dropStart, 0...]
            let kQTail = kQ[0..., 0..., dropEnd..<len, 0...]
            entries[layer].keysQ = concatenated([kQHead, kQTail], axis: 2)
            // Residual fp16 K — must mirror keysQ so the next append's
            // re-quant covers the right set of tokens.
            let kRes = entries[layer].keys
            let kResHead = kRes[0..., 0..., 0..<dropStart, 0...]
            let kResTail = kRes[0..., 0..., dropEnd..<len, 0...]
            entries[layer].keys = concatenated([kResHead, kResTail], axis: 2)
            // Slice quantised V + its per-token scales/zeros.
            let vQ = entries[layer].valuesQ!
            let vSc = entries[layer].vScales!
            let vZp = entries[layer].vZeros!
            let vQHead = vQ[0..., 0..., 0..<dropStart, 0...]
            let vQTail = vQ[0..., 0..., dropEnd..<len, 0...]
            let vScHead = vSc[0..., 0..., 0..<dropStart]
            let vScTail = vSc[0..., 0..., dropEnd..<len]
            let vZpHead = vZp[0..., 0..., 0..<dropStart]
            let vZpTail = vZp[0..., 0..., dropEnd..<len]
            entries[layer].valuesQ = concatenated([vQHead, vQTail], axis: 2)
            entries[layer].vScales = concatenated([vScHead, vScTail], axis: 2)
            entries[layer].vZeros = concatenated([vZpHead, vZpTail], axis: 2)
        } else {
            let kHead = entries[layer].keys[0..., 0..., 0..<dropStart, 0...]
            let kTail = entries[layer].keys[0..., 0..., dropEnd..<len, 0...]
            let vHead = entries[layer].values[0..., 0..., 0..<dropStart, 0...]
            let vTail = entries[layer].values[0..., 0..., dropEnd..<len, 0...]
            entries[layer].keys = concatenated([kHead, kTail], axis: 2)
            entries[layer].values = concatenated([vHead, vTail], axis: 2)
        }
    }

    public func keys(layer: Int) -> MLXArray? { entries.indices.contains(layer) ? entries[layer].keys : nil }
    public func values(layer: Int) -> MLXArray? { entries.indices.contains(layer) ? entries[layer].values : nil }

    /// Logical "occupied" length for a layer's buffer. Equal to the dense
    /// shape[2] when in-place isn't on, otherwise the in-place valid count.
    /// SDPA reads slice the buffer down to this length so the model never
    /// attends to the zero-padded tail.
    private func liveLen(_ layer: Int) -> Int {
        if preAllocCapacity != nil, layer < validLengths.count {
            return validLengths[layer]
        }
        return entries[layer].keys.shape[2]
    }

    /// Read-back with on-the-fly dtype upcast. Used by the attention
    /// extensions so SDPA always sees Q/K/V in the same dtype while the
    /// stored cache stays at `kvDtype` or KIVI-quantised (the memory-
    /// saving formats).
    ///
    /// When `preAllocCapacity` is set, the underlying buffer is sized at
    /// the model's max context; we slice it down to the live length here
    /// so SDPA's mask + matmul shapes line up with the actual token count.
    public func keys(layer: Int, asDType dt: DType) -> MLXArray? {
        guard entries.indices.contains(layer) else { return nil }
        let e = entries[layer]
        if let cfg = kivi, let q = e.keysQ, let sc = e.kScales, let zp = e.kZeros {
            return Self.dequantiseK(q: q, scale: sc, zero: zp, qMin: cfg.qMin, asDType: dt)
        }
        let k = e.keys
        let cast = k.dtype == dt ? k : k.asType(dt)
        if preAllocCapacity != nil {
            let n = liveLen(layer)
            if n < cast.shape[2] { return cast[0..., 0..., 0..<n, 0...] }
        }
        return cast
    }
    public func values(layer: Int, asDType dt: DType) -> MLXArray? {
        guard entries.indices.contains(layer) else { return nil }
        let e = entries[layer]
        if let cfg = kivi, let q = e.valuesQ, let sc = e.vScales, let zp = e.vZeros {
            return Self.dequantiseV(q: q, scale: sc, zero: zp, qMin: cfg.qMin, asDType: dt)
        }
        let v = e.values
        let cast = v.dtype == dt ? v : v.asType(dt)
        if preAllocCapacity != nil {
            let n = liveLen(layer)
            if n < cast.shape[2] { return cast[0..., 0..., 0..<n, 0...] }
        }
        return cast
    }

    /// Total cached-K/V bytes across all layers. Counts the active storage
    /// path only — dense (`keys` + `values`) when KIVI is off, or
    /// (`keysQ` + `valuesQ` + scales + zeros) when KIVI is on. Reported in
    /// `tinygpt sample`'s footer for the memory-tradeoff smoke tests.
    public func totalBytes(byteWidth: (DType) -> Int) -> (bytes: Int, populated: Int) {
        // Two byte counts to keep honest:
        //   - PHYSICAL bytes: actual allocated buffer size. In the
        //     in-place / pre-allocated mode this is the buffer at
        //     `capacity` even when only some rows are populated.
        //   - LOGICAL bytes: bytes worth of populated rows, i.e. what a
        //     concat-mode cache would be sitting on at the same valid
        //     length. For the GQA-aware audit + the autoregressive memory
        //     report, the LOGICAL count is what matters — that's the
        //     content the model actually attends to. Footer reports
        //     logical bytes so the YOCO / GQA / KV-quantise savings
        //     show up cleanly.
        var total = 0
        var populated = 0
        for (i, e) in entries.enumerated() {
            if let kQ = e.keysQ, let vQ = e.valuesQ {
                total += kQ.shape.reduce(1, *) * byteWidth(kQ.dtype)
                total += vQ.shape.reduce(1, *) * byteWidth(vQ.dtype)
                if let sc = e.kScales { total += sc.shape.reduce(1, *) * byteWidth(sc.dtype) }
                if let zp = e.kZeros { total += zp.shape.reduce(1, *) * byteWidth(zp.dtype) }
                if let sc = e.vScales { total += sc.shape.reduce(1, *) * byteWidth(sc.dtype) }
                if let zp = e.vZeros { total += zp.shape.reduce(1, *) * byteWidth(zp.dtype) }
                if kQ.shape[2] > 0 { populated += 1 }
            } else {
                // Live prefix only — in-place mode allocates the full
                // capacity buffer but only `validLengths[i]` slots are
                // semantically populated. Report the logical size.
                let live = i < validLengths.count && preAllocCapacity != nil
                    ? validLengths[i] : e.keys.shape[2]
                let kSliceShape = liveSliceShape(e.keys.shape, time: live)
                let vSliceShape = liveSliceShape(e.values.shape, time: live)
                total += kSliceShape.reduce(1, *) * byteWidth(e.keys.dtype)
                total += vSliceShape.reduce(1, *) * byteWidth(e.values.dtype)
                if live > 0 { populated += 1 }
            }
        }
        return (total, populated)
    }

    /// Substitute a different value on the time axis (index 2) of a
    /// 4-D shape, leaving the other axes alone. Used in `totalBytes`
    /// to compute "what we'd be storing if we trimmed to `time`".
    private func liveSliceShape(_ shape: [Int], time: Int) -> [Int] {
        guard shape.count == 4 else { return shape }
        return [shape[0], shape[1], time, shape[3]]
    }

    /// Physical buffer size in bytes — counts the entire allocated buffer
    /// (capacity-sized in pre-alloc mode, just the populated rows in
    /// concat mode). Useful for verifying the in-place buffer DOESN'T grow
    /// across decode steps.
    public func physicalBytes(byteWidth: (DType) -> Int) -> Int {
        var total = 0
        for e in entries {
            if let kQ = e.keysQ, let vQ = e.valuesQ {
                total += kQ.shape.reduce(1, *) * byteWidth(kQ.dtype)
                total += vQ.shape.reduce(1, *) * byteWidth(vQ.dtype)
                if let sc = e.kScales { total += sc.shape.reduce(1, *) * byteWidth(sc.dtype) }
                if let zp = e.kZeros { total += zp.shape.reduce(1, *) * byteWidth(zp.dtype) }
                if let sc = e.vScales { total += sc.shape.reduce(1, *) * byteWidth(sc.dtype) }
                if let zp = e.vZeros { total += zp.shape.reduce(1, *) * byteWidth(zp.dtype) }
            } else {
                total += e.keys.shape.reduce(1, *) * byteWidth(e.keys.dtype)
                total += e.values.shape.reduce(1, *) * byteWidth(e.values.dtype)
            }
        }
        return total
    }

    /// Persist this cache's K/V state to disk for prefix caching.
    /// Layout: per-layer (k_shape_len u32 LE) (k_shape... u32 LE)
    ///          (v_shape_len) (v_shape) then raw fp32 bytes for each.
    /// We always serialise as fp32 for portability — the in-memory
    /// kvDtype downcast / KIVI quantisation is a runtime optimisation,
    /// not a storage format. Quantised caches are dequantised at save.
    public func saveToDisk(to url: URL) throws {
        var buf = Data()
        var nLayersOut = UInt32(entries.count).littleEndian
        withUnsafeBytes(of: &nLayersOut) { buf.append(contentsOf: $0) }
        for (i, e) in entries.enumerated() {
            // Dequantise on save for portability — readers don't need to
            // know about KIVI. In pre-alloc mode we save the live prefix
            // only (slicing through `keys(layer:asDType:)` does that for
            // us); the capacity-sized zero tail would round-trip but waste
            // disk bytes 10×+ for short prompts in a long-context model.
            let kSave: MLXArray
            let vSave: MLXArray
            if kivi != nil, e.keysQ != nil {
                kSave = keys(layer: i, asDType: .float32)!
                vSave = values(layer: i, asDType: .float32)!
            } else if preAllocCapacity != nil {
                kSave = keys(layer: i, asDType: .float32)!
                vSave = values(layer: i, asDType: .float32)!
            } else {
                kSave = e.keys
                vSave = e.values
            }
            try appendTensor(kSave, to: &buf)
            try appendTensor(vSave, to: &buf)
        }
        var clen = UInt32(currentLength).littleEndian
        withUnsafeBytes(of: &clen) { buf.append(contentsOf: $0) }
        try buf.write(to: url, options: .atomic)
    }

    /// Read a previously-saved prefix cache. Returns the populated cache
    /// (currentLength + entries), or throws on shape/format error.
    public static func load(from url: URL, nLayers expectedLayers: Int) throws -> KVCache {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        var off = 0
        func readU32() throws -> UInt32 {
            guard off + 4 <= data.count else {
                throw NSError(domain: "TinyGPTKVCache", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "prefix cache truncated"])
            }
            let v = data.subdata(in: off..<(off + 4)).withUnsafeBytes { $0.load(as: UInt32.self) }
            off += 4
            return UInt32(littleEndian: v)
        }
        let nL = Int(try readU32())
        guard nL == expectedLayers else {
            throw NSError(domain: "TinyGPTKVCache", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "prefix cache layer count \(nL) ≠ model \(expectedLayers)"])
        }
        let c = KVCache(nLayers: expectedLayers)
        // `validLengths` was pre-sized to nLayers zeros in init; overwrite
        // each slot rather than appending so the array stays length-N.
        // Loaded tensors come back at their LIVE (saved-prefix) size, so
        // `validLengths[i] == keys.shape[2]`. The caller can promote to
        // in-place via `migrateToPreAlloc(capacity:)`.
        for layer in 0..<nL {
            let k = try readTensor(data, off: &off)
            let v = try readTensor(data, off: &off)
            c.entries.append(Entry(keys: k, values: v))
            if layer < c.validLengths.count {
                c.validLengths[layer] = k.shape[2]
            } else {
                c.validLengths.append(k.shape[2])
            }
        }
        c.currentLength = Int(try readU32())
        return c
    }

    /// Promote a load-from-disk dense cache to the pre-allocated in-place
    /// layout. Called by the sample driver after `KVCache.load(...)` when
    /// the user is in pre-alloc mode — we'd otherwise stay on the concat
    /// path for the rest of the session even though the persistent-cache
    /// hit was supposed to be a fast path. Allocates one `capacity`-sized
    /// buffer per layer and copies the loaded prefix into rows
    /// `[0, validLengths[i])`.
    /// Drop the trailing `n` time slots from every layer's cache. Used by
    /// the persistent-cache re-prefill: after loading a saved cache for
    /// the prompt, we rewind by one token so the last prompt token can be
    /// re-fed through `forwardCached` to produce the first generation
    /// logit (the saved cache holds K, V but not the unembed-logits).
    ///
    /// In pre-alloc mode we just shrink `validLengths`; the underlying
    /// buffers stay capacity-sized so the next `appendInPlace` will write
    /// into the (now freed) slot without re-allocating. In concat mode we
    /// slice the buffers down to the shorter prefix.
    public func rewind(by n: Int) {
        guard n > 0 else { return }
        for layer in entries.indices {
            if preAllocCapacity != nil, layer < validLengths.count {
                validLengths[layer] = max(0, validLengths[layer] - n)
            } else {
                let k = entries[layer].keys
                let v = entries[layer].values
                let cur = k.shape[2]
                let next = max(0, cur - n)
                if next < cur {
                    entries[layer].keys = k[0..., 0..., 0..<next, 0...]
                    entries[layer].values = v[0..., 0..., 0..<next, 0...]
                }
            }
        }
        currentLength = max(0, currentLength - n)
    }

    public func migrateToPreAlloc(capacity: Int) {
        // The init-time mutability check (KIVI / StreamingLLM ⇒ no pre-
        // alloc) already gated this. If a caller wires it on a KIVI cache
        // we no-op so the on-disk format stays the contract.
        if kivi != nil || sink != nil || window != nil { return }
        self.preAllocCapacity = capacity
        mountForInPlace(capacity: capacity)
    }

    /// Internal: rebuild each layer's buffer at `capacity` with the
    /// loaded prefix copied in. We never expose this in the public init
    /// because the init-time exclusion against KIVI / StreamingLLM is
    /// the place where the safety check lives. After this runs,
    /// `entries[i].keys.shape[2] == capacity` and reads return the slice
    /// `[0..validLengths[i])` via the existing `keys(layer:asDType:)`
    /// path — exactly the same shape SDPA sees from a same-length concat
    /// cache.
    private func mountForInPlace(capacity: Int) {
        for i in entries.indices {
            let live = validLengths.indices.contains(i) ? validLengths[i] : entries[i].keys.shape[2]
            guard entries[i].keys.shape[2] != capacity else { continue }
            let k = entries[i].keys
            let v = entries[i].values
            let kDtype = k.dtype
            let vDtype = v.dtype
            let B = k.shape[0]
            let H = k.shape[1]
            let D = k.shape[3]
            let Hv = v.shape[1]
            let Dv = v.shape[3]
            let newK = MLXArray.zeros([B, H, capacity, D]).asType(kDtype)
            let newV = MLXArray.zeros([B, Hv, capacity, Dv]).asType(vDtype)
            if live > 0 {
                newK[0..., 0..., 0..<live, 0...] = k
                newV[0..., 0..., 0..<live, 0...] = v
            }
            entries[i].keys = newK
            entries[i].values = newV
        }
    }

    /// Append a tensor as (rank u32, shape... u32, fp32 bytes) into `buf`.
    private func appendTensor(_ a: MLXArray, to buf: inout Data) throws {
        var rank = UInt32(a.shape.count).littleEndian
        withUnsafeBytes(of: &rank) { buf.append(contentsOf: $0) }
        for s in a.shape {
            var v = UInt32(s).littleEndian
            withUnsafeBytes(of: &v) { buf.append(contentsOf: $0) }
        }
        let f = a.asType(.float32).asArray(Float.self)
        f.withUnsafeBufferPointer { buf.append(Data(buffer: $0)) }
    }

    private static func readTensor(_ data: Data, off: inout Int) throws -> MLXArray {
        let r = data.subdata(in: off..<(off + 4)).withUnsafeBytes { $0.load(as: UInt32.self) }
        off += 4
        let rank = Int(UInt32(littleEndian: r))
        var shape: [Int] = []
        shape.reserveCapacity(rank)
        for _ in 0..<rank {
            let s = data.subdata(in: off..<(off + 4)).withUnsafeBytes { $0.load(as: UInt32.self) }
            off += 4
            shape.append(Int(UInt32(littleEndian: s)))
        }
        let n = shape.reduce(1, *)
        let bytes = n * MemoryLayout<Float>.size
        let floats = data.subdata(in: off..<(off + bytes)).withUnsafeBytes { ptr -> [Float] in
            Array(UnsafeBufferPointer(
                start: ptr.baseAddress?.assumingMemoryBound(to: Float.self),
                count: n))
        }
        off += bytes
        return MLXArray(floats, shape)
    }
}

/// KV-cached attention forward. Used by `TinyGPTModel.forwardWithCache`.
/// Returns the new keys + values that the caller should append to the
/// cache so the next call can re-use them.
extension CausalSelfAttention {
    public func forwardCached(_ x: MLXArray, cache: KVCache, layer: Int) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        // Project Q/K/V from x; reshape to [B, T, H, D] → transpose to [B, H, T, D].
        // GQA correctness: K/V projections produce nKvHeads * headDim (set
        // up in TransformerBlock.swift line 68 as `kvDim = nKvHeads * headDim`),
        // so they MUST reshape to `nKvHeads` heads. Using nHeads here would
        // either crash on a shape mismatch (GQA models) or silently break
        // attention (if nKvHeads*headDim happened to be reshape-compatible
        // with [nHeads, headDim/nHeads*nKvHeads], which doesn't generally
        // hold). The HF variant in KVCacheHF.swift gets this right; the
        // from-scratch path silently followed the standard-MHA shape for
        // months because no from-scratch preset enables GQA today. The
        // fix is forward-compatible — non-GQA presets have nKvHeads == nHeads
        // and the behaviour is unchanged.
        let q = qProj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        let kNew = kProj(x).reshaped([B, T, nKvHeads, headDim]).transposed(0, 2, 1, 3)
        let vNew = vProj(x).reshaped([B, T, nKvHeads, headDim]).transposed(0, 2, 1, 3)

        // Push into the cache — that path handles downcast-on-store (KV
        // quantisation) and sink-window pruning (StreamingLLM). Then read
        // back, upcast to q.dtype so SDPA sees consistent precision.
        cache.append(layer: layer, keys: kNew, values: vNew)
        let kFull = cache.keys(layer: layer, asDType: q.dtype)!
        let vFull = cache.values(layer: layer, asDType: q.dtype)!

        // Attention. For the prefill case (T > 1), we need causal masking
        // among the new tokens. For the per-token decode case (T == 1),
        // the new token attends to all past + itself with no masking
        // (single position is trivially valid). MLX-Fast's `.causal` mask
        // works correctly because it masks j > i within the query range
        // (which is 1 row when T_q == 1, so the mask is empty).
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: kFull, values: vFull,
            scale: scale,
            mask: T == kFull.shape[2] ? .causal : .none
        )
        let merged = out.transposed(0, 2, 1, 3).reshaped([B, T, nHeads * headDim])
        return oProj(merged)
    }
}

extension TransformerBlock {
    public func forwardCached(_ x: MLXArray, cache: KVCache, layer: Int) -> MLXArray {
        var x = x
        x = x + attn.forwardCached(ln1(x), cache: cache, layer: layer)
        if let moe = moe {
            x = x + moe(ln2(x))
        } else if let mlp = mlp {
            x = x + mlp(ln2(x))
        }
        return x
    }
}

extension TinyGPTModel {
    /// KV-cached forward pass. On the first call (when `cache` is empty),
    /// processes the full prompt and populates the cache. On subsequent
    /// calls, processes only the new token(s) — typically `idx` is
    /// `[B, 1]` for streaming generation.
    ///
    /// **YOCO**: when `config.useYOCO`, the cache only stores K, V for the
    /// FIRST HALF of layers (0..=anchorIdx). Second-half layers don't
    /// touch the cache at all — they cross-attend onto the anchor's
    /// already-cached K, V. That's the long-context memory saving:
    /// `nLayers × KVbytes(T)` → `nLayers/2 × KVbytes(T)`.
    ///
    /// Returns logits of shape `[B, T_new, vocab_size]`.
    public func forwardCached(_ idx: MLXArray, cache: KVCache) -> MLXArray {
        let T = idx.shape[1]
        let basePos = cache.currentLength
        precondition(basePos + T <= config.contextLength,
                     "KV cache + new tokens (\(basePos + T)) exceeds context \(config.contextLength)")
        let positions = MLXArray((0..<T).map { Int32($0 + basePos) })
        let posEmb = positionEmbedding(positions).expandedDimensions(axis: 0)
        var tokEmb = tokenEmbedding(idx)
        if let en = embedNorm { tokEmb = en(tokEmb) }
        var x = tokEmb + posEmb
        if config.useYOCO {
            // Only the FIRST HALF of layers grows the cache. The anchor's
            // post-RoPE K, V come back via `cache.entries[anchorIdx]`
            // after the anchor block runs — second-half blocks read it
            // back upcast to q.dtype for cross-attention.
            let anchorIdx = max(0, (blocks.count / 2) - 1)
            for (i, block) in blocks.enumerated() {
                if i <= anchorIdx {
                    x = block.forwardCached(x, cache: cache, layer: i)
                } else {
                    // Read the anchor's cached K, V. The cache stores them
                    // POST-RoPE-rotation at their absolute positions, so
                    // cross-attention only needs to rotate Q at the new
                    // tokens' absolute positions (basePos..basePos+T-1).
                    let k = cache.keys(layer: anchorIdx, asDType: x.dtype)!
                    let v = cache.values(layer: anchorIdx, asDType: x.dtype)!
                    x = block.callWithExternalKV(x, k: k, v: v, posOffset: basePos)
                }
            }
        } else {
            for (i, block) in blocks.enumerated() {
                x = block.forwardCached(x, cache: cache, layer: i)
            }
        }
        cache.currentLength = basePos + T
        x = lnFinal(x)
        return tokenEmbedding.asLinear(x)
    }
}
