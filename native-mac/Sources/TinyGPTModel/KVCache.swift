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
        sink: Int? = nil, window: Int? = nil
    ) {
        self.nLayers = nLayers
        self.entries = []
        self.entries.reserveCapacity(nLayers)
        self.kvDtype = kvDtype
        self.kivi = kivi
        self.sink = sink
        self.window = window
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
        if entries.count <= layer {
            while entries.count <= layer {
                entries.append(Entry(keys: kIn, values: vIn))
            }
        } else {
            entries[layer].keys = concatenated([entries[layer].keys, kIn], axis: 2)
            entries[layer].values = concatenated([entries[layer].values, vIn], axis: 2)
        }
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

    /// Read-back with on-the-fly dtype upcast. Used by the attention
    /// extensions so SDPA always sees Q/K/V in the same dtype while the
    /// stored cache stays at `kvDtype` or KIVI-quantised (the memory-
    /// saving formats).
    public func keys(layer: Int, asDType dt: DType) -> MLXArray? {
        guard entries.indices.contains(layer) else { return nil }
        let e = entries[layer]
        if let cfg = kivi, let q = e.keysQ, let sc = e.kScales, let zp = e.kZeros {
            return Self.dequantiseK(q: q, scale: sc, zero: zp, qMin: cfg.qMin, asDType: dt)
        }
        let k = e.keys
        return k.dtype == dt ? k : k.asType(dt)
    }
    public func values(layer: Int, asDType dt: DType) -> MLXArray? {
        guard entries.indices.contains(layer) else { return nil }
        let e = entries[layer]
        if let cfg = kivi, let q = e.valuesQ, let sc = e.vScales, let zp = e.vZeros {
            return Self.dequantiseV(q: q, scale: sc, zero: zp, qMin: cfg.qMin, asDType: dt)
        }
        let v = e.values
        return v.dtype == dt ? v : v.asType(dt)
    }

    /// Total cached-K/V bytes across all layers. Counts the active storage
    /// path only — dense (`keys` + `values`) when KIVI is off, or
    /// (`keysQ` + `valuesQ` + scales + zeros) when KIVI is on. Reported in
    /// `tinygpt sample`'s footer for the memory-tradeoff smoke tests.
    public func totalBytes(byteWidth: (DType) -> Int) -> (bytes: Int, populated: Int) {
        var total = 0
        var populated = 0
        for e in entries {
            if let kQ = e.keysQ, let vQ = e.valuesQ {
                total += kQ.shape.reduce(1, *) * byteWidth(kQ.dtype)
                total += vQ.shape.reduce(1, *) * byteWidth(vQ.dtype)
                if let sc = e.kScales { total += sc.shape.reduce(1, *) * byteWidth(sc.dtype) }
                if let zp = e.kZeros { total += zp.shape.reduce(1, *) * byteWidth(zp.dtype) }
                if let sc = e.vScales { total += sc.shape.reduce(1, *) * byteWidth(sc.dtype) }
                if let zp = e.vZeros { total += zp.shape.reduce(1, *) * byteWidth(zp.dtype) }
                if kQ.shape[2] > 0 { populated += 1 }
            } else {
                total += e.keys.shape.reduce(1, *) * byteWidth(e.keys.dtype)
                total += e.values.shape.reduce(1, *) * byteWidth(e.values.dtype)
                if e.keys.shape[2] > 0 { populated += 1 }
            }
        }
        return (total, populated)
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
            // know about KIVI.
            let kSave: MLXArray
            let vSave: MLXArray
            if kivi != nil, e.keysQ != nil {
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
        for _ in 0..<nL {
            let k = try readTensor(data, off: &off)
            let v = try readTensor(data, off: &off)
            c.entries.append(Entry(keys: k, values: v))
        }
        c.currentLength = Int(try readU32())
        return c
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
        // Project Q/K/V from x; reshape to [B, T, H, D] → transpose to [B, H, T, D]
        let q = qProj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        let kNew = kProj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        let vNew = vProj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)

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
