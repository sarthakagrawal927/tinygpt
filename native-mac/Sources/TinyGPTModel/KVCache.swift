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
public final class KVCache {
    public struct Entry {
        public var keys: MLXArray   // [B, H, T, D]
        public var values: MLXArray // [B, H, T, D]
    }

    public var entries: [Entry]
    public let nLayers: Int
    public var currentLength: Int = 0

    /// Storage dtype for cached K/V. `nil` (default) = match whatever the
    /// attention computes natively. Set to `.float16` to halve KV memory
    /// when running an fp32 model — attention output is dequantised on
    /// read, so accuracy stays within fp16 noise.
    ///
    /// Why a dtype cast and not int8: MLX-Swift's int-quantised KV path
    /// (per-row scale + zero-point) doesn't compose with `MLXFast.SDPA`
    /// yet. fp16 storage is the well-supported intermediate step; int8
    /// is a follow-up when SDPA gains a quantised-K path.
    public let kvDtype: DType?

    /// StreamingLLM (Xiao et al., 2024) — keep the FIRST `sink` tokens
    /// always, and the LAST `window` tokens, dropping everything in
    /// between when the cache exceeds `sink + window`. Sliding-window
    /// attention with the leading anchor that makes the model survive
    /// arbitrarily-long generations. `nil` (default) = no pruning,
    /// pure causal accumulation.
    public let sink: Int?
    public let window: Int?

    public init(nLayers: Int, kvDtype: DType? = nil, sink: Int? = nil, window: Int? = nil) {
        self.nLayers = nLayers
        self.entries = []
        self.entries.reserveCapacity(nLayers)
        self.kvDtype = kvDtype
        self.sink = sink
        self.window = window
    }

    public func append(layer: Int, keys: MLXArray, values: MLXArray) {
        // Downcast on store (when requested) — cheap and saves bandwidth on
        // every future SDPA read against this entry.
        let kIn = kvDtype.map { keys.asType($0) } ?? keys
        let vIn = kvDtype.map { values.asType($0) } ?? values
        if entries.count <= layer {
            while entries.count <= layer {
                entries.append(Entry(keys: kIn, values: vIn))
            }
        } else {
            // Subsequent steps — concatenate along the time axis (axis=2).
            entries[layer].keys = concatenated([entries[layer].keys, kIn], axis: 2)
            entries[layer].values = concatenated([entries[layer].values, vIn], axis: 2)
        }
        // StreamingLLM pruning: drop the middle of the cache once it
        // exceeds sink + window. Run AFTER concatenation so the just-
        // -written tail is the part we keep. Per-layer because each
        // layer's cache grows in lockstep.
        if let s = sink, let w = window {
            let len = entries[layer].keys.shape[2]
            if len > s + w {
                let dropStart = s
                let dropEnd = len - w
                let kHead = entries[layer].keys[0..., 0..., 0..<dropStart, 0...]
                let kTail = entries[layer].keys[0..., 0..., dropEnd..<len, 0...]
                let vHead = entries[layer].values[0..., 0..., 0..<dropStart, 0...]
                let vTail = entries[layer].values[0..., 0..., dropEnd..<len, 0...]
                entries[layer].keys = concatenated([kHead, kTail], axis: 2)
                entries[layer].values = concatenated([vHead, vTail], axis: 2)
            }
        }
    }

    public func keys(layer: Int) -> MLXArray? { entries.indices.contains(layer) ? entries[layer].keys : nil }
    public func values(layer: Int) -> MLXArray? { entries.indices.contains(layer) ? entries[layer].values : nil }

    /// Read-back with on-the-fly dtype upcast. Used by the attention
    /// extensions so SDPA always sees Q/K/V in the same dtype while the
    /// stored cache stays at `kvDtype` (the memory-saving format).
    public func keys(layer: Int, asDType dt: DType) -> MLXArray? {
        guard let k = keys(layer: layer) else { return nil }
        return k.dtype == dt ? k : k.asType(dt)
    }
    public func values(layer: Int, asDType dt: DType) -> MLXArray? {
        guard let v = values(layer: layer) else { return nil }
        return v.dtype == dt ? v : v.asType(dt)
    }

    /// Persist this cache's K/V state to disk for prefix caching.
    /// Layout: per-layer (k_shape_len u32 LE) (k_shape... u32 LE)
    ///          (v_shape_len) (v_shape) then raw fp32 bytes for each.
    /// We always serialise as fp32 for portability — the in-memory
    /// kvDtype downcast is a runtime optimisation, not a storage format.
    public func saveToDisk(to url: URL) throws {
        var buf = Data()
        var nLayersOut = UInt32(entries.count).littleEndian
        withUnsafeBytes(of: &nLayersOut) { buf.append(contentsOf: $0) }
        for e in entries {
            try appendTensor(e.keys, to: &buf)
            try appendTensor(e.values, to: &buf)
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
        var x = tokenEmbedding(idx) + posEmb
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
