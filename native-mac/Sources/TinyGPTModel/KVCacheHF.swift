import Foundation
import MLX
import MLXFast
import MLXNN

/// KV-cached forward pass for HF-style models. Mirrors the existing
/// `forwardCached` extensions in `KVCache.swift` (which target the
/// from-scratch TinyGPTModel + plain CausalSelfAttention with MHA + no
/// RoPE), but adapted for two differences that HF Llama-family models
/// require:
///
///   1. **GQA** — K/V projections produce `nKvHeads * headDim` outputs,
///      not `nHeads * headDim`. Reshape must use `nKvHeads` or MLX will
///      reject the shape mismatch.
///
///   2. **RoPE with a position offset** — under streaming decode, the
///      new token's RoPE rotation angle depends on its absolute position
///      in the sequence (`basePos`, the length already in the cache),
///      not 0. `MLXFast.RoPE` takes an `offset:` parameter for this.
///
/// Without the offset, every generated token would rotate with the same
/// angle as position 0, breaking the attention geometry and producing
/// garbage after the first cached step.
extension CausalSelfAttention {
    /// HF variant: GQA-correct K/V reshape + RoPE with position offset.
    /// `basePos` is the count of tokens already in the cache at this layer
    /// (same value for every layer in a single decode step).
    public func forwardCachedHF(_ x: MLXArray, cache: KVCache, layer: Int,
                                 basePos: Int) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        // Q uses nHeads; K/V use nKvHeads (= nHeads in standard MHA, less
        // for GQA models like Llama-3, Mistral, SmolLM2).
        var q = qProj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        var kNew = kProj(x).reshaped([B, T, nKvHeads, headDim]).transposed(0, 2, 1, 3)
        let vNew = vProj(x).reshaped([B, T, nKvHeads, headDim]).transposed(0, 2, 1, 3)

        // RoPE with the correct absolute-position offset. Q and K must
        // both be rotated; V is left alone. `offset` shifts the position
        // index forward so a 1-token decode at cache length N rotates as
        // position N, not position 0.
        if useRoPE {
            q = MLXFast.RoPE(q, dimensions: headDim, traditional: false,
                              base: ropeBase, scale: 1.0, offset: basePos)
            kNew = MLXFast.RoPE(kNew, dimensions: headDim, traditional: false,
                                 base: ropeBase, scale: 1.0, offset: basePos)
        }

        // Append the new K/V (post-RoPE, so rotation is paid once per
        // position). The cache encapsulates downcast-on-store (quantised
        // KV) and sink-window pruning (StreamingLLM); we read back upcast
        // to q.dtype to keep SDPA on a consistent precision.
        cache.append(layer: layer, keys: kNew, values: vNew)
        let kFull = cache.keys(layer: layer, asDType: q.dtype)!
        let vFull = cache.values(layer: layer, asDType: q.dtype)!

        // SDPA. When T == 1 (per-token decode) the single query attends
        // to every cached key — no mask needed because there's no future
        // to mask. When T > 1 (prefill) we want causal masking among the
        // T new tokens. Match KVCache.swift's existing convention.
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: kFull, values: vFull,
            scale: scale,
            mask: T == kFull.shape[2] ? .causal : .none
        )
        let merged = out.transposed(0, 2, 1, 3).reshaped([B, T, nHeads * headDim])
        return oProj(merged)
    }
}

extension TransformerBlockHF {
    public func forwardCached(_ x: MLXArray, cache: KVCache, layer: Int,
                               basePos: Int) -> MLXArray {
        var x = x
        x = x + attn.forwardCachedHF(ln1(x), cache: cache, layer: layer, basePos: basePos)
        x = x + mlp(ln2(x))
        return x
    }
}

extension TinyGPTModelHF {
    /// KV-cached forward pass for HF models. First call (empty cache)
    /// processes the full prompt and populates K/V at every layer; later
    /// calls usually pass `[B, 1]` for streaming decode.
    ///
    /// **YOCO**: when `config.useYOCO`, only the FIRST HALF of layers
    /// grows the cache. Second-half layers cross-attend onto the
    /// anchor's K, V (read back from `cache.entries[anchorIdx]`) — they
    /// allocate zero cache and skip K, V projection entirely. Halves
    /// KV memory at long-context decode.
    ///
    /// Returns logits of shape `[B, T_new, vocab_size]`.
    public func forwardCached(_ idx: MLXArray, cache: KVCache) -> MLXArray {
        let T = idx.shape[1]
        let basePos = cache.currentLength
        precondition(basePos + T <= config.contextLength,
                     "KV cache + new tokens (\(basePos + T)) exceeds context \(config.contextLength)")
        var x = tokenEmbedding(idx)
        if config.useYOCO {
            let anchorIdx = max(0, (blocks.count / 2) - 1)
            for (i, block) in blocks.enumerated() {
                if i <= anchorIdx {
                    x = block.forwardCached(x, cache: cache, layer: i, basePos: basePos)
                } else {
                    // K, V are read back from the anchor's cache slot
                    // post-RoPE (the rotation is paid once per position
                    // at the anchor; Q gets its own rotation inside
                    // CrossAttention at `basePos`).
                    let k = cache.keys(layer: anchorIdx, asDType: x.dtype)!
                    let v = cache.values(layer: anchorIdx, asDType: x.dtype)!
                    x = block.callWithExternalKV(x, k: k, v: v, posOffset: basePos)
                }
            }
        } else {
            for (i, block) in blocks.enumerated() {
                x = block.forwardCached(x, cache: cache, layer: i, basePos: basePos)
            }
        }
        cache.currentLength = basePos + T
        x = lnFinal(x)
        if let head = lmHead {
            return head(x)
        }
        return tokenEmbedding.asLinear(x)
    }
}
