import Foundation
import MLX
import MLXNN

// AUDIT FLAG: Pruning — unstructured + structured-head are FLAGGED.
//   Structured layer pruning is KEEP (real wallclock win).
//
// Tested: 50% unstructured prune on Shakespeare gallery — loss 1.27 →
//   1.32, sample stays coherent. 4/8 head zero-out per layer — loss
//   1.27 → 2.81 (degrades). 2/12 layer drop → 9.6M → 8.0M params,
//   coherent.
// Saw: unstructured pruning has NO wallclock benefit (Metal has no
//   sparse matmul). Head pruning is shape-preserving (zero-out only);
//   no actual memory or wallclock savings. ONLY structured-layer
//   pruning actually changes topology.
// When unstructured would help: post-gzip distribution size (-38%).
// When head zero-out would help: as a precursor to physical-removal
//   (queued ~200 LOC follow-up).

/// Pruning utilities — both unstructured (magnitude masks on individual
/// weights) and structured (drop whole heads or whole layers).
///
/// Two distinct payoff profiles:
///
///   - **Unstructured**: zero out the smallest-magnitude weights. Mask is
///     a 0/1 per weight. On Metal, sparse matmul is NOT available, so
///     wallclock at inference time is UNCHANGED — the win is at storage
///     time (RLE-compressed mask + zeroed weights compress much better)
///     and as a regulariser for downstream fine-tuning.
///   - **Structured**: drop entire attention heads or entire transformer
///     layers. The result is a SMALLER dense model — same forward path,
///     just fewer FLOPs and fewer parameters. Real wallclock + memory
///     win, because the inner loop's matmul shapes shrink.
///
/// File-level utilities that operate on `[Float]` weight buffers — the
/// CLI handlers (PruneUnstructured / PruneStructured) drive these.
public enum Pruning {

    // MARK: - Unstructured (magnitude) pruning

    /// Compute a 0/1 mask that zeros out the `sparsity` fraction of
    /// weights with the smallest absolute value. Returns one byte per
    /// weight (0 = pruned, 1 = kept). `sparsity == 0` → all-ones; 1 →
    /// all-zeros (don't actually do that, but we don't crash).
    ///
    /// Global magnitude pruning is the standard recipe (Han et al.,
    /// 2015; Frankle & Carbin, 2019). Per-tensor pruning would be
    /// cleaner but biases the budget against larger tensors that
    /// have proportionally smaller weights.
    public static func magnitudeMask(_ weights: [Float], sparsity: Float) -> [UInt8] {
        let n = weights.count
        guard n > 0 else { return [] }
        let s = max(0, min(1, sparsity))
        if s <= 0 { return [UInt8](repeating: 1, count: n) }
        if s >= 1 { return [UInt8](repeating: 0, count: n) }
        // Threshold = `s`-quantile of |w|. We use the standard
        // "sort + index" approach because it's simple and correct;
        // a true streaming quantile would be faster but this only
        // runs once per file.
        var abs_w = [Float](repeating: 0, count: n)
        for i in 0..<n { abs_w[i] = abs(weights[i]) }
        let k = Int(Float(n) * s)
        let kClamped = max(0, min(n - 1, k))
        let sorted = abs_w.sorted()
        let threshold = sorted[kClamped]
        var mask = [UInt8](repeating: 0, count: n)
        for i in 0..<n {
            mask[i] = abs(weights[i]) > threshold ? 1 : 0
        }
        return mask
    }

    /// Apply a 0/1 mask in-place to the float weights. `weights.count`
    /// must match `mask.count`. After this, `weights[i] == 0` wherever
    /// `mask[i] == 0`.
    public static func applyMask(_ weights: inout [Float], mask: [UInt8]) {
        precondition(weights.count == mask.count,
                     "mask/weight length mismatch: \(mask.count) vs \(weights.count)")
        for i in 0..<weights.count {
            if mask[i] == 0 { weights[i] = 0 }
        }
    }

    /// Fraction of zeros in the float buffer — handy for reporting
    /// "we pruned 50%" without storing the mask. NaN/inf are not
    /// counted as zeros.
    public static func sparsityOf(_ weights: [Float]) -> Float {
        guard !weights.isEmpty else { return 0 }
        var zeros = 0
        for w in weights { if w == 0 { zeros += 1 } }
        return Float(zeros) / Float(weights.count)
    }

    // MARK: - Mask compression for the file header

    /// Compact encoding for a 0/1 mask. Header byte selects scheme;
    /// the payload follows.
    ///
    ///   - `0x00` (bit-packed): 1 bit per element, little-endian within
    ///     each byte. Best for fine-grained ~50% sparsity (predictable
    ///     1/8 compression of the raw mask).
    ///   - `0x01` (RLE): `[value byte] [varint run length] ...`. Wins
    ///     at high sparsity (>= 80%) where runs are long, loses to
    ///     bit-packing at low sparsity.
    ///
    /// `encodeRLE` is misnamed for historical reasons — it picks
    /// whichever scheme is smaller on the input.
    public static func encodeRLE(_ mask: [UInt8]) -> Data {
        let bp = bitPack(mask)
        let rle = trueRLE(mask)
        if bp.count <= rle.count {
            var out = Data([0x00])
            // Encode the bit-count up front so the decoder doesn't
            // need to pre-know the mask length (it's also derivable
            // from the parent tensor, but explicit is cheap).
            appendVarint(&out, UInt(mask.count))
            out.append(bp)
            return out
        } else {
            var out = Data([0x01])
            appendVarint(&out, UInt(mask.count))
            out.append(rle)
            return out
        }
    }

    /// Decode whichever scheme the header byte names. Returns the
    /// reconstructed 0/1 mask. The expectedLength argument is
    /// retained for backwards compat with the no-prefix RLE format
    /// (just in case an old file lands here).
    public static func decodeRLE(_ data: Data, expectedLength: Int) -> [UInt8] {
        guard !data.isEmpty else { return [] }
        let kind = data[data.startIndex]
        if kind == 0x00 {
            let (len, advance) = readVarint(data, at: data.startIndex + 1)
            let payload = data.subdata(in: (data.startIndex + 1 + advance)..<data.endIndex)
            return bitUnpack(payload, count: Int(len))
        } else if kind == 0x01 {
            let (_, advance) = readVarint(data, at: data.startIndex + 1)
            let payload = data.subdata(in: (data.startIndex + 1 + advance)..<data.endIndex)
            return decodeTrueRLE(payload)
        } else {
            // No header byte — assume legacy RLE.
            return decodeTrueRLE(data)
        }
    }

    private static func bitPack(_ mask: [UInt8]) -> Data {
        let n = mask.count
        let nBytes = (n + 7) / 8
        var out = Data(repeating: 0, count: nBytes)
        for i in 0..<n where mask[i] != 0 {
            out[i / 8] |= UInt8(1 << (i % 8))
        }
        return out
    }

    private static func bitUnpack(_ data: Data, count: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: count)
        for i in 0..<count {
            let byteIdx = data.startIndex + i / 8
            if byteIdx >= data.endIndex { break }
            let bit = (data[byteIdx] >> (i % 8)) & 1
            out[i] = bit
        }
        return out
    }

    private static func trueRLE(_ mask: [UInt8]) -> Data {
        var out = Data()
        var i = 0
        while i < mask.count {
            let v = mask[i]
            var j = i
            while j < mask.count && mask[j] == v { j += 1 }
            out.append(v == 0 ? 0 : 1)
            appendVarint(&out, UInt(j - i))
            i = j
        }
        return out
    }

    private static func decodeTrueRLE(_ data: Data) -> [UInt8] {
        var out = [UInt8]()
        var i = data.startIndex
        while i < data.endIndex {
            let v = data[i]; i += 1
            let (run, advance) = readVarint(data, at: i)
            i += advance
            for _ in 0..<run { out.append(v == 0 ? 0 : 1) }
        }
        return out
    }

    private static func appendVarint(_ data: inout Data, _ value: UInt) {
        var v = value
        while v >= 0x80 {
            data.append(UInt8((v & 0x7F) | 0x80))
            v >>= 7
        }
        data.append(UInt8(v))
    }

    private static func readVarint(_ data: Data, at offset: Int) -> (UInt, Int) {
        var v: UInt = 0
        var shift: UInt = 0
        var idx = offset
        while idx < data.endIndex {
            let byte = data[idx]; idx += 1
            v |= UInt(byte & 0x7F) << shift
            if byte & 0x80 == 0 { return (v, idx - offset) }
            shift += 7
        }
        return (v, idx - offset)
    }

    // MARK: - Structured: head importance scoring

    /// Per-head importance score. The "right" answer (Michel et al.,
    /// 2019) is `‖∂L/∂h · h‖_2` (Taylor-expansion saliency) — but
    /// that needs gradients, which we don't have at file-pruning
    /// time. We approximate with a much cheaper alternative:
    ///
    ///     score(h) = ‖V_h‖_F + ‖O_h‖_F + ‖Q_h‖_F + ‖K_h‖_F
    ///
    /// i.e. the Frobenius norm of each head's contribution to the
    /// attention output. This catches heads whose projections have
    /// collapsed to (near-)zero — those are safe to drop. It's a
    /// coarser signal than gradient-based saliency, but it doesn't
    /// require running the model and trains exactly the same target
    /// (drop the heads whose deletion changes the output least).
    ///
    /// Inputs are the raw row-major fp32 weight buffers in PyTorch
    /// shape: `q/k/v_proj.weight ∈ [dModel_out, dModel_in]`,
    /// `o_proj.weight ∈ [dModel_out, dModel_in]`. The heads are
    /// stored CONTIGUOUSLY along the output axis of q/k/v and the
    /// input axis of o — same convention as the rest of the codebase.
    public static func headImportance(
        qProj: [Float], kProj: [Float], vProj: [Float], oProj: [Float],
        dModel: Int, nHeads: Int, nKvHeads: Int
    ) -> [Float] {
        let headDim = dModel / nHeads
        let kvHeadDim = headDim
        let qOut = nHeads * headDim
        let kvOut = nKvHeads * kvHeadDim
        var scores = [Float](repeating: 0, count: nHeads)
        // Q: [dModel, qOut] row-major. Head h's rows are [h*headDim ..< (h+1)*headDim).
        // Wait — the convention here is [out, in]. So Q maps in→out;
        // each output row is dModel_in floats long, and head h owns
        // output rows [h*headDim ..< (h+1)*headDim).
        for h in 0..<nHeads {
            let rStart = h * headDim
            let rEnd = rStart + headDim
            var s: Float = 0
            for r in rStart..<rEnd {
                let rowBase = r * dModel
                for c in 0..<dModel {
                    let v = qProj[rowBase + c]
                    s += v * v
                }
            }
            scores[h] += sqrtf(s)
        }
        // K, V: same shape, but only nKvHeads (GQA). Group queries
        // share a KV head, so we attribute that KV head's norm
        // uniformly to all its query heads.
        let groupSize = nHeads / max(1, nKvHeads)
        for kvh in 0..<nKvHeads {
            let rStart = kvh * kvHeadDim
            let rEnd = rStart + kvHeadDim
            var ks: Float = 0
            var vs: Float = 0
            for r in rStart..<rEnd {
                let rowBase = r * dModel
                for c in 0..<dModel {
                    let kv = kProj[rowBase + c]
                    let vv = vProj[rowBase + c]
                    ks += kv * kv
                    vs += vv * vv
                }
            }
            let kn = sqrtf(ks); let vn = sqrtf(vs)
            for q in 0..<groupSize {
                let qh = kvh * groupSize + q
                if qh < nHeads {
                    scores[qh] += kn + vn
                }
            }
        }
        // O: [dModel, qOut] (mapping attn output back to residual).
        // Head h owns INPUT columns [h*headDim ..< (h+1)*headDim).
        for h in 0..<nHeads {
            let cStart = h * headDim
            let cEnd = cStart + headDim
            var s: Float = 0
            for r in 0..<dModel {
                let rowBase = r * qOut
                for c in cStart..<cEnd {
                    let v = oProj[rowBase + c]
                    s += v * v
                }
            }
            scores[h] += sqrtf(s)
        }
        _ = kvOut
        return scores
    }

    /// Zero out the rows/columns of the given attention weights that
    /// correspond to heads in `headsToDrop`. Operates in place. Use
    /// when the structured-prune ATM (this implementation) chooses to
    /// keep tensor shapes unchanged — so the model still loads with
    /// the original `nHeads`, but the dropped heads contribute zero.
    ///
    /// This is the "Michel et al. zero-out" pattern: not as fast as
    /// physical head removal, but much simpler — and it preserves
    /// the existing `CausalSelfAttention` shapes (which assume
    /// `dModel % nHeads == 0` with both q,k,v,o all `dModel × dModel`).
    /// A future iteration that physically removes head columns would
    /// need an asymmetric-attention module with an explicit
    /// `attnInnerDim` field.
    public static func zeroHeadsInPlace(
        qProj: inout [Float], kProj: inout [Float],
        vProj: inout [Float], oProj: inout [Float],
        qBias: inout [Float]?, kBias: inout [Float]?,
        vBias: inout [Float]?, oBias: inout [Float]?,
        dModel: Int, nHeads: Int, nKvHeads: Int,
        headsToDrop: Set<Int>
    ) {
        let headDim = dModel / nHeads
        let qOut = nHeads * headDim
        let groupSize = nHeads / max(1, nKvHeads)
        // Map dropped query heads to KV heads — only zero a KV head
        // if EVERY query head in its group was dropped (otherwise the
        // remaining heads in the group still need that KV).
        var dropKVHeads: Set<Int> = []
        for kvh in 0..<nKvHeads {
            let group = (0..<groupSize).map { kvh * groupSize + $0 }
            let groupSet = Set(group)
            if groupSet.isSubset(of: headsToDrop) { dropKVHeads.insert(kvh) }
        }
        // Q: zero rows [h*headDim ..< (h+1)*headDim).
        for h in headsToDrop {
            let rStart = h * headDim
            let rEnd = rStart + headDim
            for r in rStart..<rEnd {
                let rowBase = r * dModel
                for c in 0..<dModel { qProj[rowBase + c] = 0 }
                if qBias != nil { qBias![r] = 0 }
            }
        }
        // K, V: zero rows per KV head we marked.
        for kvh in dropKVHeads {
            let rStart = kvh * headDim
            let rEnd = rStart + headDim
            for r in rStart..<rEnd {
                let rowBase = r * dModel
                for c in 0..<dModel {
                    kProj[rowBase + c] = 0
                    vProj[rowBase + c] = 0
                }
                if kBias != nil { kBias![r] = 0 }
                if vBias != nil { vBias![r] = 0 }
            }
        }
        // O: zero INPUT columns [h*headDim ..< (h+1)*headDim). oProj
        // is [dModel × qOut] row-major; for each row, we clear that
        // column range.
        for h in headsToDrop {
            let cStart = h * headDim
            let cEnd = cStart + headDim
            for r in 0..<dModel {
                let rowBase = r * qOut
                for c in cStart..<cEnd { oProj[rowBase + c] = 0 }
            }
        }
        // o_proj's bias is on the OUTPUT side (dModel) and shared
        // across all heads — we leave it alone. (Dropping heads
        // doesn't cancel the post-projection bias.)
    }

    // MARK: - Structured: layer importance scoring (block angular distance)

    /// Per-layer importance via "block angular distance" — the cosine
    /// distance between the layer's input residual and its output
    /// residual (Gromov et al., 2024, "The Unreasonable Ineffectiveness
    /// of the Deeper Layers"). A layer whose output is nearly
    /// IDENTICAL to its input (cosine ≈ 1, distance ≈ 0) is contributing
    /// little — safe to drop.
    ///
    /// Scoring is done by running the model on a calibration prompt
    /// and capturing the hidden state at each block boundary. The
    /// caller supplies the calibration corpus; we return one score
    /// per layer, lower = drop-me.
    ///
    /// Note: this lives in `Pruning` (the library) as the math, but
    /// the actual hidden-state capture has to happen in the CLI
    /// driver where we have an MLX model in hand. We expose the math
    /// here as pure float-array operations so it's testable.
    public static func angularDistance(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "angular-distance vector size mismatch")
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let denom = sqrtf(na) * sqrtf(nb)
        let cosSim = denom > 1e-12 ? dot / denom : 0
        // Clamp for acos numerical safety.
        let cs = max(-1, min(1, cosSim))
        // Angular distance is acos(cos_sim) / π — normalised to [0, 1].
        return acosf(cs) / .pi
    }
}
