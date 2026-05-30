import Foundation
import MLX
import TinyGPTIO

/// GPTQ-quantised HF safetensors reader (Frantar et al., 2022).
///
/// GPTQ stores a Linear's weight as four tensors instead of one. The
/// canonical layout (`auto-gptq`, `optimum`, `gptq-for-llama`) packs
/// 8 int4 values per int32 along the IN axis of the weight matrix
/// — column-packed, unlike AWQ's GEMM-packed-along-rows variant:
///
///     {name}.qweight   int32, shape [in // 8, out]
///                      8 packed int4 values per int32, low nibble = lowest in-index
///     {name}.scales    fp16/bf16, shape [in // group_size, out]
///                      per-output-channel, per-group dequant scale
///     {name}.qzeros    int32, shape [in // group_size, out // 8]
///                      8 packed int4 zero-points per int32 along the OUT axis
///     {name}.g_idx     int32, shape [in]
///                      group index per in-feature — usually `floor(i/group_size)`
///                      but can be permuted by GPTQ's activation-order pass.
///
/// The dequant recipe (HF row-major `W[out, in]`):
///
///     for o in 0..<out:
///         for i in 0..<in:
///             qint32 = qweight[i // 8, o]
///             bit    = (i % 8) * 4
///             int4   = (qint32 >> bit) & 0xF
///             g      = g_idx[i]                              // may be permuted
///             scale  = scales[g, o]
///             zint32 = qzeros[g, o // 8]
///             zbit   = (o % 8) * 4
///             zero   = ((zint32 >> zbit) & 0xF) + 1          // GPTQ convention: zero +1
///             W[o, i] = scale · (int4 − zero)
///
/// The "+1 on zero" is the historical GPTQ quirk — `auto-gptq` v0.x
/// stores zero as `(int4_zero - 1)` so re-adding 1 recovers the
/// dequant offset. AWQ does NOT do this. (Newer `auto-gptq` v1.x with
/// `desc_act=False, sym=True` skips it; we look at the metadata when
/// present, but default to +1 because that's what the vast majority
/// of public GPTQ checkpoints ship with.)
///
/// **This reader expands packed int4 to dense fp32 at load time.** The
/// runtime memory benefit is LOST — what we keep is the ability to
/// LOAD a GPTQ HF release without a separate Python dequant step.
/// A packed-int4 matmul kernel that consumes the GPTQ layout directly
/// is the follow-up that delivers the inference win (same blocker as
/// AWQ — MLX-Swift's quantized-matmul story).
public enum GPTQReader {

    /// Detect whether a `safetensors` tensor map contains GPTQ-style
    /// triples (qweight + scales + qzeros). Returns the unique base
    /// names that have all three. Bases that ALSO have an `.g_idx`
    /// sibling are unpacked with permuted-group logic; bases without
    /// fall back to `floor(i/group_size)`.
    public static func detectGptqBases(in names: any Collection<String>) -> [String] {
        let asSet = Set(names)
        var bases: [String] = []
        for n in asSet where n.hasSuffix(".qweight") {
            let base = String(n.dropLast(".qweight".count))
            if asSet.contains(base + ".scales") && asSet.contains(base + ".qzeros") {
                bases.append(base)
            }
        }
        return bases
    }

    /// Unpack one GPTQ quartet into a dense fp32 `[out, in]` weight.
    /// `qweight` is `[in/8, out]` int32 (8 nibbles per int32, low bits = lowest in-index).
    /// `scales`  is `[in/group, out]` fp16/bf16, group_size inferred from shapes.
    /// `qzeros`  is `[in/group, out/8]` int32 (same packing as qweight, on the OUT axis).
    /// `gIdx`    is optional `[in]` int32. When nil, groups follow `floor(i/group_size)`.
    /// `zeroPlusOne` toggles the "stored zero = int4 - 1" GPTQ convention. Default true
    /// (matches the dominant `auto-gptq` v0.x export); set false for newer sym-quant exports.
    public static func dequantize(
        qweight: GPTQTensor, scales: GPTQTensor, qzeros: GPTQTensor,
        gIdx: GPTQTensor? = nil, zeroPlusOne: Bool = true
    ) throws -> MLXArray {
        let qwShape = qweight.shape
        let scShape = scales.shape
        let qzShape = qzeros.shape
        guard qwShape.count == 2, scShape.count == 2, qzShape.count == 2 else {
            throw GPTQError.shapeMismatch("expected 2-D qweight/scales/qzeros, got \(qwShape) \(scShape) \(qzShape)")
        }
        let inOver8 = qwShape[0]
        let out = qwShape[1]
        let inFeatures = inOver8 * 8
        guard scShape[1] == out, qzShape[1] * 8 == out else {
            throw GPTQError.shapeMismatch("scales/qzeros output dim mismatch — qweight out=\(out) scales[1]=\(scShape[1]) qzeros[1]=\(qzShape[1])")
        }
        let groups = scShape[0]
        guard inFeatures % groups == 0 else {
            throw GPTQError.shapeMismatch("in_features (\(inFeatures)) not divisible by group count (\(groups))")
        }
        let defaultGroupSize = inFeatures / groups

        // Group index per in-feature: either explicit g_idx or floor(i/group_size).
        let gMap: [Int32]
        if let g = gIdx {
            guard g.shape == [inFeatures] else {
                throw GPTQError.shapeMismatch("g_idx shape \(g.shape) does not match in_features \(inFeatures)")
            }
            gMap = g.asInt32()
        } else {
            gMap = (0..<inFeatures).map { Int32($0 / defaultGroupSize) }
        }

        let qw: [Int32] = qweight.asInt32()
        let sc: [Float] = scales.asFloat32()
        let qz: [Int32] = qzeros.asInt32()
        let zeroBias: Int32 = zeroPlusOne ? 1 : 0

        // Output buffer in HF row-major [out, in].
        var W = [Float](repeating: 0, count: out * inFeatures)
        for o in 0..<out {
            let zeroByteCol = o / 8
            let zeroBitOffset = (o % 8) * 4
            for i in 0..<inFeatures {
                let qwInt = qw[(i / 8) * out + o]
                let bitOffset = (i % 8) * 4
                let int4 = (qwInt >> Int32(bitOffset)) & 0xF
                let g = Int(gMap[i])
                let scale = sc[g * out + o]
                let qzInt = qz[g * qzShape[1] + zeroByteCol]
                let zero4 = ((qzInt >> Int32(zeroBitOffset)) & 0xF) + zeroBias
                W[o * inFeatures + i] = scale * Float(int4 - zero4)
            }
        }
        return MLXArray(W, [out, inFeatures])
    }

    public enum GPTQError: Error, CustomStringConvertible {
        case shapeMismatch(String)
        case missingSibling(String)
        public var description: String {
            switch self {
            case .shapeMismatch(let s): return "GPTQ shape mismatch: \(s)"
            case .missingSibling(let s): return "GPTQ missing sibling tensor: \(s)"
            }
        }
    }
}

/// Decoded packed tensor handed to GPTQReader. Bridges the HF
/// safetensors raw-byte view + the dtype-specific unpacking the
/// dequantiser needs. Mirrors AWQTensor but adds bf16 (some
/// recent GPTQ exports use bf16 scales).
public struct GPTQTensor {
    public let shape: [Int]
    public let dtype: String   // "I32" or "F16" or "BF16" or "F32"
    public let bytes: Data
    public init(shape: [Int], dtype: String, bytes: Data) {
        self.shape = shape; self.dtype = dtype; self.bytes = bytes
    }

    /// View as a flat Int32 array (for qweight / qzeros / g_idx).
    public func asInt32() -> [Int32] {
        precondition(dtype == "I32", "GPTQTensor.asInt32 called with dtype \(dtype)")
        let n = shape.reduce(1, *)
        return bytes.withUnsafeBytes { ptr -> [Int32] in
            Array(UnsafeBufferPointer(
                start: ptr.baseAddress?.assumingMemoryBound(to: Int32.self),
                count: n))
        }
    }

    /// View as a flat Float32 array (for scales — F16/BF16/F32 → F32).
    public func asFloat32() -> [Float] {
        let n = shape.reduce(1, *)
        switch dtype {
        case "F32":
            return bytes.withUnsafeBytes { ptr -> [Float] in
                Array(UnsafeBufferPointer(
                    start: ptr.baseAddress?.assumingMemoryBound(to: Float.self),
                    count: n))
            }
        case "F16":
            let halves = bytes.withUnsafeBytes { ptr -> [UInt16] in
                Array(UnsafeBufferPointer(
                    start: ptr.baseAddress?.assumingMemoryBound(to: UInt16.self),
                    count: n))
            }
            return halves.map { Float(Float16(bitPattern: $0)) }
        case "BF16":
            // bf16: upper 16 bits of an fp32. Shift left 16, reinterpret.
            let bf = bytes.withUnsafeBytes { ptr -> [UInt16] in
                Array(UnsafeBufferPointer(
                    start: ptr.baseAddress?.assumingMemoryBound(to: UInt16.self),
                    count: n))
            }
            var out = [Float](repeating: 0, count: n)
            for i in 0..<n {
                let bits = UInt32(bf[i]) << 16
                out[i] = Float(bitPattern: bits)
            }
            return out
        default:
            preconditionFailure("GPTQTensor.asFloat32 called with dtype \(dtype)")
        }
    }
}
