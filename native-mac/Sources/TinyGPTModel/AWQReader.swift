import Foundation
import MLX
import TinyGPTIO

/// AWQ-quantised HF safetensors reader (Lin et al., 2023).
///
/// AWQ stores a Linear's weight as three tensors instead of one:
///
///     {name}.qweight   int32, shape [in // 8, out]
///                      8 packed int4 values per int32 along the inner axis
///     {name}.scales    fp16,  shape [in // group_size, out]
///                      per-output-channel, per-group dequant scale
///     {name}.qzeros    int32, shape [in // group_size, out // 8]
///                      8 packed int4 zero-points per int32
///
/// To recover the dense weight matrix `W[out, in]` (HF row-major):
///
///     for o in 0..<out:
///         for i in 0..<in:
///             qint32 = qweight[i // 8, o]
///             bit    = (i % 8) * 4
///             int4   = (qint32 >> bit) & 0xF
///             g      = i // group_size
///             scale  = scales[g, o]
///             zint32 = qzeros[g, o // 8]
///             zbit   = (o % 8) * 4
///             zero   = (zint32 >> zbit) & 0xF
///             W[o, i] = scale · (int4 − zero)
///
/// This reader unpacks the AWQ triples back to dense fp32, returning a
/// map of `{name}.weight → MLXArray[out, in]` that the standard
/// HFModelLoader can consume unchanged. The runtime memory benefit is
/// LOST (we expand 4-bit back to 32-bit at load time) — what we keep
/// is the ability to LOAD an AWQ HF release without a separate Python
/// dequant step. A packed-int4 matmul kernel that consumes the AWQ
/// layout directly is the follow-up that delivers the inference win.
///
/// AWQ has two main pack-order variants ("GEMM" and "GEMV"). This
/// reader implements GEMM (the default produced by `awq` / `vllm` /
/// `transformers` AWQ checkpoints). If a file uses GEMV (rare), the
/// dequantised values will be permuted — the user will see garbage on
/// sample and should re-quantise.
public enum AWQReader {

    /// Detect whether a `safetensors` tensor map contains AWQ-style
    /// triples. Trigger: any tensor name ending in `.qweight` with a
    /// sibling `.scales` and `.qzeros`. Returns the unique base names
    /// (e.g. `"model.layers.0.self_attn.q_proj"`) that have all three.
    public static func detectAwqBases(in names: any Collection<String>) -> [String] {
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

    /// Unpack one AWQ triple into a dense fp32 `[out, in]` weight.
    /// `qweight` is `[in/8, out]` int32 (8 nibbles per int32, low bits = lowest index).
    /// `scales`  is `[in/group, out]` fp16, group_size is inferred from shapes.
    /// `qzeros`  is `[in/group, out/8]` int32 (same packing as qweight, on the OUT axis).
    public static func dequantize(
        qweight: AWQTensor, scales: AWQTensor, qzeros: AWQTensor
    ) throws -> MLXArray {
        // Shapes:
        //   qweight = [in/8, out]
        //   scales  = [in/group, out]
        //   qzeros  = [in/group, out/8]
        let qwShape = qweight.shape; let scShape = scales.shape; let qzShape = qzeros.shape
        guard qwShape.count == 2, scShape.count == 2, qzShape.count == 2 else {
            throw AWQError.shapeMismatch("expected 2-D qweight/scales/qzeros, got \(qwShape) \(scShape) \(qzShape)")
        }
        let inOver8 = qwShape[0]
        let out = qwShape[1]
        let inFeatures = inOver8 * 8
        guard scShape[1] == out, qzShape[1] * 8 == out else {
            throw AWQError.shapeMismatch("scales/qzeros output dim mismatch — qweight out=\(out) scales[1]=\(scShape[1]) qzeros[1]=\(qzShape[1])")
        }
        let groups = scShape[0]
        guard inFeatures % groups == 0 else {
            throw AWQError.shapeMismatch("in_features (\(inFeatures)) not divisible by group count (\(groups))")
        }
        let groupSize = inFeatures / groups

        // Materialise the three packed tensors as Swift arrays.
        let qw: [Int32] = qweight.asInt32()
        let sc: [Float] = scales.asFloat32()
        let qz: [Int32] = qzeros.asInt32()

        // Output buffer in HF row-major [out, in].
        var W = [Float](repeating: 0, count: out * inFeatures)
        for o in 0..<out {
            // For each output channel, walk inputs in order. The naive
            // double-loop is O(out · in) — fine for the shapes that
            // ship in practice (Linear weights up to a few M elements).
            let zeroByteCol = o / 8
            let zeroBitOffset = (o % 8) * 4
            for i in 0..<inFeatures {
                let qwInt = qw[(i / 8) * out + o]
                let bitOffset = (i % 8) * 4
                let int4 = (qwInt >> Int32(bitOffset)) & 0xF
                let g = i / groupSize
                let scale = sc[g * out + o]
                let qzInt = qz[g * qzShape[1] + zeroByteCol]
                let zero4 = (qzInt >> Int32(zeroBitOffset)) & 0xF
                W[o * inFeatures + i] = scale * Float(int4 - zero4)
            }
        }
        return MLXArray(W, [out, inFeatures])
    }

    public enum AWQError: Error, CustomStringConvertible {
        case shapeMismatch(String)
        case missingSibling(String)
        public var description: String {
            switch self {
            case .shapeMismatch(let s): return "AWQ shape mismatch: \(s)"
            case .missingSibling(let s): return "AWQ missing sibling tensor: \(s)"
            }
        }
    }
}

/// Decoded packed tensor handed to AWQReader. Bridges the HF
/// safetensors reader's raw-byte view + the dtype-specific unpacking
/// the dequantiser needs.
public struct AWQTensor {
    public let shape: [Int]
    public let dtype: String   // "I32" or "F16"
    public let bytes: Data
    public init(shape: [Int], dtype: String, bytes: Data) {
        self.shape = shape; self.dtype = dtype; self.bytes = bytes
    }

    /// View as a flat Int32 array (for qweight / qzeros).
    public func asInt32() -> [Int32] {
        precondition(dtype == "I32", "AWQTensor.asInt32 called with dtype \(dtype)")
        let n = shape.reduce(1, *)
        return bytes.withUnsafeBytes { ptr -> [Int32] in
            Array(UnsafeBufferPointer(
                start: ptr.baseAddress?.assumingMemoryBound(to: Int32.self),
                count: n))
        }
    }

    /// View as a flat Float32 array (for scales — F16 → F32).
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
        default:
            preconditionFailure("AWQTensor.asFloat32 called with dtype \(dtype)")
        }
    }
}
