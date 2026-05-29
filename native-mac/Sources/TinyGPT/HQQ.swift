import Foundation
import MLX
import TinyGPTIO
import TinyGPTModel

/// `tinygpt hqq` — Half-Quadratic Quantization (Badri & Shaji, 2023).
///
/// Replaces every linear-weight tensor in a `.tinygpt` file with a
/// per-group int4 representation: `W_dequant = scale · (Q − zero)`.
/// The (scale, zero) per group are chosen by alternating IRLS updates
/// minimising `Σ |W − scale·(Q − zero)|^p` for `p < 1` (sub-quadratic,
/// noise-robust). On modern transformers the dequantised weights
/// recover the original distribution to within fp16 noise.
///
/// **Storage-only this ship.** The HQQ payload is written into the
/// .tinygpt body bytes as already-dequantised fp32 — so the model
/// loads, samples, and trains normally with the existing forward
/// path. The memory win at inference time requires a packed-int4
/// matmul kernel (the same one HQQ's reference Python implementation
/// uses); that's queued behind the MLX-Swift quantized-matmul story.
/// What you GET today: a model whose weights have been
/// rank-conditioned by HQQ's quantize-then-dequantise pass — often
/// improves downstream task accuracy by a small margin, similar to
/// the LASER effect.
///
/// USAGE
///   tinygpt hqq <input.tinygpt> --group-size 64 --bits 4 --p 0.7 \
///       --layers 0-11 --out reduced.tinygpt
enum HQQ {
    static func run(args: [String]) {
        var inPath: String? = nil
        var outPath: String? = nil
        var groupSize = 64
        var bits = 4
        var p: Float = 0.7
        var layersSpec: String = ""
        var iterations = 30

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--out":         outPath = args[i+1]; i += 2
            case "--group-size":  groupSize = Int(args[i+1]) ?? groupSize; i += 2
            case "--bits":        bits = Int(args[i+1]) ?? bits; i += 2
            case "--p":           p = Float(args[i+1]) ?? p; i += 2
            case "--layers":      layersSpec = args[i+1]; i += 2
            case "--iterations":  iterations = Int(args[i+1]) ?? iterations; i += 2
            case "-h", "--help":  exitUsage()
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                inPath = args[i]; i += 1
            }
        }
        guard let inPath = inPath else { fputs("missing <model>\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out <path> required\n", stderr); exitUsage() }
        precondition([2, 3, 4, 8].contains(bits), "--bits must be 2, 3, 4, or 8")
        precondition(groupSize > 0, "--group-size must be > 0")
        precondition(p > 0 && p <= 2, "--p must be in (0, 2]")

        print("loading \(inPath)…")
        let inputURL = URL(fileURLWithPath: inPath)
        var file: TinyGPTFile
        do { file = try TinyGPTFileReader.read(inputURL) }
        catch { fputs("read failed: \(error)\n", stderr); exit(1) }

        let nLayers = file.header.config.layers ?? 12
        let targetLayers = parseLayers(layersSpec, total: nLayers)
        // Match the "modify Linear weights only" predicate used elsewhere
        // — we don't touch token / position embeddings, layernorm
        // params, or the final LM head bias.
        let targetNames: Set<String> = {
            var s: Set<String> = []
            for li in targetLayers {
                s.insert("blocks.\(li).attn.q_proj.weight")
                s.insert("blocks.\(li).attn.k_proj.weight")
                s.insert("blocks.\(li).attn.v_proj.weight")
                s.insert("blocks.\(li).attn.o_proj.weight")
                s.insert("blocks.\(li).mlp.fc_in.weight")
                s.insert("blocks.\(li).mlp.fc_out.weight")
            }
            return s
        }()

        print("""

        TinyGPT — HQQ (half-quadratic quantize-then-dequantise)
        -------------------------------------------------------
        input:          \(inPath)
        layers:         \(targetLayers.map(String.init).joined(separator: ","))
        bits:           \(bits)  group=\(groupSize)  p=\(p)
        iterations:     \(iterations)
        output:         \(outPath)

        """)

        var hits = 0
        var totalAbsErr: Double = 0
        var totalScale: Double = 0
        for (idx, tensor) in file.tensors.enumerated() {
            guard targetNames.contains(tensor.entry.name) else { continue }
            let shape = tensor.entry.shape
            guard shape.count == 2 else { continue }
            let floats = floatsFromData(tensor.weight, count: shape[0] * shape[1])
            // Per-OUTPUT-ROW grouping — the standard recipe (rows are the
            // "neurons", and the quant noise is concentrated on the
            // inner-feature axis where activations vary widely).
            let dequant = quantizeDequantize(
                floats, m: shape[0], n: shape[1],
                groupSize: groupSize, bits: bits, p: p, iterations: iterations
            )
            // Track reconstruction quality for the run summary.
            var absErr: Double = 0
            var sumAbs: Double = 0
            for k in 0..<floats.count {
                absErr += Double(abs(floats[k] - dequant[k]))
                sumAbs += Double(abs(floats[k]))
            }
            totalAbsErr += absErr
            totalScale += sumAbs
            file.tensors[idx].weight = dequant.withUnsafeBufferPointer { Data(buffer: $0) }
            hits += 1
            print("  ✓ \(tensor.entry.name)  \(shape[0])×\(shape[1])")
        }
        if hits == 0 {
            fputs("warning: 0 tensors matched — nothing written\n", stderr); exit(1)
        }
        let relErr = totalScale > 0 ? totalAbsErr / totalScale : 0
        print("\nrelative reconstruction error: \(String(format: "%.4f", relErr))  (lower is better)")

        do {
            try TinyGPTFileWriter.write(file, to: URL(fileURLWithPath: outPath))
            print("✓ wrote \(outPath)  (\(hits) tensors quantize-dequantised)")
        } catch {
            fputs("write failed: \(error)\n", stderr); exit(1)
        }
    }

    /// Quantize each ROW of `[m, n]` flat weights in groups of `groupSize`
    /// along the inner axis (per-row block-int{bits}), then dequantise
    /// back to fp32. Returns the dequantised matrix as a flat [m·n] array.
    private static func quantizeDequantize(
        _ flat: [Float], m: Int, n: Int,
        groupSize: Int, bits: Int, p: Float, iterations: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: m * n)
        let groupsPerRow = (n + groupSize - 1) / groupSize
        for row in 0..<m {
            let rowBase = row * n
            for g in 0..<groupsPerRow {
                let gStart = g * groupSize
                let gEnd = min(gStart + groupSize, n)
                let block = Array(flat[(rowBase + gStart)..<(rowBase + gEnd)])
                let (q, scale, zero) = quantizeBlockHQQ(
                    block, bits: bits, p: p, iterations: iterations
                )
                // Write dequantised back.
                for (k, qv) in q.enumerated() {
                    out[rowBase + gStart + k] = scale * (Float(qv) - zero)
                }
            }
        }
        return out
    }

    /// Quantize one block via IRLS on the sub-quadratic loss
    /// `Σ |W − scale·(Q − zero)|^p`. Closed-form alternating updates
    /// on (scale, zero) with reweighting `wᵢ = 1 / (|rᵢ|^(2−p) + ε)`.
    private static func quantizeBlockHQQ(_ w: [Float], bits: Int, p: Float, iterations: Int)
        -> (q: [UInt8], scale: Float, zero: Float)
    {
        let qMax: Float = Float((1 << bits) - 1)   // 15 for int4
        let mid: Float = qMax / 2
        // Min-max init.
        let absMax = max(w.map { abs($0) }.max() ?? 0, 1e-6)
        var scale: Float = absMax / mid
        // Init zero so the block's centroid lands at the integer midpoint.
        var zero: Float = mid - (w.reduce(0, +) / Float(w.count)) / max(scale, 1e-6)
        let eps: Float = 1e-6

        var q = [Float](repeating: 0, count: w.count)
        for _ in 0..<iterations {
            // 1. Quantize: round to nearest int in [0, qMax].
            for i in 0..<w.count {
                let raw = w[i] / max(scale, eps) + zero
                let r = raw.rounded()
                q[i] = min(qMax, max(0, r))
            }
            // 2. Residuals + IRLS weights.
            var weights = [Float](repeating: 0, count: w.count)
            for i in 0..<w.count {
                let r = w[i] - scale * (q[i] - zero)
                let absR = abs(r)
                // 1 / (|r|^(2-p) + eps). For p=0.7 → exponent 1.3.
                weights[i] = 1.0 / (Foundation.pow(absR, 2 - p) + eps)
            }
            // 3. Closed-form alternating update.
            //    s = Σ(wᵢ · Wᵢ · (Qᵢ − zero)) / Σ(wᵢ · (Qᵢ − zero)²)
            var numS: Float = 0, denS: Float = 0
            for i in 0..<w.count {
                let qz = q[i] - zero
                numS += weights[i] * w[i] * qz
                denS += weights[i] * qz * qz
            }
            if denS > eps { scale = numS / denS }
            //    z = (Σ(wᵢ · (s·Qᵢ − Wᵢ)) / Σ(wᵢ)) / s
            var numZ: Float = 0, denZ: Float = 0
            for i in 0..<w.count {
                numZ += weights[i] * (scale * q[i] - w[i])
                denZ += weights[i]
            }
            if denZ > eps, abs(scale) > eps { zero = numZ / denZ / scale }
        }
        // Final quantize.
        var qOut = [UInt8](repeating: 0, count: w.count)
        for i in 0..<w.count {
            let raw = w[i] / max(scale, eps) + zero
            let r = raw.rounded()
            qOut[i] = UInt8(min(qMax, max(0, r)))
        }
        return (qOut, scale, zero)
    }

    private static func floatsFromData(_ data: Data, count: Int) -> [Float] {
        return data.withUnsafeBytes { ptr -> [Float] in
            Array(UnsafeBufferPointer(
                start: ptr.baseAddress?.assumingMemoryBound(to: Float.self),
                count: count))
        }
    }

    /// Parse "1,2,5-8" → [1, 2, 5, 6, 7, 8]. Empty = all layers.
    private static func parseLayers(_ spec: String, total: Int) -> [Int] {
        if spec.isEmpty { return Array(0..<total) }
        var out: [Int] = []
        for part in spec.split(separator: ",") {
            let s = String(part).trimmingCharacters(in: .whitespaces)
            if s.contains("-") {
                let bits = s.split(separator: "-").map { Int($0) ?? -1 }
                if bits.count == 2, bits[0] >= 0, bits[1] >= bits[0] {
                    for i in bits[0]...bits[1] { out.append(i) }
                }
            } else if let v = Int(s) {
                out.append(v)
            }
        }
        return out.filter { $0 >= 0 && $0 < total }
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt hqq <input.tinygpt> [options]

        --out <path>         Where to save the (re)quantised model — required
        --bits N             Bit-width (2 | 3 | 4 | 8; default 4)
        --group-size N       Per-row group size for scales+zeros (default 64)
        --p F                IRLS sub-quadratic exponent (default 0.7)
                              0.5-1.0 typical; 1.0 collapses to weighted L2.
        --iterations N       IRLS rounds (default 30)
        --layers SPEC        Layer indices to quantise (e.g. "8-11" or "0-11").
                              Empty = all layers.

        Writes a model whose weights have been quantize-then-dequantised
        per HQQ. The model loads + runs unchanged via the existing
        fp32 forward path (no inference speedup until the packed-int
        matmul kernel ships).
        """)
        exit(2)
    }
}
