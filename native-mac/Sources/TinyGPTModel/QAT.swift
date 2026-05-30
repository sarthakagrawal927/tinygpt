import Foundation
import MLX
import MLXNN

/// Quantization-Aware Training (QAT) — fake-quant + straight-through-
/// estimator (STE) primitive used by the modified TransformerBlock /
/// TransformerBlockHF forwards when `cfg.qatBits` is set.
///
/// Forward math, per Linear's `weight` (in `[out, in]` layout):
///
///     scale[o] = max(|W[o, :]|) / qMax        // per-output-row scale
///     Wq[o, i] = round(W[o, i] / scale[o]) · scale[o]
///
/// Backward: gradient flows AS IF we had used `W` directly. The trick
/// (Hubara et al., 2016; standard PyTorch QAT recipe) is
///
///     W_used = W + stopGradient(Wq − W)
///
/// Forward value: `W + (Wq − W) = Wq`  (the quantised weight).
/// Backward value: `dL/dW + 0 = dL/dW` (stopGradient kills the inner gradient).
///
/// Why round-to-nearest with a SYMMETRIC scale (no zero-point)? It's
/// the simplest grid that lets us share one scale per OUT row, matches
/// what most int8 inference kernels expect (PyTorch's `int8_weight`
/// path, Apple's CoreML int8 matmul), and avoids the bias toward 0
/// that asymmetric grids can introduce at extreme bit widths.
///
/// QAT vs. post-hoc int4 / int8 quantisation: post-hoc takes the
/// trained fp32 weights and quantises them once. The model never saw
/// the quantisation noise during training, so the optimiser couldn't
/// route around it. QAT injects the noise at training time. On
/// transformer LMs the quality gap at int4 is typically 0.5-2 perplexity
/// points at no inference cost — the QAT-trained model deploys to the
/// SAME int4 kernel, just with better weights.
public enum QAT {

    /// Apply fake-quant + STE to a Linear weight matrix `W` shaped
    /// `[out, in]`. Returns a tensor whose forward value is the
    /// quantised weight and whose backward passes the gradient through
    /// untouched. Per-output-row symmetric grid.
    public static func fakeQuant(_ W: MLXArray, bits: Int) -> MLXArray {
        precondition(W.shape.count == 2, "fakeQuant expects [out, in], got \(W.shape)")
        precondition(bits >= 2 && bits <= 8, "fakeQuant bits must be 2..8")
        // qMax: integer half-range for a symmetric grid.
        //   int4 → 7 (range −7..+7), int8 → 127 (range −127..+127).
        let qMax: Float = Float((1 << (bits - 1)) - 1)
        // Per-output-row abs-max → scale[o] = absMax[o] / qMax.
        let absW = MLX.abs(W)                                  // [out, in]
        let absMax = absW.max(axis: 1, keepDims: true)         // [out, 1]
        // Floor scale at a tiny epsilon to avoid div by zero in dead rows.
        let scale = MLX.maximum(absMax / MLXArray(qMax),
                                MLXArray(Float(1e-8)))         // [out, 1]
        // Round W / scale to nearest int in [−qMax, +qMax], then scale back.
        let scaled = W / scale
        let rounded = MLX.round(scaled)
        let clipped = MLX.clip(rounded, min: MLXArray(-qMax), max: MLXArray(qMax))
        let Wq = clipped * scale
        // Straight-through: W + stopGradient(Wq − W). Forward = Wq; backward = dL/dW.
        return W + MLX.stopGradient(Wq - W)
    }

    /// Diagnostic — relative absolute reconstruction error between W
    /// and its fake-quantised version. Use this to print a per-block
    /// "quantisation error" metric every K training steps so the
    /// operator can verify the network is actually converging toward
    /// a quant-friendly weight distribution.
    public static func relativeError(_ W: MLXArray, bits: Int) -> Float {
        let Wq = fakeQuant(W, bits: bits)
        let err = MLX.abs(W - Wq).sum()
        let scale = MLX.abs(W).sum() + MLXArray(Float(1e-8))
        let ratio = (err / scale)
        ratio.eval()
        return ratio.item(Float.self)
    }

    /// Apply a Linear's projection using fake-quantised weights. Returns
    /// `x · fakeQuant(W).T (+ bias)` so the call site can drop in as a
    /// replacement for `linear(x)` when `cfg.qatBits != nil`. Bias is
    /// NOT fake-quantised — it's a tiny tensor whose dynamic range is
    /// already representable in int8 and the paper-standard recipe is
    /// to leave it at fp32.
    public static func linearForward(_ linear: Linear, x: MLXArray, bits: Int) -> MLXArray {
        let Wq = fakeQuant(linear.weight, bits: bits)
        if let b = linear.bias {
            return MLX.addMM(b, x, Wq.T)
        }
        return MLX.matmul(x, Wq.T)
    }
}
