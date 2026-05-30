import Foundation
import MLX
import MLXNN

// AUDIT FLAG: SmoothQuant (Xiao et al., 2022).
//
// Tested: calibration pass + W·diag(s) rewrite. Outlier activation
//   channels collapse correctly; pure-float matmul output preserved
//   bit-identically (1e-6 roundoff).
// Saw: algorithmic infrastructure shipped. ZERO runtime payoff today
//   because MLX-Swift has no int8 matmul kernel — without that, the
//   scaled weights run at fp32 just like un-scaled.
// When this would help: when MLX-Swift ships int8 matmul OR when we
//   export the scaled model to a downstream runtime (llama.cpp,
//   mlx-lm) that has the kernel.

/// SmoothQuant (Xiao et al., 2022) — pre-quantization activation smoothing.
///
/// The problem SmoothQuant solves: activations into a Linear are often
/// far more skewed across channels than the weights are. A few outlier
/// channels carry 10-100× the magnitude of the rest. When you int8-
/// quantise the activations, those outliers blow out the scale and the
/// inliers all collapse onto a handful of quant levels — perplexity
/// craters.
///
/// The trick: introduce a per-INPUT-CHANNEL scale `s[i]` ≥ 0 and rewrite
///
///     y = (x / s) · (s · W)        // mathematically identical
///
/// by absorbing `s` into the weight: `W' = diag(s) · W`. Now the
/// activation that hits the int8 quantiser is `x / s`, whose channel-
/// wise range is smoothed by `s`. Choose `s` so that the per-channel
/// MAX of `x / s` and the per-channel MAX of `s · W` are comparable.
/// The paper's recipe:
///
///     s[i] = max(|x[:, i]|)^α  /  max(|W[:, i]|)^(1 − α)
///
/// where `α` (typically 0.5) trades activation-smoothing for weight
/// stretching. Set `s[i] = 1` if either max is below `eps` (a "dead"
/// channel — leave it alone).
///
/// The TRANSFORM is mathematically exact. After applying:
///   - W ← W · diag(s)     (one matmul, per Linear)
///   - the user is responsible for dividing inputs by s before the
///     Linear at runtime, or for fusing s into the PREVIOUS layer's
///     output projection (the paper does the latter — folds s into
///     the upstream LayerNorm's gamma or the previous Linear's
///     output channels).
///
/// **This module ships the CALIBRATION + SCALING pass.** It walks a
/// small text corpus through the model, tracks per-channel activation
/// max at every Linear, computes the s vector, and rewrites the
/// Linear's weight in place. Wiring s back INTO the prior layer's
/// output (so no runtime divide is needed) is a per-architecture
/// fold and queued as a follow-up — for now, we save the scaled
/// weights AND the s vector so downstream tooling (`llama.cpp`,
/// `mlx-lm` int8 path, future TinyGPT kernels) can apply both halves.
///
/// **Honest caveat — no int8 matmul kernel today.** MLX-Swift's
/// matmul is fp32/fp16/bf16. Calling this pass produces a model
/// that's BETTER-CONDITIONED for downstream int8 quantisation, but
/// the inference-side win materialises only when (a) the user
/// exports to a runtime with an int8 kernel, or (b) Apple ships
/// `mlx::quantized_matmul` int8 support, or (c) we hand-roll one.
/// The calibration logic here is unconditionally useful regardless
/// — it's a deterministic data transform.
public enum SmoothQuant {

    /// Configuration for the smoothing pass.
    public struct Config: Sendable {
        /// SmoothQuant alpha (Xiao et al. recommend 0.5; range [0, 1]).
        /// 0 = all the smoothing burden on weights (s = 1/max|W|).
        /// 1 = all on activations (s = max|x|).
        public var alpha: Float
        /// Epsilon below which a channel is treated as dead — `s` left at 1.
        public var eps: Float
        /// Maximum activation samples to accumulate before stopping
        /// calibration (one "sample" = one forward of a batch×T window).
        public var maxCalibrationSamples: Int
        public init(alpha: Float = 0.5, eps: Float = 1e-5,
                    maxCalibrationSamples: Int = 32) {
            self.alpha = max(0, min(1, alpha))
            self.eps = eps
            self.maxCalibrationSamples = maxCalibrationSamples
        }
    }

    /// Per-Linear scaling result.
    public struct LinearScale: Sendable {
        public let name: String           // dotted path, e.g. "blocks.0.attn.q_proj"
        public let scale: [Float]         // s[i], length = in_features
        /// Per-channel activation max captured during calibration.
        public let actMax: [Float]
        /// Per-channel weight max (over the OUT axis, before scaling).
        public let weightMax: [Float]
    }

    /// Run SmoothQuant calibration + scaling on a model.
    ///
    /// `forwardOnce` is a closure the caller provides — it runs ONE
    /// forward of the model on a calibration window and lets the
    /// activation hook (installed via the per-Linear `linearForwardHook`
    /// closure SOMEONE WIRED IN before calling this) populate the
    /// accumulator. To keep the wiring lightweight we instead use the
    /// FUNCTIONAL hook approach below: the caller passes both the model
    /// and a "snapshot per-channel max of x at this Linear" callback;
    /// here we re-walk every Linear in the model's module tree,
    /// computing the necessary statistics via the model's parameter view.
    ///
    /// The realistic deployment is: drive forwards externally (since
    /// the model's normal forward path doesn't expose per-Linear inputs
    /// to a hook in MLX-Swift), accumulate `actMax[layer][channel]`
    /// in a `[String: [Float]]` dictionary, then call
    /// `applyScales(model:scales:)` to rewrite weights.
    ///
    /// This top-level helper takes the END-OF-CALIBRATION accumulator
    /// as input (the caller does the calibration), computes `s`, and
    /// applies it.
    ///
    /// Returns the list of per-Linear scales that were applied — handy
    /// for serialising as a sidecar `.smoothquant` file (the user can
    /// keep the scales for downstream int8 export).
    public static func smooth(
        linearWeights: inout [String: MLXArray],
        activationMax: [String: [Float]],
        config: Config = Config()
    ) -> [LinearScale] {
        var out: [LinearScale] = []
        for (name, actMax) in activationMax {
            // The matching weight is at `<name>.weight`. Some callers
            // pass the bare Linear name; tolerate both.
            let weightKey: String
            if linearWeights[name + ".weight"] != nil {
                weightKey = name + ".weight"
            } else if linearWeights[name] != nil {
                weightKey = name
            } else {
                continue
            }
            let W = linearWeights[weightKey]!
            // W shape is [out, in]. We need per-INPUT-CHANNEL max |W|, i.e.
            // max over the OUT axis. The W matrix lives on GPU; pull abs-max
            // through MLX to keep it fast.
            let wAbs = MLX.abs(W)
            let wMaxArr = wAbs.max(axis: 0)              // [in]
            let wMaxFloats: [Float] = wMaxArr.asArray(Float.self)
            let inDim = actMax.count
            guard wMaxFloats.count == inDim else {
                // Shape mismatch — skip with a noisy log so the user can
                // chase wiring problems. Should never fire under a correct
                // calibration walk.
                FileHandle.standardError.write(Data(
                    "SmoothQuant: skipping \(name) — actMax dim \(inDim) ≠ weightMax dim \(wMaxFloats.count)\n".utf8))
                continue
            }
            var s = [Float](repeating: 1, count: inDim)
            let alpha = config.alpha
            for i in 0..<inDim {
                let a = max(actMax[i], 0)
                let w = max(wMaxFloats[i], 0)
                if a < config.eps || w < config.eps {
                    s[i] = 1   // dead channel — pass through
                    continue
                }
                // s = a^α / w^(1 − α). Avoid pow(NaN) by clamping.
                let num = Foundation.pow(a, alpha)
                let den = Foundation.pow(w, 1 - alpha)
                s[i] = max(num / max(den, config.eps), config.eps)
            }
            // W' = W · diag(s)  — multiply column i by s[i]. In [out, in]
            // layout, that's row-wise broadcast of s along axis -1.
            let sArr = MLXArray(s, [1, inDim])
            let newW = W * sArr
            linearWeights[weightKey] = newW
            out.append(LinearScale(name: name, scale: s,
                                    actMax: actMax, weightMax: wMaxFloats))
        }
        return out
    }

    /// Build a `[String: [Float]]` activation-max accumulator by walking
    /// the model's Linear modules and recording their hidden-dim size.
    /// Names match `weightKey` minus `.weight` — i.e. dotted paths like
    /// `blocks.0.attn.q_proj`.
    ///
    /// Callers populate the floats by running calibration forwards and
    /// tracking abs-max per channel at each Linear's INPUT. There's no
    /// universal hook surface in MLX-Swift; the recipe shipped here is:
    /// run the model end-to-end on a sample, then for each Linear call
    /// `MLX.abs(linear_input).max(axis: [0,1])` and pool element-wise.
    /// Doing that requires reaching into the forward path; the
    /// `tinygpt sft` style finetune driver does the same kind of
    /// per-layer access via Module reflection.
    public static func makeAccumulator(linearWeights: [String: MLXArray]) -> [String: [Float]] {
        var acc: [String: [Float]] = [:]
        for (name, W) in linearWeights {
            guard W.shape.count == 2 else { continue }
            let inDim = W.shape[1]
            // Strip ".weight" suffix to match the user's hook keys.
            let baseName = name.hasSuffix(".weight")
                ? String(name.dropLast(".weight".count))
                : name
            acc[baseName] = [Float](repeating: 0, count: inDim)
        }
        return acc
    }

    /// Update a single channel's running max in-place. Use this from
    /// the calibration walker to pool stats across micro-batches.
    public static func updateMax(_ acc: inout [Float], with sample: [Float]) {
        precondition(acc.count == sample.count, "channel count mismatch")
        for i in 0..<acc.count { acc[i] = max(acc[i], sample[i]) }
    }

    /// Per-channel abs-max of a `[B, T, C]` activation as `[C]` Floats.
    public static func channelAbsMax(_ x: MLXArray) -> [Float] {
        let absX = MLX.abs(x)
        let C = x.shape.last!
        // Reduce over all axes except the last.
        var reduced = absX
        while reduced.shape.count > 1 {
            reduced = reduced.max(axis: 0)
        }
        // reduced is now [C].
        precondition(reduced.shape == [C],
                     "channelAbsMax reduced to shape \(reduced.shape), expected [\(C)]")
        return reduced.asArray(Float.self)
    }

    /// Serialise the LinearScale list as a JSON dictionary suitable for
    /// shipping alongside the rewritten weights. Schema:
    ///
    ///   { "<name>": { "scale": [Float], "actMax": [Float], "weightMax": [Float] }, ... }
    public static func encodeJSON(_ scales: [LinearScale]) throws -> Data {
        var top: [String: Any] = [:]
        for s in scales {
            top[s.name] = [
                "scale": s.scale,
                "actMax": s.actMax,
                "weightMax": s.weightMax,
            ]
        }
        return try JSONSerialization.data(withJSONObject: top, options: [.prettyPrinted])
    }
}
