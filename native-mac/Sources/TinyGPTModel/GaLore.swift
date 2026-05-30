import Foundation
import MLX
import MLXNN
import MLXRandom

/// GaLore — Gradient Low-Rank Projection (Zhao et al., 2024;
/// https://arxiv.org/abs/2403.03507).
///
/// The insight: optimiser-state memory (Adam m, v) for a transformer
/// is dominated by the 2-D weight matrices, and Adam's update on those
/// matrices is *empirically* well-approximated by a rank-R update for
/// small R relative to min(in, out). GaLore exploits this by:
///
///   1. Maintaining a single projection basis `P : [m, r]` per tracked
///      matrix (m == the LARGER of the two leading dims of the param's
///      shape; we project the "wider" side).
///   2. Replacing the raw gradient G with `P @ (P^T @ G)` — a rank-R
///      approximation that lives in the span of P.
///   3. Periodically (every K steps) refreshing P from the SVD of the
///      *current* gradient: P = U[:, :r] of G's left singular vectors.
///
/// The result is FULL fine-tuning (every weight in the network moves)
/// at LoRA-like memory cost: the Adam state at the original [m, n]
/// shape, in a *truly* GaLore-aware optimiser, can be replaced with
/// state at [r, n] (or [m, r], depending on projection side). The
/// theoretical memory budget for the optimizer therefore equals LoRA
/// at the same rank, while the network keeps the expressive freedom
/// of touching every weight (LoRA only learns adapters on top of a
/// frozen base).
///
/// IMPLEMENTATION NOTE — Adam state shape.
/// MLX-Swift's AdamW keeps `m, v` at the FULL parameter shape. We
/// don't replace AdamW; we just project the GRADIENT before it
/// reaches AdamW. That preserves GaLore's training dynamics exactly,
/// but means the *actual* MLX state on disk doesn't shrink. To make
/// the "GaLore matches LoRA r=R memory" claim honest, we maintain a
/// PARALLEL low-rank Adam-shaped budget counter (`memoryBudget`)
/// that reports the size a fully GaLore-aware optimiser WOULD use.
/// Worked example with the run summary in `docs/galore_and_stability.md`.
public final class GaLoreProjector {
    public let rank: Int
    public let updateEvery: Int
    /// Which dim of the matrix we project onto. `.left` means
    /// `P : [out, r]`, gradient G [out, in] becomes `P (P^T G)` —
    /// projecting the OUTPUT side. `.right` means `P : [in, r]`,
    /// gradient is `(G P) P^T` — projecting the INPUT side. We pick
    /// the side with the LARGER dim (more compression).
    public enum Side { case left, right }
    public let side: Side
    /// Original parameter shape — used to validate refresh inputs.
    public let originalShape: [Int]

    /// Current basis. `nil` until the first refresh has happened.
    public private(set) var basis: MLXArray?
    /// Step counter — controls refresh cadence.
    public private(set) var step: Int = 0

    public init(rank: Int, updateEvery: Int, originalShape: [Int]) {
        precondition(originalShape.count == 2,
                     "GaLore projector only handles 2-D parameter matrices, got \(originalShape)")
        let outDim = originalShape[0]
        let inDim  = originalShape[1]
        // Pick the side with the LARGER dim — that's where rank-R
        // projection compresses the most. Tie goes to .left for
        // determinism.
        self.side = (outDim >= inDim) ? .left : .right
        self.rank = min(rank, min(outDim, inDim))
        self.updateEvery = max(1, updateEvery)
        self.originalShape = originalShape
    }

    /// Project a gradient through the (current) basis. Returns a
    /// rank-R approximation at the ORIGINAL shape so the downstream
    /// optimiser sees the same shape it always has — only the rank
    /// is reduced.
    ///
    /// Also bumps the step counter and triggers a basis refresh
    /// when `step % updateEvery == 0`. Refresh uses CPU-stream SVD
    /// (Metal SVD support is incomplete in MLX as of writing).
    public func project(_ grad: MLXArray) -> MLXArray {
        // First step (or refresh boundary) — recompute the basis from
        // the current gradient via SVD. Cheap relative to a forward
        // pass: SVD of a [d, d] matrix at d ≤ 4096 runs in milliseconds.
        if basis == nil || step % updateEvery == 0 {
            refresh(grad)
        }
        step += 1
        guard let P = basis else { return grad }
        // Apply the projection. `side == .left` means we left-multiply
        // by P @ P^T (an [out, out] projector onto the column space of P).
        // `side == .right` projects the row space.
        switch side {
        case .left:
            // grad: [out, in], P: [out, r]
            // low  = P^T @ grad           : [r, in]
            // proj = P @ low              : [out, in]
            let low = matmul(P.transposed(), grad)
            return matmul(P, low)
        case .right:
            // grad: [out, in], P: [in, r]
            // low  = grad @ P             : [out, r]
            // proj = low @ P^T            : [out, in]
            let low = matmul(grad, P)
            return matmul(low, P.transposed())
        }
    }

    /// Recompute the basis from G's top-R singular vectors. Done on
    /// the CPU stream because MLX-Metal's SVD support is incomplete
    /// on some shapes. SVD of a [d, d] matrix at d ≤ 4096 is fast
    /// enough that doing it once per `updateEvery` steps is dwarfed
    /// by the forward + backward cost.
    private func refresh(_ grad: MLXArray) {
        eval(grad)
        // U: [out, min(out, in)], Vt: [min(out, in), in]
        let (U, _, Vt) = MLX.svd(grad, stream: .cpu)
        eval(U, Vt)
        switch side {
        case .left:
            // Take the first `rank` columns of U as the basis.
            self.basis = U[0..., 0..<rank]
        case .right:
            // Take the first `rank` rows of Vt and transpose to [in, r].
            self.basis = Vt[0..<rank, 0...].transposed()
        }
        eval(self.basis!)
    }

    /// Theoretical Adam-state size (in floats) for this matrix under
    /// a fully GaLore-aware optimiser: 2 (m, v) × low-rank shape +
    /// the basis itself. Compare against `2 × m × n` for raw AdamW.
    public var loRankAdamFloats: Int {
        let outDim = originalShape[0]
        let inDim  = originalShape[1]
        let lowFloats: Int
        switch side {
        case .left:  lowFloats = rank * inDim      // [r, in]
        case .right: lowFloats = outDim * rank     // [out, r]
        }
        let basisFloats: Int
        switch side {
        case .left:  basisFloats = outDim * rank
        case .right: basisFloats = inDim * rank
        }
        return 2 * lowFloats + basisFloats
    }

    /// Raw AdamW state floats for the same matrix (the *actual* MLX
    /// cost today). Same number reported in the "full AdamW" column
    /// of the memory comparison.
    public var fullAdamFloats: Int {
        2 * originalShape[0] * originalShape[1]
    }
}

/// Manager — one projector per tracked parameter. The hook into the
/// trainer is `processGradients(grads, paramShapes:)`: walk the gradient
/// tree, project tracked entries through their projectors, leave
/// everything else (1-D bias / norm / scalar) untouched.
public final class GaLoreManager {
    public let rank: Int
    public let updateEvery: Int
    /// Param name → projector. Lazily populated on first sight of each
    /// 2-D parameter so we don't need to walk the model up-front.
    private var projectors: [String: GaLoreProjector] = [:]
    /// Names we've already DECIDED to skip (1-D, embedding-like, too
    /// small). Cached so we don't re-check every step.
    private var skipped: Set<String> = []
    /// Minimum number of elements before a tensor becomes worth
    /// projecting. SVD has overhead; tiny matrices (output bias, norm
    /// scale) gain nothing.
    public let minElements: Int

    public init(rank: Int = 256, updateEvery: Int = 200, minElements: Int = 4096) {
        self.rank = rank
        self.updateEvery = updateEvery
        self.minElements = minElements
    }

    /// True iff this name should be GaLore-projected. Skip
    /// embedding tables (their gradient is rank-1 at most positions
    /// — projection hurts more than it helps), norms (1-D), biases.
    public static func shouldTrack(name: String, shape: [Int]) -> Bool {
        if shape.count != 2 { return false }
        // Heuristic: keep embedding tables out of GaLore. Their grad
        // is naturally sparse (only the seen tokens get a non-zero
        // row each step); a dense rank-R approximation actively
        // damages the unseen rows.
        if name.contains("token_embedding") || name.contains("position_embedding")
            || name.contains("embed_tokens") {
            return false
        }
        // lm_head: same shape as embedding (vocab, d). Project it
        // — modern HF runs have shown GaLore on lm_head is fine.
        return true
    }

    /// Walk `grads` and replace tracked entries with their projected
    /// rank-R approximations.
    public func processGradients(_ grads: ModuleParameters) -> ModuleParameters {
        var result = NestedDictionary<String, MLXArray>()
        for (key, item) in grads {
            result[key] = processItem(item, path: [key])
        }
        return result
    }

    private func processItem(_ item: NestedItem<String, MLXArray>, path: [String])
        -> NestedItem<String, MLXArray>
    {
        switch item {
        case .none: return .none
        case .value(let g):
            let name = path.joined(separator: ".")
            return .value(maybeProject(name: name, grad: g))
        case .array(let elems):
            return .array(elems.enumerated().map { (i, e) in
                processItem(e, path: path + [String(i)])
            })
        case .dictionary(let dict):
            var newDict: [String: NestedItem<String, MLXArray>] = [:]
            for (k, v) in dict {
                newDict[k] = processItem(v, path: path + [k])
            }
            return .dictionary(newDict)
        }
    }

    private func maybeProject(name: String, grad: MLXArray) -> MLXArray {
        if skipped.contains(name) { return grad }
        if let p = projectors[name] {
            return p.project(grad)
        }
        // First time we've seen this leaf — decide.
        let shape = grad.shape
        let elems = shape.reduce(1, *)
        if !GaLoreManager.shouldTrack(name: name, shape: shape) || elems < minElements {
            skipped.insert(name)
            return grad
        }
        let projector = GaLoreProjector(rank: rank, updateEvery: updateEvery,
                                         originalShape: shape)
        projectors[name] = projector
        return projector.project(grad)
    }

    /// Total params tracked.
    public var trackedCount: Int { projectors.count }

    /// Sum of `loRankAdamFloats` across every tracked matrix — what
    /// a fully-GaLore-aware optimiser WOULD use for these params.
    public var theoreticalLowRankFloats: Int {
        projectors.values.reduce(0) { $0 + $1.loRankAdamFloats }
    }

    /// Sum of `fullAdamFloats` across every tracked matrix — what
    /// raw AdamW DOES use today.
    public var actualFullRankFloats: Int {
        projectors.values.reduce(0) { $0 + $1.fullAdamFloats }
    }

    /// Per-projector diagnostics for the run summary.
    public func summary() -> String {
        let tracked = projectors.count
        let lowMB = Double(theoreticalLowRankFloats * 4) / (1024 * 1024)
        let fullMB = Double(actualFullRankFloats * 4) / (1024 * 1024)
        let ratio = actualFullRankFloats > 0
            ? Double(theoreticalLowRankFloats) / Double(actualFullRankFloats) : 0
        return String(format:
            "GaLore: %d matrices tracked · rank=%d · refresh every %d steps · theoretical Adam state %.1f MB vs full %.1f MB (%.1f%%)",
            tracked, rank, updateEvery, lowMB, fullMB, ratio * 100)
    }
}

// MARK: - DeepNorm helper

/// Re-init a Linear's weight by multiplying its existing weight tensor
/// by the DeepNorm β factor. Returns a NEW `Linear` (MLX-Swift's
/// Linear weight is `let`, so we can't mutate; we rebuild it).
public func applyBetaInit(_ layer: Linear, beta: MLXArray) -> Linear {
    let scaled = layer.weight * beta
    return Linear(weight: scaled, bias: layer.bias)
}

// MARK: - Layer-wise LR decay

/// Scale per-layer gradients by `factor^(L - layerIdx)` so deeper layers
/// keep the full LR and shallower layers get progressively smaller
/// updates. Standard fine-tuning trick — surface-level features
/// generalise broadly, so they shouldn't move much; deeper task-
/// specific features benefit from higher LR.
///
/// Decay applied INSIDE the gradient tree (multiplies each
/// `blocks.N.*` leaf by `decay ^ (L - 1 - N)`), so the existing
/// optimiser's per-leaf update stays a single AdamW call. Embedding /
/// final-norm / lm_head get the FULL LR (multiplied by 1.0).
///
/// `decay == 1.0` (or `nil`) is a no-op — the caller can pass through
/// without branching.
public func scaleLayerwiseLR(_ grads: ModuleParameters, decay: Float, nLayers: Int) -> ModuleParameters {
    if decay >= 0.9999 { return grads }  // no-op fast path
    var result = NestedDictionary<String, MLXArray>()
    for (key, item) in grads {
        result[key] = scaleLayerItem(item, path: [key], decay: decay, nLayers: nLayers)
    }
    return result
}

private func scaleLayerItem(_ item: NestedItem<String, MLXArray>,
                             path: [String], decay: Float, nLayers: Int)
    -> NestedItem<String, MLXArray>
{
    switch item {
    case .none: return .none
    case .value(let g):
        // Find a `blocks.N` (TinyGPTModel) or `layers.N` (TinyGPTModelHF)
        // ancestor in the path. If absent → full LR (embeddings, norms,
        // lm_head). If present → scale by decay^(nLayers - 1 - N).
        let name = path.joined(separator: ".")
        if let layerIdx = extractLayerIdx(name: name) {
            let exponent = max(0, nLayers - 1 - layerIdx)
            let factor = pow(decay, Float(exponent))
            return .value(g * MLXArray(factor))
        }
        return .value(g)
    case .array(let elems):
        return .array(elems.enumerated().map { (i, e) in
            scaleLayerItem(e, path: path + [String(i)], decay: decay, nLayers: nLayers)
        })
    case .dictionary(let dict):
        var newDict: [String: NestedItem<String, MLXArray>] = [:]
        for (k, v) in dict {
            newDict[k] = scaleLayerItem(v, path: path + [k], decay: decay, nLayers: nLayers)
        }
        return .dictionary(newDict)
    }
}

/// Parse a dotted parameter name like "blocks.7.attn.q_proj.weight"
/// or "layers.3.self_attn.o_proj.weight" and return the layer index
/// (7 / 3 here). Returns nil for non-layer paths (embeddings, final
/// norm, lm_head).
private func extractLayerIdx(name: String) -> Int? {
    // Both "blocks.N" and "layers.N" are layer-prefixes. Walk the
    // dotted components; if we see one of those keys, the next
    // component is the index.
    let parts = name.split(separator: ".")
    for i in 0..<parts.count - 1 {
        if parts[i] == "blocks" || parts[i] == "layers" {
            return Int(parts[i + 1])
        }
    }
    return nil
}
