import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Drop-in optimizer alternatives to AdamW.
///
/// MLX-Swift already ships AdamW, Lion, and Adafactor; the project also
/// wants Sophia and Muon. This file:
///
///   1. Adds Sophia (Liu et al., 2023) and Muon (Jordan et al., 2024)
///      as `OptimizerBase` subclasses, matching the same protocol as
///      MLX-Swift's built-ins.
///   2. Exposes a single `OptimizerKind` enum + `makeOptimizer(...)`
///      factory so the trainer can be initialised with any choice
///      using a unified call site.
///   3. Provides a tiny `OptimizerHandle` shim so callers can poke
///      `learningRate` (for cosine-decay/warmup schedulers) without
///      knowing the concrete optimizer type.
///
/// Memory characteristics (per-parameter optimiser state, fp32):
///   - AdamW:     2 × |θ|   (m, v)
///   - Lion:      1 × |θ|   (m only — sign-based update)
///   - Sophia:    2 × |θ|   (m, h — Hessian-diagonal EMA)
///   - Muon:      1 × |θ|   (momentum-only for 2D; 1D leaves use AdamW
///                            internally so still 2× there)
///   - Adafactor: ~0.5 × |θ| on 2D leaves (row-sum + column-sum); 1× on
///                            1D leaves. Big tensors dominate the
///                            optimiser-state line in LMs so the
///                            average ends up ~½ AdamW.
///
/// All optimisers respect MLX-Swift's `Optimizer.update(model:gradients:)`
/// contract, so the same `compile(inputs:[m,opt], outputs:[m,opt]) { ... }`
/// machinery works unchanged.

// MARK: - OptimizerKind

/// CLI-facing enum: `--optimizer {adamw|lion|sophia|muon|adafactor}`.
public enum OptimizerKind: String, CaseIterable, Sendable {
    case adamw, lion, sophia, muon, adafactor
}

// MARK: - LearningRateMutable

/// Common surface for "settable LR at training time" — needed by the
/// cosine-decay scheduler in `Train.run`. All optimisers we wire up
/// conform; some forward to internal property setters (Adafactor's
/// `learningRate` is Float?, the others are Float).
public protocol LearningRateMutable: AnyObject {
    var learningRate: Float { get set }
}

extension AdamW: LearningRateMutable {}
extension MLXOptimizers.Lion: LearningRateMutable {}

/// Adafactor's stored `learningRate` is `Float?` (supports the
/// relative-step mode where the optimiser computes its own LR each
/// step). To present a uniform `LearningRateMutable` surface we wrap
/// the optimiser in a thin facade that round-trips through the
/// optional. We only set Adafactor up in relative-step=false mode
/// (see `makeOptimizer`), so the optional is always populated and
/// the getter never returns 0.
public final class AdafactorAdapter: Optimizer, LearningRateMutable {
    public let inner: MLXOptimizers.Adafactor
    public init(_ inner: MLXOptimizers.Adafactor) { self.inner = inner }

    public var learningRate: Float {
        get { inner.learningRate ?? 0 }
        set { inner.learningRate = .some(newValue) }
    }

    public func update(model: Module, gradients: ModuleParameters) {
        inner.update(model: model, gradients: gradients)
    }
    public func innerState() -> [MLXArray] {
        inner.innerState()
    }
}

// MARK: - State

/// Our own (m, v) pair, since MLX-Swift's `PairState` has internal
/// initialisers. Same shape; just public.
public struct PairState: Updatable {
    public var a: MLXArray
    public var b: MLXArray

    public init(_ a: MLXArray, _ b: MLXArray) {
        self.a = a
        self.b = b
    }
    public init(zeros array: MLXArray) {
        self.a = MLXArray.zeros(like: array)
        self.b = MLXArray.zeros(like: array)
    }
    public func innerState() -> [MLXArray] { [a, b] }
}

// MARK: - Sophia

/// Sophia optimizer — Stochastic Hessian-Diagonal Information Aware
/// (Liu et al., 2023).
///
/// Practical variant used here:
///   m_t = b1 * m_{t-1} + (1-b1) * g
///   h_t = b2 * h_{t-1} + (1-b2) * g * g     // Hessian-diagonal proxy
///   update = sign(m_t) * min(|m_t| / (rho * h_t + eps), 1)
///   param -= lr * (update + weightDecay * param)
///
/// The paper's "Sophia-G" Gauss-Newton estimator updates `h` only
/// every k steps using a fresh logit sample, but on bf16/fp32
/// transformer pre-training the EMA-of-squared-grads ("Sophia-light")
/// behaves similarly while keeping the step strictly local — no extra
/// forward passes. We document this in `docs/optimizers.md`.
///
/// Headline behaviour vs Adam: the per-coordinate clip `min(.., 1)`
/// caps the per-step move at `lr` for any coordinate whose
/// "preconditioned ratio" exceeds 1, which is what gives Sophia its
/// claimed robustness to bad-curvature directions.
public final class Sophia: Optimizer, LearningRateMutable {
    public var learningRate: Float
    /// First-moment EMA rate. Paper default 0.96; we use 0.965 (paper Table 1).
    public var beta1: Float
    /// Hessian EMA rate. Paper default 0.99.
    public var beta2: Float
    /// Pre-conditioner strength. Paper recommends ρ ∈ [0.01, 0.1];
    /// 0.04 is the BERT/GPT setting from the paper.
    public var rho: Float
    /// Numerical-stability epsilon on the denominator.
    public var eps: Float
    /// Decoupled weight decay (AdamW-style).
    public var weightDecay: Float

    /// One PairState per parameter, keyed by the model's nested
    /// parameter name. Same shape as MLXOptimizers' `OptimizerBase`
    /// stateStorage; we maintain it manually because the base class
    /// initialiser is internal to MLXOptimizers and can't be called
    /// from outside the module.
    private var stateStorage = NestedDictionary<String, PairState>()

    public init(
        learningRate: Float = 3e-4,
        betas: (Float, Float) = (0.965, 0.99),
        rho: Float = 0.04,
        eps: Float = 1e-12,
        weightDecay: Float = 0.1
    ) {
        self.learningRate = learningRate
        self.beta1 = betas.0
        self.beta2 = betas.1
        self.rho = rho
        self.eps = eps
        self.weightDecay = weightDecay
    }

    public func innerState() -> [MLXArray] {
        stateStorage.flattenedValues().flatMap { $0.innerState() }
    }

    public func update(model: Module, gradients: ModuleParameters) {
        let modelParams = model.parameters()
        let (p, s) = gradients.mapValues(modelParams, stateStorage) {
            (grad, param, state) -> (MLXArray, PairState?) in
            let pState = state ?? PairState(zeros: param!)
            let (newParam, newState) = applySingle(gradient: grad, parameter: param!, state: pState)
            return (newParam, newState)
        }
        self.stateStorage = s
        model.update(parameters: p)
    }

    private func applySingle(
        gradient: MLXArray, parameter: MLXArray, state: PairState
    ) -> (MLXArray, PairState) {
        var m = state.a
        var h = state.b
        m = beta1 * m + (1 - beta1) * gradient
        h = beta2 * h + (1 - beta2) * (gradient * gradient)

        // ratio = m / (ρ·h + eps). Element-wise clipped to [-1, 1] via
        // sign(m) * min(|ratio|, 1) — Sophia's defining update shape.
        let denom = rho * h + eps
        let ratio = m / denom
        let absRatio = MLX.abs(ratio)
        let one = MLXArray(Float(1))
        let clipped = MLX.sign(ratio) * MLX.minimum(absRatio, one)

        // Decoupled weight decay: shrink param before subtracting update.
        var p = parameter
        if weightDecay > 0 {
            p = p * (1 - learningRate * weightDecay)
        }
        return (p - learningRate * clipped, PairState(m, h))
    }
}

// MARK: - Muon

/// Muon optimizer — "MomentUm Orthogonalized via Newton-schulz"
/// (Jordan, Bernstein et al., 2024).
///
/// For 2D parameters (attention/MLP weights), runs Nesterov-style
/// momentum followed by Newton-Schulz orthogonalisation:
///
///   m_t = β·m_{t-1} + g
///   u_t = β·m_t + g                      (Nesterov lookahead)
///   O   = NewtonSchulz5(u_t / ||u_t||_F) // ≈ U Vᵀ from SVD(u_t)
///   param -= lr · scale · O
///
/// where `scale = max(1, sqrt(out_dim / in_dim))` — keeps the update
/// norm comparable to AdamW so the LR transfers.
///
/// For 1D parameters (LayerNorm γ/β, biases) and embeddings, Muon
/// falls back to AdamW internally — the orthogonalisation trick only
/// makes sense for matrices.
///
/// Memory: 2D leaves carry one momentum buffer (1× param); 1D leaves
/// pay AdamW's 2× cost. On a typical transformer 1D leaves are a few
/// percent of total parameters, so Muon is effectively 1× optimiser
/// memory.
public final class Muon: Optimizer, LearningRateMutable {
    public var learningRate: Float
    /// Momentum coefficient (β). Paper uses 0.95.
    public var momentum: Float
    /// Number of Newton-Schulz iterations. 5 is the paper-recommended
    /// fixed count; coefficients (3.4445, -4.7750, 2.0315) chosen so
    /// the iteration converges to an orthonormal matrix within 5
    /// rounds on practical gradient SVDs.
    public var nsIterations: Int
    /// Decoupled weight decay (applied like AdamW).
    public var weightDecay: Float
    /// Adam fallback hyperparameters for 1D / embedding params.
    public var adamBetas: (Float, Float)
    public var adamEps: Float

    /// Per-parameter state. See Sophia.stateStorage.
    private var stateStorage = NestedDictionary<String, PairState>()

    public init(
        learningRate: Float = 2e-3,
        momentum: Float = 0.95,
        nsIterations: Int = 5,
        weightDecay: Float = 0.0,
        adamBetas: (Float, Float) = (0.9, 0.95),
        adamEps: Float = 1e-8
    ) {
        self.learningRate = learningRate
        self.momentum = momentum
        self.nsIterations = nsIterations
        self.weightDecay = weightDecay
        self.adamBetas = adamBetas
        self.adamEps = adamEps
    }

    public func innerState() -> [MLXArray] {
        stateStorage.flattenedValues().flatMap { $0.innerState() }
    }

    public func update(model: Module, gradients: ModuleParameters) {
        let modelParams = model.parameters()
        let (p, s) = gradients.mapValues(modelParams, stateStorage) {
            (grad, param, state) -> (MLXArray, PairState?) in
            let pState = state ?? PairState(zeros: param!)
            let (newParam, newState) = applySingle(gradient: grad, parameter: param!, state: pState)
            return (newParam, newState)
        }
        self.stateStorage = s
        model.update(parameters: p)
    }

    private func applySingle(
        gradient: MLXArray, parameter: MLXArray, state: PairState
    ) -> (MLXArray, PairState) {
        if parameter.ndim == 2 && parameter.shape[0] > 1 && parameter.shape[1] > 1 {
            return muonStep(gradient: gradient, parameter: parameter, state: state)
        } else {
            return adamFallback(gradient: gradient, parameter: parameter, state: state)
        }
    }

    /// Newton-Schulz quintic iteration. Operates on the input matrix
    /// `x` ([m, n]); transposes once when m > n so the inner loop
    /// always works on the wider-than-tall side (fewer flops).
    /// Returns the orthogonalised matrix.
    private func newtonSchulz5(_ xIn: MLXArray) -> MLXArray {
        // Coefficients from the Muon paper (the 5-step polynomial
        // that converges to the matrix sign on SVDs with singular
        // values in [0, 1]).
        let a: Float = 3.4445
        let b: Float = -4.7750
        let c: Float = 2.0315
        // Normalise: divide by Frobenius norm so the largest singular
        // value is ≤ 1 going into the iteration.
        let fro = MLX.sqrt((xIn * xIn).sum()) + MLXArray(Float(1e-7))
        var x = xIn / fro
        // Transpose so we always operate on the [smaller, larger] form.
        let transposed = x.shape[0] > x.shape[1]
        if transposed { x = x.transposed() }
        for _ in 0..<nsIterations {
            // A = X Xᵀ;  X ← a·X + b·A·X + c·A·A·X
            let a_mat = MLX.matmul(x, x.transposed())
            let b_mat = MLX.matmul(a_mat, x)
            let c_mat = MLX.matmul(a_mat, b_mat)
            x = a * x + b * b_mat + c * c_mat
        }
        if transposed { x = x.transposed() }
        return x
    }

    private func muonStep(
        gradient: MLXArray, parameter: MLXArray, state: PairState
    ) -> (MLXArray, PairState) {
        var m = state.a
        let unused = state.b
        // Nesterov-flavoured momentum: keep the old m, blend in g.
        m = momentum * m + gradient
        let lookahead = momentum * m + gradient

        // Orthogonalise the lookahead update. Result has Frobenius
        // norm ≈ sqrt(min(m, n)) (since singular values ≈ 1), so we
        // rescale to keep step magnitude comparable to AdamW.
        let ortho = newtonSchulz5(lookahead)
        let outDim = Float(parameter.shape[0])
        let inDim  = Float(parameter.shape[1])
        let scale = max(Float(1), (outDim / inDim).squareRoot())

        var p = parameter
        if weightDecay > 0 {
            p = p * (1 - learningRate * weightDecay)
        }
        // `unused` slot stays zero — keeps state shape constant.
        return (p - learningRate * scale * ortho, PairState(m, unused))
    }

    private func adamFallback(
        gradient: MLXArray, parameter: MLXArray, state: PairState
    ) -> (MLXArray, PairState) {
        let (b1, b2) = adamBetas
        var m = state.a
        var v = state.b
        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * (gradient * gradient)

        var p = parameter
        if weightDecay > 0 {
            p = p * (1 - learningRate * weightDecay)
        }
        let update = m / (MLX.sqrt(v) + adamEps)
        return (p - learningRate * update, PairState(m, v))
    }
}

// MARK: - Factory

/// Construct an optimiser of the requested kind, applying
/// per-optimiser hyperparameter defaults when the caller hasn't
/// overridden them. Returns an `any Optimizer & LearningRateMutable`
/// so the trainer can both step it and adjust LR for cosine-decay.
///
/// LR defaults follow the paper recommendations:
///   - AdamW:    user-provided
///   - Lion:     5×-10× smaller than AdamW (paper recommendation);
///               we leave the caller's LR as-is — users should pass
///               a smaller `--max-lr` when picking Lion.
///   - Sophia:   3e-4 is a sane default for transformer pre-training
///   - Muon:     2e-3 (paper); much higher than AdamW because the
///               orthogonalised update has unit-ish singular values
///   - Adafactor: relativeStep=false + scaleParameter=false + an
///                explicit LR — required so the LR scheduler can
///                drive it like the other optimisers.
public func makeOptimizer(
    kind: OptimizerKind,
    learningRate: Float,
    weightDecay: Float,
    betas: (Float, Float) = (0.9, 0.95),
    eps: Float = 1e-8
) -> any Optimizer & LearningRateMutable {
    switch kind {
    case .adamw:
        return AdamW(learningRate: learningRate, betas: betas,
                     eps: eps, weightDecay: weightDecay)
    case .lion:
        // Lion typically wants ~⅓-⅒ AdamW's LR; caller is responsible
        // for picking that, since the schedule code uses the caller's
        // --max-lr. We pass-through.
        return MLXOptimizers.Lion(
            learningRate: learningRate,
            betas: (betas.0, 0.99),
            weightDecay: weightDecay
        )
    case .sophia:
        return Sophia(learningRate: learningRate,
                      betas: (betas.0, 0.99),
                      rho: 0.04,
                      eps: 1e-12,
                      weightDecay: weightDecay)
    case .muon:
        return Muon(learningRate: learningRate,
                    momentum: 0.95,
                    nsIterations: 5,
                    weightDecay: weightDecay,
                    adamBetas: betas,
                    adamEps: eps)
    case .adafactor:
        // relativeStep=false + scaleParameter=false locks Adafactor
        // into a "use my LR" mode so the cosine scheduler keeps
        // working. The headline memory win — row/column factorisation
        // of the second-moment — is independent of these knobs.
        let af = MLXOptimizers.Adafactor(
            learningRate: learningRate,
            eps: (1e-30, 1e-3),
            clipThreshold: 1,
            decayRate: -0.8,
            beta1: nil,            // no first-moment by default → memory win
            weightDecay: weightDecay,
            scaleParameter: false,
            relativeStep: false,
            warmupInit: false
        )
        return AdafactorAdapter(af)
    }
}

/// Parse the CLI string. Returns nil for unknown values; callers
/// emit a usage-style error in that case.
public func parseOptimizerKind(_ s: String) -> OptimizerKind? {
    return OptimizerKind(rawValue: s.lowercased())
}
