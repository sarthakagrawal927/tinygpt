# Evolution Strategies — gradient-free training

ES is a finite-difference approach to optimisation: at every step, we
sample K random perturbations of the current parameters, evaluate
each perturbed model on a shared batch, and update the parameters
along the reward-weighted average noise direction.

Useful when:
- The reward signal isn't differentiable (RL with discrete actions,
  exact-match accuracy, etc.).
- You want to train a model without exposing gradients (privacy / IP
  considerations).
- As an educational counterpoint to the SGD path: ES has the same
  asymptotic guarantees with very different constants.

Reference: Salimans et al., 2017, "Evolution Strategies as a Scalable
Alternative to Reinforcement Learning" ([arXiv:1703.03864](https://arxiv.org/abs/1703.03864)).

---

## Command

```sh
tinygpt es <model.tinygpt> --corpus <text> \
    --steps 200 --population 40 --sigma 0.02 --lr 0.01 \
    --out es-trained.tinygpt
```

Note: ES is byte-level-only in this first cut, and operates on
from-scratch models (the model's parameters are saved through the
existing `.tinygpt` manifest). The starting checkpoint can be a
fresh-train output or any prior-saved model.

## The algorithm

Per ES step:

1. **Snapshot base parameters** `w` (the current model state).
2. **Sample a shared batch** — same data for every population member.
3. **For each of K/2 pairs**, draw noise `ε ~ N(0, I)` shaped like
   `w`, evaluate `L_+(ε) = loss(w + σε)` and `L_-(ε) = loss(w - σε)`.
4. **Reward = -loss** (higher is better).
5. **Centre the rewards** by subtracting the mean across all K
   samples. Standard variance-reduction trick.
6. **Estimate the gradient** via the antithetic estimator:
   `dir = Σ_pairs ((R_+ - R_-) / 2) · ε`
7. **Apply the step**: `w ← w + (lr / (K · σ)) · dir`

The antithetic pairing — using `+ε` and `-ε` for each random vector —
cuts the gradient-estimate variance roughly in half for the same K
samples vs. one-sided estimation. Salimans 2017's headline trick.

## Hyperparameter notes

- **Population K**: must be EVEN (we pair them). 20-50 is a workable
  range for tiny models. The roadmap'd "scalable" version uses K in
  the hundreds across many machines; on one Mac, larger K just
  trades compute for variance reduction at diminishing returns.
- **Sigma σ**: 0.01-0.05 typical. Too small → no signal escapes the
  noise. Too large → perturbed models become incoherent.
- **lr**: 0.005-0.05 typical. Direct interpretation: per-step
  parameter movement is bounded by `lr / σ × max_reward_difference`.
- **Batch / context**: each population member runs ONE forward; pick
  modest sizes since K×forward is the dominant per-step cost.

## What ES is NOT

- Not a replacement for SGD on small-to-medium transformer training.
  Per-step convergence in our smoke runs is much slower than the
  AdamW baseline at the same wall-clock.
- Not differentiable-bypass for cases where SGD works fine. The
  variance per step grows with the parameter count; on a 100M-param
  model, K would need to be massive.
- Not currently parallel — the K forward passes run serially on one
  Mac. Multi-Mac ES would be a follow-up.

## Where to look

- `Sources/TinyGPT/ES.swift` — the trainer command + step routine.
- `Sources/TinyGPT/TinyGPT.swift` — CLI dispatch.
