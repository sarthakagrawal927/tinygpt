# Mixture-of-Experts — more capacity per byte of weight

MoE replaces a transformer block's single dense MLP with `N` "expert"
MLPs plus a learned router that picks the top-`K` experts for each
token. The result: a model with `~N×` the parameter capacity of the
dense baseline at the same per-token FLOP budget — once the sparse
dispatch kernel lands.

**Reference**: Fedus et al., 2021 (Switch Transformer);
Jiang et al., 2024 (Mixtral-of-Experts).

---

## Why MoE matters at our scale

The user is shipping on a single 48 GB M-series Mac. The hard wall is
"can the model fit in RAM" — not "how many FLOPs per token." A 2B-
parameter MoE (8 experts × 256M dense) loads as ~4 GB at bf16, but
activates only ~500M params per token. So the *capacity* you can
expose locally is bigger than what a dense same-FLOP model could
deliver. That's the qualitative leverage MoE buys.

The compute saving — the famous "5× cheaper than the dense
equivalent" claim — requires a real sparse scatter-gather kernel. Our
first cut runs every expert on every token, weighted by the router,
so per-token FLOPs are *higher* than dense. The architecture is
correct; the perf knob is a follow-up.

## What's wired today

`tinygpt train --moe-experts N --moe-topk K --moe-aux-weight F`

- `--moe-experts N` (default 1): how many experts per block. Set to
  the same N at every block — heterogeneous MoE isn't supported yet.
- `--moe-topk K` (default 1): how many experts each token activates.
  K=1 = Switch Transformer (one expert per token). K=2 = Mixtral.
  Capped at N at parse time.
- `--moe-aux-weight F` (default 0.01): the load-balance loss scale.
  Lower lets the router specialise faster but risks collapse to a
  single expert; higher keeps usage uniform but slows specialisation.
  Switch Transformer's recipe is 0.01; we match it.

The MoE block adds:
- A bias-free `router: Linear(d_model → n_experts)` per block.
- `N` expert MLPs per block (same architecture as the dense baseline).
- An auxiliary load-balance loss accumulated during forward and folded
  into the training loss as `α · N · Σ_e (f_e · P_e)` (Switch recipe).

**Save/load works**: MoE models serialise to `.tinygpt` with extended
manifest entries (`blocks.N.moe.router.weight`,
`blocks.N.moe.experts.E.fc_in.weight`, etc.) and `nExperts`/`moeTopK`/
`loadBalanceWeight` in the JSON header. Resume restores the same
router + expert layout. The standard `sample`, `eval`, and `inspect`
paths read these new entries via the existing header → ModelConfig
flow.

## Smoke result

On a 200 KB corpus, tiny preset, 30 steps, byte-level:

| Config | Params | Loss (init → 30) | step/s |
|---|---:|---|---:|
| Dense MLP | 842 K | 6.09 → 1.76 | 55.6 |
| MoE 4 experts top-2 | 2.42 M | 5.95 → 1.68 | 29.3 |

The MoE has 2.88× the parameters and reaches a marginally lower loss
in the same step count, despite the slower per-step throughput (every
expert runs on every token). On the real test — longer training on
real data — the parameter-capacity gap is expected to widen
meaningfully.

## What's NOT shipped yet

- **Sparse dispatch.** Today's compute path is dense — every expert
  runs on every token, multiplied by the (mostly zero) router weight.
  Real sparse dispatch (gather → per-expert forward → scatter) is the
  compute-saving win and the reason MoE exists at the lab scale.

  We investigated three possible paths:

  1. **Capacity-bounded gather/scatter with `take` + `putAlong`.**
     For each expert, argPartition tokens by router prob, take the
     top `capacity` (`= ceil(N · 1.25 / E)`), run the expert on the
     subset, scatter results back. The dealbreaker: MLX-Swift
     doesn't expose `scatter_add`. `putAlong` is an ASSIGN scatter,
     not an additive one — so overlapping writes from different
     experts (or zero-overflow writes that overwrite real values)
     corrupt the output. We can substitute "build a full `[N,C]`
     delta tensor with the rest of the rows masked to zero, then add
     it to the accumulator" — that's correct but compute-equivalent
     to the dense path we already ship.

  2. **Permutation + grouped dense matmul (megablocks-style).** Sort
     tokens by router assignment so each expert's tokens form a
     contiguous block, run one batched matmul against the expert
     weights gathered along axis 0, then inverse-permute. This needs
     a fused grouped matmul kernel — not in MLX-Swift, would have to
     be a custom Metal shader.

  3. **Custom Metal kernel.** Writing a sparse MoE forward in Metal
     and bridging through MLX-Swift's `CustomFunction` is feasible
     but is a multi-day project — outside this session.

  None of (1)/(2)/(3) ships in the current scope. The dense compute
  path is the honest "shipping-today" MoE; the compute saving is on
  the roadmap behind any of:
  - MLX-Swift exposing `scatter_add` (lets path 1 deliver real savings),
  - upstream addition of grouped matmul / megablocks kernels, or
  - this project growing a Metal kernel of its own.
- **HF-architecture MoE.** `TransformerBlockHF` (SwiGLU + RMSNorm)
  doesn't yet have an MoE variant. The from-scratch path is the only
  way to train MoE today. Wiring is mechanical: parallel changes to
  `TransformerBlockHF` to take `MoEMLP` instead of `SwiGLU` when
  `cfg.isMoE`.
- **Browser MoE loading.** The browser's gallery reader assumes the
  dense manifest. MoE checkpoints fail the browser's tensor name
  validation. The MoE manifest is a Mac-side extension for now.
- **Distillation FROM open MoE teachers.** The Phase 5 headline was
  "distill from an open-MoE teacher (Mixtral / DeepSeek) into our
  smaller MoE." That needs an HF MoE loader (the SafetensorsReader +
  HFModel path doesn't know about the router/expert layout in MoE
  safetensors). Queued behind HF MoE support above.

## Hyperparameter notes

- Start with `--moe-experts 8 --moe-topk 2` (Mixtral defaults) on
  preset huge or larger. Smaller presets don't have enough capacity
  to benefit.
- If the load-balance loss isn't dropping (router collapsed to one
  expert), raise `--moe-aux-weight` to 0.05.
- If experts aren't specialising (all probs ~uniform), lower
  `--moe-aux-weight` to 0.001 — too much pressure prevents the
  router from picking any preferences.
- Top-1 is faster to train (no renormalisation, simpler gate) but
  doesn't transfer well; top-2 is the sweet spot for actual quality.

## Where to look in the code

- `Sources/TinyGPTModel/MoE.swift` — the `MoEMLP` module, router, and
  load-balance accumulator.
- `Sources/TinyGPTModel/TransformerBlock.swift` — the swap between
  dense `MLP?` and `MoEMLP?` at construction.
- `Sources/TinyGPTModel/TinyGPTModel.swift` — `loss()` folds the
  auxiliary loss in via `sumMoEAuxLosses(blocks)`.
- `Sources/TinyGPTModel/ModelConfig.swift` — the `nExperts`,
  `moeTopK`, `loadBalanceWeight`, `isMoE` fields.
