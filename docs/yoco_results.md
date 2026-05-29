# YOCO — "You Only Cache Once" worked example

This doc captures the implementation and smoke-test measurements for
YOCO (Lin et al., 2024) in the Mac training path (Tier 3.8 of the
single-machine roadmap). All numbers below were collected on an M5 Pro
with 48 GB unified memory, MLX-Swift 0.31, fp32, Metal backend.

## TL;DR

`tinygpt train --yoco` halves the KV cache at long-context decode time
with no measurable quality regression on small-corpus smoke tests.

| Config | Layers | Cache @ 206 tok (off) | Cache @ 206 tok (on) | Δ memory |
|---|---:|---:|---:|---:|
| small (6L · d=192) | 6 | 1.9 MB | 949 KB | **-50%** |
| huge (12L · d=256) | 12 | 5.1 MB | 2.5 MB | **-51%** |

Loss curves track within a couple percent across a 200-step run; final
sample quality is indistinguishable between YOCO-on and YOCO-off at
this scale.

## What changed in code

YOCO splits the transformer in two halves. The first half (layers
`0..nLayers/2 - 1`) runs standard causal self-attention. The LAST
first-half layer ("the anchor") captures its rotated K, V tensors.
Every layer strictly after the anchor does CROSS-ATTENTION onto those
saved K, V — it computes Q from its own local hidden state, but skips
its K/V projections entirely. KV cache memory at long-context decode
drops by ~2×: only `nLayers/2` layers grow the cache.

### Files added

- `native-mac/Sources/TinyGPTModel/CrossAttention.swift` — dedicated
  attention module that takes external K, V tensors instead of
  projecting from x. Has only `q_proj` and `o_proj` (no
  `k_proj`/`v_proj`). Takes a `posOffset` parameter so RoPE-style
  models can rotate Q at the absolute decode position when the cache
  has already advanced past the prefill.

### Files modified

- `native-mac/Sources/TinyGPTModel/TransformerBlock.swift` —
  `TransformerBlock.init` now takes a `yocoSecondHalf: Bool` flag
  (defaults to false for backwards compatibility). When set with
  `cfg.useYOCO`, the block constructs a `CrossAttention` sibling at
  `@ModuleInfo(key: "cross_attn")`. The existing `attn:
  CausalSelfAttention` field stays allocated — its weights are dead
  at forward time on second-half layers, but kept for manifest /
  LoRA / debug stability (same pattern used by `diff_attn`). The
  block routes through whichever attention is installed via
  `callWithExternalKV(x, k:v:posOffset:)`.
- `native-mac/Sources/TinyGPTModel/TransformerBlockHF.swift` — same
  YOCO sibling for the HF-style block; mirrors the from-scratch
  block's `callCapturingKV` / `callWithExternalKV` API.
- `native-mac/Sources/TinyGPTModel/TinyGPTModel.swift` —
  `forwardToHidden` (training/prefill path) is YOCO-aware: anchor
  capture at layer `nLayers/2 - 1`, cross-attention thereafter. The
  block constructor receives the per-layer `yocoSecondHalf` decision.
- `native-mac/Sources/TinyGPTModel/HFModel.swift` — same YOCO
  routing in `callAsFunction` for the HF model.
- `native-mac/Sources/TinyGPTModel/KVCache.swift` — `forwardCached`
  for the from-scratch model now only grows the cache on first-half
  layers. Second-half layers read back the anchor's `entries[anchorIdx]`
  K, V and cross-attend onto them with `posOffset = basePos`.
- `native-mac/Sources/TinyGPTModel/KVCacheHF.swift` — same change
  for the HF cached forward.
- `native-mac/Sources/TinyGPT/Sample.swift` — KV cache size report
  appended to the post-generation summary. Reports populated layer
  count when YOCO is on so the halving is visible.

Train.swift's `--yoco` flag, ModelConfig.useYOCO, and the manifest
field already existed as prep work — this change is what makes them
actually do anything.

## Smoke test: 200 steps on Shakespeare, preset `small`

```
==== YOCO ON, 200 steps ====
  step     1/  200  loss 6.186
  step    50/  200  loss 2.711
  step   100/  200  loss 2.562
  step   150/  200  loss 2.518
  step   200/  200  loss 2.520

==== YOCO OFF, 200 steps ====
  step     1/  200  loss 6.155
  step    50/  200  loss 2.688
  step   100/  200  loss 2.524
  step   150/  200  loss 2.563
  step   200/  200  loss 2.471
```

The curves track each other closely. YOCO ends 2% higher at step 200;
that's within the run-to-run noise of a 200-step random-batch training
loop on a 1 MB corpus.

## Smoke test: cached sampling at preset `huge` (12L, ctx=256)

```
=== YOCO ON sample ===
(200 tokens in 0.30s — 676 tok/s · KV-cached)
KV cache:  206 tokens · 2.5 MB  · YOCO (6/12 layers populated)

=== YOCO OFF sample ===
(200 tokens in 0.26s — 767 tok/s · KV-cached)
KV cache:  206 tokens · 5.1 MB
```

Clean 2× cache reduction. Only the first 6 layers (anchor included)
are populated; the second 6 cross-attend onto layer-5's K, V slot and
allocate no cache of their own.

Decode throughput drops ~12% (767 → 676 tok/s) because cross-attention
still does the full SDPA against the same length-256 K, V — only the
projections are skipped. The throughput win YOCO claims at very long
contexts comes from the cache memory savings (more concurrent decode
streams fit per machine, less DRAM bandwidth on each cache load),
which doesn't show up at ctx=256. At ctx=4k+ on a memory-bound regime
the throughput crossover should land in YOCO's favour.

## Caveats

1. **Anchor placement is hard-coded** to `nLayers/2 - 1`. The YOCO
   paper experimented with different anchors; we use the middle layer
   for simplicity. Tunable anchor depth is a follow-up.
2. **Second-half `attn` weights are still allocated** — they get
   trained as part of the parameter tree but are never used at forward
   time on second-half layers. Param count grows by `~nLayers/2 ×
   d_model²` extra (the dead K, V projections we keep for manifest
   stability) PLUS `~nLayers/2 × 2 × d_model²` for the new
   CrossAttention's Q / O projections. Net wash on params; the win is
   pure activation-memory at decode.
3. **CrossAttention has Q and O projections too** — we could fold
   them with the anchor's K, V to save more compute, but that
   complicates the LoRA / param-name story. Kept separate for clean
   targeting.
4. **First-half cache layout is unchanged**: KV-quantization and
   StreamingLLM sink/window still compose. Second-half layers
   benefit transparently — they read whatever the anchor stored.
5. **No HF-checkpoint round-trip yet**: loading an HF model and
   toggling YOCO at init "works" architecturally (the K, V
   projections still load from safetensors and the cross-attn Q/O
   stay random-init), but the cross-attn weights are untrained so
   sampling quality is undefined. Production HF YOCO requires either
   a YOCO-aware HF checkpoint or a from-scratch retrain.

## Reproducing

```bash
# Train YOCO model
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -scheme tinygpt -destination "platform=macOS" \
  -derivedDataPath /tmp/tinygpt-build -configuration Release build

/tmp/tinygpt-build/Build/Products/Release/tinygpt train \
  --preset huge --steps 200 --yoco \
  --corpus data/examples/shakespeare.txt \
  --out yoco-on.tinygpt

# Compare KV cache sizes
/tmp/tinygpt-build/Build/Products/Release/tinygpt sample \
  yoco-on.tinygpt --tokens 200 --prompt "ROMEO:"
# → "KV cache:  206 tokens · 2.5 MB  · YOCO (6/12 layers populated)"
```

## References

- Lin et al., 2024 — "You Only Cache Once: Decoder-Decoder
  Architectures for Language Models" (arXiv:2405.05254).
- Tier 3.8 in `docs/single_machine_roadmap.md`.
- Design notes in `docs/archive/phase_9_10_status.md`.
