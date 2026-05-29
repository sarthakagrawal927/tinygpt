# First Worked Run — `tinygpt bench`

*Captured 2026-05-30. Validates the harness wired up end-to-end against
the existing in-process MLX inference path. This is the "smoke test"
that proves the scaffold produces real numbers; it is not a publishable
benchmark (n=5, no energy metrics, tiny model).*

Design doc: [`docs/benchmark_harness_design.md`](benchmark_harness_design.md).

## What I ran

A 9.6 M-parameter byte-level transformer trained in the browser
(`shakespeare.bin`, the canonical gallery checkpoint), driven through
the new harness in single-stream mode. No `powermetrics` (the laptop
isn't running with sudo, so energy/ANE are skipped — graceful
degradation is part of the design).

```bash
# from native-mac/
swift build -c release

# copy MLX kernel library next to the binary so MLX can find it via
# load_colocated_library() — see "MLX metallib lookup" caveat below.
cp /opt/homebrew/lib/mlx.metallib .build/release/mlx.metallib

./.build/release/tinygpt bench \
  --model ../browser/public/gallery/shakespeare.bin \
  --mode single \
  --prompt-tokens 64 \
  --gen-tokens 100 \
  --n-runs 5 \
  --warm-runs 1 \
  --no-energy \
  --output /tmp/tinygpt_bench_shakespeare.json
```

## Markdown table the harness printed

```
tinygpt bench — tinygpt
---------------------------------
model:           ../browser/public/gallery/shakespeare.bin
workload:        single, prompt=64 tok, gen=100 tok, batch=1
runs:            5 (+1 warm)
energy metrics:  off
loaded — 9,608,704 params
prompt: 64 bytes/tokens (byte-level)

running…
done in 1.8s across 5 runs
```

| metric | median | p95 | p99 | n |
|---|---|---|---|---|
| TTFT (ms) | 1.91 | 2.08 | 2.08 | 5 |
| decode tok/s | 794.91 | 835.06 | 835.06 | 5 |
| prefill tok/s | 33489.55 | 39190.52 | 39190.52 | 5 |
| ITL (ms) | 1.15 | 1.96 | 2.84 | 490 |
| peak RSS (MB) | 257.5 | 258.3 | 258.3 | 5 |
| energy/token (J) | — | — | — | (skip — no sudo for powermetrics) |

Warnings the harness flagged on this run:

- `n=5 is small; p95/p99 are unstable. Use --n-runs 20+ for
  paper-quality numbers.`
- `git tree is dirty — uncommitted changes; results not reproducible.
  Commit before publishing.`

Both of these are the harness behaving correctly. They are the
guardrails the design doc §1 promised.

## JSON shape (excerpt — `metrics` block from the same run)

```json
{
  "engine": { "name": "tinygpt", "commit": "bc4d4a0" },
  "git_commit": "bc4d4a0",
  "git_dirty": true,
  "harness_version": "0.1.0",
  "metrics": {
    "ttft_ms":      { "median": 1.91, "p95": 2.08, "p99": 2.08, "n": 5 },
    "decode_tps":   { "median": 794.91, "p95": 835.06, "p99": 835.06, "n": 5 },
    "prefill_tps":  { "median": 33489.55, "p95": 39190.52, "p99": 39190.52, "n": 5 },
    "itl_ms":       { "median": 1.15, "p95": 1.96, "p99": 2.84, "n": 490 },
    "peak_rss_mb":  { "median": 257.46, "p95": 258.31, "p99": 258.31, "n": 5 },
    "energy_per_token_j": { "median": null, "n": 0 },
    "ane_residency_pct":  { "median": null, "n": 0 }
  },
  "model":   { "path": "...", "params": 9608704 },
  "workload":{ "mode": "single", "batch_size": 1, "prompt_tokens": 64,
                "gen_tokens": 100, "n_runs": 5, "warm_runs": 1 },
  "system":  { "hardware_model": "Mac17,8", "macos_build": "25F71",
                "physical_ram_gb": 48 },
  "runs": [ /* one entry per run with per-run breakdown */ ],
  "warnings": [ "n=5 is small; …", "git tree is dirty — …" ]
}
```

The full JSON has a `runs` array with one entry per run for per-run
inspection (raw decode_ms, ITL median per run, etc.) — useful for
detecting drift over the run series.

## What this proves

- The `EngineAdapter` → `WorkloadController` → `MetricsCollector` →
  `Reporter` chain works end-to-end on real MLX execution.
- Wall-clock timing on the existing `AnyModel.forwardCached` path is
  internally consistent (decode median 795 tok/s, ITL median 1.15 ms
  → 1/1.15 = 870 tok/s if we discount the first-token warm-up the
  harness already excludes from the ITL distribution; the harness
  drops `interTokenLatenciesMs[0]` per `Reporter.summarize`).
- JSON validates against `python3 -m json.tool`. Markdown renders.
- The "small n" and "dirty tree" guardrails fire as designed.

## What this does NOT prove

- Anything about engine ranking (no foreign engines wired yet).
- Anything about energy efficiency (no sudo for powermetrics).
- Anything about ANE utilization (no ANE path in TinyGPT today;
  Orion-style routing is future work).
- Anything about thermal sustained performance (mode is `single`, not
  `sustained` — that mode is a placeholder).

These are the next milestones. They're scaffolded for; not in scope
for this commit.

## Caveats noted during the first run

- **MLX metallib lookup**: a clean `swift build` does not copy
  `mlx.metallib` next to the SPM-produced binary. MLX's
  `load_default_library` looks for `mlx.metallib` colocated with the
  executable first; without it, every model load fails with
  `Failed to load the default metallib`. This bites `tinygpt sample`
  too; it's a pre-existing build-system gap, not introduced by the
  bench harness. The fix above (`cp /opt/homebrew/lib/mlx.metallib
  .build/release/`) is the local workaround. The systemic fix —
  declaring the metallib as an SPM `.resource` — is an mlx-swift
  upstream change; track separately.
- **`task_info(TASK_VM_INFO).phys_footprint`** reports peak RSS at
  ~257 MB despite the model being 9.6 M params (~40 MB at fp32). The
  rest is the MLX runtime, ARC heap, tokenizer machinery, and Metal
  command buffers. Not a bug in the harness — that's what your Mac
  Activity Monitor shows too — but worth flagging when readers
  compare against bare model size.

## Next steps (post-scaffold)

1. Wire `MLXLMEngine` — subprocess `mlx_lm.generate` with per-token
   stdout timestamps; same protocol surface, no change to the
   Reporter.
2. Wire `LlamaCppEngine` via `llama-cli --simple-io`.
3. Add an `--enable-energy` run on a Mac with sudo configured and
   compare energy/token to the literature.
4. Implement `--mode sustained --duration 600` for the
   thermal-throttle regression.
5. Add a downloader for ShareGPT-v3 prompts so we stop using the
   synthetic byte filler for paper-quality runs.
