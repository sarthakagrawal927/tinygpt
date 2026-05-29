# TinyGPT Inference Benchmark Harness — Design

*Status: scaffold landed 2026-05-30. This is the foundation for the
"most powerful inference engine for modern transformers" claim; nothing
about that claim is credible without a reproducible measurement frame.*

*Background research that drives every decision below lives at
[`docs/research/inference_benchmarks_may_2026.md`](research/inference_benchmarks_may_2026.md).
Quality-benchmark survey at
[`docs/research/quality_benchmarks_may_2026.md`](research/quality_benchmarks_may_2026.md).*

## 1. Goals

1. **Reproducible** Apple-Silicon inference numbers any third party can
   replicate from this repo at a pinned commit hash. MLPerf-style
   submitter README + log replay is the gold bar; Bench360
   (arXiv 2511.16682) is the closest academic suite and is the
   architectural model we copy.
2. **Apple-specific metrics no one else publishes**: ANE residency %,
   ANE↔GPU handoff latency, sustained tok/s through thermal throttle,
   energy/token via `powermetrics` on unified-memory semantics. The
   research doc explicitly flags these four as publishable gaps.
3. **Cross-engine** apples-to-apples comparison vs MLX-LM, llama.cpp
   Metal, MLC-LLM, and Ollama on the same prompt, model SHA,
   quantization, sampling params, and thermal state.
4. **Modular**: workload controller, engine adapter, metrics collector
   are independent so we can grow each without touching the others.
   This mirrors the Bench360 split (task engine + workload controller +
   backend abstraction + metrics collector).
5. **Honest at small sample sizes**: when n<20 the harness logs a
   warning before reporting p95/p99. Single-host benchmarks rarely hit
   n≥20 in a developer workflow, so we name the limitation rather than
   hide it.

## 2. Non-goals (for this scaffold)

- Foreign-engine adapters (MLX-LM, llama.cpp, MLC-LLM, Ollama). The
  protocol shape is in place but the subprocess wrappers are stubs.
- RULER v2 / LongBench v2 long-context modes. Hooks are in the
  metrics catalog but no driver yet — that's the second milestone.
- The actual ANE+GPU routing prototype. The benchmark is the *frame*
  the routing work is measured against, not the routing itself.
- A web UI / dashboard. JSON + markdown out, that's it. Pipe into
  whatever you like.
- Distributed / multi-host benchmarking. Single-Mac scope.

## 3. Reference architecture

Direct port of the Bench360 four-component split, with one Apple-only
addition (`PowerSampler`):

```
                     ┌───────────────────┐
                     │   CLI: `tinygpt   │
                     │       bench`      │
                     └────────┬──────────┘
                              │
                  ┌───────────▼────────────┐
                  │   WorkloadController   │
                  │   (single|batch|server)│
                  └───┬──────────────┬─────┘
                      │              │
            ┌─────────▼────┐   ┌─────▼────────────┐
            │   Task / IO  │   │  MetricsCollector│
            │  (prompt gen,│   │   ─ TTFT, ITL    │
            │  token count)│   │   ─ tok/s        │
            └─────────┬────┘   │   ─ peak RSS     │
                      │        │   ─ PowerSampler ├──► powermetrics
                      │        │     (ANE, GPU,   │     (NSTask)
                      │        │      CPU, energy)│
                      │        └─────────┬────────┘
                  ┌───▼────────────────┐ │
                  │   EngineAdapter    │ │
                  │   protocol         │ │
                  │  ┌──────────────┐  │ │
                  │  │TinyGPTEngine │  │ │
                  │  │MLXLMEngine*  │  │ │
                  │  │LlamaCppEngine*│ │ │
                  │  │OllamaEngine* │  │ │
                  │  └──────────────┘  │ │
                  └────────┬───────────┘ │
                           │             │
                       ┌───▼─────────────▼──┐
                       │      Reporter      │
                       │  JSON + markdown   │
                       │  median + p95/p99  │
                       └────────────────────┘
```

`*` = protocol stub, not implemented in this milestone.

### 3.1 EngineAdapter (`EngineAdapter.swift`)

The protocol every engine must satisfy:

```swift
protocol EngineAdapter {
    var name: String { get }
    var commitHash: String? { get }
    func load(modelPath: String) throws
    func prefill(tokens: [Int]) throws -> PrefillResult     // returns TTFT + first logits
    func decode(maxTokens: Int, batchSize: Int) throws -> DecodeResult
    func peakResidentBytes() -> Int
    func reset()
}
```

`TinyGPTEngine` is implemented end-to-end against the existing MLX path
(`AnyModel.forwardCached` + `KVCache`). The three foreign engines are
declared with the same protocol and `throws .notImplemented` so the
WorkloadController can still wire them up in tests.

### 3.2 WorkloadController (`WorkloadController.swift`)

Three modes, all from Bench360:

- **single-stream** — one prompt at a time, no overlap. The default.
  Equivalent to a single user typing at the model. Reports TTFT + ITL +
  steady-state decode tok/s.
- **batch** — N prompts submitted simultaneously, all decoded
  in lockstep. Measures throughput under static batching (what a naive
  serving setup gets). `--batch-size N` controls N. The Apple-Silicon
  community usually reports batch=1 only, which understates throughput.
- **server** — concurrent requests arriving from an arrival
  distribution (Poisson, default λ matching ShareGPT-v3's pace).
  Measures continuous-batching throughput under load. This scaffold
  implements only single-stream and batch; the server mode is wired
  through the same controller but currently calls into batch as a
  placeholder. Real server-mode requires a request scheduler — that's
  the third milestone.

### 3.3 MetricsCollector + PowerSampler (`MetricsCollector.swift`)

Two cooperating pieces:

1. **In-process timer/counters** — `mach_absolute_time` for sub-ms
   timestamps; `mstats()` / `task_info(TASK_VM_INFO)` for resident /
   peak resident bytes; per-token wall-clock for ITL distribution.
2. **`PowerSampler`** — spawns `powermetrics --samplers ane_power,gpu_power,cpu_power -i 100 -f plist`
   via `Foundation.Process`. Reads the plist stream line-by-line on a
   background queue, parses each sample, accumulates into a time-series.
   On `stop()`, terminates the child process and returns the series.
   `powermetrics` requires root; if it isn't available we log a warning
   and skip the energy/ANE metrics rather than fail the run. The CPU
   metrics still come through `task_info` so the run still produces
   numbers.

The 100ms sample interval is the default for `powermetrics`; finer
intervals (10ms) measurably load the system and bias the throughput
number, so we don't.

### 3.4 Reporter (`Reporter.swift`)

Emits a single JSON object per run *and* a markdown table for human
reading. JSON schema is the canonical artifact; markdown is derived.

```json
{
  "harness_version": "0.1.0",
  "git_commit": "abc1234",
  "engine": "tinygpt",
  "engine_commit": "abc1234",
  "model": { "path": "...", "params": 9608704, "config": {...} },
  "workload": { "mode": "single", "batch_size": 1, "prompt_tokens": 128, "gen_tokens": 100, "n_runs": 5, "warm_runs": 1 },
  "system": { "hardware": "...", "macos": "...", "thermal_state": "..." },
  "metrics": {
    "ttft_ms":        { "median": ..., "p95": ..., "p99": ..., "n": 5 },
    "itl_ms":         { "median": ..., "p95": ..., "p99": ..., "n": 500 },
    "decode_tps":     { "median": ..., "p95": ..., "p99": ..., "n": 5 },
    "prefill_tps":    { "median": ..., "p95": ..., "p99": ..., "n": 5 },
    "peak_rss_mb":    { "median": ..., "max": ..., "n": 5 },
    "energy_per_token_j": { "median": ..., "n": 5 },
    "ane_residency_pct": { "median": ..., "n": 5 }
  },
  "warnings": [ "n=5 too small for stable p99 — increase --n-runs to ≥20" ]
}
```

## 4. Metrics catalog (exact definitions)

All times are wall-clock measured at the API boundary (the caller
processes a complete prompt + generation request and the engine returns
text), unless otherwise stated.

| Metric | Definition | Units | How measured |
|---|---|---|---|
| **TTFT** | `time(first_token_emitted) - time(request_received)` | ms | timer in `WorkloadController` around `engine.prefill()` returning the first sampled token |
| **ITL / TPOT** | per-token inter-arrival time during decode (excludes TTFT) | ms | timer in `engine.decode()` between successive token emits; report distribution |
| **decode tok/s (B)** | `(gen_tokens × batch_size) / (decode_wall_time)` at batch size B | tok/s | reported at batch=1, 4, 16, 64 (caller specifies `--batch-size`) |
| **prefill tok/s** | `prompt_tokens / TTFT_seconds` | tok/s | derived from TTFT |
| **peak RSS** | high-water `phys_footprint` from `task_info(TASK_VM_INFO)` over the run | MB | sampled once per token; max retained |
| **unified-memory high-water** | high-water resident over all backing stores reported by `mstats()`/`vm_statistics64`, including wired/compressed | MB | sampled once per token, same loop as RSS |
| **sustained tok/s (10 min)** | decode tok/s over a 10-minute steady-state run after a 30 s warm-up; reports both mean and the slope (regression coefficient) — slope ≈ 0 means no throttle | tok/s, tok/s/min | special `--mode sustained` flag (post-scaffold milestone) |
| **energy / token** | `∫P(t) dt` from `powermetrics` over the decode window, divided by tokens emitted | J/tok | `PowerSampler` 100 ms samples × wattage |
| **ANE residency %** | `(time_ane_active / total_decode_time)` from `powermetrics` ane_power sampler — ANE active iff `ane_power > 50 mW` (threshold tuned for noise floor) | % | `PowerSampler` time-series, post-processed |
| **ANE↔GPU handoff latency** | latency between an ANE-power-down event and the next GPU-power-up event (or vice-versa) during a decode that crosses backends. **Requires** a hybrid backend that actually does crossings — measured against the future ANE+GPU routing prototype, instrumented here so the harness is ready | ms | `PowerSampler` edge detection on the time-series |

For long-context measurements (post-scaffold), the same metrics get
reported separately at context lengths 4k, 32k, 128k, 1M, and the
research doc's distinction between cache-hit and cache-miss prefill is
preserved.

## 5. Workload modes (detail)

### single-stream

`--mode single`. One prompt, N=`--n-runs` repetitions. TTFT measured
per-run; ITL measured per-token; reported as distributions.

### batch

`--mode batch --batch-size N`. N copies of the prompt submitted as a
single batched forward. Same metric definitions; tok/s scales with N
when the engine supports batching, stays at 1× when it doesn't (a
diagnostic in itself).

### server (placeholder)

`--mode server --rps R`. Reserved. Currently calls into the batch path
and emits a warning. Implementing this needs a request scheduler that
respects an arrival distribution — Bench360's choice is Poisson with
λ derived from ShareGPT-v3 token-counts; we'll do the same when we
implement it.

### sustained (placeholder)

`--mode sustained --duration 600`. Reserved. The thermal-throttle
measurement. Will run continuous decode for `--duration` seconds and
report the regression of tok/s on time. M-series Macs throttle
visibly at ~6–8 minutes of sustained MLX load; we want that number on
the page.

## 6. Reproducibility requirements

Every run records:

- `git rev-parse HEAD` of this repo (commit hash, dirty flag).
- For TinyGPTEngine: same. For foreign engines (future): the engine
  binary's `--version` or git SHA.
- Model SHA-256 (computed lazily — header inspection plus a hash of
  the weight bytes; cached in `~/.tinygpt/bench-cache/`).
- Quantization scheme and bits (`int4` / `int8` / `fp16` / `bf16` / `fp32`).
- KV-cache dtype.
- RNG seed (default 42).
- Sampling params (temperature, top-p, top-k).
- Prompt corpus origin (default: a short deterministic Shakespeare
  excerpt; flag to switch to ShareGPT-v3 or LMSYS-Chat-1M slices once
  we add a downloader).
- Hardware SKU + RAM tier + macOS build (`sysctl hw.model`,
  `sw_vers`).
- Thermal state at run start (`pmset -g therm`) — cold/steady-state
  flag.
- Ambient temp not measured (no API); flagged as a known unaccounted-for
  source of variance.

All recorded in the JSON output. The matching markdown table includes
the full provenance block so a PDF screenshot is enough for a reader to
re-run.

## 7. Comparison plan vs MLX-LM, llama.cpp Metal, MLC-LLM, Ollama

When the foreign engines are wired (next milestone), the harness will:

1. Run each engine on the same prompt + sampling params + model SHA.
2. Use the lowest common quantization scheme each engine supports
   (typically Q4_0 / int4 group=64).
3. For MLX-LM: subprocess `mlx_lm.generate` with `--max-tokens` and a
   wrapper that prints per-token timestamps; parse stdout.
4. For llama.cpp: subprocess `./llama-cli -m model.gguf -p prompt
   --logit-bias 0 --no-display-prompt -n N --simple-io`; parse
   per-token markers from stdout.
5. For MLC-LLM: subprocess `mlc_llm chat ... --benchmark`; parse JSON.
6. For Ollama: HTTP POST against the local server with
   `stream: true`; per-chunk timestamps.
7. Pin each engine's commit hash; record both engine and harness
   commits in the output JSON.

This subprocess-based design preserves engine-native execution. A
shared-library FFI integration would be cleaner numerically but harder
to keep faithful to each engine's recommended use; subprocesses match
how a real user runs them.

## 8. Citation to research

- Bench360, arXiv 2511.16682 (Nov 2025) — architectural source.
- "Production-Grade Local LLM Inference on Apple Silicon", arXiv
  2511.05502 (Nov 2025) — the prior art this work engages with.
- TokenPowerBench, arXiv 2512.03024 (Dec 2025, AAAI) — energy-metric
  semantics; their per-token energy definition is the one we use,
  adapted to unified-memory `powermetrics` rather than NVIDIA NVML.
- MLPerf Inference v6.0 (Apr 2026) — submitter README + log replay
  reproducibility bar.
- Orion (referenced in 2511.05502) — proof that ANE is reachable from
  user code; motivates the ANE residency + handoff metrics that no
  public benchmark publishes.

Full survey, with URLs:
[`docs/research/inference_benchmarks_may_2026.md`](research/inference_benchmarks_may_2026.md).

## 9. Implementation status (2026-05-30)

| Component | Status |
|---|---|
| `tinygpt bench` CLI subcommand | ✅ wired |
| `Benchmark.swift` arg parser | ✅ |
| `WorkloadController` single / batch | ✅ |
| `WorkloadController` server / sustained | placeholder, warns |
| `MetricsCollector` timers + RSS | ✅ |
| `PowerSampler` (`powermetrics` NSTask) | ✅ scaffolded; sudo gracefully skipped |
| `TinyGPTEngine` adapter | ✅ end-to-end |
| `MLXLMEngine` / `LlamaCppEngine` / `OllamaEngine` | stubs that throw `.notImplemented` |
| `Reporter` JSON + markdown | ✅ |
| Long-context modes (4k / 32k / 128k / 1M) | not started |
| Sustained-load thermal regression | not started |
| ANE+GPU routing prototype + handoff latency | depends on routing work |

## 10. Open design questions for review

1. **`bench` subcommand name collision** — the existing
   `tinygpt bench` is a *training-throughput* benchmark vs the browser
   baseline. This work renames that to `tinygpt bench-train` and
   reassigns `bench` to inference. If preferred, we could keep
   `bench` for training and call this `tinygpt bench-infer` instead.
2. **`powermetrics` permission UX** — we currently log a warning and
   skip energy metrics if not root. Alternative: `sudo`-prompt
   interactively. The warning approach matches how the rest of the
   tinygpt CLI behaves (no privilege escalation), so it's the default.
3. **Default prompt corpus** — short Shakespeare excerpt for the
   scaffold. ShareGPT-v3 / LMSYS-Chat-1M are the conventional choices
   per the research doc; adding a downloader is a 1-day follow-up but
   wasn't in scope here.
4. **`n=5` default** — paper-quality numbers want n≥20 per the
   reproducibility bar. We default to 5 because dev-loop iteration
   speed matters and warn loudly otherwise. The CI runs (when we add
   them) should pin n=20.
