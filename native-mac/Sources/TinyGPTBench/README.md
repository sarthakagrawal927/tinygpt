# TinyGPTBench

In-process Mac LLM-inference benchmark harness for the tinygpt
project. Modeled on Bench360 (arXiv 2511.16682, Nov 2025) with two
Apple-Silicon additions no public benchmark currently publishes:
`powermetrics`-derived **energy/token** and **ANE residency** during
serving.

Full design doc: [`../../../docs/benchmark_harness_design.md`](../../../docs/benchmark_harness_design.md).
First worked run with real numbers:
[`../../../docs/benchmark_first_run.md`](../../../docs/benchmark_first_run.md).

## Quick start

```bash
# from native-mac/
swift build -c release
# Until MLX-swift declares the metallib as an SPM resource, you need
# this once after each release build:
cp /opt/homebrew/lib/mlx.metallib .build/release/mlx.metallib

./.build/release/tinygpt bench \
  --model ../browser/public/gallery/shakespeare.bin \
  --gen-tokens 100 --n-runs 5 --no-energy \
  --output /tmp/bench.json
```

## Module layout

| File | Role |
|---|---|
| `Benchmark.swift` | `tinygpt bench` CLI entry; arg parsing; main run. |
| `EngineAdapter.swift` | Protocol every engine satisfies. Includes `TinyGPTEngine` (the in-process MLX adapter, implemented end-to-end) plus stubs for `MLXLMEngine`, `LlamaCppEngine`, `OllamaEngine`. |
| `WorkloadController.swift` | Drives `--mode {single,batch,server,sustained}` on top of an `EngineAdapter`. Single + batch are implemented; server + sustained are placeholders that warn and fall back. |
| `MetricsCollector.swift` | Per-run timers, peak-RSS sampling via `task_info`, and `PowerSampler` — a child `powermetrics` process parsed plist-by-plist for ANE/GPU/CPU power time-series. |
| `Reporter.swift` | JSON + markdown emission. Computes median/p95/p99 across runs; flags warnings about small n and dirty git trees. |

The CLI wiring is in `../TinyGPT/TinyGPT.swift` — `tinygpt bench`
dispatches into `Benchmark.run`. (Heads up: the previous
`tinygpt bench` — a training-throughput benchmark vs the browser
WebGPU baseline — was renamed to `tinygpt bench-train` when this
module shipped.)

## Adding a new engine

The protocol surface lives in `EngineAdapter.swift`:

```swift
public protocol EngineAdapter {
    var name: String { get }
    var commitHash: String? { get }
    mutating func load(modelPath: String) throws
    func parameterCount() -> Int
    mutating func prefill(tokens: [Int32]) throws -> PrefillResult
    mutating func decode(maxTokens: Int, batchSize: Int) throws -> DecodeResult
    func peakResidentBytes() -> Int
    mutating func reset()
}
```

Foreign engines (MLX-LM, llama.cpp, MLC-LLM, Ollama) will be
subprocess wrappers: `Process.launchPath = "/path/to/engine-binary"`,
parse per-token markers from stdout, attribute timestamps. Specifics:

- **MLX-LM**: `mlx_lm.generate --max-tokens N --temp 0 --prompt …
  --verbose` — captures per-token timing in its verbose output.
- **llama.cpp**: `./llama-cli -m model.gguf -n N --simple-io -p …`
  with `--logit-bias 0` so sampling is deterministic. Per-token
  markers via `--token-callback`.
- **Ollama**: HTTP POST against the local daemon
  (`http://localhost:11434/api/generate`) with `stream: true`. Each
  JSON chunk's wall-clock = one token's arrival timestamp.
- **MLC-LLM**: `mlc_llm chat … --benchmark` — emits a JSON report;
  parse it.

For each engine, also record the engine's own commit hash (shell out
to `--version` or `git -C engine-dir rev-parse HEAD`) so the JSON
output is fully attributable.

## What the harness measures

| Metric | How |
|---|---|
| **TTFT** | timer around `engine.prefill()` |
| **ITL** | per-token timer inside `engine.decode()` |
| **decode tok/s** | `gen_tokens / total_decode_seconds` (excludes prefill) |
| **prefill tok/s** | `prompt_tokens / TTFT_seconds` |
| **peak RSS** | `task_info(TASK_VM_INFO).phys_footprint` high-water mark |
| **energy/token** | `∫(ane+gpu+cpu) dt` over decode window ÷ tokens, via `powermetrics` |
| **ANE residency %** | fraction of decode time with `ane_power > 50 mW` |

See the design doc §4 for exact definitions.

## Known limitations (May 2026 scaffold)

- `--mode server` and `--mode sustained` are placeholders. The
  controller accepts them but emits a warning and falls back. Real
  implementations are the next milestone.
- Foreign engines are stubs. Only `--engine tinygpt` produces numbers.
- `powermetrics` requires sudo. If not available the harness logs a
  warning and skips energy/ANE; CPU watts via task_info still work.
- TinyGPT itself runs B=1 even when `--batch-size > 1` — the model's
  KV cache is B=1 today. The harness reports what actually ran rather
  than over-claiming throughput.
- No long-context (RULER / LongBench) driver yet. Hooks in the metrics
  catalog; driver pending.

## Citations

- Bench360 — arXiv 2511.16682 (Nov 2025): modular architecture we
  copied (task engine + workload controller + backend abstraction +
  metrics collector).
- TokenPowerBench — arXiv 2512.03024 (Dec 2025, AAAI): the
  energy/token definition adapted to unified-memory `powermetrics`.
- "Production-Grade Local LLM Inference on Apple Silicon" — arXiv
  2511.05502 (Nov 2025): the prior art this harness engages with.
- Orion (referenced in 2511.05502): proof that ANE is reachable from
  user code, motivating the ANE-residency metric.

Full survey: [`../../../docs/research/inference_benchmarks_may_2026.md`](../../../docs/research/inference_benchmarks_may_2026.md).
