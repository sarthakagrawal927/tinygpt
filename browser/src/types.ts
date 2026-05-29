/**
 * types.ts — the message protocol between the UI (main.ts) and the training
 * Web Worker (worker.ts), plus the shared model/training config.
 *
 * Guide: docs/browser_notes.md ("Web Worker")
 */

export type Backend = "wasm" | "wasm-simd" | "webgpu";

/** Model + training hyperparameters chosen in the UI. Small by default so a
 *  run overfits visibly in the browser within seconds. */
export interface RunConfig {
  ctx: number;
  layers: number;
  heads: number;
  dModel: number;
  dMlp: number;
  batchSize: number;
  learningRate: number;
  gradClip: number;
  maxSteps: number;
  evalEvery: number;
  seed: number;
  /** Which compute backend trains the model. */
  backend: "wasm" | "webgpu";
}

export const DEFAULT_CONFIG: RunConfig = {
  ctx: 64,
  layers: 3,
  heads: 3,
  dModel: 96,
  dMlp: 384,
  batchSize: 16,
  learningRate: 3e-4,
  gradClip: 1.0,
  maxSteps: 1500,
  evalEvery: 50,
  seed: 42,
  backend: "wasm",
};

/** Posted to the UI on every eval interval — see docs/browser_notes.md. */
export interface TrainingProgress {
  step: number;
  maxSteps: number;
  trainLoss: number;
  valLoss?: number;
  tokensPerSecond: number;
  backend: Backend;
}

/** main -> worker */
export type ToWorker =
  | { type: "train"; text: string; config: RunConfig }
  | { type: "continue"; extraSteps: number }
  | { type: "pause" }
  | { type: "resume" }
  | { type: "sample"; prompt: string; tokens: number; temperature: number }
  | { type: "stop" }
  | { type: "restore"; state: ArrayBuffer; config: RunConfig }
  // "Watch the model think" — single introspection forward over `prompt`.
  // Returns top-K next-token probabilities per position + last-layer attention.
  | { type: "inspect"; prompt: Uint8Array; topK: number }
  // Auto-offload: main thread asks worker to free the loaded model's GPU
  // buffers after N minutes of inactivity. No-op when no model is loaded
  // or when training is in flight. Worker replies with "model_offloaded"
  // when teardown completes (or omits the reply if nothing to free).
  | { type: "offload" }
  // Benchmark runner — main thread asks worker to score the loaded model
  // against a registered benchmark (see `benchmarks/registry.ts`). Worker
  // adapts its model handle to the BenchmarkModel interface and runs it.
  | { type: "benchmark"; id: string }
  // Logit lens — interpretability tool. Worker runs forward over `prompt`
  // and returns per-layer top-K predictions (what the model "would say"
  // if it stopped at each depth).
  | { type: "lens"; prompt: Uint8Array; topK: number }
  // Ablation tool — re-runs generation with specified components zeroed
  // out, returning the resulting text. Used to study "what does this
  // layer's attention contribute?" or "is this block load-bearing?".
  | {
      type: "ablate"; prompt: string; tokens: number; temperature: number;
      ablations: { layer: number; target: "attn" | "mlp" | "all" }[];
    };

/** Logit-lens output: one layer-slot per transformer block. Each slot
 * has the top-K (token, prob) predictions at every input position,
 * the projection of THAT layer's residual stream through the LM head.
 * Useful for "when does the model learn X?" interpretability studies. */
export interface LensResult {
  tokens: number[];
  /** Per-layer: per-position top-K (token, prob), descending. */
  layers: { token: number; prob: number }[][][];
  /** Set when the active backend can't produce a lens (WASM today). */
  unavailable?: string;
}

/** Per-position introspection payload (one entry per token in the inspect prompt). */
export interface InspectResult {
  /** byte tokens fed into the model (same length as topK / attention) */
  tokens: number[];
  /** top-K next-token candidates per position, sorted by prob desc */
  topK: { token: number; prob: number }[][];
  /** attention[t][h] = Float32Array(T) — last-layer head h's weights at position t */
  attention: Float32Array[][];
  /** number of heads in the last layer */
  heads: number;
  /** present only when the active backend can't produce introspection data */
  unavailable?: string;
}

/** worker -> main */
export type FromWorker =
  | { type: "status"; message: string }
  | { type: "progress"; progress: TrainingProgress }
  // Live sample emitted periodically during training so the user sees the
  // model's output evolving from random → words → sentences in real time.
  | { type: "progress_sample"; step: number; sample: string }
  // Generation flow: sample_begin (echo prompt) → sample_chunk* (live decode
  // updates) → sample_done (final text + tokens/sec). Legacy `sample` is kept
  // for the WASM path that doesn't stream per token.
  | { type: "sample"; text: string }
  | { type: "sample_begin"; prompt: string }
  | { type: "sample_chunk"; chunk: string; count: number }
  | { type: "sample_done"; text: string; tokensPerSecond: number; firstTokenMs: number; totalMs: number }
  | { type: "checkpoint"; state: ArrayBuffer } // serialized model state for OPFS
  | { type: "restored" } // a saved model was reloaded into the worker
  | { type: "done"; reason: "finished" | "stopped" }
  | { type: "error"; message: string }
  | { type: "inspect"; result: InspectResult }
  // Updates the capability pills with paths discovered post-device-init:
  // f16Storage is set once GpuModel.prepareForInference() returns true
  // (numerics gate passed AND weights packed). cooperativeMatrix will land
  // here when #92 ships. Sent at most once per loaded model.
  | { type: "gpu_caps"; caps: { f16Storage?: boolean; cooperativeMatrix?: boolean } }
  // Fires when the worker has destroyed its loaded model (auto-offload).
  // Main thread hides the GPU-mem pill + disables Generate + shows a small
  // "model freed after idle" toast with a "reload" affordance.
  | { type: "model_offloaded" }
  // Result of a "benchmark" request. `score` is the benchmark's primary
  // ranking metric; interpret against the benchmark's `lowerIsBetter`.
  // `kind: "incompatible"` covers vocab/architecture mismatch (skip,
  // don't penalize); `kind: "failed"` is a real failure (show red).
  | { type: "benchmark_done"; id: string; score: number; details?: Record<string, unknown>; wallSeconds: number }
  | { type: "benchmark_skipped"; id: string; reason: string }
  | { type: "benchmark_failed"; id: string; message: string }
  // Logit lens result. One entry per layer; each entry has the top-K
  // (token, prob) predictions per input position. nLayers = entries.length.
  | { type: "lens"; result: LensResult }
  | { type: "ablate_done"; text: string; ablations: { layer: number; target: string }[] }
  | { type: "ablate_failed"; message: string };
