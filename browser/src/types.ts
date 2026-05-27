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
  | { type: "inspect"; prompt: Uint8Array; topK: number };

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
  | { type: "inspect"; result: InspectResult };
