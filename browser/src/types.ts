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
}

export const DEFAULT_CONFIG: RunConfig = {
  ctx: 64,
  layers: 3,
  heads: 3,
  dModel: 96,
  dMlp: 384,
  batchSize: 16,
  learningRate: 3e-3,
  gradClip: 1.0,
  maxSteps: 1500,
  evalEvery: 50,
  seed: 42,
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
  | { type: "pause" }
  | { type: "resume" }
  | { type: "sample"; prompt: string; tokens: number; temperature: number }
  | { type: "stop" }
  | { type: "restore"; state: ArrayBuffer; config: RunConfig };

/** worker -> main */
export type FromWorker =
  | { type: "status"; message: string }
  | { type: "progress"; progress: TrainingProgress }
  | { type: "sample"; text: string }
  | { type: "checkpoint"; state: ArrayBuffer } // serialized model state for OPFS
  | { type: "restored" } // a saved model was reloaded into the worker
  | { type: "done"; reason: "finished" | "stopped" }
  | { type: "error"; message: string };
