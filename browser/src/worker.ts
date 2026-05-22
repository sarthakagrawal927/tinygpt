/**
 * worker.ts — training Web Worker (Phase 4).
 *
 * Runs the whole training loop off the main thread, so the UI never freezes:
 *   load WASM backend -> create model -> setData -> { trainStep } loop
 *                     -> post TrainingProgress to the UI
 *
 * The loop runs in small chunks and yields between them, so "pause" / "stop"
 * messages from the UI are processed promptly. Single-threaded by design —
 * threaded WASM (SharedArrayBuffer + COOP/COEP) is a later concern.
 *
 * Guide: docs/browser_notes.md ("Web Worker")
 */

import { TinyGptBackend, TinyGptModel } from "./backend";
import { decode, encode } from "./tokenizer";
import type { FromWorker, RunConfig, ToWorker, TrainingProgress } from "./types";

// Minimal typed view of the worker global — avoids DOM/WebWorker lib clashes.
const ctx = self as unknown as {
  postMessage(msg: FromWorker, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null;
};

let backend: TinyGptBackend | null = null;
let model: TinyGptModel | null = null;
let paused = false;
let stopped = false;
let training = false;

const post = (msg: FromWorker, transfer?: Transferable[]) =>
  ctx.postMessage(msg, transfer);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

ctx.onmessage = (e: MessageEvent<ToWorker>) => {
  const msg = e.data;
  switch (msg.type) {
    case "train":
      void runTraining(msg.text, msg.config);
      break;
    case "pause":
      paused = true;
      break;
    case "resume":
      paused = false;
      break;
    case "stop":
      stopped = true;
      break;
    case "sample":
      doSample(msg.prompt, msg.tokens, msg.temperature);
      break;
    case "restore":
      void doRestore(msg.state, msg.config);
      break;
  }
};

async function runTraining(text: string, cfg: RunConfig): Promise<void> {
  if (training) {
    post({ type: "error", message: "a run is already in progress" });
    return;
  }
  training = true;
  stopped = false;
  paused = false;
  try {
    if (!backend) {
      post({ type: "status", message: "loading WASM backend…" });
      backend = await TinyGptBackend.load();
    }
    if (model) {
      model.free();
      model = null;
    }

    const tokens = encode(text);
    if (tokens.length < cfg.ctx + 2) {
      post({ type: "error", message: `corpus is ${tokens.length} bytes — need > ${cfg.ctx + 2}` });
      return;
    }
    model = backend.createModel({
      ctx: cfg.ctx,
      layers: cfg.layers,
      heads: cfg.heads,
      dModel: cfg.dModel,
      dMlp: cfg.dMlp,
      seed: cfg.seed,
    });
    model.setData(tokens, 0.9);
    post({
      type: "status",
      message: `${tokens.length.toLocaleString()} tokens · ${model
        .numParams()
        .toLocaleString()} params · training`,
    });

    const evalFor = (split: 0 | 1) => model!.evalLoss(split, cfg.batchSize, 5);
    post({
      type: "progress",
      progress: {
        step: 0,
        maxSteps: cfg.maxSteps,
        trainLoss: evalFor(0),
        valLoss: evalFor(1),
        tokensPerSecond: 0,
        backend: "wasm",
      },
    });

    const t0 = performance.now();
    let tokensProcessed = 0;
    let nextEval = cfg.evalEvery;
    const chunk = 8; // steps between yields — keeps pause/stop responsive

    let step = 0;
    while (step < cfg.maxSteps && !stopped) {
      if (paused) {
        await sleep(60);
        continue;
      }
      let trainLoss = 0;
      for (let i = 0; i < chunk && step < cfg.maxSteps; i++) {
        trainLoss = model.trainStep(cfg.batchSize, cfg.learningRate, cfg.gradClip);
        tokensProcessed += cfg.batchSize * cfg.ctx;
        step++;
      }
      if (step >= nextEval || step >= cfg.maxSteps) {
        const elapsed = (performance.now() - t0) / 1000;
        const progress: TrainingProgress = {
          step,
          maxSteps: cfg.maxSteps,
          trainLoss,
          valLoss: evalFor(1),
          tokensPerSecond: elapsed > 0 ? tokensProcessed / elapsed : 0,
          backend: "wasm",
        };
        post({ type: "progress", progress });
        nextEval += cfg.evalEvery;
      }
      await sleep(0); // yield so queued pause/stop messages dispatch
    }
    // Checkpoint the trained model so it survives a page refresh.
    const state = model.exportState();
    post({ type: "checkpoint", state: state.buffer as ArrayBuffer }, [
      state.buffer as ArrayBuffer,
    ]);
    post({ type: "done", reason: stopped ? "stopped" : "finished" });
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  } finally {
    training = false;
  }
}

function doSample(prompt: string, tokens: number, temperature: number): void {
  if (!model) {
    post({ type: "error", message: "train a model before sampling" });
    return;
  }
  const seed = (Date.now() & 0xffff) >>> 0;
  const out = model.generate(encode(prompt), tokens, temperature, 40, seed);
  post({ type: "sample", text: prompt + decode(out) });
}

// Rebuild a model from a saved checkpoint (milestone 7 — survives a refresh).
async function doRestore(state: ArrayBuffer, config: RunConfig): Promise<void> {
  try {
    if (!backend) backend = await TinyGptBackend.load();
    if (model) {
      model.free();
      model = null;
    }
    model = backend.createModel({
      ctx: config.ctx,
      layers: config.layers,
      heads: config.heads,
      dModel: config.dModel,
      dMlp: config.dMlp,
      seed: config.seed,
    });
    model.importState(new Uint8Array(state));
    post({ type: "restored" });
    post({
      type: "status",
      message: "restored the model from your last run — generate, or start a new run",
    });
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
}
