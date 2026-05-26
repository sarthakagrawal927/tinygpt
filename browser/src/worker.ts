/**
 * worker.ts — training Web Worker (Phase 4-5).
 *
 * Runs the whole training loop off the main thread, so the UI never freezes.
 * Two backends:
 *   - "wasm"   — the C++ TinyGPT compiled to WebAssembly (default; supports
 *                checkpointing).
 *   - "webgpu" — the GPU model in webgpu/gpu_model.ts (faster on real GPUs).
 *
 * The loop runs in small chunks and yields between them so pause / stop are
 * handled promptly. Single-threaded by design.
 *
 * Guide: docs/browser_notes.md, docs/performance.md
 */

import { GpuModel } from "../../webgpu/gpu_model";
import { createGpuContext } from "../../webgpu/tensor";
import { TinyGptBackend, TinyGptModel } from "./backend";
import { decode, encode } from "./tokenizer";
import type { FromWorker, RunConfig, ToWorker } from "./types";

const ctx = self as unknown as {
  postMessage(msg: FromWorker, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null;
};

let backend: TinyGptBackend | null = null;
let model: TinyGptModel | null = null; // WASM model
let gpuModel: GpuModel | null = null; // WebGPU model
let paused = false;
let stopped = false;
let training = false;

// Tracking for "continue training" — keep the last successful run's setup
// so a subsequent +N-steps call can pick up where we left off.
let lastCfg: RunConfig | null = null;
let lastTokens: Uint8Array | null = null;
let lastStep = 0;

const post = (msg: FromWorker, transfer?: Transferable[]) =>
  ctx.postMessage(msg, transfer);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

ctx.onmessage = (e: MessageEvent<ToWorker>) => {
  const msg = e.data;
  switch (msg.type) {
    case "train": void runTraining(msg.text, msg.config); break;
    case "continue": void runContinue(msg.extraSteps); break;
    case "pause": paused = true; break;
    case "resume": paused = false; break;
    case "stop": stopped = true; break;
    case "sample": void doSample(msg.prompt, msg.tokens, msg.temperature); break;
    case "restore": void doRestore(msg.state, msg.config); break;
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
    if (cfg.backend === "webgpu") {
      await runWebGpu(text, cfg);
    } else {
      await runWasm(text, cfg);
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  } finally {
    training = false;
  }
}

// --- WASM backend ---------------------------------------------------------
async function runWasm(text: string, cfg: RunConfig): Promise<void> {
  if (gpuModel) gpuModel = null;
  if (!backend) {
    post({ type: "status", message: "loading WASM backend…" });
    backend = await TinyGptBackend.load();
  }
  if (model) { model.free(); model = null; }

  const tokens = encode(text);
  if (tokens.length < cfg.ctx + 2) {
    post({ type: "error", message: `corpus is ${tokens.length} bytes — need > ${cfg.ctx + 2}` });
    return;
  }
  model = backend.createModel({
    ctx: cfg.ctx, layers: cfg.layers, heads: cfg.heads,
    dModel: cfg.dModel, dMlp: cfg.dMlp, seed: cfg.seed,
  });
  model.setData(tokens, 0.9);
  post({
    type: "status",
    message: `${tokens.length.toLocaleString()} tokens · ${model
      .numParams().toLocaleString()} params · training on WASM`,
  });

  const evalFor = (split: 0 | 1) => model!.evalLoss(split, cfg.batchSize, 5);
  post({
    type: "progress",
    progress: {
      step: 0, maxSteps: cfg.maxSteps, trainLoss: evalFor(0),
      valLoss: evalFor(1), tokensPerSecond: 0, backend: "wasm",
    },
  });

  const t0 = performance.now();
  let tokensProcessed = 0;
  let nextEval = cfg.evalEvery;
  const chunk = 8;
  let step = 0;
  while (step < cfg.maxSteps && !stopped) {
    if (paused) { await sleep(60); continue; }
    let trainLoss = 0;
    for (let i = 0; i < chunk && step < cfg.maxSteps; i++) {
      trainLoss = model.trainStep(cfg.batchSize, cfg.learningRate, cfg.gradClip);
      tokensProcessed += cfg.batchSize * cfg.ctx;
      step++;
    }
    if (step >= nextEval || step >= cfg.maxSteps) {
      const elapsed = (performance.now() - t0) / 1000;
      post({
        type: "progress",
        progress: {
          step, maxSteps: cfg.maxSteps, trainLoss, valLoss: evalFor(1),
          tokensPerSecond: elapsed > 0 ? tokensProcessed / elapsed : 0,
          backend: "wasm",
        },
      });
      nextEval += cfg.evalEvery;
    }
    await sleep(0);
  }
  const state = model.exportState();
  post({ type: "checkpoint", state: state.buffer as ArrayBuffer }, [state.buffer as ArrayBuffer]);
  // Remember everything needed to continue from here.
  lastCfg = cfg;
  lastTokens = tokens;
  lastStep = step;
  post({ type: "done", reason: stopped ? "stopped" : "finished" });
}

// --- WebGPU backend -------------------------------------------------------
async function runWebGpu(text: string, cfg: RunConfig): Promise<void> {
  const gpuCtx = await createGpuContext();
  if (!gpuCtx) {
    post({ type: "status", message: "WebGPU unavailable — using WASM instead" });
    await runWasm(text, cfg);
    return;
  }
  if (model) { model.free(); model = null; }

  const tokens = encode(text);
  if (tokens.length < cfg.ctx + 2) {
    post({ type: "error", message: `corpus is ${tokens.length} bytes — need > ${cfg.ctx + 2}` });
    return;
  }
  gpuModel = new GpuModel(gpuCtx, {
    vocab: 256, ctx: cfg.ctx, layers: cfg.layers, heads: cfg.heads,
    dModel: cfg.dModel, dMlp: cfg.dMlp, seed: cfg.seed,
  });
  post({
    type: "status",
    message: `${tokens.length.toLocaleString()} tokens · ${gpuModel
      .numParams().toLocaleString()} params · training on WebGPU`,
  });

  const maxStart = tokens.length - cfg.ctx - 1;
  const sampleBatch = () => {
    const ids = new Float32Array(cfg.batchSize * cfg.ctx);
    const targets = new Float32Array(cfg.batchSize * cfg.ctx);
    for (let b = 0; b < cfg.batchSize; b++) {
      const s = Math.floor(Math.random() * (maxStart + 1));
      for (let t = 0; t < cfg.ctx; t++) {
        ids[b * cfg.ctx + t] = tokens[s + t];
        targets[b * cfg.ctx + t] = tokens[s + t + 1];
      }
    }
    return { ids, targets };
  };

  const t0 = performance.now();
  let tokensProcessed = 0;
  let step = 0;
  const chunk = 4;
  while (step < cfg.maxSteps && !stopped) {
    if (paused) { await sleep(60); continue; }
    let trainLoss = 0;
    for (let i = 0; i < chunk && step < cfg.maxSteps; i++) {
      const { ids, targets } = sampleBatch();
      trainLoss = await gpuModel.trainStep(ids, targets, cfg.batchSize, cfg.learningRate);
      tokensProcessed += cfg.batchSize * cfg.ctx;
      step++;
    }
    const elapsed = (performance.now() - t0) / 1000;
    post({
      type: "progress",
      progress: {
        step, maxSteps: cfg.maxSteps, trainLoss,
        tokensPerSecond: elapsed > 0 ? tokensProcessed / elapsed : 0,
        backend: "webgpu",
      },
    });
    await sleep(0);
  }
  // The WebGPU model has no checkpoint serialization yet — survives-refresh
  // stays a WASM-backend feature.
  lastCfg = cfg;
  lastTokens = tokens;
  lastStep = step;
  post({ type: "done", reason: stopped ? "stopped" : "finished" });
}

/**
 * Continue training the existing in-memory model for `extraSteps` more steps,
 * starting from `lastStep`. Same data, same config — only the step budget
 * changes. Sends progress messages indexed against the new total so the chart
 * keeps the same x-axis.
 */
async function runContinue(extraSteps: number): Promise<void> {
  if (training) {
    post({ type: "error", message: "a run is already in progress" });
    return;
  }
  if (!lastCfg || !lastTokens) {
    post({ type: "error", message: "no prior run to continue — start a fresh one first" });
    return;
  }
  if (extraSteps <= 0) {
    post({ type: "error", message: "extra steps must be positive" });
    return;
  }
  training = true;
  stopped = false;
  paused = false;
  const startStep = lastStep;
  const newTotal = startStep + extraSteps;
  try {
    if (lastCfg.backend === "wasm") {
      if (!model) throw new Error("no WASM model in memory");
      await continueWasm(extraSteps, startStep, newTotal);
    } else {
      if (!gpuModel) throw new Error("no WebGPU model in memory");
      await continueWebgpu(extraSteps, startStep, newTotal);
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  } finally {
    training = false;
  }
}

async function continueWasm(extraSteps: number, startStep: number, newTotal: number): Promise<void> {
  if (!model || !lastCfg) return;
  const cfg = lastCfg;
  post({ type: "status", message: `continuing for ${extraSteps} more steps on WASM…` });

  const evalFor = (split: 0 | 1) => model!.evalLoss(split, cfg.batchSize, 5);
  const t0 = performance.now();
  let tokensProcessed = 0;
  let nextEval = startStep + cfg.evalEvery;
  const chunk = 8;
  let step = startStep;

  while (step < newTotal && !stopped) {
    if (paused) { await sleep(60); continue; }
    let trainLoss = 0;
    for (let i = 0; i < chunk && step < newTotal; i++) {
      trainLoss = model.trainStep(cfg.batchSize, cfg.learningRate, cfg.gradClip);
      tokensProcessed += cfg.batchSize * cfg.ctx;
      step++;
    }
    if (step >= nextEval || step >= newTotal) {
      const elapsed = (performance.now() - t0) / 1000;
      post({
        type: "progress",
        progress: {
          step, maxSteps: newTotal, trainLoss, valLoss: evalFor(1),
          tokensPerSecond: elapsed > 0 ? tokensProcessed / elapsed : 0,
          backend: "wasm",
        },
      });
      nextEval += cfg.evalEvery;
    }
    await sleep(0);
  }
  const state = model.exportState();
  post({ type: "checkpoint", state: state.buffer as ArrayBuffer }, [state.buffer as ArrayBuffer]);
  lastStep = step;
  // Update lastCfg.maxSteps so the next "continue" extends from this new total.
  lastCfg = { ...cfg, maxSteps: newTotal };
  post({ type: "done", reason: stopped ? "stopped" : "finished" });
}

async function continueWebgpu(extraSteps: number, startStep: number, newTotal: number): Promise<void> {
  if (!gpuModel || !lastCfg || !lastTokens) return;
  const cfg = lastCfg;
  const tokens = lastTokens;
  post({ type: "status", message: `continuing for ${extraSteps} more steps on WebGPU…` });

  const maxStart = tokens.length - cfg.ctx - 1;
  const sampleBatch = () => {
    const ids = new Float32Array(cfg.batchSize * cfg.ctx);
    const targets = new Float32Array(cfg.batchSize * cfg.ctx);
    for (let b = 0; b < cfg.batchSize; b++) {
      const s = Math.floor(Math.random() * (maxStart + 1));
      for (let t = 0; t < cfg.ctx; t++) {
        ids[b * cfg.ctx + t] = tokens[s + t];
        targets[b * cfg.ctx + t] = tokens[s + t + 1];
      }
    }
    return { ids, targets };
  };

  const t0 = performance.now();
  let tokensProcessed = 0;
  let step = startStep;
  const chunk = 4;
  while (step < newTotal && !stopped) {
    if (paused) { await sleep(60); continue; }
    let trainLoss = 0;
    for (let i = 0; i < chunk && step < newTotal; i++) {
      const { ids, targets } = sampleBatch();
      trainLoss = await gpuModel.trainStep(ids, targets, cfg.batchSize, cfg.learningRate);
      tokensProcessed += cfg.batchSize * cfg.ctx;
      step++;
    }
    const elapsed = (performance.now() - t0) / 1000;
    post({
      type: "progress",
      progress: {
        step, maxSteps: newTotal, trainLoss,
        tokensPerSecond: elapsed > 0 ? tokensProcessed / elapsed : 0,
        backend: "webgpu",
      },
    });
    await sleep(0);
  }
  lastStep = step;
  lastCfg = { ...cfg, maxSteps: newTotal };
  post({ type: "done", reason: stopped ? "stopped" : "finished" });
}

async function doSample(prompt: string, tokens: number, temperature: number): Promise<void> {
  const seed = (Date.now() & 0xffff) >>> 0;
  if (gpuModel) {
    const out = await gpuModel.generate([...encode(prompt)], tokens, temperature, 40, seed);
    post({ type: "sample", text: decode(Uint8Array.from(out)) });
    return;
  }
  if (model) {
    const out = model.generate(encode(prompt), tokens, temperature, 40, seed);
    post({ type: "sample", text: prompt + decode(out) });
    return;
  }
  post({ type: "error", message: "train a model before sampling" });
}

// Rebuild a model from a saved checkpoint (WASM backend only).
async function doRestore(state: ArrayBuffer, cfg: RunConfig): Promise<void> {
  try {
    if (!backend) backend = await TinyGptBackend.load();
    if (model) { model.free(); model = null; }
    gpuModel = null;
    model = backend.createModel({
      ctx: cfg.ctx, layers: cfg.layers, heads: cfg.heads,
      dModel: cfg.dModel, dMlp: cfg.dMlp, seed: cfg.seed,
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
