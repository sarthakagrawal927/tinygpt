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

const post = (msg: FromWorker, transfer?: Transferable[]) =>
  ctx.postMessage(msg, transfer);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

ctx.onmessage = (e: MessageEvent<ToWorker>) => {
  const msg = e.data;
  switch (msg.type) {
    case "train": void runTraining(msg.text, msg.config); break;
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
