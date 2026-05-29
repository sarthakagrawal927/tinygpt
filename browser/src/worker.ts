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
import { createGpuContext, type GpuContext } from "../../webgpu/tensor";
import { TinyGptBackend, TinyGptModel } from "./backend";
import { decode, encode } from "./tokenizer";
import type { FromWorker, RunConfig, ToWorker } from "./types";
import { benchmarkById } from "./benchmarks/registry";
import { BenchmarkError, type BenchmarkModel } from "./benchmarks/types";

const ctx = self as unknown as {
  postMessage(msg: FromWorker, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null;
};

let backend: TinyGptBackend | null = null;
let model: TinyGptModel | null = null; // WASM model
let gpuModel: GpuModel | null = null; // WebGPU model
// Single shared WebGPU device for the worker. Each createGpuContext() call
// returns a fresh GPUDevice + adapter context + pipeline cache; without
// caching, every gallery reload allocated a new device while the previous
// one stayed alive in the GPU process. Now: lazily create once, reuse for
// every subsequent GpuModel construction.
let gpuCtx: GpuContext | null = null;
async function getGpuCtx(): Promise<GpuContext | null> {
  if (!gpuCtx) gpuCtx = await createGpuContext();
  return gpuCtx;
}
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
    case "inspect": void doInspect(msg.prompt, msg.topK); break;
    case "offload": offloadModel(); break;
    case "benchmark": void doBenchmark(msg.id); break;
    case "lens": void doLens(msg.prompt, msg.topK); break;
    case "ablate":
      void doAblate(msg.prompt, msg.tokens, msg.temperature, msg.ablations);
      break;
  }
};

/** Free the loaded model's GPU buffers. No-op if no model is loaded or
 *  if a training run is currently active (would break the in-flight
 *  forward/backward). Posts model_offloaded back so main can update its
 *  UI (hide GPU-mem pill, disable Generate, show "freed after idle" hint). */
function offloadModel(): void {
  if (training) return; // can't free under an active run
  disposeGpuModel();
  // Don't free the WASM-backed `model` — it's smaller, the page might still
  // want to use it, and the destroy path is different. Future work.
  post({ type: "model_offloaded" });
}

/** Drop the current GpuModel AND free its GPU buffers. Use everywhere we
 *  used to set `gpuModel = null` — bare null-assignment leaks because JS
 *  GC doesn't know about WebGPU buffer lifetimes; we need an explicit
 *  destroy() chain to release them back to the GPU. Leak audit found four
 *  call sites where the previous model would orphan ~110 MB each
 *  (gallery reload, training restart, WebGPU→WASM switch, restore fallback). */
function disposeGpuModel(): void {
  if (gpuModel) {
    gpuModel.destroy();
    gpuModel = null;
  }
}

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
  disposeGpuModel(); // user switched WebGPU → WASM; release the GPU buffers
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
  const gpuCtxLocal = await getGpuCtx();
  if (!gpuCtxLocal) {
    post({ type: "status", message: "WebGPU unavailable — using WASM instead" });
    await runWasm(text, cfg);
    return;
  }
  if (model) { model.free(); model = null; }
  // If we held a previous GpuModel (e.g., user loaded a gallery model then
  // hit "Start training"), free its ~110 MB of GPU buffers BEFORE building
  // the new one — otherwise both live in GPU memory until GC notices.
  disposeGpuModel();

  const tokens = encode(text);
  if (tokens.length < cfg.ctx + 2) {
    post({ type: "error", message: `corpus is ${tokens.length} bytes — need > ${cfg.ctx + 2}` });
    return;
  }
  gpuModel = new GpuModel(gpuCtxLocal, {
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
  // Periodic live sample during training — makes the *learning* visible the
  // way the loss curve makes the loss visible. Random chars → letter patterns
  // → words. Interval is chosen so sampling stays under ~5% overhead: roughly
  // every 8% of total training (max ~12 samples per run, min every 100 steps).
  const sampleEveryN = Math.max(100, Math.floor(cfg.maxSteps / 12));
  let nextSampleAt = sampleEveryN;
  // 64-token sample with a short empty-ish prompt — costs ~0.5-2s on Small
  // depending on hardware, dominated by per-token forward passes.
  const SAMPLE_TOKENS = 64;
  const SAMPLE_PROMPT_IDS = [10]; // single space, neutral prompt
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
    // Live sample at the threshold. Re-uses the existing generate path; no
    // KV cache, so this is N forward passes. The overhead is bounded by the
    // chunk schedule and the sampleEveryN spacing — by design < 5% of run time.
    if (step >= nextSampleAt && step < cfg.maxSteps) {
      const tokens = await gpuModel.generate(SAMPLE_PROMPT_IDS, SAMPLE_TOKENS, 0.8, 40, step);
      post({ type: "progress_sample", step, sample: decode(Uint8Array.from(tokens)) });
      nextSampleAt += sampleEveryN;
    }
    await sleep(0);
  }
  // Export trained weights so the user can download the model + survives-refresh
  // works for WebGPU too (same .tinygpt format the WASM backend produces).
  if (gpuModel) {
    const state = await gpuModel.exportState();
    post({ type: "checkpoint", state }, [state]);
    // Warm up the B=1 inference pipelines while the user is still reading the
    // "training complete" message — otherwise the first Generate click pays
    // a 10–60s WebGPU pipeline-compile cost (training used B=8; inference
    // bind-group layouts differ). One small forward triggers compilation of
    // every kernel against the inference shape; subsequent generations are
    // immediate.
    await warmupGenerate(gpuModel, cfg.ctx);
  }
  lastCfg = cfg;
  lastTokens = tokens;
  lastStep = step;
  post({ type: "done", reason: stopped ? "stopped" : "finished" });
}

async function warmupGenerate(g: GpuModel, ctx: number): Promise<void> {
  post({ type: "status", message: "warming up inference pipelines…" });
  const t0 = performance.now();
  // Opportunistic f16-storage path: pack every weight to f16 IF the
  // ops-level numerics gate has passed. Falls through silently if the gate
  // failed (we stay on the f32 vec4 path; user sees identical quality).
  // Doing this here (inside the warmup) means the very first generate()
  // hits the fast path with packed buffers already in place.
  const f16Active = await g.prepareForInference();
  if (f16Active) {
    // Surface to the main thread so the +f16 pill can update post-init.
    post({ type: "gpu_caps", caps: { f16Storage: true } });
  }
  // 32 tokens is a reasonable middle-ground prompt length. One forward at
  // T=32 compiles every inference kernel against bind-group layouts that
  // also cover smaller and bigger Ts (the kernels are shape-uniform; T
  // flows through as a uniform, not a layout parameter).
  const T = Math.min(32, ctx);
  const prompt = Array.from({ length: T }, (_, i) => 10 + (i % 90));
  await g.generate(prompt, 1, 0, 0, 0);
  const note = f16Active ? " · f16-storage matmul active" : "";
  post({ type: "status", message: `inference warmed up in ${((performance.now() - t0) / 1000).toFixed(1)}s${note} — ready to generate.` });
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
  if (gpuModel) {
    const state = await gpuModel.exportState();
    post({ type: "checkpoint", state }, [state]);
    await warmupGenerate(gpuModel, cfg.ctx);
  }
  lastStep = step;
  lastCfg = { ...cfg, maxSteps: newTotal };
  post({ type: "done", reason: stopped ? "stopped" : "finished" });
}

async function doSample(prompt: string, tokens: number, temperature: number): Promise<void> {
  const seed = (Date.now() & 0xffff) >>> 0;
  if (gpuModel) {
    // Stream per-token via the onToken callback so the UI can render output
    // live instead of waiting for the whole sequence and then animating a
    // typewriter on already-finished text. Time the loop so we can report
    // a real tokens-per-second number when the run finishes.
    const promptIds = [...encode(prompt)];
    post({ type: "sample_begin", prompt });
    const tStart = performance.now();
    let firstTokenMs = 0;
    const generated: number[] = [];
    const out = await gpuModel.generate(promptIds, tokens, temperature, 40, seed, (tok, idx) => {
      if (idx === 0) firstTokenMs = performance.now() - tStart;
      generated.push(tok);
      // Decoding one byte in isolation can split UTF-8 sequences, so we
      // re-decode the running tail on every chunk — small string,
      // negligible cost compared to a full forward pass.
      const chunkText = decode(Uint8Array.from(generated));
      post({ type: "sample_chunk", chunk: chunkText, count: generated.length });
    });
    const elapsed = (performance.now() - tStart) / 1000;
    const tokensPerSecond = elapsed > 0 ? out.length / elapsed : 0;
    post({
      type: "sample_done",
      text: decode(Uint8Array.from(out)),
      tokensPerSecond,
      firstTokenMs,
      totalMs: elapsed * 1000,
    });
    return;
  }
  if (model) {
    // WASM path is synchronous — no streaming hook in the C ABI. Still time
    // it and emit a tokens-per-second number so the UI doesn't go dark.
    const tStart = performance.now();
    const out = model.generate(encode(prompt), tokens, temperature, 40, seed);
    const elapsed = (performance.now() - tStart) / 1000;
    const text = prompt + decode(out);
    const tokensPerSecond = elapsed > 0 ? out.length / elapsed : 0;
    post({ type: "sample_done", text, tokensPerSecond, firstTokenMs: 0, totalMs: elapsed * 1000 });
    return;
  }
  post({ type: "error", message: "train a model before sampling" });
}

/**
 * Run one introspection forward pass and post the result back to the UI.
 *
 * WebGPU-only for the first cut — the WASM backend doesn't expose attention
 * weights yet (it would need a sibling `tg_inspect` export). When called with
 * a WASM-only model in memory, we still return top-K probs by re-using the
 * existing generate path with maxNew=0… but that's also not exposed, so for
 * now we send back `unavailable` and the UI shows a friendly note.
 */
async function doInspect(prompt: Uint8Array, topK: number): Promise<void> {
  if (gpuModel) {
    try {
      const result = await gpuModel.inspect([...prompt], topK);
      // Transfer the underlying buffers to keep the post-message cheap.
      const transfers: Transferable[] = [];
      for (const row of result.attention) {
        for (const arr of row) transfers.push(arr.buffer);
      }
      post({
        type: "inspect",
        result: {
          tokens: result.tokens,
          topK: result.topK,
          attention: result.attention,
          heads: gpuModel.cfg.heads,
        },
      }, transfers);
    } catch (err) {
      post({ type: "error", message: err instanceof Error ? err.message : String(err) });
    }
    return;
  }
  if (model) {
    post({
      type: "inspect",
      result: {
        tokens: [...prompt],
        topK: [],
        attention: [],
        heads: 0,
        unavailable: "Switch the backend to WebGPU and re-train to see introspection — the WASM build doesn't expose attention weights yet.",
      },
    });
    return;
  }
  post({ type: "error", message: "train a model before inspecting" });
}

// Rebuild a model from a saved checkpoint. Prefers the WebGPU path when the
// saved config says backend: "webgpu" AND the host has WebGPU — that gives
// the loaded model streaming generation + lower TTFT, vs the synchronous
// per-token WASM path. Falls back to WASM if WebGPU isn't available, so
// older Safari etc. still works.
async function doRestore(state: ArrayBuffer, cfg: RunConfig): Promise<void> {
  try {
    // Try WebGPU first if the saved file was a WebGPU run.
    if (cfg.backend === "webgpu") {
      try {
        const gpuCtxLocal = await getGpuCtx();
        if (!gpuCtxLocal) throw new Error("WebGPU adapter not available");
        if (model) { model.free(); model = null; }
        // Free the previous gallery model's GPU buffers before loading the new
        // one — otherwise back-to-back gallery loads accumulate ~110 MB each.
        disposeGpuModel();
        gpuModel = new GpuModel(gpuCtxLocal, {
          vocab: 256, ctx: cfg.ctx, layers: cfg.layers, heads: cfg.heads,
          dModel: cfg.dModel, dMlp: cfg.dMlp, seed: cfg.seed,
        });
        gpuModel.importState(state);
        post({ type: "restored" });
        post({ type: "status", message: "restored on WebGPU — warming up inference pipelines…" });
        // Warm up the B=1 inference pipelines so the first Generate click
        // doesn't pay the 10–60s pipeline-compile cost. Cheap (~one forward).
        // warmupGenerate posts its own final "ready to generate." status —
        // which carries the optional " · f16-storage matmul active" suffix
        // when the fast path activates. Don't overwrite it here.
        await warmupGenerate(gpuModel, cfg.ctx);
        return;
      } catch (gpuErr) {
        // GPU restore failed (device unavailable, OOM, shape mismatch).
        // Fall through to WASM so the user at least gets something.
        post({ type: "status", message: `WebGPU restore failed (${gpuErr instanceof Error ? gpuErr.message : String(gpuErr)}); falling back to WASM` });
      }
    }
    // WASM fallback path.
    if (!backend) backend = await TinyGptBackend.load();
    if (model) { model.free(); model = null; }
    // WebGPU restore failed; free any partial GPU state before falling back.
    disposeGpuModel();
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

/**
 * Benchmark runner — adapts the currently-loaded model to the
 * `BenchmarkModel` interface (encode/decode + forwardLogits/generate)
 * and invokes the requested benchmark.
 *
 * WebGPU-only for the first cut. WASM-backed models don't currently
 * expose a full `forwardLogits` (no `tg_forward_logits` export yet),
 * so we return `benchmark_skipped` with a clear reason so the UI can
 * point users to the WebGPU backend instead of failing red.
 */
/**
 * Logit lens — per-layer top-K predictions. Pure WebGPU today (the
 * WASM model doesn't expose per-layer hidden states). When called
 * with a WASM-only model we emit an `unavailable` payload so the UI
 * shows a friendly note instead of failing red.
 */
async function doLens(prompt: Uint8Array, topK: number): Promise<void> {
  if (!gpuModel) {
    post({
      type: "lens",
      result: {
        tokens: [...prompt],
        layers: [],
        unavailable: model
          ? "Logit lens requires the WebGPU backend — switch backends and reload."
          : "Train or load a model first.",
      },
    });
    return;
  }
  try {
    const perLayer = await gpuModel.logitLens([...prompt]);
    const tokens = [...prompt];
    const V = gpuModel.cfg.vocab;
    // For each layer, softmax + top-K per input position.
    const layers: { token: number; prob: number }[][][] = [];
    for (const logits of perLayer) {
      const T = tokens.length;
      const perPos: { token: number; prob: number }[][] = [];
      for (let t = 0; t < T; t++) {
        const base = t * V;
        let mx = -1e30;
        for (let v = 0; v < V; v++) if (logits[base + v] > mx) mx = logits[base + v];
        const probs = new Float64Array(V);
        let sum = 0;
        for (let v = 0; v < V; v++) {
          const p = Math.exp(logits[base + v] - mx);
          probs[v] = p; sum += p;
        }
        for (let v = 0; v < V; v++) probs[v] /= sum;
        const indexed = Array.from(probs, (p, v) => ({ token: v, prob: p }));
        indexed.sort((a, b) => b.prob - a.prob);
        perPos.push(indexed.slice(0, topK));
      }
      layers.push(perPos);
    }
    post({ type: "lens", result: { tokens, layers } });
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
}

/**
 * Ablation generator — re-runs sampling with specific layer
 * components zeroed out. Sets GpuModel's ablation flags, generates,
 * then clears the flags so subsequent normal samples are uncorrupted.
 */
async function doAblate(
  prompt: string,
  tokens: number,
  temperature: number,
  ablations: { layer: number; target: "attn" | "mlp" | "all" }[],
): Promise<void> {
  if (!gpuModel) {
    post({ type: "ablate_failed", message: "Ablation requires the WebGPU backend." });
    return;
  }
  try {
    const seed = (Date.now() & 0xffff) >>> 0;
    const promptIds = [...encode(prompt)];
    const out = await gpuModel.generateAblated(
      promptIds, ablations, tokens, temperature, 40, seed,
    );
    post({
      type: "ablate_done",
      text: prompt + decode(Uint8Array.from(out.slice(promptIds.length))),
      ablations: ablations.map((a) => ({ layer: a.layer, target: a.target })),
    });
  } catch (err) {
    post({ type: "ablate_failed", message: err instanceof Error ? err.message : String(err) });
  }
}

async function doBenchmark(id: string): Promise<void> {
  const bench = benchmarkById(id);
  if (!bench) {
    post({ type: "benchmark_failed", id, message: `no benchmark with id '${id}'` });
    return;
  }
  if (!gpuModel) {
    post({
      type: "benchmark_skipped", id,
      reason: model
        ? "WASM backend doesn't expose forwardLogits yet — switch to WebGPU and re-load the model."
        : "no model loaded — train or load a gallery model first.",
    });
    return;
  }

  // Build the adapter. The browser is byte-level today; encode/decode go
  // through the shared tokenizer. forwardLogits / generate route to the
  // GpuModel paths added for this and existing for sample respectively.
  const gm = gpuModel;
  const adapter: BenchmarkModel = {
    vocabSize: gm.cfg.vocab,
    contextLength: gm.cfg.ctx,
    encode: (text) => [...encode(text)],
    decode: (ids) => decode(Uint8Array.from(ids)),
    forwardLogits: (ids) => gm.forwardLogits(ids),
    generate: async (prompt, maxNew, temperature) => {
      const promptIds = [...encode(prompt)];
      const seed = (Date.now() & 0xffff) >>> 0;
      const out = await gm.generate(promptIds, maxNew, temperature, 40, seed);
      return decode(Uint8Array.from(out));
    },
  };

  const t0 = performance.now();
  try {
    const result = await bench.run(adapter);
    const wallSeconds = result.wallSeconds ?? (performance.now() - t0) / 1000;
    post({
      type: "benchmark_done", id, score: result.score,
      details: result.details, wallSeconds,
    });
  } catch (err) {
    if (err instanceof BenchmarkError) {
      if (err.kind === "incompatible") {
        post({ type: "benchmark_skipped", id, reason: err.message });
      } else {
        post({ type: "benchmark_failed", id, message: err.message });
      }
      return;
    }
    post({
      type: "benchmark_failed", id,
      message: err instanceof Error ? err.message : String(err),
    });
  }
}
