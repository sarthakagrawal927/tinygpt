// bench_wasm.mjs — measure the compiled WASM module's training speed.
//
// The WASM-side counterpart of python_ref/bench.py: it loads
// browser/public/tinygpt.js and times real training steps, so the effect of a
// kernel change (allocation reuse, SIMD, ...) can be measured precisely from
// the command line — no browser needed.
//
// Run:  node tests/bench_wasm.mjs   (after wasm/build_wasm.sh)

import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const { default: createTinyGPT } = await import(
  path.join(here, "..", "browser", "public", "tinygpt.js")
);
const M = await createTinyGPT();

const N = "number";
const create = M.cwrap("tg_model_create", N, [N, N, N, N, N, N, N]);
const numParams = M.cwrap("tg_model_num_params", N, [N]);
const setData = M.cwrap("tg_set_data", null, [N, N, N, N]);
const trainStep = M.cwrap("tg_train_step", N, [N, N, N, N]);
const freeModel = M.cwrap("tg_model_free", null, [N]);

// One corpus, reused for every config.
const text = "the quick brown fox jumps over the lazy dog. ".repeat(400);
const bytes = new TextEncoder().encode(text);
const dataPtr = M._malloc(bytes.length);
M.HEAPU8.set(bytes, dataPtr);

// (label, ctx, layers, d_model, batch) — the in-browser recommendation tiers
// + the bigger sizes that exercise the cache-tiling work.
const CONFIGS = [
  ["small",    64, 3,  96, 16],
  ["medium",   96, 4, 128, 16],
  ["large",   128, 6, 192, 12],
  ["xl",      128, 8, 256,  8],
];
const WARMUP = 3;
const TIMED = 15;

console.log(`${"config".padEnd(10)} ${"params".padStart(9)} ${"ms/step".padStart(9)} ${"tok/s".padStart(10)}`);
console.log("-".repeat(42));
for (const [label, ctx, layers, dModel, batch] of CONFIGS) {
  const heads = dModel >= 256 ? dModel / 32 : (dModel === 192 ? 6 : dModel === 128 ? 4 : 3);
  const model = create(256, ctx, layers, heads, dModel, dModel * 4, 42);
  setData(model, dataPtr, bytes.length, 0.9);
  for (let i = 0; i < WARMUP; i++) trainStep(model, batch, 3e-3, 1.0);
  const t0 = performance.now();
  for (let i = 0; i < TIMED; i++) trainStep(model, batch, 3e-3, 1.0);
  const msPerStep = (performance.now() - t0) / TIMED;
  const params = numParams(model);
  freeModel(model);
  console.log(
    `${label.padEnd(10)} ${(params / 1e6).toFixed(2).padStart(8)}M ` +
    `${msPerStep.toFixed(1).padStart(9)} ${Math.round(batch * ctx / (msPerStep / 1000)).toLocaleString().padStart(10)}`,
  );
}
M._free(dataPtr);
