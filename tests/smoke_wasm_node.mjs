// smoke_wasm_node.mjs — verify the compiled WASM module from Node (Phase 4).
//
// Loads browser/public/tinygpt.js (built by wasm/build_wasm.sh) and drives it
// through the exact JS<->WASM boundary the browser Worker uses: create a model,
// upload a corpus, train, and generate. If loss falls here, the compiled module
// trains correctly — independently of any browser.
//
// Run:  node tests/smoke_wasm_node.mjs

import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const modPath = path.join(here, "..", "browser", "public", "tinygpt.js");

const { default: createTinyGPT } = await import(modPath);
const M = await createTinyGPT();

const N = "number";
const create = M.cwrap("tg_model_create", N, [N, N, N, N, N, N, N]);
const numParams = M.cwrap("tg_model_num_params", N, [N]);
const setData = M.cwrap("tg_set_data", null, [N, N, N, N]);
const trainStep = M.cwrap("tg_train_step", N, [N, N, N, N]);
const evalLoss = M.cwrap("tg_eval", N, [N, N, N, N]);
const generate = M.cwrap("tg_generate", N, [N, N, N, N, N, N, N, N]);
const freeModel = M.cwrap("tg_model_free", null, [N]);

let failed = 0;
const check = (name, ok, detail) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name.padEnd(34)} (${detail})`);
  if (!ok) failed++;
};

// --- create model + upload a tiny repeated corpus -------------------------
const model = create(256, 32, 2, 2, 64, 128, 42);
check("tg_model_create", model !== 0, model);
const params = numParams(model);
check("tg_model_num_params", params > 0, params);

const text = "the quick brown fox jumps over the lazy dog. ".repeat(70);
const bytes = new TextEncoder().encode(text);
const dataPtr = M._malloc(bytes.length);
M.HEAPU8.set(bytes, dataPtr);
setData(model, dataPtr, bytes.length, 0.9);
M._free(dataPtr);

// --- train and watch loss collapse ---------------------------------------
const initLoss = evalLoss(model, 0, 8, 5);
check("initial loss ~ ln(256)", Math.abs(initLoss - 5.545) < 0.7, initLoss.toFixed(4));

let loss = initLoss;
for (let step = 1; step <= 400; step++) {
  loss = trainStep(model, 8, 1e-3, 1.0);
  if (step % 100 === 0) console.log(`    step ${step}  loss ${loss.toFixed(4)}`);
}
check("loss fell far below initial", loss < initLoss * 0.25, loss.toFixed(4));
check("tiny overfit (loss < 0.5)", loss < 0.5, loss.toFixed(4));

// --- generate ------------------------------------------------------------
const prompt = new TextEncoder().encode("the ");
const promptPtr = M._malloc(prompt.length);
M.HEAPU8.set(prompt, promptPtr);
const outPtr = M._malloc(70);
const produced = generate(model, promptPtr, prompt.length, outPtr, 70, 0.0, 0, 7);
const sample = new TextDecoder().decode(M.HEAPU8.slice(outPtr, outPtr + produced));
M._free(promptPtr);
M._free(outPtr);
check("generate produced bytes", produced === 70, produced);
console.log(`    greedy sample: "${sample}"`);

freeModel(model);
console.log(failed === 0 ? "\nWASM module smoke test passed" : "\nWASM SMOKE TEST FAILED");
process.exit(failed === 0 ? 0 : 1);
