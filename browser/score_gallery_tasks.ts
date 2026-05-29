// score_gallery_tasks.mjs — score task-based benchmarks (sort-6,
// reverse-16) on the canonical .tinygpt models via the WASM module's
// tg_generate path.
//
// Companion to score_gallery.mjs (which handles tg_eval-based perplexity
// benchmarks). Same load + state-import path; the only differences are
// the evaluation loop (greedy generate per trial) and the score
// definition (exact-match accuracy %).
//
// Run:  node browser/score_gallery_tasks.ts

import { promises as fs } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import type { GalleryManifest } from "./src/gallery-schema.ts";

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(here, "..");
const WASM_JS = resolve(ROOT, "browser/public/tinygpt.js");
const GALLERY_DIR = resolve(ROOT, "data/gallery");
const MANIFEST_PATH = resolve(ROOT, "browser/public/gallery/manifest.json");

console.log("[score-tasks] loading WASM module…");
const { default: createTinyGPT } = await import(WASM_JS);
const M = await createTinyGPT();

const N = "number";
const tgModelCreate = M.cwrap("tg_model_create", N, [N, N, N, N, N, N, N]);
const tgGenerate = M.cwrap("tg_generate", N, [N, N, N, N, N, N, N, N]);
const tgModelFree = M.cwrap("tg_model_free", null, [N]);
const tgImportState = M.cwrap("tg_import_state", null, [N, N]);

const MAGIC = "TGPT";

function parseTinygpt(buf) {
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
  if (magic !== MAGIC) throw new Error(`bad magic: ${magic}`);
  const dv = new DataView(buf);
  const headerLen = dv.getUint32(8, true);
  const headerJson = new TextDecoder().decode(new Uint8Array(buf, 12, headerLen));
  const header = JSON.parse(headerJson);
  const stateBytes = new Uint8Array(buf.slice(12 + headerLen));
  return { config: header.config, stateBytes };
}

/// Greedy generate `maxNew` bytes given a prompt string.
function generate(handle, prompt, maxNew) {
  const promptBytes = new TextEncoder().encode(prompt);
  const promptPtr = M._malloc(promptBytes.length);
  M.HEAPU8.set(promptBytes, promptPtr);
  const outPtr = M._malloc(maxNew);
  // tg_generate(model, prompt, plen, out, maxNew, temp, topK, seed)
  // temp=0 → greedy
  const produced = tgGenerate(handle, promptPtr, promptBytes.length, outPtr, maxNew, 0, 0, 7);
  const out = new TextDecoder("utf-8", { fatal: false })
    .decode(M.HEAPU8.slice(outPtr, outPtr + produced));
  M._free(promptPtr);
  M._free(outPtr);
  return out;
}

/// Mulberry32 — match the seeded RNG used by the in-browser benchmark
/// specs so the trial sets line up exactly.
function seedRandom(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildSort6Trials() {
  const rng = seedRandom(0x517);
  const trials = [];
  for (let i = 0; i < 200; i++) {
    const digits = [];
    for (let j = 0; j < 6; j++) digits.push(Math.floor(rng() * 10));
    trials.push({
      prompt: `sort: ${digits.join(" ")} = `,
      expected: digits.slice().sort((a, b) => a - b).join(" "),
    });
  }
  return trials;
}

function buildReverse16Trials() {
  const rng = seedRandom(0x3EE5);
  const ALPHABET = "abcdefghijklmnopqrstuvwxyz";
  const trials = [];
  for (let i = 0; i < 200; i++) {
    const len = 4 + Math.floor(rng() * 13);
    let s = "";
    for (let j = 0; j < len; j++) s += ALPHABET[Math.floor(rng() * 26)];
    trials.push({
      prompt: `reverse: ${s} = `,
      expected: s.split("").reverse().join(""),
    });
  }
  return trials;
}

function scoreTask(handle, trials) {
  let correct = 0;
  const failures = [];
  for (const { prompt, expected } of trials) {
    const continuation = generate(handle, prompt, expected.length + 2);
    const got = continuation.replace(/^\s+/, "").slice(0, expected.length);
    if (got === expected) correct += 1;
    else if (failures.length < 3) failures.push(`${prompt}→ "${got}"`);
  }
  return { score: (correct / trials.length) * 100, correct, total: trials.length, failures };
}

const manifest: GalleryManifest = JSON.parse(await fs.readFile(MANIFEST_PATH, "utf8"));
const files = (await fs.readdir(GALLERY_DIR)).filter((f) => f.endsWith(".tinygpt")).sort();

const sortTrials = buildSort6Trials();
const reverseTrials = buildReverse16Trials();
console.log(`[score-tasks] ${sortTrials.length} sort-6 trials, ${reverseTrials.length} reverse-16 trials`);

const results = [];
for (const filename of files) {
  const id = filename.replace(/\.tinygpt$/, "");
  const path = resolve(GALLERY_DIR, filename);
  console.log(`\n[score-tasks] === ${id} ===`);
  try {
    const buf = (await fs.readFile(path)).buffer;
    const { config, stateBytes } = parseTinygpt(buf);
    const handle = tgModelCreate(
      256, config.ctx ?? 256, config.layers ?? 12, config.heads ?? 8,
      config.dModel ?? 256, config.dMlp ?? 1024, 42,
    );
    const statePtr = M._malloc(stateBytes.length);
    M.HEAPU8.set(stateBytes, statePtr);
    tgImportState(handle, statePtr);
    M._free(statePtr);
    const t0 = Date.now();
    const sort = scoreTask(handle, sortTrials);
    const t1 = Date.now();
    const rev = scoreTask(handle, reverseTrials);
    const t2 = Date.now();
    console.log(`  sort-6     : ${sort.score.toFixed(1).padStart(5)}%  (${sort.correct}/${sort.total}, ${((t1 - t0) / 1000).toFixed(1)}s)`);
    console.log(`  reverse-16 : ${rev.score.toFixed(1).padStart(5)}%  (${rev.correct}/${rev.total}, ${((t2 - t1) / 1000).toFixed(1)}s)`);
    tgModelFree(handle);
    results.push({ id, sort: sort.score, reverse: rev.score });
  } catch (e) {
    console.error(`[score-tasks] FAIL ${id}: ${e.message}`);
    results.push({ id, sort: null, reverse: null, error: e.message });
  }
}

console.log("\n[score-tasks] merging scores into manifest…");
const byId = new Map((manifest.models || []).map((m) => [m.id, m]));
for (const { id, sort, reverse } of results) {
  const entry = byId.get(id);
  if (!entry) continue;
  entry.benchmarks = entry.benchmarks || {};
  entry.benchmarks["sort-6"] = sort;
  entry.benchmarks["reverse-16"] = reverse;
}
await fs.writeFile(MANIFEST_PATH, JSON.stringify(manifest, null, 2));
console.log(`[score-tasks] wrote ${MANIFEST_PATH}`);
