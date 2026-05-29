// score_gallery.ts — run each canonical .tinygpt model in data/gallery/
// through the TinyStories-PPL benchmark and write the scores back into
// the gallery manifest.
//
// How it works:
//   1. Load browser/public/tinygpt.js (the WASM module the browser uses)
//   2. For each canonical .tinygpt file:
//        a. Parse magic + JSON header + state buffer
//        b. Create a matching WASM model
//        c. Import the state (tg_import_state — restores w, m, v, step)
//        d. Set the TinyStories holdout as the corpus, val=100%
//        e. Call tg_eval many times → mean loss → perplexity
//   3. Update browser/public/gallery/manifest.json with benchmarks scores
//
// Run:  node browser/score_gallery.ts        (Node 24+ native TS support)
// Outputs: browser/public/gallery/manifest.json (updated in-place)

import { promises as fs } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import type { GalleryManifest, GalleryModel } from "./src/gallery-schema.ts";

/// Emscripten WASM handle — opaque pointer to a tg_model_create result.
/// 0 is the failure value; non-zero is a valid handle.
type WasmHandle = number;

/// Minimal shape of the Emscripten module — only the surface we use.
/// The .js file produced by emcc exposes many more, but we don't reach
/// for them. Typed as `any` for the cwrap returns because their generics
/// don't add safety in practice.
interface TinyGPTModule {
  cwrap: (name: string, ret: string | null, args: (string | null)[]) => (...a: number[]) => number;
  _malloc: (n: number) => number;
  _free: (ptr: number) => void;
  HEAPU8: Uint8Array;
}

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(here, "..");
const WASM_JS = resolve(ROOT, "browser/public/tinygpt.js");
const GALLERY_DIR = resolve(ROOT, "data/gallery");
const MANIFEST_PATH = resolve(ROOT, "browser/public/gallery/manifest.json");
const HOLDOUT_PATH = resolve(ROOT, "browser/public/benchmarks/tinystories-eval.json");

console.log("[score] loading WASM module…");
const { default: createTinyGPT } = await import(WASM_JS);
const M: TinyGPTModule = await createTinyGPT();

const N = "number";
const tgModelCreate = M.cwrap("tg_model_create", N, [N, N, N, N, N, N, N]);
const tgModelNumParams = M.cwrap("tg_model_num_params", N, [N]);
const tgSetData = M.cwrap("tg_set_data", null, [N, N, N, N]);
const tgEval = M.cwrap("tg_eval", N, [N, N, N, N]);
const tgModelFree = M.cwrap("tg_model_free", null, [N]);
const tgImportState = M.cwrap("tg_import_state", null, [N, N]);

// Read the holdout text — we score all stories joined into one corpus.
console.log("[score] reading TinyStories holdout…");
interface HoldoutFile {
  source: string;
  count: number;
  totalBytes: number;
  stories: string[];
}
const holdout: HoldoutFile = JSON.parse(await fs.readFile(HOLDOUT_PATH, "utf8"));
const holdoutText = holdout.stories.join("\n\n");
const holdoutBytes = new TextEncoder().encode(holdoutText);
console.log(`[score] holdout: ${holdout.stories.length} stories, ${holdoutBytes.length} bytes`);

// Read existing manifest. We'll merge scores in.
const manifest: GalleryManifest = JSON.parse(await fs.readFile(MANIFEST_PATH, "utf8"));

// Inputs we want to score, by gallery id. Each maps to a canonical
// .tinygpt at the file path below.
const candidates = await fs.readdir(GALLERY_DIR);
const tinygptFiles = candidates.filter((f) => f.endsWith(".tinygpt"));
console.log(`[score] found ${tinygptFiles.length} canonical .tinygpt files:`,
            tinygptFiles.join(", "));

const MAGIC = "TGPT";

/// .tinygpt header subset we actually consult here. The full schema is
/// the source-of-truth one in `native-mac/Sources/TinyGPTIO/Manifest.swift` —
/// we only need the geometry + dtype check.
interface TinygptHeaderConfig {
  layers?: number;
  dModel?: number;
  ctx?: number;
  heads?: number;
  dMlp?: number;
}
interface TinygptHeader {
  config: TinygptHeaderConfig;
  weightDtype?: string;
}
interface ParsedTinygpt {
  config: TinygptHeaderConfig;
  stateBytes: Uint8Array;
}

/// Parse a .tinygpt file: returns { config, stateBytes }.
function parseTinygpt(buf: ArrayBuffer): ParsedTinygpt {
  if (buf.byteLength < 12) throw new Error("file too small");
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
  if (magic !== MAGIC) throw new Error(`bad magic: ${magic}`);
  const dv = new DataView(buf);
  const version = dv.getUint32(4, true);
  if (version !== 1 && version !== 2) {
    throw new Error(`unsupported version ${version}`);
  }
  const headerLen = dv.getUint32(8, true);
  if (12 + headerLen > buf.byteLength) throw new Error("malformed header");
  const headerJson = new TextDecoder().decode(new Uint8Array(buf, 12, headerLen));
  const header: TinygptHeader = JSON.parse(headerJson);
  // The state buffer starts right after the header. Canonical files use
  // [int32 step + per-tensor [w_fp32, m_fp32, v_fp32]] which is exactly
  // what tg_import_state expects.
  const stateBytes = new Uint8Array(buf.slice(12 + headerLen));
  if (header.weightDtype && header.weightDtype !== "fp32") {
    throw new Error(
      `score_gallery.ts only handles canonical fp32 .tinygpt files; ` +
      `this one has weightDtype=${header.weightDtype}. Use the file from ` +
      `data/gallery/, not browser/public/gallery/.`,
    );
  }
  return { config: header.config, stateBytes };
}

/// Eval the model on the held-out corpus and return perplexity.
function evalPerplexity(handle: WasmHandle, batchSize: number, nBatches: number) {
  // tg_set_data installs the holdout as the corpus. train_frac = 0.0
  // pushes 100% of the corpus into the val split, so tg_eval(..., split=1)
  // samples from the entire held-out set.
  const dataPtr = M._malloc(holdoutBytes.length);
  M.HEAPU8.set(holdoutBytes, dataPtr);
  tgSetData(handle, dataPtr, holdoutBytes.length, 0.0);
  M._free(dataPtr);

  // n_batches batches of batchSize sequences. Random sampling with
  // replacement — increase n_batches to tighten variance.
  const meanLoss = tgEval(handle, 1, batchSize, nBatches);
  // tg_eval returns mean per-token cross-entropy (in nats), same definition
  // as the browser playground's val_loss UI.
  return { loss: meanLoss, perplexity: Math.exp(meanLoss) };
}

const benchId = "tinystories-ppl";
interface ScoreUpdate {
  id: string;
  score: number | null;
  details: { error?: string; loss?: number; vocab?: number; batches?: number; tokens?: number };
}
const updated: ScoreUpdate[] = [];

for (const filename of tinygptFiles.sort()) {
  const id = filename.replace(/\.tinygpt$/, "");
  const path = resolve(GALLERY_DIR, filename);
  console.log(`\n[score] === ${id} ===`);
  try {
    const buf = (await fs.readFile(path)).buffer as ArrayBuffer;
    const { config, stateBytes } = parseTinygpt(buf);
    console.log(`        config: ${config.layers}L · d=${config.dModel} · ctx=${config.ctx} · ${stateBytes.length} state bytes`);

    // Create model with matching geometry. The 7-arg signature is:
    //   tg_model_create(vocab, ctx, layers, heads, d_model, d_mlp, seed)
    const handle: WasmHandle = tgModelCreate(
      256, // vocab — gallery models are byte-level
      config.ctx ?? 256,
      config.layers ?? 12,
      config.heads ?? 8,
      config.dModel ?? 256,
      config.dMlp ?? (config.dModel ? config.dModel * 4 : 1024),
      42,
    );
    if (handle === 0) throw new Error("tg_model_create returned 0");

    const nParams = tgModelNumParams(handle);
    console.log(`        params: ${nParams.toLocaleString()}`);

    // Import the saved state. The state buffer is exactly what
    // tg_export_state would have written.
    const statePtr = M._malloc(stateBytes.length);
    M.HEAPU8.set(stateBytes, statePtr);
    tgImportState(handle, statePtr);
    M._free(statePtr);

    // Eval. 32 batches × 8 sequences × ctx tokens per batch =
    // ~32 × 8 × 256 = ~65K tokens scored — enough to drop sampling
    // variance below ~3% in our experiments.
    const t0 = Date.now();
    const { loss, perplexity } = evalPerplexity(handle, 8, 32);
    const dt = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`        eval: loss=${loss.toFixed(3)}  perplexity=${perplexity.toFixed(2)}  (${dt}s)`);

    tgModelFree(handle);

    updated.push({
      id, score: perplexity,
      details: { loss, vocab: 256, batches: 32, tokens: 32 * 8 * (config.ctx ?? 256) },
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error(`[score] FAIL ${id}: ${msg}`);
    updated.push({ id, score: null, details: { error: msg } });
  }
}

// Merge into manifest. Each manifest entry whose id matches gets its
// benchmarks.<benchId> set; unknown ids in `updated` are appended as
// new rows (so newly-trained models that haven't been published to
// the gallery yet still appear on the leaderboard).
console.log("\n[score] merging scores into manifest…");
const byId = new Map<string, GalleryModel>((manifest.models || []).map((m) => [m.id, m]));
for (const { id, score } of updated) {
  let entry = byId.get(id);
  if (!entry) {
    // Bootstrap a minimal stub. The next finalize_gallery run will
    // overwrite with full metadata.
    entry = {
      id, name: id, file: `${id}.bin`,
      benchmarks: {},
      submission: { author: "TinyGPT", submittedAt: new Date().toISOString(), featured: false },
    };
    manifest.models = manifest.models || [];
    manifest.models.push(entry);
  }
  entry.benchmarks = entry.benchmarks || {};
  entry.benchmarks[benchId] = score;
}

await fs.writeFile(MANIFEST_PATH, JSON.stringify(manifest, null, 2));
console.log(`[score] wrote ${MANIFEST_PATH}\n`);

console.log("[score] summary:");
console.log("   id                   score (ppl)   loss");
console.log("   ───────────────────  ───────────   ─────");
for (const { id, score, details } of updated) {
  if (score == null) {
    console.log(`   ${id.padEnd(20)} ERROR  ${details.error ?? ""}`);
  } else {
    console.log(`   ${id.padEnd(20)} ${score.toFixed(2).padStart(10)}    ${details.loss?.toFixed(3) ?? "—"}`);
  }
}
