// finalize_gallery.ts — assemble the gallery from canonical training outputs.
//
// Inputs: data/gallery/<id>.{tinygpt, sample.txt, meta.json}  (one set per id)
// Outputs:
//   browser/public/gallery/<id>.tinygpt          fp16 weights-only (~18 MB)
//   browser/public/gallery/manifest.json         unified, in deterministic order
//
// Re-running is safe: existing public/gallery files are overwritten; the
// Shakespeare entry is preserved (its canonical lives in data/checkpoints/,
// not data/gallery/, so it's added from a static descriptor below).
//
// Usage:
//   node browser/finalize_gallery.ts
//
// Optional --only=id1,id2,... to convert just a subset (handy when one model
// finishes ahead of the others).

import { promises as fs } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { GalleryManifest } from "./src/gallery-schema.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const DATA_DIR = resolve(ROOT, "data/gallery");
const OUT_DIR = resolve(__dirname, "public/gallery");
await fs.mkdir(OUT_DIR, { recursive: true });

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.match(/^--([^=]+)=(.*)$/);
    return m ? [m[1], m[2]] : [a, true];
  }),
);
const onlySet = args.only ? new Set(String(args.only).split(",")) : null;

// Static descriptors for each gallery slot — these define ordering, blurb,
// icon, and corpus pointer. The dynamic bits (params, trainLoss, sample,
// steps) come from the per-id meta.json + sample.txt files.
const SLOTS = [
  {
    id: "shakespeare",
    name: "Shakespeare",
    icon: "🎭",
    blurb: "Verse + dialogue, character labels.",
    corpus: "tinyshakespeare (Karpathy)",
    corpusUrl: "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    // Shakespeare canonical lives outside data/gallery — it was the bundled
    // demo. Fall back to a static descriptor + the file already in public/gallery.
    canonicalOverride: resolve(ROOT, "data/checkpoints/huge-shakespeare-5000-loss1.22.tinygpt"),
    staticSample: "MENENIUS:\nThen you shall hear from a kinsman, ay, my lord.\n\nFIRST CITIZEN:\nWhat trade art thou? Speak, fellow!\n\nSecond Citizen:\nA shrew, my lord, I would never come.",
    staticParams: "9.6M",
    staticTrainLoss: "1.22",
    staticSteps: 5000,
    staticPrompt: "MENENIUS:\n",
  },
  {
    id: "tinystories",
    name: "TinyStories",
    icon: "📖",
    blurb: "Simple children's stories. Subject + verb + object, basic moods, repetition.",
    corpus: "roneneldan/TinyStories",
    corpusUrl: "https://huggingface.co/datasets/roneneldan/TinyStories",
  },
  {
    id: "code",
    name: "Python code",
    icon: "⌨️",
    blurb: "Source code. Imports, def, indentation, dunders, return.",
    corpus: "codeparrot/github-code-clean (Python)",
    corpusUrl: "https://huggingface.co/datasets/codeparrot/github-code-clean",
  },
  {
    id: "recipes",
    name: "Cooking recipes",
    icon: "🍳",
    blurb: "Imperative instructions. Ingredients, units, numbered directions.",
    corpus: "corbt/all-recipes",
    corpusUrl: "https://huggingface.co/datasets/corbt/all-recipes",
  },
  {
    id: "chat",
    name: "Q&A chat",
    icon: "💬",
    blurb: "User → Assistant Q&A pairs from Dolly-15k. Same arch, learned to respond in chat format.",
    corpus: "databricks/databricks-dolly-15k",
    corpusUrl: "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
  },
  {
    id: "seatales",
    name: "Sea Tales",
    icon: "🐟",
    blurb: "Maritime literature — Melville's whales and Conrad's rivers, one model.",
    corpus: "Moby-Dick (first 800K) + Heart of Darkness — both public domain",
    corpusUrl: "https://www.gutenberg.org/ebooks/2701",
    staticPrompt: "Captain Nemo",
  },
];

// fp32 → fp16 conversion — duplicate of convert_to_fp16.mjs's helper so we
// don't need to spawn child processes.
function fp32ToFp16(f) {
  const buf = new ArrayBuffer(4);
  const f32 = new Float32Array(buf);
  const u32 = new Uint32Array(buf);
  f32[0] = f;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  let mant = x & 0x007fffff;
  const exp = (x >>> 23) & 0xff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  let halfExp = exp - 127 + 15;
  if (halfExp >= 0x1f) return sign | 0x7c00;
  if (halfExp <= 0) {
    if (halfExp < -10) return sign;
    mant = (mant | 0x00800000) >> (1 - halfExp);
    if (mant & 0x00001000) mant += 0x00002000;
    return sign | (mant >> 13);
  }
  if (mant & 0x00001000) {
    mant += 0x00002000;
    if (mant & 0x00800000) { mant = 0; halfExp++; if (halfExp >= 0x1f) return sign | 0x7c00; }
  }
  return sign | (halfExp << 10) | (mant >> 13);
}

const MODEL_MAGIC = "TGPT";
const MODEL_VERSION = 2;

function buildManifest(config) {
  const { layers: L, dModel: C, ctx } = config;
  const dMlp = config.dMlp ?? C * 4;
  const V = 256;
  const entries = [];
  const push = (name, shape) => entries.push({ name, shape, size: shape.reduce((a, b) => a * b, 1) });
  push("token_embedding.weight", [V, C]);
  push("position_embedding.weight", [ctx, C]);
  push("ln_final.weight", [C]);
  push("ln_final.bias", [C]);
  for (let i = 0; i < L; i++) {
    push(`blocks.${i}.ln1.weight`, [C]);
    push(`blocks.${i}.ln1.bias`, [C]);
    push(`blocks.${i}.attn.q_proj.weight`, [C, C]);
    push(`blocks.${i}.attn.q_proj.bias`, [C]);
    push(`blocks.${i}.attn.k_proj.weight`, [C, C]);
    push(`blocks.${i}.attn.k_proj.bias`, [C]);
    push(`blocks.${i}.attn.v_proj.weight`, [C, C]);
    push(`blocks.${i}.attn.v_proj.bias`, [C]);
    push(`blocks.${i}.attn.o_proj.weight`, [C, C]);
    push(`blocks.${i}.attn.o_proj.bias`, [C]);
    push(`blocks.${i}.ln2.weight`, [C]);
    push(`blocks.${i}.ln2.bias`, [C]);
    push(`blocks.${i}.mlp.fc_in.weight`, [dMlp, C]);
    push(`blocks.${i}.mlp.fc_in.bias`, [dMlp]);
    push(`blocks.${i}.mlp.fc_out.weight`, [C, dMlp]);
    push(`blocks.${i}.mlp.fc_out.bias`, [C]);
  }
  return entries;
}

async function fp16PackCanonical(srcPath, dstPath) {
  const buf = await fs.readFile(srcPath);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const magic = String.fromCharCode(buf[0], buf[1], buf[2], buf[3]);
  if (magic !== MODEL_MAGIC) throw new Error(`bad magic ${magic} in ${srcPath}`);
  const version = view.getUint32(4, true);
  const headerLen = view.getUint32(8, true);
  const headerJson = new TextDecoder().decode(new Uint8Array(buf.buffer, buf.byteOffset + 12, headerLen));
  const header = JSON.parse(headerJson);

  // Already fp16-packed? Just copy.
  if (header.includesOptimizerState === false && header.weightDtype === "fp16") {
    await fs.copyFile(srcPath, dstPath);
    return { copied: true, srcBytes: buf.length, dstBytes: buf.length };
  }
  // Canonical fp32 [w, m, v] triplets — convert.
  const stateBytes = buf.subarray(12 + headerLen);
  const stateAB = new ArrayBuffer(stateBytes.byteLength);
  new Uint8Array(stateAB).set(stateBytes);
  const stepCount = new Int32Array(stateAB, 0, 1)[0];
  const stateF32 = new Float32Array(stateAB, 4, (stateAB.byteLength - 4) / 4);
  const manifest = buildManifest(header.config);
  const totalFloats = manifest.reduce((a, t) => a + t.size, 0);
  const expectedFloats = totalFloats * 3;
  if (stateF32.length !== expectedFloats) throw new Error(`size mismatch in ${srcPath}: got ${stateF32.length}, expected ${expectedFloats}`);

  const outBytes = 4 + totalFloats * 2;
  const out = new ArrayBuffer(outBytes);
  new Int32Array(out, 0, 1)[0] = stepCount;
  const outFp16 = new Uint16Array(out, 4);
  let inIdx = 0, outIdx = 0;
  for (const t of manifest) {
    for (let i = 0; i < t.size; i++) outFp16[outIdx + i] = fp32ToFp16(stateF32[inIdx + i]);
    inIdx += t.size * 3;
    outIdx += t.size;
  }

  const newHeader = {
    ...header,
    version: MODEL_VERSION,
    includesOptimizerState: false,
    weightDtype: "fp16",
    stateByteLength: outBytes,
    sourceConvertedFrom: srcPath,
    convertedAt: new Date().toISOString(),
  };
  const newHeaderBytes = new TextEncoder().encode(JSON.stringify(newHeader));
  const final = new Uint8Array(12 + newHeaderBytes.length + outBytes);
  final.set(new TextEncoder().encode(MODEL_MAGIC), 0);
  const fv = new DataView(final.buffer);
  fv.setUint32(4, MODEL_VERSION, true);
  fv.setUint32(8, newHeaderBytes.length, true);
  final.set(newHeaderBytes, 12);
  final.set(new Uint8Array(out), 12 + newHeaderBytes.length);
  await fs.writeFile(dstPath, final);
  return { copied: false, srcBytes: buf.length, dstBytes: final.length };
}

/** Read the .tinygpt header and compute param count from its config. The
 *  canonical (training-state) file is the source of truth; falls back to
 *  the destination (fp16-packed) file when only that exists. Returns 0
 *  if neither is present — caller falls back to a static estimate. */
async function readConfigFromTinyGpt(filePath) {
  try {
    const buf = await fs.readFile(filePath);
    if (buf.length < 12) return null;
    const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    const magic = String.fromCharCode(buf[0], buf[1], buf[2], buf[3]);
    if (magic !== "TGPT") return null;
    const headerLen = view.getUint32(8, true);
    const header = JSON.parse(new TextDecoder().decode(buf.subarray(12, 12 + headerLen)));
    return header.config ?? null;
  } catch { return null; }
}

/** Param count from a model config: per-layer matmul shapes + embeddings +
 *  layernorm scales. Matches the manifest layout in browser/src/main.ts. */
function paramsFromManifestConfig(cfg) {
  const { layers: L, dModel: C, ctx: T } = cfg;
  const M = cfg.dMlp ?? C * 4;
  const V = 256;
  let n = 0;
  n += V * C;             // token_embedding
  n += T * C;             // position_embedding
  n += 2 * C;             // ln_final (weight + bias)
  for (let i = 0; i < L; i++) {
    n += 2 * C;           // ln1 (weight + bias)
    n += 4 * (C * C + C); // q/k/v/o projections (weight + bias each)
    n += 2 * C;           // ln2 (weight + bias)
    n += C * M + M;       // fc_in (weight + bias)
    n += M * C + C;       // fc_out (weight + bias)
  }
  return n;
}

async function readJsonIfExists(path) {
  try { return JSON.parse(await fs.readFile(path, "utf8")); }
  catch { return null; }
}
async function readTextIfExists(path) {
  try { return await fs.readFile(path, "utf8"); }
  catch { return null; }
}

// Best-effort first-paragraph extraction so the gallery card preview reads
// as one coherent chunk, not a half-truncated mid-sentence string.
function trimSample(s) {
  if (!s) return "";
  const trimmed = s.trim();
  // Stop at the first double newline if reasonable; otherwise cap at 360 chars.
  const dbl = trimmed.indexOf("\n\n", 60);
  let out = dbl > 0 && dbl < 360 ? trimmed.slice(0, dbl) : trimmed.slice(0, 360);
  if (out.length < trimmed.length) out = out.replace(/\s+\S*$/, "") + " …";
  return out;
}

const built = [];
for (const slot of SLOTS) {
  if (onlySet && !onlySet.has(slot.id)) {
    console.log(`[${slot.id}] skipped (--only filter)`);
    continue;
  }
  const dstPath = resolve(OUT_DIR, `${slot.id}.bin`);

  // Pick the canonical source. Try `.bin` first (already-fp16-packed
  // shape some legacy entries shipped in), then `.tinygpt` (canonical
  // fp32 — the shape every fresh `train_gallery_one.mjs` produces).
  // `fp16PackCanonical` handles either input transparently.
  let canonicalSrc = slot.canonicalOverride ?? resolve(DATA_DIR, `${slot.id}.bin`);
  let canonicalExists = false;
  try { await fs.access(canonicalSrc); canonicalExists = true; } catch {}
  if (!canonicalExists && !slot.canonicalOverride) {
    const altSrc = resolve(DATA_DIR, `${slot.id}.tinygpt`);
    try { await fs.access(altSrc); canonicalSrc = altSrc; canonicalExists = true; } catch {}
  }

  // If no canonical, but the destination already exists (e.g., the pre-built
  // Shakespeare demo we copied earlier), keep the existing file.
  let conv = null;
  if (canonicalExists) {
    conv = await fp16PackCanonical(canonicalSrc, dstPath);
    console.log(`[${slot.id}] ${conv.copied ? "copied" : "fp16-packed"} ${(conv.srcBytes/1024/1024).toFixed(1)} MB -> ${(conv.dstBytes/1024/1024).toFixed(1)} MB`);
  } else {
    try {
      const stat = await fs.stat(dstPath);
      console.log(`[${slot.id}] no canonical found; keeping existing ${dstPath} (${(stat.size/1024/1024).toFixed(1)} MB)`);
      conv = { dstBytes: stat.size };
    } catch {
      console.log(`[${slot.id}] no canonical AND no existing file — skipping`);
      continue;
    }
  }

  // Pull dynamic stats from meta.json (if it exists) or use static fallbacks.
  const meta = await readJsonIfExists(resolve(DATA_DIR, `${slot.id}.meta.json`));
  const sampleText = (await readTextIfExists(resolve(DATA_DIR, `${slot.id}.sample.txt`))) ?? slot.staticSample;

  const params = meta?.params ?? slot.staticParams ?? "9.6M";
  const trainLoss = meta?.finalTrainLoss
    ? Number(meta.finalTrainLoss).toFixed(2)
    : (slot.staticTrainLoss ?? "");
  const steps = meta?.steps ?? slot.staticSteps ?? 5000;

  // Compute GPU memory footprint from the model's config. Read the
  // .tinygpt header (canonical preferred, dst fallback) for the exact
  // config, then walk the manifest. Each param holds w (f32) + Adam m +
  // Adam v = 12 bytes persistent on the GPU. f16-storage adds a 2-byte
  // mirror per param when active. The 12-byte figure shown is what a
  // loaded model occupies regardless of inference vs training mode.
  const cfgSource = (canonicalExists && canonicalSrc) || dstPath;
  const tinygptConfig = await readConfigFromTinyGpt(cfgSource);
  const paramCount = tinygptConfig ? paramsFromManifestConfig(tinygptConfig) : 0;
  const gpuBytes = paramCount * 12;

  // Pick a starting prompt per model so the user lands on the right
  // continuation pattern: "User: " for chat, "def " for code, "Once upon
  // a time" for stories, "MENENIUS:" for Shakespeare. Prefer the prompt
  // the model was actually sampled with at training time (in meta.json)
  // over the static slot fallback.
  const prompt = meta?.samplePrompt ?? slot.staticPrompt ?? "";

  built.push({
    id: slot.id,
    name: slot.name,
    icon: slot.icon,
    blurb: slot.blurb,
    corpus: slot.corpus,
    corpusUrl: slot.corpusUrl,
    file: `${slot.id}.bin`,
    params,
    paramCount,
    trainLoss,
    steps,
    sample: trimSample(sampleText),
    fileBytes: conv?.dstBytes ?? null,
    gpuBytes,
    prompt,
    trainWallMs: meta?.trainWallMs ?? null,
    // Curated gallery cards: featured = true, browser-trained = true,
    // authored by the project. Submission flow populates these
    // differently for community uploads.
    submission: {
      author: "TinyGPT",
      submittedAt: meta?.savedAt ?? new Date().toISOString(),
      browserTrained: true,
      featured: true,
    },
    // benchmarks populated by `score_gallery.mjs` after-the-fact, not here.
    benchmarks: {},
  });
}

const wordForCount = (n) => ["zero", "one", "two", "three", "four", "five", "six"][n] ?? String(n);
const manifest: GalleryManifest = {
  version: 1,
  note:
    `All ${wordForCount(built.length)} models share the same architecture ` +
    `(12L, d=256, ctx=256, ~9.6M params, char-level). Each was trained from ` +
    `scratch in this browser for 5000 steps on ~1.1 MB of its corpus. ` +
    `Compare the samples — same machinery, different patterns.`,
  models: built,
};
await fs.writeFile(resolve(OUT_DIR, "manifest.json"), JSON.stringify(manifest, null, 2));
console.log(`\n✨ wrote ${OUT_DIR}/manifest.json with ${built.length} models:`);
for (const m of built) {
  console.log(`  - ${m.id} · ${m.params} · loss ${m.trainLoss} · sample ${m.sample.length} chars`);
}
