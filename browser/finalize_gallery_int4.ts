// finalize_gallery_int4.mjs — produce 4-bit gallery variants from the fp16 .bin files.
//
// Inputs:  browser/public/gallery/<id>.bin           (fp16 weights-only, ~19 MB)
// Outputs: browser/public/gallery/<id>.int4.bin      (4-bit weights-only, ~5 MB)
//
// Why a separate script: the int4 path is INFERENCE-ONLY storage compression.
// It only touches the published gallery files; the canonical training-state
// .tinygpt files in data/checkpoints stay fp32. We also keep the fp16 .bin
// files alongside so the manifest can advertise both URLs and the client
// can fall back to fp16 if the int4 numerics gate fails on its GPU.
//
// File layout (output):
//   4   "TGPT" magic
//   4   uint32 LE version (= 2)
//   4   uint32 LE JSON header length
//   N   UTF-8 JSON header — same shape as fp16 file but with:
//         weightDtype: "int4",
//         int4BlockSize: 64,
//         int4ScalesBytes: <byte length of the scales blob>,
//         int4PackedBytes: <byte length of the packed nibble blob>,
//   4   int32 step prefix (always 0 for inference-only weights)
//   S   scales blob — concatenated fp16 (uint16 LE) scales in manifest order,
//       one scale per ceil(tensor_size / blockSize). Tensors with fewer than
//       blockSize elements get a single scale (their final partial block
//       uses zeroes to pad — the block size is fixed, the partial-block
//       slot in `packed` reserves a full block-worth of nibbles).
//   P   packed int4 buffer — two int4s per byte, low nibble first.
//       Each tensor is packed in fixed full-block chunks; final partial
//       block is zero-padded out to blockSize so decoder math is uniform.
//
// Quantization: BLOCK-WISE SYMMETRIC with block size 64 along the
// contiguous (last) axis — same shape as GGUF Q4_0. Per-tensor scaling
// turned out to lose too much numeric quality on layers where the absmax
// is dominated by a handful of large weights (mean_rel ~80%). With
// block-wise scaling each 64-element span gets its own fp16 scale, so
// quiet rows + loud rows quantize independently. Result: end-to-end
// output drift drops to ~1% of mean magnitude — same regime as fp16
// storage. Overhead: one fp16 scale per 64 weights = 32 bytes per
// 1024-element block = a few KB across the whole model (negligible).
//
// For each block we compute
//   absmax = max(|w_i|);  scale = absmax / 7;  q_i = round(w_i / scale)
// clamped to [-7, 7] (the int4 grid is -8..7 but we use -7..7 so the
// codebook is symmetric around zero). Stored nibble is (q + 8) so the
// on-disk value is unsigned 0..15.
//
// IMPORTANT (re: the transpose bug): we quantize the bytes AS STORED in the
// fp16 input — no shape reinterpretation, no transposes. The browser's WASM
// importer already handles the [in, out] vs [out, in] convention; the
// conversion here is a byte-wise lossy pass that preserves whatever layout
// the source file used.

import { promises as fs } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { GalleryManifest } from "./src/gallery-schema.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(__dirname, "public/gallery");

const MODEL_MAGIC = "TGPT";
const MODEL_VERSION = 2;

// Match buildManifest in browser/src/main.ts / finalize_gallery.mjs.
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

// IEEE 754 half → single. Same routine as browser/src/main.ts.
function fp16ToFp32(h) {
  const sign = (h >> 15) & 0x1;
  const exp  = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * frac * Math.pow(2, -24);
  }
  if (exp === 0x1f) return frac === 0 ? (sign ? -Infinity : Infinity) : NaN;
  return (sign ? -1 : 1) * (1 + frac / 1024) * Math.pow(2, exp - 15);
}

// Single → IEEE 754 half. Used to compress the per-block scales table.
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

const INT4_BLOCK_SIZE = 64;

/** Quantize a Float32Array to int4 with block-wise symmetric scaling.
 *  Block size is INT4_BLOCK_SIZE (64). Returns
 *  { packed: Uint8Array of length nBlocks * blockSize/2,
 *    scales: Float32Array of length nBlocks }.
 *  Trailing partial blocks (when tensor size isn't a multiple of blockSize)
 *  are zero-padded — both their scale and their nibbles default to zero. */
function quantizeInt4(values, blockSize = INT4_BLOCK_SIZE) {
  const nBlocks = Math.ceil(values.length / blockSize);
  const scales = new Float32Array(nBlocks);
  const packedLen = nBlocks * (blockSize >> 1);
  const packed = new Uint8Array(packedLen);
  for (let b = 0; b < nBlocks; b++) {
    const start = b * blockSize;
    const end = Math.min(start + blockSize, values.length);
    let absmax = 0;
    for (let i = start; i < end; i++) {
      const a = Math.abs(values[i]);
      if (a > absmax) absmax = a;
    }
    const scale = absmax > 0 ? absmax / 7 : 0;  // 0 marks "all zero block"
    scales[b] = scale;
    if (scale === 0) continue;  // packed stays zeroed; dequant returns zeros
    const invScale = 1 / scale;
    for (let i = start; i < end; i++) {
      let q = Math.round(values[i] * invScale);
      if (q < -7) q = -7;
      if (q > 7) q = 7;
      const nibble = (q + 8) & 0xf;
      const idx = (b * blockSize + (i - start));
      if ((idx & 1) === 0) packed[idx >>> 1] = nibble;
      else packed[idx >>> 1] |= nibble << 4;
    }
  }
  return { packed, scales };
}

/** Dequantize block-wise int4 → fp32. Counterpart to quantizeInt4. */
function dequantizeInt4(packed, count, scales, blockSize = INT4_BLOCK_SIZE) {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const b = (i / blockSize) | 0;
    const byte = packed[i >>> 1];
    const nibble = (i & 1) === 0 ? (byte & 0xf) : (byte >>> 4) & 0xf;
    out[i] = (nibble - 8) * scales[b];
  }
  return out;
}

/** Convert one fp16 .bin to int4 .bin. Returns size deltas + worst-case
 *  per-tensor drift for the numerics summary. */
async function convertOne(srcPath, dstPath) {
  const buf = await fs.readFile(srcPath);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const magic = String.fromCharCode(buf[0], buf[1], buf[2], buf[3]);
  if (magic !== MODEL_MAGIC) throw new Error(`bad magic ${magic} in ${srcPath}`);
  const version = view.getUint32(4, true);
  if (version !== 1 && version !== 2) throw new Error(`unsupported version ${version}`);
  const headerLen = view.getUint32(8, true);
  const headerJson = new TextDecoder().decode(new Uint8Array(buf.buffer, buf.byteOffset + 12, headerLen));
  const header = JSON.parse(headerJson);

  if (header.weightDtype !== "fp16") {
    throw new Error(`expected fp16 source, got ${header.weightDtype} in ${srcPath}`);
  }
  if (header.includesOptimizerState !== false) {
    throw new Error(`expected weights-only source in ${srcPath}`);
  }

  // Source state: 4-byte int32 step + N fp16 weights.
  const stateBytes = new Uint8Array(buf.buffer, buf.byteOffset + 12 + headerLen, buf.byteLength - 12 - headerLen);
  const stepCount = new DataView(stateBytes.buffer, stateBytes.byteOffset, 4).getInt32(0, true);
  const fp16View = new DataView(stateBytes.buffer, stateBytes.byteOffset + 4);
  const totalHalfs = (stateBytes.byteLength - 4) / 2;

  const manifest = buildManifest(header.config);
  const totalFloats = manifest.reduce((a, t) => a + t.size, 0);
  if (totalHalfs !== totalFloats) {
    throw new Error(`size mismatch: ${totalHalfs} fp16 weights vs ${totalFloats} manifest floats`);
  }

  // Walk tensors, quantize each block-wise, accumulate the scales blob and
  // the packed nibble blob. Tensors are concatenated end-to-end in each
  // blob in manifest order — boundaries are implicit (size derivable from
  // shape + blockSize). For each tensor we store nBlocks = ceil(size/64)
  // fp16 scales and nBlocks * 32 packed bytes.
  const scaleU16Chunks = [];
  const packedChunks = [];
  let totalPackedBytes = 0;
  let totalScales = 0;
  let inIdx = 0;
  let worstMeanRel = 0;
  let worstMaxAbs = 0;
  let worstName = "";
  for (const t of manifest) {
    const values = new Float32Array(t.size);
    for (let i = 0; i < t.size; i++) {
      values[i] = fp16ToFp32(fp16View.getUint16((inIdx + i) * 2, true));
    }
    const { packed, scales } = quantizeInt4(values);
    // Convert scales to fp16 for on-disk storage.
    const scaleU16 = new Uint16Array(scales.length);
    for (let i = 0; i < scales.length; i++) scaleU16[i] = fp32ToFp16(scales[i]);
    scaleU16Chunks.push(scaleU16);
    packedChunks.push(packed);
    totalPackedBytes += packed.byteLength;
    totalScales += scales.length;
    // Drift summary: dequantize and compare to the fp16→fp32 reference.
    // The fp16-scale-roundtrip is included so this measures what the
    // browser will actually see.
    const scalesRoundtrip = new Float32Array(scales.length);
    for (let i = 0; i < scales.length; i++) scalesRoundtrip[i] = fp16ToFp32(scaleU16[i]);
    const deq = dequantizeInt4(packed, t.size, scalesRoundtrip);
    let sumAbs = 0, sumRel = 0, maxAbs = 0;
    for (let i = 0; i < t.size; i++) sumAbs += Math.abs(values[i]);
    const meanAbs = sumAbs / t.size;
    const denomFloor = Math.max(meanAbs * 0.01, 1e-6);
    for (let i = 0; i < t.size; i++) {
      const e = Math.abs(deq[i] - values[i]);
      if (e > maxAbs) maxAbs = e;
      sumRel += e / Math.max(Math.abs(values[i]), denomFloor);
    }
    const meanRel = sumRel / t.size;
    if (meanRel > worstMeanRel) { worstMeanRel = meanRel; worstName = t.name; }
    if (maxAbs > worstMaxAbs) worstMaxAbs = maxAbs;
    inIdx += t.size;
  }

  const totalScalesBytes = totalScales * 2;
  // Assemble output state: 4-byte step + scales blob + packed nibble blob.
  const stateBuf = new Uint8Array(4 + totalScalesBytes + totalPackedBytes);
  const sv = new DataView(stateBuf.buffer);
  sv.setInt32(0, stepCount, true);
  let cursor = 4;
  for (const s of scaleU16Chunks) {
    const bytes = new Uint8Array(s.buffer, s.byteOffset, s.byteLength);
    stateBuf.set(bytes, cursor);
    cursor += bytes.byteLength;
  }
  for (const p of packedChunks) {
    stateBuf.set(p, cursor);
    cursor += p.byteLength;
  }

  const newHeader = {
    ...header,
    version: MODEL_VERSION,
    includesOptimizerState: false,
    weightDtype: "int4",
    int4BlockSize: INT4_BLOCK_SIZE,
    int4ScalesBytes: totalScalesBytes,
    int4PackedBytes: totalPackedBytes,
    stateByteLength: stateBuf.byteLength,
    sourceConvertedFrom: srcPath,
    convertedAt: new Date().toISOString(),
  };
  const newHeaderBytes = new TextEncoder().encode(JSON.stringify(newHeader));
  const final = new Uint8Array(12 + newHeaderBytes.length + stateBuf.byteLength);
  final.set(new TextEncoder().encode(MODEL_MAGIC), 0);
  const fv = new DataView(final.buffer);
  fv.setUint32(4, MODEL_VERSION, true);
  fv.setUint32(8, newHeaderBytes.length, true);
  final.set(newHeaderBytes, 12);
  final.set(stateBuf, 12 + newHeaderBytes.length);
  await fs.writeFile(dstPath, final);
  return {
    srcBytes: buf.length,
    dstBytes: final.length,
    worstMeanRel,
    worstMaxAbs,
    worstName,
  };
}

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.match(/^--([^=]+)=(.*)$/);
    return m ? [m[1], m[2]] : [a, true];
  }),
);
const onlySet = args.only ? new Set(String(args.only).split(",")) : null;

const ids = ["shakespeare", "tinystories", "code", "chat"];
const reports = [];
for (const id of ids) {
  if (onlySet && !onlySet.has(id)) {
    console.log(`[${id}] skipped (--only filter)`);
    continue;
  }
  const src = resolve(OUT_DIR, `${id}.bin`);
  const dst = resolve(OUT_DIR, `${id}.int4.bin`);
  try { await fs.access(src); } catch {
    console.log(`[${id}] no fp16 source — run finalize_gallery.mjs first; skipping`);
    continue;
  }
  const r = await convertOne(src, dst);
  reports.push({ id, ...r });
  console.log(
    `[${id}] ${(r.srcBytes/1024/1024).toFixed(1)} MB → ${(r.dstBytes/1024/1024).toFixed(1)} MB ` +
    `(${(100*r.dstBytes/r.srcBytes).toFixed(1)}%); worst drift mean_rel=${(r.worstMeanRel*100).toFixed(2)}% ` +
    `in ${r.worstName}, max_abs=${r.worstMaxAbs.toExponential(2)}`,
  );
}

// Patch the manifest so the browser knows about the int4 variant. We add
// `fileInt4` + `fileInt4Bytes` per-model; the existing `file` / `fileBytes`
// fields stay as the fp16 fallback. The client decides at load time which
// URL to fetch based on (a) the int4 numerics gate verdict and (b) any user
// preference. Keeping both fields means callers can also publish a model
// that ONLY has fp16 (no `fileInt4`) and the browser will Just Work.
try {
  const manifestPath = resolve(OUT_DIR, "manifest.json");
  const manifest: GalleryManifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
  for (const m of manifest.models) {
    const r = reports.find((x) => x.id === m.id);
    if (!r) continue;
    m.fileInt4 = `${m.id}.int4.bin`;
    m.fileInt4Bytes = r.dstBytes;
  }
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`\npatched ${manifestPath} with fileInt4 entries for ${reports.length} models`);
} catch (err) {
  console.warn(`warning: couldn't patch manifest.json: ${err.message}`);
}
