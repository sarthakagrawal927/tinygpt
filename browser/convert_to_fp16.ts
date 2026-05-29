// convert_to_fp16.mjs — convert a canonical .tinygpt (with fp32 [w, m, v]
// triplets) into the compact weights-only fp16 format.
//
// Used to ship the archived Huge Shakespeare checkpoint as the bundled demo:
// the canonical file is 110 MB (over Cloudflare Pages' 25 MB per-file cap);
// stripped + fp16'd, the same model fits in ~19 MB with output quality
// indistinguishable from fp32 in a generation context.
//
// Run:
//   node browser/convert_to_fp16.mjs \
//     data/checkpoints/huge-shakespeare-5000-loss1.22.tinygpt \
//     browser/public/demo.tinygpt
//
// The output is loadable by any standard tinygpt loader — see
// decodeModelFile() in browser/src/main.ts, which detects
// `header.weightDtype === "fp16"` + `includesOptimizerState === false`
// and expands back to the canonical [w, m=0, v=0] layout in memory.

import { promises as fs } from "node:fs";
import { resolve } from "node:path";

const MODEL_MAGIC = "TGPT";
const MODEL_VERSION = 2;

function buildManifest(config) {
  // Mirror of browser/src/main.ts:buildManifest — kept inline so this
  // script has no project-internal dependencies.
  const { layers: L, dModel: C, ctx } = config;
  const dMlp = config.dMlp ?? C * 4;
  const V = 256;
  const entries = [];
  const push = (name, shape) => {
    entries.push({ name, shape, size: shape.reduce((a, b) => a * b, 1) });
  };
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

// fp32 → fp16 (binary16, IEEE 754). Handles subnormals, infinities, NaN,
// round-to-nearest-even. Vectorized would be faster but this is one-shot.
function fp32ToFp16(f) {
  // Use a typed-array trick to read the raw fp32 bits.
  const buf = new ArrayBuffer(4);
  const f32 = new Float32Array(buf);
  const u32 = new Uint32Array(buf);
  f32[0] = f;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  let mant = x & 0x007fffff;
  let exp  = (x >>> 23) & 0xff;

  if (exp === 0xff) {
    // Inf / NaN.
    return sign | 0x7c00 | (mant ? 0x200 : 0);
  }
  // Bias adjustment: fp32 exp bias 127, fp16 bias 15. Half-exp = exp - 127 + 15.
  let halfExp = exp - 127 + 15;
  if (halfExp >= 0x1f) {
    // Overflow → Inf.
    return sign | 0x7c00;
  }
  if (halfExp <= 0) {
    // Subnormal or underflow.
    if (halfExp < -10) return sign;            // too small, flush to zero
    mant = (mant | 0x00800000) >> (1 - halfExp);
    // Round-to-nearest-even.
    if (mant & 0x00001000) mant += 0x00002000;
    return sign | (mant >> 13);
  }
  // Normalized half.
  // Round-to-nearest-even on the dropped 13 bits.
  if (mant & 0x00001000) {
    mant += 0x00002000;
    if (mant & 0x00800000) { mant = 0; halfExp++; if (halfExp >= 0x1f) return sign | 0x7c00; }
  }
  return sign | (halfExp << 10) | (mant >> 13);
}

const [srcPath, dstPath] = process.argv.slice(2);
if (!srcPath || !dstPath) {
  console.error("Usage: node convert_to_fp16.mjs <input.tinygpt> <output.tinygpt>");
  process.exit(1);
}

const buf = await fs.readFile(resolve(srcPath));
const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
const magic = String.fromCharCode(buf[0], buf[1], buf[2], buf[3]);
if (magic !== MODEL_MAGIC) { console.error(`bad magic ${magic}`); process.exit(1); }
const version = view.getUint32(4, true);
const headerLen = view.getUint32(8, true);
const headerJson = new TextDecoder().decode(new Uint8Array(buf.buffer, buf.byteOffset + 12, headerLen));
const header = JSON.parse(headerJson);
console.log(`read ${srcPath}: v${version}, header ${headerLen} bytes, config ${header.config.layers}L d${header.config.dModel} ctx${header.config.ctx}`);

if (header.includesOptimizerState === false) {
  console.error("source already lacks optimizer state — nothing to compact");
  process.exit(1);
}

// State layout: 4-byte int32 step + per-param [w, m, v] fp32 triplets.
// fs.readFile gives us a Node Buffer whose byteOffset may not be 4-aligned
// (it's a slice of a shared pool); copy into a fresh ArrayBuffer so the
// typed-array views can be constructed without alignment errors.
const stateBytes = buf.subarray(12 + headerLen);
const stateAB = new ArrayBuffer(stateBytes.byteLength);
new Uint8Array(stateAB).set(stateBytes);
const stepCount = new Int32Array(stateAB, 0, 1)[0];
const stateF32 = new Float32Array(stateAB, 4, (stateAB.byteLength - 4) / 4);

const manifest = buildManifest(header.config);
const totalFloats = manifest.reduce((a, t) => a + t.size, 0);
console.log(`manifest: ${manifest.length} tensors, ${totalFloats} weight floats`);

// Validate: input must contain exactly 3 × totalFloats fp32 values + step.
const expectedFloats = totalFloats * 3;
if (stateF32.length !== expectedFloats) {
  console.error(`state size mismatch: got ${stateF32.length} floats, expected ${expectedFloats}`);
  process.exit(1);
}

// Output: int32 step + N × fp16 weights (skip the m + v entries).
const outBytes = 4 + totalFloats * 2;
const out = new ArrayBuffer(outBytes);
new Int32Array(out, 0, 1)[0] = stepCount;
const outFp16 = new Uint16Array(out, 4);
let inIdx = 0;
let outIdx = 0;
let stats = { underflow: 0, overflow: 0, nan: 0, max: 0, min: Infinity };
for (const t of manifest) {
  for (let i = 0; i < t.size; i++) {
    const v = stateF32[inIdx + i];
    const absV = Math.abs(v);
    if (absV > stats.max) stats.max = absV;
    if (absV > 0 && absV < stats.min) stats.min = absV;
    if (!Number.isFinite(v)) stats.nan++;
    if (absV > 65504) stats.overflow++;            // fp16 max
    if (absV > 0 && absV < 6e-8) stats.underflow++; // fp16 subnormal floor
    outFp16[outIdx + i] = fp32ToFp16(v);
  }
  inIdx += t.size * 3;  // skip the m and v entries
  outIdx += t.size;
}
console.log(`weight range: |min|=${stats.min.toExponential(2)}, |max|=${stats.max.toExponential(2)}`);
console.log(`overflow→Inf: ${stats.overflow} weights, underflow→0: ${stats.underflow} weights, NaN: ${stats.nan}`);

// New header.
const newHeader = {
  ...header,
  version: MODEL_VERSION,
  includesOptimizerState: false,
  weightDtype: "fp16",
  stateByteLength: outBytes,
  sourceConvertedFrom: srcPath,
  convertedAt: new Date().toISOString(),
};
const newHeaderJson = JSON.stringify(newHeader);
const newHeaderBytes = new TextEncoder().encode(newHeaderJson);

// Assemble.
const final = new Uint8Array(12 + newHeaderBytes.length + outBytes);
final.set(new TextEncoder().encode(MODEL_MAGIC), 0);
const finalView = new DataView(final.buffer);
finalView.setUint32(4, MODEL_VERSION, true);
finalView.setUint32(8, newHeaderBytes.length, true);
final.set(newHeaderBytes, 12);
final.set(new Uint8Array(out), 12 + newHeaderBytes.length);

await fs.writeFile(resolve(dstPath), final);
const stat = await fs.stat(resolve(dstPath));
console.log(`\n✨ wrote ${dstPath}: ${(stat.size / 1024 / 1024).toFixed(2)} MB (was ${(buf.length / 1024 / 1024).toFixed(2)} MB, ${(stat.size / buf.length * 100).toFixed(1)}%)`);
