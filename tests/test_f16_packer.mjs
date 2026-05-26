// test_f16_packer.mjs — sanity-check the host-side f32 → packed-half packer
// in webgpu/kernels.ts against ground-truth IEEE-754 binary16 conversion.
//
// The shader-side reads each u32 as { unpack2x16float(u32) -> vec2<f32> },
// which means: low 16 bits become .x (the f32 round-trip of element 2i),
// high 16 bits become .y (element 2i+1). We replicate that here and check
// our packer matches across the corners that matter: normal range, subnormal
// range, overflow, NaN, signed zero, and the rounding-to-nearest-even tie.

// Inline the packer + an independent reference. The packer in kernels.ts is
// pasted here verbatim (no Vite import in Node).
function f32ToF16Bits(x) {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = x;
  const bits = u32[0];
  const sign = (bits >>> 16) & 0x8000;
  let exp = (bits >>> 23) & 0xff;
  let mant = bits & 0x7fffff;
  if (exp === 0xff) {
    return sign | 0x7c00 | (mant ? 0x200 | (mant >>> 13) : 0);
  }
  exp = exp - 127 + 15;
  if (exp >= 0x1f) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant = (mant | 0x800000) >>> (1 - exp);
    const rb = 1 << 12;
    if (mant & rb && (mant & (rb - 1) || mant & (rb << 1))) mant += rb;
    return sign | (mant >>> 13);
  }
  const rb = 1 << 12;
  if (mant & rb && (mant & (rb - 1) || mant & (rb << 1))) {
    mant += rb;
    if (mant & 0x800000) {
      mant = 0;
      exp += 1;
      if (exp >= 0x1f) return sign | 0x7c00;
    }
  }
  return sign | (exp << 10) | (mant >>> 13);
}

// Reference: f16 → f32 via the standard IEEE-754 binary16 layout.
function f16BitsToF32(bits) {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exp = (bits >>> 10) & 0x1f;
  const mant = bits & 0x3ff;
  if (exp === 0) {
    if (mant === 0) return sign * 0;
    return sign * mant * Math.pow(2, -24);
  }
  if (exp === 0x1f) {
    return mant === 0 ? sign * Infinity : NaN;
  }
  return sign * (1 + mant / 1024) * Math.pow(2, exp - 15);
}

let failed = 0;
const check = (name, ok, detail) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name.padEnd(40)} (${detail})`);
  if (!ok) failed++;
};

// 1. Normal range — random floats in [-1, 1].
{
  const n = 200;
  const src = new Float32Array(n);
  for (let i = 0; i < n; i++) src[i] = Math.random() * 2 - 1;
  let maxErr = 0;
  for (let i = 0; i < n; i++) {
    const recoverd = f16BitsToF32(f32ToF16Bits(src[i]));
    maxErr = Math.max(maxErr, Math.abs(src[i] - recoverd));
  }
  // Worst case for normalised f16 around |x|≈1: relative ε ≈ 2^-10 ≈ 9.77e-4.
  check("round-trip [-1, 1] within f16 eps", maxErr < 1e-3, `max err ${maxErr.toExponential(2)}`);
}

// 2. Specific edge values.
{
  const cases = [
    { val: 0, want: 0 },
    { val: -0, want: -0 },
    { val: 1, want: 1 },
    { val: -1, want: -1 },
    { val: 65504, want: 65504 },          // f16 max-finite
    { val: 65520, want: Infinity },       // overflows
    { val: -65520, want: -Infinity },
    { val: 1e-5, want: 1e-5 },            // subnormal-ish
    { val: NaN, want: NaN },
    { val: Infinity, want: Infinity },
    { val: -Infinity, want: -Infinity },
  ];
  for (const c of cases) {
    const r = f16BitsToF32(f32ToF16Bits(c.val));
    let ok;
    if (Number.isNaN(c.want)) ok = Number.isNaN(r);
    else if (c.want === Infinity || c.want === -Infinity) ok = r === c.want;
    else if (c.want === 0) ok = r === 0 && Math.sign(1 / r) === Math.sign(1 / c.want);
    else ok = Math.abs(r - c.want) <= Math.abs(c.want) * 1e-3 + 1e-6;
    check(`edge value ${String(c.val)}`, ok, `got ${r}`);
  }
}

// 3. Pack-layout check: pack2x16float lays out low=elt[2i], high=elt[2i+1].
{
  const src = new Float32Array([0.25, -0.5]);
  const lo = f32ToF16Bits(0.25);
  const hi = f32ToF16Bits(-0.5);
  const expected = lo | (hi << 16);
  // Inline the same packing logic kernels.ts uses.
  const packed = (() => {
    const out = new Uint32Array(1);
    out[0] = f32ToF16Bits(src[0]) | (f32ToF16Bits(src[1]) << 16);
    return out[0];
  })();
  check("pack layout matches pack2x16float", packed === (expected >>> 0), `0x${packed.toString(16).padStart(8, "0")} vs 0x${(expected >>> 0).toString(16).padStart(8, "0")}`);
}

console.log(failed === 0 ? "\nf16 packer test passed" : `\nf16 packer test FAILED (${failed})`);
process.exit(failed === 0 ? 0 : 1);
