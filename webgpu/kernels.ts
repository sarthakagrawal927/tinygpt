/**
 * kernels.ts — WebGPU device setup + the matmul compute kernel (Phase 5).
 *
 * JS/TS glue around webgpu/matmul.wgsl:
 *   - request a GPUAdapter + GPUDevice (feature-detected; HTTPS-only)
 *   - create GPUBuffers, a bind group, a compute pipeline
 *   - upload inputs, dispatch workgroups, read the result back
 *
 * Acceptance (milestone 6): the WebGPU matmul must equal the WASM matmul within
 * tolerance and be measurably faster on a large matrix — `benchmarkMatmul`
 * checks both against a caller-supplied reference (the WASM kernel).
 *
 * Guide: docs/browser_notes.md ("WebGPU acceleration")
 */

import matmulShader from "./matmul.wgsl?raw";
import matmulF16Shader from "./matmul_f16packed.wgsl?raw";

/** A matmul runner bound to fixed dimensions — pipeline + buffers built once. */
export interface MatmulRunner {
  /** C = A @ B for the dimensions this runner was created with. */
  run(a: Float32Array, b: Float32Array): Promise<Float32Array>;
  destroy(): void;
}

/** Request a WebGPU device, or null if the browser/platform has no WebGPU. */
export async function initWebGPU(): Promise<GPUDevice | null> {
  if (typeof navigator === "undefined" || !navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    return await adapter.requestDevice();
  } catch {
    return null;
  }
}

/** Build a matmul runner for fixed [M,K] @ [K,N] dimensions. */
export function createMatmul(
  device: GPUDevice,
  M: number,
  K: number,
  N: number,
): MatmulRunner {
  const module = device.createShaderModule({ code: matmulShader });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  const bufA = device.createBuffer({
    size: M * K * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bufB = device.createBuffer({
    size: K * N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bufC = device.createBuffer({
    size: M * N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const bufDims = device.createBuffer({
    size: 16, // 4 x u32 (M, K, N, pad)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const bufRead = device.createBuffer({
    size: M * N * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(bufDims, 0, new Uint32Array([M, K, N, 0]));

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufA } },
      { binding: 1, resource: { buffer: bufB } },
      { binding: 2, resource: { buffer: bufC } },
      { binding: 3, resource: { buffer: bufDims } },
    ],
  });

  return {
    async run(a: Float32Array, b: Float32Array): Promise<Float32Array> {
      // The cast pins the TS 5.7+ typed-array generic to ArrayBuffer; these
      // arrays are always ArrayBuffer-backed, never SharedArrayBuffer.
      device.queue.writeBuffer(bufA, 0, a as Float32Array<ArrayBuffer>);
      device.queue.writeBuffer(bufB, 0, b as Float32Array<ArrayBuffer>);

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      // workgroup_size is 16x16 — one workgroup per 16x16 output tile.
      pass.dispatchWorkgroups(Math.ceil(M / 16), Math.ceil(N / 16));
      pass.end();
      encoder.copyBufferToBuffer(bufC, 0, bufRead, 0, M * N * 4);
      device.queue.submit([encoder.finish()]);

      await bufRead.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(bufRead.getMappedRange().slice(0));
      bufRead.unmap();
      return out;
    },
    destroy() {
      for (const buf of [bufA, bufB, bufC, bufDims, bufRead]) buf.destroy();
    },
  };
}

// ===========================================================================
// Half-precision storage matmul (pack2x16float built-in, no shader-f16 needed)
// ===========================================================================

/**
 * Pack a row-major Float32Array into half-precision pairs (2 × f16 per u32)
 * via pack2x16float semantics. This matches what the WGSL shader reads back
 * with unpack2x16float in matmul_f16packed.wgsl.
 *
 * The last contiguous dimension is assumed to have even length (the runner's
 * caller ensures K and N are even — the kernel asserts the same).
 */
export function packFloat32ToHalfPairs(src: Float32Array): Uint32Array {
  if (src.length % 2 !== 0) {
    throw new Error("packFloat32ToHalfPairs: input length must be even");
  }
  const out = new Uint32Array(src.length / 2);
  // Convert one f32 to its IEEE-754 binary16 (half) representation.
  // Inline the bit-twiddle (no Float16Array on the web yet) — handles NaN,
  // Inf, subnormals, and rounding-to-nearest-even by matching what
  // pack2x16float does on the GPU side.
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  const f32ToF16Bits = (x: number): number => {
    f32[0] = x;
    const bits = u32[0];
    const sign = (bits >>> 16) & 0x8000;
    let exp = (bits >>> 23) & 0xff;
    let mant = bits & 0x7fffff;
    if (exp === 0xff) {
      // NaN | Inf
      return sign | 0x7c00 | (mant ? 0x200 | (mant >>> 13) : 0);
    }
    exp = exp - 127 + 15;
    if (exp >= 0x1f) return sign | 0x7c00;                // overflow → inf
    if (exp <= 0) {
      if (exp < -10) return sign;                          // underflow → 0
      mant = (mant | 0x800000) >>> (1 - exp);
      // round to nearest even
      const rb = 1 << 12;
      if (mant & rb && (mant & (rb - 1) || mant & (rb << 1))) mant += rb;
      return sign | (mant >>> 13);
    }
    // round to nearest even on the truncated 13 LSBs
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
  };
  for (let i = 0; i < out.length; i++) {
    // pack2x16float lays out as (low16 = .x, high16 = .y) — i.e. element 2i
    // goes in the low 16 bits, element 2i+1 in the high 16 bits.
    out[i] = f32ToF16Bits(src[2 * i]) | (f32ToF16Bits(src[2 * i + 1]) << 16);
  }
  return out;
}

/** Half-precision storage matmul runner. Same shape as MatmulRunner; uses
 * roughly half the GPU memory for A and B, and on bandwidth-bound matmuls
 * (which big-model matmuls are on M-series) is measurably faster. */
export function createMatmulF16Packed(
  device: GPUDevice,
  M: number,
  K: number,
  N: number,
): MatmulRunner {
  if (K % 2 !== 0 || N % 2 !== 0) {
    throw new Error(`createMatmulF16Packed: K and N must be even (got K=${K}, N=${N})`);
  }
  const module = device.createShaderModule({ code: matmulF16Shader });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  // A: M*K f16 values = M*K*2 bytes. B: K*N*2 bytes. C: M*N*4 (f32).
  const bufA = device.createBuffer({
    size: M * K * 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bufB = device.createBuffer({
    size: K * N * 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bufC = device.createBuffer({
    size: M * N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const bufDims = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const bufRead = device.createBuffer({
    size: M * N * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(bufDims, 0, new Uint32Array([M, K, N, 0]));

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufA } },
      { binding: 1, resource: { buffer: bufB } },
      { binding: 2, resource: { buffer: bufC } },
      { binding: 3, resource: { buffer: bufDims } },
    ],
  });

  return {
    async run(a: Float32Array, b: Float32Array): Promise<Float32Array> {
      const aPacked = packFloat32ToHalfPairs(a);
      const bPacked = packFloat32ToHalfPairs(b);
      device.queue.writeBuffer(bufA, 0, aPacked as Uint32Array<ArrayBuffer>);
      device.queue.writeBuffer(bufB, 0, bPacked as Uint32Array<ArrayBuffer>);

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(M / 16), Math.ceil(N / 16));
      pass.end();
      encoder.copyBufferToBuffer(bufC, 0, bufRead, 0, M * N * 4);
      device.queue.submit([encoder.finish()]);

      await bufRead.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(bufRead.getMappedRange().slice(0));
      bufRead.unmap();
      return out;
    },
    destroy() {
      for (const buf of [bufA, bufB, bufC, bufDims, bufRead]) buf.destroy();
    },
  };
}

export interface MatmulBenchmark {
  size: number;
  maxAbsError: number;
  gpuMs: number;
  refMs: number;
  speedup: number;
  parityOk: boolean;
}

export interface F16MatmulBenchmark {
  size: number;
  f32GpuMs: number;
  f16GpuMs: number;
  f16Speedup: number; // f32GpuMs / f16GpuMs
  maxAbsError: number; // f16-packed output vs CPU reference
  parityOk: boolean;
}

/**
 * Compare f32-storage and f16-packed-storage matmul kernels on the same
 * problem. f16-packed should be measurably faster on bandwidth-bound matrices
 * (large K against modest M/N) on M-series. Both validated against the CPU
 * reference.
 */
export async function benchmarkMatmulF16(
  device: GPUDevice,
  reference: ReferenceMatmul,
  size = 384,
  iterations = 6,
): Promise<F16MatmulBenchmark> {
  if (size % 2 !== 0) {
    throw new Error(`benchmarkMatmulF16: size must be even (got ${size})`);
  }
  const n = size * size;
  const a = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    // Narrower range than the f32 benchmark — f16 has 5-bit exponent and
    // values outside [-65504, 65504] overflow. Random in [-1, 1] is safe.
    a[i] = Math.random() * 2 - 1;
    b[i] = Math.random() * 2 - 1;
  }

  const f32Runner = createMatmul(device, size, size, size);
  const f16Runner = createMatmulF16Packed(device, size, size, size);
  try {
    // Warm-up.
    await f32Runner.run(a, b);
    const f16Out = await f16Runner.run(a, b);
    const refOut = reference(a, b, size, size, size);

    // f16-packed vs reference — tolerance is wider because storage is half.
    // f16 epsilon ≈ 9.77e-4; matmul accumulates ~K of them, so K * ε is the
    // expected envelope. Use 3× headroom.
    let maxAbsError = 0;
    for (let i = 0; i < n; i++) {
      maxAbsError = Math.max(maxAbsError, Math.abs(f16Out[i] - refOut[i]));
    }
    const tolerance = 3 * size * 9.77e-4; // ~1.13 for size=384

    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) await f32Runner.run(a, b);
    const f32GpuMs = (performance.now() - t0) / iterations;

    const t1 = performance.now();
    for (let i = 0; i < iterations; i++) await f16Runner.run(a, b);
    const f16GpuMs = (performance.now() - t1) / iterations;

    return {
      size,
      f32GpuMs,
      f16GpuMs,
      f16Speedup: f32GpuMs / f16GpuMs,
      maxAbsError,
      parityOk: maxAbsError < tolerance,
    };
  } finally {
    f32Runner.destroy();
    f16Runner.destroy();
  }
}

/** A reference matmul (e.g. the WASM kernel) to check WebGPU output against. */
export type ReferenceMatmul = (
  a: Float32Array,
  b: Float32Array,
  M: number,
  K: number,
  N: number,
) => Float32Array;

/**
 * Run the milestone-6 check: WebGPU matmul vs a reference, on a square matrix.
 * Returns max error and the GPU/reference timings.
 */
export async function benchmarkMatmul(
  device: GPUDevice,
  reference: ReferenceMatmul,
  size = 384,
  iterations = 6,
): Promise<MatmulBenchmark> {
  const n = size * size;
  const a = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    a[i] = Math.random() * 2 - 1;
    b[i] = Math.random() * 2 - 1;
  }

  const runner = createMatmul(device, size, size, size);
  try {
    const gpuOut = await runner.run(a, b); // warm-up + correctness sample
    const refOut = reference(a, b, size, size, size);

    let maxAbsError = 0;
    for (let i = 0; i < n; i++) {
      maxAbsError = Math.max(maxAbsError, Math.abs(gpuOut[i] - refOut[i]));
    }

    const gpuStart = performance.now();
    for (let i = 0; i < iterations; i++) await runner.run(a, b);
    const gpuMs = (performance.now() - gpuStart) / iterations;

    const refStart = performance.now();
    for (let i = 0; i < iterations; i++) reference(a, b, size, size, size);
    const refMs = (performance.now() - refStart) / iterations;

    // Tolerance scales with K: ~K float32 adds accumulate rounding error.
    const tolerance = 1e-3 * size;
    return {
      size,
      maxAbsError,
      gpuMs,
      refMs,
      speedup: refMs / gpuMs,
      parityOk: maxAbsError < tolerance,
    };
  } finally {
    runner.destroy();
  }
}
