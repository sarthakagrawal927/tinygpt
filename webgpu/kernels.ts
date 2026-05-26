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
import matmulTiledShader from "./matmul_tiled.wgsl?raw";
import matmulF16Shader from "./matmul_f16packed.wgsl?raw";
import matmulTiledF16Shader from "./matmul_tiled_f16.wgsl?raw";

/** A matmul runner bound to fixed dimensions — pipeline + buffers built once.
 *
 * For benchmarking on bandwidth-bound shapes, callers want to pay the upload
 * (and, for f16-packed, the pack) cost ONCE and then time the dispatch in a
 * tight loop. `uploadInputs` does the once-only work; `dispatch` runs the
 * kernel using already-uploaded inputs. `run` is the convenience that does
 * both for one-shot callers. */
export interface MatmulRunner {
  /** Upload A and B to GPU buffers. For f16-packed runners this also packs. */
  uploadInputs(a: Float32Array, b: Float32Array): void;
  /** Dispatch the kernel; returns C using inputs uploaded earlier. */
  dispatch(): Promise<Float32Array>;
  /** Convenience: upload + dispatch in one call. */
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

  const uploadInputs = (a: Float32Array, b: Float32Array) => {
    // The cast pins the TS 5.7+ typed-array generic to ArrayBuffer; these
    // arrays are always ArrayBuffer-backed, never SharedArrayBuffer.
    device.queue.writeBuffer(bufA, 0, a as Float32Array<ArrayBuffer>);
    device.queue.writeBuffer(bufB, 0, b as Float32Array<ArrayBuffer>);
  };
  const dispatch = async (): Promise<Float32Array> => {
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
  };
  return {
    uploadInputs,
    dispatch,
    async run(a: Float32Array, b: Float32Array): Promise<Float32Array> {
      uploadInputs(a, b);
      return dispatch();
    },
    destroy() {
      for (const buf of [bufA, bufB, bufC, bufDims, bufRead]) buf.destroy();
    },
  };
}

// ===========================================================================
// Tiled matmul — workgroup-shared memory, identical bind layout to naive
// ===========================================================================

/** Workgroup-shared-memory tiled matmul runner. Same shape as createMatmul,
 * different shader. On bandwidth-bound problems this avoids `size` global
 * reads per output element; on small problems the extra synchronisation
 * makes it a wash. */
export function createMatmulTiled(
  device: GPUDevice,
  M: number,
  K: number,
  N: number,
): MatmulRunner {
  const module = device.createShaderModule({ code: matmulTiledShader });
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

  const uploadInputs = (a: Float32Array, b: Float32Array) => {
    device.queue.writeBuffer(bufA, 0, a as Float32Array<ArrayBuffer>);
    device.queue.writeBuffer(bufB, 0, b as Float32Array<ArrayBuffer>);
  };
  const dispatch = async (): Promise<Float32Array> => {
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
  };
  return {
    uploadInputs,
    dispatch,
    async run(a: Float32Array, b: Float32Array): Promise<Float32Array> {
      uploadInputs(a, b);
      return dispatch();
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

/** Tiled + f16-packed combined runner — stacks both wins. */
export function createMatmulTiledF16(
  device: GPUDevice,
  M: number,
  K: number,
  N: number,
): MatmulRunner {
  if (K % 2 !== 0 || N % 2 !== 0) {
    throw new Error(`createMatmulTiledF16: K and N must be even (got K=${K}, N=${N})`);
  }
  const module = device.createShaderModule({ code: matmulTiledF16Shader });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

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

  const uploadInputs = (a: Float32Array, b: Float32Array) => {
    const aPacked = packFloat32ToHalfPairs(a);
    const bPacked = packFloat32ToHalfPairs(b);
    device.queue.writeBuffer(bufA, 0, aPacked as Uint32Array<ArrayBuffer>);
    device.queue.writeBuffer(bufB, 0, bPacked as Uint32Array<ArrayBuffer>);
  };
  const dispatch = async (): Promise<Float32Array> => {
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
  };
  return {
    uploadInputs,
    dispatch,
    async run(a: Float32Array, b: Float32Array): Promise<Float32Array> {
      uploadInputs(a, b);
      return dispatch();
    },
    destroy() {
      for (const buf of [bufA, bufB, bufC, bufDims, bufRead]) buf.destroy();
    },
  };
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

  const uploadInputs = (a: Float32Array, b: Float32Array) => {
    // Pack happens here — this is the once-per-weights cost. In a real
    // training loop this runs at upload time, not per step. The benchmark
    // calls uploadInputs() outside the timed loop so we measure only the
    // GPU dispatch cost f16-packed actually pays per step.
    const aPacked = packFloat32ToHalfPairs(a);
    const bPacked = packFloat32ToHalfPairs(b);
    device.queue.writeBuffer(bufA, 0, aPacked as Uint32Array<ArrayBuffer>);
    device.queue.writeBuffer(bufB, 0, bPacked as Uint32Array<ArrayBuffer>);
  };
  const dispatch = async (): Promise<Float32Array> => {
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
  };
  return {
    uploadInputs,
    dispatch,
    async run(a: Float32Array, b: Float32Array): Promise<Float32Array> {
      uploadInputs(a, b);
      return dispatch();
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
  /** Pure GPU dispatch cost — inputs are uploaded outside the timed loop. */
  f32GpuMs: number;
  /** Tiled-naive (f32, workgroup-shared memory) version. */
  tiledGpuMs: number;
  /** Half-precision-storage version. */
  f16GpuMs: number;
  /** Tiled + half-precision combined version. */
  tiledF16GpuMs: number;
  /** Best of the four, in ms. */
  bestGpuMs: number;
  bestVariant: "f32" | "tiled" | "f16" | "tiled-f16";
  /** Speedup of the best variant vs naive f32. */
  bestSpeedup: number;
  maxAbsError: number; // worst case across variants vs CPU reference
  parityOk: boolean;
}

/**
 * Compare f32-storage and f16-packed-storage matmul kernels on the same
 * problem. KEY DESIGN: inputs are uploaded (and for f16-packed, packed)
 * OUTSIDE the timed loop. In a real training pipeline weights pack once and
 * stay packed for thousands of steps, so charging per-step pack cost (as the
 * original version did) is unfair and gives a misleading picture.
 *
 * The timing covers just dispatch + sync + map-back — the real per-step cost.
 *
 * Tested at multiple sizes to expose the crossover: WebGPU is overhead-bound
 * on small matmuls (~hundreds of µs of dispatch + sync regardless of size)
 * and bandwidth-bound on big ones. f16-packed pays off only past the point
 * where the matmul is bandwidth-bound.
 */
export async function benchmarkMatmulF16(
  device: GPUDevice,
  reference: ReferenceMatmul,
  size = 1024,
  iterations = 8,
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
  const tiledRunner = createMatmulTiled(device, size, size, size);
  const f16Runner = createMatmulF16Packed(device, size, size, size);
  const tiledF16Runner = createMatmulTiledF16(device, size, size, size);
  try {
    // Upload OUTSIDE the timed loop — this is the once-per-weights cost.
    f32Runner.uploadInputs(a, b);
    tiledRunner.uploadInputs(a, b);
    f16Runner.uploadInputs(a, b);
    tiledF16Runner.uploadInputs(a, b);

    // Warm-up: one dispatch each.
    await f32Runner.dispatch();
    const tiledOut = await tiledRunner.dispatch();
    const f16Out = await f16Runner.dispatch();
    const tiledF16Out = await tiledF16Runner.dispatch();

    // Parity vs CPU reference — only at small sizes (CPU is slow).
    let maxAbsError = 0;
    if (size <= 512) {
      const refOut = reference(a, b, size, size, size);
      void tiledOut;
      // Report the worst error across the two f16 variants.
      for (let i = 0; i < n; i++) {
        const e1 = Math.abs(f16Out[i] - refOut[i]);
        const e2 = Math.abs(tiledF16Out[i] - refOut[i]);
        if (e1 > maxAbsError) maxAbsError = e1;
        if (e2 > maxAbsError) maxAbsError = e2;
      }
    }
    const tolerance = 3 * size * 9.77e-4;

    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) await f32Runner.dispatch();
    const f32GpuMs = (performance.now() - t0) / iterations;

    const t1 = performance.now();
    for (let i = 0; i < iterations; i++) await tiledRunner.dispatch();
    const tiledGpuMs = (performance.now() - t1) / iterations;

    const t2 = performance.now();
    for (let i = 0; i < iterations; i++) await f16Runner.dispatch();
    const f16GpuMs = (performance.now() - t2) / iterations;

    const t3 = performance.now();
    for (let i = 0; i < iterations; i++) await tiledF16Runner.dispatch();
    const tiledF16GpuMs = (performance.now() - t3) / iterations;

    const candidates: Array<{ ms: number; variant: F16MatmulBenchmark["bestVariant"] }> = [
      { ms: f32GpuMs, variant: "f32" },
      { ms: tiledGpuMs, variant: "tiled" },
      { ms: f16GpuMs, variant: "f16" },
      { ms: tiledF16GpuMs, variant: "tiled-f16" },
    ];
    candidates.sort((a, b) => a.ms - b.ms);
    const best = candidates[0];

    return {
      size,
      f32GpuMs,
      tiledGpuMs,
      f16GpuMs,
      tiledF16GpuMs,
      bestGpuMs: best.ms,
      bestVariant: best.variant,
      bestSpeedup: f32GpuMs / best.ms,
      maxAbsError,
      parityOk: size > 512 || maxAbsError < tolerance,
    };
  } finally {
    f32Runner.destroy();
    tiledRunner.destroy();
    f16Runner.destroy();
    tiledF16Runner.destroy();
  }
}

/**
 * Sweep the f16/f32 benchmark across realistic matmul sizes to expose the
 * crossover point — below it WebGPU is dispatch-bound, above it bandwidth
 * starts to matter and f16-packed pays off.
 */
export async function benchmarkMatmulF16Sweep(
  device: GPUDevice,
  reference: ReferenceMatmul,
  sizes: number[] = [256, 512, 1024, 2048],
): Promise<F16MatmulBenchmark[]> {
  const out: F16MatmulBenchmark[] = [];
  for (const s of sizes) {
    out.push(await benchmarkMatmulF16(device, reference, s, s >= 2048 ? 4 : 8));
  }
  return out;
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
