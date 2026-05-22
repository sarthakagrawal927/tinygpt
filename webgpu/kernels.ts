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

export interface MatmulBenchmark {
  size: number;
  maxAbsError: number;
  gpuMs: number;
  refMs: number;
  speedup: number;
  parityOk: boolean;
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
