/**
 * ops.ts — WebGPU compute kernels for training (Phase 5).
 *
 * Stage 1: matmul forward and backward. `GpuOps` compiles the pipelines from
 * train.wgsl once; each call dispatches a kernel over GpuTensors that stay
 * resident on the GPU. Later stages add layernorm, attention, the optimizer,
 * and a TS orchestrator that wires them into a training loop.
 *
 * Guide: docs/performance.md ("WebGPU — the real ceiling")
 */

import shader from "./train.wgsl?raw";
import { GpuTensor, type GpuContext } from "./tensor";

export class GpuOps {
  private constructor(
    private readonly device: GPUDevice,
    private readonly pipelines: Record<string, GPUComputePipeline>,
  ) {}

  static create(ctx: GpuContext): GpuOps {
    const module = ctx.device.createShaderModule({ code: shader });
    const make = (entryPoint: string) =>
      ctx.device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint },
      });
    return new GpuOps(ctx.device, {
      matmul: make("matmul"),
      matmul_abt: make("matmul_abt"),
      matmul_atb: make("matmul_atb"),
    });
  }

  /** Dispatch one matmul-family kernel. `dims` is (M, K, N); the output C is
   *  [M, N] and one invocation computes one element. */
  private dispatch(
    pipeline: GPUComputePipeline,
    a: GpuTensor,
    b: GpuTensor,
    c: GpuTensor,
    M: number,
    K: number,
    N: number,
  ): void {
    const dims = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(dims, 0, new Uint32Array([M, K, N, 0]));
    const bind = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a.buffer } },
        { binding: 1, resource: { buffer: b.buffer } },
        { binding: 2, resource: { buffer: c.buffer } },
        { binding: 3, resource: { buffer: dims } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(M / 16), Math.ceil(N / 16));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    dims.destroy();
  }

  /** C = A @ B.   A:[M,K]  B:[K,N]  ->  C:[M,N] */
  matmul(a: GpuTensor, b: GpuTensor, M: number, K: number, N: number): GpuTensor {
    const c = new GpuTensor(this.device, M * N, "matmul.C");
    this.dispatch(this.pipelines.matmul, a, b, c, M, K, N);
    return c;
  }

  /**
   * Backward of C = A @ B. Given dC:[M,N], returns
   *   dA = dC @ Bᵀ : [M,K]      dB = Aᵀ @ dC : [K,N]
   */
  matmulBackward(
    a: GpuTensor,
    b: GpuTensor,
    dC: GpuTensor,
    M: number,
    K: number,
    N: number,
  ): { dA: GpuTensor; dB: GpuTensor } {
    // dA = dC @ Bᵀ : an "A times B-transposed" matmul, output [M,K].
    const dA = new GpuTensor(this.device, M * K, "matmul.dA");
    this.dispatch(this.pipelines.matmul_abt, dC, b, dA, M, N, K);
    // dB = Aᵀ @ dC : an "A-transposed times B" matmul, output [K,N].
    const dB = new GpuTensor(this.device, K * N, "matmul.dB");
    this.dispatch(this.pipelines.matmul_atb, a, dC, dB, K, M, N);
    return { dA, dB };
  }
}
