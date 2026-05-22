/**
 * tensor.ts — GPU-resident float32 tensors + a buffer pool (Phase 5).
 *
 * Training reuses the same set of tensor shapes every step. Allocating fresh
 * GPU buffers each step and destroying them is the dominant cost — so per-step
 * ("scratch") tensors draw their buffers from a `BufferPool`: at the end of a
 * step they are returned to the pool, and the next step reuses them. After the
 * first step, a steady run does zero buffer allocation.
 *
 * Weights and optimizer moments are NOT pooled — they live for the whole run.
 *
 * Guide: docs/performance.md ("WebGPU training")
 */

export interface GpuContext {
  device: GPUDevice;
}

/** Request a WebGPU device, or null if the browser/platform has none. */
export async function createGpuContext(): Promise<GpuContext | null> {
  if (typeof navigator === "undefined" || !navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    return { device: await adapter.requestDevice() };
  } catch {
    return null;
  }
}

/** Recycles storage buffers, keyed by byte size, so steps reuse them. */
export class BufferPool {
  private readonly free = new Map<number, GPUBuffer[]>();

  constructor(private readonly device: GPUDevice) {}

  acquire(bytes: number): GPUBuffer {
    const list = this.free.get(bytes);
    if (list && list.length > 0) return list.pop() as GPUBuffer;
    return this.device.createBuffer({
      size: bytes,
      usage:
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  release(buffer: GPUBuffer): void {
    let list = this.free.get(buffer.size);
    if (!list) {
      list = [];
      this.free.set(buffer.size, list);
    }
    list.push(buffer);
  }
}

/** A flat float32 tensor living in a GPU storage buffer. */
export class GpuTensor {
  readonly buffer: GPUBuffer;
  private readonly bytes: number;
  private readonly pool: BufferPool | null;

  constructor(
    private readonly device: GPUDevice,
    readonly size: number, // element count
    opts?: { pool?: BufferPool; label?: string },
  ) {
    this.bytes = Math.max(4, size * 4);
    this.pool = opts?.pool ?? null;
    this.buffer = this.pool
      ? this.pool.acquire(this.bytes)
      : device.createBuffer({
          label: opts?.label,
          size: this.bytes,
          usage:
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
  }

  upload(data: Float32Array): void {
    this.device.queue.writeBuffer(this.buffer, 0, data as Float32Array<ArrayBuffer>);
  }

  /** Read the tensor back to the host (via a staging buffer). */
  async download(): Promise<Float32Array> {
    const staging = this.device.createBuffer({
      size: this.bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffer, 0, staging, 0, this.bytes);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0, this.size * 4));
    staging.unmap();
    staging.destroy();
    return out;
  }

  /** Allocate a (non-pooled) tensor and fill it — used for persistent weights. */
  static fromData(device: GPUDevice, data: Float32Array, label?: string): GpuTensor {
    const t = new GpuTensor(device, data.length, { label });
    t.upload(data);
    return t;
  }

  /** Return the buffer to its pool (if pooled), else free it. */
  recycle(): void {
    if (this.pool) this.pool.release(this.buffer);
    else this.buffer.destroy();
  }

  destroy(): void {
    this.buffer.destroy();
  }
}
