/**
 * tensor.ts — GPU-resident float32 tensors for WebGPU training (Phase 5).
 *
 * The point of training on the GPU is to keep every intermediate *on* the GPU
 * between ops — uploading and downloading each step would erase the speed-up.
 * `GpuTensor` wraps one storage buffer; the kernels in `ops.ts` read and write
 * these without round-tripping through JavaScript.
 *
 * Guide: docs/browser_notes.md ("WebGPU acceleration"), docs/performance.md
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

/** A flat float32 tensor living in a GPU storage buffer. */
export class GpuTensor {
  readonly buffer: GPUBuffer;
  /** byte length, rounded up to a multiple of 4 and at least 4 */
  private readonly bytes: number;

  constructor(
    private readonly device: GPUDevice,
    readonly size: number, // element count
    label?: string,
  ) {
    this.bytes = Math.max(4, size * 4);
    this.buffer = device.createBuffer({
      label,
      size: this.bytes,
      usage:
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  /** Copy host data into the tensor. */
  upload(data: Float32Array): void {
    this.device.queue.writeBuffer(this.buffer, 0, data as Float32Array<ArrayBuffer>);
  }

  /** Read the tensor back to the host (via a staging buffer — STORAGE buffers
   *  cannot be mapped directly). */
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

  /** Convenience: allocate a tensor and fill it from host data. */
  static fromData(device: GPUDevice, data: Float32Array, label?: string): GpuTensor {
    const t = new GpuTensor(device, data.length, label);
    t.upload(data);
    return t;
  }

  destroy(): void {
    this.buffer.destroy();
  }
}
