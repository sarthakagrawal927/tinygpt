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

/** Full set of opportunistic WebGPU capabilities we feature-detect at startup.
 *
 * Each flag here gates a faster code path that falls back gracefully when the
 * capability is absent. The capability set is also surfaced in the UI so users
 * can see which accelerated paths are active.
 */
export interface GpuCapabilities {
  /** Adapter `subgroups` feature. Chrome 125+ behind
   *  `chrome://flags#enable-unsafe-webgpu`. Enables subgroupAdd / shuffles. */
  subgroups: boolean;
  /** Adapter `shader-f16` feature. Chrome 121+ stable. Lets WGSL use the
   *  `f16` scalar type directly (compute precision, not just storage). */
  shaderF16: boolean;
  /** Cooperative matrix support: WGSL `enable chromium_experimental_subgroup_matrix`.
   *  Maps to tensor cores (NVIDIA), MFMA (AMD), AMX (Apple). Probed via a
   *  trial shader compile because there's no feature-flag for it as of
   *  May 2026 — only an extension that may or may not parse. Behind
   *  `chrome://flags#enable-unsafe-webgpu` + experimental features. */
  cooperativeMatrix: boolean;
  /** `timestamp-query` adapter feature. Used for in-shader perf profiling
   *  (not on the user hot path; useful for telemetry / capability UI). */
  timestampQuery: boolean;
  /** Adapter info — used to inform capability nudges (e.g., "you're on
   *  Chrome on macOS without `enable-unsafe-webgpu`, here's what you're
   *  missing"). */
  vendor: string;
  architecture: string;
  device: string;
  description: string;
}

export interface GpuContext {
  device: GPUDevice;
  capabilities: GpuCapabilities;
  /** Convenience alias for capabilities.subgroups — kept so older callers
   *  that read ctx.subgroups don't break. */
  subgroups: boolean;
}

/** Try compiling a trial shader that uses `enable chromium_experimental_subgroup_matrix`.
 *  If compilation succeeds we know the device exposes cooperative matrix
 *  primitives. No actual dispatch — pure parse-time check. */
async function probeCooperativeMatrix(device: GPUDevice): Promise<boolean> {
  const trial = `
    enable chromium_experimental_subgroup_matrix;
    @group(0) @binding(0) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(1)
    fn main() { out[0] = 0.0; }
  `;
  try {
    // Mute the console error WebGPU prints on compile failure — we expect
    // most devices to fail this probe.
    const oldOnError = device.onuncapturederror;
    device.onuncapturederror = () => {};
    const mod = device.createShaderModule({ code: trial });
    const info = await mod.getCompilationInfo();
    device.onuncapturederror = oldOnError;
    return !info.messages.some((m) => m.type === "error");
  } catch {
    return false;
  }
}

/** Request a WebGPU device with every opportunistic feature we can get,
 *  and detect all the runtime capabilities. */
export async function createGpuContext(): Promise<GpuContext | null> {
  if (typeof navigator === "undefined" || !navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;

    const wanted = [
      "subgroups",
      "shader-f16",
      "timestamp-query",
    ] as const;
    const requiredFeatures: GPUFeatureName[] = [];
    const supported = {
      subgroups: false,
      shaderF16: false,
      timestampQuery: false,
    };
    for (const name of wanted) {
      if (adapter.features.has(name as GPUFeatureName)) {
        requiredFeatures.push(name as GPUFeatureName);
        if (name === "subgroups") supported.subgroups = true;
        if (name === "shader-f16") supported.shaderF16 = true;
        if (name === "timestamp-query") supported.timestampQuery = true;
      }
    }
    const device = await adapter.requestDevice({ requiredFeatures });

    // Cooperative matrix needs a real shader compile to probe.
    const cooperativeMatrix = await probeCooperativeMatrix(device);

    // Adapter info — vendor/arch/device strings used by the UI. `adapter.info`
    // is the new (sync getter) shape; older Chrome had `requestAdapterInfo()`.
    let vendor = "", architecture = "", deviceName = "", description = "";
    try {
      const adapterAny = adapter as unknown as {
        info?: { vendor?: string; architecture?: string; device?: string; description?: string };
        requestAdapterInfo?: () => Promise<{ vendor?: string; architecture?: string; device?: string; description?: string }>;
      };
      const info = adapterAny.info ?? await adapterAny.requestAdapterInfo?.();
      if (info) {
        vendor = info.vendor ?? "";
        architecture = info.architecture ?? "";
        deviceName = info.device ?? "";
        description = info.description ?? "";
      }
    } catch { /* info getters absent on older Chrome */ }

    const capabilities: GpuCapabilities = {
      subgroups: supported.subgroups,
      shaderF16: supported.shaderF16,
      cooperativeMatrix,
      timestampQuery: supported.timestampQuery,
      vendor, architecture, device: deviceName, description,
    };
    return { device, capabilities, subgroups: supported.subgroups };
  } catch {
    return null;
  }
}

/** WebNN capability — separate from WebGPU. Lives on `navigator.ml`.
 *  Routes to OS NN runtime (CoreML on macOS, DirectML on Windows). Used
 *  by the inference path; training stays on WebGPU. */
export interface WebNNCapabilities {
  available: boolean;
  /** Whether GPU-device context can be created (the fast path). */
  gpuContext: boolean;
  /** Whether NPU-device context can be created — Apple Neural Engine on
   *  Apple Silicon, NPU on Snapdragon, etc. */
  npuContext: boolean;
}

export async function probeWebNN(): Promise<WebNNCapabilities> {
  const result: WebNNCapabilities = { available: false, gpuContext: false, npuContext: false };
  const ml = (navigator as unknown as { ml?: { createContext?: (opts?: object) => Promise<unknown> } }).ml;
  if (!ml || typeof ml.createContext !== "function") return result;
  result.available = true;
  try {
    const ctx = await ml.createContext({ deviceType: "gpu" });
    if (ctx) result.gpuContext = true;
  } catch { /* GPU device unavailable */ }
  try {
    const ctx = await ml.createContext({ deviceType: "npu" });
    if (ctx) result.npuContext = true;
  } catch { /* NPU device unavailable */ }
  return result;
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

  /** Destroy every pooled buffer. Called as part of GpuOps.destroy() during
   *  auto-offload-after-idle and other teardown paths. After this, the pool
   *  is empty and acquire() will allocate fresh buffers if used again. */
  destroyAll(): void {
    for (const list of this.free.values()) {
      for (const buf of list) buf.destroy();
    }
    this.free.clear();
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
