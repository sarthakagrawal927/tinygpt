/**
 * ops.ts — WebGPU compute kernels for training (Phase 5).
 *
 * One shader module (train.wgsl), one bind-group layout (six storage buffers +
 * a params uniform). Each method allocates its output GpuTensors, fills the
 * params, and dispatches — adding a kernel never touches the plumbing.
 *
 * Stages 1-2: matmul fwd/bwd, the elementwise ops, GELU, and layernorm.
 * Later: attention, the optimizer, and a training orchestrator.
 *
 * Guide: docs/performance.md ("WebGPU — the real ceiling")
 */

import shader from "./train.wgsl?raw";
import sgShader from "./train_sg.wgsl?raw";
import vec4Shader from "./train_vec4.wgsl?raw";
import f16Shader from "./train_f16.wgsl?raw";
import fa2Shader from "./attention_fa2.wgsl?raw";
import { BufferPool, GpuTensor, type GpuContext } from "./tensor";

const ENTRIES = [
  "matmul", "matmul_blocked", "matmul_abt", "matmul_abt_blocked",
  "matmul_atb", "matmul_atb_blocked", "add", "bias_add",
  "bias_grad", "gelu_forward", "gelu_backward", "layernorm_forward",
  "layernorm_dx", "layernorm_dgb", "attn_softmax", "attn_value", "attn_fused_sv", "attn_dscores",
  "attn_dscores_fa2", "attn_dv_fa2",
  "attn_dq", "attn_dk", "attn_dv", "embed_forward", "embed_tok_grad",
  "embed_pos_grad", "cross_entropy", "adamw",
] as const;
/** Entry points that live in train_sg.wgsl and require the WebGPU
 *  `subgroups` feature on the adapter. */
const SG_ENTRIES = ["layernorm_forward_sg", "cross_entropy_sg"] as const;
/** Entry points that live in train_vec4.wgsl — same g0-g5+p binding layout
 * from the host side, but g0/g1 are declared as array<vec4<f32>> on the
 * WGSL side for 128-bit aligned global loads. Requires K and N to be
 * multiples of 4. */
const VEC4_ENTRIES = ["matmul_blocked_vec4"] as const;
/** Entry points that live in train_f16.wgsl — same bind layout, but g1 is
 *  declared as array<u32> for packed-f16 weight storage. K and N must be
 *  even. Gated on a startup numerics check (see verifyF16Storage). */
const F16_ENTRIES = ["matmul_blocked_f16", "matmul_abt_blocked_f16", "pack_to_f16"] as const;
/** Flash-Attention-2-style fused attention forward. Same bind layout as
 * train.wgsl (g0=q g1=k g2=v g3=attn g4=ctx, p.a=B p.b=T p.c=C p.d=H
 * p.fa=1/sqrt(hd)). Workgroup-cooperative — one workgroup per
 * (b, h, ceil(T/16)) — and runs online softmax in registers across K
 * blocks. The kernel still writes the [B,H,T,T] attn matrix so the
 * existing backward kernels stay unchanged; that writeback drops when
 * FA2 backward (recompute on backward) lands. */
const FA2_ENTRIES = ["fa2_forward"] as const;
type Entry =
  | (typeof ENTRIES)[number]
  | (typeof SG_ENTRIES)[number]
  | (typeof VEC4_ENTRIES)[number]
  | (typeof F16_ENTRIES)[number]
  | (typeof FA2_ENTRIES)[number];

/** Params uniform: up to four u32 (dims) and four f32 (eps, scale, ...). */
interface Params {
  a?: number; b?: number; c?: number; d?: number;
  fa?: number; fb?: number; fc?: number; fd?: number;
}

export class GpuOps {
  // A whole training step records into one command encoder and submits once —
  // hundreds of per-kernel submits per step was the real bottleneck.
  private encoder: GPUCommandEncoder | null = null;
  private pass: GPUComputePassEncoder | null = null;
  // One uniform buffer per dispatch in a batch (a batched submit means every
  // writeBuffer lands before the submit, so dispatches cannot share one).
  private readonly uniforms: GPUBuffer[] = [];
  private uniformIdx = 0;

  private constructor(
    private readonly device: GPUDevice,
    private readonly layout: GPUBindGroupLayout,
    private readonly pipelines: Partial<Record<Entry, GPUComputePipeline>>,
    // One distinct dummy per slot — WebGPU forbids binding the same buffer to
    // two writable-storage bindings in a bind group (aliasing).
    private readonly dummies: GPUBuffer[],
    // Pool for per-step scratch buffers.
    readonly pool: BufferPool,
    /** True iff the device gave us subgroups (train_sg.wgsl pipelines exist). */
    readonly hasSubgroups: boolean,
    /** True iff the f16-storage matmul kernel both compiled AND passed the
     *  startup numerics gate. Defaults false; flipped by verifyF16Storage()
     *  during create(). When false, callers fall back to the f32 vec4 path. */
    public f16StorageActive: boolean,
  ) {}

  /** Start recording a batch of dispatches into a single command buffer. */
  beginBatch(): void {
    this.encoder = this.device.createCommandEncoder();
    this.pass = this.encoder.beginComputePass();
    this.uniformIdx = 0;
  }

  /** Finish the batch — submit every recorded dispatch in one go. */
  endBatch(): void {
    if (!this.pass || !this.encoder) return;
    this.pass.end();
    this.device.queue.submit([this.encoder.finish()]);
    this.pass = null;
    this.encoder = null;
  }

  private nextUniform(): GPUBuffer {
    if (this.uniformIdx >= this.uniforms.length) {
      this.uniforms.push(this.device.createBuffer({
        size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }));
    }
    return this.uniforms[this.uniformIdx++];
  }

  /** Promise resolving when the f16-storage numerics gate has finished.
   *  Resolves to the verdict: true iff the f16 path passed the check.
   *  Settable via the static factory; callers await this before relying on
   *  f16StorageActive being its final value. */
  public f16Ready: Promise<boolean> = Promise.resolve(false);

  static create(ctx: GpuContext): GpuOps {
    const device = ctx.device;
    const module = device.createShaderModule({ code: shader });

    const storage = (binding: number): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    });
    const layout = device.createBindGroupLayout({
      entries: [
        storage(0), storage(1), storage(2), storage(3), storage(4), storage(5),
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });

    const pipelines: Partial<Record<Entry, GPUComputePipeline>> = {};
    for (const entryPoint of ENTRIES) {
      pipelines[entryPoint] = device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module, entryPoint },
      });
    }

    // Subgroup-using variants compiled separately — train_sg.wgsl uses
    // `enable subgroups;` which only validates on devices that advertise
    // the feature. Skip compilation entirely otherwise.
    if (ctx.subgroups) {
      const sgModule = device.createShaderModule({ code: sgShader });
      for (const sgEntry of SG_ENTRIES) {
        pipelines[sgEntry] = device.createComputePipeline({
          layout: pipelineLayout,
          compute: { module: sgModule, entryPoint: sgEntry },
        });
      }
    }

    // Vec4-loaded matmul variants — same bind layout, vec4 inner type for
    // g0/g1. Always available (no device feature needed). Used when K and N
    // happen to be multiples of 4 (which is true for all preset shapes).
    const vec4Module = device.createShaderModule({ code: vec4Shader });
    for (const v4Entry of VEC4_ENTRIES) {
      pipelines[v4Entry] = device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: vec4Module, entryPoint: v4Entry },
      });
    }

    // F16-storage matmul + the pack helper. Always compiles (uses core WGSL
    // pack2x16float / unpack2x16float; no `enable f16;` extension needed),
    // but only activates after the numerics gate at the end of create().
    const f16Module = device.createShaderModule({ code: f16Shader });
    for (const f16Entry of F16_ENTRIES) {
      pipelines[f16Entry] = device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: f16Module, entryPoint: f16Entry },
      });
    }

    // FA2 fused attention forward — workgroup-cooperative, online softmax.
    // Separate module because the WGSL declares a workgroup-scope Q tile
    // sized for hd ≤ MAX_HD that train.wgsl doesn't carry.
    const fa2Module = device.createShaderModule({ code: fa2Shader });
    for (const fa2Entry of FA2_ENTRIES) {
      pipelines[fa2Entry] = device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: fa2Module, entryPoint: fa2Entry },
      });
    }

    // A distinct 1-element buffer for each slot a kernel leaves unused.
    const dummies: GPUBuffer[] = [];
    for (let i = 0; i < 6; i++) {
      dummies.push(device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }));
    }
    const ops = new GpuOps(
      device, layout, pipelines, dummies, new BufferPool(device),
      ctx.subgroups, /* f16StorageActive */ false,
    );

    // Kick off the numerics gate in the background. Until it settles, all
    // matmuls use the f32 vec4 path. Once it passes, f16StorageActive
    // flips to true and subsequent generate() / training calls can opt in.
    ops.f16Ready = ops.verifyF16Storage().catch((err) => {
      console.warn("[ops] f16-storage numerics gate threw:", err);
      return false;
    });

    return ops;
  }

  /** Numerics gate for the f16-storage matmul path.
   *
   *  Runs a representative matmul (small enough to be fast, large enough K
   *  that fp16 rounding has room to compound) on both the f32 vec4 path and
   *  the new f16 packed-storage path. Compares element-wise against a
   *  magnitude-aware tolerance so individual near-zero outputs can't
   *  inflate the relative-error metric. Two thresholds, both must hold:
   *
   *    max_abs_err < 1% of mean |reference|
   *    mean_rel < 0.5%, where rel uses denom = max(|ref|, 1% of mean |ref|)
   *
   *  The "1% of mean |reference|" floor is what makes the gate sane — it
   *  reflects "would the downstream network notice this?" rather than
   *  "did the bit-exact value match?". A single output of magnitude 1e-6
   *  diverging by 1e-4 is harmless (the activations / softmax that follow
   *  are insensitive to that scale); only systematic / large-magnitude
   *  errors should reject the path.
   *
   *  Worst-case theoretical f16 dot-product accuracy: ~sqrt(K) × eps_f16
   *  ≈ sqrt(128) × 5e-4 ≈ 5.6e-3 RMS. So the 0.5% mean threshold is right
   *  at the theoretical edge — a path that fails this is likely a real
   *  bug, not normal f16 drift. */
  private async verifyF16Storage(): Promise<boolean> {
    const M = 64, K = 128, N = 128;
    if (!this.pipelines["matmul_blocked_f16"] || !this.pipelines["pack_to_f16"]) {
      return false;
    }

    // Deterministic inputs: small magnitudes so f16 underflow isn't an issue.
    const aData = new Float32Array(M * K);
    const bData = new Float32Array(K * N);
    let seed = 12345;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return (seed / 0x7fffffff) * 0.4 - 0.2; // [-0.2, 0.2] — typical weight scale
    };
    for (let i = 0; i < aData.length; i++) aData[i] = rand();
    for (let i = 0; i < bData.length; i++) bData[i] = rand();

    const a = new GpuTensor(this.device, M * K);
    a.upload(aData);
    const b = new GpuTensor(this.device, K * N);
    b.upload(bData);

    // f32 reference via the existing matmul path (matmul_blocked_vec4).
    const cF32 = this.matmul(a, b, M, K, N);
    const refOut = await cF32.download();

    // Pack B to f16 storage in a fresh GPU buffer.
    const packedBytes = (K * N) * 2; // K*N/2 u32 = K*N halfs = K*N*2 bytes
    const bPackedBuf = this.device.createBuffer({
      size: packedBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.packToF16(b.buffer, bPackedBuf, K, N);

    // f16-storage forward matmul.
    const cF16 = this.matmulF16Weight(a, bPackedBuf, M, K, N);
    const f16Out = await cF16.download();

    // Compare forward outputs.
    const fwdResult = this.compareF16Output(refOut, f16Out, "fwd");
    cF16.recycle();

    // --- Backward matmul check: dA = dC @ B^T ---
    // Use the SAME packed buffer (the packing is layout-agnostic between
    // the forward matmul and matmulAbt variants — both index the same
    // packed-along-N storage, just with different row/col interpretations).
    // Build a fresh A_bwd to keep the gate independent of the forward run.
    const Mb = 64, Kb = 128, Nb = 64; // matmulAbt: output [Mb, Nb], inner Kb
    // For matmulAbtF16Weight to share a packed B buffer, we need K (= Kb) to
    // match the packing axis of the original weight (= N of the [K, N] view).
    // So the "weight" we pack is shape [Nb, Kb] with Kb as the contiguous /
    // packed axis. Reuse bData but re-pack with that shape.
    const aBwdData = new Float32Array(Mb * Kb);
    const wBwdData = new Float32Array(Nb * Kb);
    for (let i = 0; i < aBwdData.length; i++) aBwdData[i] = rand();
    for (let i = 0; i < wBwdData.length; i++) wBwdData[i] = rand();

    const aBwd = new GpuTensor(this.device, Mb * Kb);
    aBwd.upload(aBwdData);
    const wBwd = new GpuTensor(this.device, Nb * Kb);
    wBwd.upload(wBwdData);

    // f32 reference: matmulAbt(aBwd, wBwd) = aBwd @ wBwd^T → [Mb, Nb]
    const cAbtF32 = this.matmulAbt(aBwd, wBwd, Mb, Kb, Nb);
    const abtRefOut = await cAbtF32.download();
    cAbtF32.recycle();

    // Pack wBwd along its second axis (Kb), matching what the backward
    // kernel expects (it indexes packed by [bRow * halfK + bCol/2]).
    const wBwdPackedBuf = this.device.createBuffer({
      size: Nb * Kb * 2,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.packToF16(wBwd.buffer, wBwdPackedBuf, Nb, Kb);

    const cAbtF16 = this.matmulAbtF16Weight(aBwd, wBwdPackedBuf, Mb, Kb, Nb);
    const abtF16Out = await cAbtF16.download();
    const bwdResult = this.compareF16Output(abtRefOut, abtF16Out, "bwd");

    aBwd.destroy();
    wBwd.destroy();
    wBwdPackedBuf.destroy();
    cAbtF16.recycle();

    // Both forward AND backward must pass for the f16 path to activate.
    const passed = fwdResult.passed && bwdResult.passed;
    console.info(
      `[ops] f16-storage gate (fwd): ${fwdResult.summary}`,
    );
    console.info(
      `[ops] f16-storage gate (bwd): ${bwdResult.summary}`,
    );
    console.info(
      `[ops] f16-storage gate verdict: ${passed ? "PASS — f16 path active" : "FAIL — staying on f32"}`,
    );

    this.f16StorageActive = passed;
    a.destroy();
    b.destroy();
    bPackedBuf.destroy();
    cF32.recycle();
    return passed;
  }

  /** Shared comparison helper for the f16 gate. Magnitude-aware tolerance,
   *  same thresholds for both forward and backward. */
  private compareF16Output(
    refOut: Float32Array, f16Out: Float32Array, label: string,
  ): { passed: boolean; summary: string } {
    let sumAbsRef = 0;
    for (let i = 0; i < refOut.length; i++) sumAbsRef += Math.abs(refOut[i]);
    const meanAbsRef = sumAbsRef / refOut.length;
    const denomFloor = Math.max(meanAbsRef * 0.01, 1e-6);
    let maxAbs = 0, maxRel = 0, sumRel = 0;
    for (let i = 0; i < refOut.length; i++) {
      const r = refOut[i];
      const f = f16Out[i];
      const absErr = Math.abs(f - r);
      const denom = Math.max(Math.abs(r), denomFloor);
      const rel = absErr / denom;
      if (rel > maxRel) maxRel = rel;
      if (absErr > maxAbs) maxAbs = absErr;
      sumRel += rel;
    }
    const meanRel = sumRel / refOut.length;
    const maxAbsThreshold = meanAbsRef * 0.01;
    const passed = maxAbs < maxAbsThreshold && meanRel < 5e-3;
    const summary =
      `mean|ref|=${meanAbsRef.toExponential(2)}, ` +
      `max_abs=${maxAbs.toExponential(2)} (limit ${maxAbsThreshold.toExponential(2)}), ` +
      `mean_rel=${(meanRel * 100).toFixed(3)}% (limit 0.500%), ` +
      `max_rel=${(maxRel * 100).toFixed(2)}% — ` +
      `${passed ? "PASS" : "FAIL"} [${label}]`;
    return { passed, summary };
  }

  private newTensor(size: number, label: string): GpuTensor {
    return new GpuTensor(this.device, size, { pool: this.pool, label });
  }

  /** A pooled tensor filled with host data — for per-step inputs (ids/targets). */
  upload(data: Float32Array): GpuTensor {
    const t = new GpuTensor(this.device, data.length, { pool: this.pool });
    t.upload(data);
    return t;
  }

  /** Dispatch one kernel. `buffers` fill g0..g5 in order; unused slots get a
   *  dummy. `wgX`/`wgY` are workgroup counts. */
  private dispatch(
    entry: Entry,
    buffers: GpuTensor[],
    params: Params,
    wgX: number,
    wgY = 1,
  ): void {
    // A standalone call (e.g. a parity test) records and submits on its own;
    // inside beginBatch/endBatch it just records into the shared pass.
    const ownBatch = this.pass === null;
    if (ownBatch) this.beginBatch();

    const u = new ArrayBuffer(32);
    const dv = new DataView(u);
    dv.setUint32(0, params.a ?? 0, true);
    dv.setUint32(4, params.b ?? 0, true);
    dv.setUint32(8, params.c ?? 0, true);
    dv.setUint32(12, params.d ?? 0, true);
    dv.setFloat32(16, params.fa ?? 0, true);
    dv.setFloat32(20, params.fb ?? 0, true);
    dv.setFloat32(24, params.fc ?? 0, true);
    dv.setFloat32(28, params.fd ?? 0, true);
    const ubuf = this.nextUniform();
    this.device.queue.writeBuffer(ubuf, 0, u);

    const entries: GPUBindGroupEntry[] = [];
    for (let i = 0; i < 6; i++) {
      entries.push({
        binding: i,
        resource: {
          buffer: i < buffers.length ? buffers[i].buffer : this.dummies[i],
        },
      });
    }
    entries.push({ binding: 6, resource: { buffer: ubuf } });
    const bind = this.device.createBindGroup({ layout: this.layout, entries });

    const pass = this.pass as GPUComputePassEncoder;
    const pipeline = this.pipelines[entry];
    if (!pipeline) throw new Error(`pipeline missing for ${entry} (subgroups not available?)`);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(wgX, wgY);

    if (ownBatch) this.endBatch();
  }

  // --- matmul --------------------------------------------------------------
  /** C = A @ B.   A:[M,K]  B:[K,N]  ->  [M,N]
   *
   * Uses the thread-blocked variant (matmul_blocked) — 5.18× faster than the
   * naive kernel at 2048³ on M-series WebGPU. Workgroup is 16×16 threads but
   * each thread computes a 4×4 register block of output, so workgroup
   * dispatch is ceil(M/64) × ceil(N/64). The original naive `matmul` kernel
   * is kept in train.wgsl as a reference / fallback. */
  matmul(a: GpuTensor, b: GpuTensor, M: number, K: number, N: number): GpuTensor {
    const c = this.newTensor(M * N, "matmul.C");
    // Prefer vec4-loaded variant when K and N are 4-aligned (always true
    // for preset shapes). Measured 1.37× faster than scalar blocked4 at
    // 2048³ standalone. (First integration attempt hit a WGSL access-mode
    // mismatch with the shared bind-group layout — see train_vec4.wgsl
    // comment block.)
    const vec4Ok = (K % 4 === 0) && (N % 4 === 0);
    const entry = vec4Ok ? "matmul_blocked_vec4" : "matmul_blocked";
    this.dispatch(entry, [a, b, c], { a: M, b: K, c: N },
      Math.ceil(M / 64), Math.ceil(N / 64));
    return c;
  }

  /** C = A @ Bᵀ.   A:[M,K]  B:[N,K]  ->  [M,N]   (also the tied output head).
   *
   * Uses the thread-blocked variant — same 4×4-register + 64×64-workgroup tile
   * pattern as matmul, adapted to B's [N,K] layout. */
  matmulAbt(a: GpuTensor, b: GpuTensor, M: number, K: number, N: number): GpuTensor {
    const c = this.newTensor(M * N, "matmul.abt");
    this.dispatch("matmul_abt_blocked", [a, b, c], { a: M, b: K, c: N },
      Math.ceil(M / 64), Math.ceil(N / 64));
    return c;
  }

  /** C = Aᵀ @ B.   A:[K,M]  B:[K,N]  ->  [M,N]
   *
   * Uses the thread-blocked variant — same pattern, adapted to A's [K,M]
   * (transposed-row) access. */
  matmulAtb(a: GpuTensor, b: GpuTensor, M: number, K: number, N: number): GpuTensor {
    const c = this.newTensor(M * N, "matmul.atb");
    this.dispatch("matmul_atb_blocked", [a, b, c], { a: M, b: K, c: N },
      Math.ceil(M / 64), Math.ceil(N / 64));
    return c;
  }

  /** Backward of C = A @ B:  dA = dC@Bᵀ [M,K],  dB = Aᵀ@dC [K,N]. */
  matmulBackward(
    a: GpuTensor, b: GpuTensor, dC: GpuTensor, M: number, K: number, N: number,
  ): { dA: GpuTensor; dB: GpuTensor } {
    return {
      dA: this.matmulAbt(dC, b, M, N, K), // dC:[M,N] @ B:[K,N]ᵀ -> [M,K]
      dB: this.matmulAtb(a, dC, K, M, N), // A:[M,K]ᵀ @ dC:[M,N] -> [K,N]
    };
  }

  // --- f16-storage matmul --------------------------------------------------
  /** Same shape as matmul (C = A @ B → [M,N]) but B is a pre-packed f16
   *  storage buffer instead of an f32 GpuTensor. A stays f32 (activations
   *  are produced by other f32 kernels in the same submit), accumulation
   *  stays f32. K and N must be even. The B buffer length is K*N/2 u32
   *  (two consecutive f16 packed per u32 along the N axis).
   *
   *  Use this when the caller has already packed a weight tensor via
   *  packToF16 (typically once on importState, then again after each AdamW
   *  step). For matmuls where B is a transient activation/gradient, stick
   *  with the f32 matmul — the pack-pass cost dominates the bandwidth win
   *  when B is single-use. */
  matmulF16Weight(
    a: GpuTensor, bPackedF16: GPUBuffer, M: number, K: number, N: number,
  ): GpuTensor {
    if (K % 2 !== 0 || N % 2 !== 0) {
      throw new Error(`matmulF16Weight: K and N must be even (got K=${K}, N=${N})`);
    }
    const c = this.newTensor(M * N, "matmul.f16.C");
    this.dispatchMixed(
      "matmul_blocked_f16",
      [a.buffer, bPackedF16, c.buffer],
      { a: M, b: K, c: N },
      Math.ceil(M / 64), Math.ceil(N / 64),
    );
    return c;
  }

  /** Backward dA = dC @ B^T where B is a pre-packed f16 storage buffer
   *  (the weight, packed along its N axis at upload time). Same shape
   *  contract as matmulAbt(a, b, M, K, N) → output is [M, N], inner is K.
   *
   *  For a forward layer y = x @ w, the backward call is
   *  matmulAbt(dy, w, batchT, cout, cin). With matmulAbt's convention:
   *  M = batchT, K = cout (the original N-axis of w), N = cin (rows of w).
   *  The kernel reads w packed along its original N-axis (= matmulAbt's K).
   *  Same K-evenness requirement as matmulF16Weight.
   *
   *  Activates only when the f16-storage gate's matmulAbt-side check has
   *  also passed (see verifyF16Storage). Callers should consult
   *  f16StorageActive before dispatching. */
  matmulAbtF16Weight(
    a: GpuTensor, bPackedF16: GPUBuffer, M: number, K: number, N: number,
  ): GpuTensor {
    if (K % 2 !== 0) {
      throw new Error(`matmulAbtF16Weight: K must be even (got K=${K})`);
    }
    const c = this.newTensor(M * N, "matmul.abt.f16.C");
    this.dispatchMixed(
      "matmul_abt_blocked_f16",
      [a.buffer, bPackedF16, c.buffer],
      { a: M, b: K, c: N },
      Math.ceil(M / 64), Math.ceil(N / 64),
    );
    return c;
  }

  /** Pack a f32 weight buffer into a packed-f16 storage buffer (each pair
   *  of consecutive f32 values along the N axis becomes one packed u32).
   *  Dispatched once on weight load (after importState) and again after
   *  each AdamW step. N must be even.
   *
   *  This kernel reads from srcF32 (length M*N f32) and writes to dstF16
   *  (length M*N/2 u32 = M*N halfs). Both buffers must be raw GPUBuffers
   *  with STORAGE usage. */
  packToF16(srcF32: GPUBuffer, dstF16: GPUBuffer, M: number, N: number): void {
    if (N % 2 !== 0) {
      throw new Error(`packToF16: N must be even (got ${N})`);
    }
    const totalPairs = (M * N) / 2;
    this.dispatchMixed(
      "pack_to_f16",
      [srcF32, dstF16],
      { a: M, b: N },
      Math.ceil(totalPairs / 64),
    );
  }

  /** Variant of dispatch() that accepts raw GPUBuffers in the binding slots
   *  rather than GpuTensors. Used by the f16-storage paths where one buffer
   *  is the pre-packed weight (not a GpuTensor wrapper).
   *
   *  Identical command-recording shape as dispatch — same uniform encoding,
   *  same beginBatch/endBatch coordination. */
  private dispatchMixed(
    entry: Entry,
    buffers: GPUBuffer[],
    params: Params,
    wgX: number,
    wgY = 1,
  ): void {
    const ownBatch = this.pass === null;
    if (ownBatch) this.beginBatch();

    const u = new ArrayBuffer(32);
    const dv = new DataView(u);
    dv.setUint32(0, params.a ?? 0, true);
    dv.setUint32(4, params.b ?? 0, true);
    dv.setUint32(8, params.c ?? 0, true);
    dv.setUint32(12, params.d ?? 0, true);
    dv.setFloat32(16, params.fa ?? 0, true);
    dv.setFloat32(20, params.fb ?? 0, true);
    dv.setFloat32(24, params.fc ?? 0, true);
    dv.setFloat32(28, params.fd ?? 0, true);
    const ubuf = this.nextUniform();
    this.device.queue.writeBuffer(ubuf, 0, u);

    const entries: GPUBindGroupEntry[] = [];
    for (let i = 0; i < 6; i++) {
      entries.push({
        binding: i,
        resource: { buffer: i < buffers.length ? buffers[i] : this.dummies[i] },
      });
    }
    entries.push({ binding: 6, resource: { buffer: ubuf } });
    const bind = this.device.createBindGroup({ layout: this.layout, entries });

    const pass = this.pass as GPUComputePassEncoder;
    const pipeline = this.pipelines[entry];
    if (!pipeline) throw new Error(`pipeline missing for ${entry}`);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(wgX, wgY);

    if (ownBatch) this.endBatch();
  }

  // --- elementwise ---------------------------------------------------------
  /** c = a + b, length n (residual add). */
  add(a: GpuTensor, b: GpuTensor, n: number): GpuTensor {
    const c = this.newTensor(n, "add");
    this.dispatch("add", [a, b, c], { a: n }, Math.ceil(n / 64));
    return c;
  }

  /** y += bias, broadcast over `rows` rows of width D. In place on y. */
  biasAdd(y: GpuTensor, bias: GpuTensor, rows: number, D: number): void {
    this.dispatch("bias_add", [y, bias], { a: rows, b: D },
      Math.ceil((rows * D) / 64));
  }

  /** db[d] = sum over rows of dy[row,d]. */
  biasGrad(dy: GpuTensor, rows: number, D: number): GpuTensor {
    const db = this.newTensor(D, "db");
    this.dispatch("bias_grad", [dy, db], { a: rows, b: D }, Math.ceil(D / 64));
    return db;
  }

  /** y = GELU(x), length n. */
  gelu(x: GpuTensor, n: number): GpuTensor {
    const y = this.newTensor(n, "gelu");
    this.dispatch("gelu_forward", [x, y], { a: n }, Math.ceil(n / 64));
    return y;
  }

  /** dx = dy * GELU'(x), length n. */
  geluBackward(x: GpuTensor, dy: GpuTensor, n: number): GpuTensor {
    const dx = this.newTensor(n, "dgelu");
    this.dispatch("gelu_backward", [x, dy, dx], { a: n }, Math.ceil(n / 64));
    return dx;
  }

  // --- layernorm -----------------------------------------------------------
  /** LayerNorm over the last dim D, for N rows. Returns y and the cached
   *  mean/rstd the backward pass needs. */
  layernormForward(
    x: GpuTensor, gamma: GpuTensor, beta: GpuTensor, N: number, D: number,
    eps = 1e-5,
  ): { y: GpuTensor; mean: GpuTensor; rstd: GpuTensor } {
    const y = this.newTensor(N * D, "ln.y");
    const mean = this.newTensor(N, "ln.mean");
    const rstd = this.newTensor(N, "ln.rstd");
    // Prefer the subgroup-cooperative variant when the device offers it.
    // The SG kernel runs one workgroup per row (vs one thread per row in the
    // base kernel) and uses subgroupAdd for the reductions — big win at
    // d_model ≥ 256 where the serial scan dominates.
    if (this.hasSubgroups) {
      this.dispatch("layernorm_forward_sg", [x, gamma, beta, y, mean, rstd],
        { a: N, b: D, fa: eps }, N);
    } else {
      this.dispatch("layernorm_forward", [x, gamma, beta, y, mean, rstd],
        { a: N, b: D, fa: eps }, Math.ceil(N / 64));
    }
    return { y, mean, rstd };
  }

  /** LayerNorm backward: dx, plus dgamma/dbeta summed over rows. */
  layernormBackward(
    x: GpuTensor, gamma: GpuTensor, mean: GpuTensor, rstd: GpuTensor,
    dy: GpuTensor, N: number, D: number,
  ): { dx: GpuTensor; dgamma: GpuTensor; dbeta: GpuTensor } {
    const dx = this.newTensor(N * D, "ln.dx");
    this.dispatch("layernorm_dx", [x, gamma, mean, rstd, dy, dx],
      { a: N, b: D }, Math.ceil(N / 64));
    const dgamma = this.newTensor(D, "ln.dgamma");
    const dbeta = this.newTensor(D, "ln.dbeta");
    this.dispatch("layernorm_dgb", [x, mean, rstd, dy, dgamma, dbeta],
      { a: N, b: D }, Math.ceil(D / 64));
    return { dx, dgamma, dbeta };
  }

  // --- causal multi-head attention (the SDPA core) -------------------------
  /** Scaled dot-product attention over q,k,v:[B,T,C], H heads. Returns the
   *  softmax weights attn:[B,H,T,T] and the context ctx:[B,T,C]. */
  attentionForward(
    q: GpuTensor, k: GpuTensor, v: GpuTensor,
    B: number, T: number, C: number, H: number,
  ): { attn: GpuTensor; ctx: GpuTensor; L: GpuTensor | null } {
    const params = { a: B, b: T, c: C, d: H, fa: 1 / Math.sqrt(C / H) };
    const attn = this.newTensor(B * H * T * T, "attn");
    const ctx = this.newTensor(B * T * C, "ctx");
    // Prefer the FA2 forward kernel when the head dim fits the workgroup
    // storage budget (MAX_HD=64 in attention_fa2.wgsl). One workgroup per
    // (batch, head, ceil(T/16)) tile of Q; online softmax in registers
    // across K blocks; writes the full attn matrix in a second pass so the
    // existing backward kernels stay unchanged. Fallback: attn_fused_sv,
    // which is the FA1-style fused kernel for shapes the FA2 kernel can't
    // handle yet (hd > 64).
    const hd = C / H;
    let L: GpuTensor | null = null;
    if (hd <= 64) {
      // FA2 path: also save L = m + log(l) so the FA2-aware backward
      // kernels (attn_dscores_fa2, attn_dv_fa2) can reconstruct P from
      // q/k without reading the attn matrix.
      L = this.newTensor(B * H * T, "L");
      this.dispatch("fa2_forward", [q, k, v, attn, ctx, L], params,
        Math.ceil(T / 16), B * H);
    } else {
      const wg = Math.ceil((B * H * T) / 64);
      this.dispatch("attn_fused_sv", [q, k, v, attn, ctx], params, wg);
    }
    return { attn, ctx, L };
  }

  /** Backward of attention. Given the forward outputs and dctx, returns
   *  dq, dk, dv : each [B,T,C]. If `L` is provided (FA2 forward path),
   *  the two backward kernels that read attn (dscores + dv) get replaced
   *  with FA2-aware variants that recompute P = exp(S − L) from q/k
   *  instead. The attn matrix itself isn't read in that case — meaning
   *  the FA2 forward can drop its second-pass writeback. */
  attentionBackward(
    q: GpuTensor, k: GpuTensor, v: GpuTensor, attn: GpuTensor, dctx: GpuTensor,
    B: number, T: number, C: number, H: number,
    L: GpuTensor | null = null,
  ): { dq: GpuTensor; dk: GpuTensor; dv: GpuTensor } {
    const params = { a: B, b: T, c: C, d: H, fa: 1 / Math.sqrt(C / H) };
    const wg = Math.ceil((B * H * T) / 64);
    const dscores = this.newTensor(B * H * T * T, "dscores");
    if (L !== null) {
      // FA2 path: recompute P from q/k/L; never touch the attn matrix.
      this.dispatch("attn_dscores_fa2", [q, k, L, dctx, v, dscores], params, wg);
    } else {
      this.dispatch("attn_dscores", [dctx, v, attn, dscores], params, wg);
    }
    const dq = this.newTensor(B * T * C, "dq");
    this.dispatch("attn_dq", [dscores, k, dq], params, wg);
    const dk = this.newTensor(B * T * C, "dk");
    this.dispatch("attn_dk", [dscores, q, dk], params, wg);
    const dv = this.newTensor(B * T * C, "dv");
    if (L !== null) {
      // FA2 path: recompute P inside the kernel; doesn't need attn.
      this.dispatch("attn_dv_fa2", [q, k, L, dctx, dv], params, wg);
    } else {
      this.dispatch("attn_dv", [attn, dctx, dv], params, wg);
    }
    return { dq, dk, dv };
  }

  // --- embeddings, cross-entropy, optimizer --------------------------------
  /** x[n] = tok_emb[id[n]] + pos_emb[n mod T].  ids holds int values as f32. */
  embedForward(
    tokEmb: GpuTensor, posEmb: GpuTensor, ids: GpuTensor,
    N: number, C: number, T: number,
  ): GpuTensor {
    const x = this.newTensor(N * C, "embed.x");
    this.dispatch("embed_forward", [tokEmb, posEmb, ids, x],
      { a: N, b: C, c: T }, Math.ceil((N * C) / 64));
    return x;
  }

  /** Token-embedding gradient: dtok[v] = sum of dx over rows with that id. */
  embedTokGrad(dx: GpuTensor, ids: GpuTensor, N: number, C: number, V: number): GpuTensor {
    const dtok = this.newTensor(V * C, "embed.dtok");
    this.dispatch("embed_tok_grad", [dx, ids, dtok],
      { a: N, b: C, c: V }, Math.ceil((V * C) / 64));
    return dtok;
  }

  /** Position-embedding gradient: dpos[t] = sum of dx over the batch. */
  embedPosGrad(dx: GpuTensor, N: number, C: number, T: number): GpuTensor {
    const dpos = this.newTensor(T * C, "embed.dpos");
    this.dispatch("embed_pos_grad", [dx, dpos],
      { a: N, b: C, c: T }, Math.ceil((T * C) / 64));
    return dpos;
  }

  /** Cross-entropy. Returns dlogits:[N,V] and per-row loss:[N] (sum on host). */
  crossEntropy(
    logits: GpuTensor, targets: GpuTensor, N: number, V: number,
  ): { dlogits: GpuTensor; loss: GpuTensor } {
    const dlogits = this.newTensor(N * V, "ce.dlogits");
    const loss = this.newTensor(N, "ce.loss");
    if (this.hasSubgroups) {
      this.dispatch("cross_entropy_sg", [logits, targets, dlogits, loss],
        { a: N, b: V }, N);
    } else {
      this.dispatch("cross_entropy", [logits, targets, dlogits, loss],
        { a: N, b: V }, Math.ceil(N / 64));
    }
    return { dlogits, loss };
  }

  /** In-place AdamW step over one parameter buffer (betas/eps are fixed). */
  adamwStep(
    param: GpuTensor, grad: GpuTensor, m: GpuTensor, v: GpuTensor,
    count: number, step: number, lr: number, weightDecay: number,
  ): void {
    this.dispatch("adamw", [param, grad, m, v],
      { a: count, b: step, fa: lr, fb: weightDecay }, Math.ceil(count / 64));
  }
}
