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
import { GpuTensor, type GpuContext } from "./tensor";

const ENTRIES = [
  "matmul", "matmul_abt", "matmul_atb", "add", "bias_add", "bias_grad",
  "gelu_forward", "gelu_backward", "layernorm_forward", "layernorm_dx",
  "layernorm_dgb", "attn_softmax", "attn_value", "attn_dscores", "attn_dq",
  "attn_dk", "attn_dv", "embed_forward", "embed_tok_grad", "embed_pos_grad",
  "cross_entropy", "adamw",
] as const;
type Entry = (typeof ENTRIES)[number];

/** Params uniform: up to four u32 (dims) and four f32 (eps, scale, ...). */
interface Params {
  a?: number; b?: number; c?: number; d?: number;
  fa?: number; fb?: number; fc?: number; fd?: number;
}

export class GpuOps {
  private constructor(
    private readonly device: GPUDevice,
    private readonly layout: GPUBindGroupLayout,
    private readonly pipelines: Record<Entry, GPUComputePipeline>,
    // One distinct dummy per slot — WebGPU forbids binding the same buffer to
    // two writable-storage bindings in a bind group (aliasing).
    private readonly dummies: GPUBuffer[],
  ) {}

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

    const pipelines = {} as Record<Entry, GPUComputePipeline>;
    for (const entryPoint of ENTRIES) {
      pipelines[entryPoint] = device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module, entryPoint },
      });
    }
    // A distinct 1-element buffer for each slot a kernel leaves unused.
    const dummies: GPUBuffer[] = [];
    for (let i = 0; i < 6; i++) {
      dummies.push(device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }));
    }
    return new GpuOps(device, layout, pipelines, dummies);
  }

  private newTensor(size: number, label: string): GpuTensor {
    return new GpuTensor(this.device, size, label);
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
    const pbuf = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(pbuf, 0, u);

    const entries: GPUBindGroupEntry[] = [];
    for (let i = 0; i < 6; i++) {
      entries.push({
        binding: i,
        resource: {
          buffer: i < buffers.length ? buffers[i].buffer : this.dummies[i],
        },
      });
    }
    entries.push({ binding: 6, resource: { buffer: pbuf } });
    const bind = this.device.createBindGroup({ layout: this.layout, entries });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines[entry]);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    pbuf.destroy();
  }

  // --- matmul --------------------------------------------------------------
  /** C = A @ B.   A:[M,K]  B:[K,N]  ->  [M,N] */
  matmul(a: GpuTensor, b: GpuTensor, M: number, K: number, N: number): GpuTensor {
    const c = this.newTensor(M * N, "matmul.C");
    this.dispatch("matmul", [a, b, c], { a: M, b: K, c: N },
      Math.ceil(M / 16), Math.ceil(N / 16));
    return c;
  }

  /** C = A @ Bᵀ.   A:[M,K]  B:[N,K]  ->  [M,N]   (also the tied output head). */
  matmulAbt(a: GpuTensor, b: GpuTensor, M: number, K: number, N: number): GpuTensor {
    const c = this.newTensor(M * N, "matmul.abt");
    this.dispatch("matmul_abt", [a, b, c], { a: M, b: K, c: N },
      Math.ceil(M / 16), Math.ceil(N / 16));
    return c;
  }

  /** C = Aᵀ @ B.   A:[K,M]  B:[K,N]  ->  [M,N] */
  matmulAtb(a: GpuTensor, b: GpuTensor, M: number, K: number, N: number): GpuTensor {
    const c = this.newTensor(M * N, "matmul.atb");
    this.dispatch("matmul_atb", [a, b, c], { a: M, b: K, c: N },
      Math.ceil(M / 16), Math.ceil(N / 16));
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
    this.dispatch("layernorm_forward", [x, gamma, beta, y, mean, rstd],
      { a: N, b: D, fa: eps }, Math.ceil(N / 64));
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
  ): { attn: GpuTensor; ctx: GpuTensor } {
    const params = { a: B, b: T, c: C, d: H, fa: 1 / Math.sqrt(C / H) };
    const wg = Math.ceil((B * H * T) / 64);
    const attn = this.newTensor(B * H * T * T, "attn");
    this.dispatch("attn_softmax", [q, k, attn], params, wg);
    const ctx = this.newTensor(B * T * C, "ctx");
    this.dispatch("attn_value", [attn, v, ctx], params, wg);
    return { attn, ctx };
  }

  /** Backward of attention. Given the cached attn weights and dctx, returns
   *  dq, dk, dv : each [B,T,C]. */
  attentionBackward(
    q: GpuTensor, k: GpuTensor, v: GpuTensor, attn: GpuTensor, dctx: GpuTensor,
    B: number, T: number, C: number, H: number,
  ): { dq: GpuTensor; dk: GpuTensor; dv: GpuTensor } {
    const params = { a: B, b: T, c: C, d: H, fa: 1 / Math.sqrt(C / H) };
    const wg = Math.ceil((B * H * T) / 64);
    const dscores = this.newTensor(B * H * T * T, "dscores");
    this.dispatch("attn_dscores", [dctx, v, attn, dscores], params, wg);
    const dq = this.newTensor(B * T * C, "dq");
    this.dispatch("attn_dq", [dscores, k, dq], params, wg);
    const dk = this.newTensor(B * T * C, "dk");
    this.dispatch("attn_dk", [dscores, q, dk], params, wg);
    const dv = this.newTensor(B * T * C, "dv");
    this.dispatch("attn_dv", [attn, dctx, dv], params, wg);
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
    this.dispatch("cross_entropy", [logits, targets, dlogits, loss],
      { a: N, b: V }, Math.ceil(N / 64));
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
