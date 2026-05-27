/**
 * gpu_model.ts — a TinyGPT that trains entirely on the GPU (Phase 5, stage 5).
 *
 * Wires the WGSL kernels in ops.ts into a full forward + backward + AdamW loop.
 * Every weight, gradient, optimizer moment, and activation is a GpuTensor that
 * stays resident on the GPU — the host only uploads the batch and downloads the
 * scalar loss. Architecture matches python_ref/model.py and wasm/src/model.cpp:
 * pre-LayerNorm blocks, GELU MLP, tied input/output embeddings.
 *
 * Per-step intermediates are tracked and freed each step so a long run does not
 * exhaust GPU memory; weights / moments persist.
 *
 * Guide: docs/performance.md ("WebGPU — the real ceiling")
 */

import { GpuOps } from "./ops";
import { type GpuContext, GpuTensor } from "./tensor";

export interface GpuModelConfig {
  vocab: number;
  ctx: number;
  layers: number;
  heads: number;
  dModel: number;
  dMlp: number;
  seed: number;
}

/** A trainable tensor: weights + the two AdamW moments. Grad is per-step. */
interface Param {
  w: GpuTensor;
  m: GpuTensor;
  v: GpuTensor;
  size: number;
  decay: boolean;
}

interface Layer {
  ln1g: Param; ln1b: Param;
  wq: Param; bq: Param; wk: Param; bk: Param; wv: Param; bv: Param;
  wo: Param; bo: Param;
  ln2g: Param; ln2b: Param;
  fcInW: Param; fcInB: Param; fcOutW: Param; fcOutB: Param;
}

/** Forward activations one block needs for its backward pass. */
interface LayerCache {
  blockIn: GpuTensor; ln1o: GpuTensor; m1: GpuTensor; r1s: GpuTensor;
  q: GpuTensor; k: GpuTensor; v: GpuTensor; attn: GpuTensor; ctx: GpuTensor;
  /** Log-sum-exp from FA2 forward, when that path runs (hd ≤ 64). The FA2
   * backward kernels reconstruct P = exp(S − L) from q/k/L instead of
   * reading the attn matrix; if null, backward uses the legacy attn-cached
   * kernels. */
  L: GpuTensor | null;
  r1: GpuTensor; ln2o: GpuTensor; m2: GpuTensor; r2s: GpuTensor;
  hpre: GpuTensor; hact: GpuTensor;
}

// Deterministic RNG (mulberry32) + Box-Muller, for reproducible weight init.
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class GpuModel {
  private readonly ops: GpuOps;
  private readonly device: GPUDevice;
  private scratch: GpuTensor[] = [];
  private stepCount = 0;

  private tokEmb!: Param;
  private posEmb!: Param;
  private lnfG!: Param;
  private lnfB!: Param;
  private layers: Layer[] = [];
  private params: Param[] = [];

  constructor(ctx: GpuContext, readonly cfg: GpuModelConfig) {
    this.device = ctx.device;
    this.ops = GpuOps.create(ctx);
    this.initWeights();
  }

  /** Track a per-step tensor so it is freed at the end of the step. */
  private keep<T extends GpuTensor>(t: T): T {
    this.scratch.push(t);
    return t;
  }

  // Per-step input tensors (ids, targets) are pooled — recycled each step.
  private tensorFrom(data: Float32Array): GpuTensor {
    return this.ops.upload(data);
  }

  private makeParam(size: number, fill: Float32Array, decay: boolean): Param {
    const p: Param = {
      w: GpuTensor.fromData(this.device, fill, "w"),
      m: GpuTensor.fromData(this.device, new Float32Array(size), "m"),
      v: GpuTensor.fromData(this.device, new Float32Array(size), "v"),
      size,
      decay,
    };
    this.params.push(p);
    return p;
  }

  private initWeights(): void {
    const { vocab, ctx, layers, dModel: C, dMlp: M } = this.cfg;
    const rng = makeRng(this.cfg.seed);
    let spare: number | null = null;
    const randn = (std: number): number => {
      if (spare !== null) {
        const r = spare;
        spare = null;
        return r * std;
      }
      const u = Math.max(1e-9, rng()), w = rng();
      const mag = Math.sqrt(-2 * Math.log(u));
      spare = mag * Math.sin(2 * Math.PI * w);
      return mag * Math.cos(2 * Math.PI * w) * std;
    };
    const normal = (n: number, std: number) => {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = randn(std);
      return a;
    };
    const filled = (n: number, value: number) => new Float32Array(n).fill(value);
    const scaled = 0.02 / Math.sqrt(2 * layers); // residual-path init

    this.tokEmb = this.makeParam(vocab * C, normal(vocab * C, 0.02), true);
    this.posEmb = this.makeParam(ctx * C, normal(ctx * C, 0.02), true);
    this.lnfG = this.makeParam(C, filled(C, 1), false);
    this.lnfB = this.makeParam(C, filled(C, 0), false);

    for (let l = 0; l < layers; l++) {
      this.layers.push({
        ln1g: this.makeParam(C, filled(C, 1), false),
        ln1b: this.makeParam(C, filled(C, 0), false),
        wq: this.makeParam(C * C, normal(C * C, 0.02), true),
        bq: this.makeParam(C, filled(C, 0), false),
        wk: this.makeParam(C * C, normal(C * C, 0.02), true),
        bk: this.makeParam(C, filled(C, 0), false),
        wv: this.makeParam(C * C, normal(C * C, 0.02), true),
        bv: this.makeParam(C, filled(C, 0), false),
        wo: this.makeParam(C * C, normal(C * C, scaled), true),
        bo: this.makeParam(C, filled(C, 0), false),
        ln2g: this.makeParam(C, filled(C, 1), false),
        ln2b: this.makeParam(C, filled(C, 0), false),
        fcInW: this.makeParam(C * M, normal(C * M, 0.02), true),
        fcInB: this.makeParam(M, filled(M, 0), false),
        fcOutW: this.makeParam(M * C, normal(M * C, scaled), true),
        fcOutB: this.makeParam(C, filled(C, 0), false),
      });
    }
  }

  numParams(): number {
    return this.params.reduce((n, p) => n + p.size, 0);
  }

  step(): number {
    return this.stepCount;
  }

  // --- linear = matmul + bias ----------------------------------------------
  private linear(x: GpuTensor, w: Param, b: Param, N: number, cin: number, cout: number) {
    const y = this.keep(this.ops.matmul(x, w.w, N, cin, cout));
    this.ops.biasAdd(y, b.w, N, cout);
    return y;
  }

  private linearBackward(
    x: GpuTensor, w: Param, dy: GpuTensor, N: number, cin: number, cout: number,
    grads: Map<Param, GpuTensor>, wOut: Param, bOut: Param,
  ): GpuTensor {
    const { dA, dB } = this.ops.matmulBackward(x, w.w, dy, N, cin, cout);
    this.keep(dA); this.keep(dB);
    grads.set(wOut, dB);
    grads.set(bOut, this.keep(this.ops.biasGrad(dy, N, cout)));
    return dA;
  }

  /** Forward pass. Returns logits and everything the backward pass needs. */
  private forward(ids: Float32Array, batch: number, T: number) {
    const { vocab: V, layers: L, heads: H, dModel: C, dMlp: M } = this.cfg;
    const N = batch * T;
    const idsT = this.keep(this.tensorFrom(ids));
    const x0 = this.keep(
      this.ops.embedForward(this.tokEmb.w, this.posEmb.w, idsT, N, C, T));
    const caches: LayerCache[] = [];
    let x = x0;
    for (let l = 0; l < L; l++) {
      const ly = this.layers[l];
      const blockIn = x;
      const ln1 = this.ops.layernormForward(blockIn, ly.ln1g.w, ly.ln1b.w, N, C);
      this.keep(ln1.y); this.keep(ln1.mean); this.keep(ln1.rstd);
      const q = this.linear(ln1.y, ly.wq, ly.bq, N, C, C);
      const k = this.linear(ln1.y, ly.wk, ly.bk, N, C, C);
      const v = this.linear(ln1.y, ly.wv, ly.bv, N, C, C);
      const att = this.ops.attentionForward(q, k, v, batch, T, C, H);
      this.keep(att.attn); this.keep(att.ctx);
      if (att.L) this.keep(att.L);
      const ao = this.linear(att.ctx, ly.wo, ly.bo, N, C, C);
      const r1 = this.keep(this.ops.add(blockIn, ao, N * C));
      const ln2 = this.ops.layernormForward(r1, ly.ln2g.w, ly.ln2b.w, N, C);
      this.keep(ln2.y); this.keep(ln2.mean); this.keep(ln2.rstd);
      const hpre = this.linear(ln2.y, ly.fcInW, ly.fcInB, N, C, M);
      const hact = this.keep(this.ops.gelu(hpre, N * M));
      const mo = this.linear(hact, ly.fcOutW, ly.fcOutB, N, M, C);
      const r2 = this.keep(this.ops.add(r1, mo, N * C));
      caches.push({
        blockIn, ln1o: ln1.y, m1: ln1.mean, r1s: ln1.rstd, q, k, v,
        attn: att.attn, ctx: att.ctx, L: att.L, r1, ln2o: ln2.y, m2: ln2.mean,
        r2s: ln2.rstd, hpre, hact,
      });
      x = r2;
    }
    const lnf = this.ops.layernormForward(x, this.lnfG.w, this.lnfB.w, N, C);
    this.keep(lnf.y); this.keep(lnf.mean); this.keep(lnf.rstd);
    // tied head: logits[N,V] = lnf[N,C] @ tok_emb[V,C]ᵀ
    const logits = this.keep(this.ops.matmulAbt(lnf.y, this.tokEmb.w, N, C, V));
    return { logits, caches, lastX: x, lnf, idsT, N };
  }

  /**
   * Serialize the trained model into the same flat buffer layout the WASM
   * backend produces (see `tg_export_state` in `wasm/src/model.cpp` and the
   * matching reader in `browser/src/main.ts:buildManifest` /
   * `encodeModelFile`):
   *
   *   4 bytes  int32 LE  step counter
   *   ...      float32   per-param triplets [w, m, v], in the order params
   *                      were created in `initWeights()` (token + position
   *                      embeddings, final layernorm, then per-block:
   *                      ln1, q/k/v/o projections, ln2, MLP fc_in/fc_out).
   *
   * This is the on-disk format the .tinygpt header keys off, so the resulting
   * ArrayBuffer can be passed straight into `encodeModelFile()` and the saved
   * file is loadable in any TinyGPT backend (WASM or another WebGPU session).
   */
  async exportState(): Promise<ArrayBuffer> {
    let totalFloats = 0;
    for (const p of this.params) totalFloats += p.size * 3; // w + m + v
    const buf = new ArrayBuffer(4 + totalFloats * 4);
    new Int32Array(buf, 0, 1)[0] = this.stepCount;
    const f32 = new Float32Array(buf, 4);
    let off = 0;
    for (const p of this.params) {
      // Pull weights and Adam moments back to CPU. Each download is an async
      // GPU→CPU readback; doing them serially keeps memory pressure manageable
      // (one Float32Array per param at a time) and avoids overlapping mapAsync.
      const w = await p.w.download(); f32.set(w, off); off += w.length;
      const m = await p.m.download(); f32.set(m, off); off += m.length;
      const v = await p.v.download(); f32.set(v, off); off += v.length;
    }
    return buf;
  }

  /** Autoregressive generation from a prompt. temperature <= 0 is greedy.
   *  Optional `onToken` callback fires once per newly-sampled token so the
   *  caller can stream output instead of waiting for the full sequence. */
  async generate(
    promptIds: number[], maxNew: number, temperature: number, topK: number,
    seed: number, onToken?: (tok: number, idxIntoMaxNew: number) => void,
  ): Promise<number[]> {
    const { vocab: V, ctx } = this.cfg;
    const ids = promptIds.length > 0 ? [...promptIds] : [10];
    const rng = makeRng(seed);
    for (let s = 0; s < maxNew; s++) {
      const T = Math.min(ids.length, ctx);
      const window = new Float32Array(ids.slice(ids.length - T));
      this.ops.beginBatch();
      const fwd = this.forward(window, 1, T);
      this.ops.endBatch();
      const logits = await fwd.logits.download();
      const base = (T - 1) * V;
      let next = 0;
      if (temperature <= 0) {
        for (let v = 1; v < V; v++) if (logits[base + v] > logits[base + next]) next = v;
      } else {
        const probs = new Float32Array(V);
        let mx = -1e30;
        for (let v = 0; v < V; v++) mx = Math.max(mx, logits[base + v]);
        for (let v = 0; v < V; v++) probs[v] = Math.exp((logits[base + v] - mx) / temperature);
        if (topK > 0 && topK < V) {
          const thresh = [...probs].sort((a, b) => b - a)[topK - 1];
          for (let v = 0; v < V; v++) if (probs[v] < thresh) probs[v] = 0;
        }
        let sum = 0;
        for (const z of probs) sum += z;
        let r = rng() * sum;
        next = V - 1;
        for (let v = 0; v < V; v++) {
          r -= probs[v];
          if (r <= 0) { next = v; break; }
        }
      }
      ids.push(next);
      this.freeScratch();
      onToken?.(next, s);
    }
    return ids;
  }

  /**
   * Introspection forward pass — used by the "Watch the model think" panel.
   *
   * Runs a single forward over `promptIds` (B=1), then returns, per token
   * position t:
   *   - topK probability bars (the actual distribution over next-byte for
   *     position t, softmaxed without temperature), top `k` entries
   *   - attention weights from the LAST block: one Float32Array per head,
   *     length T, representing which earlier tokens that head looked at
   *     when producing position t. Future positions are zero (causal mask
   *     handled by the WGSL kernel).
   *
   * Cost: one forward over the prompt. No backward, no training.
   * Memory: the full attn buffer is [B=1, H, T, T] — at T=64 H=4 that's
   * 64 KB. Fine to download.
   */
  async inspect(
    promptIds: number[], k: number,
  ): Promise<{
    tokens: number[];
    topK: { token: number; prob: number }[][];
    attention: Float32Array[][];
  }> {
    const { vocab: V, ctx, heads: H, layers: L } = this.cfg;
    if (promptIds.length === 0) {
      return { tokens: [], topK: [], attention: [] };
    }
    const T = Math.min(promptIds.length, ctx);
    const window = new Float32Array(promptIds.slice(promptIds.length - T));
    this.ops.beginBatch();
    const fwd = this.forward(window, 1, T);
    this.ops.endBatch();
    const logits = await fwd.logits.download();
    // Last block's attn — shape [B=1, H, T, T]
    const lastAttn = await fwd.caches[L - 1].attn.download();

    const tokens = Array.from(window).map((x) => x | 0);
    const topK: { token: number; prob: number }[][] = [];
    const attention: Float32Array[][] = [];

    for (let t = 0; t < T; t++) {
      // Softmax logits[t, :] (no temperature — show the raw model belief).
      const base = t * V;
      let mx = -1e30;
      for (let v = 0; v < V; v++) if (logits[base + v] > mx) mx = logits[base + v];
      const probs = new Float64Array(V);
      let sum = 0;
      for (let v = 0; v < V; v++) {
        const p = Math.exp(logits[base + v] - mx);
        probs[v] = p;
        sum += p;
      }
      for (let v = 0; v < V; v++) probs[v] /= sum;

      // Top-k by probability.
      const indexed = Array.from(probs, (p, v) => ({ token: v, prob: p }));
      indexed.sort((a, b) => b.prob - a.prob);
      topK.push(indexed.slice(0, k));

      // Attention per head for query position t — slice [h, t, :].
      const headRows: Float32Array[] = [];
      for (let h = 0; h < H; h++) {
        // Layout: [B=1, H, T, T] -> offset = ((0*H + h)*T + t) * T
        const off = (h * T + t) * T;
        headRows.push(lastAttn.slice(off, off + T));
      }
      attention.push(headRows);
    }

    this.freeScratch();
    return { tokens, topK, attention };
  }

  /** One training step: forward, cross-entropy, backward, AdamW. Returns loss. */
  async trainStep(
    ids: Float32Array, targets: Float32Array, batch: number, lr: number,
  ): Promise<number> {
    const { vocab: V, ctx: T, layers: L, heads: H, dModel: C, dMlp: M } = this.cfg;
    const grads = new Map<Param, GpuTensor>();

    // Record the whole step (forward + backward + AdamW) into one submission.
    this.ops.beginBatch();
    const f = this.forward(ids, batch, T);
    const { logits, caches, lnf, idsT } = f;
    const N = f.N;

    // --- loss + dlogits ----------------------------------------------------
    const targetsT = this.keep(this.tensorFrom(targets));
    const ce = this.ops.crossEntropy(logits, targetsT, N, V);
    this.keep(ce.dlogits); this.keep(ce.loss);

    // --- backward ----------------------------------------------------------
    // head backward: dlnf = dlogits @ tok_emb; d(tok_emb)_head = dlogitsᵀ @ lnf
    const dlnf = this.keep(this.ops.matmul(ce.dlogits, this.tokEmb.w, N, V, C));
    const dTokHead = this.keep(this.ops.matmulAtb(ce.dlogits, lnf.y, V, N, C));
    const lnfBack = this.ops.layernormBackward(
      f.lastX, this.lnfG.w, lnf.mean, lnf.rstd, dlnf, N, C);
    this.keep(lnfBack.dx); this.keep(lnfBack.dgamma); this.keep(lnfBack.dbeta);
    grads.set(this.lnfG, lnfBack.dgamma);
    grads.set(this.lnfB, lnfBack.dbeta);

    let dnext = lnfBack.dx;
    for (let l = L - 1; l >= 0; l--) {
      const ly = this.layers[l];
      const c = caches[l];
      // r2 = r1 + mo
      const dmo = dnext, dr1a = dnext;
      const dhact = this.linearBackward(c.hact, ly.fcOutW, dmo, N, M, C, grads, ly.fcOutW, ly.fcOutB);
      const dhpre = this.keep(this.ops.geluBackward(c.hpre, dhact, N * M));
      const dln2o = this.linearBackward(c.ln2o, ly.fcInW, dhpre, N, C, M, grads, ly.fcInW, ly.fcInB);
      const ln2b = this.ops.layernormBackward(c.r1, ly.ln2g.w, c.m2, c.r2s, dln2o, N, C);
      this.keep(ln2b.dx); this.keep(ln2b.dgamma); this.keep(ln2b.dbeta);
      grads.set(ly.ln2g, ln2b.dgamma);
      grads.set(ly.ln2b, ln2b.dbeta);
      const dr1 = this.keep(this.ops.add(dr1a, ln2b.dx, N * C));
      // r1 = blockIn + ao
      const dao = dr1, dBlockA = dr1;
      const dctx = this.linearBackward(c.ctx, ly.wo, dao, N, C, C, grads, ly.wo, ly.bo);
      const attBack = this.ops.attentionBackward(c.q, c.k, c.v, c.attn, dctx, batch, T, C, H, c.L);
      this.keep(attBack.dq); this.keep(attBack.dk); this.keep(attBack.dv);
      const dx1 = this.linearBackward(c.ln1o, ly.wq, attBack.dq, N, C, C, grads, ly.wq, ly.bq);
      const dx2 = this.linearBackward(c.ln1o, ly.wk, attBack.dk, N, C, C, grads, ly.wk, ly.bk);
      const dx3 = this.linearBackward(c.ln1o, ly.wv, attBack.dv, N, C, C, grads, ly.wv, ly.bv);
      const dln1o = this.keep(this.ops.add(
        this.keep(this.ops.add(dx1, dx2, N * C)), dx3, N * C));
      const ln1b = this.ops.layernormBackward(c.blockIn, ly.ln1g.w, c.m1, c.r1s, dln1o, N, C);
      this.keep(ln1b.dx); this.keep(ln1b.dgamma); this.keep(ln1b.dbeta);
      grads.set(ly.ln1g, ln1b.dgamma);
      grads.set(ly.ln1b, ln1b.dbeta);
      dnext = this.keep(this.ops.add(dBlockA, ln1b.dx, N * C));
    }
    // embedding backward — grad w.r.t. x0 splits into tok + pos
    const dTokEmbed = this.keep(this.ops.embedTokGrad(dnext, idsT, N, C, V));
    grads.set(this.tokEmb, this.keep(this.ops.add(dTokHead, dTokEmbed, V * C)));
    grads.set(this.posEmb, this.keep(this.ops.embedPosGrad(dnext, N, C, T)));

    // --- AdamW -------------------------------------------------------------
    this.stepCount++;
    for (const p of this.params) {
      const g = grads.get(p);
      if (!g) continue;
      this.ops.adamwStep(p.w, g, p.m, p.v, p.size, this.stepCount, lr,
        p.decay ? 0.1 : 0.0);
    }

    this.ops.endBatch(); // submit the whole step at once
    const lossArr = await ce.loss.download();
    let total = 0;
    for (const x2 of lossArr) total += x2;
    this.freeScratch();
    return total / N;
  }

  // Return every per-step tensor to the buffer pool (not destroyed) so the
  // next step reuses the buffers — after step 1, a run does no allocation.
  private freeScratch(): void {
    for (const t of this.scratch) t.recycle();
    this.scratch = [];
  }
}
