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
  /** Shape of the weight as [rows, cols] when this param is a 2D matrix
   *  used in a matmul (q/k/v/o projections, MLP fc_in/fc_out, embeddings).
   *  Null for 1D params (biases, layernorm gain/bias) that never appear as
   *  the B side of a matmul. Drives the f16 packing geometry. */
  matShape: [number, number] | null;
  /** Packed-f16 storage buffer mirroring `w`, populated by
   *  GpuModel.prepareForInference() iff the f16-storage numerics gate
   *  passed. Null otherwise. */
  wF16: GPUBuffer | null;
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

  private makeParam(
    size: number, fill: Float32Array, decay: boolean,
    matShape: [number, number] | null = null,
  ): Param {
    const p: Param = {
      w: GpuTensor.fromData(this.device, fill, "w"),
      m: GpuTensor.fromData(this.device, new Float32Array(size), "m"),
      v: GpuTensor.fromData(this.device, new Float32Array(size), "v"),
      size,
      decay,
      matShape,
      wF16: null,
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

    // Embeddings are matmul-shaped (tied head uses tokEmb in matmulAbt) but
    // they're consumed via embed_forward / matmulAbt, neither of which uses
    // the f16-storage path today, so leave matShape null on them.
    this.tokEmb = this.makeParam(vocab * C, normal(vocab * C, 0.02), true);
    this.posEmb = this.makeParam(ctx * C, normal(ctx * C, 0.02), true);
    this.lnfG = this.makeParam(C, filled(C, 1), false);
    this.lnfB = this.makeParam(C, filled(C, 0), false);

    for (let l = 0; l < layers; l++) {
      this.layers.push({
        ln1g: this.makeParam(C, filled(C, 1), false),
        ln1b: this.makeParam(C, filled(C, 0), false),
        // Q/K/V/O projections + MLP fc_in/fc_out are the matmul-B targets in
        // the forward pass. matShape = [K, N] matches the matmul C = A @ B
        // signature where B is laid out [K, N] in row-major order.
        wq: this.makeParam(C * C, normal(C * C, 0.02), true, [C, C]),
        bq: this.makeParam(C, filled(C, 0), false),
        wk: this.makeParam(C * C, normal(C * C, 0.02), true, [C, C]),
        bk: this.makeParam(C, filled(C, 0), false),
        wv: this.makeParam(C * C, normal(C * C, 0.02), true, [C, C]),
        bv: this.makeParam(C, filled(C, 0), false),
        wo: this.makeParam(C * C, normal(C * C, scaled), true, [C, C]),
        bo: this.makeParam(C, filled(C, 0), false),
        ln2g: this.makeParam(C, filled(C, 1), false),
        ln2b: this.makeParam(C, filled(C, 0), false),
        fcInW: this.makeParam(C * M, normal(C * M, 0.02), true, [C, M]),
        fcInB: this.makeParam(M, filled(M, 0), false),
        fcOutW: this.makeParam(M * C, normal(M * C, scaled), true, [M, C]),
        fcOutB: this.makeParam(C, filled(C, 0), false),
      });
    }
  }

  /** Has the f16-storage path been activated for this model? Initially false;
   *  flipped by prepareForInference() once the gate passes AND all matShape
   *  weights have been packed. linear() uses this to choose the dispatch. */
  private useF16Storage = false;

  /** Pack every matmul-shaped weight into a packed-f16 storage buffer, IF
   *  the ops-level numerics gate passes. Idempotent — second call is a
   *  no-op. Should be invoked after importState() and before the hot loop
   *  (warmupGenerate / generate / trainStep) so the first matmul on the
   *  fast path doesn't pay any setup cost.
   *
   *  Returns true if the f16 path is now active; false if the gate failed
   *  (in which case linear() continues to use the f32 matmul path
   *  unchanged). Always resolves — no throws on gate failure. */
  async prepareForInference(): Promise<boolean> {
    if (this.useF16Storage) return true;
    let gatePassed = false;
    try {
      gatePassed = await this.ops.f16Ready;
    } catch {
      gatePassed = false;
    }
    if (!gatePassed) return false;

    // Pack every matShape'd weight. Skip params that aren't 2D matmul-B
    // targets (biases, layernorm gain/bias) — those stay f32, used directly
    // in bias_add / layernorm kernels which already handle f32 scalars.
    for (const p of this.params) {
      if (!p.matShape) continue;
      const [rows, cols] = p.matShape;
      if (cols % 2 !== 0) continue; // f16 packing requires even N axis
      const bytes = (rows * cols) * 2; // K*N halfs
      const buf = this.device.createBuffer({
        label: "wF16",
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.ops.packToF16(p.w.buffer, buf, rows, cols);
      p.wF16 = buf;
    }
    this.useF16Storage = true;
    return true;
  }

  /** True iff inference-time matmuls on weight matrices use the f16-storage
   *  path. Exposed so the worker can post this back to the main thread for
   *  the capability pill cluster. */
  get f16StorageActive(): boolean {
    return this.useF16Storage;
  }

  numParams(): number {
    return this.params.reduce((n, p) => n + p.size, 0);
  }

  step(): number {
    return this.stepCount;
  }

  // --- linear = matmul + bias ----------------------------------------------
  private linear(x: GpuTensor, w: Param, b: Param, N: number, cin: number, cout: number) {
    // f16-storage fast path: matmulF16Weight reads B as packed-half (half
    // the bytes per inner-loop K-step), f32 accumulate. Numerics-gated at
    // GpuOps create time and again at prepareForInference time, so this
    // branch is only ever taken on configurations that pass both checks.
    let y: GpuTensor;
    if (this.useF16Storage && w.wF16 && w.matShape && cin % 2 === 0 && cout % 2 === 0) {
      y = this.keep(this.ops.matmulF16Weight(x, w.wF16, N, cin, cout));
    } else {
      y = this.keep(this.ops.matmul(x, w.w, N, cin, cout));
    }
    this.ops.biasAdd(y, b.w, N, cout);
    return y;
  }

  private linearBackward(
    x: GpuTensor, w: Param, dy: GpuTensor, N: number, cin: number, cout: number,
    grads: Map<Param, GpuTensor>, wOut: Param, bOut: Param,
  ): GpuTensor {
    // dB = x^T @ dy never reads the weight — always stays on f32 vec4.
    const dB = this.keep(this.ops.matmulAtb(x, dy, cin, N, cout));
    // dA = dy @ W^T reads the weight. If the f16-storage path is active
    // and this weight has a packed mirror, dispatch the f16 variant.
    let dA: GpuTensor;
    if (this.useF16Storage && w.wF16 && w.matShape && cin % 2 === 0 && cout % 2 === 0) {
      dA = this.keep(this.ops.matmulAbtF16Weight(dy, w.wF16, N, cout, cin));
    } else {
      dA = this.keep(this.ops.matmulAbt(dy, w.w, N, cout, cin));
    }
    grads.set(wOut, dB);
    grads.set(bOut, this.keep(this.ops.biasGrad(dy, N, cout)));
    return dA;
  }

  /** Per-layer ablation flags. When `attnAblated[l]` is true the layer's
   *  attention OUTPUT contribution is dropped before the residual add — the
   *  block becomes "MLP-only" at depth l. `mlpAblated[l]` similarly drops
   *  the MLP contribution. Both true at l = whole layer skipped (residual
   *  passes through unchanged). Editable from JS for interpretability
   *  probes; cleared after each `generateAblated` call. */
  public attnAblated: boolean[] = [];
  public mlpAblated: boolean[] = [];

  /** Position-level patch: at layer l, position p, ZERO OUT the
   *  block's residual stream value (replace x[p,:] with zeros for the
   *  remainder of the forward pass). The simplest causal intervention
   *  — pinpoints whether a specific token's representation at a
   *  specific depth is load-bearing for the model's output.
   *
   *  Full donor → recipient activation patching (Meng et al., 2022)
   *  swaps in another forward's saved hidden state instead of zeroing.
   *  That requires a tensor-slice "replace one row" kernel — queued
   *  behind this simpler intervention.
   *
   *  Each entry is consumed by `forward` and cleared after
   *  `generatePatched` returns. */
  public patchPositions: { layer: number; position: number }[] = [];

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
      // Ablation: when set, the attn output is dropped (residual unchanged).
      // We still RUN attn — cheaper than a per-step recompile, and the
      // downstream tensors stay shape-consistent for the rest of the trace.
      const r1 = this.attnAblated[l]
        ? blockIn
        : this.keep(this.ops.add(blockIn, ao, N * C));
      const ln2 = this.ops.layernormForward(r1, ly.ln2g.w, ly.ln2b.w, N, C);
      this.keep(ln2.y); this.keep(ln2.mean); this.keep(ln2.rstd);
      const hpre = this.linear(ln2.y, ly.fcInW, ly.fcInB, N, C, M);
      const hact = this.keep(this.ops.gelu(hpre, N * M));
      const mo = this.linear(hact, ly.fcOutW, ly.fcOutB, N, M, C);
      var r2 = this.mlpAblated[l]
        ? r1
        : this.keep(this.ops.add(r1, mo, N * C));
      // Position-zero patch: at the specified (layer, position),
      // replace the block's residual-stream output with zeros for that
      // single position. Pure causal intervention — exposes whether
      // that token's representation at this depth is load-bearing.
      for (const patch of this.patchPositions) {
        if (patch.layer !== l) continue;
        if (patch.position < 0 || patch.position >= T) continue;
        r2 = this.keep(this.ops.zeroRow(r2, patch.position, T, C));
      }
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

  /**
   * Inverse of `exportState`. Reads the same flat float32 buffer layout
   * (4-byte int32 step prefix + per-param triplets [w, m, v]) and uploads
   * each tensor back to the corresponding GPU buffer. The param order in
   * this.params matches initWeights(), which matches the on-disk manifest,
   * so a straight sequential walk is correct.
   *
   * Throws if the buffer size doesn't match the model's expected param
   * footprint — typically means the saved file was for a different config.
   */
  importState(state: ArrayBuffer): void {
    let totalFloats = 0;
    for (const p of this.params) totalFloats += p.size * 3;
    const expectedBytes = 4 + totalFloats * 4;
    if (state.byteLength !== expectedBytes) {
      throw new Error(
        `state size mismatch: got ${state.byteLength} bytes, expected ${expectedBytes} ` +
        `for ${this.params.length} params totaling ${totalFloats} floats. ` +
        `The saved checkpoint was probably for a different config.`,
      );
    }
    this.stepCount = new Int32Array(state, 0, 1)[0];
    const f32 = new Float32Array(state, 4);
    let off = 0;
    for (const p of this.params) {
      p.w.upload(new Float32Array(f32.buffer, f32.byteOffset + off * 4, p.size)); off += p.size;
      p.m.upload(new Float32Array(f32.buffer, f32.byteOffset + off * 4, p.size)); off += p.size;
      p.v.upload(new Float32Array(f32.buffer, f32.byteOffset + off * 4, p.size)); off += p.size;
    }
  }

  /** Generate WITH per-(layer, position) zero-patches applied.
   *  Phase-8 activation patching, simplest variant: the residual
   *  stream at the specified (layer, position) is zeroed AFTER that
   *  layer's MLP add, before the next layer's input. The model's
   *  output reveals how load-bearing that token's representation
   *  was at that depth. Full donor → recipient swap (Meng et al.,
   *  2022) is the next iteration. */
  async generatePatched(
    promptIds: number[],
    patches: { layer: number; position: number }[],
    maxNew: number, temperature: number, topK: number, seed: number,
    onToken?: (tok: number, idxIntoMaxNew: number) => void,
  ): Promise<number[]> {
    this.patchPositions = patches.slice();
    try {
      return await this.generate(promptIds, maxNew, temperature, topK, seed, onToken);
    } finally {
      this.patchPositions = [];
    }
  }

  /** Generate WITH per-layer ablations applied. Sets the `attnAblated` /
   *  `mlpAblated` masks for the duration of the run, then clears them.
   *  Used by the interpretability "ablate" tool — gives a direct
   *  comparison of "what the model says without this layer's attention"
   *  vs the baseline. */
  async generateAblated(
    promptIds: number[],
    ablations: { layer: number; target: "attn" | "mlp" | "all" }[],
    maxNew: number, temperature: number, topK: number, seed: number,
    onToken?: (tok: number, idxIntoMaxNew: number) => void,
  ): Promise<number[]> {
    const { layers: L } = this.cfg;
    this.attnAblated = new Array(L).fill(false);
    this.mlpAblated = new Array(L).fill(false);
    for (const ab of ablations) {
      if (ab.layer < 0 || ab.layer >= L) continue;
      if (ab.target === "attn" || ab.target === "all") this.attnAblated[ab.layer] = true;
      if (ab.target === "mlp"  || ab.target === "all") this.mlpAblated[ab.layer]  = true;
    }
    try {
      return await this.generate(promptIds, maxNew, temperature, topK, seed, onToken);
    } finally {
      this.attnAblated = new Array(L).fill(false);
      this.mlpAblated = new Array(L).fill(false);
    }
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
   * Logit lens (Nostalgebraist 2020): project each layer's residual-
   * stream hidden state through the final layernorm + tied LM head,
   * revealing what the model "would predict" if it stopped at that
   * depth. The lens isn't a true prediction (the later layers always
   * shift the distribution), but it shows when specific knowledge or
   * structure first appears in the residual stream.
   *
   * Returns one `Float32Array` per layer, each shaped row-major
   * `[T, vocab]`. Index 0 = first block's output, index L-1 = final
   * block's output (which equals the model's real prediction).
   *
   * Cost: one extra LN + lm-head matmul per block. ~L× the head
   * cost of a normal forward; still cheap for the L≤12 configs we
   * actually run.
   */
  async logitLens(promptIds: number[]): Promise<Float32Array[]> {
    const { vocab: V, ctx, layers: L, heads: H, dModel: C, dMlp: M } = this.cfg;
    if (promptIds.length === 0) return [];
    const T = Math.min(promptIds.length, ctx);
    const N = T;
    const window = new Float32Array(promptIds.slice(promptIds.length - T));

    this.ops.beginBatch();
    const idsT = this.keep(this.tensorFrom(window));
    const x0 = this.keep(
      this.ops.embedForward(this.tokEmb.w, this.posEmb.w, idsT, N, C, T));
    let x = x0;
    const perLayerLogits: import("./tensor").GpuTensor[] = [];

    for (let l = 0; l < L; l++) {
      const ly = this.layers[l];
      // Standard block forward — kept inline (vs. calling private `forward`)
      // so we control what gets `keep`'d. The training caches aren't needed.
      const ln1 = this.ops.layernormForward(x, ly.ln1g.w, ly.ln1b.w, N, C);
      this.keep(ln1.y); this.keep(ln1.mean); this.keep(ln1.rstd);
      const q = this.linear(ln1.y, ly.wq, ly.bq, N, C, C);
      const k = this.linear(ln1.y, ly.wk, ly.bk, N, C, C);
      const v = this.linear(ln1.y, ly.wv, ly.bv, N, C, C);
      const att = this.ops.attentionForward(q, k, v, 1, T, C, H);
      this.keep(att.attn); this.keep(att.ctx);
      if (att.L) this.keep(att.L);
      const ao = this.linear(att.ctx, ly.wo, ly.bo, N, C, C);
      const r1 = this.keep(this.ops.add(x, ao, N * C));
      const ln2 = this.ops.layernormForward(r1, ly.ln2g.w, ly.ln2b.w, N, C);
      this.keep(ln2.y); this.keep(ln2.mean); this.keep(ln2.rstd);
      const hpre = this.linear(ln2.y, ly.fcInW, ly.fcInB, N, C, M);
      const hact = this.keep(this.ops.gelu(hpre, N * M));
      const mo = this.linear(hact, ly.fcOutW, ly.fcOutB, N, M, C);
      const r2 = this.keep(this.ops.add(r1, mo, N * C));
      x = r2;

      // Lens projection: re-apply final LN + tied LM head to the
      // current depth's residual stream. The final LN's stats are
      // computed locally (not the same as if all layers ran first)
      // — this is the standard interpretation of "logit lens."
      const lnfL = this.ops.layernormForward(x, this.lnfG.w, this.lnfB.w, N, C);
      this.keep(lnfL.y); this.keep(lnfL.mean); this.keep(lnfL.rstd);
      const logitsL = this.keep(this.ops.matmulAbt(lnfL.y, this.tokEmb.w, N, C, V));
      perLayerLogits.push(logitsL);
    }
    this.ops.endBatch();

    const out: Float32Array[] = [];
    for (const lg of perLayerLogits) {
      out.push(await lg.download());
    }
    this.freeScratch();
    return out;
  }

  /**
   * Full-sequence logits for one prompt.
   *
   * The benchmark runner uses this to score every position in a held-out
   * sequence in one pass — vs the autoregressive `generate` which can
   * only sample. Returns the entire `[T, vocab]` logits matrix row-major.
   *
   * Promtps are clipped to `ctx` from the LEFT (matching `generate`'s
   * windowing) so long sequences don't OOM the attention buffer.
   */
  async forwardLogits(promptIds: number[]): Promise<Float32Array> {
    const { ctx } = this.cfg;
    if (promptIds.length === 0) return new Float32Array(0);
    const T = Math.min(promptIds.length, ctx);
    const window = new Float32Array(promptIds.slice(promptIds.length - T));
    this.ops.beginBatch();
    const fwd = this.forward(window, 1, T);
    this.ops.endBatch();
    const logits = await fwd.logits.download();
    this.freeScratch();
    return logits;
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

  /** Repack every matmul-shaped weight's f16 mirror from its (just-updated)
   *  f32 source. Called at the end of trainStep after AdamW has written
   *  fresh values to p.w. Cost: one bandwidth-bound dispatch per packed
   *  weight; total work is roughly one pass through the weight buffer set
   *  per step (~10MB on the Huge preset → ~0.07ms on M-series). The
   *  alternative (invalidating + falling back to f32 for training) would
   *  leave the matmul reads on f32 and forfeit the bandwidth win on the
   *  forward+dA matmuls of training. */
  private repackF16Mirrors(): void {
    if (!this.useF16Storage) return;
    for (const p of this.params) {
      if (!p.wF16 || !p.matShape) continue;
      const [rows, cols] = p.matShape;
      this.ops.packToF16(p.w.buffer, p.wF16, rows, cols);
    }
  }

  /** One training step: forward, cross-entropy, backward, AdamW. Returns loss. */
  async trainStep(
    ids: Float32Array, targets: Float32Array, batch: number, lr: number,
  ): Promise<number> {
    // Lazy f16-storage activation: the ops-level numerics gate runs at
    // construction time and settles independently. If it passed and we
    // haven't yet packed weights for this model, do so on the first
    // training step. After that, useF16Storage stays true; per-step
    // overhead is the repackF16Mirrors() pass at the bottom of this method.
    // No-op (and no await) on subsequent steps because prepareForInference
    // short-circuits when useF16Storage is already true.
    if (!this.useF16Storage) {
      // prepareForInference is idempotent and resolves false fast when the
      // gate failed, so this won't slow training on devices where the f16
      // path isn't activating.
      await this.prepareForInference();
    }
    // If the f16-storage path is active, training reads weights from the
    // packed mirrors set up by prepareForInference. We keep those mirrors
    // in sync at the END of each step (after AdamW writes new values to
    // p.w). The forward + dA matmuls within this step still read the
    // mirror state from the previous step's repack, which exactly matches
    // p.w pre-AdamW — correct semantics.
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

    // Keep the f16 mirrors in sync with the just-updated f32 weights so the
    // next step's forward + dA matmuls see fresh values. No-op when
    // useF16Storage is false (training on the f32 vec4 path).
    this.repackF16Mirrors();

    this.ops.endBatch(); // submit the whole step at once
    const lossArr = await ce.loss.download();
    let total = 0;
    for (const x2 of lossArr) total += x2;
    this.freeScratch();
    return total / N;
  }

  /** Free every GPU resource this model owns. Use case: auto-offload after
   *  the user has been idle for N minutes — the model occupies ~110 MB of
   *  GPU memory just sitting there. After destroy() the instance is dead:
   *  any subsequent method call will throw or produce garbage. Callers
   *  should null out their reference and reload the model from scratch
   *  (gallery refetch or OPFS restore) when the user comes back. */
  destroy(): void {
    for (const p of this.params) {
      p.w.destroy();
      p.m.destroy();
      p.v.destroy();
      if (p.wF16) { p.wF16.destroy(); p.wF16 = null; }
    }
    this.params.length = 0;
    this.layers.length = 0;
    this.scratch.length = 0;
    this.ops.destroy();
  }

  // Return every per-step tensor to the buffer pool (not destroyed) so the
  // next step reuses the buffers — after step 1, a run does no allocation.
  private freeScratch(): void {
    for (const t of this.scratch) t.recycle();
    this.scratch = [];
  }
}
