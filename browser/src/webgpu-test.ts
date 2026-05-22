/**
 * webgpu-test.ts — parity tests for the WebGPU training kernels (Phase 5).
 *
 * Runs in webgpu-test.html. Every kernel is checked against a plain-JS
 * reference; results go to #results for the Playwright runner (webgpu_test.mjs).
 * This is the verification harness every WebGPU-training stage adds to.
 */

import { GpuModel } from "../../webgpu/gpu_model";
import { GpuOps } from "../../webgpu/ops";
import { createGpuContext, GpuTensor } from "../../webgpu/tensor";

const out = document.getElementById("results") as HTMLPreElement;
const lines: string[] = [];
let failed = 0;

function check(name: string, ok: boolean, detail: string): void {
  lines.push(`${ok ? "ok  " : "FAIL"} ${name.padEnd(28)} ${detail}`);
  if (!ok) failed++;
  out.textContent = lines.join("\n");
}

function rand(n: number, scale = 1): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (Math.random() * 2 - 1) * scale;
  return a;
}

function maxError(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

// --- plain-JS references --------------------------------------------------
function refMatmul(A: Float32Array, B: Float32Array, M: number, K: number, N: number) {
  const c = new Float32Array(M * N);
  for (let m = 0; m < M; m++)
    for (let n = 0; n < N; n++) {
      let s = 0;
      for (let k = 0; k < K; k++) s += A[m * K + k] * B[k * N + n];
      c[m * N + n] = s;
    }
  return c;
}
function refDA(dC: Float32Array, B: Float32Array, M: number, K: number, N: number) {
  const o = new Float32Array(M * K);
  for (let m = 0; m < M; m++)
    for (let k = 0; k < K; k++) {
      let s = 0;
      for (let n = 0; n < N; n++) s += dC[m * N + n] * B[k * N + n];
      o[m * K + k] = s;
    }
  return o;
}
function refDB(A: Float32Array, dC: Float32Array, M: number, K: number, N: number) {
  const o = new Float32Array(K * N);
  for (let k = 0; k < K; k++)
    for (let n = 0; n < N; n++) {
      let s = 0;
      for (let m = 0; m < M; m++) s += A[m * K + k] * dC[m * N + n];
      o[k * N + n] = s;
    }
  return o;
}
// Abramowitz & Stegun 7.1.26 — the same approximation the WGSL kernel uses.
function erf(x: number): number {
  const s = Math.sign(x), ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) *
      t +
      0.254829592) *
      t *
      Math.exp(-ax * ax);
  return s * y;
}
const gelu = (x: number) => 0.5 * x * (1 + erf(x * 0.70710678118));
const geluGrad = (x: number) =>
  0.5 * (1 + erf(x * 0.70710678118)) + x * 0.3989422804 * Math.exp(-0.5 * x * x);

async function main(): Promise<void> {
  const ctx = await createGpuContext();
  if (!ctx) {
    out.textContent = "SKIP — no WebGPU in this browser";
    return;
  }
  const ops = GpuOps.create(ctx);
  const dev = ctx.device;
  const tol = 2e-3;

  // --- matmul (stage 1) ---------------------------------------------------
  {
    const M = 24, K = 40, N = 18;
    const A = rand(M * K), B = rand(K * N), dC = rand(M * N);
    const At = GpuTensor.fromData(dev, A), Bt = GpuTensor.fromData(dev, B);
    const dCt = GpuTensor.fromData(dev, dC);
    const C = await ops.matmul(At, Bt, M, K, N).download();
    check("matmul forward", maxError(C, refMatmul(A, B, M, K, N)) < tol, "");
    const { dA, dB } = ops.matmulBackward(At, Bt, dCt, M, K, N);
    check("matmul backward dA", maxError(await dA.download(), refDA(dC, B, M, K, N)) < tol, "");
    check("matmul backward dB", maxError(await dB.download(), refDB(A, dC, M, K, N)) < tol, "");
  }

  // --- elementwise (stage 2) ---------------------------------------------
  {
    const n = 200;
    const a = rand(n), b = rand(n);
    const sum = await ops.add(GpuTensor.fromData(dev, a), GpuTensor.fromData(dev, b), n).download();
    const refSum = a.map((v, i) => v + b[i]);
    check("add", maxError(sum, refSum) < tol, "");

    const x = rand(n, 3);
    const xt = GpuTensor.fromData(dev, x);
    const gy = await ops.gelu(xt, n).download();
    check("gelu forward", maxError(gy, x.map(gelu)) < tol, "");
    const dy = rand(n);
    const gdx = await ops.geluBackward(xt, GpuTensor.fromData(dev, dy), n).download();
    check("gelu backward", maxError(gdx, x.map((v, i) => dy[i] * geluGrad(v))) < tol, "");
  }

  // --- bias ---------------------------------------------------------------
  {
    const rows = 12, D = 16;
    const y = rand(rows * D), bias = rand(D);
    const yt = GpuTensor.fromData(dev, y);
    ops.biasAdd(yt, GpuTensor.fromData(dev, bias), rows, D);
    const got = await yt.download();
    const refY = y.map((v, i) => v + bias[i % D]);
    check("bias add", maxError(got, refY) < tol, "");

    const dyB = rand(rows * D);
    const db = await ops.biasGrad(GpuTensor.fromData(dev, dyB), rows, D).download();
    const refDb = new Float32Array(D);
    for (let r = 0; r < rows; r++) for (let d = 0; d < D; d++) refDb[d] += dyB[r * D + d];
    check("bias grad", maxError(db, refDb) < tol, "");
  }

  // --- layernorm ----------------------------------------------------------
  {
    const N = 20, D = 48, eps = 1e-5;
    const x = rand(N * D, 2), gamma = rand(D, 1), beta = rand(D, 1), dy = rand(N * D);
    const xt = GpuTensor.fromData(dev, x);
    const gt = GpuTensor.fromData(dev, gamma);
    const fwd = ops.layernormForward(xt, gt, GpuTensor.fromData(dev, beta), N, D, eps);

    // reference forward
    const refY = new Float32Array(N * D);
    const mean = new Float32Array(N), rstd = new Float32Array(N);
    for (let n = 0; n < N; n++) {
      let mu = 0;
      for (let d = 0; d < D; d++) mu += x[n * D + d];
      mu /= D;
      let v = 0;
      for (let d = 0; d < D; d++) v += (x[n * D + d] - mu) ** 2;
      v /= D;
      mean[n] = mu;
      rstd[n] = 1 / Math.sqrt(v + eps);
      for (let d = 0; d < D; d++)
        refY[n * D + d] = gamma[d] * (x[n * D + d] - mu) * rstd[n] + beta[d];
    }
    check("layernorm forward", maxError(await fwd.y.download(), refY) < tol, "");

    const bwd = ops.layernormBackward(xt, gt, fwd.mean, fwd.rstd,
      GpuTensor.fromData(dev, dy), N, D);
    // reference backward
    const refDx = new Float32Array(N * D);
    const refDg = new Float32Array(D), refDbeta = new Float32Array(D);
    for (let n = 0; n < N; n++) {
      let mdx = 0, mdxx = 0;
      for (let d = 0; d < D; d++) {
        const xhat = (x[n * D + d] - mean[n]) * rstd[n];
        const dxhat = dy[n * D + d] * gamma[d];
        mdx += dxhat;
        mdxx += dxhat * xhat;
        refDg[d] += dy[n * D + d] * xhat;
        refDbeta[d] += dy[n * D + d];
      }
      mdx /= D;
      mdxx /= D;
      for (let d = 0; d < D; d++) {
        const xhat = (x[n * D + d] - mean[n]) * rstd[n];
        const dxhat = dy[n * D + d] * gamma[d];
        refDx[n * D + d] = rstd[n] * (dxhat - mdx - xhat * mdxx);
      }
    }
    check("layernorm backward dx", maxError(await bwd.dx.download(), refDx) < tol, "");
    check("layernorm backward dgamma", maxError(await bwd.dgamma.download(), refDg) < tol, "");
    check("layernorm backward dbeta", maxError(await bwd.dbeta.download(), refDbeta) < tol, "");
  }

  // --- attention (stage 3) ------------------------------------------------
  {
    const B = 2, T = 12, C = 24, H = 3;
    const hd = C / H, scale = 1 / Math.sqrt(hd);
    const q = rand(B * T * C), k = rand(B * T * C), v = rand(B * T * C);
    const qt = GpuTensor.fromData(dev, q);
    const kt = GpuTensor.fromData(dev, k);
    const vt = GpuTensor.fromData(dev, v);
    const fwd = ops.attentionForward(qt, kt, vt, B, T, C, H);

    // reference forward: causal scaled-dot-product attention
    const refAttn = new Float32Array(B * H * T * T);
    const refCtx = new Float32Array(B * T * C);
    for (let b = 0; b < B; b++)
      for (let h = 0; h < H; h++) {
        const off = h * hd;
        for (let t1 = 0; t1 < T; t1++) {
          const sc: number[] = [];
          let mx = -1e30;
          for (let t2 = 0; t2 <= t1; t2++) {
            let s = 0;
            for (let d = 0; d < hd; d++)
              s += q[(b * T + t1) * C + off + d] * k[(b * T + t2) * C + off + d];
            s *= scale;
            sc[t2] = s;
            if (s > mx) mx = s;
          }
          let sum = 0;
          for (let t2 = 0; t2 <= t1; t2++) { sc[t2] = Math.exp(sc[t2] - mx); sum += sc[t2]; }
          const arow = ((b * H + h) * T + t1) * T;
          for (let t2 = 0; t2 <= t1; t2++) refAttn[arow + t2] = sc[t2] / sum;
          for (let d = 0; d < hd; d++) {
            let acc = 0;
            for (let t2 = 0; t2 <= t1; t2++)
              acc += refAttn[arow + t2] * v[(b * T + t2) * C + off + d];
            refCtx[(b * T + t1) * C + off + d] = acc;
          }
        }
      }
    check("attention forward attn", maxError(await fwd.attn.download(), refAttn) < tol, "");
    check("attention forward ctx", maxError(await fwd.ctx.download(), refCtx) < tol, "");

    // reference backward
    const dctx = rand(B * T * C);
    const bwd = ops.attentionBackward(qt, kt, vt, fwd.attn,
      GpuTensor.fromData(dev, dctx), B, T, C, H);
    const refDq = new Float32Array(B * T * C);
    const refDk = new Float32Array(B * T * C);
    const refDv = new Float32Array(B * T * C);
    for (let b = 0; b < B; b++)
      for (let h = 0; h < H; h++) {
        const off = h * hd;
        for (let t1 = 0; t1 < T; t1++) {
          const arow = ((b * H + h) * T + t1) * T;
          const dattn: number[] = [];
          let dot = 0;
          for (let t2 = 0; t2 <= t1; t2++) {
            let da = 0;
            for (let d = 0; d < hd; d++)
              da += dctx[(b * T + t1) * C + off + d] * v[(b * T + t2) * C + off + d];
            dattn[t2] = da;
            dot += da * refAttn[arow + t2];
          }
          for (let t2 = 0; t2 <= t1; t2++) {
            const ds = refAttn[arow + t2] * (dattn[t2] - dot) * scale;
            for (let d = 0; d < hd; d++) {
              refDq[(b * T + t1) * C + off + d] += ds * k[(b * T + t2) * C + off + d];
              refDk[(b * T + t2) * C + off + d] += ds * q[(b * T + t1) * C + off + d];
            }
          }
          for (let t2 = 0; t2 <= t1; t2++) {
            const a = refAttn[arow + t2];
            for (let d = 0; d < hd; d++)
              refDv[(b * T + t2) * C + off + d] += a * dctx[(b * T + t1) * C + off + d];
          }
        }
      }
    check("attention backward dq", maxError(await bwd.dq.download(), refDq) < tol, "");
    check("attention backward dk", maxError(await bwd.dk.download(), refDk) < tol, "");
    check("attention backward dv", maxError(await bwd.dv.download(), refDv) < tol, "");
  }

  // --- embeddings / cross-entropy / optimizer (stage 4) ------------------
  {
    const B = 2, T = 6, C = 8, V = 20, N = B * T;
    const tok = rand(V * C), pos = rand(T * C);
    const ids = new Float32Array(N);
    for (let i = 0; i < N; i++) ids[i] = Math.floor(Math.random() * V);

    const x = await ops
      .embedForward(GpuTensor.fromData(dev, tok), GpuTensor.fromData(dev, pos),
        GpuTensor.fromData(dev, ids), N, C, T)
      .download();
    const refX = new Float32Array(N * C);
    for (let n = 0; n < N; n++)
      for (let c = 0; c < C; c++)
        refX[n * C + c] = tok[ids[n] * C + c] + pos[(n % T) * C + c];
    check("embed forward", maxError(x, refX) < tol, "");

    const dx = rand(N * C);
    const dtok = await ops
      .embedTokGrad(GpuTensor.fromData(dev, dx), GpuTensor.fromData(dev, ids), N, C, V)
      .download();
    const refDtok = new Float32Array(V * C);
    for (let n = 0; n < N; n++)
      for (let c = 0; c < C; c++) refDtok[ids[n] * C + c] += dx[n * C + c];
    check("embed tok grad", maxError(dtok, refDtok) < tol, "");

    const dpos = await ops.embedPosGrad(GpuTensor.fromData(dev, dx), N, C, T).download();
    const refDpos = new Float32Array(T * C);
    for (let n = 0; n < N; n++)
      for (let c = 0; c < C; c++) refDpos[(n % T) * C + c] += dx[n * C + c];
    check("embed pos grad", maxError(dpos, refDpos) < tol, "");

    const Nc = 10, Vc = 20;
    const logits = rand(Nc * Vc, 3);
    const tgts = new Float32Array(Nc);
    for (let i = 0; i < Nc; i++) tgts[i] = Math.floor(Math.random() * Vc);
    const ce = ops.crossEntropy(GpuTensor.fromData(dev, logits),
      GpuTensor.fromData(dev, tgts), Nc, Vc);
    const dl = await ce.dlogits.download();
    const refDl = new Float32Array(Nc * Vc);
    for (let n = 0; n < Nc; n++) {
      const base = n * Vc;
      let mx = logits[base];
      for (let v = 1; v < Vc; v++) mx = Math.max(mx, logits[base + v]);
      let sum = 0;
      for (let v = 0; v < Vc; v++) sum += Math.exp(logits[base + v] - mx);
      for (let v = 0; v < Vc; v++) {
        const pr = Math.exp(logits[base + v] - mx) / sum;
        refDl[base + v] = (pr - (v === tgts[n] ? 1 : 0)) / Nc;
      }
    }
    check("cross-entropy dlogits", maxError(dl, refDl) < tol, "");

    const cnt = 16, step = 3, lr = 0.01, wd = 0.1;
    const pm = rand(cnt), gr = rand(cnt);
    const mm = rand(cnt, 0.1);
    const vv = rand(cnt, 0.1).map((z) => Math.abs(z));
    const pt = GpuTensor.fromData(dev, pm);
    const mt = GpuTensor.fromData(dev, mm);
    const vt = GpuTensor.fromData(dev, vv);
    ops.adamwStep(pt, GpuTensor.fromData(dev, gr), mt, vt, cnt, step, lr, wd);
    const pAfter = await pt.download();
    const refP = new Float32Array(cnt);
    const b1 = 0.9, b2 = 0.95, eps = 1e-8;
    for (let i = 0; i < cnt; i++) {
      const m = b1 * mm[i] + (1 - b1) * gr[i];
      const v = b2 * vv[i] + (1 - b2) * gr[i] * gr[i];
      const mh = m / (1 - Math.pow(b1, step));
      const vh = v / (1 - Math.pow(b2, step));
      refP[i] = pm[i] - lr * (mh / (Math.sqrt(vh) + eps) + wd * pm[i]);
    }
    check("adamw step", maxError(pAfter, refP) < tol, "");
  }

  // --- full GPU training: the overfit gate (stage 5) ----------------------
  // If the entire GPU forward + backward + AdamW is correct, a tiny model
  // drives the loss on one fixed batch from ~ln(256) to near zero.
  {
    const cfg = { vocab: 256, ctx: 16, layers: 2, heads: 2, dModel: 32, dMlp: 64, seed: 42 };
    const model = new GpuModel(ctx, cfg);
    const batch = 8;
    const corpus = new TextEncoder().encode(
      "the quick brown fox jumps over the lazy dog. ".repeat(20));
    const ids = new Float32Array(batch * cfg.ctx);
    const targets = new Float32Array(batch * cfg.ctx);
    for (let b = 0; b < batch; b++) {
      const s = b * 7;
      for (let t = 0; t < cfg.ctx; t++) {
        ids[b * cfg.ctx + t] = corpus[s + t];
        targets[b * cfg.ctx + t] = corpus[s + t + 1];
      }
    }
    const first = await model.trainStep(ids, targets, batch, 5e-3);
    let loss = first;
    for (let i = 0; i < 150; i++) loss = await model.trainStep(ids, targets, batch, 5e-3);
    check("gpu training: initial loss ~ ln(256)", Math.abs(first - 5.545) < 0.7,
      first.toFixed(3));
    check("gpu training: overfits a batch (loss < 0.5)", loss < 0.5,
      `${first.toFixed(2)} -> ${loss.toFixed(3)}`);
  }

  out.textContent =
    lines.join("\n") + "\n\n" + (failed === 0 ? "ALL PASS" : "SOME TESTS FAILED");
}

void main();
