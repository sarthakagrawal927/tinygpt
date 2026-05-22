/**
 * webgpu-test.ts — parity tests for the WebGPU training kernels (Phase 5).
 *
 * Runs in webgpu-test.html. Every kernel is checked against a plain-JS
 * reference; results go to #results for the Playwright runner (webgpu_test.mjs).
 * This is the verification harness every WebGPU-training stage adds to.
 */

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

  out.textContent =
    lines.join("\n") + "\n\n" + (failed === 0 ? "ALL PASS" : "SOME TESTS FAILED");
}

void main();
