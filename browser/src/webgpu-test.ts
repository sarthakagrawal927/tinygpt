/**
 * webgpu-test.ts — parity tests for the WebGPU training kernels (Phase 5).
 *
 * Runs in the page webgpu-test.html. Each kernel is checked against a plain-JS
 * reference; results are written to #results for the Playwright runner
 * (browser/webgpu_test.mjs) to read. This is the verification harness every
 * WebGPU-training stage adds to.
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

function rand(n: number): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = Math.random() * 2 - 1;
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
// dA[m,k] = sum_n dC[m,n] * B[k,n]
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
// dB[k,n] = sum_m A[m,k] * dC[m,n]
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

async function main(): Promise<void> {
  const ctx = await createGpuContext();
  if (!ctx) {
    out.textContent = "SKIP — no WebGPU in this browser";
    return;
  }
  const ops = GpuOps.create(ctx);

  // Non-square, non-multiple-of-16 dims exercise the ragged-edge guard.
  const M = 24, K = 40, N = 18;
  const A = rand(M * K), B = rand(K * N), dC = rand(M * N);
  const At = GpuTensor.fromData(ctx.device, A);
  const Bt = GpuTensor.fromData(ctx.device, B);
  const dCt = GpuTensor.fromData(ctx.device, dC);

  const C = await ops.matmul(At, Bt, M, K, N).download();
  const eC = maxError(C, refMatmul(A, B, M, K, N));
  check("matmul forward", eC < 1e-3, `max err ${eC.toExponential(2)}`);

  const { dA, dB } = ops.matmulBackward(At, Bt, dCt, M, K, N);
  const dAg = await dA.download();
  const dBg = await dB.download();
  const eA = maxError(dAg, refDA(dC, B, M, K, N));
  const eB = maxError(dBg, refDB(A, dC, M, K, N));
  check("matmul backward dA", eA < 1e-3, `max err ${eA.toExponential(2)}`);
  check("matmul backward dB", eB < 1e-3, `max err ${eB.toExponential(2)}`);

  out.textContent =
    lines.join("\n") + "\n\n" + (failed === 0 ? "ALL PASS" : "SOME TESTS FAILED");
}

void main();
