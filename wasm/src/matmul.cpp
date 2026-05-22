// matmul.cpp — matrix multiply, forward + backward (Phase 4).
//
// The single most performance-critical kernel. Scalar and correct first; a
// SIMD build (-msimd128) and the WebGPU port (Phase 5) come later, each
// verified against this scalar version.
//
//   Forward:   C = A @ B            A:[M,K]  B:[K,N]  C:[M,N]
//   Backward:  dA = dC @ B^T        dB = A^T @ dC
//
// Used by: Linear layers, the attention projections, the output head.
//
// Guide: docs/browser_notes.md ("WASM backend", "WebGPU acceleration")

#include "kernels.h"

// C = A @ B. The k-loop is innermost-but-one so the B and C row walks are
// sequential in memory — friendlier to the cache than a naive i,j,k order.
WASM_EXPORT void matmul_forward(const float* A, const float* B, float* C,
                                int M, int K, int N) {
  for (int m = 0; m < M; ++m) {
    float* c_row = C + static_cast<long>(m) * N;
    for (int n = 0; n < N; ++n) c_row[n] = 0.0f;
    const float* a_row = A + static_cast<long>(m) * K;
    for (int k = 0; k < K; ++k) {
      const float a = a_row[k];
      const float* b_row = B + static_cast<long>(k) * N;
      for (int n = 0; n < N; ++n) c_row[n] += a * b_row[n];
    }
  }
}

// dA = dC @ B^T  and  dB = A^T @ dC. Both outputs are overwritten.
WASM_EXPORT void matmul_backward(const float* A, const float* B, const float* dC,
                                 float* dA, float* dB, int M, int K, int N) {
  // dA[m,k] = sum_n dC[m,n] * B[k,n]
  for (int m = 0; m < M; ++m) {
    float* da_row = dA + static_cast<long>(m) * K;
    for (int k = 0; k < K; ++k) da_row[k] = 0.0f;
    const float* dc_row = dC + static_cast<long>(m) * N;
    for (int k = 0; k < K; ++k) {
      const float* b_row = B + static_cast<long>(k) * N;
      float acc = 0.0f;
      for (int n = 0; n < N; ++n) acc += dc_row[n] * b_row[n];
      da_row[k] = acc;
    }
  }
  // dB[k,n] = sum_m A[m,k] * dC[m,n]
  for (int i = 0; i < K * N; ++i) dB[i] = 0.0f;
  for (int m = 0; m < M; ++m) {
    const float* a_row = A + static_cast<long>(m) * K;
    const float* dc_row = dC + static_cast<long>(m) * N;
    for (int k = 0; k < K; ++k) {
      const float a = a_row[k];
      float* db_row = dB + static_cast<long>(k) * N;
      for (int n = 0; n < N; ++n) db_row[n] += a * dc_row[n];
    }
  }
}
