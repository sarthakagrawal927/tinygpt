// matmul.cpp — matrix multiply, forward + backward (Phase 4 + threading pass).
//
// The single most performance-critical kernel. The inner loop is the original
// "iko" order — emcc autovectorises this cleanly with -msimd128. Threading
// splits the M dimension across CPU cores via std::thread.
//
//   Forward:   C = A @ B            A:[M,K]  B:[K,N]  C:[M,N]
//   Backward:  dA = dC @ B^T        dB = A^T @ dC
//
// Used by: Linear layers, the attention projections, the output head.
//
// For matmul_forward and the dA path, every output row is independent —
// each thread gets a contiguous slice of M and writes only to its own region.
// dB accumulates over M so we use per-thread scratch and a final reduction.
//
// Threading kicks in only when M is large enough that thread-creation
// overhead is amortised (M >= 64). Below that, single-threaded.
//
// Guide: docs/browser_notes.md ("WASM backend", "WebGPU acceleration")

#include "kernels.h"

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

// Cap pthread count to match the WASM build's PTHREAD_POOL_SIZE.
static constexpr int MAX_THREADS = 8;
// Threshold below which threading overhead exceeds the parallel win.
static constexpr int MIN_M_FOR_THREADS = 64;

static int chosen_threads(int M) {
  if (M < MIN_M_FOR_THREADS) return 1;
  const int hw = static_cast<int>(std::thread::hardware_concurrency());
  const int cap = hw > 0 ? std::min(hw, MAX_THREADS) : MAX_THREADS;
  // Each thread should chew on at least ~16 rows.
  return std::min(cap, std::max(1, M / 16));
}

// ---------------------------------------------------------------------------
// C = A @ B    A:[M,K]  B:[K,N]  C:[M,N]
// ---------------------------------------------------------------------------
static void matmul_forward_serial(const float* A, const float* B, float* C,
                                   int m_lo, int m_hi, int K, int N) {
  for (int m = m_lo; m < m_hi; ++m) {
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

WASM_EXPORT void matmul_forward(const float* A, const float* B, float* C,
                                int M, int K, int N) {
  const int nthreads = chosen_threads(M);
  if (nthreads == 1) {
    matmul_forward_serial(A, B, C, 0, M, K, N);
    return;
  }
  const int chunk = (M + nthreads - 1) / nthreads;
  std::vector<std::thread> workers;
  workers.reserve(nthreads - 1);
  for (int t = 1; t < nthreads; ++t) {
    int m_lo = t * chunk;
    int m_hi = std::min(m_lo + chunk, M);
    if (m_lo >= M) break;
    workers.emplace_back(matmul_forward_serial, A, B, C, m_lo, m_hi, K, N);
  }
  matmul_forward_serial(A, B, C, 0, std::min(chunk, M), K, N);
  for (auto& w : workers) w.join();
}

// ---------------------------------------------------------------------------
// dA = dC @ B^T     dB = A^T @ dC.   Both outputs overwritten.
//
// dA: each row dA[m,:] is independent — splittable like forward.
// dB: each cell dB[k,n] accumulates over m. Per-thread scratch + reduction.
// ---------------------------------------------------------------------------
static void backward_dA_serial(const float* B, const float* dC, float* dA,
                                int m_lo, int m_hi, int K, int N) {
  for (int m = m_lo; m < m_hi; ++m) {
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
}

static void backward_dB_partial(const float* A, const float* dC, float* dB_local,
                                 int m_lo, int m_hi, int K, int N) {
  for (long i = 0; i < static_cast<long>(K) * N; ++i) dB_local[i] = 0.0f;
  for (int m = m_lo; m < m_hi; ++m) {
    const float* a_row = A + static_cast<long>(m) * K;
    const float* dc_row = dC + static_cast<long>(m) * N;
    for (int k = 0; k < K; ++k) {
      const float a = a_row[k];
      float* db_row = dB_local + static_cast<long>(k) * N;
      for (int n = 0; n < N; ++n) db_row[n] += a * dc_row[n];
    }
  }
}

WASM_EXPORT void matmul_backward(const float* A, const float* B, const float* dC,
                                 float* dA, float* dB, int M, int K, int N) {
  const int nthreads = chosen_threads(M);
  if (nthreads == 1) {
    backward_dA_serial(B, dC, dA, 0, M, K, N);
    backward_dB_partial(A, dC, dB, 0, M, K, N);
    return;
  }
  const int chunk = (M + nthreads - 1) / nthreads;

  // --- dA: independent rows, no reduction needed --------------------------
  {
    std::vector<std::thread> workers;
    workers.reserve(nthreads - 1);
    for (int t = 1; t < nthreads; ++t) {
      int m_lo = t * chunk;
      int m_hi = std::min(m_lo + chunk, M);
      if (m_lo >= M) break;
      workers.emplace_back(backward_dA_serial, B, dC, dA, m_lo, m_hi, K, N);
    }
    backward_dA_serial(B, dC, dA, 0, std::min(chunk, M), K, N);
    for (auto& w : workers) w.join();
  }

  // --- dB: accumulate per-thread, reduce at the end -----------------------
  const long kn = static_cast<long>(K) * N;
  std::vector<std::vector<float>> partials(nthreads);
  for (auto& p : partials) p.resize(kn);

  std::vector<std::thread> workers;
  workers.reserve(nthreads - 1);
  for (int t = 1; t < nthreads; ++t) {
    int m_lo = t * chunk;
    int m_hi = std::min(m_lo + chunk, M);
    if (m_lo >= M) break;
    workers.emplace_back(backward_dB_partial, A, dC, partials[t].data(),
                         m_lo, m_hi, K, N);
  }
  backward_dB_partial(A, dC, partials[0].data(), 0, std::min(chunk, M), K, N);
  for (auto& w : workers) w.join();

  // Reduction: dB = sum_t partials[t]. O(KN), << the matmul itself.
  std::memcpy(dB, partials[0].data(), kn * sizeof(float));
  for (int t = 1; t < nthreads; ++t) {
    const float* src = partials[t].data();
    for (long i = 0; i < kn; ++i) dB[i] += src[i];
  }
}
