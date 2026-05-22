// test_wasm_kernels.cpp — correctness tests for the WASM CPU kernels (Phase 4).
//
// Each kernel carries its own hand-written backward (no autograd), so each
// backward is verified by a finite-difference gradient check: perturb every
// input element by +/-h, measure the change in a scalar loss, and compare to
// the analytic gradient. A wrong backward formula shows up as a large relative
// error here long before it corrupts a training run.
//
// Build + run natively (no Emscripten needed):
//     bash wasm/build_native.sh
// or directly:
//     clang++ -std=c++17 -O2 wasm/src/*.cpp tests/test_wasm_kernels.cpp \
//         -o /tmp/test_wasm_kernels && /tmp/test_wasm_kernels

#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "../wasm/src/kernels.h"

// --------------------------------------------------------------------------
// Test harness
// --------------------------------------------------------------------------
static int g_failed = 0;
static std::mt19937 g_rng(1234);

static void fill_random(std::vector<float>& buf, float scale = 1.0f) {
  std::uniform_real_distribution<float> dist(-scale, scale);
  for (float& x : buf) x = dist(g_rng);
}

static void report(const char* name, bool ok, double detail) {
  if (ok) {
    std::printf("ok  %-34s (%.2e)\n", name, detail);
  } else {
    std::printf("FAIL %-33s (%.2e)\n", name, detail);
    ++g_failed;
  }
}

// Central-difference gradient check. Returns the worst error between the
// analytic and a numerical gradient, normalized by the largest gradient
// component in the buffer. Normalizing by the buffer scale (rather than each
// element) is the standard robust metric: it does not blow up on the small,
// near-cancelling entries that the softmax path produces, while a genuinely
// wrong backward still lands far above the tolerance.
static double grad_check(float* param, int n, const float* analytic,
                         const std::function<double()>& loss, float h = 1e-3f) {
  double scale = 1e-6;
  for (int i = 0; i < n; ++i)
    scale = std::max(scale, static_cast<double>(std::fabs(analytic[i])));

  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const float orig = param[i];
    param[i] = orig + h;
    const double lp = loss();
    param[i] = orig - h;
    const double lm = loss();
    param[i] = orig;
    const double numeric = (lp - lm) / (2.0 * h);
    const double err = std::fabs(numeric - analytic[i]) / scale;
    if (err > worst) worst = err;
  }
  return worst;
}

// --------------------------------------------------------------------------
// matmul
// --------------------------------------------------------------------------
static void test_matmul() {
  const int M = 5, K = 7, N = 4;
  std::vector<float> A(M * K), B(K * N), C(M * N), U(M * N);
  fill_random(A);
  fill_random(B);
  fill_random(U);
  matmul_forward(A.data(), B.data(), C.data(), M, K, N);

  // forward vs a naive reference
  double fwd_err = 0.0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double ref = 0.0;
      for (int k = 0; k < K; ++k) ref += A[m * K + k] * B[k * N + n];
      fwd_err = std::max(fwd_err, std::fabs(ref - C[m * N + n]));
    }
  report("matmul forward", fwd_err < 1e-4, fwd_err);

  // loss = sum(C * U)  =>  dC == U
  std::vector<float> dA(M * K), dB(K * N);
  matmul_backward(A.data(), B.data(), U.data(), dA.data(), dB.data(), M, K, N);
  auto loss = [&]() {
    matmul_forward(A.data(), B.data(), C.data(), M, K, N);
    double s = 0.0;
    for (int i = 0; i < M * N; ++i) s += static_cast<double>(C[i]) * U[i];
    return s;
  };
  const double eA = grad_check(A.data(), M * K, dA.data(), loss);
  const double eB = grad_check(B.data(), K * N, dB.data(), loss);
  report("matmul backward dA", eA < 2e-2, eA);
  report("matmul backward dB", eB < 2e-2, eB);
}

// --------------------------------------------------------------------------
// layernorm
// --------------------------------------------------------------------------
static void test_layernorm() {
  const int N = 6, D = 8;
  const float eps = 1e-5f;
  std::vector<float> x(N * D), gamma(D), beta(D), y(N * D);
  std::vector<float> mean(N), rstd(N), U(N * D);
  fill_random(x, 2.0f);
  fill_random(gamma, 1.0f);
  fill_random(beta, 1.0f);
  fill_random(U);

  // forward sanity: with gamma=1, beta=0 each row is zero-mean, unit-variance.
  std::vector<float> g1(D, 1.0f), b0(D, 0.0f);
  layernorm_forward(x.data(), g1.data(), b0.data(), y.data(), mean.data(),
                    rstd.data(), N, D, eps);
  double worst_stat = 0.0;
  for (int r = 0; r < N; ++r) {
    double m = 0.0, v = 0.0;
    for (int d = 0; d < D; ++d) m += y[r * D + d];
    m /= D;
    for (int d = 0; d < D; ++d) v += (y[r * D + d] - m) * (y[r * D + d] - m);
    v /= D;
    worst_stat = std::max({worst_stat, std::fabs(m), std::fabs(v - 1.0)});
  }
  report("layernorm forward (mean~0,var~1)", worst_stat < 1e-3, worst_stat);

  // backward grad check with real gamma/beta
  layernorm_forward(x.data(), gamma.data(), beta.data(), y.data(), mean.data(),
                    rstd.data(), N, D, eps);
  std::vector<float> dx(N * D), dgamma(D), dbeta(D);
  layernorm_backward(x.data(), gamma.data(), mean.data(), rstd.data(), U.data(),
                     dx.data(), dgamma.data(), dbeta.data(), N, D);
  auto loss = [&]() {
    layernorm_forward(x.data(), gamma.data(), beta.data(), y.data(),
                      mean.data(), rstd.data(), N, D, eps);
    double s = 0.0;
    for (int i = 0; i < N * D; ++i) s += static_cast<double>(y[i]) * U[i];
    return s;
  };
  const double ex = grad_check(x.data(), N * D, dx.data(), loss);
  const double eg = grad_check(gamma.data(), D, dgamma.data(), loss);
  const double eb = grad_check(beta.data(), D, dbeta.data(), loss);
  report("layernorm backward dx", ex < 2e-2, ex);
  report("layernorm backward dgamma", eg < 2e-2, eg);
  report("layernorm backward dbeta", eb < 2e-2, eb);
}

// --------------------------------------------------------------------------
// attention
// --------------------------------------------------------------------------
static void test_attention() {
  const int B = 2, T = 4, C = 8, H = 2;
  const int BT = B * T, WN = C * C;
  std::vector<float> x(BT * C), Wq(WN), Wk(WN), Wv(WN), Wo(WN);
  std::vector<float> bq(C), bk(C), bv(C), bo(C);
  std::vector<float> out(BT * C), q(BT * C), k(BT * C), v(BT * C);
  std::vector<float> attn(B * H * T * T), U(BT * C);
  fill_random(x);
  fill_random(Wq, 0.5f);
  fill_random(Wk, 0.5f);
  fill_random(Wv, 0.5f);
  fill_random(Wo, 0.5f);
  fill_random(bq, 0.2f);
  fill_random(bk, 0.2f);
  fill_random(bv, 0.2f);
  fill_random(bo, 0.2f);
  fill_random(U);

  auto run_forward = [&]() {
    attention_forward(x.data(), Wq.data(), bq.data(), Wk.data(), bk.data(),
                      Wv.data(), bv.data(), Wo.data(), bo.data(), out.data(),
                      q.data(), k.data(), v.data(), attn.data(), B, T, C, H);
  };
  run_forward();

  // forward sanity: each attn row sums to 1 over [0,t1]; future is masked to 0.
  double worst_sm = 0.0;
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int t1 = 0; t1 < T; ++t1) {
        const float* row = attn.data() + (((b * H + h) * T) + t1) * T;
        double sum = 0.0;
        for (int t2 = 0; t2 <= t1; ++t2) sum += row[t2];
        worst_sm = std::max(worst_sm, std::fabs(sum - 1.0));
        for (int t2 = t1 + 1; t2 < T; ++t2)
          worst_sm = std::max(worst_sm, static_cast<double>(std::fabs(row[t2])));
      }
  report("attention forward (causal softmax)", worst_sm < 1e-5, worst_sm);

  std::vector<float> dx(BT * C), dWq(WN), dWk(WN), dWv(WN), dWo(WN);
  std::vector<float> dbq(C), dbk(C), dbv(C), dbo(C);
  attention_backward(x.data(), Wq.data(), Wk.data(), Wv.data(), Wo.data(),
                     q.data(), k.data(), v.data(), attn.data(), U.data(),
                     dx.data(), dWq.data(), dbq.data(), dWk.data(), dbk.data(),
                     dWv.data(), dbv.data(), dWo.data(), dbo.data(), B, T, C, H);
  auto loss = [&]() {
    run_forward();
    double s = 0.0;
    for (int i = 0; i < BT * C; ++i) s += static_cast<double>(out[i]) * U[i];
    return s;
  };
  const double ex = grad_check(x.data(), BT * C, dx.data(), loss);
  const double eq = grad_check(Wq.data(), WN, dWq.data(), loss);
  const double ek = grad_check(Wk.data(), WN, dWk.data(), loss);
  const double ev = grad_check(Wv.data(), WN, dWv.data(), loss);
  const double eo = grad_check(Wo.data(), WN, dWo.data(), loss);
  const double ebq = grad_check(bq.data(), C, dbq.data(), loss);
  const double ebv = grad_check(bv.data(), C, dbv.data(), loss);
  const double ebo = grad_check(bo.data(), C, dbo.data(), loss);
  report("attention backward dx", ex < 2e-2, ex);
  report("attention backward dWq", eq < 2e-2, eq);
  report("attention backward dWk", ek < 2e-2, ek);
  report("attention backward dWv", ev < 2e-2, ev);
  report("attention backward dWo", eo < 2e-2, eo);
  const double eb = std::max({ebq, ebv, ebo});
  report("attention backward dbias", eb < 2e-2, eb);
}

// --------------------------------------------------------------------------
// adamw
// --------------------------------------------------------------------------
static void test_adamw() {
  // One step against a hand-computed value (see adamw.cpp formula).
  // p=1, g=0.5, t=1, lr=0.1, b1=0.9, b2=0.95, eps=1e-8, wd=0.1  =>  p' = 0.89
  float p = 1.0f, g = 0.5f, m = 0.0f, v = 0.0f;
  adamw_step(&p, &g, &m, &v, 1, 1, 0.1f, 0.9f, 0.95f, 1e-8f, 0.1f);
  report("adamw one step (vs hand-computed)", std::fabs(p - 0.89f) < 1e-5,
         std::fabs(p - 0.89f));

  // Convergence: minimize f(w) = sum(w^2), grad = 2w. Should drive w -> ~0.
  const int n = 32;
  std::vector<float> w(n), gr(n), mm(n, 0.0f), vv(n, 0.0f);
  fill_random(w, 1.0f);
  double start_norm = grad_l2_norm(w.data(), n);
  for (int step = 1; step <= 400; ++step) {
    for (int i = 0; i < n; ++i) gr[i] = 2.0f * w[i];
    adamw_step(w.data(), gr.data(), mm.data(), vv.data(), n, step, 0.05f, 0.9f,
               0.95f, 1e-8f, 0.01f);
  }
  double end_norm = grad_l2_norm(w.data(), n);
  report("adamw convergence (||w|| shrinks)", end_norm < start_norm * 1e-2,
         end_norm);

  // grad_l2_norm: [3,4] -> 5;  grad_scale halves it.
  std::vector<float> gn = {3.0f, 4.0f};
  const double norm = grad_l2_norm(gn.data(), 2);
  report("grad_l2_norm (3,4)->5", std::fabs(norm - 5.0) < 1e-5,
         std::fabs(norm - 5.0));
  grad_scale(gn.data(), 2, 0.5f);
  const double scaled = grad_l2_norm(gn.data(), 2);
  report("grad_scale halves the norm", std::fabs(scaled - 2.5) < 1e-5,
         std::fabs(scaled - 2.5));
}

int main() {
  test_matmul();
  test_layernorm();
  test_attention();
  test_adamw();
  std::printf("\n%s\n", g_failed == 0 ? "all WASM kernel tests passed"
                                      : "SOME WASM KERNEL TESTS FAILED");
  return g_failed == 0 ? 0 : 1;
}
