// kernels.h — declarations for the WASM CPU backend (Phase 4).
//
// One header for the five hand-written kernels: tensor, matmul, layernorm,
// attention, adamw. No general autograd — each op carries its own forward and
// backward. Functions use a C ABI so Emscripten can export them and so the
// native test harness (tests/test_wasm_kernels.cpp) can link them directly.
//
// Weight convention (math, not PyTorch): a Linear is y = x @ W + b with
// W shaped [in, out]; x rows are contiguous. This matches docs/model_guide.md.
//
// Guide: docs/browser_notes.md ("WASM backend"), wasm/README.md
#ifndef TINYGPT_WASM_KERNELS_H
#define TINYGPT_WASM_KERNELS_H

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define WASM_EXPORT extern "C" EMSCRIPTEN_KEEPALIVE
#else
#define WASM_EXPORT extern "C"
#endif

// --- tensor.cpp ----------------------------------------------------------
// A flat float32 buffer plus a shape. Up to 4 dims is plenty for TinyGPT.
struct Tensor {
  float* data;
  int shape[4];
  int ndim;
  int size;
};

Tensor tensor_alloc(int ndim, const int* shape);
void tensor_free(Tensor* t);
void tensor_zero(float* data, int n);
void tensor_fill(float* data, int n, float v);

// Raw allocation entry points — the browser glue calls these to hand buffers
// across the JS/WASM boundary.
WASM_EXPORT float* tg_malloc(int n_floats);
WASM_EXPORT void tg_free(float* p);

// --- matmul.cpp ----------------------------------------------------------
// C = A @ B          A:[M,K]  B:[K,N]  C:[M,N]
WASM_EXPORT void matmul_forward(const float* A, const float* B, float* C,
                                int M, int K, int N);
// dA = dC @ B^T      dB = A^T @ dC      (dA, dB are overwritten, not accumulated)
WASM_EXPORT void matmul_backward(const float* A, const float* B, const float* dC,
                                 float* dA, float* dB, int M, int K, int N);

// --- layernorm.cpp -------------------------------------------------------
// Normalizes over the last dimension D, across N independent rows.
// y = gamma * (x - mean) / sqrt(var + eps) + beta
// `mean` and `rstd` (1/sqrt(var+eps)) are cached per row for the backward pass.
WASM_EXPORT void layernorm_forward(const float* x, const float* gamma,
                                   const float* beta, float* y, float* mean,
                                   float* rstd, int N, int D, float eps);
// dx per element; dgamma/dbeta accumulated across all N rows.
WASM_EXPORT void layernorm_backward(const float* x, const float* gamma,
                                    const float* mean, const float* rstd,
                                    const float* dy, float* dx, float* dgamma,
                                    float* dbeta, int N, int D);

// --- attention.cpp -------------------------------------------------------
// Causal multi-head self-attention over x:[B,T,C], heads H, head_dim = C/H.
// Forward caches q,k,v:[B,T,C] and the post-softmax weights attn:[B,H,T,T]
// for the backward pass.
WASM_EXPORT void attention_forward(
    const float* x, const float* Wq, const float* bq, const float* Wk,
    const float* bk, const float* Wv, const float* bv, const float* Wo,
    const float* bo, float* out, float* q, float* k, float* v, float* attn,
    int B, int T, int C, int H);
WASM_EXPORT void attention_backward(
    const float* x, const float* Wq, const float* Wk, const float* Wv,
    const float* Wo, const float* q, const float* k, const float* v,
    const float* attn, const float* dout, float* dx, float* dWq, float* dbq,
    float* dWk, float* dbk, float* dWv, float* dbv, float* dWo, float* dbo,
    int B, int T, int C, int H);

// --- adamw.cpp -----------------------------------------------------------
// In-place AdamW step over one parameter buffer. `t` is the 1-based step count
// (used for bias correction). Clip gradients BEFORE calling this.
WASM_EXPORT void adamw_step(float* p, const float* g, float* m, float* v, int n,
                            int t, float lr, float beta1, float beta2, float eps,
                            float weight_decay);
// L2 norm of a gradient buffer — building block for global grad clipping.
WASM_EXPORT float grad_l2_norm(const float* g, int n);
// Scale a gradient buffer in place (apply after computing the clip factor).
WASM_EXPORT void grad_scale(float* g, int n, float factor);

#endif  // TINYGPT_WASM_KERNELS_H
