// tensor.cpp — flat float32 tensor type + helpers (Phase 4).
//
// The minimal building block for the WASM CPU backend: a contiguous float32
// buffer plus a shape. No general autograd — each op (matmul, layernorm,
// attention, adamw) carries its own forward and backward.
//
// Guide: docs/browser_notes.md ("WASM backend")

#include "kernels.h"

#include <cstdlib>
#include <cstring>

// Product of a shape — the element count of a tensor.
static int shape_size(int ndim, const int* shape) {
  int n = 1;
  for (int i = 0; i < ndim; ++i) n *= shape[i];
  return n;
}

Tensor tensor_alloc(int ndim, const int* shape) {
  Tensor t;
  t.ndim = ndim;
  t.size = shape_size(ndim, shape);
  for (int i = 0; i < 4; ++i) t.shape[i] = (i < ndim) ? shape[i] : 1;
  t.data = static_cast<float*>(std::calloc(t.size, sizeof(float)));
  return t;
}

void tensor_free(Tensor* t) {
  if (t && t->data) {
    std::free(t->data);
    t->data = nullptr;
    t->size = 0;
  }
}

void tensor_zero(float* data, int n) {
  std::memset(data, 0, static_cast<size_t>(n) * sizeof(float));
}

void tensor_fill(float* data, int n, float v) {
  for (int i = 0; i < n; ++i) data[i] = v;
}

// JS/WASM boundary: the browser glue allocates buffers through these so the
// WASM heap and the JS views agree on ownership.
WASM_EXPORT float* tg_malloc(int n_floats) {
  return static_cast<float*>(std::malloc(static_cast<size_t>(n_floats) * sizeof(float)));
}

WASM_EXPORT void tg_free(float* p) { std::free(p); }
