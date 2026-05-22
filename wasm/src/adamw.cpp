// adamw.cpp — AdamW optimizer step (Phase 4).
//
// Per parameter, given gradient g (config: configs/training.json):
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g * g
//   m_hat = m / (1 - beta1^t)
//   v_hat = v / (1 - beta2^t)
//   p = p - lr * ( m_hat / (sqrt(v_hat) + eps) + weight_decay * p )
//
// "AdamW" = decoupled weight decay: the wd term acts directly on p, not through
// the moments. Apply gradient clipping (grad_l2_norm + grad_scale) BEFORE the
// step. m and v are persisted in checkpoints — that is what makes resume exact.
//
// Guide: docs/model_guide.md ("Training config")

#include "kernels.h"

#include <cmath>

WASM_EXPORT void adamw_step(float* p, const float* g, float* m, float* v, int n,
                            int t, float lr, float beta1, float beta2, float eps,
                            float weight_decay) {
  // Bias-correction denominators depend on the step count, not the parameter.
  const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t));
  const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t));
  for (int i = 0; i < n; ++i) {
    const float gi = g[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
    v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;
    const float m_hat = m[i] / bc1;
    const float v_hat = v[i] / bc2;
    p[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * p[i]);
  }
}

// L2 norm of a gradient buffer. Global clipping sums the squares across every
// parameter buffer, takes the sqrt, then scales each if the norm exceeds the
// threshold.
WASM_EXPORT float grad_l2_norm(const float* g, int n) {
  float sum_sq = 0.0f;
  for (int i = 0; i < n; ++i) sum_sq += g[i] * g[i];
  return std::sqrt(sum_sq);
}

WASM_EXPORT void grad_scale(float* g, int n, float factor) {
  for (int i = 0; i < n; ++i) g[i] *= factor;
}
