// layernorm.cpp — LayerNorm, forward + backward (Phase 4).
//
// Pre-LayerNorm runs before attention and before the MLP in every block:
//   y = gamma * (x - mean) / sqrt(var + eps) + beta
// Normalization is over the last dimension D, independently for each of the
// N rows. mean and rstd = 1/sqrt(var+eps) are cached for the backward pass.
//
// Guide: docs/model_guide.md ("Transformer block"), docs/browser_notes.md

#include "kernels.h"

#include <cmath>

WASM_EXPORT void layernorm_forward(const float* x, const float* gamma,
                                   const float* beta, float* y, float* mean,
                                   float* rstd, int N, int D, float eps) {
  for (int row = 0; row < N; ++row) {
    const float* xr = x + static_cast<long>(row) * D;
    float* yr = y + static_cast<long>(row) * D;

    float mu = 0.0f;
    for (int d = 0; d < D; ++d) mu += xr[d];
    mu /= D;

    float var = 0.0f;
    for (int d = 0; d < D; ++d) {
      const float diff = xr[d] - mu;
      var += diff * diff;
    }
    var /= D;

    const float rs = 1.0f / std::sqrt(var + eps);
    mean[row] = mu;
    rstd[row] = rs;
    for (int d = 0; d < D; ++d)
      yr[d] = gamma[d] * ((xr[d] - mu) * rs) + beta[d];
  }
}

// With xhat = (x - mean) * rstd:
//   dgamma += dy * xhat            dbeta += dy
//   dxhat  = dy * gamma
//   dx     = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
// dgamma and dbeta accumulate across all N rows; dx is written per element.
WASM_EXPORT void layernorm_backward(const float* x, const float* gamma,
                                    const float* mean, const float* rstd,
                                    const float* dy, float* dx, float* dgamma,
                                    float* dbeta, int N, int D) {
  for (int d = 0; d < D; ++d) {
    dgamma[d] = 0.0f;
    dbeta[d] = 0.0f;
  }
  for (int row = 0; row < N; ++row) {
    const float* xr = x + static_cast<long>(row) * D;
    const float* dyr = dy + static_cast<long>(row) * D;
    float* dxr = dx + static_cast<long>(row) * D;
    const float mu = mean[row];
    const float rs = rstd[row];

    // Two row-wise reductions needed by the dx formula.
    float mean_dxhat = 0.0f;       // mean of dxhat
    float mean_dxhat_xhat = 0.0f;  // mean of dxhat * xhat
    for (int d = 0; d < D; ++d) {
      const float xhat = (xr[d] - mu) * rs;
      const float dxhat = dyr[d] * gamma[d];
      mean_dxhat += dxhat;
      mean_dxhat_xhat += dxhat * xhat;
      dgamma[d] += dyr[d] * xhat;
      dbeta[d] += dyr[d];
    }
    mean_dxhat /= D;
    mean_dxhat_xhat /= D;

    for (int d = 0; d < D; ++d) {
      const float xhat = (xr[d] - mu) * rs;
      const float dxhat = dyr[d] * gamma[d];
      dxr[d] = rs * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat);
    }
  }
}
