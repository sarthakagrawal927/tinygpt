// attention.cpp — causal multi-head self-attention, forward + backward (Phase 4).
//
// Per block (batch B, seq T, d_model C, heads H, head_dim = C / H):
//   q = x @ Wq + bq;  k = x @ Wk + bk;  v = x @ Wv + bv
//   scores = q @ k^T / sqrt(head_dim)
//   scores = causal_mask(scores)          // position t1 sees only t2 <= t1
//   attn   = softmax(scores)              // over the unmasked t2
//   ctx    = attn @ v
//   out    = ctx @ Wo + bo
//
// The backward propagates through the softmax, the masked scores, and all four
// projections. Forward caches q, k, v and the post-softmax weights `attn`; ctx
// is small and is recomputed in the backward rather than carried.
//
// Guide: docs/model_guide.md ("Causal self-attention")

#include "kernels.h"

#include <cmath>
#include <vector>

// y[r,:] = x[r,:] @ W + b      W:[in,out]
static void linear_forward(const float* x, const float* W, const float* b,
                           float* y, int rows, int in, int out) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<long>(r) * in;
    float* yr = y + static_cast<long>(r) * out;
    for (int j = 0; j < out; ++j) yr[j] = b[j];
    for (int i = 0; i < in; ++i) {
      const float xi = xr[i];
      const float* Wi = W + static_cast<long>(i) * out;
      for (int j = 0; j < out; ++j) yr[j] += xi * Wi[j];
    }
  }
}

// dx ACCUMULATES (caller zeroes it once, then sums q/k/v contributions);
// dW and db are overwritten. dx += dy @ W^T,  dW = x^T @ dy,  db = colsum(dy).
static void linear_backward(const float* x, const float* W, const float* dy,
                            float* dx, float* dW, float* db, int rows, int in,
                            int out) {
  for (int i = 0; i < in * out; ++i) dW[i] = 0.0f;
  for (int j = 0; j < out; ++j) db[j] = 0.0f;
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<long>(r) * in;
    const float* dyr = dy + static_cast<long>(r) * out;
    float* dxr = dx + static_cast<long>(r) * in;
    for (int j = 0; j < out; ++j) db[j] += dyr[j];
    for (int i = 0; i < in; ++i) {
      const float* Wi = W + static_cast<long>(i) * out;
      float* dWi = dW + static_cast<long>(i) * out;
      const float xi = xr[i];
      float acc = 0.0f;
      for (int j = 0; j < out; ++j) {
        acc += dyr[j] * Wi[j];
        dWi[j] += xi * dyr[j];
      }
      dxr[i] += acc;
    }
  }
}

// ctx[b,t1, h-slice] = sum_{t2<=t1} attn[b,h,t1,t2] * v[b,t2, h-slice]
static void compute_ctx(const float* attn, const float* v, float* ctx, int B,
                        int T, int C, int H) {
  const int hd = C / H;
  for (int i = 0; i < B * T * C; ++i) ctx[i] = 0.0f;
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const int off = h * hd;
      for (int t1 = 0; t1 < T; ++t1) {
        const float* arow = attn + ((static_cast<long>((b * H + h) * T) + t1) * T);
        float* c1 = ctx + (static_cast<long>(b * T + t1) * C) + off;
        for (int t2 = 0; t2 <= t1; ++t2) {
          const float a = arow[t2];
          const float* v2 = v + (static_cast<long>(b * T + t2) * C) + off;
          for (int d = 0; d < hd; ++d) c1[d] += a * v2[d];
        }
      }
    }
}

WASM_EXPORT void attention_forward(
    const float* x, const float* Wq, const float* bq, const float* Wk,
    const float* bk, const float* Wv, const float* bv, const float* Wo,
    const float* bo, float* out, float* q, float* k, float* v, float* attn,
    int B, int T, int C, int H) {
  const int hd = C / H;
  const int BT = B * T;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  linear_forward(x, Wq, bq, q, BT, C, C);
  linear_forward(x, Wk, bk, k, BT, C, C);
  linear_forward(x, Wv, bv, v, BT, C, C);

  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const int off = h * hd;
      for (int t1 = 0; t1 < T; ++t1) {
        const float* q1 = q + (static_cast<long>(b * T + t1) * C) + off;
        float* arow = attn + ((static_cast<long>((b * H + h) * T) + t1) * T);

        // scores over the unmasked range [0, t1], with max-subtraction.
        std::vector<float> sc(t1 + 1);
        float maxv = -1e30f;
        for (int t2 = 0; t2 <= t1; ++t2) {
          const float* k2 = k + (static_cast<long>(b * T + t2) * C) + off;
          float s = 0.0f;
          for (int d = 0; d < hd; ++d) s += q1[d] * k2[d];
          s *= scale;
          sc[t2] = s;
          if (s > maxv) maxv = s;
        }
        float sum = 0.0f;
        for (int t2 = 0; t2 <= t1; ++t2) {
          sc[t2] = std::exp(sc[t2] - maxv);
          sum += sc[t2];
        }
        const float inv = 1.0f / sum;
        for (int t2 = 0; t2 < T; ++t2) arow[t2] = 0.0f;  // masked future = 0
        for (int t2 = 0; t2 <= t1; ++t2) arow[t2] = sc[t2] * inv;
      }
    }

  std::vector<float> ctx(static_cast<long>(BT) * C);
  compute_ctx(attn, v, ctx.data(), B, T, C, H);
  linear_forward(ctx.data(), Wo, bo, out, BT, C, C);
}

WASM_EXPORT void attention_backward(
    const float* x, const float* Wq, const float* Wk, const float* Wv,
    const float* Wo, const float* q, const float* k, const float* v,
    const float* attn, const float* dout, float* dx, float* dWq, float* dbq,
    float* dWk, float* dbk, float* dWv, float* dbv, float* dWo, float* dbo,
    int B, int T, int C, int H) {
  const int hd = C / H;
  const int BT = B * T;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  // out = ctx @ Wo + bo  ->  dctx, dWo, dbo
  std::vector<float> ctx(static_cast<long>(BT) * C);
  compute_ctx(attn, v, ctx.data(), B, T, C, H);
  std::vector<float> dctx(static_cast<long>(BT) * C, 0.0f);
  linear_backward(ctx.data(), Wo, dout, dctx.data(), dWo, dbo, BT, C, C);

  // Propagate dctx back through attn @ v and the softmax to dq, dk, dv.
  std::vector<float> dq(static_cast<long>(BT) * C, 0.0f);
  std::vector<float> dk(static_cast<long>(BT) * C, 0.0f);
  std::vector<float> dv(static_cast<long>(BT) * C, 0.0f);

  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const int off = h * hd;
      for (int t1 = 0; t1 < T; ++t1) {
        const float* dc1 = dctx.data() + (static_cast<long>(b * T + t1) * C) + off;
        const float* arow = attn + ((static_cast<long>((b * H + h) * T) + t1) * T);

        // dattn[t2] = sum_d dctx[t1,d] * v[t2,d];  dv accumulates attn^T @ dctx.
        std::vector<float> dattn(t1 + 1);
        for (int t2 = 0; t2 <= t1; ++t2) {
          const float* v2 = v + (static_cast<long>(b * T + t2) * C) + off;
          float* dv2 = dv.data() + (static_cast<long>(b * T + t2) * C) + off;
          float da = 0.0f;
          const float a = arow[t2];
          for (int d = 0; d < hd; ++d) {
            da += dc1[d] * v2[d];
            dv2[d] += a * dc1[d];
          }
          dattn[t2] = da;
        }

        // softmax backward: dscores = p * (dattn - sum(dattn * p))
        float dot = 0.0f;
        for (int t2 = 0; t2 <= t1; ++t2) dot += dattn[t2] * arow[t2];

        // scores = scale * q . k  ->  dq, dk
        const float* q1 = q + (static_cast<long>(b * T + t1) * C) + off;
        float* dq1 = dq.data() + (static_cast<long>(b * T + t1) * C) + off;
        for (int t2 = 0; t2 <= t1; ++t2) {
          const float ds = arow[t2] * (dattn[t2] - dot) * scale;
          const float* k2 = k + (static_cast<long>(b * T + t2) * C) + off;
          float* dk2 = dk.data() + (static_cast<long>(b * T + t2) * C) + off;
          for (int d = 0; d < hd; ++d) {
            dq1[d] += ds * k2[d];
            dk2[d] += ds * q1[d];
          }
        }
      }
    }

  // dq, dk, dv -> dx (accumulated across the three) and the projection grads.
  for (int i = 0; i < BT * C; ++i) dx[i] = 0.0f;
  linear_backward(x, Wq, dq.data(), dx, dWq, dbq, BT, C, C);
  linear_backward(x, Wk, dk.data(), dx, dWk, dbk, BT, C, C);
  linear_backward(x, Wv, dv.data(), dx, dWv, dbv, BT, C, C);
}
