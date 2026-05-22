// model.cpp — the WASM-side TinyGPT (Phase 4).
//
// A full byte-level GPT assembled from the kernels in kernels.h. No autograd:
// every op below (embedding, GELU, MLP, the tied head, cross-entropy) carries a
// hand-written backward, and the four heavy kernels supply theirs. The browser
// Worker drives everything through the C-ABI in model.h.
//
// Architecture mirrors python_ref/model.py: pre-LayerNorm blocks, GELU MLP,
// tied input/output embeddings. Correctness is checked by tests/test_wasm_model.cpp
// (the tiny-overfit gate) before this is ever compiled to WASM.

#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr float kLnEps = 1e-5f;
constexpr float kGeluC = 0.7071067811865476f;       // 1 / sqrt(2)
constexpr float kInvSqrt2Pi = 0.3989422804014327f;  // 1 / sqrt(2*pi)

// --- small element-wise helpers -----------------------------------------
void add_bias(float* y, const float* b, int rows, int d) {
  for (int r = 0; r < rows; ++r)
    for (int j = 0; j < d; ++j) y[r * d + j] += b[j];
}

void bias_grad(const float* dy, float* db, int rows, int d) {
  for (int j = 0; j < d; ++j) db[j] = 0.0f;
  for (int r = 0; r < rows; ++r)
    for (int j = 0; j < d; ++j) db[j] += dy[r * d + j];
}

// Exact (erf-based) GELU, matching torch.nn.functional.gelu's default.
void gelu_forward(const float* x, float* y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = 0.5f * x[i] * (1.0f + std::erf(x[i] * kGeluC));
}

void gelu_backward(const float* x, const float* dy, float* dx, int n) {
  for (int i = 0; i < n; ++i) {
    const float xi = x[i];
    const float cdf = 0.5f * (1.0f + std::erf(xi * kGeluC));
    const float pdf = kInvSqrt2Pi * std::exp(-0.5f * xi * xi);
    dx[i] = dy[i] * (cdf + xi * pdf);
  }
}

// --- parameters ----------------------------------------------------------
// One trainable tensor: weights, gradient, and the two AdamW moments.
struct Param {
  std::vector<float> w, g, m, v;
  bool decay = false;  // weight decay applies only to matrices/embeddings
  void init(int n, bool dec) {
    w.assign(n, 0.0f);
    g.assign(n, 0.0f);
    m.assign(n, 0.0f);
    v.assign(n, 0.0f);
    decay = dec;
  }
  int size() const { return static_cast<int>(w.size()); }
};

struct Layer {
  Param ln1_g, ln1_b, Wq, bq, Wk, bk, Wv, bv, Wo, bo;
  Param ln2_g, ln2_b, fc_in_W, fc_in_b, fc_out_W, fc_out_b;
};

// Cached forward activations for one block (needed by its backward).
struct LayerAct {
  std::vector<float> ln1o, mean1, rstd1;
  std::vector<float> q, k, v, attn, ao;
  std::vector<float> r1;
  std::vector<float> ln2o, mean2, rstd2;
  std::vector<float> hpre, hact, mo;
  std::vector<float> r2;
};

struct Acts {
  int B = 0, T = 0;
  std::vector<float> x0, lnf_o, meanf, rstdf, logits;
  std::vector<LayerAct> layer;
};

struct Model {
  int V, ctx, L, H, C, M, hd;
  long step = 0;
  std::mt19937 rng;

  Param tok_emb, pos_emb, ln_f_g, ln_f_b;
  std::vector<Layer> layers;
  std::vector<Param*> params;  // every Param, for AdamW + grad clipping

  std::vector<unsigned char> data;
  int n_train = 0;

  Acts acts;            // forward cache, reused across steps
  std::vector<float> et;  // transposed tok_emb [C,V] for the tied head
};

void collect_params(Model& m) {
  m.params = {&m.tok_emb, &m.pos_emb, &m.ln_f_g, &m.ln_f_b};
  for (Layer& l : m.layers) {
    Param* ps[] = {&l.ln1_g,   &l.ln1_b, &l.Wq,      &l.bq,
                   &l.Wk,      &l.bk,    &l.Wv,      &l.bv,
                   &l.Wo,      &l.bo,    &l.ln2_g,   &l.ln2_b,
                   &l.fc_in_W, &l.fc_in_b, &l.fc_out_W, &l.fc_out_b};
    for (Param* p : ps) m.params.push_back(p);
  }
}

void init_weights(Model& m) {
  std::normal_distribution<float> n02(0.0f, 0.02f);
  // GPT-2 scaled init for residual-path output projections.
  std::normal_distribution<float> n_scaled(
      0.0f, 0.02f / std::sqrt(2.0f * static_cast<float>(m.L)));
  auto randn = [&](Param& p) { for (float& x : p.w) x = n02(m.rng); };
  auto randn_scaled = [&](Param& p) { for (float& x : p.w) x = n_scaled(m.rng); };
  auto ones = [&](Param& p) { std::fill(p.w.begin(), p.w.end(), 1.0f); };

  m.tok_emb.init(m.V * m.C, true);
  randn(m.tok_emb);
  m.pos_emb.init(m.ctx * m.C, true);
  randn(m.pos_emb);
  m.ln_f_g.init(m.C, false);
  ones(m.ln_f_g);
  m.ln_f_b.init(m.C, false);

  for (Layer& l : m.layers) {
    l.ln1_g.init(m.C, false);  ones(l.ln1_g);
    l.ln1_b.init(m.C, false);
    l.ln2_g.init(m.C, false);  ones(l.ln2_g);
    l.ln2_b.init(m.C, false);
    l.Wq.init(m.C * m.C, true);  randn(l.Wq);  l.bq.init(m.C, false);
    l.Wk.init(m.C * m.C, true);  randn(l.Wk);  l.bk.init(m.C, false);
    l.Wv.init(m.C * m.C, true);  randn(l.Wv);  l.bv.init(m.C, false);
    l.Wo.init(m.C * m.C, true);  randn_scaled(l.Wo);  l.bo.init(m.C, false);
    l.fc_in_W.init(m.C * m.M, true);   randn(l.fc_in_W);
    l.fc_in_b.init(m.M, false);
    l.fc_out_W.init(m.M * m.C, true);  randn_scaled(l.fc_out_W);
    l.fc_out_b.init(m.C, false);
  }
}

void resize_acts(Model& m, int B, int T) {
  Acts& a = m.acts;
  if (a.B == B && a.T == T) return;
  a.B = B;
  a.T = T;
  const int N = B * T, C = m.C, NC = N * C, NM = N * m.M;
  a.x0.assign(NC, 0.0f);
  a.lnf_o.assign(NC, 0.0f);
  a.meanf.assign(N, 0.0f);
  a.rstdf.assign(N, 0.0f);
  a.logits.assign(N * m.V, 0.0f);
  a.layer.resize(m.L);
  for (LayerAct& la : a.layer) {
    la.ln1o.assign(NC, 0.0f);  la.mean1.assign(N, 0.0f);  la.rstd1.assign(N, 0.0f);
    la.q.assign(NC, 0.0f);     la.k.assign(NC, 0.0f);     la.v.assign(NC, 0.0f);
    la.attn.assign(static_cast<long>(B) * m.H * T * T, 0.0f);
    la.ao.assign(NC, 0.0f);    la.r1.assign(NC, 0.0f);
    la.ln2o.assign(NC, 0.0f);  la.mean2.assign(N, 0.0f);  la.rstd2.assign(N, 0.0f);
    la.hpre.assign(NM, 0.0f);  la.hact.assign(NM, 0.0f);  la.mo.assign(NC, 0.0f);
    la.r2.assign(NC, 0.0f);
  }
}

// Forward pass. ids has B*T byte tokens; logits land in m.acts.logits.
void forward(Model& m, const int* ids, int B, int T) {
  resize_acts(m, B, T);
  Acts& a = m.acts;
  const int N = B * T, C = m.C;

  // token + position embedding
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T; ++t) {
      const int id = ids[b * T + t];
      float* dst = a.x0.data() + (b * T + t) * C;
      const float* te = m.tok_emb.w.data() + id * C;
      const float* pe = m.pos_emb.w.data() + t * C;
      for (int c = 0; c < C; ++c) dst[c] = te[c] + pe[c];
    }

  const float* cur = a.x0.data();
  for (int l = 0; l < m.L; ++l) {
    Layer& w = m.layers[l];
    LayerAct& la = a.layer[l];

    layernorm_forward(cur, w.ln1_g.w.data(), w.ln1_b.w.data(), la.ln1o.data(),
                      la.mean1.data(), la.rstd1.data(), N, C, kLnEps);
    attention_forward(la.ln1o.data(), w.Wq.w.data(), w.bq.w.data(),
                      w.Wk.w.data(), w.bk.w.data(), w.Wv.w.data(), w.bv.w.data(),
                      w.Wo.w.data(), w.bo.w.data(), la.ao.data(), la.q.data(),
                      la.k.data(), la.v.data(), la.attn.data(), B, T, C, m.H);
    for (int i = 0; i < N * C; ++i) la.r1[i] = cur[i] + la.ao[i];

    layernorm_forward(la.r1.data(), w.ln2_g.w.data(), w.ln2_b.w.data(),
                      la.ln2o.data(), la.mean2.data(), la.rstd2.data(), N, C,
                      kLnEps);
    matmul_forward(la.ln2o.data(), w.fc_in_W.w.data(), la.hpre.data(), N, C, m.M);
    add_bias(la.hpre.data(), w.fc_in_b.w.data(), N, m.M);
    gelu_forward(la.hpre.data(), la.hact.data(), N * m.M);
    matmul_forward(la.hact.data(), w.fc_out_W.w.data(), la.mo.data(), N, m.M, C);
    add_bias(la.mo.data(), w.fc_out_b.w.data(), N, C);
    for (int i = 0; i < N * C; ++i) la.r2[i] = la.r1[i] + la.mo[i];

    cur = la.r2.data();
  }

  layernorm_forward(cur, m.ln_f_g.w.data(), m.ln_f_b.w.data(), a.lnf_o.data(),
                    a.meanf.data(), a.rstdf.data(), N, C, kLnEps);

  // tied head: logits = lnf_o @ tok_emb^T  (build the transpose et:[C,V])
  m.et.resize(static_cast<long>(C) * m.V);
  for (int vi = 0; vi < m.V; ++vi)
    for (int c = 0; c < C; ++c) m.et[c * m.V + vi] = m.tok_emb.w[vi * C + c];
  matmul_forward(a.lnf_o.data(), m.et.data(), a.logits.data(), N, C, m.V);
}

// Cross-entropy over the cached logits. Returns mean loss and writes
// dlogits = (softmax - onehot) / N.
float cross_entropy(const Model& m, const int* targets, float* dlogits) {
  const Acts& a = m.acts;
  const int N = a.B * a.T, V = m.V;
  const float* logits = a.logits.data();
  double total = 0.0;
  for (int n = 0; n < N; ++n) {
    const float* lr = logits + static_cast<long>(n) * V;
    float maxv = lr[0];
    for (int v = 1; v < V; ++v) maxv = std::max(maxv, lr[v]);
    double sum = 0.0;
    for (int v = 0; v < V; ++v) sum += std::exp(lr[v] - maxv);
    const int tgt = targets[n];
    total += -((lr[tgt] - maxv) - std::log(sum));
    float* dr = dlogits + static_cast<long>(n) * V;
    const float inv_n = 1.0f / N;
    for (int v = 0; v < V; ++v) {
      const float p = static_cast<float>(std::exp(lr[v] - maxv) / sum);
      dr[v] = (p - (v == tgt ? 1.0f : 0.0f)) * inv_n;
    }
  }
  return static_cast<float>(total / N);
}

// Backward pass. Fills every Param.g; assumes forward() just ran on these ids.
void backward(Model& m, const int* ids, const float* dlogits) {
  Acts& a = m.acts;
  const int B = a.B, T = a.T, N = B * T, C = m.C, V = m.V, M = m.M;

  for (Param* p : m.params) std::fill(p->g.begin(), p->g.end(), 0.0f);

  // head backward: logits = lnf_o @ et  ->  dlnf_o, det  (det:[C,V] -> tok_emb)
  std::vector<float> dlnf(N * C), det(static_cast<long>(C) * V);
  matmul_backward(a.lnf_o.data(), m.et.data(), dlogits, dlnf.data(), det.data(),
                  N, C, V);
  for (int vi = 0; vi < V; ++vi)
    for (int c = 0; c < C; ++c) m.tok_emb.g[vi * C + c] += det[c * V + vi];

  // final LayerNorm backward
  const float* last = m.layers.empty() ? a.x0.data() : a.layer.back().r2.data();
  std::vector<float> dnext(N * C);
  layernorm_backward(last, m.ln_f_g.w.data(), a.meanf.data(), a.rstdf.data(),
                     dlnf.data(), dnext.data(), m.ln_f_g.g.data(),
                     m.ln_f_b.g.data(), N, C);

  // reusable scratch
  std::vector<float> dmo(N * C), dr1(N * C), dr1ln(N * C);
  std::vector<float> dhact(N * M), dhpre(N * M);
  std::vector<float> dln2o(N * C), dao(N * C), dln1o(N * C);
  std::vector<float> dbi(N * C), dbiln(N * C);

  for (int l = m.L - 1; l >= 0; --l) {
    Layer& w = m.layers[l];
    LayerAct& la = a.layer[l];
    const float* block_in = (l == 0) ? a.x0.data() : a.layer[l - 1].r2.data();

    // r2 = r1 + mo
    dmo = dnext;
    dr1 = dnext;  // residual contribution; the LN2 path is added below
    // fc_out: mo = hact @ fc_out_W + fc_out_b
    matmul_backward(la.hact.data(), w.fc_out_W.w.data(), dmo.data(),
                    dhact.data(), w.fc_out_W.g.data(), N, M, C);
    bias_grad(dmo.data(), w.fc_out_b.g.data(), N, C);
    gelu_backward(la.hpre.data(), dhact.data(), dhpre.data(), N * M);
    // fc_in: hpre = ln2o @ fc_in_W + fc_in_b
    matmul_backward(la.ln2o.data(), w.fc_in_W.w.data(), dhpre.data(),
                    dln2o.data(), w.fc_in_W.g.data(), N, C, M);
    bias_grad(dhpre.data(), w.fc_in_b.g.data(), N, M);
    // ln2 = LN2(r1)
    layernorm_backward(la.r1.data(), w.ln2_g.w.data(), la.mean2.data(),
                       la.rstd2.data(), dln2o.data(), dr1ln.data(),
                       w.ln2_g.g.data(), w.ln2_b.g.data(), N, C);
    for (int i = 0; i < N * C; ++i) dr1[i] += dr1ln[i];

    // r1 = block_in + ao
    dao = dr1;
    dbi = dr1;  // residual contribution; the LN1 path is added below
    attention_backward(la.ln1o.data(), w.Wq.w.data(), w.Wk.w.data(),
                       w.Wv.w.data(), w.Wo.w.data(), la.q.data(), la.k.data(),
                       la.v.data(), la.attn.data(), dao.data(), dln1o.data(),
                       w.Wq.g.data(), w.bq.g.data(), w.Wk.g.data(),
                       w.bk.g.data(), w.Wv.g.data(), w.bv.g.data(),
                       w.Wo.g.data(), w.bo.g.data(), B, T, C, m.H);
    layernorm_backward(block_in, w.ln1_g.w.data(), la.mean1.data(),
                       la.rstd1.data(), dln1o.data(), dbiln.data(),
                       w.ln1_g.g.data(), w.ln1_b.g.data(), N, C);
    for (int i = 0; i < N * C; ++i) dbi[i] += dbiln[i];

    dnext = dbi;  // grad w.r.t. this block's input == r2 of the block below
  }

  // embedding backward: x0[b,t] = tok_emb[id] + pos_emb[t]
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T; ++t) {
      const int id = ids[b * T + t];
      const float* g = dnext.data() + (b * T + t) * C;
      float* dte = m.tok_emb.g.data() + id * C;
      float* dpe = m.pos_emb.g.data() + t * C;
      for (int c = 0; c < C; ++c) {
        dte[c] += g[c];
        dpe[c] += g[c];
      }
    }
}

// Global-norm gradient clipping, then one AdamW step over every parameter.
void optimizer_step(Model& m, float lr, float grad_clip) {
  double sq = 0.0;
  for (Param* p : m.params)
    for (float g : p->g) sq += static_cast<double>(g) * g;
  const float norm = static_cast<float>(std::sqrt(sq));
  if (grad_clip > 0.0f && norm > grad_clip) {
    const float scale = grad_clip / (norm + 1e-6f);
    for (Param* p : m.params) grad_scale(p->g.data(), p->size(), scale);
  }
  m.step += 1;
  for (Param* p : m.params) {
    const float wd = p->decay ? 0.1f : 0.0f;
    adamw_step(p->w.data(), p->g.data(), p->m.data(), p->v.data(), p->size(),
               static_cast<int>(m.step), lr, 0.9f, 0.95f, 1e-8f, wd);
  }
}

// Build a random (ids, targets) batch from one split. split: 0 train, 1 val.
void sample_batch(Model& m, int split, int B, int T, std::vector<int>& ids,
                  std::vector<int>& targets) {
  const int lo = (split == 0) ? 0 : m.n_train;
  const int hi = (split == 0) ? m.n_train : static_cast<int>(m.data.size());
  const int max_start = hi - T - 1;
  std::uniform_int_distribution<int> pick(lo, max_start);
  ids.resize(B * T);
  targets.resize(B * T);
  for (int b = 0; b < B; ++b) {
    const int s = pick(m.rng);
    for (int t = 0; t < T; ++t) {
      ids[b * T + t] = m.data[s + t];
      targets[b * T + t] = m.data[s + t + 1];
    }
  }
}

}  // namespace

// ===========================================================================
// C-ABI surface (declared in model.h)
// ===========================================================================
WASM_EXPORT TgModel tg_model_create(int vocab, int ctx, int n_layers,
                                    int n_heads, int d_model, int d_mlp,
                                    unsigned int seed) {
  if (d_model % n_heads != 0) return nullptr;
  Model* m = new Model();
  m->V = vocab;
  m->ctx = ctx;
  m->L = n_layers;
  m->H = n_heads;
  m->C = d_model;
  m->M = d_mlp;
  m->hd = d_model / n_heads;
  m->rng.seed(seed);
  m->layers.resize(n_layers);
  init_weights(*m);
  collect_params(*m);
  return static_cast<TgModel>(m);
}

WASM_EXPORT void tg_model_free(TgModel handle) {
  delete static_cast<Model*>(handle);
}

WASM_EXPORT int tg_model_num_params(TgModel handle) {
  Model* m = static_cast<Model*>(handle);
  int n = 0;
  for (Param* p : m->params) n += p->size();
  return n;
}

WASM_EXPORT int tg_model_step(TgModel handle) {
  return static_cast<int>(static_cast<Model*>(handle)->step);
}

WASM_EXPORT void tg_set_data(TgModel handle, const unsigned char* tokens,
                             int n_tokens, float train_frac) {
  Model* m = static_cast<Model*>(handle);
  m->data.assign(tokens, tokens + n_tokens);
  m->n_train = static_cast<int>(n_tokens * train_frac);
}

WASM_EXPORT float tg_train_step(TgModel handle, int batch_size, float lr,
                                float grad_clip) {
  Model* m = static_cast<Model*>(handle);
  const int T = m->ctx;
  std::vector<int> ids, targets;
  sample_batch(*m, 0, batch_size, T, ids, targets);
  forward(*m, ids.data(), batch_size, T);
  std::vector<float> dlogits(static_cast<long>(batch_size) * T * m->V);
  const float loss = cross_entropy(*m, targets.data(), dlogits.data());
  backward(*m, ids.data(), dlogits.data());
  optimizer_step(*m, lr, grad_clip);
  return loss;
}

WASM_EXPORT float tg_eval(TgModel handle, int split, int batch_size,
                          int n_batches) {
  Model* m = static_cast<Model*>(handle);
  const int T = m->ctx;
  std::vector<int> ids, targets;
  std::vector<float> dlogits(static_cast<long>(batch_size) * T * m->V);
  double total = 0.0;
  for (int i = 0; i < n_batches; ++i) {
    sample_batch(*m, split, batch_size, T, ids, targets);
    forward(*m, ids.data(), batch_size, T);
    total += cross_entropy(*m, targets.data(), dlogits.data());
  }
  return static_cast<float>(total / n_batches);
}

WASM_EXPORT int tg_generate(TgModel handle, const unsigned char* prompt,
                            int prompt_len, unsigned char* out, int max_new,
                            float temperature, int top_k, unsigned int seed) {
  Model* m = static_cast<Model*>(handle);
  std::mt19937 gen(seed);
  std::vector<int> ids;
  for (int i = 0; i < prompt_len; ++i) ids.push_back(prompt[i]);
  if (ids.empty()) ids.push_back('\n');

  int produced = 0;
  for (int step = 0; step < max_new; ++step) {
    const int T = std::min(static_cast<int>(ids.size()), m->ctx);
    const int* window = ids.data() + (ids.size() - T);
    forward(*m, window, 1, T);
    const float* logits = m->acts.logits.data() + static_cast<long>(T - 1) * m->V;

    int next;
    if (temperature <= 0.0f) {  // greedy
      next = 0;
      for (int v = 1; v < m->V; ++v)
        if (logits[v] > logits[next]) next = v;
    } else {
      std::vector<float> probs(m->V);
      float maxv = logits[0];
      for (int v = 1; v < m->V; ++v) maxv = std::max(maxv, logits[v]);
      for (int v = 0; v < m->V; ++v)
        probs[v] = std::exp((logits[v] - maxv) / temperature);
      if (top_k > 0 && top_k < m->V) {  // keep only the top-k probabilities
        std::vector<float> sorted = probs;
        std::nth_element(sorted.begin(), sorted.begin() + (m->V - top_k),
                         sorted.end());
        const float thresh = sorted[m->V - top_k];
        for (int v = 0; v < m->V; ++v)
          if (probs[v] < thresh) probs[v] = 0.0f;
      }
      float sum = 0.0f;
      for (float p : probs) sum += p;
      std::uniform_real_distribution<float> u(0.0f, sum);
      float r = u(gen);
      next = m->V - 1;
      for (int v = 0; v < m->V; ++v) {
        r -= probs[v];
        if (r <= 0.0f) { next = v; break; }
      }
    }
    ids.push_back(next);
    out[produced++] = static_cast<unsigned char>(next);
  }
  return produced;
}

// --- checkpointing: serialise step + every param's w / m / v -------------
WASM_EXPORT int tg_state_bytes(TgModel handle) {
  Model* m = static_cast<Model*>(handle);
  long floats = 0;
  for (Param* p : m->params) floats += p->size();
  return static_cast<int>(4 + floats * 3 * 4);  // int32 step + 3 buffers
}

WASM_EXPORT void tg_export_state(TgModel handle, unsigned char* dst) {
  Model* m = static_cast<Model*>(handle);
  const int step = static_cast<int>(m->step);
  std::memcpy(dst, &step, 4);
  long off = 4;
  for (Param* p : m->params) {
    const long n = static_cast<long>(p->size()) * 4;
    std::memcpy(dst + off, p->w.data(), n);  off += n;
    std::memcpy(dst + off, p->m.data(), n);  off += n;
    std::memcpy(dst + off, p->v.data(), n);  off += n;
  }
}

WASM_EXPORT void tg_import_state(TgModel handle, const unsigned char* src) {
  Model* m = static_cast<Model*>(handle);
  int step = 0;
  std::memcpy(&step, src, 4);
  m->step = step;
  long off = 4;
  for (Param* p : m->params) {
    const long n = static_cast<long>(p->size()) * 4;
    std::memcpy(p->w.data(), src + off, n);  off += n;
    std::memcpy(p->m.data(), src + off, n);  off += n;
    std::memcpy(p->v.data(), src + off, n);  off += n;
  }
}
