// test_wasm_model.cpp — correctness gate for the WASM-side TinyGPT (Phase 4).
//
// The kernel tests prove each op in isolation; this proves the whole model
// assembled from them — forward, every hand-written backward, and the AdamW
// step — by running the test that matters most (tests/README.md):
//
//     can a tiny model drive loss down on a few KB of repeated text?
//
// If loss does not fall here, a backward formula or the block wiring is wrong.
// Runs natively via clang (see wasm/build_native.sh) before any WASM build.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../wasm/src/model.h"

static int g_failed = 0;

static void report(const char* name, bool ok, double detail) {
  std::printf("%s %-34s (%.4f)\n", ok ? "ok  " : "FAIL", name, detail);
  if (!ok) ++g_failed;
}

int main() {
  // A small, fast config — same shape as the browser demo would use.
  const int V = 256, ctx = 32, layers = 2, heads = 2, d_model = 64, d_mlp = 128;
  TgModel m = tg_model_create(V, ctx, layers, heads, d_model, d_mlp, 42);
  if (!m) {
    std::printf("FAIL tg_model_create returned null\n");
    return 1;
  }
  const int n_params = tg_model_num_params(m);
  std::printf("model: %d params  (%d layers, d_model %d)\n\n", n_params, layers,
              d_model);
  report("model has parameters", n_params > 0, n_params);

  // A few KB of repeated text — the overfit corpus.
  std::string corpus;
  const std::string unit = "the quick brown fox jumps over the lazy dog. ";
  while (corpus.size() < 3000) corpus += unit;
  tg_set_data(m, reinterpret_cast<const unsigned char*>(corpus.data()),
              static_cast<int>(corpus.size()), 0.9f);

  // A random model's loss should sit near ln(vocab).
  const float init_loss = tg_eval(m, 0, 8, 5);
  report("initial loss ~ ln(256)=5.545", std::fabs(init_loss - 5.545f) < 0.7f,
         init_loss);

  // THE test: train and watch loss collapse on the repeated text.
  float loss = init_loss;
  for (int step = 1; step <= 400; ++step) {
    loss = tg_train_step(m, 8, 1e-3f, 1.0f);
    if (step % 100 == 0) std::printf("    step %3d  loss %.4f\n", step, loss);
  }
  report("loss fell far below initial", loss < init_loss * 0.25f, loss);
  report("tiny overfit (loss < 0.5)", loss < 0.5f, loss);
  report("optimizer step count", tg_model_step(m) == 400, tg_model_step(m));

  // Greedy generation should run and reproduce the memorized pattern.
  unsigned char out[80];
  const char* prompt = "the ";
  const int produced =
      tg_generate(m, reinterpret_cast<const unsigned char*>(prompt), 4, out, 70,
                  0.0f, 0, 7);
  report("generate produced max_new bytes", produced == 70, produced);
  std::printf("    greedy sample: \"%.*s\"\n", produced,
              reinterpret_cast<char*>(out));

  // Checkpoint round-trip: export the state, load it into a fresh model, and
  // confirm the fresh model produces the identical greedy continuation.
  std::vector<unsigned char> blob(tg_state_bytes(m));
  tg_export_state(m, blob.data());
  TgModel m2 = tg_model_create(V, ctx, layers, heads, d_model, d_mlp, 999);
  tg_import_state(m2, blob.data());
  unsigned char out2[80];
  const int produced2 =
      tg_generate(m2, reinterpret_cast<const unsigned char*>(prompt), 4, out2, 70,
                  0.0f, 0, 7);
  const bool identical =
      produced2 == produced && std::memcmp(out, out2, produced) == 0;
  report("checkpoint round-trip reproduces output", identical, identical ? 1 : 0);
  tg_model_free(m2);

  tg_model_free(m);
  std::printf("\n%s\n", g_failed == 0 ? "all WASM model tests passed"
                                      : "SOME WASM MODEL TESTS FAILED");
  return g_failed == 0 ? 0 : 1;
}
