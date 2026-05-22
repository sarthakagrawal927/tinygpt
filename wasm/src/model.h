// model.h — the WASM-side TinyGPT: a full byte-level GPT assembled from the
// kernels in kernels.h (Phase 4).
//
// The kernels (matmul, layernorm, attention, adamw) are the heavy ops; this
// layer adds the connective tissue with NO autograd — embeddings, GELU/MLP,
// the tied output head, cross-entropy, and the AdamW training step, each with
// its own hand-written backward. The browser Worker drives the model entirely
// through the C-ABI functions below, so all the math stays in WASM.
//
// Architecture matches python_ref/model.py and configs/model.byte-tinygpt-v0.json:
// byte-level vocab, pre-LayerNorm blocks, tied input/output embeddings.
#ifndef TINYGPT_WASM_MODEL_H
#define TINYGPT_WASM_MODEL_H

#include "kernels.h"

// Opaque handle — the browser never sees the C++ struct layout.
typedef void* TgModel;

// Create / destroy. d_model must be divisible by n_heads.
WASM_EXPORT TgModel tg_model_create(int vocab, int ctx, int n_layers,
                                    int n_heads, int d_model, int d_mlp,
                                    unsigned int seed);
WASM_EXPORT void tg_model_free(TgModel m);
WASM_EXPORT int tg_model_num_params(TgModel m);

// Attach a byte-token corpus and split it train/val (e.g. train_frac 0.9).
// The bytes are copied in; the caller may free its buffer afterwards.
WASM_EXPORT void tg_set_data(TgModel m, const unsigned char* tokens,
                             int n_tokens, float train_frac);

// One AdamW step on a freshly sampled random batch. Returns the batch loss.
WASM_EXPORT float tg_train_step(TgModel m, int batch_size, float lr,
                                float grad_clip);

// Average loss over n_batches random batches. split: 0 = train, 1 = val.
WASM_EXPORT float tg_eval(TgModel m, int split, int batch_size, int n_batches);

// Autoregressive generation. Writes up to max_new bytes into `out` and returns
// the count. temperature <= 0 is greedy; top_k <= 0 disables top-k.
WASM_EXPORT int tg_generate(TgModel m, const unsigned char* prompt,
                            int prompt_len, unsigned char* out, int max_new,
                            float temperature, int top_k, unsigned int seed);

// Step counter (AdamW step / number of train_step calls so far).
WASM_EXPORT int tg_model_step(TgModel m);

// --- checkpointing -------------------------------------------------------
// Serialise the full trainable state — weights, the AdamW moments m/v, and the
// step count — so a run can be saved to OPFS and resumed after a page refresh.
// import requires a model created with the SAME config (identical layout).
WASM_EXPORT int tg_state_bytes(TgModel m);
WASM_EXPORT void tg_export_state(TgModel m, unsigned char* dst);
WASM_EXPORT void tg_import_state(TgModel m, const unsigned char* src);

#endif  // TINYGPT_WASM_MODEL_H
