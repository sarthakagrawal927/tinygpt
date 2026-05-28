# Parked: the four "multi-model" directions

User decision (this session): hold these four options and focus on the
HF-compat capabilities (SwiGLU + RoPE + GQA + BPE tokenizer) first.
Once those land, the Mac app can load any modern open-weight model
and LoRA-fine-tune it. Then these four become more interesting
because we'd have real models to apply them to.

Each entry below is a brief — what it is, what it costs, what it buys,
and where the wiring would go.

## 1. Multi-modal (vision + text)

Add image input to the Mac app. The standard recipe (LLaVA-style):

  - Pre-trained vision encoder: CLIP-ViT-B/32 or SigLIP-B. Loaded
    once at startup (~150 MB int8).
  - Tiny projector: 2-layer MLP from vision encoder dim → LLM token
    embedding dim. Trained briefly on image-caption pairs.
  - At inference: image → encoder → projector → "vision tokens" →
    prepend to text prompt → LLM processes the joined sequence.

**Engineering**: ~1-2 focused weeks.
**Dependency**: image-caption training data (~100 MB of paired data
for the projector). LAION-400M's filtered subsets are accessible.
**Demo value**: high. "Train your model to describe images" reads as
flagship multi-modal capability.
**Quality bar**: passable. Real multi-modal models need instruction-
tuning on multi-modal chat data (LLaVA-Instruct, ShareGPT4V) — we'd
land "describes images" not "answers questions about images well."

## 2. Multiple models in one session

Two slightly different things:

a) Multiple bases simultaneously: load Llama-3-3B AND Phi-3-mini at
   once in the same process. 48 GB easily holds 3-4. Each is its
   own `TinyGPTModel` instance; the SwiftUI app shows a "Model A vs
   Model B" side-by-side view. ~1 day of UI work — purely additive.

b) Multiple LoRA adapters over one base: ALREADY SHIPPED. Tonight's
   `LoraStackInjection` composes any number of adapters over a single
   base. Use `--lora` multiple times on the sample CLI.

So the only un-shipped piece is (a)'s UI. Tag this as a v0.2 polish item.

## 3. Mixture of Experts (MoE)

Architecturally the most interesting of the four. Replace each MLP
with N parallel "expert" MLPs (typically 8) + a small "router" that
picks the top-2 experts per token based on a learned gating signal.

Why: a model with 8 experts × 7B params each = 56B parameters total,
but only 14B *active* per token (since each token picks 2 of 8).
Compute cost matches a 14B dense model; quality matches a 56B dense.
Mixtral 8x7B is the open-weight reference.

**Engineering**: ~3-5 focused days.
**Cost**: ~200 lines (router + top-k expert dispatch).
**Constraint**: training MoE requires a load-balancing auxiliary loss
to keep experts from collapsing onto the same routing. Adds a hyper-
parameter and complicates the training loop slightly.
**Demo value**: medium. The pitch ("47B params on a laptop") is real
but doesn't tweet as well as multi-modal.
**Combines with HF loading**: yes. Once we can load HF models, we
could load Mixtral and exercise the MoE path.

## 4. Model ensembling

Run 2-3 small models in parallel, average logits before sampling.
Cheap (~50 lines) but doubles/triples compute for modest quality
gain. Mostly a research technique these days; production stacks
prefer one bigger model over averaged smaller ones.

**Engineering**: ~50 lines.
**Cost**: real-time perf gets divided by N.
**Demo value**: low. Hard to convey why this matters.

Recommend: skip unless we find a specific use case (e.g.,
ensembling LoRA adapters trained on different domains).

## When to revisit

After the HF-compat pieces land. At that point:
- Multi-modal becomes "add vision to your LoRA-fine-tuned Llama"
- MoE becomes "train your own Mixtral-style architecture"
- Multi-base UI becomes a polish item once people are actually using
  the Mac app

If forced to pick one: **multi-modal**. It's the highest visual-impact
and most "wow" feature. MoE is more technically interesting but harder
to convey to anyone who isn't deep in the field.
