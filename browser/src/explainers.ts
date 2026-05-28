/**
 * explainers.ts — the content behind every "?" popover in the UI.
 *
 * Each entry pairs a one-paragraph plain-English explanation with the single
 * best link for someone who wants to go deeper. Keep prose terse: the popover
 * is a glance, the link is the rabbit hole.
 */

export interface Explainer {
  title: string;
  body: string;
  link?: { href: string; label: string };
}

const docsLink = (path: string, label: string) => ({
  href: `https://github.com/sarthakagrawal927/tinygpt/blob/main/${path}`,
  label,
});

export const EXPLAINERS: Record<string, Explainer> = {
  // --- fine-tuning -------------------------------------------------------
  loraOverview: {
    title: "LoRA — fine-tuning, cheaply",
    body: "Instead of nudging all 800k weights, freeze them all and slip in a few tiny new matrices (rank 4 here) into the attention layers. You train ONLY those — typically 1–10% of the base size. One base model can then host many swappable adapters: 'formal writer', 'children's stories', 'Shakespeare-style' — each adapter is a few KB.",
    link: docsLink("docs/lora_guide.md", "docs/lora_guide.md — full LoRA walkthrough"),
  },

  // --- size preset -------------------------------------------------------
  sizePreset: {
    title: "Model size presets",
    body: "Five curated sizes from Tiny (70k params, finishes in seconds) to XL (6.4M params, WebGPU recommended). Picking one auto-fills every knob; switching to Custom lets you tweak individually. The pre-flight estimate below updates live.",
    link: docsLink("docs/performance.md", "docs/performance.md — perf work"),
  },

  // --- model config ------------------------------------------------------
  layers: {
    title: "Transformer layers",
    body: "Each layer runs attention (every token looks at every other token) then a small MLP. Stacking more layers lets the model do deeper chains of reasoning. Three is plenty for a tiny GPT — every layer doubles training time.",
    link: {
      href: "https://jalammar.github.io/illustrated-transformer/",
      label: "The Illustrated Transformer",
    },
  },
  dModel: {
    title: "d_model — embedding width",
    body: "Every token is turned into a vector of d_model numbers; that vector is what the network actually moves around. Wider = more capacity but quadratically more compute in attention. 96 is the sweet spot here.",
    link: {
      href: "https://www.youtube.com/watch?v=kCc8FmEb1nY",
      label: "Karpathy — Let's build GPT",
    },
  },
  ctx: {
    title: "Context length",
    body: "How many tokens (bytes, here) the model can see at once. The attention matrix is ctx × ctx, so doubling context quadruples that cost. 64 is enough to learn short-range structure; longer needs more compute.",
    link: docsLink("docs/model_guide.md", "model_guide.md in this repo"),
  },
  maxSteps: {
    title: "Max training steps",
    body: "One step = forward pass, backward pass, AdamW update on one batch. With batch 16 and ctx 64 that's ~1,000 bytes processed per step. 1,500 steps gets a tiny model from random to noticeably patterned in well under a minute.",
  },
  batch: {
    title: "Batch size",
    body: "How many independent text windows we train on per step. Bigger batch = smoother gradients (less noise) but more memory and compute per step. 16 is comfortable for an in-browser run.",
    link: {
      href: "https://www.deeplearningbook.org/contents/optimization.html",
      label: "Goodfellow et al — Optimization",
    },
  },
  lr: {
    title: "Learning rate",
    body: "How aggressively to nudge each weight on every step. Too high → the loss oscillates (or plateaus, which looks like a modelling ceiling but isn't). Too low → training stalls. 0.0003 is the safe default for char-level GPTs trained with AdamW, matching the Python reference. Raise to 1e-3 only if the model and dataset are both very small.",
    link: {
      href: "https://arxiv.org/abs/1711.05101",
      label: "AdamW — Loshchilov & Hutter",
    },
  },
  backend: {
    title: "Backend — WASM or WebGPU",
    body: "WASM runs the same model on the CPU via hand-derived C++ kernels — fully supported. WebGPU runs the forward, backward, and AdamW on the GPU. Measured on Apple M-series: WebGPU is ~2.6× faster than WASM on small models and ~12× faster on the XL preset — the speedup grows with model size as the GPU's arithmetic throughput dominates. The kernels are parity-checked, so it's correct. If your machine has WebGPU, use it.",
    link: docsLink("docs/browser_notes.md", "browser_notes.md — WASM vs WebGPU"),
  },
  corpus: {
    title: "The training corpus",
    body: "The model has nothing to learn except patterns in this text. The default is the full TinyShakespeare corpus (~1.1 MB) — enough that a small model learns letter and word patterns, and a larger model trained for ~15 minutes starts producing readable pseudo-Shakespeare. Paste your own text, upload a file, or pull a dataset from Hugging Face to change what it learns.",
    link: docsLink("docs/learn.md", "docs/learn.md — the guided path"),
  },

  // --- sampling ----------------------------------------------------------
  prompt: {
    title: "Prompt",
    body: "The starting bytes the model conditions on. It predicts the next byte, appends it, then conditions on the new sequence — repeat until the requested length. A common prompt the model has seen helps it 'find its footing.'",
  },
  temp: {
    title: "Temperature",
    body: "Controls how random sampling is. 0 = always pick the most likely next byte (deterministic, repetitive). 1 = sample from the model's raw distribution. >1 = flatter, more chaotic. 0.7–0.9 usually reads best.",
    link: {
      href: "https://docs.cohere.com/docs/temperature",
      label: "Cohere — temperature explained",
    },
  },
  genTokens: {
    title: "Tokens to generate",
    body: "How many bytes to sample after the prompt. A tiny model loses coherence past a few dozen tokens — beyond that you're mostly watching it loop on patterns it learned.",
  },

  // --- loss & metrics ----------------------------------------------------
  perplexity: {
    title: "Perplexity",
    body: "exp(loss). It's the more intuitive cousin of cross-entropy: 'on average, the model thinks the next byte could be one of these N options.' Perplexity of 256 = pure random guess (no information). Perplexity of 10 = the model has narrowed each next byte down to ~10 plausible options. Lower is better.",
    link: {
      href: "https://en.wikipedia.org/wiki/Perplexity",
      label: "Perplexity — Wikipedia",
    },
  },
  elapsed: {
    title: "Elapsed time",
    body: "Wall-clock time since you clicked Start. Includes the first-step overhead (loading the WASM module, allocating buffers), which is why the first second or two often shows a low tokens/sec — it stabilises within ~5 steps.",
  },
  bestVal: {
    title: "Best validation loss",
    body: "The lowest val loss seen so far, and the step at which it occurred. Once 'Best' stops getting beaten while train loss keeps falling, you're memorising — the right time to either stop or reduce capacity. This is the practical signal pros watch.",
    link: {
      href: "https://www.deeplearningbook.org/contents/regularization.html",
      label: "Goodfellow — early stopping",
    },
  },
  trainValGap: {
    title: "Train ↔ holdout gap",
    body: "val_loss − train_loss. Near zero = the model generalises (good). Steadily growing = the train loss is improving on text the model has seen but val isn't, i.e. the model is memorising rather than learning patterns. A gap > 0.5 on a tiny corpus like this is normal.",
  },

  trainLoss: {
    title: "Training loss",
    body: "Cross-entropy on the batch the model just trained on. A uniform guess over 256 bytes is ln(256) ≈ 5.55; getting under ~2 means the model has learned real structure. It should fall quickly at first, then slowly.",
    link: {
      href: "https://en.wikipedia.org/wiki/Cross-entropy",
      label: "Cross-entropy — Wikipedia",
    },
  },
  valLoss: {
    title: "Validation loss",
    body: "Same metric, but evaluated on text the model didn't train on. If val loss stops falling (or rises) while train loss keeps dropping, the model is memorising rather than generalising — overfit.",
    link: {
      href: "https://www.deeplearningbook.org/contents/regularization.html",
      label: "Goodfellow — generalisation & regularisation",
    },
  },
  tokensPerSec: {
    title: "Throughput",
    body: "Tokens (bytes here) processed per second. WASM uses multi-threaded SIMD; WebGPU runs the whole training loop on the GPU. On a modern laptop expect roughly 10k–80k tok/s for a small model under WASM, and ~3× that under WebGPU on the smaller presets — growing to ~12× on XL.",
    link: docsLink("docs/performance.md", "docs/performance.md — perf work"),
  },
  eta: {
    title: "Estimated time remaining",
    body: "tokens_left ÷ current_throughput. Updates live, so it stabilises after a few steps. Trust the estimate — it'll be within 10% by step 50.",
  },

  // --- machine capabilities ----------------------------------------------
  webgpuPill: {
    title: "WebGPU",
    body: "Modern browser GPU API — lets web code run shader programs (WGSL) on your real GPU. Chrome 113+, Safari 18+, Edge 113+. The kernels in webgpu/ run here when this is on.",
    link: {
      href: "https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API",
      label: "MDN — WebGPU",
    },
  },
  wasmSimd: {
    title: "WASM SIMD",
    body: "WebAssembly 128-bit vector instructions. With this on, four floats are multiplied per cycle in the matmul inner loop instead of one — about 1.6× faster on this project's hot paths.",
    link: {
      href: "https://github.com/WebAssembly/simd/blob/main/proposals/simd/SIMD.md",
      label: "WASM SIMD proposal",
    },
  },
  lossMeaning: {
    title: "What the loss number actually means",
    body: "TinyGPT predicts one byte at a time, so loss is in natural-log nats per byte. Random guessing on a 256-byte vocab is ln(256) ≈ 5.55. Useful thresholds for this codebase:\n\n• > 3.0 — letter-pair statistics only, no recognisable words\n• > 2.0 — short n-grams, letters look right but words don't form\n• > 1.5 — words start forming, grammar still random\n• < 1.5 — real grammar emerges, sentences start parsing\n• < 1.0 — fluent local grammar; on tiny corpora this is usually memorisation, so watch the train↔val gap\n\nIf you want grammatical English, aim for val loss below 1.5 — that's the bar. Below that you're in the regime where the prose starts looking like prose.",
  },
  memory64: {
    title: "WebAssembly Memory64",
    body: "64-bit pointers in the WASM module. Without it, the tab heap is capped at 4 GB (~250M fp32 params with Adam state); with it, the cap rises into the tens of GB — enough to host the Behemoth preset. Chromium 133+, Firefox 134+. When this pill is on, you're loading tinygpt64.wasm.",
    link: {
      href: "https://github.com/WebAssembly/memory64",
      label: "WASM Memory64 proposal",
    },
  },
  crossOriginIsolated: {
    title: "Cross-origin isolated",
    body: "Required by browsers to allow SharedArrayBuffer and high-resolution timers. Not strictly needed for this app's single-threaded training, but a 'yes' here means the page is configured for performance-sensitive work.",
    link: {
      href: "https://web.dev/articles/coop-coep",
      label: "web.dev — COOP/COEP",
    },
  },
  gpuMem: {
    title: "GPU memory footprint",
    body: "How much GPU memory the current model occupies. Computed as 12 bytes per parameter — each weight needs the f32 value itself plus its two AdamW optimizer moments (m and v) for resuming training. A 9.6M-param Huge model = ~110 MB of persistent GPU buffers, regardless of whether you're sampling or training. During training there's additional transient memory (per-step activations + gradients, typically 30-100 MB more on Huge) but that's released between steps. Note: WebGPU doesn't expose actual GPU memory usage to the page — this number is OUR allocation accounting, which is the only thing we can measure precisely.",
  },
  f16Storage: {
    title: "f16-storage matmul",
    body: "The matmul kernel reads weight tensors from a packed-half (f16) GPU buffer instead of the usual f32 — half the bytes per K-step, accumulation stays in f32 for numerical safety. Activates after the startup numerics gate confirms loss is preserved (max relative error < 5%, mean < 0.5% against the f32 reference). On bandwidth-bound matmuls this is ~1.5-2× faster; on Apple Silicon WebGPU the matmul path IS bandwidth-bound, so this lights up across the board. Uses pack2x16float / unpack2x16float — core WGSL, available on every WebGPU device (no shader-f16 extension needed).",
    link: {
      href: "https://www.w3.org/TR/WGSL/#pack2x16float-builtin",
      label: "WGSL — pack2x16float",
    },
  },
  shaderF16: {
    title: "WGSL shader-f16",
    body: "WGSL extension that lets shaders use the f16 (half-precision) scalar type directly — not just for storage, but for compute. On bandwidth-bound matmuls this halves the memory traffic and on f16-capable hardware also halves the compute throughput cost. Chrome 121+ stable. When on, the matmul + MLP path runs in f16; final accumulation stays in f32 for numerical safety.",
    link: {
      href: "https://www.w3.org/TR/WGSL/#extension-f16",
      label: "WGSL spec — f16 extension",
    },
  },
  subgroups: {
    title: "WebGPU subgroups",
    body: "Cross-lane primitives (subgroupAdd, subgroupBroadcast, etc.) for the threads in a workgroup. In TinyGPT we use them for the reductions inside layernorm, softmax, and cross-entropy — one cooperative pass across the row instead of a workgroup-shared-memory scan. Behind chrome://flags#enable-unsafe-webgpu on Chrome 125+; ships in Chrome stable later.",
    link: {
      href: "https://developer.chrome.com/blog/new-in-webgpu-125",
      label: "Chrome — subgroups in WebGPU",
    },
  },
  coopMatrix: {
    title: "Cooperative matrix (WMMA)",
    body: "WGSL extension that maps to hardware matrix-multiply-accumulate units — NVIDIA's tensor cores, AMD's MFMA, Apple's AMX. Single instruction replaces a tile of multiply-adds; on NVIDIA that's a ~3× speedup, on Apple ~1.3× (the AMX path is more constrained). Behind chrome://flags#enable-unsafe-webgpu + experimental features. The extension name is chromium_experimental_subgroup_matrix and is still evolving.",
    link: {
      href: "https://www.w3.org/TR/WGSL/#cooperative-matrix",
      label: "WGSL spec (proposal)",
    },
  },
  webnn: {
    title: "WebNN API",
    body: "Web Neural Network API. Hands a model graph to the OS — CoreML on macOS (which can route to the Apple Neural Engine), DirectML on Windows, TFLite on Android. Used here for inference / sampling only; training stays on WebGPU. Behind chrome://flags#enable-webnn-api.",
    link: {
      href: "https://developer.chrome.com/docs/ai/built-in",
      label: "Chrome — built-in AI APIs",
    },
  },
  heap: {
    title: "JavaScript heap usage",
    body: "Live JS heap (in MB), updated every 2 s. Shows used / V8's per-tab limit. Training a 0.8M model typically lives in 30–80 MB; bigger configs scale roughly linearly with params plus the Adam optimizer state (≈ 8× params bytes). Chromium-only — Safari/Firefox don't expose this.",
    link: {
      href: "https://developer.mozilla.org/en-US/docs/Web/API/Performance/memory",
      label: "MDN — performance.memory",
    },
  },

  // --- section-level -----------------------------------------------------
  whatIsTraining: {
    title: "What 'training' means here",
    body: "Sample a window of bytes, ask the model to predict the next byte at every position, measure how wrong it is (cross-entropy), then nudge every one of the model's 800k weights very slightly in the direction that would have been less wrong. Repeat thousands of times. That's it.",
    link: {
      href: "https://www.3blue1brown.com/topics/neural-networks",
      label: "3Blue1Brown — Neural Networks",
    },
  },
  whatIsLoss: {
    title: "Reading the loss curve",
    body: "Loss = average surprisal. It starts near 5.55 (random guess over 256 bytes) and falls as the model learns. The curve is the most honest single signal of whether the model is learning at all — flat means dead, falling fast means learning, slow tail means soaking up the hard bits.",
    link: docsLink("docs/notes.md", "docs/notes.md — the project's loss numbers"),
  },
  whatIsSampling: {
    title: "Sampling from the model",
    body: "Feed the prompt in, ask the model for the next-byte distribution, draw one byte from it (biased by temperature), append, repeat. The model has no plan or goal — it just predicts one byte at a time, forever.",
    link: {
      href: "https://github.com/karpathy/nanoGPT/blob/master/sample.py",
      label: "Karpathy nanoGPT — sample.py",
    },
  },

  liveInference: {
    title: "Sampling while training is still running",
    body: "The model is a single C++ instance inside the Web Worker. Training mutates its weights in place; Sample runs a forward pass against the same weights between training steps. No copy, no pause, no separate 'trained model' object — you're snapshotting the same neurons as they learn. Sample twice during a run and you can see what the model picked up in those few seconds.",
    link: docsLink("docs/browser_notes.md", "docs/browser_notes.md — Web Worker"),
  },
  machine: {
    title: "Your machine, and what it can train",
    body: "TinyGPT detects your browser's capabilities (WebGPU, WASM SIMD, cross-origin isolation) and your hardware (cores, RAM, a quick CPU probe) and suggests a model size that should train in a sensible amount of time. Click Apply to use it.",
    link: docsLink("src/runtime_detect.ts", "runtime_detect.ts — how the probe works"),
  },
};
