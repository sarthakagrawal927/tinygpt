# Leaderboard — benchmarks for tiny models

The TinyGPT leaderboard at [/leaderboard.html](https://tinygpt.sarthakagrawal.dev/leaderboard.html)
ranks small, browser-runnable language models on a curated set of
benchmarks. Three properties are non-negotiable:

1. **Browser-runnable evaluation.** Every benchmark scores a loaded
   `.tinygpt` file in a browser tab. No Python, no GPU cluster, no
   server-side judgment. Verifiability matters more than expressivity.
2. **Deterministic.** Given the same model and benchmark, the score is
   reproducible. Trial sets are seeded; no LLM-as-judge noise.
3. **Trainable at our scale.** A benchmark belongs on this leaderboard
   only if a community member can credibly compete with a model trained
   in their browser. Big-lab benchmarks (MMLU, GSM8k) need too many
   parameters; this leaderboard is for the 100K-100M-param range.

---

## 1. Launch benchmarks

Three at launch, each picked for clean differentiation at our scale:

### TinyStories PPL

> Held-out perplexity over a 50-story slice of TinyStories. Lower is
> better. Score = exp(mean per-byte cross-entropy).

The universal language-modeling baseline. TinyStories was designed for
sub-100M models specifically (Eldan & Li, 2023) — at 1M params, a model
that's seen enough TinyStories text can drop perplexity below 5, which
is roughly when its outputs become recognizable as English children's
stories.

Every byte-level LM gets a real number. Models trained on disjoint
distributions (Shakespeare, code) score 5-10× worse, so the benchmark
differentiates "generally good" from "narrowly trained."

**Eval data**: `browser/public/benchmarks/tinystories-eval.json` — 50
stories, ~13 KB, ~3,000 byte-tokens scored.

**Scorer**: `browser/score_gallery.mjs` — Node + WASM module, ~40s per
model.

### Sort-6

> Given 6 random digits, output them sorted ascending. Greedy
> continuation, exact match over 200 deterministic trials. Higher %
> is better.

Karpathy's minGPT algorithmic-task family. Prompt format:
`"sort: 5 1 4 2 6 3 = "`, expected continuation: `"1 2 3 4 5 6"`.

A task-specific benchmark — the gallery models (Shakespeare,
TinyStories, code, etc.) all score 0% because they were never trained
on this format. **That's the point**: the leaderboard slot is open
for someone to train a tiny task-specific model and beat the gallery.

**Eval data**: 200 trials, seeded with `0x517` Mulberry32.

**Scorer**: `browser/score_gallery_tasks.mjs` (Node + WASM), or the
browser-side benchmark runner (coming).

### Reverse-16

> Reverse a lowercase string of 4-16 chars. Same task-family as Sort-6.
> Higher % is better.

Slightly harder than Sort-6 because the last input character becomes
the first of the output — longer dependency tail across the equals
sign. Prompt: `"reverse: hello = "`, expected: `"olleh"`.

**Eval data**: 200 trials, seeded with `0x3EE5`.

---

## 2. How to read the leaderboard

Each row has four numbers:

| Column | What |
|---|---|
| **Rank** | Score-sorted position |
| **Params** | Total model parameters (formatted 1.2M / 100K etc.) |
| **Score** | The benchmark-specific number |
| **Train** | Wall-clock training time (`trainWallMs` in the manifest) |

Two pareto views matter:

- **Score per parameter** — a 100K-param sorter that hits 95% is more
  impressive than a 9.6M-param model at 95%. Tiny + accurate = great.
- **Score per training compute** — efficient training is rewarded; a
  model that converges in 5 minutes beats one that takes 50 hours, all
  else equal.

Click any row's "try →" to load that model in the playground and
generate from it directly.

---

## 3. Submitting a model

Until the upload UI ships (work-in-progress), submission is via a PR:

1. Train a model in the browser playground. Hit "Download model" to
   get a `.tinygpt` file.
2. Drop it into `data/gallery/<your-id>.tinygpt`.
3. Add a slot definition to `browser/finalize_gallery.mjs` SLOTS list:
   ```js
   { id: "<your-id>", name: "Your name", icon: "🎯",
     blurb: "What makes this one different",
     corpus: "How it was trained" }
   ```
4. Run `node browser/finalize_gallery.mjs` to fp16-pack it +
   `node browser/score_gallery.mjs` and `score_gallery_tasks.mjs` to
   score it on the launch benchmarks. Both update the manifest in
   place.
5. Open a PR. We merge after eyeballing.

The browser-side upload + auto-score flow is on the roadmap;
fundamentally the same artifact, different UX.

---

## 4. Adding a new benchmark

The benchmark engine lives in `browser/src/benchmarks/`. Each benchmark
implements the `Benchmark` interface from `types.ts`:

```typescript
export interface Benchmark {
  id: string;                              // matches manifest.benchmarks.<id>
  name: string;
  description: string;
  lowerIsBetter: boolean;
  requiresByteLevel: boolean;              // true for byte-level only
  approxSeconds: number;                   // shown as ETA in the UI
  browserTrainable: boolean;               // whether browser-trained
                                           // submissions can credibly compete
  run(model: BenchmarkModel): Promise<BenchmarkResult>;
}
```

The `BenchmarkModel` handle (see `types.ts`) exposes encode/decode,
`forwardLogits(ids)` for perplexity benchmarks, and
`generate(prompt, n, temp)` for task benchmarks. Three concrete
patterns:

- **Perplexity benchmark** — see `tinystories-ppl.ts`. Fetches a
  held-out JSON, runs `forwardLogits` on each story, computes per-byte
  cross-entropy.
- **Task-with-exact-match benchmark** — see `sort-6.ts` / `reverse-16.ts`.
  Builds N deterministic trials with seeded RNG, calls `generate` per
  trial, scores by exact match.
- **Future: classification** — pick the most likely of K canned
  continuations. Not shipped yet; design is in
  `docs/benchmark_classification.md` (drafted).

After implementing the benchmark, add it to `registry.ts` and add a
tab button to `leaderboard.astro` with the matching `data-bench` id.
The Node scorer (`score_gallery.mjs` or `score_gallery_tasks.mjs`)
needs the same logic in its task list — they're not yet auto-generated
from the registry.

---

## 5. Design philosophy

A few non-obvious choices worth surfacing:

**Why no human or LLM judges at launch.** LLM-as-judge introduces an
opaque dependency on a bigger model that the submitter doesn't run.
Two leaderboard entries get scored against the same judge today, a
different judge tomorrow. Reproducibility breaks. Deterministic
benchmarks survive infrastructure changes.

**Why a fixed eval set, not random.** A randomly-sampled eval (e.g., a
HuggingFace dataset's test split fetched at score-time) creates a
moving target. Our checked-in `tinystories-eval.json` is small, fixed,
versioned with the rest of the code. Cheating by training-on-the-eval
is possible but visible (any high-quality but high-PPL model would
look suspicious).

**Why the existing gallery cards are "featured" while submissions
aren't.** Two reasons. First, editorial — the gallery sidebar has
limited space, and curating it to a thematic set (one Shakespeare, one
code, one stories, etc.) is more useful than auto-ranking by score.
Second, anti-spam — featured entries land in the gallery sidebar of
the playground for everyone; community submissions appear on the
leaderboard but not in the playground gallery until promoted.

**Why include task-specific benchmarks at all.** The point of the
leaderboard isn't to find the best general LM — that competition is
won by big labs with big compute. The point is to find clever,
*small* models. Sort-6 and Reverse-16 are tiny enough that a 50K-param
model can solve them perfectly with the right training data. Generic
PPL benchmarks don't have that property — perplexity scales smoothly
with params and data forever.

---

## 6. Roadmap

The leaderboard at launch is the minimum viable thing. Planned
additions:

- **Browser-side "Run benchmark on loaded model"** — currently scores
  are pre-computed by the Node scorer; the browser leaderboard reads
  static numbers. A button to evaluate the user's current model
  in-browser is what closes the submission loop.
- **More benchmarks**: `shakespeare-ppl` (narrow distribution
  perplexity), `code-completion-pass` (function-body completion that
  passes a test), `multiple-choice-mini` (10 hand-curated facts).
- **Pareto-frontier views**: side-by-side scatter of score-vs-params
  and score-vs-compute, not just sorted lists.
- **Provenance verification**: top-N entries get re-trained from their
  declared config to confirm reported scores are real (anti-cheat,
  best-effort).
- **R2/D1 backend**: replaces the current manifest-as-source-of-truth
  with a server-side database, supports community submissions at scale.
