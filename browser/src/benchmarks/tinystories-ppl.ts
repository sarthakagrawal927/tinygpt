import type { Benchmark, BenchmarkModel, BenchmarkResult } from "./types";
import { BenchmarkError } from "./types";

/// Held-out perplexity on a 50-story slice of TinyStories. The "is the
/// model a competent narrative language model?" baseline.
///
/// Why TinyStories specifically: by design it uses a constrained
/// (~1500-word) vocabulary at a 3-5 year old's reading level. That
/// makes it the cleanest universal benchmark — a byte-level model with
/// a couple million params CAN reach low perplexity on it (Eldan &
/// Li, 2023 showed 1M-param models cross the coherence threshold),
/// but a model trained on disjoint text (Shakespeare, code) won't —
/// so the benchmark differentiates "generally trained" from
/// "narrowly trained" models without any task setup.
///
/// Score = exp(mean per-byte cross-entropy). Lower is better.
/// On the gallery models we'd predict:
///   TinyStories model: ~3-5
///   Shakespeare model: ~50-200 (out-of-distribution)
///   Chat model: ~30-80
const HOLDOUT_URL = "/benchmarks/tinystories-eval.json";

type HoldoutFile = {
  source: string;
  count: number;
  totalBytes: number;
  stories: string[];
};

let _holdoutCache: HoldoutFile | null = null;
async function loadHoldout(): Promise<HoldoutFile> {
  if (_holdoutCache) return _holdoutCache;
  const resp = await fetch(HOLDOUT_URL, { cache: "force-cache" });
  if (!resp.ok) {
    throw new BenchmarkError("failed", `couldn't fetch held-out set: ${resp.status}`);
  }
  _holdoutCache = (await resp.json()) as HoldoutFile;
  return _holdoutCache;
}

/// Log-sum-exp over one row of the logits matrix. Numerically stable —
/// subtract the row max before exp.
function logSumExp(logits: Float32Array, offset: number, vocabSize: number): number {
  let m = -Infinity;
  for (let i = 0; i < vocabSize; i++) {
    const v = logits[offset + i];
    if (v > m) m = v;
  }
  let sumExp = 0;
  for (let i = 0; i < vocabSize; i++) {
    sumExp += Math.exp(logits[offset + i] - m);
  }
  return m + Math.log(sumExp);
}

export const tinyStoriesPpl: Benchmark = {
  id: "tinystories-ppl",
  name: "TinyStories PPL",
  description: "Perplexity on a 50-story TinyStories holdout (lower = better)",
  lowerIsBetter: true,
  // Byte-level for now — BPE models would need an encode-side join with
  // the model's tokenizer; first-pass scope is byte-level only.
  requiresByteLevel: true,
  approxSeconds: 15,
  browserTrainable: true,

  async run(model: BenchmarkModel): Promise<BenchmarkResult> {
    if (model.vocabSize !== 256) {
      throw new BenchmarkError(
        "incompatible",
        `tinystories-ppl currently scores byte-level models only (vocab=256), got vocab=${model.vocabSize}`,
      );
    }
    const holdout = await loadHoldout();

    let totalNll = 0;
    let totalTokens = 0;
    const perStoryPpl: number[] = [];

    for (const story of holdout.stories) {
      const ids = model.encode(story);
      if (ids.length < 2) continue;
      // Chunk to context length. For each chunk, score every position
      // 1..end (position 0's target would be the byte before the start,
      // which is undefined; the engine just skips it).
      const ctx = model.contextLength;
      let storyNll = 0;
      let storyToks = 0;
      for (let chunkStart = 0; chunkStart < ids.length; chunkStart += ctx) {
        const chunk = ids.slice(chunkStart, Math.min(chunkStart + ctx, ids.length));
        if (chunk.length < 2) break;
        // Forward returns row-major [T, vocab].
        const logits = await model.forwardLogits(chunk);
        const vocab = model.vocabSize;
        // Score positions 0..T-2: each predicts chunk[t+1].
        for (let t = 0; t < chunk.length - 1; t++) {
          const target = chunk[t + 1];
          const lse = logSumExp(logits, t * vocab, vocab);
          const logp = logits[t * vocab + target] - lse;
          storyNll += -logp;
          storyToks += 1;
        }
      }
      totalNll += storyNll;
      totalTokens += storyToks;
      perStoryPpl.push(storyToks > 0 ? Math.exp(storyNll / storyToks) : Infinity);
    }

    if (totalTokens === 0) {
      throw new BenchmarkError("failed", "no scorable positions in the held-out set");
    }
    const meanNll = totalNll / totalTokens;
    const ppl = Math.exp(meanNll);
    return {
      score: ppl,
      details: {
        stories: holdout.stories.length,
        tokens: totalTokens,
        meanNll,
        bytes: holdout.totalBytes,
        // Per-story P50 / P90 for the "is this generally good or great
        // on a few outliers?" question.
        p50Ppl: perStoryPpl.slice().sort((a, b) => a - b)[
          Math.floor(perStoryPpl.length * 0.5)
        ],
        p90Ppl: perStoryPpl.slice().sort((a, b) => a - b)[
          Math.floor(perStoryPpl.length * 0.9)
        ],
      },
    };
  },
};
