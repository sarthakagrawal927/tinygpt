import type { Benchmark, BenchmarkModel, BenchmarkResult } from "./types";
import { BenchmarkError } from "./types";

/// Sort-6: given 6 random digits, output them sorted ascending.
///
/// Prompt format:  "sort: 5 1 4 2 6 3 = "
/// Expected:       "1 2 3 4 5 6"
///
/// This is the Karpathy minGPT algorithmic task — proven to converge
/// at ~100K-param models in minutes of browser-time training. The
/// score is exact-match accuracy over a 200-trial deterministic suite
/// (seeded). Higher is better.
///
/// Note: gallery models trained on Shakespeare / TinyStories / code
/// have never seen this prompt format and will score near 0%. The
/// benchmark exists for SUBMISSIONS — people train a tiny model
/// specifically for sort and submit. The pareto view "score per
/// params" rewards getting to 90%+ accuracy with the fewest params.
const N_TRIALS = 200;
const SEQ_LEN = 6;

function seedRandom(seed: number): () => number {
  // Mulberry32 — small deterministic PRNG.
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildTrials(): { prompt: string; expected: string }[] {
  const rng = seedRandom(0x517);
  const trials: { prompt: string; expected: string }[] = [];
  for (let i = 0; i < N_TRIALS; i++) {
    const digits: number[] = [];
    for (let j = 0; j < SEQ_LEN; j++) digits.push(Math.floor(rng() * 10));
    const sorted = digits.slice().sort((a, b) => a - b);
    trials.push({
      prompt: `sort: ${digits.join(" ")} = `,
      expected: sorted.join(" "),
    });
  }
  return trials;
}

export const sort6: Benchmark = {
  id: "sort-6",
  name: "Sort-6",
  description: "Sort 6 random digits. Greedy continuation, exact match (higher = better).",
  lowerIsBetter: false,
  requiresByteLevel: true,
  approxSeconds: 60,
  browserTrainable: true,

  async run(model: BenchmarkModel): Promise<BenchmarkResult> {
    if (model.vocabSize !== 256) {
      throw new BenchmarkError(
        "incompatible",
        `sort-6 currently scores byte-level models only`,
      );
    }
    const trials = buildTrials();
    let correct = 0;
    const failures: string[] = [];
    for (const { prompt, expected } of trials) {
      const continuation = await model.generate(prompt, expected.length + 2, 0);
      // Match the FIRST `expected.length` chars of the continuation after
      // trimming leading whitespace.
      const got = continuation.replace(/^\s+/, "").slice(0, expected.length);
      if (got === expected) correct += 1;
      else if (failures.length < 5) failures.push(`${prompt}→ "${got}" (expected "${expected}")`);
    }
    return {
      score: (correct / trials.length) * 100,
      details: { trials: trials.length, correct, failures },
    };
  },
};
