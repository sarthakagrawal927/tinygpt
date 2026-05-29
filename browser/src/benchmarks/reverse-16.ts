import type { Benchmark, BenchmarkModel, BenchmarkResult } from "./types";
import { BenchmarkError } from "./types";

/// Reverse-16: given a string ≤ 16 lowercase chars, output it reversed.
///
/// Prompt format:  "reverse: hello = "
/// Expected:       "olleh"
///
/// More dependency-tail than sort-6 (the last character of the input
/// becomes the first of the output, so the model has to "remember"
/// across the equals sign). Same Karpathy algorithmic-task family.
const N_TRIALS = 200;
const MIN_LEN = 4;
const MAX_LEN = 16;
const ALPHABET = "abcdefghijklmnopqrstuvwxyz";

function seedRandom(seed: number): () => number {
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
  const rng = seedRandom(0x3EE5);
  const trials: { prompt: string; expected: string }[] = [];
  for (let i = 0; i < N_TRIALS; i++) {
    const len = MIN_LEN + Math.floor(rng() * (MAX_LEN - MIN_LEN + 1));
    let s = "";
    for (let j = 0; j < len; j++) {
      s += ALPHABET[Math.floor(rng() * ALPHABET.length)];
    }
    trials.push({
      prompt: `reverse: ${s} = `,
      expected: s.split("").reverse().join(""),
    });
  }
  return trials;
}

export const reverse16: Benchmark = {
  id: "reverse-16",
  name: "Reverse-16",
  description: "Reverse a string ≤16 chars. Greedy continuation, exact match (higher = better).",
  lowerIsBetter: false,
  requiresByteLevel: true,
  approxSeconds: 90,
  browserTrainable: true,

  async run(model: BenchmarkModel): Promise<BenchmarkResult> {
    if (model.vocabSize !== 256) {
      throw new BenchmarkError(
        "incompatible",
        `reverse-16 currently scores byte-level models only`,
      );
    }
    const trials = buildTrials();
    let correct = 0;
    const failures: string[] = [];
    for (const { prompt, expected } of trials) {
      const continuation = await model.generate(prompt, expected.length + 2, 0);
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
