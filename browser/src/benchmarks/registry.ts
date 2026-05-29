import type { Benchmark } from "./types";
import { tinyStoriesPpl } from "./tinystories-ppl";
import { sort6 } from "./sort-6";
import { reverse16 } from "./reverse-16";

/// All benchmarks the leaderboard knows about. Add new ones here.
/// Manifest keys (`benchmarks.<id>`) match `Benchmark.id`.
///
/// Launch set:
///   - `tinystories-ppl` — universal language-modeling baseline,
///     scored via `tg_eval` from `score_gallery.mjs` (no browser
///     worker needed).
///   - `sort-6`, `reverse-16` — Karpathy minGPT algorithmic tasks.
///     Score via generation; requires the browser worker's
///     `generate` path or a task-aware Node scorer (see
///     `browser/score_gallery_tasks.mjs` — coming next ship).
export const benchmarks: readonly Benchmark[] = [
  tinyStoriesPpl,
  sort6,
  reverse16,
];

export function benchmarkById(id: string): Benchmark | undefined {
  return benchmarks.find((b) => b.id === id);
}
