/// Browser-runnable benchmark suite for the TinyGPT leaderboard.
///
/// Each Benchmark takes a loaded model handle and produces a single
/// numeric score plus optional structured details. Lower scores are
/// better when `lowerIsBetter === true`. All evaluation is
/// deterministic given the same model and the same eval data; the
/// eval data is checked into `browser/public/benchmarks/`.
///
/// The same engine drives:
///   - "Run benchmark on this loaded model" from the gallery dialog
///   - The `score_gallery.mjs` CLI that pre-scores curated entries
///   - The community submission flow's auto-eval step

/// One probe-able handle to the loaded model. The benchmark engine
/// uses two things only: a tokenizer pair (encode/decode) and a logits
/// function for a full prompt.
export interface BenchmarkModel {
  /// Vocabulary size — needed for softmax over the full logits vector.
  /// Browser-trained models are byte-level (256); BPE models report
  /// their real vocab. Benchmarks that assume byte-level (e.g., the
  /// holdout file is raw bytes) check this and skip if mismatched.
  vocabSize: number;

  /// Encode a UTF-8 string into token ids.
  encode(text: string): number[];

  /// Decode token ids back to a UTF-8 string.
  decode(ids: number[]): string;

  /// Forward pass: token ids `[T]` → logits `[T, vocab]`.
  ///
  /// Returns the FULL logits sequence (not just the last position),
  /// so perplexity-style benchmarks can score every position in one
  /// shot. The Float32Array is row-major: position `t`'s logits start
  /// at index `t * vocabSize`.
  ///
  /// For long sequences (T > context length), the engine chunks the
  /// input internally; benchmarks see one logical Float32Array.
  forwardLogits(ids: number[]): Promise<Float32Array>;

  /// Greedy / temperature sampling, matching the playground's
  /// generate path. Used by task-based benchmarks (sort, reverse,
  /// completion) that score by exact-match on the produced text.
  generate(prompt: string, maxNewTokens: number, temperature: number): Promise<string>;

  /// The model's context-length cap; needed so the engine can chunk
  /// long perplexity sequences appropriately.
  contextLength: number;
}

/// A single benchmark definition. Registered in `registry.ts`.
export interface Benchmark {
  /// Stable id — used as the manifest key for stored scores
  /// (`benchmarks.<id>` in `manifest.json`).
  id: string;
  /// Human-readable name shown on leaderboard rows.
  name: string;
  /// One-line description shown next to the name.
  description: string;
  /// Lower score is better (perplexity, error rate) or higher score
  /// is better (accuracy, BLEU). Drives the sort direction.
  lowerIsBetter: boolean;
  /// Whether the benchmark requires byte-level vocab. Most do today;
  /// BPE benchmarks would set false. Used by the engine to skip
  /// incompatible models with a clear reason instead of failing.
  requiresByteLevel: boolean;
  /// Approximate wall-clock time on a 9.6M-param browser-trained
  /// model — shown as "ETA" in the UI so users know roughly how long
  /// to wait when they click "Run".
  approxSeconds: number;
  /// Whether browser-only training can plausibly produce a good
  /// score. Used by leaderboard filters: "show only the benchmarks
  /// that browser-trained submissions can meaningfully compete on".
  browserTrainable: boolean;
  /// Run the benchmark. The engine times the call, wraps any error
  /// into a `BenchmarkError`, and persists `score` + `details`.
  run(model: BenchmarkModel): Promise<BenchmarkResult>;
}

export interface BenchmarkResult {
  /// The single number that drives leaderboard ranking. Interpreted
  /// against `Benchmark.lowerIsBetter`.
  score: number;
  /// Optional structured detail blob — surfaces as collapsed JSON in
  /// the leaderboard row's "details" expander. Keep small (< 4 KB).
  details?: Record<string, unknown>;
  /// Wall-clock seconds spent running. Engine-filled, but benchmarks
  /// can override (e.g., to exclude a one-time fetch cost).
  wallSeconds?: number;
}

/// Engine-thrown wrapper when a benchmark fails or isn't applicable.
export class BenchmarkError extends Error {
  /// `"incompatible"` (skip, don't penalize) vs `"failed"` (real
  /// failure, show in UI as red). Used by the manifest writer:
  /// incompatible → `null`, failed → don't store anything (try again).
  readonly kind: "incompatible" | "failed";
  constructor(kind: "incompatible" | "failed", message: string) {
    super(message);
    this.kind = kind;
  }
}
