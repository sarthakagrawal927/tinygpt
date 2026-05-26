/**
 * sizing.ts — model size presets, head-count derivation, and a pre-flight
 * time estimator calibrated to single-threaded WASM SIMD on this codebase.
 *
 * The estimator uses the CPU probe from runtime_detect.ts (a 160³ matmul timed
 * in plain JS) and a tuning constant fit to two known data points:
 *   0.36M params → ~0.4 s/step (~6 400 tok/s on a fast laptop, batch 16, ctx 64)
 *   1.30M params → ~2.6 s/step (~  985 tok/s, same)
 * Both correspond to cpuProbeMs ≈ 10 ms on the calibration laptop. Throughput
 * scales roughly as 1/params, giving K ≈ 1.7e9 tok·params/s at that reference,
 * and we scale linearly by the inverse of the probe.
 */

export interface Preset {
  id: string;
  label: string;
  layers: number;
  dModel: number;
  ctx: number;
  batch: number;
  maxSteps: number;
  recommendedBackend: "wasm" | "webgpu";
  note: string;
}

export const PRESETS: Preset[] = [
  {
    id: "tiny",
    label: "Tiny (~70k params)",
    layers: 2, dModel: 48, ctx: 32, batch: 16, maxSteps: 2000,
    recommendedBackend: "wasm",
    note: "finishes in well under a minute — best for kicking the tyres",
  },
  {
    id: "small",
    label: "Small (~360k params)",
    layers: 3, dModel: 96, ctx: 64, batch: 16, maxSteps: 1500,
    recommendedBackend: "wasm",
    note: "the default — a real GPT, comfortable on any laptop",
  },
  {
    id: "medium",
    label: "Medium (~830k params)",
    layers: 4, dModel: 128, ctx: 96, batch: 16, maxSteps: 1000,
    recommendedBackend: "wasm",
    note: "loss curve is more interesting; a few minutes",
  },
  {
    id: "large",
    label: "Large (~2.7M params)",
    layers: 6, dModel: 192, ctx: 128, batch: 12, maxSteps: 600,
    recommendedBackend: "wasm",
    note: "WASM is slow here — switch to WebGPU if your browser supports it",
  },
  {
    id: "xl",
    label: "XL (~6.4M params, GPU)",
    layers: 8, dModel: 256, ctx: 128, batch: 8, maxSteps: 400,
    recommendedBackend: "webgpu",
    note: "WebGPU recommended — WASM will take a long time at this size",
  },
  {
    id: "huge",
    label: "Huge (~10M params, GPU only)",
    layers: 12, dModel: 256, ctx: 256, batch: 8, maxSteps: 1500,
    recommendedBackend: "webgpu",
    note: "~15 minutes on M-series WebGPU — first preset where you can see word-shaped output",
  },
  {
    id: "massive",
    label: "Massive (~25M params, GPU only, ~40 min)",
    layers: 14, dModel: 384, ctx: 256, batch: 4, maxSteps: 1000,
    recommendedBackend: "webgpu",
    note: "the wow moment — locally grammatical output on a real dataset. Plan ~40 minutes",
  },
  {
    id: "mega",
    label: "Mega (~25M, ctx 512, ~1.5 h)",
    layers: 14, dModel: 384, ctx: 512, batch: 2, maxSteps: 800,
    recommendedBackend: "webgpu",
    note: "ctx 512 — the regime where attention dominates compute and Flash Attention would matter. Long-range coherence becomes possible. Plan ~1.5 h",
  },
];

/**
 * Pick a sensible head count for d_model. Targets head_dim ≈ 32 where possible
 * and falls back to 3 (the project's historical default) for the legacy
 * d_model values that aren't divisible by 32.
 */
const HEADS_BY_D: Record<number, number> = {
  48: 3, 64: 2, 96: 3, 128: 4, 144: 3, 192: 6, 256: 8, 384: 12,
};
export function headsFor(dModel: number): number {
  if (HEADS_BY_D[dModel]) return HEADS_BY_D[dModel];
  if (dModel % 32 === 0) return dModel / 32;
  if (dModel % 3 === 0) return 3;
  return 1;
}

/** Total params: byte embeddings + position embeddings + ~12·d² per layer. */
export function estimateParams(layers: number, dModel: number, ctx: number): number {
  return 256 * dModel + ctx * dModel + layers * 12 * dModel * dModel;
}

/**
 * Estimate tokens/sec for the WASM SIMD backend on this machine.
 * See header — derived from two empirical data points; expect ±2× accuracy.
 */
const CALIBRATION = 1.7e10; // K · ms · params for the reference machine
export function estimateTokensPerSec(params: number, cpuProbeMs: number): number {
  return CALIBRATION / (cpuProbeMs * Math.max(params, 1));
}

/**
 * Pre-flight estimate of total training time for the given config, in seconds.
 * Uses the WASM throughput estimate; WebGPU on a real GPU will likely be
 * faster (but in-browser WebGPU is unmeasured here, see docs/notes.md §10).
 */
export function estimateTrainSeconds(
  layers: number,
  dModel: number,
  ctx: number,
  batch: number,
  maxSteps: number,
  cpuProbeMs: number,
): number {
  const params = estimateParams(layers, dModel, ctx);
  const tps = estimateTokensPerSec(params, cpuProbeMs);
  const totalTokens = maxSteps * batch * ctx;
  return totalTokens / Math.max(tps, 1);
}

export function formatParams(p: number): string {
  if (p >= 1_000_000) return `${(p / 1_000_000).toFixed(p < 10_000_000 ? 1 : 0)}M`;
  if (p >= 1000) return `${(p / 1000).toFixed(p < 10_000 ? 1 : 0)}k`;
  return `${p}`;
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 90) return `${Math.round(seconds)} s`;
  const minutes = seconds / 60;
  if (minutes < 90) return `${minutes.toFixed(minutes < 10 ? 1 : 0)} min`;
  const hours = minutes / 60;
  return `${hours.toFixed(hours < 10 ? 1 : 0)} h`;
}
