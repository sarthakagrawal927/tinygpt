/**
 * runtime_detect.ts — browser capability + hardware detection (Phase 4).
 *
 * Two jobs:
 *   1. detectCapabilities() — which compute backends the browser supports.
 *   2. detectHardware() + recommendModel() — inspect the user's machine and
 *      suggest a model size it can train comfortably *while they watch*.
 *
 * The honest signal for "how big a model can this machine train" is not the
 * core count — browser training is single-threaded WASM — it is raw CPU speed.
 * So detectHardware() runs a small timed matmul and the recommendation is keyed
 * off that measurement.
 *
 * Guide: docs/browser_notes.md ("WebGPU acceleration", "Browser facts")
 */

import type { Backend } from "./types";

export interface Capabilities {
  webgpu: boolean;
  wasmSimd: boolean;
  crossOriginIsolated: boolean; // needed only for threaded WASM
  /** Best backend that actually has a working kernel build today. */
  active: Backend;
}

/** WebAssembly SIMD support: validate a tiny module that uses a v128 opcode. */
function hasWasmSimd(): boolean {
  // Minimal module containing the `i32x4.splat` instruction.
  const probe = new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1,
    8, 0, 65, 0, 253, 17, 253, 98, 11,
  ]);
  try {
    return WebAssembly.validate(probe);
  } catch {
    return false;
  }
}

async function hasWebGpu(): Promise<boolean> {
  // WebGPU types are not in the standard DOM lib — probe with a minimal shape.
  const gpu = (navigator as unknown as {
    gpu?: { requestAdapter(): Promise<unknown> };
  }).gpu;
  if (!gpu) return false;
  try {
    return (await gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

/** Probe the browser. `active` is the backend the worker will really use. */
export async function detectCapabilities(): Promise<Capabilities> {
  const webgpu = await hasWebGpu();
  const wasmSimd = hasWasmSimd();
  return {
    webgpu,
    wasmSimd,
    crossOriginIsolated:
      typeof crossOriginIsolated !== "undefined" && crossOriginIsolated,
    // Only the scalar WASM kernel is built today — that is what runs.
    active: "wasm",
  };
}

// ===========================================================================
// Hardware detection + model-size recommendation
// ===========================================================================

export type MachineTier = "modest" | "standard" | "capable" | "strong";

export interface Hardware {
  cores: number; // navigator.hardwareConcurrency
  deviceMemoryGB: number | null; // navigator.deviceMemory — coarse, Chrome-only
  cpuProbeMs: number; // time for a fixed matmul — the real speed signal
}

export interface ModelRecommendation {
  tier: MachineTier;
  ctx: number;
  layers: number;
  dModel: number;
  maxSteps: number;
  approxParams: number;
  note: string;
}

/**
 * Time a fixed 160×160×160 matmul in plain JS. Browser training is
 * single-threaded, so this directly measures how fast a run will be — a far
 * more honest signal than core count or the (coarse, Chrome-only) memory hint.
 */
function probeCpuSpeed(): number {
  const n = 160;
  const a = new Float32Array(n * n);
  const b = new Float32Array(n * n);
  const c = new Float32Array(n * n);
  for (let i = 0; i < n * n; i++) {
    a[i] = Math.sin(i);
    b[i] = Math.cos(i);
  }
  let best = Infinity;
  for (let run = 0; run < 3; run++) {
    c.fill(0);
    const t0 = performance.now();
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < n; k++) {
        const av = a[i * n + k];
        for (let j = 0; j < n; j++) c[i * n + j] += av * b[k * n + j];
      }
    }
    best = Math.min(best, performance.now() - t0);
  }
  return best;
}

/** Inspect the machine: core count, the memory hint, and a CPU speed probe. */
export function detectHardware(): Hardware {
  const nav = navigator as Navigator & { deviceMemory?: number };
  return {
    cores: nav.hardwareConcurrency || 1,
    deviceMemoryGB: typeof nav.deviceMemory === "number" ? nav.deviceMemory : null,
    cpuProbeMs: probeCpuSpeed(),
  };
}

// Per-tier model configs — sized so a run finishes while the user watches.
// d_model values are multiples of 3 (the app uses 3 attention heads).
const TIERS: Record<MachineTier, Omit<ModelRecommendation, "tier" | "approxParams">> = {
  modest: { ctx: 32, layers: 2, dModel: 48, maxSteps: 2000,
    note: "overfits a tiny corpus in seconds" },
  standard: { ctx: 64, layers: 3, dModel: 96, maxSteps: 1500,
    note: "a real run in well under a minute" },
  capable: { ctx: 96, layers: 4, dModel: 96, maxSteps: 1200,
    note: "a real run in about a minute" },
  strong: { ctx: 128, layers: 5, dModel: 144, maxSteps: 800,
    note: "a ~1.3M-param run in a couple of minutes" },
};

/** Rough parameter count: embeddings + ~12·d² per transformer layer. */
function estimateParams(ctx: number, layers: number, d: number): number {
  return 256 * d + ctx * d + layers * 12 * d * d;
}

/** Map detected hardware to a model the browser can train while you watch. */
export function recommendModel(hw: Hardware): ModelRecommendation {
  const ms = hw.cpuProbeMs;
  let tier: MachineTier =
    ms < 8 ? "strong" : ms < 18 ? "capable" : ms < 40 ? "standard" : "modest";

  // Conservative downgrades on weak secondary signals.
  if (hw.cores <= 4 && tier === "strong") tier = "capable";
  if (hw.cores <= 2 && (tier === "strong" || tier === "capable")) tier = "standard";
  if (hw.deviceMemoryGB != null && hw.deviceMemoryGB <= 2) tier = "modest";

  const t = TIERS[tier];
  return { tier, ...t, approxParams: estimateParams(t.ctx, t.layers, t.dModel) };
}
