/**
 * runtime_detect.ts — browser capability detection (Phase 4).
 *
 * Reports which compute backends the browser could use. The current build ships
 * only the scalar WASM kernel, so training always runs on "wasm"; this module
 * still probes WebGPU and WASM-SIMD so the UI capability panel can show what an
 * accelerated build (milestones 5 SIMD / 6 WebGPU) would unlock.
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
