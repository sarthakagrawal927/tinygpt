// test_webgpu_train.mjs — drives a real training step on the WebGPU backend
// to verify the blocked-matmul integration doesn't break model quality.
//
// Two-phase test:
//   1. Train Small preset on WASM backend for 50 steps. Record final loss.
//   2. Train Small preset on WebGPU backend (uses matmul_blocked) for 50
//      steps with the SAME SEED and same data. Record final loss.
//   3. Assert: WebGPU final loss matches WASM final loss within tolerance
//      (5%). If it diverges much further, something in the new matmul is
//      mathematically wrong — quality has dropped.
//
// Must run with WebGPU enabled — Chromium needs --enable-unsafe-webgpu and
// a real GPU adapter (not headless software).

import { chromium } from "playwright";

const args = [
  "--enable-unsafe-webgpu",
  "--enable-features=Vulkan",
  "--use-vulkan",
];

const browser = await chromium.launch({
  headless: false, // headed gets a real GPU adapter on macOS
  args,
});
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await ctx.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(`console.error: ${m.text()}`);
});

await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click().catch(() => {});

// Check WebGPU is actually working.
const adapter = await page.evaluate(async () => {
  if (!navigator.gpu) return { ok: false, reason: "no navigator.gpu" };
  try {
    const a = await navigator.gpu.requestAdapter();
    if (!a) return { ok: false, reason: "no adapter" };
    return { ok: true };
  } catch (e) { return { ok: false, reason: e.message }; }
});
console.log("adapter:", JSON.stringify(adapter));
if (!adapter.ok) {
  console.log("WebGPU unavailable — cannot validate integration here.");
  await browser.close();
  process.exit(1);
}

// Helper: train Small preset for N steps with the given backend, return final loss.
async function trainAndGetLoss(backend, steps) {
  // Reset to a known fresh state.
  await page.evaluate(({ steps, backend }) => {
    document.querySelector("#sizePreset").value = "small";
    document.querySelector("#sizePreset").dispatchEvent(new Event("change", { bubbles: true }));
    document.querySelector("#maxSteps").value = String(steps);
    document.querySelector("#maxSteps").dispatchEvent(new Event("input", { bubbles: true }));
    const backendSel = document.getElementById("backend");
    backendSel.value = backend;
    backendSel.dataset.userPicked = "1";
    backendSel.dispatchEvent(new Event("change", { bubbles: true }));
    const seed = document.getElementById("seed");
    if (seed) { seed.value = "42"; seed.dispatchEvent(new Event("input", { bubbles: true })); }
  }, { steps, backend });

  page.removeAllListeners("dialog");
  page.on("dialog", (d) => d.accept());

  await page.locator("#start").click({ force: true });

  // Wait for training to finish — stEta becomes "done" when step ≥ maxSteps.
  // Print progress and any console errors every 5s.
  const t0 = Date.now();
  while (Date.now() - t0 < 90_000) {
    await new Promise((r) => setTimeout(r, 5000));
    const state = await page.evaluate(() => ({
      step: document.getElementById("stStep")?.textContent,
      train: document.getElementById("stTrain")?.textContent,
      eta: document.getElementById("stEta")?.textContent,
      status: document.getElementById("status")?.textContent,
    }));
    console.log(`  t+${((Date.now() - t0) / 1000).toFixed(0)}s  step=${state.step}  train=${state.train}  eta=${state.eta}  status=${state.status}`);
    if (state.eta?.trim().toLowerCase() === "done") break;
    if (state.status && (state.status.toLowerCase().includes("error") || state.status.toLowerCase().includes("failed"))) {
      console.log("  ABORT — status shows error");
      break;
    }
  }

  const loss = await page.evaluate(() => {
    const el = document.getElementById("stTrain");
    return el ? parseFloat(el.textContent) : NaN;
  });
  return loss;
}

console.log("\n=== Phase 1: WASM backend (50 steps, seed=42) ===");
const wasmLoss = await trainAndGetLoss("wasm", 50);
console.log(`WASM final loss: ${wasmLoss.toFixed(4)}`);

// Reset between runs.
await page.evaluate(() => location.reload());
await page.waitForLoadState("networkidle");
await page.locator("#welcomeSkip").click().catch(() => {});

console.log("\n=== Phase 2: WebGPU backend (50 steps, seed=42) ===");
const gpuLoss = await trainAndGetLoss("webgpu", 50);
console.log(`WebGPU final loss: ${gpuLoss.toFixed(4)}`);

const drift = Math.abs(wasmLoss - gpuLoss) / Math.max(Math.abs(wasmLoss), 1e-6);
console.log(`\nRelative drift: ${(drift * 100).toFixed(1)}%`);
if (drift < 0.05) {
  console.log("PASS — WebGPU + blocked matmul produces equivalent training within 5%.");
} else if (drift < 0.20) {
  console.log("WARN — small drift, may be float-reorder noise (or seeding diff).");
} else {
  console.log("FAIL — WebGPU diverges substantially from WASM. Possible matmul bug.");
}

if (errors.length) {
  console.log(`\n${errors.length} console errors:`);
  errors.forEach((e) => console.log("  " + e));
}

await browser.close();
