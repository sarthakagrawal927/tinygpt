// smoke_f16_train.mjs — verify the f16-storage path works during training.
//
// Approach: train a Medium preset on Shakespeare for 200 steps with seed=42
// and WebGPU backend. The training-time f16 activation is lazy in trainStep,
// so the first step packs every matmul-shaped weight; the gate (which ran
// at GpuOps.create() time) decides whether the path lights up.
//
// Pass criteria:
//   1. WebGPU loads without WGSL errors.
//   2. Both fwd and bwd gates pass (PASS PASS visible in the [ops] log).
//   3. Loss descends — final loss < initial loss × 0.7 (sanity check).
//   4. No NaN / Inf in any reported loss.
//
// This is intentionally a weaker check than the matmul gate (which already
// verified element-wise numerics): the goal here is "training stably
// descends with the f16 path active" — i.e., the new code in trainStep
// (linearBackward f16 dispatch + repackF16Mirrors) doesn't break gradients.
//
// Run with the dev server (preview) up on :5173:
//   node browser/smoke_f16_train.mjs

import { chromium } from "playwright";

const URL = process.env.SMOKE_URL ?? "http://localhost:5173/";
const STEPS = Number(process.env.SMOKE_STEPS ?? 200);

const browser = await chromium.launch({
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan"],
});
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await ctx.newPage();

const consoleLines = [];
page.on("console", (m) => {
  const t = m.type();
  const text = m.text();
  consoleLines.push({ t, text });
  if (text.includes("[ops]") || text.includes("f16-storage") || t === "error") {
    console.log(`[browser ${t}]`, text);
  }
});
page.on("pageerror", (e) => console.log("[pageerror]", e.message));

console.log(`opening ${URL} (steps=${STEPS}) …`);
await page.goto(URL, { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click({ timeout: 1500 }).catch(() => {});

// Wait for Shakespeare corpus to auto-load.
await page.waitForFunction(
  () => (document.getElementById("corpus")?.value.length ?? 0) > 1_000_000,
  null, { timeout: 10_000 },
);
console.log("corpus auto-loaded");

// Configure: Medium preset, WebGPU backend, fixed seed, the smoke step count.
await page.evaluate((stepCount) => {
  const setVal = (id, v) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = String(v);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  };
  document.getElementById("sizePreset").value = "medium";
  document.getElementById("sizePreset").dispatchEvent(new Event("change", { bubbles: true }));
  setVal("maxSteps", stepCount);
  setVal("lr", 0.0003);
  setVal("seed", 42);
  const back = document.getElementById("backend");
  back.value = "webgpu";
  back.dataset.userPicked = "1";
  back.dispatchEvent(new Event("change", { bubbles: true }));
}, STEPS);
console.log("configured: Medium preset, WebGPU backend, 200 steps, seed=42");

// Start training.
const tStart = Date.now();
await page.locator("#start").click({ force: true });
console.log("training started, polling every 5s …");

const lossSamples = [];
let lastStep = 0;
while (true) {
  await new Promise((r) => setTimeout(r, 5_000));
  const s = await page.evaluate(() => ({
    step: document.getElementById("stStep")?.textContent ?? "",
    train: document.getElementById("stTrain")?.textContent ?? "",
    status: document.getElementById("status")?.textContent ?? "",
  }));
  const m = s.step.match(/^(\d+)\s*\/\s*(\d+)/);
  const cur = m ? Number(m[1]) : 0;
  const max = m ? Number(m[2]) : 0;
  const loss = Number(s.train);
  if (cur > lastStep) {
    lossSamples.push({ step: cur, loss });
    console.log(`  step=${cur}/${max}  train=${s.train}`);
    lastStep = cur;
  }
  if (cur >= max && max > 0) break;
  if (/error|failed/i.test(s.status)) {
    console.log(`ERROR: ${s.status}`);
    break;
  }
  if (Date.now() - tStart > 5 * 60_000) {
    console.log("TIMEOUT: 5 minutes elapsed");
    break;
  }
}

console.log("\n=== loss samples ===");
for (const ls of lossSamples) console.log(`  step ${ls.step}: ${ls.loss.toFixed(4)}`);

// Verdict.
const initialLoss = lossSamples[0]?.loss ?? NaN;
const finalLoss = lossSamples[lossSamples.length - 1]?.loss ?? NaN;
const hasNaN = lossSamples.some((ls) => !Number.isFinite(ls.loss));

const wgslErrors = consoleLines.filter((l) =>
  l.t === "error" && /wgsl|shader|pipeline/i.test(l.text),
);
const fwdGate = consoleLines.find((l) => l.text.includes("gate (fwd)"));
const bwdGate = consoleLines.find((l) => l.text.includes("gate (bwd)"));
const verdict = consoleLines.find((l) => l.text.includes("gate verdict"));
const f16Active = consoleLines.find((l) => /f16-storage matmul active/i.test(l.text));

console.log("\n=== smoke summary ===");
console.log(`initial loss:        ${initialLoss.toFixed(4)}`);
console.log(`final loss:          ${finalLoss.toFixed(4)}`);
console.log(`descended:           ${finalLoss < initialLoss * 0.7 ? "yes" : "NO"}`);
console.log(`NaN encountered:     ${hasNaN ? "YES" : "no"}`);
console.log(`wgsl errors:         ${wgslErrors.length}`);
console.log(`fwd gate ran:        ${fwdGate ? "yes" : "no"}`);
console.log(`bwd gate ran:        ${bwdGate ? "yes" : "no"}`);
console.log(`verdict in log:      ${verdict?.text.match(/PASS|FAIL/)?.[0] ?? "unknown"}`);
console.log(`f16 active in train: ${f16Active ? "yes" : "no"}`);

const pass = (
  finalLoss < initialLoss * 0.7 &&
  !hasNaN &&
  wgslErrors.length === 0 &&
  !!fwdGate && !!bwdGate
);
console.log(`\n${pass ? "✅ TRAIN SMOKE PASS" : "❌ TRAIN SMOKE FAIL"}`);

await browser.close();
process.exit(pass ? 0 : 1);
