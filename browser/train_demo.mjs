// train_demo.mjs — produce a real public/demo.tinygpt
//
// Steps:
//  1. Paste full TinyShakespeare into #corpus
//  2. Huge preset (12L, d=256, ctx=256, batch=8 — ~9.6M params)
//  3. WebGPU backend, 5000 steps, seed 42
//  4. Poll progress every 30s
//  5. When training completes, click #downloadModel and capture the file
//  6. Move it into browser/public/demo.tinygpt
//
// Expected wall time: ~45 min (5000 × ~536 ms/step measured in feasibility run)

import { chromium } from "playwright";
import { promises as fs } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const CORPUS_PATH = resolve(ROOT, "data/examples/shakespeare.txt");
const DEMO_OUT = resolve(__dirname, "public/demo.tinygpt");
const STEPS = 5000;
const PRESET = "huge";
const POLL_MS = 30_000;
const MAX_WALL_MS = 120 * 60 * 1000; // 2-hour hard cap

const corpus = await fs.readFile(CORPUS_PATH, "utf8");
console.log(`corpus: ${corpus.length} bytes from ${CORPUS_PATH}`);

const browser = await chromium.launch({
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan"],
});
const ctx = await browser.newContext({
  viewport: { width: 1400, height: 900 },
  acceptDownloads: true,
});
const page = await ctx.newPage();
page.on("dialog", (d) => d.accept().catch(() => {}));
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
page.on("console", (m) => {
  const t = m.type();
  if (t === "error") console.log(`[err]`, m.text());
});

await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click({ timeout: 1500 }).catch(() => {});

// Set corpus value via JS — bypasses fill's visibility/editable wait
// (textarea may be inside a hidden tab panel during init).
await page.evaluate((text) => {
  const el = document.getElementById("corpus");
  el.value = text;
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
}, corpus);
const corpusLen = await page.evaluate(() => document.getElementById("corpus").value.length);
console.log(`corpus set: ${corpusLen} chars in #corpus`);
if (corpusLen < corpus.length - 10) throw new Error(`corpus truncated: got ${corpusLen} expected ${corpus.length}`);

await page.evaluate(({ preset, steps }) => {
  const setVal = (id, v) => {
    const el = document.getElementById(id);
    el.value = String(v);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  };
  document.getElementById("sizePreset").value = preset;
  document.getElementById("sizePreset").dispatchEvent(new Event("change", { bubbles: true }));
  setVal("maxSteps", steps);
  // Browser default lr=3e-3 is 10x higher than the Python ref (3e-4) and
  // causes loss to plateau ~2.5 on char-level Shakespeare. Use the ref value.
  setVal("lr", 0.0003);
  const back = document.getElementById("backend");
  back.value = "webgpu";
  back.dataset.userPicked = "1";
  back.dispatchEvent(new Event("change", { bubbles: true }));
  const seed = document.getElementById("seed");
  if (seed) setVal("seed", 42);
}, { preset: PRESET, steps: STEPS });

const cfg = await page.evaluate(() => ({
  preset: document.getElementById("sizePreset").value,
  maxSteps: document.getElementById("maxSteps").value,
  backend: document.getElementById("backend").value,
  status: document.getElementById("status")?.textContent ?? "",
}));
console.log("config:", JSON.stringify(cfg));

const tStart = Date.now();
await page.locator("#start").click({ force: true });
console.log(`training started at ${new Date().toISOString()}`);

while (true) {
  await new Promise((r) => setTimeout(r, POLL_MS));
  const wall = Date.now() - tStart;
  const s = await page.evaluate(() => ({
    step: document.getElementById("stStep")?.textContent ?? "",
    train: document.getElementById("stTrain")?.textContent ?? "",
    val: document.getElementById("stVal")?.textContent ?? "",
    status: document.getElementById("status")?.textContent ?? "",
  }));
  const m = s.step.match(/^(\d+)\s*\/\s*(\d+)/);
  const cur = m ? Number(m[1]) : 0;
  const max = m ? Number(m[2]) : 0;
  console.log(
    `t+${(wall / 60000).toFixed(1)}min  step=${cur}/${max}  train=${s.train}  val=${s.val}  status="${s.status.slice(0, 80)}"`,
  );
  if (cur >= max && max > 0) { console.log("=== training complete ==="); break; }
  if (/error|failed/i.test(s.status)) { console.log("=== ERROR ==="); break; }
  if (wall > MAX_WALL_MS) { console.log("=== HARD-CAP ==="); break; }
}

console.log("\n--- saving checkpoint ---");
// Wait for the worker's "checkpoint" message to fire and main.ts to assign
// `latestState` — that's what enables the download button. The downloadModel
// click handler returns early ("nothing trained yet") if `latestState` is
// still null, so polling for the enabled state avoids a silent no-op click.
await page.waitForFunction(() => {
  const btn = document.getElementById("downloadModel");
  return btn && !btn.disabled;
}, null, { timeout: 60_000 }).catch(() => {
  console.log("WARN: #downloadModel never enabled — worker may not have posted checkpoint");
});

// #modelMenuBtn lives in a controls cluster that's visibility-toggled —
// Playwright's locator.click() rejects with "Element is not visible". Bypass
// the visibility check by invoking the DOM click handler directly. Same for
// #downloadModel, which lives inside the menu and only becomes interactable
// after the parent click.
try {
  await page.evaluate(() => document.getElementById("modelMenuBtn").click());
  await page.waitForTimeout(150);
  const [download] = await Promise.all([
    page.waitForEvent("download", { timeout: 120_000 }),
    page.evaluate(() => document.getElementById("downloadModel").click()),
  ]);
  const tmp = await download.path();
  console.log(`download arrived at ${tmp}; copying to ${DEMO_OUT}`);
  await fs.copyFile(tmp, DEMO_OUT);
  const stat = await fs.stat(DEMO_OUT);
  console.log(`demo.tinygpt: ${stat.size} bytes`);
  await browser.close();
  console.log("done.");
} catch (err) {
  console.log("\n=== DOWNLOAD FAILED — leaving browser open for manual rescue ===");
  console.log(`error: ${err?.message ?? err}`);
  console.log("The trained model is still in the page's memory.");
  console.log("In the open browser window:");
  console.log("  1. Click the 'Model ▾' button (right of Start)");
  console.log("  2. Click 'Download .tinygpt'");
  console.log(`  3. Save the file as: ${DEMO_OUT}`);
  console.log("Then quit this script with Ctrl+C.\n");
  // Keep the script (and browser) alive so the user can rescue the model.
  await new Promise(() => {}); // never resolves
}
