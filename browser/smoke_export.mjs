// smoke_export.mjs — validate the WebGPU checkpoint-export path end to end
// before committing to a 60-minute training run. Trains a Small model for
// ~30 seconds, then exercises the same download flow train_demo.mjs uses.
// Saves to /tmp/tinygpt-smoke.tinygpt — does NOT touch public/demo.tinygpt.

import { chromium } from "playwright";
import { promises as fs } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const CORPUS_PATH = resolve(ROOT, "data/examples/shakespeare.txt");
const SMOKE_OUT = "/tmp/tinygpt-smoke.tinygpt";
const STEPS = 50; // ~30s on Small preset

const corpus = await fs.readFile(CORPUS_PATH, "utf8");
console.log(`smoke test: ${STEPS} steps Small preset on Shakespeare (${corpus.length} bytes)`);

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
  if (m.type() === "error") console.log("[err]", m.text());
});

await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click({ timeout: 1500 }).catch(() => {});

await page.evaluate((text) => {
  const el = document.getElementById("corpus");
  el.value = text;
  el.dispatchEvent(new Event("input", { bubbles: true }));
}, corpus);

await page.evaluate((steps) => {
  const setVal = (id, v) => {
    const el = document.getElementById(id);
    el.value = String(v);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  };
  document.getElementById("sizePreset").value = "small";
  document.getElementById("sizePreset").dispatchEvent(new Event("change", { bubbles: true }));
  setVal("maxSteps", steps);
  setVal("lr", 0.0003);
  const back = document.getElementById("backend");
  back.value = "webgpu";
  back.dataset.userPicked = "1";
  back.dispatchEvent(new Event("change", { bubbles: true }));
  const seed = document.getElementById("seed");
  if (seed) setVal("seed", 42);
}, STEPS);

const tStart = Date.now();
await page.locator("#start").click({ force: true });
console.log("training started…");

// Poll until step counter catches up OR download button becomes enabled.
const result = await page.waitForFunction(
  (target) => {
    const stStep = document.getElementById("stStep")?.textContent ?? "";
    const m = stStep.match(/^(\d+)\s*\/\s*(\d+)/);
    const btn = document.getElementById("downloadModel");
    const cur = m ? Number(m[1]) : 0;
    const max = m ? Number(m[2]) : 0;
    if (btn && !btn.disabled) return { ok: true, reason: "button-enabled", cur, max };
    if (cur >= max && max > 0 && cur > 0) return { ok: false, reason: "done-but-no-checkpoint", cur, max };
    return false;
  },
  STEPS,
  { timeout: 120_000, polling: 500 },
).then((h) => h.jsonValue());

const elapsed = Date.now() - tStart;
console.log(`training+checkpoint wait: ${(elapsed / 1000).toFixed(1)}s`);
console.log(`result:`, result);

if (!result.ok) {
  console.log("FAIL: training completed but checkpoint never arrived. WebGPU exportState is broken.");
  await browser.close();
  process.exit(1);
}

// Try the download
console.log("\nattempting download…");
await page.evaluate(() => document.getElementById("modelMenuBtn").click());
await page.waitForTimeout(150);
const [download] = await Promise.all([
  page.waitForEvent("download", { timeout: 30_000 }),
  page.evaluate(() => document.getElementById("downloadModel").click()),
]).catch((err) => {
  console.log("FAIL on download:", err.message);
  return [null];
});

if (!download) {
  console.log("FAIL: download event never fired");
  await browser.close();
  process.exit(1);
}

const tmp = await download.path();
await fs.copyFile(tmp, SMOKE_OUT);
const stat = await fs.stat(SMOKE_OUT);
console.log(`\nPASS — saved ${stat.size} bytes to ${SMOKE_OUT}`);
console.log("File contents:");
const head = await fs.readFile(SMOKE_OUT);
const magic = String.fromCharCode(...head.slice(0, 4));
const version = head.readUInt32LE(4);
const headerLen = head.readUInt32LE(8);
const headerJson = JSON.parse(head.slice(12, 12 + headerLen).toString("utf8"));
console.log(`  magic="${magic}" version=${version} headerLen=${headerLen}`);
console.log(`  config:`, JSON.stringify(headerJson.config));
console.log(`  finalLoss=${headerJson.finalLoss}  savedAt=${headerJson.savedAt}`);
console.log(`  manifest entries: ${headerJson.manifest?.length ?? "—"}`);
console.log(`  loss history points: ${headerJson.lossHistory?.length ?? 0}`);
console.log(`  state bytes: ${stat.size - 12 - headerLen}`);
await browser.close();
console.log("\nWebGPU exportState works end-to-end. Safe to run train_demo.mjs.");
