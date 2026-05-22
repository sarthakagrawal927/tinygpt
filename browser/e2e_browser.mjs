// e2e_browser.mjs — end-to-end check of the browser app (Phase 4-5).
//
// Drives the real page in headless Chromium: loads it, starts a training run,
// and verifies that (1) training runs to completion in the Web Worker,
// (2) the loss actually falls, (3) sampling works, (4) the WebGPU matmul
// benchmark matches WASM (milestone 6 — skipped if headless has no WebGPU),
// (5) the trained model survives a page refresh (milestone 7), and (6) no
// console or page errors occurred.
//
// Run from browser/:
//   npm run build && npm run preview &   # build + serve the app
//   npm run e2e                          # this script
//
// BASE_URL overrides the default http://localhost:4173.

import { chromium } from "playwright";

const BASE = process.env.BASE_URL || "http://localhost:4173";
let failed = 0;
const check = (name, ok, detail) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name.padEnd(40)} (${detail})`);
  if (!ok) failed++;
};

const browser = await chromium.launch({
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
});
const page = await browser.newPage();

const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(`console: ${m.text()}`);
});

await page.goto(BASE, { waitUntil: "load" });
check("page loaded", true, BASE);

// Machine detection + model recommendation.
const caps = (await page.textContent("#caps")) || "";
check("machine detection + recommendation shown",
  /Your machine:/.test(caps) && /Suggested model:/.test(caps),
  caps.replace(/\s+/g, " ").trim().slice(0, 88));

// Keep the run short so the e2e is quick.
await page.fill("#maxSteps", "400");
await page.fill("#corpus", "the quick brown fox jumps over the lazy dog. ".repeat(60));

await page.click("#start");

// Training runs in the Worker; wait for the UI to report completion (or error).
await page.waitForFunction(
  () => {
    const s = document.getElementById("status")?.textContent || "";
    return s.includes("complete") || s.startsWith("error") || s.includes("worker error");
  },
  undefined,
  { timeout: 180000 },
);
const finalStatus = await page.textContent("#status");
check("training finished without error", !/error/i.test(finalStatus || ""), finalStatus);

const step = await page.textContent("#stStep");
const trainLoss = parseFloat((await page.textContent("#stTrain")) || "NaN");
const valLoss = parseFloat((await page.textContent("#stVal")) || "NaN");
check("training reached the step budget", step?.startsWith("400"), step);
check("train loss fell below 1.0", trainLoss < 1.0, trainLoss.toFixed(4));
check("val loss is a finite number", Number.isFinite(valLoss), valLoss.toFixed(4));

// Sampling from the trained model.
await page.click("#sample");
await page.waitForFunction(
  () => {
    const t = document.getElementById("output")?.textContent || "";
    return t.length > 0 && t !== "generating…";
  },
  undefined,
  { timeout: 30000 },
);
const sample = await page.textContent("#output");
check("generated a non-empty sample", (sample || "").length > 10, `${(sample || "").length} chars`);
console.log(`    sample: ${JSON.stringify((sample || "").slice(0, 80))}`);

// WebGPU matmul benchmark (milestone 6). Headless Chromium may lack WebGPU —
// that is reported as a skip, not a failure.
await page.click("#bench");
await page.waitForFunction(
  () => {
    const t = document.getElementById("benchOut")?.textContent || "";
    return t !== "Not run yet." && !t.endsWith("…");
  },
  undefined,
  { timeout: 90000 },
);
const bench = (await page.textContent("#benchOut")) || "";
if (/not available/i.test(bench)) {
  console.log(`skip WebGPU benchmark                       (no WebGPU in headless Chromium)`);
} else {
  check("WebGPU matmul matches WASM", /parity OK/.test(bench), bench.replace(/\n/g, " "));
}

// Checkpointing (milestone 7): the trained model survives a page refresh.
await page.reload({ waitUntil: "load" });
await page.waitForFunction(
  () => {
    const s = document.getElementById("status")?.textContent || "";
    return /restored/i.test(s) || s.startsWith("error");
  },
  undefined,
  { timeout: 30000 },
);
const restoredStatus = await page.textContent("#status");
check("model restored after page refresh", /restored/i.test(restoredStatus || ""), restoredStatus);
check("loss chart restored after refresh", (await page.textContent("#stStep")) === "400",
  await page.textContent("#stStep"));
await page.click("#sample");
await page.waitForFunction(
  () => {
    const t = document.getElementById("output")?.textContent || "";
    return t.length > 0 && t !== "generating…";
  },
  undefined,
  { timeout: 30000 },
);
check("restored model still generates", ((await page.textContent("#output")) || "").length > 10,
  "ok");

// The "Apply" button fills the config inputs with the recommended model.
await page.click("#applyRec");
const ctxAfter = await page.inputValue("#ctx");
const layersAfter = await page.inputValue("#layers");
check("Apply sets a recommended config",
  /^\d+$/.test(ctxAfter) && /^\d+$/.test(layersAfter),
  `ctx=${ctxAfter} layers=${layersAfter}`);

check("no console / page errors", errors.length === 0, errors.join(" | ") || "none");

await browser.close();
console.log(failed === 0 ? "\nbrowser e2e passed" : "\nBROWSER E2E FAILED");
process.exit(failed === 0 ? 0 : 1);
