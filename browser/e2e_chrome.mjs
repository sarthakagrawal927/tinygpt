// e2e_chrome.mjs — exhaustive Chrome end-to-end test of tinygpt.sarthakagrawal.dev.
// Reads as the "everything works" gate. Each block is an independent check;
// failures are reported but don't stop the run, so a single broken thing
// doesn't mask the others.
//
// Run: node browser/e2e_chrome.mjs

import { chromium } from "playwright";

const SITE = process.env.E2E_URL || "https://tinygpt.sarthakagrawal.dev";

const results = [];
const ok = (n, msg = "") => { results.push({ name: n, ok: true, msg }); console.log(`✅ ${n}${msg ? "  — " + msg : ""}`); };
const fail = (n, msg) => { results.push({ name: n, ok: false, msg }); console.log(`❌ ${n}  — ${msg}`); };

const browser = await chromium.launch({
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan"],
});
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 }, acceptDownloads: true });
const page = await ctx.newPage();
page.on("dialog", (d) => d.accept().catch(() => {}));
const pageErrors = [];
const consoleErrors = [];
page.on("pageerror", (e) => pageErrors.push(e.message));
page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });

console.log(`\n>> E2E ${SITE}\n${"=".repeat(60)}\n`);

// ----- 1. Initial page load -----
console.log("PHASE 1 — Initial page load\n");
try {
  const resp = await page.goto(SITE, { waitUntil: "networkidle", timeout: 30_000 });
  if (resp?.status() === 200) ok("page loads HTTP 200");
  else fail("page loads HTTP 200", `got ${resp?.status()}`);
} catch (e) {
  fail("page loads", e.message);
  await browser.close();
  process.exit(1);
}
await page.locator("#welcomeSkip").click({ timeout: 2000 }).catch(() => {});

// ----- 2. Critical UI elements present + correct copy -----
console.log("\nPHASE 2 — UI elements + copy\n");
const introVisible = await page.locator("#introCard").isVisible().catch(() => false);
introVisible ? ok("intro card visible on first-visit") : fail("intro card", "not visible");

const introText = await page.locator("#introCard").textContent().catch(() => "");
if (introText?.includes("What is this?") && introText.includes("language model")) ok("intro copy correct");
else fail("intro copy", "missing 'What is this?' or 'language model'");

const bannerVisible = await page.locator("#demoBanner").isVisible().catch(() => false);
bannerVisible ? ok("demo banner visible") : fail("demo banner", "not visible");
const bannerText = await page.locator("#demoBanner").textContent().catch(() => "");
if (bannerText?.includes("Two ways") && bannerText.includes("Load pretrained")) ok("banner copy correct");
else fail("banner copy", "missing 'Two ways' or 'Load pretrained'");

// ----- 3. Corpus auto-loads (with idle-callback delay) -----
console.log("\nPHASE 3 — Default corpus lazy-loads\n");
await page.waitForTimeout(3500); // give requestIdleCallback time to fire
const corpusLen = await page.evaluate(() => document.getElementById("corpus")?.value.length ?? 0);
if (corpusLen > 1_000_000) ok("Shakespeare auto-fetched", `${corpusLen} chars`);
else fail("Shakespeare auto-fetched", `only ${corpusLen} chars`);

// ----- 4. Demo file size + reachability -----
console.log("\nPHASE 4 — Demo model file\n");
const demoMeta = await page.evaluate(async () => {
  const r = await fetch("/demo.tinygpt");
  if (!r.ok) return { ok: false, size: 0 };
  const buf = await r.arrayBuffer();
  return { ok: true, size: buf.byteLength };
});
if (demoMeta.ok) ok("/demo.tinygpt fetchable");
else fail("/demo.tinygpt fetchable", "404 or network failure");
const sizeMB = demoMeta.size / 1024 / 1024;
if (sizeMB > 0 && sizeMB < 25) ok("demo size under CF cap", `${sizeMB.toFixed(1)} MB`);
else fail("demo size under CF cap", `${sizeMB.toFixed(1)} MB`);

// ----- 5. Load pretrained → auto-switch to Watch + Generate focused -----
console.log("\nPHASE 5 — Load pretrained flow\n");
const tLoad = Date.now();
await page.locator("#loadDemoBtn").click({ force: true });
await page.locator("#demoBanner").waitFor({ state: "hidden", timeout: 90_000 }).catch(() => {});
const loadMs = Date.now() - tLoad;
if (loadMs < 90_000) ok("pretrained model loads", `${(loadMs / 1000).toFixed(1)}s`);
else fail("pretrained model loads", `timeout at ${loadMs}ms`);

await page.waitForTimeout(800); // settle for auto-switch
const screenState = await page.evaluate(() => ({
  active: document.getElementById("screens")?.getAttribute("data-active"),
  focused: document.activeElement?.id ?? "",
  watchTabEnabled: !document.querySelector(".screen-tab[data-screen='watch']")?.disabled,
}));
if (screenState.active === "watch") ok("auto-switched to Watch screen");
else fail("auto-switched to Watch screen", `data-active=${screenState.active}`);
if (screenState.focused === "sample") ok("Generate button focused");
else fail("Generate button focused", `activeElement=#${screenState.focused}`);
if (screenState.watchTabEnabled) ok("Watch tab enabled");
else fail("Watch tab enabled", "still disabled");

// ----- 6. Generate produces text + real tok/s + real TTFT -----
console.log("\nPHASE 6 — Generate from pretrained\n");
const tGen = Date.now();
await page.evaluate(() => document.getElementById("sample")?.click());

let firstByteMs = 0;
await page.waitForFunction(() => {
  const out = document.getElementById("output");
  return out && !out.classList.contains("empty") && out.textContent.length > 10;
}, null, { timeout: 60_000 }).then(() => { firstByteMs = Date.now() - tGen; }).catch(() => {});

await page.waitForTimeout(12_000);
const output = await page.evaluate(() => document.getElementById("output")?.textContent ?? "");
if (output.length > 40) ok("Generate produced text", `${output.length} chars, first byte ${firstByteMs}ms`);
else fail("Generate produced text", `only ${output.length} chars`);
console.log(`\n   sample preview: ${output.slice(0, 200)}\n`);

const stats = await page.evaluate(() => document.getElementById("sampleStats")?.textContent ?? "");
if (stats.includes("tok/s")) ok("tok/s appears in stats");
else fail("tok/s appears in stats", "missing");
if (stats.match(/\d+\s*ms\s*to first token/) || stats.match(/\d+\.\d+\s*ms/)) ok("TTFT is a real number (not n/a)", stats.replace(/\s+/g, " ").trim());
else if (stats.includes("n/a")) fail("TTFT real (not n/a)", "still says n/a — WebGPU path not being used for loaded model");
else fail("TTFT real", stats.slice(0, 80));

// ----- 7. Back to Setup → state intact -----
console.log("\nPHASE 7 — Navigate back to Setup\n");
await page.locator(".screen-tab[data-screen='setup']").click({ force: true }).catch(() => {});
await page.waitForTimeout(600);
const backState = await page.evaluate(() => ({
  active: document.getElementById("screens")?.getAttribute("data-active"),
  corpusFilled: (document.getElementById("corpus")?.value.length ?? 0) > 100_000,
  startVisible: document.getElementById("start")?.offsetWidth > 0,
}));
if (backState.active === "setup") ok("returned to Setup screen");
else fail("returned to Setup", `data-active=${backState.active}`);
if (backState.corpusFilled) ok("corpus still populated after switch");
else fail("corpus still populated", "lost");
if (backState.startVisible) ok("Start button visible after switch back");
else fail("Start button visible", "hidden / removed");

// ----- 8. Behemoth preset is blocked with a helpful error -----
console.log("\nPHASE 8 — Behemoth preset is blocked\n");
let confirmFired = false;
let alertFired = false;
page.removeAllListeners("dialog");
page.on("dialog", (d) => {
  if (d.type() === "alert" && (d.message().includes("memory.grow") || d.message().includes("racing") || d.message().includes("WASM heap"))) {
    alertFired = true;
  }
  confirmFired = d.type() === "confirm";
  void d.dismiss();
});
await page.evaluate(() => {
  const sel = document.getElementById("sizePreset");
  if (sel) { sel.value = "behemoth"; sel.dispatchEvent(new Event("change", { bubbles: true })); }
});
await page.waitForTimeout(400);
await page.evaluate(() => document.getElementById("start")?.click());
await page.waitForTimeout(1200);
if (alertFired) ok("Behemoth blocked with WASM-heap warning");
else if (confirmFired) fail("Behemoth blocked", "got a confirm() not the heap-warning alert — old code path");
else fail("Behemoth blocked", "no dialog fired");

// Restore Small preset for the next test
await page.evaluate(() => {
  const sel = document.getElementById("sizePreset");
  if (sel) { sel.value = "small"; sel.dispatchEvent(new Event("change", { bubbles: true })); }
});

// Re-arm dialog handler to auto-accept any further dialogs
page.removeAllListeners("dialog");
page.on("dialog", (d) => d.accept().catch(() => {}));

// ----- 9. Train a Small preset from scratch — loss starts at ~5.55 -----
console.log("\nPHASE 9 — Train Small preset from scratch\n");
await page.evaluate(() => {
  const setVal = (id, v) => {
    const el = document.getElementById(id);
    if (el) { el.value = String(v); el.dispatchEvent(new Event("input", { bubbles: true })); el.dispatchEvent(new Event("change", { bubbles: true })); }
  };
  setVal("maxSteps", 30);   // short run so the test stays fast
  setVal("lr", 0.0003);
  const back = document.getElementById("backend");
  if (back) { back.value = "webgpu"; back.dataset.userPicked = "1"; back.dispatchEvent(new Event("change", { bubbles: true })); }
});
await page.waitForTimeout(200);
await page.evaluate(() => document.getElementById("start")?.click());
// Wait specifically for a progress message from the NEW run — loading the
// pretrained model set stStep to "5000 / 5000" with stTrain=1.22 from its
// saved history, so a naive >=1 wait would match that stale state. The new
// run's maxSteps is 30, so look for that as the denominator.
await page.waitForFunction(() => {
  const stStep = document.getElementById("stStep")?.textContent ?? "";
  const m = stStep.match(/^(\d+)\s*\/\s*(\d+)/);
  return m && Number(m[1]) >= 1 && Number(m[2]) === 30;
}, null, { timeout: 30_000 }).catch(() => {});

const firstLoss = await page.evaluate(() => parseFloat(document.getElementById("stTrain")?.textContent ?? "NaN"));
if (!Number.isFinite(firstLoss)) fail("first-step loss is a number", `got ${firstLoss}`);
else if (firstLoss > 1 && firstLoss < 10) ok("first-step loss in expected range", `${firstLoss.toFixed(3)} (random init for char-level)`);
else fail("first-step loss in expected range", `${firstLoss.toFixed(3)} — expected ~5.55, training is producing garbage`);

await page.waitForFunction(() => {
  const stStep = document.getElementById("stStep")?.textContent ?? "";
  const m = stStep.match(/^(\d+)\s*\/\s*(\d+)/);
  return m && Number(m[1]) >= Number(m[2]) && Number(m[2]) > 0;
}, null, { timeout: 60_000 }).catch(() => {});
const finalLoss = await page.evaluate(() => parseFloat(document.getElementById("stTrain")?.textContent ?? "NaN"));
if (Number.isFinite(finalLoss) && finalLoss < firstLoss) ok("loss descended during training", `${firstLoss.toFixed(2)} → ${finalLoss.toFixed(2)}`);
else fail("loss descended", `start ${firstLoss}, end ${finalLoss}`);

// Live-sample card should have appeared if training was long enough
const liveSampleSeen = await page.evaluate(() => {
  const card = document.getElementById("liveSampleCard");
  return card && !card.hidden && (card.textContent?.trim().length ?? 0) > 0;
});
liveSampleSeen ? ok("live sample card appeared during training") : console.log("ℹ live sample didn't fire (30 steps may be too short — interval is min 100)");

// ----- 10. /docs index + each doc renders -----
console.log("\nPHASE 10 — Docs library\n");
const docsResp = await page.goto(`${SITE}/docs`, { waitUntil: "domcontentloaded", timeout: 15_000 }).catch(() => null);
docsResp?.status() === 200 ? ok("/docs index loads") : fail("/docs index loads", `HTTP ${docsResp?.status()}`);
const docCards = await page.locator(".doc-card").count();
docCards === 6 ? ok("/docs lists 6 cards") : fail("/docs lists 6 cards", `found ${docCards}`);

for (const slug of ["lessons", "qa_log", "decision_log", "study_guide", "annotated_transcript", "session_retrospective"]) {
  const r = await page.goto(`${SITE}/docs/${slug}`, { waitUntil: "domcontentloaded", timeout: 15_000 }).catch(() => null);
  const h1Visible = r?.status() === 200 ? await page.locator(".prose h1").isVisible().catch(() => false) : false;
  h1Visible ? ok(`/docs/${slug} renders`) : fail(`/docs/${slug} renders`, `HTTP ${r?.status()}`);
}

// ----- 11. Other key routes -----
console.log("\nPHASE 11 — Other routes\n");
for (const path of ["/speedup", "/devlog", "/roadmap"]) {
  const r = await page.goto(`${SITE}${path}`, { waitUntil: "domcontentloaded", timeout: 15_000 }).catch(() => null);
  r?.status() === 200 ? ok(`${path} loads`) : fail(`${path} loads`, `HTTP ${r?.status()}`);
}

// ----- 12. Roadmap has 4 planned levers (gallery, diverse data, quant+LoRA, Mac) -----
console.log("\nPHASE 12 — Roadmap state\n");
const plannedCount = await page.evaluate(() => document.querySelectorAll(".status.planned").length);
plannedCount === 4 ? ok("4 planned levers on /roadmap") : fail("4 planned levers", `found ${plannedCount}`);

// ----- 13. No page errors / console errors -----
console.log("\nPHASE 13 — Error budget\n");
pageErrors.length === 0 ? ok("no page errors") : fail("no page errors", pageErrors.slice(0, 3).join(" | "));
consoleErrors.length === 0 ? ok("no console errors") : fail("no console errors", consoleErrors.slice(0, 3).join(" | "));

// ----- Summary -----
const passed = results.filter((r) => r.ok).length;
console.log(`\n${"=".repeat(60)}`);
console.log(`${passed} / ${results.length} checks passed`);
if (passed < results.length) {
  console.log(`\nFailures:`);
  for (const r of results.filter((r) => !r.ok)) console.log(`  ❌ ${r.name} — ${r.msg}`);
}

await browser.close();
process.exit(passed === results.length ? 0 : 1);
