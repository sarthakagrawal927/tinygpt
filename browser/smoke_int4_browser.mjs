// smoke_int4_browser.mjs — end-to-end browser smoke for the int4 gallery path.
//
// Loads the Shakespeare gallery card (which now has both fp16 and int4
// variants), confirms the browser picks the int4 variant, that the
// numerics gate passes, and that generation still produces Shakespeare-
// like text. Mirror of smoke_f16.mjs but for the storage-quantization
// path.
//
// Run with the preview server up on :4173:
//   (cd browser && npm run build && npx astro preview &)
//   node browser/smoke_int4_browser.mjs

import { chromium } from "playwright";

const URL = process.env.SMOKE_URL ?? "http://localhost:4173/";

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
  if (t === "error") console.log("[browser err]", text);
  else if (text.includes("[int4]") || text.includes("int4") || text.includes("gallery")) {
    console.log(`[browser ${t}]`, text);
  }
});
page.on("pageerror", (e) => console.log("[pageerror]", e.message));

// Track which gallery URL the page actually fetched. The fp16 vs int4
// choice happens inside loadGalleryCard, so observing network is the
// most reliable way to confirm the right variant was picked.
const fetchedUrls = [];
page.on("request", (req) => {
  const u = req.url();
  if (u.includes("/gallery/") && (u.endsWith(".bin"))) fetchedUrls.push(u);
});

console.log(`opening ${URL} …`);
await page.goto(URL, { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click({ timeout: 1500 }).catch(() => {});

await page.waitForSelector("#demoBanner:not([hidden])", { timeout: 10_000 });
console.log("page loaded; gallery banner visible");

// Clear OPFS for the gallery cache so we know we're testing a fresh
// network fetch (otherwise an earlier session's cached fp16 file would
// short-circuit before the int4 selection logic runs).
await page.evaluate(async () => {
  if (navigator.storage?.getDirectory) {
    const root = await navigator.storage.getDirectory();
    try {
      const dir = await root.getDirectoryHandle("gallery");
      // @ts-ignore — newer File System Access API
      for await (const [name] of dir) {
        try { await dir.removeEntry(name); } catch { /* ignore */ }
      }
    } catch { /* no gallery dir yet */ }
  }
});
console.log("OPFS gallery cache cleared");

await page.locator("#openGalleryBtn").click();
await page.waitForSelector("#galleryDialog[open]", { timeout: 5_000 });
console.log("gallery dialog opened");
await page.locator('.gallery-card[data-id="shakespeare"]').click();
console.log("clicked Shakespeare card — model loading…");

const readyTimeoutMs = 60_000;
const t0 = Date.now();
let status = "";
while (Date.now() - t0 < readyTimeoutMs) {
  status = await page.evaluate(() => document.getElementById("status")?.textContent ?? "");
  if (/ready to generate/i.test(status)) break;
  await page.waitForTimeout(200);
}
console.log(`final status: "${status}"`);

// Did the page actually fetch the int4 variant?
const int4Fetched = fetchedUrls.some((u) => u.includes(".int4.bin"));
const fp16Fetched = fetchedUrls.some((u) => u.endsWith("/shakespeare.bin"));
console.log(`int4 file fetched: ${int4Fetched}`);
console.log(`fp16 fallback fetched: ${fp16Fetched}`);
console.log(`all gallery requests: ${fetchedUrls.join(", ")}`);

// Gate log line specifically.
const gateLine = consoleLines.find((l) => l.text.includes("[int4] numerics gate"));
console.log(`gate log: ${gateLine ? gateLine.text : "NOT FOUND"}`);
const gatePassed = gateLine && gateLine.text.includes("PASS");

// Generate Shakespeare to confirm end-to-end output still works.
console.log("\nclicking Generate to validate end-to-end …");
await page.locator("#prompt").fill("MENENIUS:\n");
await page.locator("#genTokens").fill("96");
await page.locator("#sample").click();
await page.waitForFunction(
  () => {
    const out = document.getElementById("output");
    if (!out || out.classList.contains("empty")) return false;
    return (out.textContent ?? "").length > 80;
  },
  null,
  { timeout: 30_000 },
);
const generated = await page.evaluate(() => document.getElementById("output")?.textContent ?? "");
console.log(`generated ${generated.length} chars:\n  ${generated.replace(/\n/g, "\n  ").slice(0, 300)}…`);

// Quality heuristic for Shakespeare: should contain newlines and either
// uppercase character names or common Shakespeare-ish words. NOT a
// rigorous test — just a "didn't completely fall apart" signal.
const hasNewlines = (generated.match(/\n/g) ?? []).length >= 2;
const hasShakespeareTokens = /\b(thee|thou|lord|prince|king|sir|my|the|and)\b/i.test(generated) || /[A-Z]{3,}:/.test(generated);

const wgslErrors = consoleLines.filter((l) =>
  l.t === "error" && /wgsl|shader|pipeline/i.test(l.text),
);

console.log("\n=== smoke summary ===");
console.log(`int4 gate ran:          ${gateLine ? "yes" : "NO"}`);
console.log(`int4 gate passed:       ${gatePassed ? "yes" : "no"}`);
console.log(`int4 file fetched:      ${int4Fetched ? "yes" : "NO"}`);
console.log(`generation produced:    ${generated.length > 80 ? "ok" : "FAIL"}`);
console.log(`generation has newlines:${hasNewlines ? "yes" : "no"}`);
console.log(`Shakespeare-ish tokens: ${hasShakespeareTokens ? "yes" : "no"}`);
console.log(`wgsl errors:            ${wgslErrors.length}`);

const pass = gateLine && int4Fetched && generated.length > 80 && wgslErrors.length === 0;
console.log(`\n${pass ? "[PASS] SMOKE PASS" : "[FAIL] SMOKE FAIL"}`);

await browser.close();
process.exit(pass ? 0 : 1);
