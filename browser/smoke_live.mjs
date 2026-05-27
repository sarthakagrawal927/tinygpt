// smoke_live.mjs — pre-launch walkthrough of the deployed playground.
// Verifies the full user flow end-to-end against https://tinygpt.sarthakagrawal.dev
// (or whatever SMOKE_URL points at). Reports each step pass/fail with timing.
//
// Run:  node browser/smoke_live.mjs
//       SMOKE_URL=http://localhost:5173 node browser/smoke_live.mjs   (against preview)

import { chromium } from "playwright";
import dns from "node:dns/promises";

const SITE = process.env.SMOKE_URL || "https://tinygpt.sarthakagrawal.dev";
const isLive = SITE.startsWith("https://");

const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok, detail });
  const mark = ok ? "✅" : "❌";
  console.log(`${mark} ${name}${detail ? "  — " + detail : ""}`);
};

// Sidestep local DNS cache when going against the live host.
const launchArgs = [
  "--enable-unsafe-webgpu",
  "--enable-features=Vulkan",
  "--use-vulkan",
];
if (isLive) {
  const host = new URL(SITE).host;
  try {
    const ips = await dns.resolve4(host);
    launchArgs.push(`--host-resolver-rules=MAP ${host} ${ips[0]}`);
    console.log(`Resolved ${host} → ${ips[0]}`);
  } catch { /* fall back to whatever the OS gives us */ }
}

const browser = await chromium.launch({ headless: false, args: launchArgs });
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 }, acceptDownloads: true });
const page = await ctx.newPage();
page.on("dialog", (d) => d.accept().catch(() => {}));
const pageErrors = [];
page.on("pageerror", (e) => pageErrors.push(e.message));
const consoleErrors = [];
page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });

console.log(`\n>> SITE = ${SITE}\n`);

// 1. Site loads
try {
  const resp = await page.goto(SITE, { waitUntil: "networkidle", timeout: 30_000 });
  check("site responds 200", resp?.status() === 200, `HTTP ${resp?.status()}`);
} catch (e) {
  check("site responds", false, e.message);
  await browser.close();
  process.exit(1);
}
await page.locator("#welcomeSkip").click({ timeout: 3000 }).catch(() => {});

// 2. COOP/COEP headers — needed for SharedArrayBuffer
try {
  const h = await page.evaluate(async () => {
    const r = await fetch(location.href, { method: "HEAD" });
    return { coop: r.headers.get("cross-origin-opener-policy"), coep: r.headers.get("cross-origin-embedder-policy") };
  });
  check("COOP header", h.coop === "same-origin", `value=${h.coop}`);
  check("COEP header", h.coep === "require-corp", `value=${h.coep}`);
} catch (e) { check("COOP/COEP", false, e.message); }

// 3. Banner copy
const banner = await page.locator("#demoBanner").isVisible().catch(() => false);
check("demo banner visible", banner);
if (banner) {
  const text = (await page.locator("#demoBanner").textContent()) ?? "";
  check("banner has new 'Two ways' copy", text.includes("Two ways"));
  check("banner has 'Load pretrained' label", text.includes("Load pretrained"));
}

// 4. Default corpus
await page.waitForTimeout(2500);
const corpusLen = await page.evaluate(() => document.getElementById("corpus")?.value.length ?? 0);
check("Shakespeare corpus auto-loaded", corpusLen > 1_000_000, `${corpusLen} bytes`);

// 5. demo.tinygpt — is the new Medium model under 25 MB? CF Pages strips
// Content-Length on HEAD for binary files, so we do a GET and measure the
// actual bytes.
const demoMeta = await page.evaluate(async () => {
  const r = await fetch("/demo.tinygpt");
  if (!r.ok) return { ok: false, size: 0 };
  const buf = await r.arrayBuffer();
  return { ok: true, size: buf.byteLength };
});
check("/demo.tinygpt reachable", demoMeta.ok);
check("demo size under CF cap", demoMeta.size > 0 && demoMeta.size < 25 * 1024 * 1024,
  `${(demoMeta.size / 1024 / 1024).toFixed(1)} MB`);

// 6. Load pretrained → expect auto-switch to Watch + Generate focused
const tLoad = Date.now();
await page.locator("#loadDemoBtn").click({ force: true });
await page.locator("#demoBanner").waitFor({ state: "hidden", timeout: 90_000 }).catch(() => {});
check("pretrained model loads", Date.now() - tLoad < 90_000, `${((Date.now() - tLoad) / 1000).toFixed(1)}s`);

// Wait a beat for the 50ms setTimeout + screen swap animation
await page.waitForTimeout(600);

const screenState = await page.evaluate(() => ({
  active: document.getElementById("screens")?.getAttribute("data-active"),
  focused: document.activeElement?.id ?? "",
  watchTabEnabled: !document.querySelector(".screen-tab[data-screen='watch']")?.disabled,
}));
check("auto-switched to Watch screen", screenState.active === "watch", `data-active=${screenState.active}`);
check("Generate button focused", screenState.focused === "sample", `activeElement=#${screenState.focused}`);
check("Watch tab is enabled", screenState.watchTabEnabled);
const tGen = Date.now();
await page.evaluate(() => document.getElementById("sample")?.click());

let firstByteMs = 0;
await page.waitForFunction(
  () => {
    const out = document.getElementById("output");
    return out && !out.classList.contains("empty") && out.textContent.length > 10;
  },
  null, { timeout: 60_000 },
).then(() => { firstByteMs = Date.now() - tGen; }).catch(() => {});

await page.waitForTimeout(10_000);
const output = await page.evaluate(() => document.getElementById("output")?.textContent ?? "");
check("Generate produces text", output.length > 40, `${output.length} chars, first byte ${firstByteMs}ms`);
console.log(`\n--- generated sample (live) ---\n${output.slice(0, 300)}\n--- /sample ---\n`);

// 8. tok/s stats
const statsText = await page.evaluate(() => document.getElementById("sampleStats")?.textContent ?? "");
check("tok/s stats rendered", statsText.includes("tok/s"), statsText.replace(/\s+/g, " ").trim().slice(0, 80));

// 9. /docs index
const docsResp = await page.goto(`${SITE}/docs`, { waitUntil: "domcontentloaded", timeout: 20_000 }).catch(() => null);
check("/docs index loads", docsResp?.status() === 200, `HTTP ${docsResp?.status()}`);
const docsCards = await page.locator(".doc-card").count().catch(() => 0);
check("/docs lists 6 cards", docsCards === 6, `found ${docsCards}`);

// 10. each doc page
for (const slug of ["lessons", "qa_log", "decision_log", "study_guide", "annotated_transcript", "session_retrospective"]) {
  const r = await page.goto(`${SITE}/docs/${slug}`, { waitUntil: "domcontentloaded", timeout: 15_000 }).catch(() => null);
  const h1 = r?.status() === 200 ? await page.locator(".prose h1").isVisible().catch(() => false) : false;
  check(`/docs/${slug}`, h1, `HTTP ${r?.status()}`);
}

// 11. other routes
for (const path of ["/speedup", "/devlog", "/roadmap"]) {
  const r = await page.goto(`${SITE}${path}`, { waitUntil: "domcontentloaded", timeout: 15_000 }).catch(() => null);
  check(path, r?.status() === 200, `HTTP ${r?.status()}`);
}

// 12. Console errors
check("no page errors", pageErrors.length === 0, pageErrors.slice(0, 3).join(" | ") || "");
check("no console errors", consoleErrors.length === 0, consoleErrors.slice(0, 3).join(" | ") || "");

const passed = results.filter((r) => r.ok).length;
console.log(`\n${"=".repeat(50)}`);
console.log(`${passed} / ${results.length} checks passed`);
if (passed < results.length) {
  console.log(`\nFailed:`);
  for (const r of results.filter((r) => !r.ok)) console.log(`  ❌ ${r.name}  — ${r.detail}`);
}

await browser.close();
process.exit(passed === results.length ? 0 : 1);
