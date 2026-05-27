// smoke_live.mjs — verify the live tinygpt.sarthakagrawal.dev deploy in a
// real Chromium with WebGPU enabled. Sidesteps the local DNS cache by
// using a Playwright hostResolver fallback that points the domain at the
// resolved Cloudflare IP.
import { chromium } from "playwright";
import dns from "node:dns/promises";

const HOST = "tinygpt.sarthakagrawal.dev";

// Resolve via public DNS so we're not at the mercy of the local cache.
const ipResult = await dns.resolve4(HOST).catch(async () => {
  // Fallback: ask Cloudflare directly over HTTPS.
  const r = await fetch(`https://cloudflare-dns.com/dns-query?name=${HOST}&type=A`, {
    headers: { accept: "application/dns-json" },
  });
  const j = await r.json();
  return j.Answer.filter((a) => a.type === 1).map((a) => a.data);
});
const ip = ipResult[0];
console.log(`Resolved ${HOST} → ${ip}`);

const browser = await chromium.launch({
  headless: false,
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan",
    "--use-vulkan",
    `--host-resolver-rules=MAP ${HOST} ${ip}`,
  ],
});
const ctx = await browser.newContext();
const page = await ctx.newPage();
const consoleErrors = [];
page.on("pageerror", (e) => consoleErrors.push(`pageerror: ${e.message}`));
page.on("console", (m) => {
  if (m.type() === "error") consoleErrors.push(`console.error: ${m.text()}`);
});

console.log(`Loading https://${HOST}/ ...`);
await page.goto(`https://${HOST}/`, { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click({ timeout: 2000 }).catch(() => {});
await page.waitForTimeout(800);

// 1. Memory64 pill
const mem64 = await page.evaluate(() => {
  const el = document.querySelector('[data-explain="memory64"]');
  return el ? { text: el.textContent.trim(), on: el.classList.contains("on") } : null;
});
console.log(`Memory64 pill: ${JSON.stringify(mem64)}`);

// 2. WebGPU pill
const webgpu = await page.evaluate(() => {
  const el = document.getElementById("webgpuPill");
  return el ? { text: el.textContent.trim(), on: el.classList.contains("on") } : null;
});
console.log(`WebGPU pill:   ${JSON.stringify(webgpu)}`);

// 3. Behemoth preset present?
const hasBehemoth = await page.evaluate(() => {
  const opts = [...document.querySelectorAll("#sizePreset option")].map((o) => o.textContent.trim());
  return { count: opts.length, hasBehemoth: opts.some((o) => /behemoth/i.test(o)) };
});
console.log(`Preset list:   ${JSON.stringify(hasBehemoth)}`);

// 4. Demo load → sample button enables → introspection panel appears
await page.locator("#loadDemoBtn").click({ timeout: 5000 }).catch(() => {});
await page.waitForFunction(
  () => { const el = document.getElementById("sample"); return el && !el.disabled; },
  null, { timeout: 60_000 },
).catch((e) => console.log(`  sample never enabled: ${e.message}`));

const sampleReady = await page.evaluate(() => {
  const el = document.getElementById("sample");
  return el ? !el.disabled : null;
});
console.log(`Sample btn:    ${sampleReady ? "enabled (demo loaded ok)" : "still disabled"}`);

// Generate once + verify the introspection panel shows up
await page.locator("#sample").click().catch(() => {});
await page.waitForTimeout(2500);
const thinkCardVisible = await page.evaluate(() => {
  const el = document.querySelector(".think-card");
  if (!el) return null;
  return { exists: true, hidden: el.hidden, hasContent: el.textContent.includes("Watch") };
});
console.log(`Think card:    ${JSON.stringify(thinkCardVisible)}`);

// 5. The other four routes load?
for (const route of ["/roadmap", "/devlog", "/speedup", "/webgpu-test"]) {
  const resp = await page.goto(`https://${HOST}${route}`, { waitUntil: "networkidle" });
  console.log(`${route.padEnd(14)} ${resp.status()}  ${resp.headers()["content-type"]?.split(";")[0]}`);
}

console.log("\n=== Console errors ===");
if (consoleErrors.length) {
  consoleErrors.forEach((e) => console.log("  " + e));
} else {
  console.log("  none");
}

await browser.close();
