// run_f16_bench.mjs — click the WebGPU benchmark button and report the
// f16-packed vs f32 numbers from the page.
import { chromium } from "playwright";

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await ctx.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message));

await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click().catch(() => {});

// Open Diagnostics so the benchmark button is visible (not strictly needed —
// we can call .click on the element even if hidden).
// Click directly via DOM — the button lives inside a collapsed <details>.
await page.evaluate(() => document.getElementById("bench").click());

// Wait for the output to settle. Look for the "speed-up" line which only
// appears once both runs are done.
await page.waitForFunction(() => {
  const out = document.getElementById("benchOut");
  if (!out) return false;
  const t = out.textContent || "";
  return t.includes("speed-up") && (t.includes("f16-packed") || t.includes("PARITY FAILED"));
}, null, { timeout: 60_000 });

const result = await page.locator("#benchOut").innerText();
console.log("BENCH OUTPUT:");
console.log(result);
await browser.close();
