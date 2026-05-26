// take_screenshot.mjs — drive a headless Chromium to capture the page state.
// Lets me actually SEE what the rendered playground looks like instead of
// guessing from CSS code.
//
// Usage: node take_screenshot.mjs [url] [outPath]

import { chromium } from "playwright";

const url = process.argv[2] || "http://localhost:5173/";
const out = process.argv[3] || "/tmp/tinygpt-screenshot.png";

const browser = await chromium.launch();
const ctx = await browser.newContext({
  viewport: { width: 1400, height: 900 },
  deviceScaleFactor: 2,
});
const page = await ctx.newPage();
page.on("console", (msg) => {
  if (msg.type() === "error" || msg.type() === "warning") {
    console.log(`[${msg.type()}] ${msg.text()}`);
  }
});
page.on("pageerror", (e) => console.log("[pageerror]", e.message));

await page.goto(url, { waitUntil: "networkidle" });
// Dismiss the welcome dialog if it appears.
const welcome = await page.locator("dialog#welcome").isVisible().catch(() => false);
if (welcome) {
  await page.locator("#welcomeSkip").click();
  await page.waitForTimeout(300);
}

await page.screenshot({ path: out, fullPage: true });
console.log(`saved -> ${out}`);
console.log(`  page height: ${await page.evaluate(() => document.documentElement.scrollHeight)}`);
console.log(`  viewport: 1400x900`);
await browser.close();
