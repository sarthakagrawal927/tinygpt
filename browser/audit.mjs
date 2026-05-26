// audit.mjs — drive the page through key states and screenshot each.
import { chromium } from "playwright";

const browser = await chromium.launch();
const ctx = await browser.newContext({
  viewport: { width: 1400, height: 900 },
  deviceScaleFactor: 2,
});
const page = await ctx.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message));

await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click().catch(() => {});
await page.waitForTimeout(300);

// 1. Setup screen (fresh)
await page.screenshot({ path: "/tmp/audit-1-setup.png", fullPage: true });
console.log("1. setup screen captured");

// 2. Click "Try a trained model" to load demo (use force + longer timeout)
try {
  await page.locator("#loadDemoBtn").click({ timeout: 5000 });
  await page.waitForTimeout(4000); // demo file is ~10 MB
  await page.screenshot({ path: "/tmp/audit-2-demo-loaded.png", fullPage: true });
  console.log("2. demo-loaded captured");

  // 3. Watch tab
  await page.locator('.screen-tab[data-screen="watch"]').click({ timeout: 5000 });
  await page.waitForTimeout(500);
  await page.screenshot({ path: "/tmp/audit-3-watch.png", fullPage: true });
  console.log("3. watch screen captured");

  // 4. Generate
  const sampleBtn = page.locator("#sample");
  if (!(await sampleBtn.isDisabled())) {
    await sampleBtn.click();
    await page.waitForTimeout(2500);
    await page.screenshot({ path: "/tmp/audit-4-generated.png", fullPage: true });
    console.log("4. generated text captured");
  }
} catch (e) {
  console.log("audit error:", e.message.split("\n")[0]);
}

await browser.close();
