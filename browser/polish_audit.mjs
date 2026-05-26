// polish_audit.mjs — walk through the playground at multiple viewports
// and capture screenshots so I can spot what needs fixing before ship.
import { chromium } from "playwright";

const browser = await chromium.launch();
const pageErrors = [];

async function audit(label, width, height) {
  const ctx = await browser.newContext({
    viewport: { width, height },
    deviceScaleFactor: 2,
  });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => pageErrors.push(`[${label}] ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") pageErrors.push(`[${label}] console.error: ${m.text()}`);
  });

  // 1. Home (Setup)
  await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
  await page.locator("#welcomeSkip").click({ timeout: 1500 }).catch(() => {});
  await page.waitForTimeout(400);
  await page.screenshot({ path: `/tmp/polish-${label}-1-home.png`, fullPage: true });

  // 2. After demo load
  await page.locator("#loadDemoBtn").click().catch(() => {});
  await page.waitForFunction(() => {
    const el = document.getElementById("sample");
    return el && !el.disabled;
  }, { timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(500);

  // 3. Watch screen
  await page.locator('.screen-tab[data-screen="watch"]').click().catch(() => {});
  await page.waitForTimeout(400);
  await page.screenshot({ path: `/tmp/polish-${label}-2-watch.png`, fullPage: true });

  // 4. Generate
  const sampleBtn = page.locator("#sample");
  if (!(await sampleBtn.isDisabled())) {
    await sampleBtn.click();
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `/tmp/polish-${label}-3-sample.png`, fullPage: true });
  }

  // 5. Roadmap + Devlog + Speedup pages
  for (const route of ["roadmap", "devlog", "speedup"]) {
    await page.goto(`http://localhost:5173/${route}.html`, { waitUntil: "networkidle" });
    await page.waitForTimeout(300);
    await page.screenshot({ path: `/tmp/polish-${label}-${route}.png`, fullPage: true });
  }

  await ctx.close();
}

await audit("desktop", 1400, 900);
await audit("mobile", 390, 844);

if (pageErrors.length) {
  console.log("\nERRORS:\n" + pageErrors.join("\n"));
} else {
  console.log("\nNo page errors across desktop + mobile.");
}
console.log("Screenshots in /tmp/polish-*.png");

await browser.close();
