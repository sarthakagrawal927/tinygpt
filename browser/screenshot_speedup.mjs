import { chromium } from "playwright";
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1200, height: 1000 }, deviceScaleFactor: 2 });
const page = await ctx.newPage();
await page.goto("http://localhost:5173/speedup.html", { waitUntil: "networkidle" });
await page.waitForTimeout(200);
await page.screenshot({ path: "/tmp/speedup-full.png", fullPage: true });
console.log("-> /tmp/speedup-full.png");
// headline area only
const headline = page.locator(".headline").first();
await headline.screenshot({ path: "/tmp/speedup-headline.png" });
console.log("-> /tmp/speedup-headline.png");
// stack only
const stack = page.locator(".stack").first();
await stack.screenshot({ path: "/tmp/speedup-stack.png" });
console.log("-> /tmp/speedup-stack.png");
await browser.close();
