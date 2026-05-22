// webgpu_test.mjs — run the WebGPU kernel parity tests in a headless browser.
//
// Loads webgpu-test.html (which runs the checks from src/webgpu-test.ts) and
// reports the result. A "SKIP" — no WebGPU in this browser — is not a failure.
//
// Run from browser/:  npm run build && npm run preview &  then  npm run webgpu-test

import { chromium } from "playwright";

const BASE = process.env.BASE_URL || "http://localhost:4173";
const browser = await chromium.launch({
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
});
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text());
});

await page.goto(`${BASE}/webgpu-test.html`, { waitUntil: "load" });
await page.waitForFunction(
  () => /ALL PASS|FAILED|SKIP/.test(document.getElementById("results")?.textContent || ""),
  undefined,
  { timeout: 60000 },
);

const text = (await page.textContent("#results")) || "";
console.log(text);
if (errors.length) console.log("page errors: " + errors.join(" | "));
await browser.close();

const ok = /ALL PASS|SKIP/.test(text) && errors.length === 0;
process.exit(ok ? 0 : 1);
