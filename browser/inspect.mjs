// inspect.mjs — print bounding rects of key elements to diagnose layout gaps.
import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await (await browser.newContext({ viewport: { width: 1400, height: 900 } })).newPage();
await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
const welcome = await page.locator("dialog#welcome").isVisible().catch(() => false);
if (welcome) { await page.locator("#welcomeSkip").click(); await page.waitForTimeout(200); }

const targets = [
  ".brand",
  ".lede",
  ".screen-nav",
  ".demo-banner",
  ".screens",
  'section.screen[data-screen="setup"]',
  'section.screen[data-screen="setup"] .train-grid',
  'section.screen[data-screen="setup"] .card.hero-config',
  'section.screen[data-screen="watch"]',
  ".machine-card",
  ".section-divider",
  "footer.notes",
];
for (const sel of targets) {
  const box = await page.locator(sel).first().boundingBox().catch(() => null);
  if (!box) {
    console.log(`${sel.padEnd(48)}  (not found)`);
    continue;
  }
  console.log(`${sel.padEnd(48)}  y=${Math.round(box.y).toString().padStart(4)}  h=${Math.round(box.height).toString().padStart(4)}  → bottom=${Math.round(box.y + box.height)}`);
}
// Find anything between screens-bottom and machine-card-top
const screensEnd = await page.locator(".screens").boundingBox();
const machineStart = await page.locator(".machine-card").boundingBox();
if (screensEnd && machineStart) {
  console.log(`\nGap between .screens bottom (${Math.round(screensEnd.y + screensEnd.height)}) and .machine-card top (${Math.round(machineStart.y)}) = ${Math.round(machineStart.y - (screensEnd.y + screensEnd.height))}px`);
}

// Dump computed style of .screens
const screensStyle = await page.locator(".screens").evaluate((el) => {
  const cs = getComputedStyle(el);
  return {
    paddingBottom: cs.paddingBottom,
    marginBottom: cs.marginBottom,
    height: cs.height,
    minHeight: cs.minHeight,
  };
});
console.log(".screens computed:", screensStyle);

// What's the bottom of the LAST child of .screens (probably the watch screen)
const watchBox = await page.locator('section.screen[data-screen="watch"]').evaluate((el) => {
  const cs = getComputedStyle(el);
  const rect = el.getBoundingClientRect();
  return {
    display: cs.display,
    visibility: cs.visibility,
    hidden: el.hasAttribute("hidden"),
    rectY: Math.round(rect.y),
    rectH: Math.round(rect.height),
  };
});
console.log("watch screen:", watchBox);

const machineStyle = await page.locator(".machine-card").evaluate((el) => {
  const cs = getComputedStyle(el);
  return {
    marginTop: cs.marginTop, marginBottom: cs.marginBottom,
    paddingTop: cs.paddingTop, paddingBottom: cs.paddingBottom,
    border: cs.border, animation: cs.animation,
  };
});
console.log("machine-card computed:", machineStyle);

// Inspect children of .wrap to see what's between screens and machine
const wrapKids = await page.locator(".wrap > *").evaluateAll((els) =>
  els.map((e) => ({
    tag: e.tagName.toLowerCase(),
    id: e.id || null,
    cls: e.className || null,
    y: Math.round(e.getBoundingClientRect().y),
    h: Math.round(e.getBoundingClientRect().height),
    display: getComputedStyle(e).display,
  }))
);
console.log("\n.wrap children:");
for (const k of wrapKids) {
  console.log(`  ${k.tag} ${k.id || ""} .${k.cls?.slice(0, 40) || ""}  y=${k.y} h=${k.h} display=${k.display}`);
}

// Trace ancestry of machine-card
const ancestry = await page.locator(".machine-card").evaluate((el) => {
  const chain = [];
  let cur = el;
  while (cur && cur.tagName !== "BODY") {
    chain.push(`${cur.tagName.toLowerCase()}${cur.id ? "#" + cur.id : ""}${cur.className ? "." + String(cur.className).split(/\s+/).slice(0, 2).join(".") : ""}`);
    cur = cur.parentElement;
  }
  return chain;
});
console.log("\nmachine-card ancestry (closest first):");
ancestry.forEach((a, i) => console.log(`  ${" ".repeat(i)}${a}`));

await browser.close();
