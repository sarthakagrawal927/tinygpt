// audit_live.mjs — walk every route on the live site, surface anything that
// looks stale, broken, or off-brand. Designed for a pre-launch read-through
// without manually clicking through twelve pages.
//
// For each route, reports:
//   - HTTP status + final URL after redirects
//   - <title>, first <h1>
//   - text-content red-flags (old copy that should have been replaced)
//   - every internal link → HEAD-check each, flag any non-2xx
//   - rough word count (sanity check that the page rendered, not just an empty shell)

import { chromium } from "playwright";

const SITE = process.env.AUDIT_URL || "https://tinygpt.sarthakagrawal.dev";

const ROUTES = [
  "/",
  "/docs",
  "/docs/lessons",
  "/docs/session_retrospective",
  "/docs/qa_log",
  "/docs/decision_log",
  "/docs/study_guide",
  "/docs/annotated_transcript",
  "/speedup",
  "/devlog",
  "/roadmap",
  "/webgpu-test",
];

// Patterns that, if found, suggest stale content. We list each as
// { needle, why, skipIf }. `skipIf` is a route-prefix list — matches there
// are intentional history (the lessons + retrospective docs explicitly tell
// the story of old-bad-value → new-good-value, so they MUST contain the old
// strings as quoted historical context). Empty skipIf = check every route.
const STALE = [
  // Always-bad: these signal current copy hasn't been updated.
  { needle: "Want to skip the wait", why: "old banner copy — should be 'Two ways to see this work'", skipIf: [] },
  { needle: "A small model learns one thing at a time", why: "old default-corpus paragraph — should be Shakespeare", skipIf: ["/docs/"] },
  { needle: " 9.7× faster than WASM",          why: "flat '9.7×' headline — replaced by the curve", skipIf: ["/docs/", "/devlog", "/roadmap", "/speedup"] /* historical refs allowed */ },
  // Old-config values — only worry about them outside doc-history contexts.
  { needle: "value=\"0.003\"",                  why: "old LR default in a form input; should be 0.0003", skipIf: ["/docs/"] },
  { needle: " 0.003 is a safe default",        why: "old LR explainer text", skipIf: ["/docs/"] },
  { needle: "learningRate: 3e-3",               why: "old default in code-quote", skipIf: ["/docs/"] },
  { needle: "~0.8M-parameter GPT",              why: "stale params-as-identity claim", skipIf: ["/docs/"] },
  // Fixed-bug references — historical in docs, must not appear in code-rendered pages.
  { needle: "WebGPU model has no checkpoint serialization yet", why: "stale comment from before the fix; allowed only in retrospectives", skipIf: ["/docs/"] },
];

const results = [];
const linksCache = new Map(); // url → status, dedup

const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await ctx.newPage();

async function checkLink(url) {
  if (linksCache.has(url)) return linksCache.get(url);
  try {
    const r = await fetch(url, { method: "HEAD", redirect: "follow" });
    const out = { status: r.status, ok: r.ok };
    linksCache.set(url, out);
    return out;
  } catch (e) {
    const out = { status: 0, ok: false, error: e.message };
    linksCache.set(url, out);
    return out;
  }
}

for (const route of ROUTES) {
  const url = SITE + route;
  const r = { route, url, status: 0, title: "", h1: "", words: 0, stale: [], brokenLinks: [], internalLinks: 0 };

  try {
    const resp = await page.goto(url, { waitUntil: "networkidle", timeout: 30_000 });
    r.status = resp?.status() ?? 0;
    r.finalUrl = page.url();
  } catch (e) {
    r.error = e.message;
    results.push(r);
    continue;
  }

  r.title = await page.title();
  r.h1 = (await page.locator("h1").first().textContent().catch(() => ""))?.trim() ?? "";

  // Rough word count — body innerText
  const bodyText = await page.evaluate(() => document.body.innerText || "");
  r.words = bodyText.trim().split(/\s+/).length;

  // Stale-needle scan, respecting per-pattern route exclusions.
  for (const { needle, why, skipIf } of STALE) {
    const skipped = (skipIf ?? []).some((prefix) => route === prefix || route.startsWith(prefix));
    if (skipped) continue;
    if (bodyText.includes(needle)) r.stale.push({ needle, why });
  }

  // Internal links — same origin only
  const links = await page.evaluate((origin) => {
    const set = new Set();
    for (const a of document.querySelectorAll("a[href]")) {
      const href = a.getAttribute("href");
      if (!href) continue;
      if (href.startsWith("#") || href.startsWith("mailto:")) continue;
      try {
        const u = new URL(href, location.href);
        if (u.origin === origin) set.add(u.href.split("#")[0]); // strip fragments
      } catch { /* ignore */ }
    }
    return [...set];
  }, new URL(SITE).origin);

  r.internalLinks = links.length;
  for (const link of links) {
    const c = await checkLink(link);
    if (!c.ok) r.brokenLinks.push({ link, status: c.status, error: c.error });
  }

  results.push(r);
}

await browser.close();

// --- report ---
console.log(`\nAuditing ${SITE}\n${"=".repeat(60)}\n`);

let totalStale = 0;
let totalBroken = 0;
for (const r of results) {
  const ok = r.status === 200 && r.stale.length === 0 && r.brokenLinks.length === 0;
  const mark = ok ? "✅" : (r.status !== 200 ? "❌" : "⚠️ ");
  console.log(`${mark} ${r.route.padEnd(32)} HTTP ${r.status}  ${r.words} words · ${r.internalLinks} internal links`);
  if (r.error) console.log(`     ERROR: ${r.error}`);
  if (r.title) console.log(`     title: ${r.title.slice(0, 80)}`);
  if (r.h1)    console.log(`     h1:    ${r.h1.slice(0, 80)}`);
  for (const s of r.stale) {
    console.log(`     ⚠ stale: "${s.needle}"  — ${s.why}`);
    totalStale++;
  }
  for (const b of r.brokenLinks) {
    console.log(`     ❌ broken link → ${b.link}  (status ${b.status}${b.error ? " · " + b.error : ""})`);
    totalBroken++;
  }
  console.log();
}

console.log("=".repeat(60));
console.log(`${results.length} routes audited · ${totalStale} stale matches · ${totalBroken} broken internal links`);
if (totalStale === 0 && totalBroken === 0 && results.every((r) => r.status === 200)) {
  console.log("\nClean. Site is ready to share.");
  process.exit(0);
} else {
  console.log("\nIssues above need a look before launch.");
  process.exit(1);
}
