// train_gallery_one.mjs — train a single gallery model from a corpus file
//
// Same pattern as train_demo.mjs (Huge preset, 5000 steps, WebGPU, seed 42)
// but parameterized:
//   --corpus=PATH    corpus text file
//   --out=NAME       output filename (no extension) → data/gallery/NAME.tinygpt
//   --prompt=TEXT    prompt for the post-training sample (default: corpus head)
//
// Produces three files:
//   data/gallery/NAME.tinygpt       canonical fp32 [w, m, v] triplets
//   data/gallery/NAME.sample.txt    ~400-char generation
//   data/gallery/NAME.meta.json     { name, corpusBytes, finalTrainLoss, trainWallMs }
//
// The fp16-pack + manifest assembly happens in finalize_gallery.mjs.

import { chromium } from "playwright";
import { promises as fs } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.match(/^--([^=]+)=(.*)$/);
    return m ? [m[1], m[2]] : [a, true];
  }),
);
if (!args.corpus || !args.out) {
  console.error("usage: node train_gallery_one.mjs --corpus=PATH --out=NAME [--prompt=TEXT]");
  process.exit(1);
}

const CORPUS_PATH = resolve(args.corpus);
const OUT_DIR = resolve(ROOT, "data/gallery");
const OUT_FILE = resolve(OUT_DIR, `${args.out}.tinygpt`);
const SAMPLE_FILE = resolve(OUT_DIR, `${args.out}.sample.txt`);
const META_FILE = resolve(OUT_DIR, `${args.out}.meta.json`);
await fs.mkdir(OUT_DIR, { recursive: true });

const STEPS = 5000;
const PRESET = "huge";
const POLL_MS = 30_000;
const MAX_WALL_MS = 120 * 60 * 1000;

const corpus = await fs.readFile(CORPUS_PATH, "utf8");
console.log(`[${args.out}] corpus: ${corpus.length} bytes from ${CORPUS_PATH}`);

const prompt = args.prompt ?? corpus.slice(0, 60).replace(/\n/g, " ").trim();
console.log(`[${args.out}] sample prompt: "${prompt}"`);

const browser = await chromium.launch({
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan"],
});
const ctx = await browser.newContext({
  viewport: { width: 1400, height: 900 },
  acceptDownloads: true,
});
const page = await ctx.newPage();
page.on("dialog", (d) => d.accept().catch(() => {}));
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
page.on("console", (m) => {
  const t = m.type();
  if (t === "error") console.log("[err]", m.text());
});

// `?autoSave=NAME` makes the browser app fire the model download itself
// the moment training completes (case "done" with reason "finished").
// We then await page.waitForEvent("download") and copy the file — no
// "wait for downloadModel button → click" dance needed, so the artifact
// persists even if the post-training sample step crashes or times out.
const APP_URL = `http://localhost:5173/?autoSave=${encodeURIComponent(args.out)}`;
await page.goto(APP_URL, { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click({ timeout: 1500 }).catch(() => {});

await page.evaluate((text) => {
  const el = document.getElementById("corpus") as HTMLTextAreaElement;
  el.value = text;
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
}, corpus);
const corpusLen = await page.evaluate(() => (document.getElementById("corpus") as HTMLTextAreaElement).value.length);
// Textarea silently strips some bytes (NULs, lone surrogates, weird control chars).
// Tolerate up to 5% loss — for a 1.2MB corpus that's plenty of training data.
if (corpusLen < corpus.length * 0.95) throw new Error(`corpus truncated badly: got ${corpusLen}/${corpus.length}`);
if (corpusLen < corpus.length) console.log(`[${args.out}] corpus normalized by textarea: ${corpus.length} -> ${corpusLen} (${((1 - corpusLen/corpus.length) * 100).toFixed(2)}% stripped)`);

await page.evaluate(({ preset, steps }) => {
  const setVal = (id: string, v: number | string) => {
    const el = document.getElementById(id) as HTMLInputElement;
    el.value = String(v);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  };
  (document.getElementById("sizePreset") as HTMLSelectElement).value = preset;
  document.getElementById("sizePreset")!.dispatchEvent(new Event("change", { bubbles: true }));
  setVal("maxSteps", steps);
  setVal("lr", 0.0003);
  const back = document.getElementById("backend") as HTMLSelectElement;
  back.value = "webgpu";
  back.dataset.userPicked = "1";
  back.dispatchEvent(new Event("change", { bubbles: true }));
  const seed = document.getElementById("seed") as HTMLInputElement | null;
  if (seed) setVal("seed", 42);
}, { preset: PRESET, steps: STEPS });

const cfg = await page.evaluate(() => ({
  preset: (document.getElementById("sizePreset") as HTMLSelectElement).value,
  maxSteps: (document.getElementById("maxSteps") as HTMLInputElement).value,
  backend: (document.getElementById("backend") as HTMLSelectElement).value,
}));
console.log(`[${args.out}] config:`, JSON.stringify(cfg));

// Register the auto-save download listener BEFORE we start training. The
// `case "done"` handler in main.ts will fire the download the moment the
// last step lands, and we don't know exactly when between two polling
// ticks that'll be. `waitForEvent` registered up-front catches the event
// whenever it actually fires. Timeout is set to MAX_WALL_MS + a buffer.
const downloadPromise = page.waitForEvent("download", { timeout: MAX_WALL_MS + 5 * 60 * 1000 });
// Silence unhandled-rejection warnings if training errors out before we
// await the download. The same promise is still awaited below — the
// no-op .catch here only swallows the case where we never get there.
downloadPromise.catch(() => {});

const tStart = Date.now();
await page.locator("#start").click({ force: true });
console.log(`[${args.out}] training started at ${new Date().toISOString()}`);

let finalTrain = "";
while (true) {
  await new Promise((r) => setTimeout(r, POLL_MS));
  const wall = Date.now() - tStart;
  const s = await page.evaluate(() => ({
    step: document.getElementById("stStep")?.textContent ?? "",
    train: document.getElementById("stTrain")?.textContent ?? "",
    val: document.getElementById("stVal")?.textContent ?? "",
    status: document.getElementById("status")?.textContent ?? "",
  }));
  const m = s.step.match(/^(\d+)\s*\/\s*(\d+)/);
  const cur = m ? Number(m[1]) : 0;
  const max = m ? Number(m[2]) : 0;
  finalTrain = s.train;
  console.log(
    `[${args.out}] t+${(wall / 60000).toFixed(1)}min  step=${cur}/${max}  train=${s.train}  val=${s.val}`,
  );
  if (cur >= max && max > 0) { console.log(`[${args.out}] === training complete ===`); break; }
  if (/error|failed/i.test(s.status)) throw new Error(`training error: ${s.status}`);
  if (wall > MAX_WALL_MS) throw new Error("hard cap exceeded");
}
const trainWallMs = Date.now() - tStart;

// SAVE CHECKPOINT FIRST — the .tinygpt is the only critical artifact.
// The download was kicked off by the browser app's auto-save (URL param)
// and we registered the listener BEFORE training started, so the event
// is either pending now or already fired into the listener.
console.log(`[${args.out}] --- saving checkpoint (auto-save) ---`);
const download = await downloadPromise;
const tmp = await download.path();
await fs.copyFile(tmp, OUT_FILE);
const stat = await fs.stat(OUT_FILE);
console.log(`[${args.out}] wrote ${OUT_FILE}: ${(stat.size / 1024 / 1024).toFixed(2)} MB`);

// Generate a sample so the gallery card can show real model output.
// Failure here no longer loses training: the checkpoint is already on
// disk. `try/catch` so we can still write meta with the loss + wall time.
let sample = "";
try {
  console.log(`[${args.out}] --- generating sample ---`);
  await page.waitForFunction(
    () => {
      const btn = document.getElementById("sample") as HTMLButtonElement | null;
      return btn && !btn.disabled;
    },
    null,
    { timeout: 120_000 },
  );
  await page.evaluate((p) => {
    const setVal = (id: string, v: number | string) => {
      const el = document.getElementById(id) as HTMLInputElement | null;
      if (!el) return;
      el.value = String(v);
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    };
    setVal("prompt", p);
    setVal("temp", 0.8);
    setVal("genTokens", 400);
  }, prompt);
  await page.locator("#sample").click({ force: true });

  await page.waitForFunction(
    () => {
      const out = document.getElementById("output");
      if (!out) return false;
      const txt = (out.textContent ?? "").trim();
      if (out.classList.contains("empty")) return false;
      return txt.length > 50;
    },
    null,
    { timeout: 120_000 },
  );
  await page.waitForTimeout(3000);
  sample = await page.evaluate(() => document.getElementById("output")?.textContent ?? "");
  console.log(`[${args.out}] sample (${sample.length} chars): ${sample.slice(0, 160)}…`);
  await fs.writeFile(SAMPLE_FILE, sample);
} catch (err) {
  console.log(`[${args.out}] sample step failed (non-fatal — checkpoint is saved): ${err.message}`);
  await fs.writeFile(SAMPLE_FILE, "(sample generation failed — see train log)");
}

await fs.writeFile(
  META_FILE,
  JSON.stringify(
    {
      name: args.out,
      corpusPath: CORPUS_PATH,
      corpusBytes: corpus.length,
      finalTrainLoss: finalTrain,
      trainWallMs,
      steps: STEPS,
      preset: PRESET,
      samplePrompt: prompt,
      sampleChars: sample.length,
    },
    null,
    2,
  ),
);
console.log(`[${args.out}] wrote ${META_FILE}`);

await browser.close();
console.log(`[${args.out}] done.`);
