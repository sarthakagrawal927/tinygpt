/**
 * main.ts — UI / main-thread controller (Phase 4).
 *
 * The main thread owns ONLY the UI: capability panel, corpus input, controls,
 * and the loss chart. Training runs entirely in worker.ts, so the page stays
 * responsive — this file never does model math.
 *
 * Guide: docs/browser_notes.md ("Web Worker")
 */

import { benchmarkMatmul, initWebGPU } from "../../webgpu/kernels";
import { TinyGptBackend } from "./backend";
import { HF_CATALOG, HfFetchError, fetchHfText } from "./datasets";
import { LossChart } from "./charts";
import {
  detectBrowser,
  detectCapabilities,
  detectHardware,
  getGpuName,
  recommendModel,
  type ModelRecommendation,
} from "./runtime_detect";
import { initPopovers } from "./popover";
import {
  PRESETS,
  estimateParams,
  estimateTrainSeconds,
  formatDuration,
  formatParams,
  headsFor,
} from "./sizing";
import { loadRun, loadState, requestDurableStorage, saveRun, saveState } from "./storage";
import { hasSeenTour, markTourSeen, startTour } from "./tour";
import { DEFAULT_CONFIG, type FromWorker, type RunConfig, type ToWorker } from "./types";

const byId = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

const els = {
  caps: byId<HTMLDivElement>("caps"),
  corpus: byId<HTMLTextAreaElement>("corpus"),
  start: byId<HTMLButtonElement>("start"),
  pause: byId<HTMLButtonElement>("pause"),
  stop: byId<HTMLButtonElement>("stop"),
  sample: byId<HTMLButtonElement>("sample"),
  status: byId<HTMLDivElement>("status"),
  output: byId<HTMLDivElement>("output"),
  stStep: byId<HTMLElement>("stStep"),
  stTrain: byId<HTMLElement>("stTrain"),
  stVal: byId<HTMLElement>("stVal"),
  stToks: byId<HTMLElement>("stToks"),
  stEta: byId<HTMLElement>("stEta"),
  stBackend: byId<HTMLElement>("stBackend"),
  bench: byId<HTMLButtonElement>("bench"),
  benchOut: byId<HTMLDivElement>("benchOut"),
  hfDataset: byId<HTMLSelectElement>("hfDataset"),
  hfStatus: byId<HTMLSpanElement>("hfStatus"),
  hfCustom: byId<HTMLDivElement>("hfCustom"),
  hfCustomDataset: byId<HTMLInputElement>("hfCustomDataset"),
  hfCustomConfig: byId<HTMLInputElement>("hfCustomConfig"),
  hfCustomSplit: byId<HTMLInputElement>("hfCustomSplit"),
  hfCustomColumn: byId<HTMLInputElement>("hfCustomColumn"),
  hfCustomLoad: byId<HTMLButtonElement>("hfCustomLoad"),
  progressBar: byId<HTMLDivElement>("progressBar"),
  sizePreset: byId<HTMLSelectElement>("sizePreset"),
  estParams: byId<HTMLElement>("estParams"),
  estTime: byId<HTMLElement>("estTime"),
  estNote: byId<HTMLElement>("estNote"),
  notify: byId<HTMLInputElement>("notify"),
  notifyStatus: byId<HTMLSpanElement>("notifyStatus"),
  stPpl: byId<HTMLElement>("stPpl"),
  stElapsed: byId<HTMLElement>("stElapsed"),
  stBestVal: byId<HTMLElement>("stBestVal"),
  stGap: byId<HTMLElement>("stGap"),
  milestone: byId<HTMLDivElement>("milestone"),
  milestoneText: byId<HTMLElement>("milestoneText"),
  mpPreview: byId<HTMLElement>("mpPreview"),
  mpBody: byId<HTMLElement>("mpBody"),
  downloadModel: byId<HTMLButtonElement>("downloadModel"),
  downloadSafetensors: byId<HTMLButtonElement>("downloadSafetensors"),
  uploadModel: byId<HTMLInputElement>("uploadModel"),
  modelStatus: byId<HTMLElement>("modelStatus"),
  continueBtn: byId<HTMLButtonElement>("continueBtn"),
  continueSteps: byId<HTMLInputElement>("continueSteps"),
  uploadCorpus: byId<HTMLInputElement>("uploadCorpus"),
  wikiTitle: byId<HTMLInputElement>("wikiTitle"),
  wikiLoad: byId<HTMLButtonElement>("wikiLoad"),
  urlInput: byId<HTMLInputElement>("urlInput"),
  urlLoad: byId<HTMLButtonElement>("urlLoad"),
  fetchSize: byId<HTMLSelectElement>("fetchSize"),
  dataLoadStatus: byId<HTMLElement>("dataLoadStatus"),
  shareBtn: byId<HTMLButtonElement>("shareBtnTop"),
  shareToast: byId<HTMLDivElement>("shareToast"),
  hfTokenRow: byId<HTMLDivElement>("hfTokenRow"),
  hfToken: byId<HTMLInputElement>("hfToken"),
  hfTokenSave: byId<HTMLButtonElement>("hfTokenSave"),
  hfTokenClear: byId<HTMLButtonElement>("hfTokenClear"),
};

function setOutput(text: string, empty = false): void {
  els.output.textContent = text;
  els.output.classList.toggle("empty", empty);
  // Show the copy button only when we have a real generation (not the empty
  // state or the "Training…" placeholder).
  const copyBtn = document.getElementById("copyOutput");
  if (copyBtn) copyBtn.hidden = empty;
}

/**
 * Type a sample into the output element one character at a time. Skipped on
 * reduced-motion preference. Returns immediately and lets the animation run.
 */
let typewriterToken = 0;
function typewriteOutput(text: string): void {
  els.output.classList.remove("empty");
  if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
    els.output.textContent = text;
    return;
  }
  const myToken = ++typewriterToken;
  els.output.textContent = "";
  const chunk = Math.max(1, Math.floor(text.length / 240));
  let i = 0;
  const step = () => {
    if (myToken !== typewriterToken) return;
    i = Math.min(text.length, i + chunk);
    els.output.textContent = text.slice(0, i);
    if (i < text.length) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/**
 * Briefly flash a stat value in the accent color when it ticks. Lets the eye
 * track which numbers are updating without being noisy.
 *
 * Throttled: each stat only re-renders every ~150 ms even if progress messages
 * arrive faster (they often do — every step on a fast machine). This keeps the
 * main thread free to handle the chart redraw and the user's clicks.
 */
const flashTimers = new WeakMap<HTMLElement, number>();
const flashLast = new WeakMap<HTMLElement, number>();
const flashPending = new WeakMap<HTMLElement, string>();
const FLASH_MIN_INTERVAL_MS = 150;

function flashStat(el: HTMLElement, value: string): void {
  if (el.textContent === value) return;
  const now = performance.now();
  const last = flashLast.get(el) ?? 0;
  const elapsed = now - last;
  if (elapsed < FLASH_MIN_INTERVAL_MS) {
    // Defer — keep the most-recent target value; collapse rapid updates.
    flashPending.set(el, value);
    const pendingTimer = flashTimers.get(el);
    if (pendingTimer) return;
    const t = window.setTimeout(() => {
      flashTimers.delete(el);
      const pending = flashPending.get(el);
      flashPending.delete(el);
      if (pending != null) flashStat(el, pending);
    }, FLASH_MIN_INTERVAL_MS - elapsed);
    flashTimers.set(el, t);
    return;
  }
  el.textContent = value;
  el.classList.add("flash");
  flashLast.set(el, now);
  const prev = flashTimers.get(el);
  if (prev) clearTimeout(prev);
  const t = window.setTimeout(() => el.classList.remove("flash"), 280);
  flashTimers.set(el, t);
}

function setStatus(text: string, isError = false): void {
  els.status.textContent = text;
  els.status.classList.toggle("error", isError && text.length > 0);
}

function setProgress(step: number, maxSteps: number): void {
  const pct = maxSteps > 0 ? Math.min(100, (step / maxSteps) * 100) : 0;
  els.progressBar.style.width = `${pct}%`;
}

const canvas = byId<HTMLCanvasElement>("chart");
// Let CSS drive the canvas size — the LossChart.setupHiDpi() reads the
// rendered rect and scales the backing store accordingly. Hard-coding these
// here overrode the CSS that wanted a 420px-tall hero chart.
canvas.width = canvas.clientWidth || canvas.width;
canvas.height = canvas.clientHeight || canvas.height;
const chart = new LossChart(canvas);

const worker = new Worker(new URL("./worker.ts", import.meta.url), {
  type: "module",
});
const send = (msg: ToWorker) => worker.postMessage(msg);

let paused = false;
let history: { step: number; trainLoss: number; valLoss?: number }[] = [];
let lastConfig: RunConfig | null = null; // config of the in-flight / last run
let savedThisRun = false; // did this run produce an OPFS checkpoint?
let runStartTime = 0; // performance.now() when training started
let bestVal = Infinity;
let bestValStep = 0;
let milestonesHit = new Set<string>();
let elapsedTimer: number | undefined;

const LN_256 = Math.log(256); // 5.545 — the random-baseline loss
let firstRunCelebrated = (() => {
  try { return localStorage.getItem("tinygpt.firstRunCelebrated") === "1"; } catch { return false; }
})();

/**
 * Render a post-run "verdict" — what happened + a concrete "do better next
 * time" suggestion. Goal: the user never has to leave the playground to
 * figure out *why* the output was bad.
 */
function renderRunVerdict(finished: boolean): void {
  const verdict = document.getElementById("runVerdict");
  if (!verdict) return;
  if (!finished || history.length === 0) {
    verdict.hidden = true;
    return;
  }

  const final = history[history.length - 1];
  const trainLoss = final.trainLoss;
  const valLoss = final.valLoss;
  const cfg = readConfig();
  const params = estimateParams(cfg.layers, cfg.dModel, cfg.ctx);
  const corpusBytes = els.corpus.value.length;
  const bytesPerParamSeen = (corpusBytes * 6) / Math.max(params, 1);
  const valGap = valLoss != null ? valLoss - trainLoss : null;

  // Headline interpretation of the achieved loss.
  let headline: string;
  if (trainLoss > 4.0) headline = "Barely moved from random.";
  else if (trainLoss > 3.0) headline = "Letter pairs learned, no words yet.";
  else if (trainLoss > 2.3) headline = "Common short words emerging, no grammar.";
  else if (trainLoss > 1.7) headline = "Word shapes formed, local grammar rough.";
  else if (trainLoss > 1.2) headline = "Local grammar visible. Generation will read.";
  else if (trainLoss > 0.8) headline = "Substantial memorisation of the corpus.";
  else headline = "Near-perfect memorisation.";

  // Pick the single most actionable improvement.
  const actions: Array<{ label: string; apply: () => void }> = [];
  let why = "";

  if (trainLoss > 1.8 && bytesPerParamSeen < 5) {
    why = `Your model has ~${Math.round(params / 1e3)}k params but only saw ~${bytesPerParamSeen.toFixed(1)} bytes per parameter (Chinchilla floor is ~20). The data can't fill the capacity.`;
    actions.push({
      label: "Load Tiny Shakespeare (1 MB)",
      apply: () => {
        switchTab("url");
        els.urlInput.value = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
        els.urlLoad.click();
      },
    });
  } else if (trainLoss > 2.3 && cfg.maxSteps < 3000) {
    why = `${cfg.maxSteps} steps is undertrained for this model. Loss is still falling — give it more.`;
    actions.push({
      label: "Continue +2000 steps",
      apply: () => {
        els.continueSteps.value = "2000";
        els.continueBtn.click();
      },
    });
  } else if (valGap != null && valGap > 0.8 && trainLoss < 1.5) {
    why = `Train loss is ${trainLoss.toFixed(2)} but val is ${valLoss!.toFixed(2)} — you're memorising the training text, not learning patterns. Smaller model or bigger corpus.`;
    actions.push({
      label: "Switch to a smaller preset",
      apply: () => {
        els.sizePreset.value = "small";
        els.sizePreset.dispatchEvent(new Event("change"));
      },
    });
  } else if (trainLoss > 2.0 && params < 1_000_000) {
    why = `${formatParams(params)} params is narrow for the data you have. A bigger model would chew through more of it.`;
    actions.push({
      label: "Try Medium preset",
      apply: () => {
        els.sizePreset.value = "medium";
        els.sizePreset.dispatchEvent(new Event("change"));
      },
    });
  } else if (trainLoss < 1.5) {
    why = `Train loss ${trainLoss.toFixed(2)} is the sweet spot for this size. Generate now — output should look word-shaped.`;
    actions.push({
      label: "Generate text →",
      apply: () => {
        els.sample.scrollIntoView({ behavior: "smooth", block: "center" });
        els.sample.click();
      },
    });
  } else {
    why = `Loss ${trainLoss.toFixed(2)} is "${headline.toLowerCase()}" Try continuing or switching dataset.`;
    actions.push({
      label: "Continue +1000 steps",
      apply: () => {
        els.continueSteps.value = "1000";
        els.continueBtn.click();
      },
    });
  }

  const actionHtml = actions
    .map((_, i) => `<button class="verdict-cta" data-action-idx="${i}">${actions[i].label}</button>`)
    .join("");

  verdict.innerHTML =
    `<strong>${headline}</strong> ${why}` +
    (actionHtml ? `<div class="verdict-actions">${actionHtml}</div>` : "");
  verdict.hidden = false;
  verdict.querySelectorAll<HTMLButtonElement>(".verdict-cta").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = parseInt(btn.dataset.actionIdx || "0", 10);
      actions[idx]?.apply();
    });
  });
}

function showFirstRunCelebration(): void {
  const final = history[history.length - 1];
  const lossStr = final ? final.trainLoss.toFixed(2) : "?";
  const sub = `You trained your first transformer. Loss: ${lossStr}. Click Generate to see what it learned.`;
  const div = document.createElement("div");
  div.className = "first-run-toast";
  div.innerHTML = `
    <span class="first-run-icon" aria-hidden="true">✓</span>
    <div>
      <div class="first-run-title">First model trained</div>
      <div class="first-run-sub">${sub}</div>
    </div>
    <button class="first-run-cta" type="button">Generate ↓</button>
  `;
  document.body.appendChild(div);
  div.querySelector<HTMLButtonElement>(".first-run-cta")?.addEventListener("click", () => {
    div.remove();
    els.sample.scrollIntoView({ behavior: "smooth", block: "center" });
    els.sample.classList.add("nudge");
    setTimeout(() => els.sample.classList.remove("nudge"), 1200);
  });
  setTimeout(() => div.classList.add("dismissing"), 9000);
  setTimeout(() => div.remove(), 9600);
}

function formatPerplexity(loss: number): string {
  if (!Number.isFinite(loss) || loss < 0) return "–";
  const p = Math.exp(loss);
  if (p >= 100) return p.toFixed(0);
  if (p >= 10) return p.toFixed(1);
  return p.toFixed(2);
}

function showMilestone(text: string): void {
  els.milestoneText.textContent = text;
  els.milestone.hidden = false;
}

function clearMilestone(): void {
  els.milestone.hidden = true;
  els.milestoneText.textContent = "";
}

function startElapsedClock(): void {
  if (elapsedTimer != null) clearInterval(elapsedTimer);
  elapsedTimer = window.setInterval(() => {
    const seconds = (performance.now() - runStartTime) / 1000;
    els.stElapsed.textContent = formatElapsed(seconds);
  }, 250);
}
function stopElapsedClock(): void {
  if (elapsedTimer != null) { clearInterval(elapsedTimer); elapsedTimer = undefined; }
}
function formatElapsed(s: number): string {
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s % 60);
  return `${m}m ${r.toString().padStart(2, "0")}s`;
}

function checkMilestones(_step: number, trainLoss: number, valLoss?: number): void {
  if (trainLoss < LN_256 - 0.3 && !milestonesHit.has("baseline")) {
    milestonesHit.add("baseline");
    showMilestone(`Past random baseline (ln 256 = 5.55) — the model is no longer guessing uniformly.`);
    return;
  }
  if (trainLoss < 3.0 && !milestonesHit.has("under3")) {
    milestonesHit.add("under3");
    showMilestone(`Loss under 3.0 — the model has locked onto byte frequencies and short n-grams.`);
    return;
  }
  if (trainLoss < 2.0 && !milestonesHit.has("under2")) {
    milestonesHit.add("under2");
    showMilestone(`Loss under 2.0 — real structure has been learned. Generated samples should start looking word-shaped.`);
    return;
  }
  if (trainLoss < 1.0 && !milestonesHit.has("under1")) {
    milestonesHit.add("under1");
    showMilestone(`Loss under 1.0 — substantial memorisation of this corpus. Holdout loss is now the honest signal.`);
    return;
  }
  if (valLoss != null && valLoss - trainLoss > 1.0 && !milestonesHit.has("overfit")) {
    milestonesHit.add("overfit");
    showMilestone(`Train ↔ holdout gap > 1.0 — overfitting territory. The model is memorising rather than generalising.`);
    return;
  }
}

// --- config ---------------------------------------------------------------
function readConfig(): RunConfig {
  const intOf = (id: string) => parseInt(byId<HTMLInputElement>(id).value, 10);
  const dModel = parseInt(byId<HTMLSelectElement>("dModel").value, 10);
  return {
    ctx: intOf("ctx"),
    layers: intOf("layers"),
    heads: headsFor(dModel),
    dModel,
    dMlp: dModel * 4,
    batchSize: intOf("batch"),
    learningRate: parseFloat(byId<HTMLInputElement>("lr").value),
    gradClip: DEFAULT_CONFIG.gradClip,
    maxSteps: intOf("maxSteps"),
    evalEvery: DEFAULT_CONFIG.evalEvery,
    seed: DEFAULT_CONFIG.seed,
    backend: byId<HTMLSelectElement>("backend").value === "webgpu" ? "webgpu" : "wasm",
  };
}

// --- button state ---------------------------------------------------------
function setRunning(on: boolean): void {
  els.start.disabled = on;
  els.pause.disabled = !on;
  els.stop.disabled = !on;
  if (on) els.continueBtn.disabled = true;
  const live = document.getElementById("liveBadge");
  if (live) live.hidden = !on;
  // Notify the sticky-stats observer.
  const hook = (window as unknown as { __tgSetTraining?: (b: boolean) => void }).__tgSetTraining;
  if (typeof hook === "function") hook(on);
}

function updateStickyStats(step: number, maxSteps: number, loss: number, toks: number, etaText: string): void {
  const s = document.getElementById("stickyStep");
  const l = document.getElementById("stickyLoss");
  const t = document.getElementById("stickyToks");
  const e = document.getElementById("stickyEta");
  if (s) s.textContent = `${step}/${maxSteps}`;
  if (l) l.textContent = loss.toFixed(3);
  if (t) t.textContent = Math.round(toks).toLocaleString();
  if (e) e.textContent = etaText;
}

// --- hardware-aware recommendation ----------------------------------------
/** Human-readable duration, for the live training-time estimate. */
function formatTime(seconds: number): string {
  if (seconds < 1) return "<1s";
  if (seconds < 90) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

/** Fill the config inputs with a recommended model for the detected machine. */
function applyRecommendation(rec: ModelRecommendation): void {
  byId<HTMLInputElement>("ctx").value = String(rec.ctx);
  byId<HTMLInputElement>("layers").value = String(rec.layers);
  byId<HTMLSelectElement>("dModel").value = String(rec.dModel);
  byId<HTMLInputElement>("maxSteps").value = String(rec.maxSteps);
}

// --- controls -------------------------------------------------------------
els.start.addEventListener("click", () => {
  const text = els.corpus.value;
  if (text.length < 80) {
    setStatus("corpus is very short — add more text", true);
    return;
  }
  const cfg = readConfig();
  const estParams = estimateParams(cfg.layers, cfg.dModel, cfg.ctx);
  const corpusBytes = text.length;
  const bytesPerParamSeen = (corpusBytes * 6) / Math.max(estParams, 1);

  // Pre-flight warning when the config is going to disappoint. Catch the
  // most common mistakes before they cost 20 minutes of compute:
  //   - data-starved (way more params than bytes can fill)
  //   - undertrained (big model + low step budget)
  const warnings: string[] = [];
  if (bytesPerParamSeen < 5 && estParams > 500_000) {
    warnings.push(
      `Your corpus has only ~${bytesPerParamSeen.toFixed(1)} bytes per parameter ` +
      `(Chinchilla floor is ~20). The model will plateau early; output will be ` +
      `letter-level, not word-level. Pick a bigger dataset (TinyStories, ` +
      `Tiny Shakespeare, Wikipedia topic) before running.`,
    );
  }
  if (estParams > 5_000_000 && cfg.maxSteps < 2000) {
    warnings.push(
      `${(estParams / 1e6).toFixed(0)}M-param models typically need 3000+ steps to converge. ` +
      `${cfg.maxSteps} steps will leave it undertrained — loss will look stuck.`,
    );
  }
  if (warnings.length > 0) {
    const ok = window.confirm(
      "⚠ Heads up — this config is likely to produce poor output:\n\n" +
      warnings.map((w, i) => `${i + 1}. ${w}`).join("\n\n") +
      "\n\nRun anyway?",
    );
    if (!ok) return;
  }

  // For the big presets, also confirm the time cost.
  if (estParams > 5_000_000) {
    const minutes = estimateTrainSeconds(
      cfg.layers, cfg.dModel, cfg.ctx, cfg.batchSize, cfg.maxSteps, cachedCpuProbeMs,
    ) / 60;
    const onWebgpu = cfg.backend === "webgpu";
    const wallClock = onWebgpu ? Math.max(2, Math.round(minutes / 7)) : Math.round(minutes);
    const backendLabel = onWebgpu ? "WebGPU" : "WASM (CPU)";
    const ok = window.confirm(
      `This is a ${(estParams / 1e6).toFixed(1)}M-param run on ${backendLabel}.\n\n` +
      `Estimated time: ~${wallClock} minute${wallClock === 1 ? "" : "s"}.\n\n` +
      `Your machine may feel slower while it's running — WebGPU shares the GPU with the system compositor.\n\n` +
      `Continue?`,
    );
    if (!ok) return;
  }
  history = [];
  chart.reset();
  setOutput("Training… generate once a few steps have run.", true);
  setStatus("");
  setProgress(0, 1);
  els.stEta.textContent = "…";
  // Reset all the new metrics + start the elapsed clock + milestone tracker.
  runStartTime = performance.now();
  bestVal = Infinity;
  bestValStep = 0;
  milestonesHit = new Set();
  els.stPpl.textContent = "–";
  els.stBestVal.textContent = "–";
  els.stGap.textContent = "–";
  els.stElapsed.textContent = "0.0 s";
  clearMilestone();
  // Hide any previous run's verdict + reset the time-left label
  const verdict = document.getElementById("runVerdict");
  if (verdict) verdict.hidden = true;
  const timeLabel = document.getElementById("stTimeLabel");
  if (timeLabel) timeLabel.textContent = "Time left";
  startElapsedClock();
  savedThisRun = false;
  paused = false;
  els.pause.textContent = "Pause";
  setRunning(true);
  els.sample.disabled = false;
  lastConfig = readConfig();
  send({ type: "train", text, config: lastConfig });
});

els.pause.addEventListener("click", () => {
  paused = !paused;
  els.pause.textContent = paused ? "Resume" : "Pause";
  send({ type: paused ? "pause" : "resume" });
});

els.stop.addEventListener("click", () => send({ type: "stop" }));

// --- model actions dropdown -----------------------------------------------
function setupModelMenu(): void {
  const btn = document.getElementById("modelMenuBtn");
  const menu = document.getElementById("modelMenu");
  const container = btn?.parentElement;
  if (!btn || !menu || !container) return;

  const open = () => {
    menu.hidden = false;
    container.setAttribute("data-open", "true");
    btn.setAttribute("aria-expanded", "true");
  };
  const close = () => {
    menu.hidden = true;
    container.setAttribute("data-open", "false");
    btn.setAttribute("aria-expanded", "false");
  };
  const toggle = () => (menu.hidden ? open() : close());

  btn.addEventListener("click", (e) => { e.stopPropagation(); toggle(); });
  // Close on outside click or Esc.
  document.addEventListener("click", (e) => {
    if (!menu.hidden && !container.contains(e.target as Node)) close();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !menu.hidden) close();
  });
  // Close after any menu item is activated (except the inline number input).
  menu.querySelectorAll<HTMLElement>(".menu-item").forEach((item) => {
    item.addEventListener("click", (e) => {
      if ((e.target as HTMLElement).classList.contains("menu-input")) return;
      // Let the upload label trigger its file input before closing.
      setTimeout(close, 0);
    });
  });
}
setupModelMenu();

// --- copy sample output ---------------------------------------------------
const copyBtn = document.getElementById("copyOutput") as HTMLButtonElement | null;
if (copyBtn) {
  copyBtn.addEventListener("click", async () => {
    const text = els.output.textContent || "";
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      copyBtn.classList.add("copied");
      copyBtn.textContent = "✓";
      setTimeout(() => {
        copyBtn.classList.remove("copied");
        copyBtn.textContent = "⧉";
      }, 1500);
    } catch {
      // Clipboard unavailable; fall back to selecting the text.
      const range = document.createRange();
      range.selectNodeContents(els.output);
      const sel = window.getSelection();
      sel?.removeAllRanges();
      sel?.addRange(range);
    }
  });
}

// --- reset ---------------------------------------------------------------
// Reset config + corpus + chart + trained model. Asks for confirmation if
// there's something to lose (existing trained state or a non-default corpus).
const RESET_CORPUS = `A small model learns one thing at a time. First it learns that letters follow letters. After a space, another word begins. After a period, a capital letter often comes next.
Then it learns short words. The word "the" appears again and again, and so do "and", "of", "to", and "a". A model that predicts the next byte quickly learns to expect these common words, because guessing them lowers the loss on almost every line.
Training is slow at the start and fast in the middle. At the start the model knows nothing, so its loss sits near the value of a uniform guess. In the middle the loss falls quickly as the model picks up the easy patterns. Near the end it falls slowly, because what remains is hard.
The goal here is understanding, not a clever model. If you build it yourself, byte by byte and layer by layer, you will understand how every larger model works.`;

byId<HTMLButtonElement>("reset").addEventListener("click", () => {
  const hasModel = latestState !== null;
  const corpusChanged = els.corpus.value !== RESET_CORPUS;
  if (hasModel || corpusChanged) {
    if (!window.confirm("Reset everything? Your trained model, corpus changes, and config will be cleared.")) {
      return;
    }
  }

  // Reset config to Small preset (the default).
  els.sizePreset.value = "small";
  els.sizePreset.dispatchEvent(new Event("change"));

  // Reset corpus.
  els.corpus.value = RESET_CORPUS;

  // Reset tabs.
  switchTab("curated");
  els.hfDataset.value = "";
  setDataLoadStatus("");

  // Reset trained model + chart + stats.
  latestState = null;
  latestStateConfig = null;
  history = [];
  chart.reset();
  bestVal = Infinity;
  bestValStep = 0;
  milestonesHit = new Set();
  clearMilestone();
  setProgress(0, 1);
  setStatus("");
  setOutput("Train a model, then generate text here.", true);
  els.stStep.textContent = "0";
  els.stTrain.textContent = "–";
  els.stEta.textContent = "–";
  els.stPpl.textContent = "–";
  els.stElapsed.textContent = "–";
  els.stVal.textContent = "–";
  els.stBestVal.textContent = "–";
  els.stToks.textContent = "0";
  els.stGap.textContent = "–";
  els.downloadModel.disabled = true;
  els.downloadSafetensors.disabled = true;
  els.continueBtn.disabled = true;
  els.sample.disabled = true;

  // Clear OPFS-saved model + history.
  void saveState(new Uint8Array(0));
  void saveRun({ savedAt: new Date().toISOString(), config: DEFAULT_CONFIG, lossHistory: [] });

  // Strip query params so a refresh stays clean.
  window.history.replaceState(null, "", window.location.pathname);

  refreshEstimate("reset to defaults");
  refreshSampleNote();
  setModelStatus("✓ reset", "ok");
});

els.continueBtn.addEventListener("click", () => {
  const extra = parseInt(els.continueSteps.value, 10);
  if (!Number.isFinite(extra) || extra <= 0) {
    setStatus("extra steps must be a positive number", true);
    return;
  }
  savedThisRun = false;
  paused = false;
  els.pause.textContent = "Pause";
  setRunning(true);
  setStatus(`continuing for ${extra} more steps…`);
  // Don't reset history/chart — the new progress points extend the same curve.
  runStartTime = performance.now();
  startElapsedClock();
  send({ type: "continue", extraSteps: extra });
});

els.sample.addEventListener("click", () => {
  send({
    type: "sample",
    prompt: byId<HTMLInputElement>("prompt").value,
    tokens: parseInt(byId<HTMLInputElement>("genTokens").value, 10),
    temperature: parseFloat(byId<HTMLInputElement>("temp").value),
  });
  setOutput("generating…", true);
});

// --- size presets + pre-flight estimate -----------------------------------
let cachedCpuProbeMs = 12; // overwritten by init() once hardware is detected.

function populatePresetOptions(): void {
  for (const p of PRESETS) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.label;
    els.sizePreset.appendChild(opt);
  }
}
populatePresetOptions();

function applyPreset(id: string): void {
  const preset = PRESETS.find((p) => p.id === id);
  if (!preset) return;
  byId<HTMLInputElement>("layers").value = String(preset.layers);
  // Make sure the d_model exists in the <select>; if not, add an option.
  const dSel = byId<HTMLSelectElement>("dModel");
  if (!Array.from(dSel.options).some((o) => o.value === String(preset.dModel))) {
    const opt = document.createElement("option");
    opt.value = String(preset.dModel);
    opt.textContent = String(preset.dModel);
    dSel.appendChild(opt);
  }
  dSel.value = String(preset.dModel);
  byId<HTMLInputElement>("ctx").value = String(preset.ctx);
  byId<HTMLInputElement>("batch").value = String(preset.batch);
  byId<HTMLInputElement>("maxSteps").value = String(preset.maxSteps);
  byId<HTMLSelectElement>("backend").value = preset.recommendedBackend;
  refreshEstimate(preset.note);
}

/**
 * Render a context-aware "what to expect" note for the Sample card. Depends on
 * current model size, corpus size in bytes, and (if a run has happened) the
 * best loss reached. This replaces a static "garbled is expected" paragraph
 * with something the user can read against the actual numbers in front of them.
 */
function refreshSampleNote(): void {
  const noteEl = document.getElementById("sampleNoteContent");
  if (!noteEl) return;
  const cfg = readConfig();
  const params = estimateParams(cfg.layers, cfg.dModel, cfg.ctx);
  const corpusBytes = els.corpus.value.length;
  const corpusKb = corpusBytes / 1024;
  const corpusLabel = corpusBytes < 5000
    ? `${corpusBytes} bytes`
    : corpusBytes < 1_000_000
      ? `${corpusKb.toFixed(0)} KB`
      : `${(corpusBytes / 1_000_000).toFixed(1)} MB`;

  // Reasonable "params-per-byte-seen" thresholds. With ~6 epochs over the
  // corpus, total bytes seen ≈ corpusBytes × 6. The model can usefully exploit
  // ~50 bytes per parameter (the empirical "Chinchilla-ish" rule, scaled down).
  const bytesPerParamSeen = (corpusBytes * 6) / Math.max(params, 1);
  const haveTrained = history.length > 0;
  const finalTrain = haveTrained ? history[history.length - 1].trainLoss : null;
  const finalVal = haveTrained ? history[history.length - 1].valLoss : null;

  let regime: string;
  let detail: string;

  if (params < 200_000 && corpusBytes < 5000) {
    regime = "Microscope mode";
    detail = `${formatParams(params)} params on ${corpusLabel} — the model has nothing to chew on. Expect letter shapes and nothing else. The win here is watching the loss curve fall; the prose is incidental.`;
  } else if (params < 1_000_000 && corpusBytes < 50_000) {
    regime = "Tiny / Tiny";
    detail = `${formatParams(params)} params on ${corpusLabel}. The model can learn letter co-occurrence and the existence of common words. Grammar requires more data than this — sentences will read like word-salad. The loss curve is the honest signal.`;
  } else if (params < 1_000_000 && corpusBytes >= 50_000) {
    regime = "Tiny model, decent corpus";
    detail = `${formatParams(params)} params on ${corpusLabel} — there's enough text, but the model is too narrow to capture it. Loss will plateau early. Try Medium or Large preset to get more out of this corpus.`;
  } else if (params >= 1_000_000 && corpusBytes < 50_000) {
    regime = "Big model, thin data";
    detail = `${formatParams(params)} params on ${corpusLabel} — model has more capacity than the corpus has structure. Train loss will crash to near-zero (memorisation); val loss will lag, and that gap tells you the model is reciting, not generalising. Pick a bigger Hugging Face dataset.`;
  } else if (bytesPerParamSeen < 30) {
    regime = "Reasonable, but data-starved";
    detail = `${formatParams(params)} params seeing ~${bytesPerParamSeen.toFixed(0)} bytes per parameter — under the Chinchilla-ish floor of ~50. You'll see word shapes and some local grammar; expect val loss to plateau around 2.0–2.5.`;
  } else {
    regime = "Well-matched";
    detail = `${formatParams(params)} params on ${corpusLabel}, ~${bytesPerParamSeen.toFixed(0)} bytes per param — well-matched. With enough steps you should see val loss drop below 2.0 and the samples start looking word-shaped (real grammar emerges around val 1.5).`;
  }

  let outcomeLine = "";
  if (haveTrained && finalTrain != null) {
    const bits = (finalTrain / Math.log(2)).toFixed(2);
    const ppx = Math.exp(finalTrain).toFixed(1);
    const valBit = finalVal != null
      ? `, val ${(finalVal / Math.log(2)).toFixed(2)} bits`
      : "";
    let interpret = "";
    if (finalTrain > 3.0) interpret = "Still in the 'letter pairs' regime.";
    else if (finalTrain > 2.0) interpret = "Letter frequencies + short n-grams learned.";
    else if (finalTrain > 1.5) interpret = "Word shapes emerging; grammar still rough.";
    else if (finalTrain > 1.0) interpret = "Local grammar should be visible. Long-range still random.";
    else interpret = "Substantial memorisation — train ↔ val gap is the honest signal now.";
    outcomeLine = `<br><br><strong>This run:</strong> ${bits} bits/byte${valBit} · perplexity ${ppx}. ${interpret}`;
  }

  noteEl.innerHTML =
    `<strong>${regime}.</strong> ${detail}${outcomeLine}` +
    `<br><br>` +
    `<span style="color: var(--faint); font-size: 12.5px;">` +
    `For genuinely grammatical English you need at least val loss ~1.5 — roughly ` +
    `Medium/Large preset on a real dataset (TinyStories, Tiny Shakespeare). ` +
    `For coherent prose: 10M+ params on MB of text via the ` +
    `<a href="#sec-diagnostics">Python CLI</a>.` +
    `</span>`;
}

function refreshEstimate(extraNote = ""): void {
  const cfg = readConfig();
  const params = estimateParams(cfg.layers, cfg.dModel, cfg.ctx);
  els.estParams.textContent =
    `${formatParams(params)} params · ${cfg.layers}L · d_model ${cfg.dModel} · ctx ${cfg.ctx} · ${cfg.heads} heads`;
  if (cfg.backend === "webgpu") {
    els.estTime.textContent = "depends on your GPU";
    els.estNote.textContent =
      extraNote || "WebGPU runs aren't pre-flight benchmarked here — start the run to see real tokens/sec.";
    return;
  }
  const secs = estimateTrainSeconds(
    cfg.layers, cfg.dModel, cfg.ctx, cfg.batchSize, cfg.maxSteps, cachedCpuProbeMs,
  );
  els.estTime.textContent = `~${formatDuration(secs)} on this machine (WASM SIMD, single-threaded)`;
  els.estNote.textContent = extraNote;
}

els.sizePreset.addEventListener("change", () => {
  const id = els.sizePreset.value;
  const hp = document.getElementById("hyperparamDetails") as HTMLDetailsElement | null;
  if (id === "custom") {
    if (hp) hp.open = true;
    refreshEstimate("");
    return;
  }
  if (hp) hp.open = false;
  applyPreset(id);
});

for (const id of ["layers", "dModel", "ctx", "batch", "maxSteps", "backend"]) {
  byId(id).addEventListener("change", () => {
    if (els.sizePreset.value !== "custom") els.sizePreset.value = "custom";
    refreshEstimate("");
    refreshSampleNote();
  });
  byId(id).addEventListener("input", () => {
    if (els.sizePreset.value !== "custom") els.sizePreset.value = "custom";
    refreshEstimate("");
    refreshSampleNote();
  });
}
// Corpus text changes affect the sample-note regime too.
els.corpus.addEventListener("input", () => refreshSampleNote());

// --- personalised "what this machine means for you" ----------------------
interface PersonaliseInput {
  caps: Awaited<ReturnType<typeof detectCapabilities>>;
  hw: ReturnType<typeof detectHardware>;
  rec: ModelRecommendation;
  browser: ReturnType<typeof detectBrowser>;
}

function classifyCpuSpeed(probeMs: number): { tier: string; verdict: string } {
  if (probeMs < 8) return { tier: "very fast", verdict: "near the in-browser ceiling — try Large or XL." };
  if (probeMs < 18) return { tier: "fast", verdict: "Medium runs in a few minutes; Large is feasible if patient." };
  if (probeMs < 40) return { tier: "comfortable", verdict: "Small (~360k) is the sweet spot — under a minute." };
  return { tier: "modest", verdict: "stick to Tiny (~70k); larger sizes will be slow." };
}

function renderPersonalisation({ caps, hw, rec, browser }: PersonaliseInput): void {
  const cpu = classifyCpuSpeed(hw.cpuProbeMs);
  const tps = 1.7e10 / (hw.cpuProbeMs * rec.approxParams);
  const recRunSec = (rec.maxSteps * 16 * rec.ctx) / Math.max(tps, 1);
  const recRunStr = recRunSec < 90 ? `~${Math.round(recRunSec)} s` : `~${(recRunSec / 60).toFixed(1)} min`;

  const ramLine = hw.deviceMemoryGB
    ? `Browser reports ${hw.deviceMemoryGB >= 8 ? "≥" : "~"}${hw.deviceMemoryGB} GB RAM available — that's a privacy-capped value, your real RAM may be higher. Either way, plenty for in-browser training; the wall here is CPU speed, not memory.`
    : `Your browser doesn't expose RAM info (Safari and Firefox don't implement <code>navigator.deviceMemory</code>). It almost certainly isn't the bottleneck — CPU is.`;

  const webgpuLine = caps.webgpu
    ? `Your browser supports <strong>WebGPU</strong>. <em>One measured datapoint: ~7× faster than WASM on Apple M-series</em>. The kernels are parity-checked, so correctness is solid. <strong>Switch the backend to WebGPU</strong> and you'll likely see a similar speedup. (Yours may vary — post your tokens/sec to help calibrate.)`
    : `Your browser <strong>doesn't expose WebGPU</strong>. You're CPU-only for now. Chrome / Edge 113+ and Safari 18+ have it; Firefox is gated behind a flag.`;

  const simdLine = caps.wasmSimd
    ? `<strong>WASM SIMD is on</strong> — your matmuls run ~1.6× faster than the scalar path. Already accelerated.`
    : `<strong>WASM SIMD is unavailable</strong> in your browser. You're running scalar WASM — about 1.6× slower than what most modern browsers see.`;

  const coiLine = caps.crossOriginIsolated
    ? `<strong>Cross-origin isolated: yes.</strong> Your page is set up for SharedArrayBuffer — meaning if multi-threaded WASM ships, your browser is already ready to run it.`
    : `<strong>Cross-origin isolated: no.</strong> The deployment isn't serving the COOP/COEP headers SharedArrayBuffer needs. Single-threaded only — for now.`;

  const browserNote = browser.note ? `${browser.name} — ${browser.note}` : browser.name;

  const fastestLeverHtml = !caps.webgpu
    ? `your biggest open speed lever is <strong>enabling WebGPU</strong> (upgrade browser or try Chrome / Safari 18+) — potential 3–10×.`
    : hw.cpuProbeMs >= 18
      ? `your biggest open speed lever is <strong>running the Python CLI locally</strong> (CUDA / Apple MPS) — 50–100× faster than this browser path.`
      : `your machine is already near the ceiling for in-browser. Real gains now come from <strong>kernel-level work</strong> (tiled matmul, multi-threaded WASM) — see <a href="/roadmap">the roadmap</a>.`;

  els.mpPreview.textContent = `${cpu.tier} · ${hw.cores}c · ${recRunStr} for the suggested model`;

  els.mpBody.innerHTML = `
    <div class="mp-row">
      <span class="mp-label">CPU speed</span>
      <span class="mp-value">
        Your CPU probe came in at <code>${hw.cpuProbeMs.toFixed(0)} ms</code> for a 160³ matmul — that's
        <strong>${cpu.tier}</strong>. ${cpu.verdict}
      </span>
    </div>
    <div class="mp-row">
      <span class="mp-label">Recommended run</span>
      <span class="mp-value">
        For <strong>${rec.layers} layers · d_model ${rec.dModel} · ctx ${rec.ctx}</strong>
        (~${(rec.approxParams / 1000).toFixed(0)}k params), expect
        <strong>${recRunStr}</strong> end-to-end on this machine.
        Estimated throughput: <code>~${Math.round(tps).toLocaleString()}</code> tokens/sec.
      </span>
    </div>
    <div class="mp-row">
      <span class="mp-label">Memory</span>
      <span class="mp-value">${ramLine}</span>
    </div>
    <div class="mp-row">
      <span class="mp-label">WebGPU</span>
      <span class="mp-value">${webgpuLine}</span>
    </div>
    <div class="mp-row">
      <span class="mp-label">WASM SIMD</span>
      <span class="mp-value">${simdLine}</span>
    </div>
    <div class="mp-row">
      <span class="mp-label">Threading</span>
      <span class="mp-value">${coiLine}</span>
    </div>
    <div class="mp-row">
      <span class="mp-label">Browser</span>
      <span class="mp-value">${browserNote}</span>
    </div>
    <div class="mp-row">
      <span class="mp-label">Your fastest lever</span>
      <span class="mp-value">${fastestLeverHtml}</span>
    </div>
  `;
}

// --- live JS heap polling (Chromium only) --------------------------------
interface PerfWithMemory {
  memory?: { usedJSHeapSize: number; jsHeapSizeLimit: number };
}
function startHeapPoll(): void {
  const perfMem = performance as unknown as PerfWithMemory;
  if (!perfMem.memory) {
    const pillEl = document.getElementById("heapPill");
    if (pillEl) pillEl.hidden = true;
    return;
  }
  const update = () => {
    const m = perfMem.memory;
    if (!m) return;
    const usedMb = m.usedJSHeapSize / (1024 * 1024);
    const limitMb = m.jsHeapSizeLimit / (1024 * 1024);
    const pillEl = document.getElementById("heapPill");
    if (pillEl) {
      pillEl.textContent = `heap ${usedMb.toFixed(0)} / ${limitMb.toFixed(0)} MB`;
    }
  };
  update();
  setInterval(update, 2000);
}

// --- model file download / upload ----------------------------------------
//
// On-disk format (extension .tinygpt) — a tiny self-describing binary:
//   4 bytes "TGPT"             magic
//   4 bytes uint32 LE          version = 1
//   4 bytes uint32 LE          JSON header length
//   N bytes UTF-8              JSON header (config + metadata)
//   M bytes                    serialized model state
//
// The JSON header carries the RunConfig so the receiver can configure the
// worker identically before calling restore.
const MODEL_MAGIC = "TGPT";
const MODEL_VERSION = 2; // v2 adds: manifest, loss history, sample, optimizer state flag
let latestState: ArrayBuffer | null = null;
let latestStateConfig: RunConfig | null = null;
let lastSampleText: string = "";

function setModelStatus(text: string, kind: "" | "ok" | "error" = ""): void {
  els.modelStatus.textContent = text;
  els.modelStatus.className = `model-status${kind ? ` ${kind}` : ""}`;
}

/**
 * Build the named-tensor manifest that describes what's in the flat state
 * buffer. Mirrors `collect_params` in wasm/src/model.cpp — token + position
 * embeddings, final layernorm, then per-layer attention + MLP. Each entry has
 * an offset (in float units, into the per-param region) and a shape.
 *
 * The C++ export writes 3 buffers per param (w, m, v). We surface only the
 * weight buffer here — Adam momentum is private to a run and not useful to
 * share. The `includesOptimizerState` flag in the header tells the receiver
 * whether m/v follow each w or not.
 */
interface TensorEntry {
  name: string;
  shape: number[];
  /** Float offset within the WEIGHT-ONLY view of the state (post-step prefix). */
  floatOffset: number;
}

function buildManifest(cfg: RunConfig): TensorEntry[] {
  const { layers: L, dModel: C, ctx, heads: _H } = cfg;
  const dMlp = cfg.dMlp ?? C * 4;
  const V = 256; // byte-level vocab is hard-coded in this project
  const ents: TensorEntry[] = [];
  let off = 0;
  const push = (name: string, shape: number[]) => {
    ents.push({ name, shape, floatOffset: off });
    off += shape.reduce((a, b) => a * b, 1);
  };
  push("token_embedding.weight", [V, C]);
  push("position_embedding.weight", [ctx, C]);
  push("ln_final.weight", [C]);
  push("ln_final.bias", [C]);
  for (let i = 0; i < L; i++) {
    push(`blocks.${i}.ln1.weight`, [C]);
    push(`blocks.${i}.ln1.bias`, [C]);
    push(`blocks.${i}.attn.q_proj.weight`, [C, C]);
    push(`blocks.${i}.attn.q_proj.bias`, [C]);
    push(`blocks.${i}.attn.k_proj.weight`, [C, C]);
    push(`blocks.${i}.attn.k_proj.bias`, [C]);
    push(`blocks.${i}.attn.v_proj.weight`, [C, C]);
    push(`blocks.${i}.attn.v_proj.bias`, [C]);
    push(`blocks.${i}.attn.o_proj.weight`, [C, C]);
    push(`blocks.${i}.attn.o_proj.bias`, [C]);
    push(`blocks.${i}.ln2.weight`, [C]);
    push(`blocks.${i}.ln2.bias`, [C]);
    push(`blocks.${i}.mlp.fc_in.weight`, [dMlp, C]);
    push(`blocks.${i}.mlp.fc_in.bias`, [dMlp]);
    push(`blocks.${i}.mlp.fc_out.weight`, [C, dMlp]);
    push(`blocks.${i}.mlp.fc_out.bias`, [C]);
  }
  return ents;
}

/**
 * File format (v2):
 *   4   "TGPT" magic
 *   4   uint32 LE version (= 2)
 *   4   uint32 LE JSON header length
 *   N   UTF-8 JSON header — { config, manifest, lossHistory[], sample,
 *                              finalLoss, savedAt, includesOptimizerState }
 *   M   raw float32 weight buffer (with the 4-byte int32 step prefix +
 *       per-param triplets of w/m/v exactly as the WASM exports it).
 */
function encodeModelFile(config: RunConfig, state: ArrayBuffer): Blob {
  const final = history[history.length - 1];
  const headerObj = {
    version: MODEL_VERSION,
    savedAt: new Date().toISOString(),
    config,
    manifest: buildManifest(config),
    includesOptimizerState: true, // current WASM export bundles Adam m, v
    stateByteLength: state.byteLength,
    lossHistory: history.slice(-512).map((p) => ({
      step: p.step,
      train: +p.trainLoss.toFixed(4),
      val: p.valLoss != null ? +p.valLoss.toFixed(4) : null,
    })),
    finalLoss: final ? { step: final.step, train: final.trainLoss, val: final.valLoss ?? null } : null,
    sample: lastSampleText.slice(0, 320),
    bestVal: Number.isFinite(bestVal) ? { loss: bestVal, step: bestValStep } : null,
    project: "https://github.com/sarthakagrawal927/tinygpt",
  };
  const headerJson = JSON.stringify(headerObj);
  const headerBytes = new TextEncoder().encode(headerJson);
  const prefix = new ArrayBuffer(12);
  const view = new DataView(prefix);
  new Uint8Array(prefix, 0, 4).set(new TextEncoder().encode(MODEL_MAGIC));
  view.setUint32(4, MODEL_VERSION, true);
  view.setUint32(8, headerBytes.byteLength, true);
  return new Blob([prefix, headerBytes, state], { type: "application/octet-stream" });
}

async function decodeModelFile(file: File): Promise<{ config: RunConfig; state: ArrayBuffer; header: Record<string, unknown> }> {
  const buf = await file.arrayBuffer();
  if (buf.byteLength < 12) throw new Error("file too small to be a .tinygpt model");
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
  if (magic !== MODEL_MAGIC) throw new Error("not a .tinygpt model file (bad magic)");
  const view = new DataView(buf);
  const version = view.getUint32(4, true);
  if (version !== 1 && version !== 2) throw new Error(`unsupported model version ${version}`);
  const headerLen = view.getUint32(8, true);
  if (12 + headerLen > buf.byteLength) throw new Error("model header is malformed");
  const headerJson = new TextDecoder().decode(new Uint8Array(buf, 12, headerLen));
  const header = JSON.parse(headerJson) as { config: RunConfig };
  const state = buf.slice(12 + headerLen);
  return { config: header.config, state, header: header as Record<string, unknown> };
}

els.downloadModel.addEventListener("click", () => {
  if (!latestState || !latestStateConfig) {
    setModelStatus("nothing trained yet — start a run first", "error");
    return;
  }
  const blob = encodeModelFile(latestStateConfig, latestState);
  const sizeKb = (blob.size / 1024).toFixed(0);
  const filename = `tinygpt-${latestStateConfig.layers}L-d${latestStateConfig.dModel}-ctx${latestStateConfig.ctx}.tinygpt`;
  triggerDownload(blob, filename);
  setModelStatus(`✓ saved ${filename} (${sizeKb} KB)`, "ok");
});

els.downloadSafetensors.addEventListener("click", () => {
  if (!latestState || !latestStateConfig) {
    setModelStatus("nothing trained yet — start a run first", "error");
    return;
  }
  try {
    const blob = encodeSafetensorsFile(latestStateConfig, latestState);
    const sizeKb = (blob.size / 1024).toFixed(0);
    const filename = `tinygpt-${latestStateConfig.layers}L-d${latestStateConfig.dModel}-ctx${latestStateConfig.ctx}.safetensors`;
    triggerDownload(blob, filename);
    setModelStatus(`✓ saved ${filename} (${sizeKb} KB) — load in Python with safetensors.numpy.load_file`, "ok");
  } catch (err) {
    setModelStatus(`couldn't export: ${err instanceof Error ? err.message : String(err)}`, "error");
  }
});

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/**
 * Encode the model as a standard safetensors file. Adopts huggingface's format
 * verbatim: u64 header length, JSON header mapping name → {dtype, shape,
 * data_offsets: [start, end]}, then concatenated tensor data. Names match the
 * PyTorch state_dict from python_ref/model.py — so the file is loadable with
 * `safetensors.numpy.load_file` and the dict drops straight into PyTorch.
 *
 * We only serialise weights (skip the Adam m/v moments) — sharing Adam state
 * is rarely useful and triples the file size.
 */
function encodeSafetensorsFile(config: RunConfig, state: ArrayBuffer): Blob {
  const manifest = buildManifest(config);
  // The C++ state buffer is: int32 step + per-param triplets of [w, m, v].
  // Each param's `w` is at offset `4 + (cumulativeFloats * 3) * 4` from the
  // start, where cumulativeFloats counts only the *first* of the three blocks.
  // i.e. for param i with `nFloats` floats and prior cumulative `cum`:
  //   wOff = 4 + cum * 3 * 4
  //   mOff = wOff + nFloats * 4
  //   vOff = mOff + nFloats * 4
  // We extract only `w`.

  const tensors: { name: string; shape: number[]; bytes: Uint8Array }[] = [];
  let cum = 0;
  for (const t of manifest) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    const wByteOff = 4 + cum * 3 * 4;
    if (wByteOff + n * 4 > state.byteLength) {
      throw new Error(`manifest exceeds state buffer at ${t.name}`);
    }
    const bytes = new Uint8Array(state, wByteOff, n * 4).slice();
    tensors.push({ name: t.name, shape: t.shape, bytes });
    cum += n;
  }

  // Build safetensors JSON header.
  const headerObj: Record<string, unknown> = {
    __metadata__: {
      project: "https://github.com/sarthakagrawal927/tinygpt",
      tinygpt_version: "1",
      config: JSON.stringify(config),
    },
  };
  let dataOffset = 0;
  for (const t of tensors) {
    headerObj[t.name] = {
      dtype: "F32",
      shape: t.shape,
      data_offsets: [dataOffset, dataOffset + t.bytes.byteLength],
    };
    dataOffset += t.bytes.byteLength;
  }
  let headerJson = JSON.stringify(headerObj);
  // safetensors recommends padding the header to an 8-byte boundary.
  while (headerJson.length % 8 !== 0) headerJson += " ";
  const headerBytes = new TextEncoder().encode(headerJson);
  const prefix = new ArrayBuffer(8);
  new DataView(prefix).setBigUint64(0, BigInt(headerBytes.byteLength), true);

  return new Blob(
    [prefix, headerBytes as BlobPart, ...tensors.map((t) => t.bytes as BlobPart)],
    { type: "application/octet-stream" },
  );
}

async function loadModelFromFile(file: File, label = file.name): Promise<void> {
  setModelStatus(`loading ${label}…`);
  try {
    const { config, state, header } = await decodeModelFile(file);
    const meta = header as {
      finalLoss?: { step: number; train: number; val: number | null };
      sample?: string;
      lossHistory?: { step: number; train: number; val: number | null }[];
      savedAt?: string;
    };
    if (meta.finalLoss) {
      const trainTxt = meta.finalLoss.train.toFixed(3);
      const valTxt = meta.finalLoss.val != null ? `, val ${meta.finalLoss.val.toFixed(3)}` : "";
      setModelStatus(`✓ ${label} · final loss ${trainTxt}${valTxt} @ step ${meta.finalLoss.step}`, "ok");
    }
    if (meta.lossHistory && meta.lossHistory.length > 0) {
      history = meta.lossHistory.map((p) => ({ step: p.step, trainLoss: p.train, valLoss: p.val ?? undefined }));
      chart.reset();
      for (const pt of history) chart.addPoint(pt);
      // Populate the primary stats with the loaded model's final state so
      // the Watch screen reads as "this is what you'd see if you'd trained
      // this from scratch" rather than empty placeholders.
      const last = history[history.length - 1];
      const final = meta.finalLoss;
      els.stStep.textContent = `${last.step} / ${last.step}`;
      els.stTrain.textContent = last.trainLoss.toFixed(4);
      els.stEta.textContent = "done";
      const timeLabel = document.getElementById("stTimeLabel");
      if (timeLabel) timeLabel.textContent = "Trained";
      els.stElapsed.textContent = "loaded";
      els.stPpl.textContent = formatPerplexity(last.trainLoss);
      if (last.valLoss != null) {
        els.stVal.textContent = last.valLoss.toFixed(4);
        const best = history.reduce(
          (b, p) => (p.valLoss != null && p.valLoss < b.loss ? { loss: p.valLoss, step: p.step } : b),
          { loss: Infinity, step: 0 },
        );
        if (Number.isFinite(best.loss)) {
          els.stBestVal.textContent = `${best.loss.toFixed(4)} @ ${best.step}`;
          bestVal = best.loss;
          bestValStep = best.step;
        }
        els.stGap.textContent = (last.valLoss - last.trainLoss).toFixed(3);
      }
      void final; // reserved for future use
    }
    if (meta.sample) {
      typewriteOutput(meta.sample);
    }
    byId<HTMLInputElement>("layers").value = String(config.layers);
    const dSel = byId<HTMLSelectElement>("dModel");
    if (!Array.from(dSel.options).some((o) => o.value === String(config.dModel))) {
      const opt = document.createElement("option");
      opt.value = String(config.dModel);
      opt.textContent = String(config.dModel);
      dSel.appendChild(opt);
    }
    dSel.value = String(config.dModel);
    byId<HTMLInputElement>("ctx").value = String(config.ctx);
    byId<HTMLInputElement>("batch").value = String(config.batchSize);
    byId<HTMLSelectElement>("backend").value = config.backend;
    if (els.sizePreset.value !== "custom") els.sizePreset.value = "custom";
    refreshEstimate("loaded from model file");
    refreshSampleNote();

    lastConfig = config;
    latestState = state.slice(0);
    latestStateConfig = config;
    els.downloadModel.disabled = false;
    els.downloadSafetensors.disabled = false;
    els.continueBtn.disabled = false;
    worker.postMessage({ type: "restore", state, config }, [state]);
    if (!els.modelStatus.classList.contains("ok")) {
      setModelStatus(`✓ loaded ${label} — ready to sample or continue training`, "ok");
    }
  } catch (err) {
    setModelStatus(`couldn't load: ${err instanceof Error ? err.message : String(err)}`, "error");
    throw err;
  }
}

els.uploadModel.addEventListener("change", async () => {
  const file = els.uploadModel.files?.[0];
  if (!file) return;
  try {
    await loadModelFromFile(file);
  } catch { /* status already set */ }
  els.uploadModel.value = "";
});

// --- share & restore from URL --------------------------------------------
const SHARE_KEYS = ["layers", "dModel", "ctx", "maxSteps", "batch", "lr", "backend"] as const;

function buildShareUrl(): string {
  const params = new URLSearchParams();
  for (const k of SHARE_KEYS) {
    const el = byId<HTMLInputElement | HTMLSelectElement>(k);
    if (el && el.value) params.set(k, el.value);
  }
  // Share the dataset choice too — but never the corpus text itself (URL-limit
  // hostile and often private). Pull from whichever data-source tab is active.
  const activeTab = document.querySelector<HTMLButtonElement>(".tab-btn.active")?.dataset.tab;
  if (activeTab === "custom") {
    const ds = els.hfCustomDataset.value.trim();
    if (ds) {
      params.set("hf", "custom");
      params.set("hfDataset", ds);
      params.set("hfConfig", els.hfCustomConfig.value.trim() || "default");
      params.set("hfSplit", els.hfCustomSplit.value.trim() || "train");
      params.set("hfColumn", els.hfCustomColumn.value.trim() || "text");
    }
  } else if (activeTab === "curated" && els.hfDataset.value) {
    params.set("hf", els.hfDataset.value);
  }
  const url = new URL(window.location.href);
  url.search = params.toString();
  url.hash = "";
  return url.toString();
}

function applyConfigFromUrl(): boolean {
  const params = new URLSearchParams(window.location.search);
  let applied = false;
  for (const k of SHARE_KEYS) {
    const v = params.get(k);
    if (v == null) continue;
    const el = byId<HTMLInputElement | HTMLSelectElement>(k);
    if (el) {
      el.value = v;
      applied = true;
    }
  }
  // Dataset restoration — drive the relevant tab + control.
  const hfId = params.get("hf");
  if (hfId === "custom") {
    const ds = params.get("hfDataset");
    if (ds) {
      els.hfCustomDataset.value = ds;
      els.hfCustomConfig.value = params.get("hfConfig") || "default";
      els.hfCustomSplit.value = params.get("hfSplit") || "train";
      els.hfCustomColumn.value = params.get("hfColumn") || "text";
      switchTab("custom");
      els.hfCustomLoad.click();
      applied = true;
    }
  } else if (hfId && HF_CATALOG.some((d) => d.id === hfId)) {
    switchTab("curated");
    els.hfDataset.value = hfId;
    els.hfDataset.dispatchEvent(new Event("change"));
    applied = true;
  }
  if (applied) {
    // The size preset doesn't correspond to a shared config — mark as Custom.
    if (els.sizePreset.value !== "custom") els.sizePreset.value = "custom";
    refreshEstimate("loaded from shared link");
  }
  return applied;
}

let toastTimer: number | undefined;
function showToast(message: string): void {
  els.shareToast.textContent = message;
  els.shareToast.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { els.shareToast.hidden = true; }, 2500);
}

els.shareBtn.addEventListener("click", async () => {
  const url = buildShareUrl();
  try {
    await navigator.clipboard.writeText(url);
    showToast("✓ Link copied to clipboard");
  } catch {
    // Clipboard may be unavailable (non-HTTPS, permission denied). Fall back
    // to updating the address bar so the user can copy manually.
    window.history.replaceState(null, "", url);
    showToast("Couldn't copy — URL updated in address bar");
  }
});

// --- "notify me when done" ------------------------------------------------
function setNotifyStatus(text: string, kind: "" | "ok" | "error" = ""): void {
  els.notifyStatus.textContent = text;
  els.notifyStatus.className = `notify-status${kind ? ` ${kind}` : ""}`;
}

function notificationsSupported(): boolean {
  return typeof Notification !== "undefined";
}

els.notify.addEventListener("change", async () => {
  if (!els.notify.checked) {
    setNotifyStatus("");
    return;
  }
  if (!notificationsSupported()) {
    els.notify.checked = false;
    setNotifyStatus("not supported in this browser", "error");
    return;
  }
  if (Notification.permission === "granted") {
    setNotifyStatus("on", "ok");
    return;
  }
  if (Notification.permission === "denied") {
    els.notify.checked = false;
    setNotifyStatus("blocked — re-enable in browser settings", "error");
    return;
  }
  const result = await Notification.requestPermission();
  if (result === "granted") {
    setNotifyStatus("on", "ok");
  } else {
    els.notify.checked = false;
    setNotifyStatus("permission denied", "error");
  }
});

function fireDoneNotification(message: string): void {
  if (!els.notify.checked || !notificationsSupported() || Notification.permission !== "granted") return;
  // Always fire — the user explicitly opted in. Skipping when the tab is
  // focused turned out to be surprising more than helpful.
  try {
    new Notification("TinyGPT — training complete", {
      body: message,
      icon: "/favicon.svg",
      tag: "tinygpt-training",
    });
  } catch {
    // Some browsers throw on Notification construction in non-secure contexts.
  }
}

// --- WebGPU matmul benchmark (milestone 6) --------------------------------
els.bench.addEventListener("click", async () => {
  els.bench.disabled = true;
  els.benchOut.textContent = "initialising WebGPU…";
  try {
    const device = await initWebGPU();
    if (!device) {
      els.benchOut.textContent = "WebGPU is not available in this browser.";
      return;
    }
    els.benchOut.textContent = "loading the WASM matmul kernel…";
    const backend = await TinyGptBackend.load();
    els.benchOut.textContent = "running 384×384 matmul on WebGPU and WASM…";
    const r = await benchmarkMatmul(
      device,
      (a, b, M, K, N) => backend.matmul(a, b, M, K, N),
      384,
    );
    els.benchOut.textContent =
      `${r.size}×${r.size} matmul — ${r.parityOk ? "parity OK ✓" : "PARITY FAILED"} ` +
      `(max abs error ${r.maxAbsError.toExponential(2)})\n` +
      `WASM ${r.refMs.toFixed(1)} ms · WebGPU ${r.gpuMs.toFixed(1)} ms · ` +
      `${r.speedup.toFixed(1)}× speed-up`;
  } catch (err) {
    els.benchOut.textContent = `benchmark error: ${
      err instanceof Error ? err.message : String(err)
    }`;
  } finally {
    els.bench.disabled = false;
  }
});

// --- Hugging Face dataset picker ------------------------------------------
// (Custom-HF flow now has its own tab — no sentinel value needed in the dropdown.)

for (const d of HF_CATALOG) {
  const opt = document.createElement("option");
  opt.value = d.id;
  opt.textContent = `${d.label} — ${d.blurb}`;
  els.hfDataset.appendChild(opt);
}
// Custom HF dataset now has its own tab — no need for a dropdown sentinel.

let hfLoadToken = 0;
let lastHfEntry: import("./datasets").HfDataset | null = null;
const HF_TOKEN_KEY = "tinygpt.hf.token";

function getStoredHfToken(): string {
  try { return localStorage.getItem(HF_TOKEN_KEY) ?? ""; } catch { return ""; }
}
function setStoredHfToken(value: string): void {
  try {
    if (value) localStorage.setItem(HF_TOKEN_KEY, value);
    else localStorage.removeItem(HF_TOKEN_KEY);
  } catch { /* private mode */ }
}

async function loadHfDataset(entry: import("./datasets").HfDataset): Promise<void> {
  lastHfEntry = entry;
  const callToken = ++hfLoadToken;
  els.hfDataset.disabled = true;
  els.hfCustomLoad.disabled = true;
  setDataLoadStatus(`fetching ${entry.label}…`);
  const hfToken = getStoredHfToken();
  const maxChars = parseInt(els.fetchSize.value, 10) || 2_000_000;
  try {
    const text = await fetchHfText(entry, maxChars, (chars) => {
      if (callToken !== hfLoadToken) return;
      setDataLoadStatus(`fetching ${entry.label}… ${(chars / 1024).toFixed(0)} KB / ${(maxChars / 1024).toFixed(0)} KB`);
    }, hfToken || undefined);
    if (callToken !== hfLoadToken) return;
    els.corpus.value = text;
    setDataLoadStatus(`✓ loaded ${Math.round(text.length / 1000)} KB from ${entry.label} · ${entry.license}`, "ok");
    els.hfTokenRow.hidden = !hfToken;
    refreshSampleNote();
  } catch (err) {
    if (callToken !== hfLoadToken) return;
    const message = err instanceof Error ? err.message : String(err);
    setDataLoadStatus(`✗ ${message}`, "error");
    if (err instanceof HfFetchError && (err.kind === "auth" || err.kind === "ratelimit")) {
      els.hfTokenRow.hidden = false;
      els.hfToken.focus();
    }
  } finally {
    if (callToken === hfLoadToken) {
      els.hfDataset.disabled = false;
      els.hfCustomLoad.disabled = false;
    }
  }
}

// Restore stored token on load; if present, surface the row so the user can
// see / clear it.
{
  const existing = getStoredHfToken();
  if (existing) {
    els.hfToken.value = existing;
    els.hfTokenRow.hidden = false;
  }
}

els.hfTokenSave.addEventListener("click", () => {
  const value = els.hfToken.value.trim();
  setStoredHfToken(value);
  setDataLoadStatus(value ? "token saved — retrying…" : "token cleared");
  if (value && lastHfEntry) void loadHfDataset(lastHfEntry);
});

els.hfTokenClear.addEventListener("click", () => {
  els.hfToken.value = "";
  setStoredHfToken("");
  setDataLoadStatus("token cleared");
});

els.hfDataset.addEventListener("change", () => {
  const value = els.hfDataset.value;
  if (!value) {
    setDataLoadStatus("");
    return;
  }
  const entry = HF_CATALOG.find((d) => d.id === value);
  if (entry) void loadHfDataset(entry);
});

els.hfCustomLoad.addEventListener("click", () => {
  const dataset = els.hfCustomDataset.value.trim();
  if (!dataset || !dataset.includes("/")) {
    setDataLoadStatus("dataset path must look like owner/name", "error");
    els.hfCustomDataset.focus();
    return;
  }
  const entry: import("./datasets").HfDataset = {
    id: "custom",
    label: dataset,
    dataset,
    config: els.hfCustomConfig.value.trim() || "default",
    split: els.hfCustomSplit.value.trim() || "train",
    textColumn: els.hfCustomColumn.value.trim() || "text",
    license: "custom",
    blurb: "custom dataset",
  };
  void loadHfDataset(entry);
});

els.hfCustomDataset.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    els.hfCustomLoad.click();
  }
});

// --- data-source tab switching --------------------------------------------
function switchTab(name: string): void {
  document.querySelectorAll<HTMLButtonElement>(".tab-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name);
    b.setAttribute("aria-selected", b.dataset.tab === name ? "true" : "false");
  });
  document.querySelectorAll<HTMLElement>(".tab-panel").forEach((p) => {
    p.hidden = p.dataset.panel !== name;
  });
}
document.querySelectorAll<HTMLButtonElement>(".tab-btn").forEach((b) => {
  b.addEventListener("click", () => {
    const name = b.dataset.tab;
    if (name) switchTab(name);
  });
});

// --- corpus from a local .txt file ----------------------------------------
function setDataLoadStatus(text: string, kind: "" | "ok" | "error" = ""): void {
  els.dataLoadStatus.textContent = text;
  els.dataLoadStatus.className = `data-load-status${kind ? ` ${kind}` : ""}`;
}

els.uploadCorpus.addEventListener("change", async () => {
  const file = els.uploadCorpus.files?.[0];
  if (!file) return;
  setDataLoadStatus(`reading ${file.name}…`);
  try {
    if (file.size > 5 * 1024 * 1024) {
      throw new Error("file too big (max 5 MB) — browser training is bounded by corpus size anyway");
    }
    const text = await file.text();
    els.corpus.value = text;
    els.hfDataset.value = "";
    els.hfCustom.hidden = true;
    setDataLoadStatus(`✓ loaded ${(file.size / 1024).toFixed(0)} KB from ${file.name}`, "ok");
    refreshSampleNote();
  } catch (err) {
    setDataLoadStatus(`couldn't read: ${err instanceof Error ? err.message : String(err)}`, "error");
  } finally {
    els.uploadCorpus.value = "";
  }
});

// --- corpus from Wikipedia — multi-article search-based fetch ------------
// Wikipedia exposes two CORS-friendly endpoints:
//   1. /w/api.php?action=query&list=search&srsearch=<topic>&srlimit=50 — returns
//      article titles matching the topic
//   2. /api/rest_v1/page/plain/<title> — returns plain-text content
// We hit the search endpoint, then fetch the top N articles in parallel until
// we've accumulated the user's requested corpus size. This turns a topic like
// "quantum mechanics" into ~1-5 MB of real prose.
els.wikiLoad.addEventListener("click", () => void loadWikipediaTopic());
els.wikiTitle.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); void loadWikipediaTopic(); }
});

async function loadWikipediaTopic(): Promise<void> {
  const raw = els.wikiTitle.value.trim();
  if (!raw) {
    setDataLoadStatus("type a topic or article title first", "error");
    return;
  }
  els.wikiLoad.disabled = true;
  const maxChars = parseInt(els.fetchSize.value, 10) || 2_000_000;

  // Accept a single-article URL too — extract the slug if present.
  const urlMatch = /\/wiki\/([^?#]+)/.exec(raw);
  const isSingleArticleUrl = !!urlMatch;
  const seedTopic = isSingleArticleUrl ? decodeURIComponent(urlMatch![1]).replace(/_/g, " ") : raw;

  try {
    setDataLoadStatus(`searching Wikipedia for "${seedTopic}"…`);
    const searchUrl =
      `https://en.wikipedia.org/w/api.php?action=query&list=search` +
      `&srsearch=${encodeURIComponent(seedTopic)}&srlimit=50&srprop=&format=json&origin=*`;
    const searchResp = await fetch(searchUrl);
    if (!searchResp.ok) throw new Error(`Wikipedia search returned HTTP ${searchResp.status}`);
    const searchJson = await searchResp.json() as { query?: { search?: { title: string }[] } };
    const titles = (searchJson.query?.search ?? []).map((s) => s.title);
    if (titles.length === 0) throw new Error(`no Wikipedia articles found for "${seedTopic}"`);

    // If user pasted a single-article URL, lead with that exact title.
    if (isSingleArticleUrl) {
      const targeted = decodeURIComponent(urlMatch![1]).replace(/_/g, " ");
      const without = titles.filter((t) => t.toLowerCase() !== targeted.toLowerCase());
      titles.length = 0;
      titles.push(targeted, ...without);
    }

    const parts: string[] = [];
    let chars = 0;
    let fetched = 0;
    for (const title of titles) {
      if (chars >= maxChars) break;
      try {
        const slug = title.replace(/\s+/g, "_");
        const articleUrl =
          `https://en.wikipedia.org/api/rest_v1/page/plain/${encodeURIComponent(slug)}`;
        const articleResp = await fetch(articleUrl, { headers: { Accept: "text/plain" } });
        if (!articleResp.ok) continue;
        const text = await articleResp.text();
        if (text.length < 200) continue;
        parts.push(`# ${title}\n\n${text}`);
        chars += text.length + title.length + 4;
        fetched += 1;
        setDataLoadStatus(`fetched ${fetched} articles · ${(chars / 1024).toFixed(0)} KB / ${(maxChars / 1024).toFixed(0)} KB`);
      } catch { /* skip individual failures */ }
    }
    if (parts.length === 0) throw new Error("none of the matching articles loaded — try a different topic");

    const corpus = parts.join("\n\n").slice(0, maxChars);
    els.corpus.value = corpus;
    els.hfDataset.value = "";
    els.hfCustom.hidden = true;
    setDataLoadStatus(`✓ loaded ${(corpus.length / 1024).toFixed(0)} KB from ${fetched} Wikipedia articles about "${seedTopic}"`, "ok");
    refreshSampleNote();
  } catch (err) {
    setDataLoadStatus(`couldn't load: ${err instanceof Error ? err.message : String(err)}`, "error");
  } finally {
    els.wikiLoad.disabled = false;
  }
}

// --- corpus from an arbitrary CORS-friendly URL ---------------------------
// Works for GitHub raw, jsDelivr, gists, anywhere that serves text with CORS
// headers. The browser will reject non-CORS targets, so the user gets a clear
// error — we don't try to proxy.
els.urlLoad.addEventListener("click", () => void loadFromUrl());
els.urlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); void loadFromUrl(); }
});

async function loadFromUrl(): Promise<void> {
  let url = els.urlInput.value.trim();
  if (!url) { setDataLoadStatus("paste a URL first", "error"); return; }

  // Quality-of-life: github.com/.../blob/ → raw.githubusercontent.com/.../
  if (/^https?:\/\/github\.com\/.+\/blob\//.test(url)) {
    url = url
      .replace("github.com", "raw.githubusercontent.com")
      .replace("/blob/", "/");
  }
  if (!/^https?:\/\//.test(url)) {
    setDataLoadStatus("URL must start with http:// or https://", "error");
    return;
  }

  els.urlLoad.disabled = true;
  setDataLoadStatus(`fetching ${new URL(url).hostname}…`);
  const maxChars = parseInt(els.fetchSize.value, 10) || 2_000_000;
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    let text = await resp.text();
    if (text.length === 0) throw new Error("empty file");
    // If it looks like HTML, warn — but still load it; the user might want it.
    if (/^\s*<!doctype html|^\s*<html/i.test(text)) {
      setDataLoadStatus("⚠ looks like an HTML page, not plain text — may train poorly. Loading anyway…");
    }
    if (text.length > maxChars) text = text.slice(0, maxChars);
    els.corpus.value = text;
    els.hfDataset.value = "";
    els.hfCustom.hidden = true;
    setDataLoadStatus(`✓ loaded ${(text.length / 1024).toFixed(0)} KB from ${new URL(url).hostname}`, "ok");
    refreshSampleNote();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setDataLoadStatus(
      msg.includes("Failed to fetch")
        ? `✗ blocked by CORS — that host doesn't allow browser fetch. Try GitHub raw or jsDelivr.`
        : `✗ ${msg}`,
      "error",
    );
  } finally {
    els.urlLoad.disabled = false;
  }
}

// --- worker messages ------------------------------------------------------
worker.onmessage = (e: MessageEvent<FromWorker>) => {
  const msg = e.data;
  switch (msg.type) {
    case "status":
      setStatus(msg.message);
      break;
    case "progress": {
      const p = msg.progress;
      history.push({ step: p.step, trainLoss: p.trainLoss, valLoss: p.valLoss });
      chart.addPoint({ step: p.step, trainLoss: p.trainLoss, valLoss: p.valLoss });
      flashStat(els.stStep, `${p.step} / ${p.maxSteps}`);
      flashStat(els.stTrain, p.trainLoss.toFixed(4));
      flashStat(els.stVal, p.valLoss?.toFixed(4) ?? "–");
      flashStat(els.stToks, Math.round(p.tokensPerSecond).toLocaleString());
      els.stBackend.textContent = p.backend;
      flashStat(els.stPpl, formatPerplexity(p.trainLoss));
      if (p.valLoss != null) {
        if (p.valLoss < bestVal) { bestVal = p.valLoss; bestValStep = p.step; }
        flashStat(els.stBestVal, Number.isFinite(bestVal)
          ? `${bestVal.toFixed(4)} @ ${bestValStep}`
          : "–");
        flashStat(els.stGap, (p.valLoss - p.trainLoss).toFixed(3));
      }
      checkMilestones(p.step, p.trainLoss, p.valLoss);
      setProgress(p.step, p.maxSteps);
      refreshSampleNote();
      updateStickyStats(p.step, p.maxSteps, p.trainLoss, p.tokensPerSecond, els.stEta.textContent || "–");
      // Live time estimate: tokens left ÷ current throughput.
      if (lastConfig && p.tokensPerSecond > 0 && p.step < p.maxSteps) {
        const tokensLeft = (p.maxSteps - p.step) * lastConfig.batchSize * lastConfig.ctx;
        els.stEta.textContent = formatTime(tokensLeft / p.tokensPerSecond);
      } else {
        els.stEta.textContent = p.step >= p.maxSteps ? "done" : "…";
      }
      break;
    }
    case "sample":
      lastSampleText = msg.text;
      typewriteOutput(msg.text);
      break;
    case "checkpoint":
      // The worker exported the trained model — persist it to OPFS and stash
      // a copy so the user can download it via the "Download model" button.
      savedThisRun = true;
      void saveState(new Uint8Array(msg.state));
      void saveRun({
        savedAt: new Date().toISOString(),
        config: lastConfig ?? readConfig(),
        lossHistory: history,
      });
      latestState = msg.state.slice(0); // copy — the original is now detached after transfer
      latestStateConfig = lastConfig ?? readConfig();
      els.downloadModel.disabled = false;
      els.downloadSafetensors.disabled = false;
      els.continueBtn.disabled = false;
      // Allow navigation to the Watch screen once a model is in memory.
      (window as unknown as { __tgEnableWatch?: () => void }).__tgEnableWatch?.();
      break;
    case "restored":
      els.sample.disabled = false;
      // A restored model lives in the worker — the Watch screen is now valid.
      (window as unknown as { __tgEnableWatch?: () => void }).__tgEnableWatch?.();
      break;
    case "done": {
      setRunning(false);
      stopElapsedClock();
      const totalSec = (performance.now() - runStartTime) / 1000;
      const elapsedStr = formatElapsed(totalSec);
      els.stElapsed.textContent = elapsedStr;
      // Swap the "Time left" stat to "Trained in" once the run completes —
      // the elapsed time is the question every user asks after the run.
      const timeLabel = document.getElementById("stTimeLabel");
      if (msg.reason === "finished") {
        setProgress(1, 1);
        if (timeLabel) timeLabel.textContent = "Trained in";
        els.stEta.textContent = elapsedStr;
        if (!firstRunCelebrated) {
          firstRunCelebrated = true;
          try { localStorage.setItem("tinygpt.firstRunCelebrated", "1"); } catch {}
          showFirstRunCelebration();
        }
      } else {
        els.stEta.textContent = "–";
        if (timeLabel) timeLabel.textContent = "Time left";
      }
      // Show the post-run verdict — concrete interpretation of the final loss
      // right next to the stats so the user knows what they're looking at.
      renderRunVerdict(msg.reason === "finished");
      {
        const doneMsg =
          msg.reason !== "finished"
            ? "training stopped"
            : savedThisRun
              ? "training complete — saved to storage, survives a refresh"
              : "training complete (WebGPU run — not checkpointed)";
        setStatus(doneMsg);
        if (msg.reason === "finished") fireDoneNotification(doneMsg);
      }
      break;
    }
    case "error":
      setRunning(false);
      setStatus(`error: ${msg.message}`, true);
      break;
  }
};

worker.onerror = (e) => {
  setRunning(false);
  els.status.textContent = `worker error: ${e.message}`;
};

// --- startup --------------------------------------------------------------
async function init(): Promise<void> {
  const caps = await detectCapabilities();
  await requestDurableStorage();
  const hw = detectHardware();
  cachedCpuProbeMs = hw.cpuProbeMs;
  refreshEstimate();
  refreshSampleNote();
  const rec = recommendModel(hw);
  const browser = detectBrowser();
  const pill = (label: string, on: boolean, explainKey?: string) => {
    const cls = `pill ${on ? "on" : "off"}`;
    const text = `${label} ${on ? "✓" : "—"}`;
    return explainKey
      ? `<button type="button" class="${cls} pill-btn" data-explain="${explainKey}">${text}</button>`
      : `<span class="${cls}">${text}</span>`;
  };

  // navigator.deviceMemory is capped at 8 GB across Chromium for privacy
  // (and unsupported in Safari/Firefox). Show "≥8 GB" at the cap, otherwise
  // "~X GB" with the bucketed value the browser reported.
  const ramBit = hw.deviceMemoryGB
    ? ` · ${hw.deviceMemoryGB >= 8 ? "≥" : "~"}${hw.deviceMemoryGB} GB RAM`
    : "";
  els.caps.innerHTML =
    pill("WebGPU", caps.webgpu, "webgpuPill") +
    pill("WASM SIMD", caps.wasmSimd, "wasmSimd") +
    pill("cross-origin isolated", caps.crossOriginIsolated, "crossOriginIsolated") +
    `<span class="pill on">backend: ${caps.active}</span>` +
    `<span class="pill off">${browser.name}</span>` +
    `<span class="pill off">${hw.cores} cores${ramBit}</span>` +
    `<button type="button" class="pill off pill-btn" id="heapPill" data-explain="heap" title="JS heap used">heap —</button>` +
    `<span class="pill off" id="gpuPill" hidden></span>` +
    `<span class="muted" style="margin-left:6px">` +
    `Suggested: <strong>${rec.layers}L · d_model ${rec.dModel} · ctx ${rec.ctx}</strong> ` +
    `(~${formatParams(rec.approxParams)} params, ${rec.tier})` +
    `</span>` +
    `<button id="applyRec" class="ghost" style="margin-left:4px">Apply</button>`;

  // GPU adapter name (Chromium / Safari WebGPU). Display only if available.
  if (caps.webgpu) {
    void getGpuName().then((name) => {
      const pillEl = document.getElementById("gpuPill");
      if (pillEl && name) {
        pillEl.textContent = `GPU: ${name}`;
        pillEl.hidden = false;
      }
    });
  }

  // Live JS-heap usage (Chromium only — performance.memory is non-standard).
  startHeapPoll();

  // If WebGPU is available, default the backend dropdown to it — measured
  // ~7× faster on Apple M-series and increasingly accurate elsewhere. Users
  // who explicitly want WASM can switch back; this just changes the default.
  if (caps.webgpu) {
    const backendSel = byId<HTMLSelectElement>("backend");
    // Only override if the user hasn't already touched it (i.e. it's still
    // on the HTML-default "wasm" value).
    if (backendSel.value === "wasm" && backendSel.dataset.userPicked !== "1") {
      backendSel.value = "webgpu";
      refreshEstimate("");
    }
    backendSel.addEventListener("change", () => {
      backendSel.dataset.userPicked = "1";
    });
  }

  byId<HTMLButtonElement>("applyRec").addEventListener("click", () => {
    applyRecommendation(rec);
    byId<HTMLButtonElement>("applyRec").textContent = "Applied ✓";
  });

  renderPersonalisation({ caps, hw, rec, browser });

  // Restore the previous run — the loss chart and the trained model — so it
  // survives a page refresh (milestone 7).
  const prev = await loadRun();
  const prevState = await loadState();
  if (prev && prev.lossHistory.length > 0) {
    history = prev.lossHistory;
    for (const pt of history) chart.addPoint(pt);
    const last = history[history.length - 1];
    els.stStep.textContent = String(last.step);
    els.stTrain.textContent = last.trainLoss.toFixed(4);
    els.stVal.textContent = last.valLoss?.toFixed(4) ?? "–";
  }
  if (prev && prevState) {
    lastConfig = prev.config as RunConfig;
    const buffer = prevState.buffer as ArrayBuffer;
    // Stash a copy for download before transferring to the worker.
    latestState = buffer.slice(0);
    latestStateConfig = lastConfig;
    els.downloadModel.disabled = false;
    els.downloadSafetensors.disabled = false;
    els.continueBtn.disabled = false;
    worker.postMessage({ type: "restore", state: buffer, config: lastConfig }, [buffer]);
    els.status.textContent = "restoring your last model from storage…";
  }
}

void init().then(() => {
  initPopovers();
  setupTour();
  applyConfigFromUrl();
  setupIntroCard();
  setupStickyStats();
  setupKeyboardShortcuts();
  setupDemoBanner();
  setupScreens();
  // Mark landing animation as done after first paint — subsequent navigations
  // skip the brand-draw animation (it's a one-time wow).
  setTimeout(() => document.body.classList.add("landing-done"), 2200);
});

// --- keyboard shortcuts ---------------------------------------------------
function setupKeyboardShortcuts(): void {
  const dialog = document.getElementById("shortcuts") as HTMLDialogElement | null;
  const closeBtn = document.getElementById("shortcutsClose") as HTMLButtonElement | null;
  if (!dialog || !closeBtn) return;
  closeBtn.addEventListener("click", () => dialog.close());

  // Don't trigger shortcuts when the user is typing in a text input.
  const isEditing = () => {
    const el = document.activeElement as HTMLElement | null;
    if (!el) return false;
    const tag = el.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
  };

  document.addEventListener("keydown", (e) => {
    // Cmd/Ctrl chord shortcuts always allowed (even in inputs).
    const cmd = e.metaKey || e.ctrlKey;
    if (cmd && e.key === "Enter") {
      e.preventDefault();
      if (!els.start.disabled) els.start.click();
      return;
    }
    if (cmd && (e.key === "g" || e.key === "G")) {
      e.preventDefault();
      if (!els.sample.disabled) els.sample.click();
      return;
    }
    if (isEditing()) return;
    if (e.key === "?") {
      e.preventDefault();
      if (typeof dialog.showModal === "function") dialog.showModal();
      return;
    }
    if (e.key === "t" || e.key === "T") {
      const tourBtn = document.getElementById("tourBtn") as HTMLButtonElement | null;
      tourBtn?.click();
      return;
    }
    if (e.key === "s" || e.key === "S") {
      els.shareBtn?.click();
      return;
    }
    if (e.key === "p" || e.key === "P") {
      if (!els.pause.disabled) els.pause.click();
      return;
    }
    if (/^[1-5]$/.test(e.key)) {
      const order = ["tiny", "small", "medium", "large", "xl"];
      const presetId = order[parseInt(e.key, 10) - 1];
      if (presetId) {
        els.sizePreset.value = presetId;
        els.sizePreset.dispatchEvent(new Event("change"));
      }
    }
  });
}

// --- two-screen navigation: Setup ⟷ Watch --------------------------------
// Screen 1 (Setup): the Train card with corpus + presets + Start.
// Screen 2 (Watch): the Loss chart + Sample area + verdict.
// Click Start → transition to Watch. Click the Setup tab → back to Setup.
function setupScreens(): void {
  const screens = document.getElementById("screens");
  if (!screens) return;
  const tabs = document.querySelectorAll<HTMLButtonElement>(".screen-tab");
  let modelLoaded = latestState != null; // determines if "Watch" tab is enabled

  const setActive = (name: "setup" | "watch") => {
    screens.setAttribute("data-active", name);
    tabs.forEach((t) => {
      const isActive = t.dataset.screen === name;
      t.classList.toggle("active", isActive);
    });
    document.querySelectorAll<HTMLElement>(".screen").forEach((s) => {
      s.hidden = s.dataset.screen !== name;
    });
    // Scroll back to the top so the user sees the new screen from the start.
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const enableWatch = () => {
    modelLoaded = true;
    tabs.forEach((t) => {
      if (t.dataset.screen === "watch") t.disabled = false;
    });
  };

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const name = tab.dataset.screen as "setup" | "watch";
      if (name === "watch" && !modelLoaded) return; // gated
      setActive(name);
    });
  });

  // Auto-switch when training starts OR when a model is loaded.
  els.start.addEventListener("click", () => {
    // Defer to next tick so the existing Start handler (validation,
    // pre-flight warning) runs first. If validation fails, the run never
    // begins — but we still moved screens. Acceptable: the loss chart
    // will be empty + ready when validation passes.
    setTimeout(() => {
      enableWatch();
      setActive("watch");
    }, 0);
  });

  // Watching is also valid after loading a saved model (no fresh training).
  // We poll the latestState every checkpoint message via the existing handler;
  // hook the model upload too.
  const origRender = renderRunVerdict;
  void origRender; // referenced to silence TS

  // Expose enableWatch globally so other code paths can call it.
  (window as unknown as { __tgEnableWatch?: () => void }).__tgEnableWatch = enableWatch;
}

// --- demo banner — "Try a trained model" CTA -----------------------------
// On load, HEAD /demo.tinygpt. If it exists (200), reveal the banner.
// Click → fetch + load via the same path as model upload.
async function setupDemoBanner(): Promise<void> {
  const banner = document.getElementById("demoBanner");
  const btn = document.getElementById("loadDemoBtn") as HTMLButtonElement | null;
  if (!banner || !btn) return;
  // Hide once any model is loaded — avoid clutter for return visitors.
  if (latestState) { banner.hidden = true; return; }

  try {
    const head = await fetch("/demo.tinygpt", { method: "HEAD" });
    if (!head.ok) return; // demo not deployed yet; banner stays hidden
  } catch {
    return;
  }
  banner.hidden = false;

  btn.addEventListener("click", async () => {
    btn.classList.add("loading");
    btn.textContent = "fetching trained model…";
    try {
      const resp = await fetch("/demo.tinygpt");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const blob = await resp.blob();
      const file = new File([blob], "demo.tinygpt", { type: "application/octet-stream" });
      await loadModelFromFile(file, "pre-trained Shakespeare demo");
      banner.hidden = true;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      btn.classList.remove("loading");
      btn.innerHTML = `<span aria-hidden="true">▶</span> Try a trained model`;
      setStatus(`couldn't load demo model: ${msg}`, true);
    }
  });
}

// --- intro card (first-visit "what is this") ------------------------------
function setupIntroCard(): void {
  const card = document.getElementById("introCard");
  const dismiss = document.getElementById("introDismiss");
  if (!card || !dismiss) return;
  const KEY = "tinygpt.introDismissed";
  let dismissed = false;
  try { dismissed = localStorage.getItem(KEY) === "1"; } catch {}
  card.hidden = dismissed;
  dismiss.addEventListener("click", () => {
    card.hidden = true;
    try { localStorage.setItem(KEY, "1"); } catch {}
  });
}

// --- sticky mini-stats — shown when training + loss card scrolled offscreen
function setupStickyStats(): void {
  const sticky = document.getElementById("stickyStats");
  const jumpBtn = document.getElementById("stickyJump") as HTMLButtonElement | null;
  const lossCard = document.querySelector<HTMLElement>(".loss-card");
  if (!sticky || !jumpBtn || !lossCard) return;

  let lossCardVisible = true;
  const io = new IntersectionObserver(
    (entries) => {
      lossCardVisible = entries[0]?.isIntersecting ?? true;
      updateSticky();
    },
    { threshold: 0.12 },
  );
  io.observe(lossCard);

  let isTraining = false;
  const origSetRunning = setRunning;
  // Wrap setRunning by patching the function reference is tricky; just keep
  // sticky in sync via a small hook fired from setRunning.
  Object.defineProperty(window, "__tgSetTraining", {
    value: (on: boolean) => { isTraining = on; updateSticky(); },
    writable: false,
  });

  function updateSticky(): void {
    if (!sticky) return;
    sticky.hidden = !(isTraining && !lossCardVisible);
  }

  jumpBtn.addEventListener("click", () => {
    lossCard?.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  // Reference the unused helper so TS doesn't complain.
  void origSetRunning;
}

function setupTour(): void {
  const tourBtn = document.getElementById("tourBtn") as HTMLButtonElement | null;
  const welcome = document.getElementById("welcome") as HTMLDialogElement | null;
  const welcomeStart = document.getElementById("welcomeStart") as HTMLButtonElement | null;
  const welcomeSkip = document.getElementById("welcomeSkip") as HTMLButtonElement | null;

  tourBtn?.addEventListener("click", () => startTour());

  if (welcome && welcomeStart && welcomeSkip) {
    welcomeStart.addEventListener("click", () => {
      welcome.close();
      startTour();
    });
    welcomeSkip.addEventListener("click", () => {
      welcome.close();
      markTourSeen();
    });
    if (!hasSeenTour() && typeof welcome.showModal === "function") {
      welcome.showModal();
    }
  }
}
