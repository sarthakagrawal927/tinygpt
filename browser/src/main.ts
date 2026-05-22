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
import { HF_CATALOG, fetchHfText } from "./datasets";
import { LossChart } from "./charts";
import {
  detectBrowser,
  detectCapabilities,
  detectHardware,
  recommendModel,
  type ModelRecommendation,
} from "./runtime_detect";
import { loadRun, loadState, requestDurableStorage, saveRun, saveState } from "./storage";
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
  hfLoad: byId<HTMLButtonElement>("hfLoad"),
  hfStatus: byId<HTMLSpanElement>("hfStatus"),
};

const canvas = byId<HTMLCanvasElement>("chart");
canvas.width = canvas.clientWidth || 560;
canvas.height = 220;
const chart = new LossChart(canvas);

const worker = new Worker(new URL("./worker.ts", import.meta.url), {
  type: "module",
});
const send = (msg: ToWorker) => worker.postMessage(msg);

let paused = false;
let history: { step: number; trainLoss: number; valLoss?: number }[] = [];
let lastConfig: RunConfig | null = null; // config of the in-flight / last run

// --- config ---------------------------------------------------------------
function readConfig(): RunConfig {
  const intOf = (id: string) => parseInt(byId<HTMLInputElement>(id).value, 10);
  const dModel = parseInt(byId<HTMLSelectElement>("dModel").value, 10);
  return {
    ctx: intOf("ctx"),
    layers: intOf("layers"),
    heads: 3, // d_model options are all multiples of 3
    dModel,
    dMlp: dModel * 4,
    batchSize: intOf("batch"),
    learningRate: parseFloat(byId<HTMLInputElement>("lr").value),
    gradClip: DEFAULT_CONFIG.gradClip,
    maxSteps: intOf("maxSteps"),
    evalEvery: DEFAULT_CONFIG.evalEvery,
    seed: DEFAULT_CONFIG.seed,
  };
}

// --- button state ---------------------------------------------------------
function setRunning(on: boolean): void {
  els.start.disabled = on;
  els.pause.disabled = !on;
  els.stop.disabled = !on;
}

// --- hardware-aware recommendation ----------------------------------------
function formatParams(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : `${Math.round(n / 1000)}k`;
}

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
    els.status.textContent = "corpus is very short — add more text";
    return;
  }
  history = [];
  chart.reset();
  els.output.textContent = "Training… generate once a few steps have run.";
  els.stEta.textContent = "…";
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

els.sample.addEventListener("click", () => {
  send({
    type: "sample",
    prompt: byId<HTMLInputElement>("prompt").value,
    tokens: parseInt(byId<HTMLInputElement>("genTokens").value, 10),
    temperature: parseFloat(byId<HTMLInputElement>("temp").value),
  });
  els.output.textContent = "generating…";
});

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
for (const d of HF_CATALOG) {
  const opt = document.createElement("option");
  opt.value = d.id;
  opt.textContent = `${d.label} — ${d.blurb}`;
  els.hfDataset.appendChild(opt);
}

els.hfLoad.addEventListener("click", async () => {
  const entry = HF_CATALOG.find((d) => d.id === els.hfDataset.value);
  if (!entry) {
    els.hfStatus.textContent = "pick a dataset from the list first";
    return;
  }
  els.hfLoad.disabled = true;
  els.hfStatus.textContent = `fetching ${entry.label}…`;
  try {
    const text = await fetchHfText(entry, 120_000, (chars) => {
      els.hfStatus.textContent = `fetching ${entry.label}… ${Math.round(chars / 1000)} KB`;
    });
    els.corpus.value = text;
    els.hfStatus.textContent =
      `loaded ${Math.round(text.length / 1000)} KB from ${entry.label} · ${entry.license}`;
  } catch (err) {
    els.hfStatus.textContent =
      `couldn't load: ${err instanceof Error ? err.message : String(err)}`;
  } finally {
    els.hfLoad.disabled = false;
  }
});

// --- worker messages ------------------------------------------------------
worker.onmessage = (e: MessageEvent<FromWorker>) => {
  const msg = e.data;
  switch (msg.type) {
    case "status":
      els.status.textContent = msg.message;
      break;
    case "progress": {
      const p = msg.progress;
      history.push({ step: p.step, trainLoss: p.trainLoss, valLoss: p.valLoss });
      chart.addPoint({ step: p.step, trainLoss: p.trainLoss, valLoss: p.valLoss });
      els.stStep.textContent = `${p.step} / ${p.maxSteps}`;
      els.stTrain.textContent = p.trainLoss.toFixed(4);
      els.stVal.textContent = p.valLoss?.toFixed(4) ?? "–";
      els.stToks.textContent = Math.round(p.tokensPerSecond).toLocaleString();
      els.stBackend.textContent = p.backend;
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
      els.output.textContent = msg.text;
      break;
    case "checkpoint":
      // The worker exported the trained model — persist it to OPFS.
      void saveState(new Uint8Array(msg.state));
      void saveRun({
        savedAt: new Date().toISOString(),
        config: lastConfig ?? readConfig(),
        lossHistory: history,
      });
      break;
    case "restored":
      els.sample.disabled = false;
      break;
    case "done":
      setRunning(false);
      els.stEta.textContent = msg.reason === "finished" ? "done" : "–";
      els.status.textContent =
        msg.reason === "finished"
          ? "training complete — saved to storage, survives a refresh"
          : "training stopped — progress saved";
      break;
    case "error":
      setRunning(false);
      els.status.textContent = `error: ${msg.message}`;
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
  const storage = await requestDurableStorage();
  const hw = detectHardware();
  const rec = recommendModel(hw);
  const browser = detectBrowser();
  const pill = (label: string, on: boolean) =>
    `<span class="pill ${on ? "on" : "off"}">${label} ${on ? "✓" : "—"}</span>`;

  els.caps.innerHTML =
    `<div>` +
    pill("WebGPU", caps.webgpu) +
    pill("WASM SIMD", caps.wasmSimd) +
    pill("cross-origin isolated", caps.crossOriginIsolated) +
    `<span class="pill on">backend: ${caps.active}</span>` +
    `<span class="pill off">OPFS ~${storage.quotaMB} MB</span>` +
    `</div>` +
    `<div style="margin-top:10px;font-size:13px;color:#adbac7">` +
    `<b style="color:#e6edf3">Browser:</b> ${browser.name} — ${browser.note}` +
    `</div>` +
    `<div style="margin-top:6px;font-size:13px;color:#adbac7">` +
    `<b style="color:#e6edf3">Your machine:</b> ${hw.cores} logical cores` +
    (hw.deviceMemoryGB ? ` · ~${hw.deviceMemoryGB} GB RAM` : "") +
    ` · CPU probe ${hw.cpuProbeMs.toFixed(0)} ms` +
    `</div>` +
    `<div style="margin-top:6px;font-size:13px;color:#adbac7">` +
    `<b style="color:#e6edf3">Suggested model:</b> ~${formatParams(rec.approxParams)} params ` +
    `— context ${rec.ctx}, ${rec.layers} layers, d_model ${rec.dModel} ` +
    `<span style="color:#7d8590">(${rec.tier} machine — ${rec.note})</span> ` +
    `<button id="applyRec" class="secondary" style="padding:3px 10px;margin-left:4px">Apply</button>` +
    `</div>`;

  byId<HTMLButtonElement>("applyRec").addEventListener("click", () => {
    applyRecommendation(rec);
    byId<HTMLButtonElement>("applyRec").textContent = "Applied ✓";
  });

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
    worker.postMessage({ type: "restore", state: buffer, config: lastConfig }, [buffer]);
    els.status.textContent = "restoring your last model from storage…";
  }
}

void init();
