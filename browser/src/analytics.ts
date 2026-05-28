/**
 * analytics.ts — thin PostHog wrapper for the playground.
 *
 * Three events only — `playground_loaded`, `train_started`, `sample_generated`.
 * No PII (no corpus text, no generated text, no prompts). The page MUST keep
 * working when analytics is off, so every call here is a no-op if init didn't
 * complete: missing VITE_POSTHOG_KEY, DNT signalled, or user opt-out present.
 *
 * Owner-level decisions baked in here (see README):
 *  - default host: https://us.i.posthog.com (PostHog Cloud US ingest)
 *  - opt-out: localStorage["tg_analytics_opt_out"] === "1" or navigator.doNotTrack === "1"
 *  - autocapture: enabled (PostHog default) — clicks on public buttons only
 *  - session_recording: DISABLED (privacy + bundle weight)
 *  - capture_pageview: enabled once at init (single SPA, no router)
 */

import posthog from "posthog-js";

// import.meta.env is typed locally in src/vite-env.d.ts — no unknown,
// no casts, no aliasing. Read directly at call sites so Vite's static
// replacement plugin can substitute the build-time values.

const DEFAULT_HOST = "https://us.i.posthog.com";
const OPT_OUT_KEY = "tg_analytics_opt_out";

let enabled = false;

function userHasOptedOut(): boolean {
  // navigator.doNotTrack is the spec; some engines also stash it on window.
  try {
    if (typeof navigator !== "undefined" && navigator.doNotTrack === "1") return true;
  } catch {
    // ignore
  }
  try {
    if (typeof localStorage !== "undefined" && localStorage.getItem(OPT_OUT_KEY) === "1") {
      return true;
    }
  } catch {
    // localStorage can throw in sandboxed iframes; treat as not-opted-out.
  }
  return false;
}

/**
 * Initialise PostHog. Safe to call multiple times — only the first call wires
 * the SDK; subsequent calls are no-ops. Returns true if analytics is live.
 */
export function initAnalytics(): boolean {
  if (enabled) return true;

  if (userHasOptedOut()) {
    // Quiet on purpose — opt-out is a user choice, not a config error.
    return false;
  }

  // Read import.meta.env.VITE_X DIRECTLY here so Vite's static
  // replacement plugin can substitute the literal value at build time.
  // Aliasing (`const env = import.meta.env`) defeats the replacement and
  // leaves a runtime lookup against an empty object.
  const key = import.meta.env.VITE_POSTHOG_KEY;
  if (!key) {
    // One-time info-level breadcrumb so devs running locally aren't confused.
    // eslint-disable-next-line no-console
    console.info("Analytics disabled: VITE_POSTHOG_KEY not set");
    return false;
  }

  const host = import.meta.env.VITE_POSTHOG_HOST || DEFAULT_HOST;

  try {
    posthog.init(key, {
      api_host: host,
      // Single-page playground — fire one pageview on load, that's it.
      capture_pageview: true,
      capture_pageleave: true,
      // Keep autocapture on (PostHog default) — buttons and link clicks only,
      // no input contents. Useful for "did anyone click X?" without bespoke events.
      autocapture: true,
      // Hard off — we don't ship recordings, the playground is a single page.
      disable_session_recording: true,
      // Don't collect IP at the project level (still set on server unless the
      // PostHog project is configured to discard it; this is the client signal).
      ip: false,
      // Respect DNT inside the SDK as well as our own gate.
      respect_dnt: true,
      // Use localStorage — the cookie-jar fallback is unnecessary here.
      persistence: "localStorage",
    });
    enabled = true;
    return true;
  } catch (err) {
    // Never let a broken analytics init break the page.
    // eslint-disable-next-line no-console
    console.warn("Analytics init failed:", err);
    return false;
  }
}

function capture(event: string, props: object): void {
  if (!enabled) return;
  try {
    posthog.capture(event, props as Record<string, unknown>);
  } catch {
    // Swallow — analytics must never throw into the playground.
  }
}

// --- the three events ----------------------------------------------------

export interface PlaygroundLoadedProps {
  browser: string;
  has_webgpu: boolean;
  has_memory64: boolean;
  has_wasm_simd: boolean;
  cores: number;
  device_memory_gb: number | null;
  cpu_probe_ms: number;
}

export function trackPlaygroundLoaded(props: PlaygroundLoadedProps): void {
  capture("playground_loaded", props);
}

export interface TrainStartedProps {
  preset: string;
  backend: "wasm" | "webgpu";
  layers: number;
  d_model: number;
  ctx: number;
  batch: number;
  max_steps: number;
  corpus_bytes: number;
  est_params: number;
}

export function trackTrainStarted(props: TrainStartedProps): void {
  capture("train_started", props);
}

export interface SampleGeneratedProps {
  prompt_bytes: number;
  output_bytes: number;
  temperature: number;
  top_k: number;
  final_train_loss: number | null;
  final_val_loss: number | null;
}

export function trackSampleGenerated(props: SampleGeneratedProps): void {
  capture("sample_generated", props);
}

// --- fleet page crash monitoring (page_crash_capture requirement) ------------
// Standard browser crash capture hook (documented equivalent for TinyGPT's
// Astro + vanilla-TS bundle; see saas-maker/standards/hardening/error-kit).
// Wires global listeners so uncaught errors and unhandled rejections in
// production are captured with project slug. Always console.errors (never
// silent). Sends `foundry_page_crash` via PostHog only when analytics enabled
// (respects existing VITE_POSTHOG_KEY + opt-out gates; no behavior change).

const PROJECT_SLUG = "tinygpt";

function route(): string | undefined {
  if (typeof window === "undefined") return undefined;
  return `${window.location.origin}${window.location.pathname}`;
}

function messageFrom(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  return String(error);
}

export function capturePageCrash(
  error: unknown,
  source: "window_error" | "unhandled_rejection",
): void {
  const payload = {
    project_id: PROJECT_SLUG,
    route: route(),
    source,
    message: messageFrom(error),
    stack: error instanceof Error ? error.stack : undefined,
  };

  // Always surface to console — ops + debugging visibility even when
  // analytics is fully disabled (matches tinygpt "page must keep working"
  // rule while giving signal).
  // eslint-disable-next-line no-console
  console.error("[tinygpt:page_crash]", payload);

  if (!enabled) return;
  try {
    posthog.capture("foundry_page_crash", payload as Record<string, unknown>);
  } catch {
    // Swallow — crash monitoring must never throw or break the page.
  }
}

export function installBrowserMonitoring(): () => void {
  if (typeof window === "undefined") return () => {};

  const onError = (event: ErrorEvent) => {
    capturePageCrash(event.error ?? event.message, "window_error");
  };
  const onUnhandledRejection = (event: PromiseRejectionEvent) => {
    capturePageCrash(event.reason, "unhandled_rejection");
  };

  window.addEventListener("error", onError);
  window.addEventListener("unhandledrejection", onUnhandledRejection);

  return () => {
    window.removeEventListener("error", onError);
    window.removeEventListener("unhandledrejection", onUnhandledRejection);
  };
}
