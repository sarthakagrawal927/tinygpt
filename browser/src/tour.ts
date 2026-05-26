/**
 * tour.ts — a guided first-visit tour of the playground.
 *
 * Walks the user from "this is your machine" → "click Start" → "watch the loss
 * fall" → "generate text" in seven steps. Each step has a popover anchored to
 * a target element and a soft spotlight (everything else dimmed) so the user
 * always knows what we're talking about.
 *
 * Built on the same Floating UI / native popover primitives as popover.ts —
 * one extra dep would be wasteful when the building blocks are already loaded.
 */

import {
  autoUpdate,
  computePosition,
  flip,
  offset,
  shift,
  type Placement,
} from "@floating-ui/dom";

interface TourStep {
  /** CSS selector for the element to spotlight + anchor the popover to. */
  target: string;
  title: string;
  body: string;
  link?: { href: string; label: string };
  placement?: Placement;
  /** Pause the tour until the user clicks this selector (e.g., #start). */
  awaitClickOn?: string;
}

const STEPS: TourStep[] = [
  {
    target: ".machine",
    title: "Step 1 · Your machine",
    body: "We detected your CPU, RAM, and which graphics APIs your browser supports. The Apply button drops in a model size that should train in well under a minute here.",
    placement: "bottom-start",
  },
  {
    target: "#corpus",
    title: "Step 2 · The training corpus",
    body: "This is the only thing the model will ever see. Paste your own text, or pick a small Hugging Face dataset from the dropdown above. With ~0.8M params, kilobytes of text is plenty.",
    placement: "right",
  },
  {
    target: ".form-grid",
    title: "Step 3 · Model size & training knobs",
    body: "Layers, d_model, context length — the architecture. Max steps, batch size, backend — how the training loop runs. Click any '?' next to a label for what each one actually does.",
    placement: "right",
  },
  {
    target: "#start",
    title: "Step 4 · Click Start",
    body: "Training runs in a Web Worker so the page stays responsive. The first step takes a moment (loading the WASM module); after that the loss starts updating live.",
    placement: "top",
    awaitClickOn: "#start",
  },
  {
    target: "#chart",
    title: "Step 5 · Watch the loss curve",
    body: "Loss = how surprised the model is on average. It starts near 5.55 (random guess over 256 bytes). Under ~2 means real structure has been learned. Below 1 is genuine memorisation — and on a corpus this small, that's the regime you'll hit.",
    placement: "left",
  },
  {
    target: "#sample",
    title: "Step 6 · Generate text",
    body: "Once training is going, click Generate. The model writes one byte at a time, picking each from its predicted distribution. Temperature 0.8 reads best — try 0 (deterministic) and 1.5 (chaotic) to feel the dial.",
    placement: "top",
  },
  {
    target: "footer.notes",
    title: "You're done",
    body: "Your model and loss curve are checkpointed to OPFS — refresh the page and they'll come back. Every '?' on the page links out to the maths if you want to go deeper. There's also a WebGPU kernel-parity test at /webgpu-test.",
    placement: "top-end",
  },
];

const STORAGE_KEY = "tinygpt.tour.completed";

export function hasSeenTour(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

export function markTourSeen(): void {
  try {
    localStorage.setItem(STORAGE_KEY, "1");
  } catch {
    // Ignore — private mode, quota, etc. Tour just won't be remembered.
  }
}

export function startTour(): void {
  let stepIdx = 0;
  let cleanupPosition: (() => void) | undefined;
  let awaitController: AbortController | undefined;

  const overlay = document.createElement("div");
  overlay.className = "tour-overlay";
  overlay.setAttribute("aria-hidden", "true");

  const spotlight = document.createElement("div");
  spotlight.className = "tour-spotlight";
  overlay.appendChild(spotlight);

  const pop = document.createElement("div");
  pop.className = "tour-pop";
  pop.setAttribute("role", "dialog");
  pop.setAttribute("aria-live", "polite");

  document.body.appendChild(overlay);
  document.body.appendChild(pop);

  function teardown(): void {
    cleanupPosition?.();
    awaitController?.abort();
    awaitController = undefined;
    overlay.remove();
    pop.remove();
    document.removeEventListener("keydown", onKey);
    markTourSeen();
  }

  function onKey(e: KeyboardEvent): void {
    if (e.key === "Escape") teardown();
    else if (e.key === "ArrowRight") goNext();
    else if (e.key === "ArrowLeft") goPrev();
  }
  document.addEventListener("keydown", onKey);

  function goPrev(): void {
    if (stepIdx > 0) {
      stepIdx -= 1;
      renderStep();
    }
  }
  function goNext(): void {
    if (stepIdx < STEPS.length - 1) {
      stepIdx += 1;
      renderStep();
    } else {
      teardown();
    }
  }

  function renderStep(): void {
    cleanupPosition?.();
    awaitController?.abort();
    awaitController = undefined;

    const step = STEPS[stepIdx];
    const target = document.querySelector<HTMLElement>(step.target);
    if (!target) {
      // Target missing (e.g., a section hasn't been mounted yet). Skip ahead.
      goNext();
      return;
    }

    target.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });

    const isLast = stepIdx === STEPS.length - 1;
    const isFirst = stepIdx === 0;
    const linkHtml = step.link
      ? `<a class="tour-link" href="${step.link.href}" target="_blank" rel="noopener noreferrer">${step.link.label} ↗</a>`
      : "";
    const awaitNote = step.awaitClickOn
      ? `<div class="tour-await">Click <b>Start training</b> below to continue…</div>`
      : "";

    pop.innerHTML = `
      <div class="tour-step">${stepIdx + 1} / ${STEPS.length}</div>
      <h3>${escapeText(step.title)}</h3>
      <p>${escapeText(step.body)}</p>
      ${linkHtml}
      ${awaitNote}
      <div class="tour-actions">
        <button type="button" class="tour-skip" data-action="skip">Skip tour</button>
        <div class="tour-nav">
          <button type="button" class="tour-prev" data-action="prev" ${isFirst ? "disabled" : ""}>Back</button>
          <button type="button" class="tour-next" data-action="next" ${step.awaitClickOn ? "disabled" : ""}>
            ${isLast ? "Done" : "Next →"}
          </button>
        </div>
      </div>
    `;

    pop.querySelector('[data-action="skip"]')?.addEventListener("click", teardown);
    pop.querySelector('[data-action="prev"]')?.addEventListener("click", goPrev);
    pop.querySelector('[data-action="next"]')?.addEventListener("click", goNext);

    // Position the popover relative to the target.
    cleanupPosition = autoUpdate(target, pop, () => {
      const r = target.getBoundingClientRect();
      spotlight.style.top = `${r.top - 6}px`;
      spotlight.style.left = `${r.left - 6}px`;
      spotlight.style.width = `${r.width + 12}px`;
      spotlight.style.height = `${r.height + 12}px`;

      void computePosition(target, pop, {
        strategy: "fixed",
        placement: step.placement ?? "bottom",
        middleware: [offset(14), flip(), shift({ padding: 14 })],
      }).then(({ x, y }) => {
        pop.style.left = `${x}px`;
        pop.style.top = `${y}px`;
      });
    });

    // If this step is gated on a user action, advance once it fires.
    if (step.awaitClickOn) {
      const watchTarget = document.querySelector(step.awaitClickOn);
      if (watchTarget) {
        awaitController = new AbortController();
        watchTarget.addEventListener(
          "click",
          () => setTimeout(() => goNext(), 400),
          { once: true, signal: awaitController.signal },
        );
      }
    }
  }

  renderStep();
}

const escapeText = (s: string): string =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
