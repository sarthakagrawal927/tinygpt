/**
 * popover.ts — wires every [data-explain] button to a popover.
 *
 * Uses the browser's native Popover API (light-dismiss + ESC + a11y for free)
 * and Floating UI to keep the popover anchored to its trigger.
 *
 * Add a button anywhere in the HTML:
 *   <button class="info" data-explain="layers">?</button>
 *
 * On load, initPopovers() finds it, builds the matching popover from
 * EXPLAINERS, and wires up the toggle.
 */

import { autoUpdate, computePosition, flip, offset, shift } from "@floating-ui/dom";
import { EXPLAINERS } from "./explainers";

let counter = 0;

export function initPopovers(root: ParentNode = document): void {
  const triggers = root.querySelectorAll<HTMLButtonElement>("[data-explain]");
  for (const trigger of triggers) {
    if (trigger.dataset.popoverWired === "1") continue;
    const key = trigger.dataset.explain;
    if (!key) continue;
    const explainer = EXPLAINERS[key];
    if (!explainer) {
      console.warn("[popover] missing explainer for key:", key);
      continue;
    }

    const id = `explain-${key}-${counter++}`;
    const pop = document.createElement("div");
    pop.id = id;
    pop.setAttribute("popover", "auto");
    pop.className = "explain-pop";

    const linkHtml = explainer.link
      ? `<a href="${escapeAttr(explainer.link.href)}" target="_blank" rel="noopener noreferrer">${escapeText(explainer.link.label)} ↗</a>`
      : "";
    pop.innerHTML = `
      <h4>${escapeText(explainer.title)}</h4>
      <p>${escapeText(explainer.body)}</p>
      ${linkHtml}
    `;
    document.body.appendChild(pop);

    trigger.setAttribute("popovertarget", id);
    trigger.setAttribute("aria-label", `Explain: ${explainer.title}`);
    trigger.setAttribute("aria-haspopup", "dialog");
    trigger.type = "button";
    // Only style as a "?" info circle when the trigger has no text of its own
    // (so pre-styled pills and other custom triggers are left alone).
    if (!trigger.textContent?.trim()) {
      trigger.textContent = "?";
      trigger.classList.add("info");
    }
    trigger.dataset.popoverWired = "1";

    let cleanup: (() => void) | undefined;
    pop.addEventListener("toggle", (e) => {
      const newState = (e as ToggleEvent).newState;
      if (newState === "open") {
        trigger.setAttribute("aria-expanded", "true");
        trigger.classList.add("is-open");
        cleanup = autoUpdate(trigger, pop, () => {
          void computePosition(trigger, pop, {
            strategy: "fixed",
            placement: "bottom-start",
            middleware: [offset(8), flip(), shift({ padding: 10 })],
          }).then(({ x, y }) => {
            pop.style.left = `${x}px`;
            pop.style.top = `${y}px`;
          });
        });
      } else {
        trigger.setAttribute("aria-expanded", "false");
        trigger.classList.remove("is-open");
        if (cleanup) {
          cleanup();
          cleanup = undefined;
        }
      }
    });
  }
}

const escapeText = (s: string): string =>
  s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

const escapeAttr = (s: string): string => escapeText(s).replace(/"/g, "&quot;");
