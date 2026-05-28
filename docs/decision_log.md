# TinyGPT Decision Log

This is the meta-log: not the lessons learned, not the open questions, but the moments the direction actually changed. Across one long working session between Sarthak and the agent (Claude Opus 4.7), the project drifted, snapped, doubled back, and sometimes lurched. The entries below are the turning points — the user messages and findings that forced a course correction, the calls that followed, and what those calls eventually shipped. Lessons live elsewhere. This file is just the inflection points, in the order they happened.

---

### Decision 1: Withdraw the flat "9.7×" headline in favor of a curve
**When:** Opening phase of the session, immediately after the FA2 forward-pass shipping work.
**Context:** The repo had just landed Flash Attention 2 forward and was advertising a single headline number — "9.7× WebGPU vs WASM" — across the README, the blog, the status doc, and the Astro pages. The number came from one preset run and was treated as the canonical speedup claim.
**The trigger:** Sarthak's pushback — "Just 9.7×?" — calling out that a single number both undersold the bigger presets and overpromised the smaller ones.
**The call:** Replace the flat number with a measured curve across four presets and publish the function, not the point.
**Why it was right:** A curve is honest. A point estimate at one model size implies the speedup is constant; the curve reveals it grows monotonically with model dimension, which is the actual interesting finding (memory-bandwidth-bound kernels widen the WebGPU lead as tensors grow).
**What it shipped:** Edits to `README.md`, `BLOG.md`, `docs/status.md`, `browser/src/pages/speedup.astro`, `browser/src/pages/roadmap.astro`, `browser/src/pages/devlog.astro`. The published numbers became Small 2.6×, Medium 6.8×, Large 9.3×, XL 12.1×.

---

### Decision 2: Investigate before retrying
**When:** Early measurement phase, after the curve decision but before any new data was captured.
**Context:** The measurement script `browser/measure_curve.mjs` was failing on the XL preset under WASM. The reflex was to bump timeouts, add retries, and try again with slightly different flags.
**The trigger:** Sarthak's interrupt — "First I want you to understand why it is failing" — cutting off the retry loop after the third unproductive attempt.
**The call:** Stop retrying. Pivot to writing a diagnostic script (`browser/diagnose_behemoth.mjs`) that isolates the failure surface.
**Why it was right:** The retries were treating a structural bug (WASM Memory64 out-of-bounds on memory growth in a SharedArrayBuffer + pthread context inside the browser) as if it were a flake. The diagnostic surfaced the real root cause in one run.
**What it shipped:** `browser/diagnose_behemoth.mjs`; the finding that `tinygpt64.wasm` runs cleanly under Node but fails under the browser pthread+SAB combination during memory growth.

---

### Decision 3: Refuse to manipulate the step count to make the ratio "work"
**When:** Mid-measurement, after the curve was being filled out preset by preset.
**Context:** One preset's WASM-vs-WebGPU comparison kept timing out at the 15-step default. The agent proposed dropping to 5 steps to "make the run complete."
**The trigger:** Sarthak's response — "This does not spark joy" — flagging that fewer steps shrinks the ratio, because the WASM amortization-per-step gets worse as step counts drop while WebGPU stays flat.
**The call:** Keep 15 steps as the comparison standard. Fix the underlying timeout instead.
**Why it was right:** A speedup claim that depends on a hand-tuned step count is not a speedup claim — it's a benchmark gaming exercise. Keeping the step count constant made the curve comparable across presets.
**What it shipped:** No file change directly — but the integrity of every number on the speedup page traces back to this refusal.

---

### Decision 4: Accept that the in-browser demo cannot show off "great output" without 15 minutes of training
**When:** Quality-of-output discussion, after the curve was published and attention shifted to what users would actually see in the live demo.
**Context:** The Huge preset converges nicely — but only after roughly 3000 steps at ~150 ms/step on WebGPU. That's a quarter-hour of held attention from a user who clicked a link.
**The trigger:** Math, not a single message — the realization that no amount of kernel tuning collapses 15 minutes of compute into a 30-second demo.
**The call:** Stop pretending the in-browser train-from-scratch path can produce coherent Shakespeare in the demo's natural attention window. Plan instead around a two-track product.
**Why it was right:** The acceptance unlocked Decision 5. Trying to hide the training cost would have meant either lying about model quality or shipping a worse model.
**What it shipped:** No file directly. The realization seeded the pretrained-model path.

---

### Decision 5: Ship pre-trained and train-your-own as two product paths
**When:** Immediately following Decision 4.
**Context:** The demo page was trying to do everything in one flow: pick a preset, click train, see good output. It was failing at the "good output" step for anything below Huge + lots of steps.
**The trigger:** Sarthak's framing — "We should give them a sample model which works well and we should also ask them if they want to train on their machine also." Two audiences, one page.
**The call:** Bifurcate the landing experience. A pre-trained checkpoint loads instantly for the "look how good this is" audience; a clearly-labelled "train your own" path serves the "look how fast this trains" audience.
**Why it was right:** Each path now has a credible story. The pre-trained model demonstrates output quality without lying about training time; the train-your-own path demonstrates the kernel speedups honestly because the speedup is the point, not the loss curve.
**What it shipped:** Banner rework and CTA rework in `browser/src/pages/index.astro`; the pretrained model swap; corpus swap (Decision 6).

---

### Decision 6: Replace the inline 863-byte default corpus with full TinyShakespeare
**When:** Mid-session, after a 1500-step Huge training run reported a suspiciously good loss of 0.14.
**Context:** The default corpus shipped inline in the source as an 863-byte snippet. A 9.6M-parameter model trained on 863 bytes for 1500 steps is not learning a language — it is memorizing a passage.
**The trigger:** The 0.14 loss itself, which is too good for any honest language modelling objective at that scale.
**The call:** Replace the snippet with the full TinyShakespeare corpus and stop shipping the trivial-overfit default.
**Why it was right:** The number 0.14 was a lie of context. The new corpus produces honest loss curves that decay into the 1.5–2.0 range, which is the regime where samples start looking like language rather than copy-paste.
**What it shipped:** `data/examples/shakespeare.txt`, `browser/public/shakespeare.txt`; the default-data wiring in `browser/src/pages/index.astro` and `browser/src/main.ts`.

---

### Decision 7: Lower the default learning rate to 3e-4
**When:** After the corpus swap, watching the new honest-loss curves stall.
**Context:** Training on the full corpus plateaued at loss ≈ 2.45 for any preset above Small. The plateau looked like underfitting, but the dataset was now large enough that underfitting at 1500 steps should not have happened.
**The trigger:** A side-by-side check of the Python reference's optimizer config showed `lr=3e-4`. The browser default was `3e-3` — 10× higher, off in the regime where Adam updates blow past minima rather than settling into them.
**The call:** Drop the default to `3e-4` and let users override upward if they want.
**Why it was right:** The plateau broke. Loss curves resumed monotone decrease past 2.0. The 10× error came from a copy-paste during the early WebGPU port and had survived because nobody had trained long enough to notice the plateau.
**What it shipped:** `browser/src/types.ts:35` (default constant) and `browser/src/pages/index.astro:2621` (UI default value).

---

### Decision 8: Ship the curve, not the lessons-loss curve
**When:** Documentation phase, after the speedup numbers settled.
**Context:** Two artefacts competed for the headline slot on the speedup page: a per-preset speedup curve (Small → XL) and a loss-vs-step curve from training. Both told true stories. Only one fit the page's question.
**The trigger:** A realization that the speedup page is about "how much faster is the kernel" — not "how well does it learn." Mixing the two muddied both.
**The call:** Publish the preset-to-speedup function on the speedup page. Keep loss curves in `docs/status.md` and the devlog.
**Why it was right:** Separation of concerns inside the docs. A reader asking "is this fast" gets a clean answer; a reader asking "does this work" gets a different clean answer.
**What it shipped:** Final `browser/src/pages/speedup.astro` layout — preset axis, speedup axis, four marked points, a fitted shape.

---

### Decision 9: Hand off and restart in a fresh session when macOS TCC revoked Desktop access
**When:** Roughly two-thirds through, mid-edit.
**Context:** The session was in the middle of wiring up the pre-trained model loader. macOS Transparency, Consent, and Control prompted for Desktop folder access and the sandbox lost write permission to the working tree.
**The trigger:** Edit calls returning permission errors with no recoverable fallback inside the sandbox.
**The call:** Commit in-progress work via dictated commands while there was still read access, close the harness, restart Warp from a fresh shell, resume in a new session with TCC re-granted.
**Why it was right:** The alternative was an hour of trying to coax permissions back through a session that had already lost trust. Restarting cost five minutes and zero work.
**What it shipped:** A clean commit boundary and a fresh session that finished the pre-trained loader work.

---

### Decision 10: Variety beats volume for in-browser corpora
**When:** Post-corpus-swap, when discussion turned to "what else should ship as a sample dataset."
**Context:** With TinyShakespeare working as the default, the obvious next move looked like "ship a much bigger Shakespeare-style corpus." That intuition was wrong for this product.
**The trigger:** Sarthak's question — "Do you think having more data ... would be helpful?" — which forced the agent to actually think about the training budget the browser allows.
**The call:** Add genre variety, not volume. Ship four curated Hugging Face mini-corpora covering domains the demo could not otherwise touch.
**Why it was right:** The 15-minute training window caps how much of any corpus the model can see. More tokens than fit in that window are wasted bytes. Different genres, in contrast, let users feel the model learn a different shape of language each time.
**What it shipped:** Curated corpora — code, recipes, PubMed abstracts, poetry — wired into the dataset picker in `browser/src/pages/index.astro`. Total curated corpus count reached 19 across the session.

---

### Decision 11: Make compute pressure visible via PressureObserver
**When:** Late session, after Sarthak ran a long training job and noticed the laptop slowing down.
**Context:** The demo could pin a CPU and saturate the GPU and the only signal the user got was a slow page. Heat, fan, and system pressure were invisible.
**The trigger:** Sarthak's request — "show how much process it is taking when this is running because my machine gets visibly slower."
**The call:** Add a live pulse-dot pressure chip next to the backend pill, driven by `PressureObserver` where available, with a graceful no-op fallback elsewhere.
**Why it was right:** Honest UI. The cost of training is now legible; the user is not surprised by their fan.
**What it shipped:** A `setupSystemPressure` function in `browser/src/main.ts` and the chip markup near `#stBackend` in `browser/src/pages/index.astro`.

---

### Decision 12: Fix the Memory64 OOB via INITIAL_MEMORY bump, not C++ kernel rewrite
**When:** After the diagnostic from Decision 2 was complete.
**Context:** `tinygpt64.wasm` ran XL and Massive cleanly under Node but crashed in-browser with an out-of-bounds access. The crash sat at a growth boundary inside the pthread + SharedArrayBuffer interaction.
**The trigger:** The diagnostic confirming the kernels themselves were correct — the failure was in `WebAssembly.Memory.grow()` semantics under the browser's threading model.
**The call:** Skip any C++ kernel rewrite. Raise `INITIAL_MEMORY` from 32 MB to 256 MB in `wasm/build_wasm64.sh` so growth simply never fires for the presets users actually run.
**Why it was right:** The fix took one line. A kernel rewrite would have taken days, addressed a problem the browser engines will likely fix upstream anyway, and added new risk. The trade is a 224 MB up-front allocation in exchange for never tripping a known browser race.
**What it shipped:** `wasm/build_wasm64.sh` INITIAL_MEMORY change; `tests/test_wasm64_xl_node.mjs` as the regression-catcher.

---

### Decision 13: Smoke-test the export path before betting an hour on it
**When:** Just before what would have been a full Huge training run for checkpoint validation.
**Context:** The plan was to train a real model end-to-end and then test whether the download button actually produced a valid checkpoint. The cost of finding a bug at the end of a Huge run is one Huge run.
**The trigger:** Sarthak's sharpest feedback of the session — "why don't you train a smaller model, test how to download, and then do this?"
**The call:** Build a 30-second smoke test that exercises the entire flow on a Small WebGPU model and asserts the checkpoint loads back.
**Why it was right:** The smoke script (`browser/smoke_export.mjs`, ~50 steps Small on WebGPU) immediately surfaced the next bug — Decision 14 — without any of the long-training cost. The cost-of-finding-bugs curve flattened from "hours" to "seconds."
**What it shipped:** `browser/smoke_export.mjs`; the discipline of testing the export path on every relevant commit.

---

### Decision 14: Implement WebGPU checkpoint serialization
**When:** Immediately after the smoke test surfaced the issue.
**Context:** A comment in `browser/src/worker.ts:212` had openly admitted: "The WebGPU model has no checkpoint serialization yet — survives-refresh stays a WASM-backend feature." The smoke script turned that admission into a hard failure.
**The trigger:** The smoke test failing on the WebGPU path's first download attempt.
**The call:** Add `GpuModel.exportState()` — copy every weight tensor and every Adam state tensor back from GPU buffers into CPU `Float32Array`s in the exact flat layout the WASM backend already uses, so the existing loader works unchanged.
**Why it was right:** A shared on-disk format meant the loader did not need a second code path. The serialization function fit into the existing worker contract; the diff was small; the smoke test passed on the next run.
**What it shipped:** `GpuModel.exportState()` and the wiring inside `doTrainWebGPU` in `browser/src/worker.ts` (both the initial-train and continue-train branches).

---

### Decision 15: Keep the browser alive on script failure
**When:** Tooling polish, after a `train_demo.mjs` run lost a partial download to a script-exit race.
**Context:** The training-and-download helper was auto-closing the browser when its download assertion failed. That closure killed any chance of manually clicking through to recover the checkpoint.
**The trigger:** A lost run where the model had finished training, but the script error path closed Chromium before the download click finished.
**The call:** On any download-path failure, leave the browser open with a clear console message pointing the user at `Model ▾ → Download .tinygpt`.
**Why it was right:** Failure-tolerance matters most exactly when training succeeded. The cost of a stuck-open browser is one Cmd-Q; the cost of a lost trained model is the whole run.
**What it shipped:** The error handler in `browser/train_demo.mjs`.

---

### Decision 16: Stop polling. Use harness notifications
**When:** Repeatedly, across the session. Marked here once for the record.
**Context:** Background tasks in the harness auto-notify on completion. The agent kept reaching for sleep-based polling anyway.
**The trigger:** The harness itself, which blocks long leading `sleep` commands and pushes the agent toward the notification flow.
**The call:** Use `run_in_background` for long jobs and trust the notification. Use `Monitor` with an until-loop where polling really is needed. Never chain short sleeps.
**Why it was right:** Less wasted wall-clock, less context bloat. The harness was right; the reflex to poll was a holdover from non-harness environments.
**What it shipped:** No file change. A behavioral correction visible in the run-history.

---

### Decision 17: Defer fine-tuning, quantization, gallery, Mac app
**When:** Scope-setting, roughly mid-session.
**Context:** With the basic in-browser demo working, several adjacent ideas competed for attention: a LoRA fine-tuning UI, INT8 quantization, a curated samples gallery, a native Mac app wrapper. Each had its own argument.
**The trigger:** Sarthak's framing — "Fine tuning is for the future. Fine tuning and quantization are later tasks."
**The call:** Hold the line at bugs + data + polish + documentation for this session. Push the four big features to later.
**Why it was right:** Scope discipline. The session shipped a clean, honest, working demo because it refused the larger features; attempting them would have left everything half-finished.
**What it shipped:** Nothing — which is the point. The deferral let the rest of the work land.

---

### Decision 18: Build all four documentation formats in parallel
**When:** Final stretch of the session, when the question of how to document the work surfaced.
**Context:** Four documentation artefacts competed: the README, the BLOG, the devlog page, and the status doc. Each had a different audience.
**The trigger:** Sarthak's response when asked which to prioritize — "I would say build them all and people can choose what they want to read."
**The call:** Author all four in parallel rather than picking one and trimming the others.
**Why it was right:** The four documents have genuinely different shapes. The README is scannable; the BLOG is narrative; the devlog is chronological; the status doc is structured. Forcing them into one doc would have served no audience well.
**What it shipped:** Coordinated updates to `README.md`, `BLOG.md`, `browser/src/pages/devlog.astro`, and `docs/status.md`. Plus the lessons file (`docs/lessons.md`) as a fifth artefact for a fifth audience.

---

## Decisions still in flight

The session closed with several open calls deliberately not made:

- **Whether to add KV caching.** The inference path would benefit, especially for the pre-trained model's interactive sampling. The decision is whether the implementation cost is worth it before the gallery and Mac app ship.
- **Whether the next session is the Mac app or the curated gallery.** The Mac app extends reach; the gallery extends depth. Both are blocked by Decision 17's deferral and neither is more urgent than the other yet.
- **Whether to ship more curated corpora or hold at 19.** Variety past 19 hits diminishing returns; the picker UI starts to feel crowded; each new corpus is a hosting and licensing burden. The case for stopping is as strong as the case for continuing.
- **Whether the pre-trained checkpoint should be one model or a small zoo.** Currently one. A zoo (Shakespeare, code, recipes) would let the pretrained path showcase the same genre variety the train-your-own path has. The cost is hosting and the cognitive load on the landing page.
- **Whether to revisit the WASM Memory64 OOB upstream.** Decision 12's fix works. But the underlying browser bug is real and reporting it would be a public good. Open question whether this session's bandwidth extends to a minimal reproduction.
- **Whether to invest in a real evaluation harness.** Loss is published. Perplexity on a held-out chunk is not. Sample quality is judged by eyeballing. A small eval suite would tighten the honesty story; it would also pull effort from the next feature.

These are not the unknowns. These are the known unknowns that have a decision waiting on the next session's first hour.

---

## Second session — the browser-frontier perf quest

### Decision 18: Don't post to HN yet; first lock down the opportunistic-edge perf push
**When:** Late in the gallery-build thread, after the gallery dialog + manifest + Shakespeare entry had landed and three new corpora (TinyStories, code, recipes) were ready to train.
**Context:** The agent was preparing to ship the gallery and post to HN. The user asked whether there was a way to make in-browser training 30-40× faster.
**The trigger:** Sarthak's direction — "I just want the best possible performance for the people who are at the latest version and we can do a graceful degradation depending on what's available… also inform them about it" — followed by "we can prompt users to enable `chrome://flags#enable-unsafe-webgpu` for further speed."
**The call:** Pause the HN launch. Build the full opportunistic-acceleration stack — storage-f16, shader-f16, cooperative matrix, WebNN inference — each gated on feature detection, each surfaced in the capability pills, with a soft dismissible nudge for Chromium users who don't yet have the experimental flag enabled. Ship the gallery on the now-faster path so the launch's "trained in this browser" claim has the new speed behind it.
**Why it was right:** The Mac app (lever 20) is ~30× over WebGPU because it has access to MLX + Metal + ANE. Stacking every browser frontier together gets ~3-5× on flag-enabled Chrome — small next to the Mac app but real for the 1% of HN visitors who already run `chrome://flags#enable-unsafe-webgpu`. More importantly, the capability pill + nudge UI makes the bleeding edge visible — the page itself becomes a demonstration of where browser ML can reach today.
**What it shipped (foundation, commit `28f2533`):** `webgpu/tensor.ts` extended to opportunistically request `subgroups` + `shader-f16` + `timestamp-query` and probe cooperative-matrix via trial shader compile; `runtime_detect.ts` carries `gpuFeatures: GpuSubFeatures` + `webnnPresent`; `main.ts` renders `+f16` / `+subgroups` / `+WebNN` pills + a dismissible "Power user?" nudge; `explainers.ts` carries four new pill explainers; roadmap lever 21 ("Browser frontier — tech we're tracking") catalogues each lever with expected gain + skip rationale + revisit trigger. The kernel work itself (storage-f16 → shader-f16 → cooperative matrix → WebNN) is queued as tasks #90-#94 with the hard rule **no quality regression on any fast path** — each must pass a 500-step Shakespeare loss-curve gate or auto-disable.

---

### Decision 19: Refuse to ship any fast path that regresses model quality
**When:** Inside the opportunistic-acceleration scope conversation, immediately after Decision 18.
**Context:** The four-lever push (storage-f16, shader-f16, cooperative matrix, WebNN) brings real precision risk — f16 accumulators can drift, cooperative-matrix is an experimental WGSL extension, WebNN routes through OS NN runtimes whose numerics behavior isn't fully specified. The temptation in a launch-prep window is to ship the speedup and ignore the drift.
**The trigger:** Sarthak's constraint — "I just want to ensure that the model quality does not drop and the speed is as fast as whatever flags that the user has enabled on their Chrome."
**The call:** Bake a hard numerics gate into every fast path. Train Shakespeare for 500 steps on each enabled path at first use; require the loss curve to match the f32 reference within 1% at step 500. Paths that fail the gate disable themselves silently for the session and the user gets the slower correct path. No path activates if it can't pass the gate.
**Why it was right:** Opportunistic optimization without a correctness gate is a bug factory. Each new accelerator has its own precision profile, and "it ran faster" is the easy thing to measure and observe — "it produced subtly worse output" is the hard thing to notice without instrumentation. The gate makes "speed" a derived property of "speed AND correctness," not a substitute for correctness.
**What it shipped:** The constraint is encoded in task #94's title ("Numerics gate: no quality regression on ANY fast path") and described in `docs/perf_quest.md`. Implementation lands alongside #90-#93 (one verifier per fast path; the gate also gets its own page `docs/precision.md` capturing measured deltas).

---

### Decision 20: Sequential under `caffeinate`, never parallel-on-one-GPU
**When:** After 2 hours of failed parallel training during the gallery-build phase.
**Context:** Three Playwright Chromium instances were launched concurrently to train TinyStories + code + recipes against the same M-series GPU. The reasoning was that the GPU could time-share. The reality: every model slowed from ~100 steps/min (solo) to ~17-30 steps/min (3-way share), plus three separate 15-minute stalls in each log when the system slept. After 2 hours, all three hit the 2-hour wall-clock cap at 40-46% completion — and none had downloaded, because the script's download step lives after training completion. The work was discarded.
**The trigger:** All three background tasks notifying completion simultaneously, all with the same `Error: hard cap exceeded`. The pattern in the logs (uniform 15-min gaps across all three logs at the same wall-clock timestamps) pointed at display-sleep suspending the Chromium tabs.
**The call:** Future training runs are **sequential**, one Playwright at a time, **wrapped in `caffeinate -i`** to block display sleep. Three sequential Huge runs at ~75 min each = 3.75 hr total — slower in theory than perfectly-parallel 3 × 75 min ÷ 3 = 75 min, but faster in practice than "parallel + contention + display sleep + zero output."
**Why it was right:** GPU contention on a single GPU is sub-linear. The math is "3× the work to do" plus "1 GPU's worth of throughput" minus "scheduler thrash" — at best you'd expect 2× wall-time of sequential, not equal. Plus the OS isn't aware that a backgrounded Chromium tab is doing real work, so it sleeps the tab when the user's display sleeps. Both effects compound. Sequential side-steps both.
**What it shipped:** Task #85's description now reads "Retrain gallery (sequential, caffeinate, after perf work)." The gallery retrain is queued for the post-perf-work phase explicitly under this constraint, with no path back to parallel.

---

These three decisions reshape the next session: it's not a launch session. It's a perf-quest session. The gallery, the HN post, and the deploy all wait on it.
