/**
 * charts.ts — training-loss chart, polished (Phase 4 + UX pass).
 *
 * A dependency-free canvas line chart of train/val loss vs step. Fed from
 * TrainingProgress messages on the main thread.
 *
 * Visuals:
 *   - Dark background that matches the rest of the page
 *   - Train line: accent teal with a filled gradient area underneath
 *   - Val line: warm amber, dashed
 *   - Leading-edge marker: pulsing accent dot at the latest train point
 *   - Faint horizontal reference at ln(256) = 5.55 (random baseline)
 *
 * Guide: docs/browser_notes.md ("Web Worker")
 */

interface Point {
  step: number;
  trainLoss: number;
  valLoss?: number;
}

// Colors aligned with the page's accent/warn tokens.
const COLOR_BG = "#08090a";
const COLOR_GRID = "#1d1f23";
const COLOR_AXIS_TEXT = "#787d88";
const COLOR_TRAIN = "#48e5c2";
const COLOR_TRAIN_FILL = "rgba(72, 229, 194, 0.14)";
const COLOR_TRAIN_FILL_FADE = "rgba(72, 229, 194, 0.0)";
const COLOR_VAL = "#f5b14a";
const COLOR_BASELINE = "rgba(245, 177, 74, 0.28)";

const LN_256 = Math.log(256); // 5.545

export class LossChart {
  private readonly ctx: CanvasRenderingContext2D;
  private points: Point[] = [];
  private pulsePhase = 0;
  private pulseTimer: number | null = null;
  private hoverX: number | null = null; // CSS pixels relative to canvas
  private lastPadL = 44;
  private lastPlotW = 0;
  private tooltip: HTMLDivElement | null = null;
  /** When the consumer knows the target step count (e.g., maxSteps from
   *  the training config), pass it in so the x-axis spans the FULL run
   *  from step 0. Without this, the axis auto-scales to "last seen step",
   *  which makes the first ~10% of training look squished into the left
   *  edge and unreadable. */
  private maxStepHint = 0;
  private rafPending = false;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2D canvas context unavailable");
    this.ctx = ctx;
    this.setupHiDpi();
    this.attachHoverInspect();
    this.draw();
    // Pause the pulse-redraw when the tab is hidden — the user can't see it,
    // and 4Hz redraws keep the renderer thread alive in background tabs.
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) this.stopPulseLoop();
      else if (this.points.length > 0) this.startPulseLoop();
    });
  }

  private attachHoverInspect(): void {
    const tip = document.createElement("div");
    tip.className = "chart-tooltip";
    tip.hidden = true;
    // Inserted next to the canvas; absolutely positioned within the loss card.
    this.canvas.parentElement?.appendChild(tip);
    this.tooltip = tip;

    this.canvas.addEventListener("mousemove", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      this.hoverX = e.clientX - rect.left;
      this.draw();
      this.updateTooltip(e.clientX - rect.left, e.clientY - rect.top);
    });
    this.canvas.addEventListener("mouseleave", () => {
      this.hoverX = null;
      if (this.tooltip) this.tooltip.hidden = true;
      this.draw();
    });
  }

  private updateTooltip(cssX: number, _cssY: number): void {
    if (!this.tooltip || this.points.length === 0) return;
    // Map cssX to nearest point.
    const fractionalIdx = Math.max(
      0,
      Math.min(1, (cssX - this.lastPadL) / Math.max(1, this.lastPlotW)),
    );
    const idx = Math.round(fractionalIdx * (this.points.length - 1));
    const p = this.points[idx];
    if (!p) return;
    const trainBits = (p.trainLoss / Math.log(2)).toFixed(2);
    const valBit = p.valLoss != null
      ? `<span class="tip-val">val ${p.valLoss.toFixed(3)}</span>`
      : "";
    this.tooltip.innerHTML = `
      <div class="tip-step">step ${p.step}</div>
      <div class="tip-train">train ${p.trainLoss.toFixed(3)} <span class="tip-bits">${trainBits} bits</span></div>
      ${valBit}
    `;
    this.tooltip.hidden = false;
    // Position to the right of the cursor by default, flip if too close to edge.
    const tipW = this.tooltip.offsetWidth || 140;
    const canvasW = this.canvas.clientWidth;
    const onLeft = cssX > canvasW - tipW - 24;
    this.tooltip.style.left = `${onLeft ? cssX - tipW - 12 : cssX + 12}px`;
    this.tooltip.style.top = `${10}px`;
  }

  /**
   * Bump the backing-store resolution to match devicePixelRatio for crisp
   * lines. Sizes from the canvas's actual rendered bounding box so CSS rules
   * (e.g. width: 100%, height: 420px on .hero-loss canvas#chart) drive the
   * logical chart size — not the static width/height attributes.
   */
  private setupHiDpi(): void {
    const dpr = window.devicePixelRatio || 1;
    const { canvas } = this;
    const rect = canvas.getBoundingClientRect();
    const cssW = rect.width || canvas.width;
    const cssH = rect.height || canvas.height;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    this.ctx.scale(dpr, dpr);
  }

  reset(): void {
    this.points = [];
    this.stopPulseLoop();
    this.draw();
  }

  /** Hint the chart about the full target step count so the x-axis spans
   *  step 0..maxSteps from the very first frame, instead of squishing
   *  early points into the left edge as new data arrives. */
  setMaxStep(maxSteps: number): void {
    this.maxStepHint = Math.max(0, maxSteps);
    this.draw();
  }

  addPoint(p: Point): void {
    this.points.push(p);
    this.startPulseLoop();
    // The pulse loop redraws at 4Hz; an extra synchronous draw per point is
    // redundant under training (30+ points/sec) and catastrophic during model
    // restoration (1500-point loops would each force a full canvas redraw).
    // Coalesce via rAF so multiple addPoints within one frame collapse to one
    // draw, and the rAF naturally pauses while the tab is hidden.
    if (this.rafPending) return;
    this.rafPending = true;
    requestAnimationFrame(() => {
      this.rafPending = false;
      this.draw();
    });
  }

  /** Throttled redraw loop — at 60fps the constant draw was visibly
   *  taxing the CPU on lower-end machines (user reported "feels like
   *  something's happening a lot" lag). 4Hz is plenty for the pulse to
   *  read as alive, and the main thread stays mostly idle between ticks. */
  private startPulseLoop(): void {
    if (this.pulseTimer != null) return;
    const tick = () => {
      this.pulsePhase += 0.5;
      this.draw();
    };
    this.pulseTimer = window.setInterval(tick, 250); // 4 redraws/sec
  }

  private stopPulseLoop(): void {
    if (this.pulseTimer != null) {
      window.clearInterval(this.pulseTimer);
      this.pulseTimer = null;
    }
  }

  private logicalSize(): { w: number; h: number } {
    const dpr = window.devicePixelRatio || 1;
    return { w: this.canvas.width / dpr, h: this.canvas.height / dpr };
  }

  private draw(): void {
    const { ctx } = this;
    const { w: W, h: H } = this.logicalSize();
    const pad = { l: 44, r: 14, t: 14, b: 28 };
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;
    this.lastPadL = pad.l;
    this.lastPlotW = plotW;

    ctx.fillStyle = COLOR_BG;
    ctx.fillRect(0, 0, W, H);

    if (this.points.length === 0) {
      this.drawEmptyState(pad, plotW, plotH, W, H);
      return;
    }

    // --- X range -----------------------------------------------------
    // Live training spans 0..maxStepHint (line grows from the left).
    // Loaded models span first..last of the actual saved data — no
    // synthetic prefix, no fake interpolation. If the saved history was
    // truncated, the user sees exactly what's stored; the gap (if any)
    // is disclosed via the "saved from step N" label below.
    const firstStep = this.points[0].step;
    const lastStep = this.points[this.points.length - 1].step;
    const targetMax = this.maxStepHint || lastStep;
    const liveTraining = firstStep <= Math.max(1, targetMax * 0.05);
    const minStep = liveTraining ? 0 : firstStep;
    const maxStep = liveTraining ? Math.max(1, targetMax) : Math.max(firstStep + 1, lastStep);
    const xSpan = Math.max(1, maxStep - minStep);

    // --- Y range -----------------------------------------------------
    // Auto-scale to the data with sensible padding. Only EXTEND to
    // include the ln(256) random baseline (5.55) when the data is
    // genuinely close to it — otherwise that reference line just stretches
    // the chart's Y range and makes the descended-loss curve look squished
    // against the bottom edge. Threshold: include the baseline only when
    // data reaches at least 60% of it.
    let dataMax = 0, dataMin = Infinity;
    for (const p of this.points) {
      dataMax = Math.max(dataMax, p.trainLoss, p.valLoss ?? 0);
      dataMin = Math.min(dataMin, p.trainLoss, p.valLoss ?? p.trainLoss);
    }
    const includeBaseline = dataMax >= LN_256 * 0.6;
    let maxLoss = (includeBaseline ? Math.max(dataMax, LN_256) : dataMax) * 1.12;
    // Lower bound: 0 when data spans most of the range, dataMin-padded
    // otherwise so a tight clump near 0.9 doesn't get visually crushed.
    const minLoss = dataMin < dataMax * 0.4 ? 0 : Math.max(0, dataMin - (dataMax - dataMin) * 0.4);
    const lossSpan = Math.max(0.01, maxLoss - minLoss);

    const x = (step: number) => pad.l + ((step - minStep) / xSpan) * plotW;
    const y = (loss: number) => pad.t + (1 - (loss - minLoss) / lossSpan) * plotH;

    this.drawGrid(pad, plotW, plotH, maxLoss, y);

    // ln(256) random-baseline reference line — drawn only when the
    // current Y range actually reaches it. Otherwise it's just a label
    // outside the visible plot area.
    if (LN_256 >= minLoss && LN_256 <= maxLoss) {
      ctx.save();
      ctx.strokeStyle = COLOR_BASELINE;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(pad.l, y(LN_256));
      ctx.lineTo(pad.l + plotW, y(LN_256));
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = COLOR_BASELINE;
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText("random  ln(256) = 5.55", pad.l + plotW - 132, y(LN_256) - 4);
    }

    // Quality threshold bands — tells the user what each loss level actually
    // means in human terms. Drawn only when they're within the visible y range.
    const drawThreshold = (lossLevel: number, color: string, label: string) => {
      if (lossLevel > maxLoss || lossLevel < minLoss) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 5]);
      ctx.beginPath();
      ctx.moveTo(pad.l, y(lossLevel));
      ctx.lineTo(pad.l + plotW, y(lossLevel));
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = color;
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText(label, pad.l + 8, y(lossLevel) - 4);
    };
    // Loss = 2.0 — words start forming below this line.
    drawThreshold(2.0, "rgba(245, 208, 74, 0.65)", "loss 2.0  ↓ words form");
    // Loss = 1.5 — grammar emerges below this line.
    drawThreshold(1.5, "rgba(72, 229, 194, 0.7)", "loss 1.5  ↓ grammar emerges");

    // Train: filled gradient area + line.
    this.drawTrainArea(x, y, pad, plotH);
    this.drawSeries(x, y, (p) => p.trainLoss, COLOR_TRAIN, false);
    // Val: dashed.
    this.drawSeries(x, y, (p) => p.valLoss, COLOR_VAL, true);

    // Leading-edge pulse on the latest train point.
    const last = this.points[this.points.length - 1];
    const lx = x(last.step);
    const ly = y(last.trainLoss);
    const pulse = (Math.sin(this.pulsePhase) + 1) / 2; // 0..1
    const radius = 3 + pulse * 2.2;
    const haloAlpha = 0.18 + pulse * 0.18;
    ctx.fillStyle = `rgba(72, 229, 194, ${haloAlpha.toFixed(3)})`;
    ctx.beginPath();
    ctx.arc(lx, ly, radius + 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = COLOR_TRAIN;
    ctx.beginPath();
    ctx.arc(lx, ly, radius, 0, Math.PI * 2);
    ctx.fill();

    // Hover crosshair + nearest-point marker, if hovering.
    if (this.hoverX != null && this.points.length > 0) {
      const cssH = H;
      const cssPlotW = plotW;
      const fractionalIdx = Math.max(
        0,
        Math.min(1, (this.hoverX - pad.l) / Math.max(1, cssPlotW)),
      );
      const idx = Math.round(fractionalIdx * (this.points.length - 1));
      const p = this.points[idx];
      if (p) {
        const hx = x(p.step);
        const hy = y(p.trainLoss);
        ctx.save();
        ctx.strokeStyle = "rgba(146, 150, 160, 0.35)";
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(hx, pad.t);
        ctx.lineTo(hx, cssH - pad.b);
        ctx.stroke();
        ctx.restore();
        // Halo + dot on the hovered train point.
        ctx.fillStyle = "rgba(72, 229, 194, 0.22)";
        ctx.beginPath();
        ctx.arc(hx, hy, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = COLOR_TRAIN;
        ctx.beginPath();
        ctx.arc(hx, hy, 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Legend.
    this.drawLegend(pad);

    // X-axis labels — when minStep is nonzero (a loaded model whose saved
    // history was truncated), show both ends so it's obvious which range
    // the chart represents. Live training (minStep=0) only labels the
    // right side to avoid the redundant "step 0".
    ctx.fillStyle = COLOR_AXIS_TEXT;
    ctx.font = "10.5px ui-monospace, monospace";
    if (minStep > 0) {
      ctx.textAlign = "left";
      ctx.fillText(`step ${minStep}`, pad.l, H - 10);
    }
    ctx.textAlign = "right";
    ctx.fillText(`step ${maxStep}`, pad.l + plotW, H - 10);
    ctx.textAlign = "left";
  }

  private drawEmptyState(
    pad: { l: number; t: number },
    plotW: number,
    plotH: number,
    W: number,
    H: number,
  ): void {
    const { ctx } = this;
    this.drawFrame(pad, plotW, plotH);

    // Faint preview curve — exp decay from 5.55 → ~1.5 across the plot width.
    // Hints at what the user is about to see without faking real data.
    const maxLoss = 6.2;
    const y = (loss: number) => pad.t + (1 - loss / maxLoss) * plotH;
    ctx.save();
    ctx.strokeStyle = "rgba(72, 229, 194, 0.18)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    const steps = 80;
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const loss = 1.4 + (5.55 - 1.4) * Math.exp(-3.2 * t);
      const px = pad.l + t * plotW;
      const py = y(loss);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.restore();

    // Random-baseline reference dashes — same as in the populated chart.
    ctx.save();
    ctx.strokeStyle = "rgba(245, 177, 74, 0.22)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.l, y(LN_256));
    ctx.lineTo(pad.l + plotW, y(LN_256));
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = "rgba(245, 177, 74, 0.5)";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText("random  ln(256) = 5.55", pad.l + plotW - 132, y(LN_256) - 4);

    // y-axis tick labels (lightly).
    ctx.fillStyle = "rgba(120, 125, 136, 0.55)";
    ctx.font = "10.5px ui-monospace, monospace";
    for (let i = 0; i <= 4; i++) {
      const loss = (maxLoss / 4) * i;
      ctx.fillText(loss.toFixed(1), 8, y(loss) + 4);
    }

    ctx.fillStyle = "rgba(231, 232, 234, 0.7)";
    ctx.font = "500 13px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("loss curve appears here once training begins", W / 2, H / 2 - 4);
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillStyle = COLOR_AXIS_TEXT;
    ctx.fillText("starts near 5.55 — falls toward 1.5 as the model learns", W / 2, H / 2 + 16);
    ctx.textAlign = "left";
  }

  private drawGrid(
    pad: { l: number; t: number },
    plotW: number,
    plotH: number,
    maxLoss: number,
    y: (l: number) => number,
  ): void {
    const { ctx } = this;
    ctx.strokeStyle = COLOR_GRID;
    ctx.lineWidth = 1;
    ctx.fillStyle = COLOR_AXIS_TEXT;
    ctx.font = "10.5px ui-monospace, monospace";
    for (let i = 0; i <= 4; i++) {
      const loss = (maxLoss / 4) * i;
      const yy = y(loss);
      ctx.beginPath();
      ctx.moveTo(pad.l, yy);
      ctx.lineTo(pad.l + plotW, yy);
      ctx.stroke();
      ctx.fillText(loss.toFixed(1), 8, yy + 4);
    }
    this.drawFrame(pad, plotW, plotH);
  }

  private drawTrainArea(
    x: (s: number) => number,
    y: (l: number) => number,
    pad: { l: number; t: number },
    plotH: number,
  ): void {
    const { ctx } = this;
    if (this.points.length < 2) return;
    const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + plotH);
    grad.addColorStop(0, COLOR_TRAIN_FILL);
    grad.addColorStop(1, COLOR_TRAIN_FILL_FADE);
    ctx.fillStyle = grad;
    ctx.beginPath();
    let started = false;
    for (const p of this.points) {
      const px = x(p.step);
      const py = y(p.trainLoss);
      if (!started) { ctx.moveTo(px, py); started = true; }
      else ctx.lineTo(px, py);
    }
    // close the path down to the x-axis baseline.
    const lastX = x(this.points[this.points.length - 1].step);
    const firstX = x(this.points[0].step);
    ctx.lineTo(lastX, pad.t + plotH);
    ctx.lineTo(firstX, pad.t + plotH);
    ctx.closePath();
    ctx.fill();
  }

  private drawSeries(
    x: (s: number) => number,
    y: (l: number) => number,
    pick: (p: Point) => number | undefined,
    color: string,
    dashed: boolean,
  ): void {
    const { ctx } = this;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.9;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (dashed) ctx.setLineDash([4, 3]);
    ctx.beginPath();
    let started = false;
    for (const p of this.points) {
      const value = pick(p);
      if (value == null) continue;
      const px = x(p.step);
      const py = y(value);
      if (started) ctx.lineTo(px, py);
      else { ctx.moveTo(px, py); started = true; }
    }
    ctx.stroke();
    ctx.restore();
  }

  private drawLegend(pad: { l: number; t: number }): void {
    const { ctx } = this;
    const y = pad.t + 4;
    ctx.font = "500 10.5px ui-monospace, monospace";

    // train swatch
    ctx.fillStyle = COLOR_TRAIN;
    ctx.fillRect(pad.l + 8, y + 4, 10, 2);
    ctx.fillText("train", pad.l + 22, y + 8);

    // val swatch (dashed)
    ctx.strokeStyle = COLOR_VAL;
    ctx.lineWidth = 2;
    ctx.setLineDash([2.5, 2]);
    ctx.beginPath();
    ctx.moveTo(pad.l + 56, y + 5);
    ctx.lineTo(pad.l + 66, y + 5);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COLOR_VAL;
    ctx.fillText("val", pad.l + 70, y + 8);
  }

  private drawFrame(
    pad: { l: number; t: number },
    plotW: number,
    plotH: number,
  ): void {
    this.ctx.strokeStyle = COLOR_GRID;
    this.ctx.lineWidth = 1;
    this.ctx.strokeRect(pad.l, pad.t, plotW, plotH);
  }
}
