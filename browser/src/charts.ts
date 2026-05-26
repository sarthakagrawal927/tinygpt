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
  private rafId: number | null = null;
  private hoverX: number | null = null; // CSS pixels relative to canvas
  private lastPadL = 44;
  private lastPlotW = 0;
  private tooltip: HTMLDivElement | null = null;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2D canvas context unavailable");
    this.ctx = ctx;
    this.setupHiDpi();
    this.attachHoverInspect();
    this.draw();
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

  addPoint(p: Point): void {
    this.points.push(p);
    this.startPulseLoop();
    this.draw();
  }

  private startPulseLoop(): void {
    if (this.rafId != null) return;
    const tick = () => {
      this.pulsePhase += 0.06;
      this.draw();
      this.rafId = requestAnimationFrame(tick);
    };
    this.rafId = requestAnimationFrame(tick);
  }

  private stopPulseLoop(): void {
    if (this.rafId != null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
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

    const maxStep = Math.max(1, this.points[this.points.length - 1].step);
    let maxLoss = 0;
    for (const p of this.points) {
      maxLoss = Math.max(maxLoss, p.trainLoss, p.valLoss ?? 0);
    }
    maxLoss = Math.max(maxLoss, LN_256) * 1.08;

    const x = (step: number) => pad.l + (step / maxStep) * plotW;
    const y = (loss: number) => pad.t + (1 - loss / maxLoss) * plotH;

    this.drawGrid(pad, plotW, plotH, maxLoss, y);

    // ln(256) random-baseline reference line — soft amber dashes.
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

    // x-axis end label.
    ctx.fillStyle = COLOR_AXIS_TEXT;
    ctx.font = "10.5px ui-monospace, monospace";
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
