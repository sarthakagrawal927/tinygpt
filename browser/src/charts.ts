/**
 * charts.ts — minimal training-loss chart (Phase 4).
 *
 * A dependency-free canvas line chart of train/val loss vs step. Fed from
 * TrainingProgress messages on the main thread.
 *
 * Guide: docs/browser_notes.md ("Web Worker")
 */

interface Point {
  step: number;
  trainLoss: number;
  valLoss?: number;
}

export class LossChart {
  private readonly ctx: CanvasRenderingContext2D;
  private points: Point[] = [];

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2D canvas context unavailable");
    this.ctx = ctx;
    this.draw();
  }

  reset(): void {
    this.points = [];
    this.draw();
  }

  addPoint(p: Point): void {
    this.points.push(p);
    this.draw();
  }

  private draw(): void {
    const { ctx, canvas } = this;
    const W = canvas.width;
    const H = canvas.height;
    const pad = { l: 44, r: 12, t: 12, b: 26 };
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;

    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, W, H);

    if (this.points.length === 0) {
      ctx.fillStyle = "#7d8590";
      ctx.font = "13px system-ui, sans-serif";
      ctx.fillText("loss chart — start a run", pad.l, H / 2);
      this.drawFrame(pad, plotW, plotH);
      return;
    }

    const maxStep = Math.max(1, this.points[this.points.length - 1].step);
    let maxLoss = 0;
    for (const p of this.points) {
      maxLoss = Math.max(maxLoss, p.trainLoss, p.valLoss ?? 0);
    }
    maxLoss = Math.max(maxLoss, 0.1) * 1.05;

    const x = (step: number) => pad.l + (step / maxStep) * plotW;
    const y = (loss: number) => pad.t + (1 - loss / maxLoss) * plotH;

    this.drawFrame(pad, plotW, plotH);

    // y-axis ticks
    ctx.fillStyle = "#7d8590";
    ctx.font = "11px system-ui, sans-serif";
    for (let i = 0; i <= 4; i++) {
      const loss = (maxLoss / 4) * i;
      const yy = y(loss);
      ctx.fillText(loss.toFixed(1), 6, yy + 4);
      ctx.strokeStyle = "#21262d";
      ctx.beginPath();
      ctx.moveTo(pad.l, yy);
      ctx.lineTo(pad.l + plotW, yy);
      ctx.stroke();
    }
    ctx.fillText(`step ${maxStep}`, pad.l + plotW - 56, H - 8);

    this.drawSeries(x, y, (p) => p.trainLoss, "#58a6ff");
    this.drawSeries(x, y, (p) => p.valLoss, "#f0883e");

    // legend
    ctx.fillStyle = "#58a6ff";
    ctx.fillText("train", pad.l + 6, pad.t + 14);
    ctx.fillStyle = "#f0883e";
    ctx.fillText("val", pad.l + 52, pad.t + 14);
  }

  private drawSeries(
    x: (s: number) => number,
    y: (l: number) => number,
    pick: (p: Point) => number | undefined,
    color: string,
  ): void {
    const { ctx } = this;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.75;
    ctx.beginPath();
    let started = false;
    for (const p of this.points) {
      const value = pick(p);
      if (value == null) continue;
      const px = x(p.step);
      const py = y(value);
      if (started) ctx.lineTo(px, py);
      else {
        ctx.moveTo(px, py);
        started = true;
      }
    }
    ctx.stroke();
  }

  private drawFrame(
    pad: { l: number; t: number },
    plotW: number,
    plotH: number,
  ): void {
    this.ctx.strokeStyle = "#30363d";
    this.ctx.lineWidth = 1;
    this.ctx.strokeRect(pad.l, pad.t, plotW, plotH);
  }
}
