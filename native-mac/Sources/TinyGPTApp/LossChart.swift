import SwiftUI

/// One sample of the loss curve. Shared by both the train and fine-tune
/// tabs so the chart can render either.
struct LossPoint: Identifiable {
    let id = UUID()
    let step: Int
    let loss: Float
}

/// A live, low-chrome loss chart. Renders via Canvas for tight control over
/// the look — matches the browser playground's mint-line chart.
struct LossChart: View {
    let points: [LossPoint]

    /// Convenience init that takes `(step, loss)` tuples — useful for
    /// callers building points inline from arbitrary state.
    init(points: [LossPoint], targetSteps: Int) {
        self.points = points
        self.targetSteps = targetSteps
    }
    init(points: [Point], targetSteps: Int) {
        self.points = points.map { LossPoint(step: $0.step, loss: $0.loss) }
        self.targetSteps = targetSteps
    }
    struct Point { let step: Int; let loss: Float }
    /// The expected upper-bound step so the curve grows leftward as training
    /// progresses, instead of squishing into the left edge.
    let targetSteps: Int

    private static let yLossMax: Float = 8.0   // a fresh model starts near ln(256)≈5.55
    private static let yLossMin: Float = 0.0

    var body: some View {
        Canvas { context, size in
            let pad: (top: CGFloat, right: CGFloat, bottom: CGFloat, left: CGFloat) =
                (12, 16, 24, 40)
            let plotW = size.width - pad.left - pad.right
            let plotH = size.height - pad.top - pad.bottom
            guard plotW > 0, plotH > 0 else { return }

            // Background
            let bg = Path(CGRect(x: pad.left, y: pad.top, width: plotW, height: plotH))
            context.fill(bg, with: .color(Color(red: 0.04, green: 0.04, blue: 0.05)))

            // Grid lines (horizontal — at integer loss values).
            for lossLevel in stride(from: Float(0), through: Self.yLossMax, by: 2) {
                let y = pad.top + plotH * CGFloat((Self.yLossMax - lossLevel) / (Self.yLossMax - Self.yLossMin))
                var line = Path()
                line.move(to: CGPoint(x: pad.left, y: y))
                line.addLine(to: CGPoint(x: pad.left + plotW, y: y))
                context.stroke(line, with: .color(Color(white: 1, opacity: 0.04)), lineWidth: 1)

                // Y-axis label
                let text = Text(String(format: "%.0f", lossLevel))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(Color(white: 0.45))
                context.draw(text, at: CGPoint(x: pad.left - 6, y: y), anchor: .trailing)
            }

            // Milestone markers — words form at 2.0, grammar at 1.5
            for (loss, label) in [(Float(2.0), "↓ words form"),
                                  (Float(1.5), "↓ grammar emerges")] {
                let y = pad.top + plotH * CGFloat((Self.yLossMax - loss) / (Self.yLossMax - Self.yLossMin))
                var dashed = Path()
                dashed.move(to: CGPoint(x: pad.left, y: y))
                dashed.addLine(to: CGPoint(x: pad.left + plotW, y: y))
                context.stroke(dashed, with: .color(Color(red: 0.96, green: 0.69, blue: 0.29, opacity: 0.5)),
                               style: StrokeStyle(lineWidth: 1, dash: [3, 4]))
                let labelText = Text("loss \(String(format: "%.1f", loss))  \(label)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(Color(red: 0.96, green: 0.69, blue: 0.29, opacity: 0.7))
                context.draw(labelText, at: CGPoint(x: pad.left + 6, y: y - 8), anchor: .leading)
            }

            // The loss curve.
            guard points.count >= 2 else {
                // Empty state — show a friendly hint.
                let txt = Text("Press Start to train. Loss will appear here.")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(Color(white: 0.4))
                context.draw(txt, at: CGPoint(x: size.width / 2, y: size.height / 2), anchor: .center)
                return
            }

            let maxStep = max(targetSteps, points.last!.step)
            let xForStep: (Int) -> CGFloat = { step in
                pad.left + plotW * CGFloat(step) / CGFloat(max(maxStep, 1))
            }
            let yForLoss: (Float) -> CGFloat = { loss in
                let clamped = max(Self.yLossMin, min(Self.yLossMax, loss))
                return pad.top + plotH * CGFloat((Self.yLossMax - clamped) / (Self.yLossMax - Self.yLossMin))
            }

            var path = Path()
            path.move(to: CGPoint(x: xForStep(points[0].step), y: yForLoss(points[0].loss)))
            for p in points.dropFirst() {
                path.addLine(to: CGPoint(x: xForStep(p.step), y: yForLoss(p.loss)))
            }
            // Mint stroke
            context.stroke(path,
                           with: .color(Color(red: 72/255, green: 229/255, blue: 194/255)),
                           lineWidth: 1.8)

            // Leading edge — a small filled dot at the latest point.
            if let last = points.last {
                let lx = xForStep(last.step)
                let ly = yForLoss(last.loss)
                let halo = Path(ellipseIn: CGRect(x: lx - 6, y: ly - 6, width: 12, height: 12))
                context.fill(halo, with: .color(Color(red: 72/255, green: 229/255, blue: 194/255, opacity: 0.25)))
                let dot = Path(ellipseIn: CGRect(x: lx - 3, y: ly - 3, width: 6, height: 6))
                context.fill(dot, with: .color(Color(red: 72/255, green: 229/255, blue: 194/255)))
            }

            // X-axis: step number at the right edge.
            let stepText = Text("step \(points.last?.step ?? 0) / \(maxStep)")
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(Color(white: 0.45))
            context.draw(stepText,
                         at: CGPoint(x: pad.left + plotW, y: pad.top + plotH + 12),
                         anchor: .trailing)
        }
        .background(Theme.panel)
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Theme.line, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
