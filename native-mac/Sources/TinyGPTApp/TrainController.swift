import Foundation
import MLX
import TinyGPTIO
import TinyGPTModel

/// Owns a training run and streams progress to the UI. Lives on the main
/// actor so views can subscribe directly; training-step work happens
/// inline (MLX is GPU-async, so the per-step cost on the main thread
/// is mostly orchestration, not compute).
@MainActor
final class TrainController: ObservableObject {
    @Published var lossHistory: [LossPoint] = []
    @Published var status: String = "configure a run and press Start"
    @Published var stepCount: Int = 0
    @Published var targetSteps: Int = 1000
    @Published var stepsPerSec: Double = 0
    @Published var isTraining: Bool = false
    @Published var currentLoss: Float = 0
    @Published var sampleText: String = ""
    @Published var presetIdx: Int = 0  // index into Self.presets

    struct LossPoint: Identifiable {
        let id = UUID()
        let step: Int
        let loss: Float
    }

    /// Picks the user can choose between. Each is a (name, config) pair.
    static let presets: [(name: String, cfg: ModelConfig)] = [
        ("Tiny",  ModelConfig(vocabSize: 256, contextLength: 128, nLayers: 4,
                              nHeads: 4, dModel: 128, dMlp: 512)),
        ("Small", ModelConfig(vocabSize: 256, contextLength: 256, nLayers: 6,
                              nHeads: 6, dModel: 192, dMlp: 768)),
        ("Huge",  ModelConfig.huge),
        ("Mega",  ModelConfig.mega),
    ]

    private var trainTask: Task<Void, Never>? = nil

    func start(corpus: Data) {
        cancel()
        let cfg = Self.presets[presetIdx].cfg
        let presetName = Self.presets[presetIdx].name
        lossHistory.removeAll()
        stepCount = 0
        currentLoss = 0
        sampleText = ""
        isTraining = true
        status = "building \(presetName) (\(formatParams(estimateParams(cfg))) params)…"

        trainTask = Task {
            await runTraining(corpus: corpus, cfg: cfg, presetName: presetName)
        }
    }

    func cancel() {
        trainTask?.cancel()
        trainTask = nil
        if isTraining {
            isTraining = false
            status = "stopped at step \(stepCount)"
        }
    }

    private func runTraining(corpus: Data, cfg: ModelConfig, presetName: String) async {
        let model = TinyGPTModel(cfg)
        let trainer = Trainer(model: model, compileStep: true)
        let byteCorpus = ByteCorpus(corpus)
        let batchSize = batchSizeFor(cfg)

        status = "training \(presetName) · batch \(batchSize) · ctx \(cfg.contextLength)"
        let t0 = Date()

        for step in 0..<targetSteps {
            if Task.isCancelled { break }
            let (x, y) = byteCorpus.sampleBatch(batchSize: batchSize, contextLength: cfg.contextLength)
            let loss = trainer.step(inputs: x, targets: y)

            self.stepCount = step + 1
            self.currentLoss = loss
            // Append every 5th step (keeps the chart light at 1000+ steps).
            if step % 5 == 0 || step == targetSteps - 1 {
                self.lossHistory.append(LossPoint(step: step + 1, loss: loss))
            }
            let elapsed = -t0.timeIntervalSinceNow
            if elapsed > 0 { self.stepsPerSec = Double(step + 1) / elapsed }

            // Yield to give the UI a chance to repaint.
            await Task.yield()
        }
        self.isTraining = false
        self.status = "done — \(stepCount) steps in \(String(format: "%.1f", -t0.timeIntervalSinceNow))s, final loss \(String(format: "%.3f", currentLoss))"
    }

    /// Conservative batch sizes — Mega at B=8 is tight on a 16 GB box; smaller
    /// presets can go larger but the perf difference is small at this scale.
    private func batchSizeFor(_ cfg: ModelConfig) -> Int {
        if cfg.dModel >= 512 { return 4 }
        if cfg.dModel >= 256 { return 8 }
        return 16
    }

    private func estimateParams(_ cfg: ModelConfig) -> Int {
        let v = cfg.vocabSize, c = cfg.dModel, ctx = cfg.contextLength
        let m = cfg.dMlp, l = cfg.nLayers
        return v*c + ctx*c + 2*c + l*(4*(c*c + c) + 4*c + 2*(c*m + m + c))
    }

    private func formatParams(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n)/1_000_000) }
        if n >= 1_000 { return String(format: "%.0fK", Double(n)/1_000) }
        return "\(n)"
    }
}
