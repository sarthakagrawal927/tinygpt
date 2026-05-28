import Foundation
import MLX
import MLXRandom
import TinyGPTIO
import TinyGPTModel

/// Owns the loaded model + the in-flight generation. Views observe
/// `@Published` properties; the controller serialises model operations onto
/// its own task queue so the UI never blocks.
@MainActor
final class ModelController: ObservableObject {
    @Published var loadedItem: GalleryItem? = nil
    @Published var status: String = "no model loaded"
    @Published var paramCount: Int = 0
    @Published var deviceName: String = ""
    @Published var generated: String = ""
    @Published var isGenerating: Bool = false
    @Published var tokensPerSec: Double = 0

    private var model: TinyGPTModel? = nil
    private var modelConfig: ModelConfig? = nil
    private var generationTask: Task<Void, Never>? = nil

    init() {
        deviceName = "\(Device.defaultDevice())"
    }

    func load(_ item: GalleryItem) async {
        cancelGeneration()
        status = "loading \(item.displayName)…"
        loadedItem = nil
        model = nil
        modelConfig = nil

        // Stay on the main actor — the model isn't Sendable, and the load
        // work is small (read ~20MB file, decode fp16 to fp32, build the
        // module). The actual heavy MLX evaluation happens on the GPU
        // stream independently. SwiftUI yields to the run loop between
        // statements, so the UI stays responsive enough.
        let url = item.url
        do {
            let file = try TinyGPTFileReader.read(url)
            let h = file.header.config
            let cfg = ModelConfig(
                vocabSize: 256,
                contextLength: h.ctx ?? 256,
                nLayers: h.layers ?? 12,
                nHeads: h.heads ?? 8,
                dModel: h.dModel ?? 256,
                dMlp: h.dMlp ?? 1024
            )
            let m = TinyGPTModel(cfg)
            try TinyGPTWeightLoader.load(file, into: m)
            self.model = m
            self.modelConfig = cfg
            self.loadedItem = item
            self.paramCount = m.numParameters()
            self.status = "ready — \(formatParams(self.paramCount)) parameters on \(self.deviceName)"
        } catch {
            self.status = "failed to load \(item.displayName): \(error)"
        }
    }

    /// Stream-generate tokens. Cancel any in-flight generation first.
    func generate(prompt: String, maxTokens: Int, temperature: Float) {
        cancelGeneration()
        guard let model, let cfg = modelConfig else { return }
        generated = prompt
        isGenerating = true
        tokensPerSec = 0
        status = "generating…"

        generationTask = Task {
            await runGenerate(model: model, cfg: cfg,
                              prompt: prompt, maxTokens: maxTokens,
                              temperature: temperature)
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        generationTask = nil
        if isGenerating {
            isGenerating = false
            status = "ready"
        }
    }

    private func runGenerate(model: TinyGPTModel, cfg: ModelConfig,
                             prompt: String, maxTokens: Int, temperature: Float) async {
        let promptBytes = [UInt8](prompt.utf8)
        var idx = MLXArray(promptBytes.map { Int32($0) }, [1, promptBytes.count])

        let t0 = Date()
        var streamed = 0
        for _ in 0..<maxTokens {
            if Task.isCancelled { break }
            let T = idx.shape.last!
            let lo = max(0, T - cfg.contextLength)
            let cond = idx[0..., lo..<T]
            let logits = model(cond)
            let last = logits[0..., logits.shape[1] - 1, 0...]

            let nextId: MLXArray
            if temperature <= 0 {
                nextId = argMax(last, axis: -1).reshaped([1, 1])
            } else {
                let scaled = last / MLXArray(temperature)
                nextId = MLXRandom.categorical(scaled).reshaped([1, 1])
            }
            eval(nextId)
            let id = Int(nextId.item(Int32.self))
            idx = concatenated([idx, nextId.asType(idx.dtype)], axis: 1)
            streamed += 1

            // Stream one token at a time to the UI; coalesce updates via
            // an actor-isolated assign to avoid SwiftUI churn (re-render
            // per token is fine at 100 tok/s).
            let glyph: String
            if let scalar = UnicodeScalar(id), id >= 9 {
                glyph = String(scalar)
            } else {
                glyph = ""
            }
            await MainActor.run { [glyph] in
                self.generated.append(glyph)
                let elapsed = -t0.timeIntervalSinceNow
                if elapsed > 0 {
                    self.tokensPerSec = Double(streamed) / elapsed
                }
            }
        }
        await MainActor.run {
            self.isGenerating = false
            self.status = "done — \(streamed) tokens at \(String(format: "%.0f", self.tokensPerSec)) tok/s"
        }
    }

    private func formatParams(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }
}
