import Foundation
import MLX
import TinyGPTIO
import TinyGPTModel

/// Drives a LoRA fine-tune from the SwiftUI tab. Same pieces as the CLI
/// `tinygpt finetune` (ModelLoader → AnyModel.injectLora → step loop →
/// saveLora), but with state exposed as `@Published` so the chart and
/// counters update as training runs.
///
/// Routes through `AnyModel`, so loading a `.tinygpt` file gets the
/// from-scratch path and loading a HuggingFace directory gets the
/// HF SwiGLU+RoPE+GQA path automatically — including BPE-tokenized
/// corpus prep when the loaded model is HF.
@MainActor
final class FinetuneController: ObservableObject {
    @Published var status: String = "pick a base model and a corpus"
    @Published var lossHistory: [LossPoint] = []
    @Published var stepCount: Int = 0
    @Published var targetSteps: Int = 200
    @Published var stepsPerSec: Double = 0
    @Published var isTraining: Bool = false
    @Published var currentLoss: Float = 0
    @Published var rank: Int = 4
    @Published var alpha: Double = 8.0
    @Published var learningRate: Double = 1e-3
    @Published var basePath: String? = nil
    @Published var corpusPath: String? = nil
    @Published var savedAdapterPath: String? = nil

    private var trainTask: Task<Void, Never>? = nil

    func start() {
        cancel()
        guard let basePath, let corpusPath else {
            status = "pick both a base model and a corpus first"
            return
        }
        lossHistory.removeAll()
        stepCount = 0
        currentLoss = 0
        savedAdapterPath = nil
        isTraining = true
        status = "loading base \(URL(fileURLWithPath: basePath).lastPathComponent)…"

        let basePathC = basePath
        let corpusPathC = corpusPath
        let rankC = rank
        let alphaC = Float(alpha)
        let lrC = Float(learningRate)
        let stepsC = targetSteps

        trainTask = Task {
            await runFinetune(basePath: basePathC, corpusPath: corpusPathC,
                              rank: rankC, alpha: alphaC,
                              lr: lrC, steps: stepsC)
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

    private func runFinetune(basePath: String, corpusPath: String,
                              rank: Int, alpha: Float, lr: Float,
                              steps: Int) async {
        // 1. Load base.
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(basePath) }
        catch {
            isTraining = false
            status = "couldn't load base: \(error)"
            return
        }
        let cfg = load.config

        // 2. Inject LoRA.
        let loraCfg = LoraConfig(rank: rank, alpha: alpha,
                                  targetSuffixes: ["q_proj", "v_proj"])
        let nTrainable = load.model.injectLora(config: loraCfg)
        let nTotal = load.model.numParameters()

        // 3. Build corpus.
        let corpusURL = URL(fileURLWithPath: corpusPath)
        let sampleBatch: (Int, Int) -> (MLXArray, MLXArray)
        let corpusLabel: String
        switch load.model {
        case .fromScratch:
            let corpus: ByteCorpus
            do { corpus = try ByteCorpus(contentsOf: corpusURL) }
            catch {
                isTraining = false
                status = "couldn't read corpus: \(error)"
                return
            }
            sampleBatch = { B, T in corpus.sampleBatch(batchSize: B, contextLength: T) }
            corpusLabel = "\(formatBytes(corpus.bytes.count)) bytes"
        case .huggingFace:
            guard let tokDir = load.hfTokenizerDir else {
                isTraining = false; status = "HF model has no tokenizer dir"; return
            }
            status = "tokenizing corpus through model's BPE…"
            let text: String
            do { text = try String(contentsOf: corpusURL, encoding: .utf8) }
            catch {
                isTraining = false; status = "couldn't read corpus: \(error)"; return
            }
            let tok: HFTokenizer
            do { tok = try HFTokenizer.loadBlocking(from: tokDir) }
            catch {
                isTraining = false; status = "tokenizer load failed: \(error)"; return
            }
            let ids: [Int]
            do { ids = try tok.encode(text) }
            catch {
                isTraining = false; status = "tokenization failed: \(error)"; return
            }
            let tokens = ids.map { Int32($0) }
            let corpus = TokenizedCorpus(tokens: tokens, vocabSize: cfg.vocabSize)
            sampleBatch = { B, T in corpus.sampleBatch(batchSize: B, contextLength: T) }
            corpusLabel = "\(tokens.count) BPE tokens"
        }

        // 4. Build the right trainer (dispatch via AnyModel.makeStepFn-like
        // pattern — same as the CLI Finetune.swift).
        let stepFn = makeStepFn(load.model, lr: lr)

        let B = defaultBatch(cfg)
        status = "training · \(formatNum(nTrainable)) trainable / \(formatNum(nTotal)) total · \(corpusLabel) · B=\(B)"
        let t0 = Date()
        var lastLoss: Float = 0
        for step in 0..<steps {
            if Task.isCancelled { break }
            let (x, y) = sampleBatch(B, cfg.contextLength)
            let loss = stepFn(x, y)
            lastLoss = loss
            self.stepCount = step + 1
            self.currentLoss = loss
            if step % 5 == 0 || step == steps - 1 {
                self.lossHistory.append(LossPoint(step: step + 1, loss: loss))
            }
            let elapsed = -t0.timeIntervalSinceNow
            if elapsed > 0 { self.stepsPerSec = Double(step + 1) / elapsed }
            await Task.yield()
        }

        // 5. Save the adapter to a default location in a temp file the
        // user can grab from a "Reveal in Finder" or copy elsewhere.
        let outURL = defaultAdapterURL(basePath: basePath, corpusPath: corpusPath)
        do {
            try load.model.saveLora(baseConfig: cfg, loraConfig: loraCfg,
                                     finalLoss: lastLoss, to: outURL)
            self.savedAdapterPath = outURL.path
            self.status = "done — \(stepCount) steps · final loss \(String(format: "%.3f", lastLoss)) · saved \(outURL.lastPathComponent)"
        } catch {
            self.status = "training ran but save failed: \(error)"
        }
        self.isTraining = false
    }

    /// Dispatch trainer construction to the right backing type.
    private nonisolated func makeStepFn(_ model: AnyModel, lr: Float) -> (MLXArray, MLXArray) -> Float {
        switch model {
        case .fromScratch(let m):
            let trainer = Trainer(model: m, learningRate: lr, weightDecay: 0.0)
            return { x, y in trainer.step(inputs: x, targets: y) }
        case .huggingFace(let m):
            let trainer = TrainerHF(model: m, learningRate: lr, weightDecay: 0.0)
            return { x, y in trainer.step(inputs: x, targets: y) }
        }
    }

    private nonisolated func defaultBatch(_ cfg: ModelConfig) -> Int {
        if cfg.dModel >= 512 { return 1 }   // HF models — tight on memory
        if cfg.dModel >= 256 { return 8 }
        return 16
    }

    private nonisolated func defaultAdapterURL(basePath: String, corpusPath: String) -> URL {
        let baseName = URL(fileURLWithPath: basePath).deletingPathExtension().lastPathComponent
        let corpusName = URL(fileURLWithPath: corpusPath).deletingPathExtension().lastPathComponent
        let stamp = Int(Date().timeIntervalSince1970)
        let name = "\(baseName)-\(corpusName)-\(stamp).lora"
        return URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(name)
    }

    private nonisolated func formatNum(_ n: Int) -> String {
        let f = NumberFormatter(); f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private nonisolated func formatBytes(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1f MB", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0f KB", Double(n) / 1_000) }
        return "\(n)"
    }
}
