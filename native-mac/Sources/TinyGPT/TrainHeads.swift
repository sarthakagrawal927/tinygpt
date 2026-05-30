import Foundation
import MLX
import MLXNN
import MLXOptimizers
import TinyGPTIO
import TinyGPTModel

/// `tinygpt train-heads` — train Medusa / EAGLE-2 speculative-decode
/// heads on top of a FROZEN base model. Sidecar output (`.heads`),
/// loadable via `tinygpt sample --heads <path>`.
///
/// The base is closure-captured during the training step (the same
/// trick `tinygpt tuned-lens` uses) so MLX autograd treats its
/// parameters as constants — no freeze gymnastics required, no
/// accidental base-weight drift. Only the heads / draft net update.
///
/// Realistic expectations: a meaningful Medusa or EAGLE training run
/// takes hours and ideally uses logit-distillation from the base
/// rather than raw CE on a corpus (the heads should match the
/// base's NEXT-token distribution, not just predict the right
/// token). This first-cut implementation does plain shifted-CE on a
/// corpus — fast to wire, slower to converge, but architecturally
/// correct. The CLI's smoke run (50 steps) is meant to verify the
/// loop executes cleanly and the loss curve points the right way;
/// production-quality acceptance rates need 10k+ steps.
///
/// USAGE
///   tinygpt train-heads <model.tinygpt> --type {medusa|eagle} \
///       --corpus <text.txt> --steps 500 --num-heads 4 \
///       --out heads.heads
enum TrainHeads {

    enum HeadKind: String { case medusa, eagle }

    static func run(args: [String]) {
        var modelPath: String? = nil
        var corpusPath: String? = nil
        var outPath: String? = nil
        var kindRaw: String = "medusa"
        var steps = 500
        var lr: Float = 1e-3
        var batchSize: Int? = nil
        var ctxOverride: Int? = nil
        var numHeads: Int = 4
        var hiddenDim: Int? = nil

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--type":       kindRaw = args[i+1]; i += 2
            case "--corpus":     corpusPath = args[i+1]; i += 2
            case "--out":        outPath = args[i+1]; i += 2
            case "--steps":      steps = Int(args[i+1]) ?? steps; i += 2
            case "--lr":         lr = Float(args[i+1]) ?? lr; i += 2
            case "--batch":      batchSize = Int(args[i+1]); i += 2
            case "--ctx":        ctxOverride = Int(args[i+1]); i += 2
            case "--num-heads":  numHeads = max(1, Int(args[i+1]) ?? numHeads); i += 2
            case "--hidden-dim": hiddenDim = Int(args[i+1]); i += 2
            case "-h", "--help": exitUsage()
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                modelPath = args[i]; i += 1
            }
        }
        guard let modelPath = modelPath else { fputs("train-heads: missing <model>\n", stderr); exitUsage() }
        guard let corpusPath = corpusPath else { fputs("--corpus required\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out required\n", stderr); exitUsage() }
        guard let kind = HeadKind(rawValue: kindRaw) else {
            fputs("--type must be 'medusa' or 'eagle', got '\(kindRaw)'\n", stderr); exit(2)
        }

        // Load the base. First-cut targets from-scratch byte-level models —
        // the same constraint TunedLens has. BPE-base training-heads is a
        // 5-line plumbing change once needed (corpus = TokenizedCorpus).
        print("loading base model from \(modelPath)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(modelPath) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        guard case .fromScratch(let base) = load.model else {
            fputs("train-heads first-cut targets from-scratch byte-level models.\n", stderr); exit(2)
        }
        let cfg = load.config
        guard cfg.tokenizerSource == nil else {
            fputs("train-heads first-cut is byte-level only — BPE coming.\n", stderr); exit(2)
        }

        let corpus: ByteCorpus
        do { corpus = try ByteCorpus(contentsOf: URL(fileURLWithPath: corpusPath)) }
        catch { fputs("corpus read failed: \(error)\n", stderr); exit(1) }

        let B = batchSize ?? defaultBatch(cfg)
        let T = ctxOverride ?? cfg.contextLength
        let hidden = hiddenDim ?? cfg.dModel

        // Construct heads. Both kinds use the shared SpeculativeHeadConfig.
        let headCfg = ModelConfig.SpeculativeHeadConfig(
            kind: kind == .medusa ? .medusa : .eagle,
            numHeads: numHeads, hiddenDim: hidden
        )

        switch kind {
        case .medusa:
            let stack = MedusaHeadStack(cfg: headCfg, dModel: cfg.dModel, vocabSize: cfg.vocabSize)
            print("""

            TinyGPT — Medusa head training
            ------------------------------
            base:            \(modelPath)  (\(cfg.nLayers)L · d=\(cfg.dModel) · vocab=\(cfg.vocabSize))
            base frozen:     yes (closure-captured)
            corpus:          \(corpusPath) (\(corpus.bytes.count) bytes · byte-level)
            heads:           \(numHeads)  (head k predicts token at offset k+1)
            head params:     \(formatLargeInt(stack.numParameters()))
            steps / lr:      \(steps) / \(lr)
            batch / ctx:     \(B) / \(T)
            output:          \(outPath)

            """)
            trainMedusa(base: base, stack: stack, corpus: corpus,
                         B: B, T: T, steps: steps, lr: lr,
                         cfg: cfg, outPath: outPath)
        case .eagle:
            let draft = EagleDraft(dModel: cfg.dModel, vocabSize: cfg.vocabSize,
                                    numHeads: numHeads)
            // Warm-start from the base: copy token_embedding + lm_head weights
            // into the draft. The draft will train its in_proj / hidden_proj
            // / out_norm fresh; the embedding + vocab_proj stay close to the
            // base's (the gradients still flow into them but slow drift is
            // fine and matches the EAGLE recipe).
            do { try EagleWarmStart.fromBase(base, into: draft) }
            catch {
                fputs("warning: warm-start failed (\(error)); training from random init\n", stderr)
            }
            print("""

            TinyGPT — EAGLE-2 draft training
            --------------------------------
            base:            \(modelPath)  (\(cfg.nLayers)L · d=\(cfg.dModel) · vocab=\(cfg.vocabSize))
            base frozen:     yes (closure-captured)
            corpus:          \(corpusPath) (\(corpus.bytes.count) bytes · byte-level)
            unroll steps:    \(numHeads)  (draft net auto-regressively predicts k tokens ahead)
            draft params:    \(formatLargeInt(draft.numParameters()))
            warm-start:      tok_embed + vocab_proj copied from base
            steps / lr:      \(steps) / \(lr)
            batch / ctx:     \(B) / \(T)
            output:          \(outPath)

            """)
            trainEagle(base: base, draft: draft, corpus: corpus,
                        B: B, T: T, steps: steps, lr: lr,
                        cfg: cfg, hiddenDim: hidden, outPath: outPath)
        }
    }

    // MARK: - Medusa training

    private static func trainMedusa(
        base: TinyGPTModel,
        stack: MedusaHeadStack,
        corpus: ByteCorpus,
        B: Int, T: Int, steps: Int, lr: Float,
        cfg: ModelConfig,
        outPath: String
    ) {
        // Closure-capture the base — its weights show up as constants in
        // autograd. Loss = mean per-head shifted CE.
        let baseModel = base
        let lossFn = { (s: MedusaHeadStack, x: MLXArray, y: MLXArray) -> MLXArray in
            let hidden = baseModel.forwardToHidden(x)        // [B, T, d]
            let headLogits = s(hidden)                       // [[B, T, vocab], ...]
            return medusaHeadsLoss(headLogits: headLogits, targets: y)
        }
        let gradFn = valueAndGrad(model: stack, lossFn)
        let opt = AdamW(learningRate: lr, weightDecay: 0)
        TrainSupport.installSigintHandler()
        TrainSupport.stopRequested.reset()

        let t0 = Date()
        var lastLoss: Float = 0
        var firstLoss: Float = 0
        var stoppedEarly = false
        var lastStep = 0
        var lossCurve: [(Int, Float)] = []
        for step in 0..<steps {
            if TrainSupport.stopRequested.isSet { stoppedEarly = true; break }
            let (x, y) = corpus.sampleBatch(batchSize: B, contextLength: T)
            let (loss, grads) = gradFn(stack, x, y)
            opt.update(model: stack, gradients: grads)
            MLX.eval(loss, stack, opt)
            lastLoss = loss.item(Float.self)
            if step == 0 { firstLoss = lastLoss }
            lastStep = step + 1
            if step == 0 || (step + 1) % 10 == 0 || step == steps - 1 {
                lossCurve.append((step + 1, lastLoss))
            }
            if step == 0 || (step + 1) % 25 == 0 || step == steps - 1 {
                let elapsed = -t0.timeIntervalSinceNow
                let sps = Double(step + 1) / elapsed
                fputs(String(format: "  step %4d/%4d  loss %.3f  · %.1f step/s · eta %.0fs\n",
                             step + 1, steps, lastLoss, sps,
                             Double(steps - step - 1) / sps), stderr)
            }
        }
        let elapsed = -t0.timeIntervalSinceNow
        print(stoppedEarly
            ? String(format: "\ninterrupted at step %d · final mean head loss %.3f", lastStep, lastLoss)
            : String(format: "\ndone — %d steps in %.1fs · mean head loss %.3f → %.3f (Δ=%.3f)",
                      steps, elapsed, firstLoss, lastLoss, firstLoss - lastLoss))
        // Mini loss-curve print so the agent log captures whether loss
        // actually moved during the smoke run.
        if !lossCurve.isEmpty {
            fputs("loss curve sample:\n", stderr)
            for (step, loss) in lossCurve {
                fputs(String(format: "    step %4d   loss %.3f\n", step, loss), stderr)
            }
        }
        do {
            try MedusaHeadsIO.write(stack: stack, baseConfig: cfg,
                                     finalLoss: lastLoss, to: URL(fileURLWithPath: outPath))
            print("✓ wrote \(outPath)")
        } catch {
            fputs("save failed: \(error)\n", stderr); exit(1)
        }
        if stoppedEarly { exit(130) }
    }

    // MARK: - EAGLE-2 training

    private static func trainEagle(
        base: TinyGPTModel,
        draft: EagleDraft,
        corpus: ByteCorpus,
        B: Int, T: Int, steps: Int, lr: Float,
        cfg: ModelConfig,
        hiddenDim: Int,
        outPath: String
    ) {
        let baseModel = base
        let lossFn = { (d: EagleDraft, x: MLXArray, y: MLXArray) -> MLXArray in
            let hidden = baseModel.forwardToHidden(x)
            let stepLogits = eagleTrainingForward(draft: d, baseHidden: hidden, tokens: x)
            return eagleDraftLoss(stepLogits: stepLogits, targets: y)
        }
        let gradFn = valueAndGrad(model: draft, lossFn)
        let opt = AdamW(learningRate: lr, weightDecay: 0)
        TrainSupport.installSigintHandler()
        TrainSupport.stopRequested.reset()

        let t0 = Date()
        var lastLoss: Float = 0
        var firstLoss: Float = 0
        var stoppedEarly = false
        var lastStep = 0
        var lossCurve: [(Int, Float)] = []
        for step in 0..<steps {
            if TrainSupport.stopRequested.isSet { stoppedEarly = true; break }
            let (x, y) = corpus.sampleBatch(batchSize: B, contextLength: T)
            let (loss, grads) = gradFn(draft, x, y)
            opt.update(model: draft, gradients: grads)
            MLX.eval(loss, draft, opt)
            lastLoss = loss.item(Float.self)
            if step == 0 { firstLoss = lastLoss }
            lastStep = step + 1
            if step == 0 || (step + 1) % 10 == 0 || step == steps - 1 {
                lossCurve.append((step + 1, lastLoss))
            }
            if step == 0 || (step + 1) % 25 == 0 || step == steps - 1 {
                let elapsed = -t0.timeIntervalSinceNow
                let sps = Double(step + 1) / elapsed
                fputs(String(format: "  step %4d/%4d  loss %.3f  · %.1f step/s · eta %.0fs\n",
                             step + 1, steps, lastLoss, sps,
                             Double(steps - step - 1) / sps), stderr)
            }
        }
        let elapsed = -t0.timeIntervalSinceNow
        print(stoppedEarly
            ? String(format: "\ninterrupted at step %d · final mean draft loss %.3f", lastStep, lastLoss)
            : String(format: "\ndone — %d steps in %.1fs · mean draft loss %.3f → %.3f (Δ=%.3f)",
                      steps, elapsed, firstLoss, lastLoss, firstLoss - lastLoss))
        if !lossCurve.isEmpty {
            fputs("loss curve sample:\n", stderr)
            for (step, loss) in lossCurve {
                fputs(String(format: "    step %4d   loss %.3f\n", step, loss), stderr)
            }
        }
        do {
            try EagleDraftIO.write(draft: draft, baseConfig: cfg,
                                    hiddenDim: hiddenDim,
                                    finalLoss: lastLoss,
                                    to: URL(fileURLWithPath: outPath))
            print("✓ wrote \(outPath)")
        } catch {
            fputs("save failed: \(error)\n", stderr); exit(1)
        }
        if stoppedEarly { exit(130) }
    }

    // MARK: - Helpers

    private static func defaultBatch(_ cfg: ModelConfig) -> Int {
        if cfg.dModel >= 1024 { return 1 }
        if cfg.dModel >= 512 { return 2 }
        if cfg.dModel >= 256 { return 4 }
        return 8
    }

    private static func formatLargeInt(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt train-heads <model.tinygpt> --type {medusa|eagle} --corpus <text> --out <path> [options]

        --type {medusa|eagle}   Head architecture
        --corpus <text>         UTF-8 byte-level text to fit the heads on
        --out <path>            Where to save the .heads sidecar (required)
        --steps N               Training steps (default 500)
        --lr F                  Learning rate (default 1e-3)
        --batch N               Batch size (default by preset)
        --ctx N                 Context length override
        --num-heads N           Look-ahead horizon (default 4; head k predicts t+k+1)
        --hidden-dim N          Head/draft internal width (default = base d_model)

        Trains the heads on top of a FROZEN base model. The base is
        closure-captured so MLX autograd never updates its parameters.
        Output: a .heads sidecar loadable via `tinygpt sample --heads`.
        """)
        exit(2)
    }
}
