import Foundation
import MLX
import MLXNN
import MLXOptimizers
import TinyGPTIO
import TinyGPTModel

/// `tinygpt dpo` — preference-optimisation trainer with four loss
/// variants selectable via `--loss-type`:
///
///   • **dpo** (Rafailov et al., 2023) — the original. Needs a frozen
///     reference; the implicit reward is `log π_pol − log π_ref`.
///   • **simpo** (Meng et al., 2024) — reference-free. Length-normalises
///     log-probabilities and adds a target reward margin γ; halves
///     memory cost (no ref model).
///   • **orpo** (Hong et al., 2024) — reference-free, combines an SFT
///     NLL term with a log-odds-ratio preference term. Often replaces
///     SFT entirely (single-stage post-training).
///   • **kto** (Ethayarajh et al., 2024) — utility-theory framing.
///     Treats `chosen` as desirable, `rejected` as undesirable; the
///     two have asymmetric loss shapes (good for unpaired feedback).
///     Needs the reference model.
///
/// All variants share the same data format (the existing JSONL of
/// `{prompt, chosen, rejected}` triplets) so users can A/B test loss
/// shapes without re-collecting data. SimPO and ORPO skip the
/// reference load entirely, cutting memory by ~half.
///
/// USAGE
///   tinygpt dpo <base> --data path.jsonl --loss-type simpo \
///       --beta 2.0 --gamma 1.0 --rank 4 --steps 500 --out my.lora
enum DPO {

    /// Preference-optimisation loss variants. `dpo` and `kto` need a
    /// reference model; `simpo` and `orpo` are reference-free.
    enum LossType: String { case dpo, simpo, orpo, kto }

    static func run(args: [String]) {
        var basePath: String?
        var dataPath: String?
        var outPath: String?
        var templateName = "chatml"
        var rank = 4
        var alpha: Float = 8.0
        var steps = 200
        var lr: Float = 5e-5             // smaller than SFT — DPO is sharp
        var beta: Float = 0.1             // 0.05-0.3 typical for DPO
        var batchSize: Int? = nil
        var maxSeqLen: Int = 1024
        var nefTuneAlpha: Float = 0
        var gradClipNorm: Float = 1.0
        var loraPlusRatio: Float = 1.0
        var useDora: Bool = false
        // Curated-recipe default: SimPO (reference-free, ½ the memory of
        // DPO at equivalent quality on published benchmarks). Override with
        // `--loss-type dpo` for the classical recipe, `--loss-type orpo`
        // for SFT+DPO in one pass, or `--loss-type kto` for single-side
        // (thumbs up/down) preference data.
        var lossType: LossType = .simpo
        // SimPO's reward-margin γ (paper recommends 1.0; loss is sensitive).
        var simpoGamma: Float = 1.0
        // ORPO's preference-term weight λ (paper recommends 0.1).
        var orpoLambda: Float = 0.1
        // PEFT variants — see Finetune.swift / docs/peft_variants.md.
        var peftVariant: PeftVariant = .lora
        var adaLoraTargetRank = 0
        var layerDropProb: Float = 0
        var optimizerKind: OptimizerKind = .adamw
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--data":          dataPath = args[i+1]; i += 2
            case "--out":           outPath = args[i+1]; i += 2
            case "--template":      templateName = args[i+1]; i += 2
            case "--rank":          rank = Int(args[i+1]) ?? rank; i += 2
            case "--alpha":         alpha = Float(args[i+1]) ?? alpha; i += 2
            case "--steps":         steps = Int(args[i+1]) ?? steps; i += 2
            case "--lr":            lr = Float(args[i+1]) ?? lr; i += 2
            case "--beta":          beta = Float(args[i+1]) ?? beta; i += 2
            case "--batch":         batchSize = Int(args[i+1]); i += 2
            case "--max-seq":       maxSeqLen = Int(args[i+1]) ?? maxSeqLen; i += 2
            case "--neftune-alpha": nefTuneAlpha = Float(args[i+1]) ?? nefTuneAlpha; i += 2
            case "--grad-clip":     gradClipNorm = Float(args[i+1]) ?? gradClipNorm; i += 2
            case "--lora-plus-ratio": loraPlusRatio = Float(args[i+1]) ?? loraPlusRatio; i += 2
            case "--dora":            useDora = true; i += 1
            case "--vera":          peftVariant = .vera; i += 1
            case "--rs-lora":       peftVariant = .rsLora; i += 1
            case "--lora-fa":       peftVariant = .loraFA; i += 1
            case "--pissa-init":    peftVariant = .pissa; i += 1
            case "--loftq":         peftVariant = .loftq; i += 1
            case "--adalora-target-rank":
                peftVariant = .adaLora
                adaLoraTargetRank = Int(args[i+1]) ?? adaLoraTargetRank; i += 2
            case "--layer-drop":    layerDropProb = Float(args[i+1]) ?? layerDropProb; i += 2
            case "--optimizer":
                guard let k = parseOptimizerKind(args[i+1]) else {
                    fputs("unknown --optimizer '\(args[i+1])'. Pick adamw|lion|sophia|muon|adafactor.\n", stderr); exit(2)
                }
                optimizerKind = k; i += 2
            case "--loss-type":
                guard let lt = LossType(rawValue: args[i+1].lowercased()) else {
                    fputs("unknown --loss-type '\(args[i+1])'. Pick dpo|simpo|orpo|kto.\n", stderr); exit(2)
                }
                lossType = lt; i += 2
            case "--gamma":         simpoGamma = Float(args[i+1]) ?? simpoGamma; i += 2
            case "--orpo-lambda":   orpoLambda = Float(args[i+1]) ?? orpoLambda; i += 2
            case "-h", "--help":    exitUsage()
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                basePath = args[i]; i += 1
            }
        }
        guard let basePath = basePath else { fputs("missing base path\n", stderr); exitUsage() }
        guard let dataPath = dataPath else { fputs("--data <jsonl> required\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out <path.lora> required\n", stderr); exitUsage() }
        guard let template = PromptTemplate(name: templateName) else {
            fputs("unknown template '\(templateName)'\n", stderr); exit(2)
        }

        // Load the policy base. The reference base is only loaded for
        // loss variants that actually need it (dpo, kto) — SimPO and ORPO
        // are reference-free and skip the second load, halving the memory.
        print("loading policy base from \(basePath)…")
        let policyLoad: ModelLoader.LoadResult
        do { policyLoad = try ModelLoader.load(basePath) }
        catch { fputs("policy load failed: \(error)\n", stderr); exit(1) }
        let needsRef = (lossType == .dpo || lossType == .kto)
        var refModel: AnyModel? = nil
        if needsRef {
            print("loading reference base from \(basePath)…")
            let refLoad: ModelLoader.LoadResult
            do { refLoad = try ModelLoader.load(basePath) }
            catch { fputs("reference load failed: \(error)\n", stderr); exit(1) }
            refModel = refLoad.model
        }
        let cfg = policyLoad.config

        guard let tokDir = policyLoad.hfTokenizerDir else {
            fputs("DPO needs a tokenizer-pinned base — either an HF model dir or a from-scratch model trained with `--tokenizer`.\n", stderr)
            exit(2)
        }
        print("loading tokenizer from \(tokDir.lastPathComponent)…")
        let tokenizer: HFTokenizer
        do { tokenizer = try HFTokenizer.loadBlocking(from: tokDir) }
        catch { fputs("tokenizer load failed: \(error)\n", stderr); exit(1) }

        // Inject the chosen PEFT variant on the policy ONLY. Reference
        // stays untouched (it must score the same as before training).
        let loraCfg = LoraConfig(rank: rank, alpha: alpha,
                                  targetSuffixes: ["q_proj", "v_proj"],
                                  useDora: useDora,
                                  variant: peftVariant,
                                  adaLoraTargetRank: adaLoraTargetRank)
        LayerDropState.probability = layerDropProb
        defer { LayerDropState.disable() }
        let nTrainable = policyLoad.model.injectLora(config: loraCfg)
        let nTotal = policyLoad.model.numParameters()

        // NEFTune on the POLICY only — the reference is meant to be a clean
        // baseline; noising it would shift the implicit reward we're
        // optimising against. Default 0 (off).
        if nefTuneAlpha > 0 {
            switch policyLoad.model {
            case .fromScratch(let m): m.nefTuneAlpha = nefTuneAlpha
            case .huggingFace(let m): m.nefTuneAlpha = nefTuneAlpha
            }
        }

        // Read + templatize + tokenize the preference data.
        print("reading preference data from \(dataPath)…")
        let records: [PreferenceRecord]
        do { records = try PreferenceReader.readJSONL(URL(fileURLWithPath: dataPath)) }
        catch { fputs("read failed: \(error)\n", stderr); exit(1) }
        print("templating + tokenizing \(records.count) records…")
        let effectiveMax = min(maxSeqLen, cfg.contextLength)
        var examples: [PreferenceExample] = []
        examples.reserveCapacity(records.count)
        for r in records {
            do {
                let ex = try PreferenceBuilder.buildExample(
                    record: r, template: template, tokenizer: tokenizer,
                    maxSeqLen: effectiveMax)
                examples.append(ex)
            } catch {
                fputs("tokenize failed for one record: \(error)\n", stderr)
            }
        }
        guard !examples.isEmpty else {
            fputs("no usable preference examples after tokenization\n", stderr); exit(1)
        }
        let corpus = PreferenceCorpus(examples, vocabSize: cfg.vocabSize)

        let B = batchSize ?? defaultBatch(cfg)
        let T = effectiveMax

        print("""

        TinyGPT — preference optimisation (\(lossType.rawValue))
        -------------
        base:           \(basePath)
        template:       \(template.rawValue)
        data:           \(dataPath) · \(records.count) preference triplets
        config:         \(cfg.nLayers)L · d=\(cfg.dModel) · ctx=\(cfg.contextLength) · max-seq=\(T)
        variant:        \(Finetune.describeVariant(useDora ? .dora : peftVariant, target: adaLoraTargetRank))
        \(useDora ? "DoRA" : "LoRA"):           rank=\(rank) alpha=\(alpha) targets=q_proj,v_proj  (policy only)
        loss:           \(describeLoss(lossType, beta: beta, gamma: simpoGamma, lambda: orpoLambda))
        LayerDrop:      \(layerDropProb > 0 ? "p=\(layerDropProb)" : "off")
        reference:      \(needsRef ? "loaded (memory ~2× base)" : "skipped (reference-free loss)")
        trainable:      \(formatNum(nTrainable))  /  total \(formatNum(nTotal))  (\(String(format: "%.2f%%", 100 * Float(nTrainable) / Float(nTotal))))
        steps:          \(steps)
        batch / lr:     \(B) / \(lr)
        NEFTune:        \(nefTuneAlpha > 0 ? "alpha=\(nefTuneAlpha) (policy only)" : "off")
        grad clip:      \(gradClipNorm > 0 ? "global L2 ≤ \(gradClipNorm)" : "off")
        LoRA+:          \(loraPlusRatio > 1 ? "B-LR × \(loraPlusRatio)" : "off")
        device:         \(Device.defaultDevice())

        """)
        fflush(stdout)

        let stepFn = makePreferenceStepFn(
            policy: policyLoad.model, ref: refModel,
            lossType: lossType, lr: lr, beta: beta,
            simpoGamma: simpoGamma, orpoLambda: orpoLambda,
            gradClipNorm: gradClipNorm > 0 ? gradClipNorm : nil,
            loraPlusRatio: loraPlusRatio > 1 ? loraPlusRatio : nil,
            optimizerKind: optimizerKind)

        TrainSupport.installSigintHandler()
        TrainSupport.stopRequested.reset()

        let t0 = Date()
        var lastLoss: Float = 0
        var stoppedEarly = false
        var lastStep = 0
        for step in 0..<steps {
            let batch = corpus.sampleBatch(batchSize: B, contextLength: T)
            lastLoss = stepFn(batch.chosen, batch.rejected)
            lastStep = step + 1
            if step == 0 || (step + 1) % 25 == 0 || step == steps - 1 {
                let elapsed = -t0.timeIntervalSinceNow
                let sps = Double(step + 1) / elapsed
                fputs(String(format: "  step %4d/%4d  loss %.4f  · %.1f step/s · eta %.0fs\n",
                             step + 1, steps, lastLoss, sps,
                             Double(steps - step - 1) / sps), stderr)
            }
            if TrainSupport.stopRequested.isSet {
                fputs("\n[SIGINT] saving adapter at step \(lastStep)…\n", stderr)
                stoppedEarly = true
                break
            }
        }
        let elapsed = -t0.timeIntervalSinceNow
        if stoppedEarly {
            print(String(format: "\ninterrupted at step %d of %d after %.1fs · loss %.4f",
                          lastStep, steps, elapsed, lastLoss))
        } else {
            print(String(format: "\ndone — %d steps in %.1fs (%.1f step/s) · final loss %.4f",
                          steps, elapsed, Double(steps) / elapsed, lastLoss))
        }
        do {
            try policyLoad.model.saveLora(baseConfig: cfg, loraConfig: loraCfg,
                                           finalLoss: lastLoss,
                                           to: URL(fileURLWithPath: outPath))
            print("✓ wrote \(outPath)")
        } catch {
            fputs("save failed: \(error)\n", stderr); exit(1)
        }
        if stoppedEarly { exit(130) }
    }

    /// Per-position forward bundle used by every preference loss:
    /// returns `(sumLogp, sumMaskedOdds)` for the response positions
    /// only, both shaped `[B]`. `sumMaskedOdds` is `Σ_t mask_t · log(1−p_t)`
    /// — needed by ORPO's log-odds-ratio; cheap to compute alongside CE so
    /// we always produce it (DPO/SimPO/KTO branches just discard it).
    private struct PreferenceForward {
        let sumLogp: MLXArray       // [B] — Σ_t mask_t · log p_t(y_t | y_{<t})
        let sumLogOneMinusP: MLXArray  // [B] — Σ_t mask_t · log(1 − p_t)
        let maskSum: MLXArray       // [B] — Σ_t mask_t  (response-token count)
    }

    private static func preferenceForward(
        _ model: AnyModel, inputs: MLXArray, targets: MLXArray, mask: MLXArray
    ) -> PreferenceForward {
        let logits: MLXArray
        switch model {
        case .fromScratch(let m): logits = m(inputs)
        case .huggingFace(let m): logits = m(inputs)
        }
        let B = inputs.shape[0]
        let T = inputs.shape[1]
        let v = logits.shape.last!
        let flatMask = mask.reshaped([-1])
        // Per-token NLL = -log p_t.
        let perTokCE = crossEntropy(
            logits: logits.reshaped([-1, v]),
            targets: targets.reshaped([-1]),
            reduction: .none
        )
        let perTokLogp = -perTokCE
        // log(1 − p_t) = log1p(-exp(log p_t)). Numerically OK for log p < 0
        // (true except at perfect predictions). At log p ≈ 0 we clamp to a
        // tiny floor so the gradient stays finite.
        let one = MLXArray(Float(1))
        let perTokOneMinusP = MLX.maximum(one - MLX.exp(perTokLogp), MLXArray(Float(1e-12)))
        let perTokLogOneMinusP = MLX.log(perTokOneMinusP)

        let sumLogp = (perTokLogp * flatMask).reshaped([B, T]).sum(axis: -1)
        let sumLogOM = (perTokLogOneMinusP * flatMask).reshaped([B, T]).sum(axis: -1)
        let maskSum = flatMask.reshaped([B, T]).sum(axis: -1) + MLXArray(Float(1e-6))
        return PreferenceForward(sumLogp: sumLogp,
                                  sumLogOneMinusP: sumLogOM,
                                  maskSum: maskSum)
    }

    /// Build the per-step preference-optimisation train function.
    ///
    /// Closure-captures the loss type and hyperparameters; routes to the
    /// right loss expression. DPO and KTO use the reference forward
    /// passes; SimPO and ORPO skip them entirely (ref is nil and never
    /// invoked).
    private static func makePreferenceStepFn(
        policy: AnyModel, ref: AnyModel?,
        lossType: LossType, lr: Float, beta: Float,
        simpoGamma: Float, orpoLambda: Float,
        gradClipNorm: Float?, loraPlusRatio: Float?,
        optimizerKind: OptimizerKind
    ) -> ((MLXArray, MLXArray, MLXArray), (MLXArray, MLXArray, MLXArray)) -> Float {
        let clip = gradClipNorm
        let lpRatio = loraPlusRatio
        // Closure-captured per-step batch (the second triplet — chosen comes
        // through valueAndGrad's two-array slot; the rejected triplet rides
        // through the box, same pattern as SFT's mask-box).
        final class Box: @unchecked Sendable {
            var chosenM = MLXArray(0)
            var rejectedX = MLXArray(0); var rejectedY = MLXArray(0); var rejectedM = MLXArray(0)
        }
        let box = Box()
        let betaA = MLXArray(beta)
        let gammaA = MLXArray(simpoGamma)
        let lambdaA = MLXArray(orpoLambda)

        /// Build the loss MLXArray from policy + (optional) ref forward
        /// bundles. Pure-MLX so it composes inside `valueAndGrad`.
        func computeLoss(polC: PreferenceForward, polR: PreferenceForward,
                          refC: PreferenceForward?, refR: PreferenceForward?) -> MLXArray {
            switch lossType {
            case .dpo:
                // L = -log σ(β · (polLogratio − refLogratio))
                let polLogratio = polC.sumLogp - polR.sumLogp
                let refLogratio = refC!.sumLogp - refR!.sumLogp
                let logits = betaA * (polLogratio - refLogratio)
                return (-logSigmoid(logits)).mean()
            case .simpo:
                // Length-normalise, subtract reward margin γ. No reference.
                // L = -log σ((β/|y_w|)·Σlogp_w − (β/|y_r|)·Σlogp_r − γ)
                let nC = polC.sumLogp / polC.maskSum
                let nR = polR.sumLogp / polR.maskSum
                let logits = betaA * (nC - nR) - gammaA
                return (-logSigmoid(logits)).mean()
            case .orpo:
                // SFT NLL term on the chosen response + λ · -log σ(log-odds
                // ratio). log_odds = sum_logp − sum_log(1 − p).
                let nllChosen = -(polC.sumLogp / polC.maskSum).mean()
                let oddsC = polC.sumLogp - polC.sumLogOneMinusP
                let oddsR = polR.sumLogp - polR.sumLogOneMinusP
                let prefTerm = (-logSigmoid(oddsC - oddsR)).mean()
                return nllChosen + lambdaA * prefTerm
            case .kto:
                // Asymmetric utility loss. Chosen → desirable side;
                // rejected → undesirable side. z₀ approximated as 0
                // (the original recipe averages a KL estimate; the
                // simplified form trains well in practice).
                //   L_des  = 1 − σ(β·(logp_pol − logp_ref))
                //   L_undes= 1 − σ(β·(logp_ref − logp_pol))
                let chosenAdv = polC.sumLogp - refC!.sumLogp
                let rejectAdv = polR.sumLogp - refR!.sumLogp
                let lDes   = MLXArray(Float(1)) - sigmoid(betaA * chosenAdv)
                let lUndes = MLXArray(Float(1)) - sigmoid(betaA * (-rejectAdv))
                return (lDes.mean() + lUndes.mean()) * MLXArray(Float(0.5))
            }
        }

        switch policy {
        case .fromScratch(let polM):
            let refLocal = ref
            let lossFn = { (m: TinyGPTModel, cX: MLXArray, cY: MLXArray) -> MLXArray in
                let polC = preferenceForward(.fromScratch(m), inputs: cX, targets: cY, mask: box.chosenM)
                let polR = preferenceForward(.fromScratch(m), inputs: box.rejectedX, targets: box.rejectedY, mask: box.rejectedM)
                let refC = refLocal.map { preferenceForward($0, inputs: cX, targets: cY, mask: box.chosenM) }
                let refR = refLocal.map { preferenceForward($0, inputs: box.rejectedX, targets: box.rejectedY, mask: box.rejectedM) }
                return computeLoss(polC: polC, polR: polR, refC: refC, refR: refR)
            }
            let gradFn = valueAndGrad(model: polM, lossFn)
            let opt = makeOptimizer(kind: optimizerKind, learningRate: lr, weightDecay: 0)
            return { chosen, rejected in
                box.chosenM = chosen.2
                box.rejectedX = rejected.0; box.rejectedY = rejected.1; box.rejectedM = rejected.2
                let (loss, grads) = gradFn(polM, chosen.0, chosen.1)
                var final = clip.map { clipGradNorm(grads, maxNorm: $0) } ?? grads
                if let r = lpRatio { final = scaleLoraBGradients(final, ratio: r) }
                opt.update(model: polM, gradients: final)
                MLX.eval(loss, polM, opt)
                return loss.item(Float.self)
            }
        case .huggingFace(let polM):
            let refLocal = ref
            let lossFn = { (m: TinyGPTModelHF, cX: MLXArray, cY: MLXArray) -> MLXArray in
                let polC = preferenceForward(.huggingFace(m), inputs: cX, targets: cY, mask: box.chosenM)
                let polR = preferenceForward(.huggingFace(m), inputs: box.rejectedX, targets: box.rejectedY, mask: box.rejectedM)
                let refC = refLocal.map { preferenceForward($0, inputs: cX, targets: cY, mask: box.chosenM) }
                let refR = refLocal.map { preferenceForward($0, inputs: box.rejectedX, targets: box.rejectedY, mask: box.rejectedM) }
                return computeLoss(polC: polC, polR: polR, refC: refC, refR: refR)
            }
            let gradFn = valueAndGrad(model: polM, lossFn)
            let opt = makeOptimizer(kind: optimizerKind, learningRate: lr, weightDecay: 0)
            return { chosen, rejected in
                box.chosenM = chosen.2
                box.rejectedX = rejected.0; box.rejectedY = rejected.1; box.rejectedM = rejected.2
                let (loss, grads) = gradFn(polM, chosen.0, chosen.1)
                var final = clip.map { clipGradNorm(grads, maxNorm: $0) } ?? grads
                if let r = lpRatio { final = scaleLoraBGradients(final, ratio: r) }
                opt.update(model: polM, gradients: final)
                MLX.eval(loss, polM, opt)
                return loss.item(Float.self)
            }
        }
    }

    /// Numerically stable log-sigmoid. `log σ(x) = -log(1 + exp(-x))`;
    /// for very negative x the exp blows up, so we branch on sign.
    private static func logSigmoid(_ x: MLXArray) -> MLXArray {
        // log σ(x) = -softplus(-x) = -log(1 + exp(-x))
        return -MLX.logAddExp(MLXArray(0.0), -x)
    }

    /// Element-wise sigmoid via `exp(logSigmoid(·))`, defined for the
    /// KTO loss where we need σ directly (not log σ). Going through
    /// logSigmoid keeps the stable form.
    private static func sigmoid(_ x: MLXArray) -> MLXArray {
        MLX.exp(logSigmoid(x))
    }

    /// One-line description of the active loss for the run-summary block.
    private static func describeLoss(_ t: LossType, beta: Float, gamma: Float, lambda: Float) -> String {
        switch t {
        case .dpo:   return "DPO · β=\(beta) (Rafailov et al., 2023)"
        case .simpo: return "SimPO · β=\(beta) γ=\(gamma) · ref-free (Meng et al., 2024)"
        case .orpo:  return "ORPO · λ=\(lambda) · NLL+odds-ratio · ref-free (Hong et al., 2024)"
        case .kto:   return "KTO · β=\(beta) · z₀=0 approx (Ethayarajh et al., 2024)"
        }
    }

    private static func defaultBatch(_ cfg: ModelConfig) -> Int {
        // DPO needs 2× the activations of SFT (chosen + rejected passes).
        if cfg.dModel >= 1024 { return 1 }
        if cfg.dModel >= 512 { return 1 }
        if cfg.dModel >= 256 { return 4 }
        return 8
    }

    private static func formatNum(_ n: Int) -> String {
        let f = NumberFormatter(); f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt dpo <base> [options]

        --data path.jsonl        JSONL of {prompt, chosen, rejected} triplets (required)
        --out path.lora          Where to save the adapter (required)
        --template chatml|alpaca|llama|plain   (default: chatml)
        --rank N                 LoRA rank (default 4)
        --alpha F                LoRA scale (default 8.0)
        --steps N                Training steps (default 200)
        --lr F                   Learning rate (default 5e-5 — smaller than SFT)
        --beta F                 DPO temperature (default 0.1; lower = stay near ref,
                                   higher = sharper preferences)
        --batch N                Batch size (default by preset)
        --max-seq N              Truncate examples longer than N (default 1024)
        --neftune-alpha F        Noisy-embedding regulariser on the POLICY (default 0=off;
                                   5 is typical). Reference stays clean.
        --grad-clip F            Global L2 grad-norm cap (default 1.0). Pass 0 to disable.
        --lora-plus-ratio F      LoRA+ B-matrix LR multiplier (default 1.0; 16 is the recipe).
        --dora                   Use DoRA (Liu et al., 2024) instead of LoRA — adds a
                                   learnable per-output magnitude. In-session only.
        --loss-type T            Loss variant: dpo (default) | simpo | orpo | kto.
                                   SimPO and ORPO are reference-free (½ memory).
        --gamma F                SimPO reward-margin γ (default 1.0).
        --orpo-lambda F          ORPO preference-term weight λ (default 0.1).

        Memory: DPO/KTO hold the base TWICE (policy + ref). SimPO/ORPO need
        only one copy. Use --dtype bfloat16
        on the original `train` if memory is tight; the adapter itself is fp32.

        PEFT variants (mutually exclusive — see docs/peft_variants.md):
        --vera                   VeRA — frozen random A/B, train per-rank scalars.
        --rs-lora                Rank-stabilized LoRA — scale = α/√r.
        --lora-fa                LoRA-FA — freeze A, train only B.
        --pissa-init             PISSA — init A,B from top-r SVD of base.
        --loftq                  LoftQ — init compensates a simulated int4 quantization error.
        --adalora-target-rank R  AdaLoRA — train per-rank importance, target avg rank R.
        --layer-drop F           LayerDrop fraction (0.0-0.5) — stochastically skip whole blocks.
        """)
        exit(2)
    }
}
