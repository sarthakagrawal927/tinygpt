import Foundation
import MLX
import MLXNN
import MLXOptimizers
import TinyGPTIO
import TinyGPTModel

/// `tinygpt sft` — supervised fine-tuning with response-only loss
/// masking. Same LoRA injection + checkpoint plumbing as `tinygpt
/// finetune`, but the training corpus is a JSONL of
/// `{instruction, input?, response}` and the loss is computed ONLY
/// on the response tokens (instruction tokens are seen but don't
/// contribute to the gradient).
///
/// This is the step that turns a base language model into one that
/// follows instructions. Without it, the model learns to autocomplete
/// the instruction itself; with it, the model learns "given THIS
/// prompt, produce THAT response."
///
/// USAGE
///   tinygpt sft <base> --data path.jsonl --template chatml --out my.lora
enum SFT {
    static func run(args: [String]) {
        var basePath: String?
        var dataPath: String?
        var outPath: String?
        var templateName = "chatml"
        var rank = 4
        var alpha: Float = 8.0
        var steps = 200
        var lr: Float = 1e-3
        var batchSize: Int? = nil
        var maxSeqLen: Int = 1024
        var nefTuneAlpha: Float = 0
        var gradClipNorm: Float = 1.0
        var loraPlusRatio: Float = 1.0
        // Curated-recipe default: DoRA on. 5-10% better than vanilla LoRA
        // at same rank for a modest compute cost. Pass `--no-dora` to fall
        // back to vanilla LoRA. See docs/audit_2026.md "PEFT — KEEP".
        var useDora: Bool = true
        var packSequences = false
        var optimizerKind: OptimizerKind = .adamw
        // PEFT variants — see Finetune.swift / docs/peft_variants.md.
        var peftVariant: PeftVariant = .lora
        var adaLoraTargetRank = 0
        var layerDropProb: Float = 0
        // pack-mode: explicit selector replacing the legacy --pack flag.
        //   none     — uniform pick (sampleBatch)
        //   sequence — multi-example rows (sampleBatchPacked, == --pack)
        //   sample   — inverse-length weighted (sampleBatchWeighted)
        //   bucket   — length-bucket uniform (sampleBatchBucketed)
        // When --pack is passed without --pack-mode, it maps to "sequence"
        // for backwards compat.
        var packMode = "none"
        var lengthBuckets = 0
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
            case "--batch":         batchSize = Int(args[i+1]); i += 2
            case "--max-seq":       maxSeqLen = Int(args[i+1]) ?? maxSeqLen; i += 2
            case "--neftune-alpha": nefTuneAlpha = Float(args[i+1]) ?? nefTuneAlpha; i += 2
            case "--grad-clip":     gradClipNorm = Float(args[i+1]) ?? gradClipNorm; i += 2
            case "--lora-plus-ratio": loraPlusRatio = Float(args[i+1]) ?? loraPlusRatio; i += 2
            case "--pack":          packSequences = true; i += 1
            case "--pack-mode":     packMode = args[i+1]; i += 2
            case "--length-bucket": lengthBuckets = Int(args[i+1]) ?? lengthBuckets; i += 2
            case "--dora":          useDora = true; i += 1
            case "--no-dora":       useDora = false; i += 1
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
            fputs("unknown template '\(templateName)'. Options: chatml, alpaca, llama, plain\n", stderr); exit(2)
        }

        print("loading base from \(basePath)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(basePath) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        let cfg = load.config

        // SFT requires a BPE tokenizer — instruction-tuning datasets
        // contain special chat markers (`<|im_start|>` etc) that need
        // BPE-level encoding. Byte-level can't see those as single
        // tokens; the response-mask boundary would be wrong.
        guard let tokDir = load.hfTokenizerDir else {
            fputs("SFT needs a tokenizer-pinned base — either an HF model dir, or a from-scratch model trained with `--tokenizer`.\nByte-level bases don't support instruction templates.\n", stderr)
            exit(2)
        }
        print("loading tokenizer from \(tokDir.lastPathComponent)…")
        let tokenizer: HFTokenizer
        do { tokenizer = try HFTokenizer.loadBlocking(from: tokDir) }
        catch { fputs("tokenizer load failed: \(error)\n", stderr); exit(1) }

        // Inject LoRA (or DoRA / one of the PEFT variants).
        let loraCfg = LoraConfig(
            rank: rank, alpha: alpha,
            targetSuffixes: ["q_proj", "v_proj"],
            useDora: useDora,
            variant: peftVariant,
            adaLoraTargetRank: adaLoraTargetRank
        )
        LayerDropState.probability = layerDropProb
        defer { LayerDropState.disable() }
        let nTrainable = load.model.injectLora(config: loraCfg)
        let nTotal = load.model.numParameters()

        // NEFTune (Jain et al., 2024): uniform noise on the embedding output
        // during training. Tiny code change, ~5-10% SFT win on small models
        // when alpha ≈ 5. Off by default; flipped on per the CLI flag.
        if nefTuneAlpha > 0 {
            switch load.model {
            case .fromScratch(let m): m.nefTuneAlpha = nefTuneAlpha
            case .huggingFace(let m): m.nefTuneAlpha = nefTuneAlpha
            }
        }

        // Build the SFT corpus: read JSONL, apply template, tokenize each
        // example with its response-only mask.
        print("reading SFT data from \(dataPath)…")
        let records: [SFTRecord]
        do { records = try SFTReader.readJSONL(URL(fileURLWithPath: dataPath)) }
        catch { fputs("read failed: \(error)\n", stderr); exit(1) }
        print("templating + tokenizing \(records.count) records…")
        let effectiveMax = min(maxSeqLen, cfg.contextLength)
        var examples: [SFTExample] = []
        examples.reserveCapacity(records.count)
        for r in records {
            do {
                let ex = try SFTBuilder.buildExample(
                    record: r, template: template, tokenizer: tokenizer,
                    maxSeqLen: effectiveMax
                )
                examples.append(ex)
            } catch {
                fputs("tokenize failed for one record: \(error)\n", stderr); continue
            }
        }
        guard !examples.isEmpty else {
            fputs("no usable SFT examples after tokenization\n", stderr); exit(1)
        }
        let corpus = SFTCorpus(examples, vocabSize: cfg.vocabSize)

        // Resolve pack-mode: legacy --pack wins if --pack-mode is left at
        // default. Sanity-check unknown modes.
        if packSequences && packMode == "none" { packMode = "sequence" }
        switch packMode {
        case "none", "sequence", "sample", "bucket": break
        default:
            fputs("unknown --pack-mode '\(packMode)'. Options: none, sequence, sample, bucket\n", stderr)
            exit(2)
        }
        if packMode == "bucket" && lengthBuckets < 1 { lengthBuckets = 4 }

        let B = batchSize ?? defaultBatch(cfg)
        let T = effectiveMax
        let templatedTokens = examples.map { $0.tokens.count }.reduce(0, +)
        let maskedTokens = examples.flatMap { $0.responseMask }.filter { $0 }.count
        print("""

        TinyGPT — SFT (response-only loss)
        ----------------------------------
        base:           \(basePath)
        template:       \(template.rawValue)
        data:           \(dataPath) · \(records.count) records · \(formatNum(templatedTokens)) tokens · \(formatNum(maskedTokens)) scored (mask=1)
        config:         \(cfg.nLayers)L · d=\(cfg.dModel) · ctx=\(cfg.contextLength) · max-seq=\(T)
        variant:        \(Finetune.describeVariant(useDora ? .dora : peftVariant, target: adaLoraTargetRank))
        \(useDora ? "DoRA" : "LoRA"):           rank=\(rank) alpha=\(alpha) targets=q_proj,v_proj
        LayerDrop:      \(layerDropProb > 0 ? "p=\(layerDropProb)" : "off")
        trainable:      \(formatNum(nTrainable))  /  total \(formatNum(nTotal))  (\(String(format: "%.2f%%", 100 * Float(nTrainable) / Float(nTotal))))
        steps:          \(steps)
        batch / lr:     \(B) / \(lr)
        NEFTune:        \(nefTuneAlpha > 0 ? "alpha=\(nefTuneAlpha)" : "off")
        grad clip:      \(gradClipNorm > 0 ? "global L2 ≤ \(gradClipNorm)" : "off")
        LoRA+:          \(loraPlusRatio > 1 ? "B-LR × \(loraPlusRatio)" : "off")
        packing:        \(packModeDescription(packMode, buckets: lengthBuckets))
        device:         \(Device.defaultDevice())

        """)
        fflush(stdout)

        // Build a masked-loss train step for whichever model variant.
        let stepFn = makeMaskedStepFn(load.model, lr: lr,
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
            let (x, y, m): (MLXArray, MLXArray, MLXArray)
            switch packMode {
            case "sequence":
                (x, y, m) = corpus.sampleBatchPacked(batchSize: B, contextLength: T)
            case "sample":
                (x, y, m) = corpus.sampleBatchWeighted(batchSize: B, contextLength: T)
            case "bucket":
                (x, y, m) = corpus.sampleBatchBucketed(
                    batchSize: B, contextLength: T, nBuckets: lengthBuckets
                )
            default:
                (x, y, m) = corpus.sampleBatch(batchSize: B, contextLength: T)
            }
            lastLoss = stepFn(x, y, m)
            lastStep = step + 1
            if step == 0 || (step + 1) % 25 == 0 || step == steps - 1 {
                let elapsed = -t0.timeIntervalSinceNow
                let sps = Double(step + 1) / elapsed
                fputs(String(format: "  step %4d/%4d  loss %.3f  · %.1f step/s · eta %.0fs\n",
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
            print(String(format: "\ninterrupted at step %d of %d after %.1fs · loss %.3f",
                          lastStep, steps, elapsed, lastLoss))
        } else {
            print(String(format: "\ndone — %d steps in %.1fs (%.1f step/s) · final loss %.3f",
                          steps, elapsed, Double(steps) / elapsed, lastLoss))
        }
        do {
            try load.model.saveLora(baseConfig: cfg, loraConfig: loraCfg,
                                     finalLoss: lastLoss,
                                     to: URL(fileURLWithPath: outPath))
            print("✓ wrote \(outPath)")
        } catch {
            fputs("save failed: \(error)\n", stderr); exit(1)
        }
        if stoppedEarly { exit(130) }
    }

    /// Per-step train function. MLX-Swift's `valueAndGrad` only supports
    /// 2-MLXArray loss signatures, so we smuggle the per-batch mask via a
    /// closure-captured `var`. `compileStep` is implicitly off — each
    /// step re-traces the loss against the current mask value.
    ///
    /// `gradClipNorm`: when non-nil, clips grads to that global L2 norm
    /// before the optimizer step.
    /// `loraPlusRatio`: when non-nil, scales any `loraB` leaf in the
    /// gradient tree by this factor (LoRA+, Hayou et al., 2024).
    /// Scaling happens AFTER clipping to mirror per-param-LR semantics.
    private static func makeMaskedStepFn(_ model: AnyModel, lr: Float,
                                          gradClipNorm: Float?,
                                          loraPlusRatio: Float?,
                                          optimizerKind: OptimizerKind)
        -> (MLXArray, MLXArray, MLXArray) -> Float
    {
        // Wrap the mask in a class so the closure captures by reference
        // (Swift closures capture vars by reference, but to be explicit
        // and isolation-safe we box it).
        final class MaskBox: @unchecked Sendable { var value: MLXArray = MLXArray(0) }
        let maskBox = MaskBox()
        let clip = gradClipNorm
        let lpRatio = loraPlusRatio

        switch model {
        case .fromScratch(let m):
            let opt = makeOptimizer(kind: optimizerKind, learningRate: lr, weightDecay: 0)
            let lossFn = { (mm: TinyGPTModel, x: MLXArray, y: MLXArray) -> MLXArray in
                AnyModel.fromScratch(mm).maskedLoss(x, y, maskBox.value)
            }
            let gradFn = valueAndGrad(model: m, lossFn)
            return { x, y, msk in
                maskBox.value = msk
                let (loss, grads) = gradFn(m, x, y)
                var final = clip.map { clipGradNorm(grads, maxNorm: $0) } ?? grads
                if let r = lpRatio { final = scaleLoraBGradients(final, ratio: r) }
                opt.update(model: m, gradients: final)
                MLX.eval(loss, m, opt)
                return loss.item(Float.self)
            }
        case .huggingFace(let m):
            let opt = makeOptimizer(kind: optimizerKind, learningRate: lr, weightDecay: 0)
            let lossFn = { (mm: TinyGPTModelHF, x: MLXArray, y: MLXArray) -> MLXArray in
                AnyModel.huggingFace(mm).maskedLoss(x, y, maskBox.value)
            }
            let gradFn = valueAndGrad(model: m, lossFn)
            return { x, y, msk in
                maskBox.value = msk
                let (loss, grads) = gradFn(m, x, y)
                var final = clip.map { clipGradNorm(grads, maxNorm: $0) } ?? grads
                if let r = lpRatio { final = scaleLoraBGradients(final, ratio: r) }
                opt.update(model: m, gradients: final)
                MLX.eval(loss, m, opt)
                return loss.item(Float.self)
            }
        }
    }

    private static func defaultBatch(_ cfg: ModelConfig) -> Int {
        if cfg.dModel >= 1024 { return 1 }
        if cfg.dModel >= 512 { return 2 }
        if cfg.dModel >= 256 { return 8 }
        return 16
    }

    private static func packModeDescription(_ mode: String, buckets: Int) -> String {
        switch mode {
        case "sequence": return "sequence (multi-example rows, greedy fit)"
        case "sample":   return "sample (inverse-length weighted; short examples over-represented)"
        case "bucket":   return "bucket (uniform over \(buckets) length buckets)"
        default:         return "none (uniform pick, one-per-row)"
        }
    }

    private static func formatNum(_ n: Int) -> String {
        let f = NumberFormatter(); f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt sft <base> [options]

        --data path.jsonl        JSONL of {instruction,input?,response} records (required)
        --out path.lora          Where to save the adapter (required)
        --template chatml|alpaca|llama|plain   (default: chatml)
        --rank N                 LoRA rank (default 4)
        --alpha F                LoRA scale (default 8.0)
        --steps N                Training steps (default 200)
        --lr F                   Learning rate (default 1e-3)
        --batch N                Batch size (default by preset)
        --max-seq N              Truncate examples longer than N tokens (default 1024)
        --neftune-alpha F        Noisy-embedding regulariser scale (Jain et al., 2024).
                                   0 = off (default); 5 is a common SFT setting.
        --grad-clip F            Global L2 grad-norm cap (default 1.0). Pass 0 to disable.
        --lora-plus-ratio F      LoRA+ B-matrix LR multiplier (Hayou et al., 2024).
                                   1.0 (default) = standard LoRA; 16.0 is the recipe.
        --pack                   Alias for --pack-mode sequence (back-compat).
        --pack-mode MODE         How to construct each batch (default none):
                                   none     — uniform random pick (one example per row)
                                   sequence — multi-example rows (greedy fit, ~3-10×
                                              effective batch on short SFT data;
                                              attention not block-masked)
                                   sample   — inverse-length weighted sampling. Short
                                              examples picked more often so each
                                              example contributes ~equally per step.
                                              Counters the natural batch bias toward
                                              long examples.
                                   bucket   — length-bucket uniform: bin examples into
                                              --length-bucket buckets, pick a bucket
                                              uniformly, then a uniform example from it.
        --length-bucket N        Number of length buckets for --pack-mode bucket
                                   (default 4 when --pack-mode bucket is set).
        --dora                   Use DoRA instead of LoRA (Liu et al., 2024).
                                   Adds a learnable per-output magnitude vector to
                                   each wrapped Linear; better quality at same rank.
                                   In-session only — DoRA adapters aren't yet on disk.

        PEFT variants (mutually exclusive; pick at most one — see docs/peft_variants.md):
        --vera                   VeRA — frozen random A/B, train per-rank scalars (~10× fewer params).
        --rs-lora                Rank-stabilized LoRA — scale = α/√r.
        --lora-fa                LoRA-FA — freeze A, train only B (½ trainable params).
        --pissa-init             PISSA — init A,B from top-r SVD of base.
        --loftq                  LoftQ — init compensates a simulated int4 quantization error.
        --adalora-target-rank R  AdaLoRA — train per-rank importance, target avg rank R.
        --layer-drop F           LayerDrop fraction (0.0-0.5) — stochastically skip whole blocks.
        """)
        exit(2)
    }
}
