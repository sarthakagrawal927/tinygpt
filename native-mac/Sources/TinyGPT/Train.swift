import Foundation
import MLX
import TinyGPTIO
import TinyGPTModel

/// `tinygpt train` — train a model from scratch on a UTF-8 text corpus.
///
/// Long-run features (Tier 0 safety nets):
///   --resume <path.tinygpt>     Resume weights + step from a checkpoint
///                               (Adam state restarts — 100-step warmup,
///                                see `docs/training_phases_roadmap.md`)
///   --save-every N              Atomic checkpoint every N steps. A crash
///                               leaves the last successful checkpoint
///                               intact (write-to-.tmp then rename).
///   --lr-schedule cosine        Linear warmup + cosine decay.
///   --warmup N                  Warmup steps (default 0 — constant LR).
///   --max-lr / --min-lr         Cosine endpoints (defaults 3e-4 / 3e-5).
///   --val-split 0.0-0.2         Hold out last fraction of corpus for val.
///   --val-every N               Eval val loss every N steps (default 200).
///
/// Ctrl-C is cooperative: the next step finishes, a final checkpoint is
/// flushed, and the process exits cleanly.
enum Train {
    static func run(args: [String]) {
        var preset = "tiny"
        var steps = 500
        var corpusPath: String? = nil
        var outPath: String? = nil
        var dtype = "float32"
        var batchSize: Int? = nil
        var sampleEvery = 100
        // Tier 0 additions:
        var resumePath: String? = nil
        var saveEvery: Int? = nil
        var lrSchedule = "constant"
        var warmupSteps: Int = 0
        var maxLR: Float = 3e-4
        var minLR: Float = 3e-5
        var valSplit: Double = 0
        var valEvery: Int = 200
        var tokenizerDir: String? = nil
        var ctxOverride: Int? = nil
        var accumSteps: Int = 1
        // Default-on at 1.0 — standard transformer-LM stability lever, almost
        // never a no-op cost on well-behaved runs, saves bf16 blowups.
        // Pass `--grad-clip 0` to disable.
        var gradClipNorm: Float = 1.0
        // Mixture-of-Experts. `nExperts == 1` = standard dense MLP. When
        // > 1, every block's MLP becomes an MoE with a learned router.
        // Top-K is the number of experts each token activates (1 = Switch
        // Transformer, 2 = Mixtral-style). aux weight scales the load-
        // balance loss that keeps the router from collapsing.
        var moeExperts: Int = 1
        var moeTopK: Int = 1
        var moeAuxWeight: Float = 0.01
        // Multi-Token Prediction horizons (Gloeckle et al., 2024;
        // DeepSeek-V3). 1 = standard next-token. 2-4 typical for the
        // regulariser to bite without ballooning per-step compute.
        var mtpHorizons: Int = 1
        // Sliding-window attention (Mistral / GPT-OSS). nil = full causal.
        // When set, each query attends to only the last `slidingWindow`
        // positions — bounds attn memory/compute at long context.
        var slidingWindow: Int? = nil
        // ALiBi position bias (Press et al., 2021). When set, the model
        // drops learned positional embeddings and uses a per-head linear-
        // distance bias instead. Cleaner generalisation to longer contexts.
        var useALiBi: Bool = false
        // Mixture-of-Depths: per-token sigmoid gate on each block's
        // residual contribution (Raposo et al., 2024). Pure architecture
        // change; the dense compute path is unchanged.
        var useMoD: Bool = false
        // Differential attention (Ye et al., 2024). Doubles the Q/K
        // projections per block + adds a learnable λ — used for less-
        // noisy attention. Mutually exclusive with the standard path.
        var useDiffAttn: Bool = false
        // YOCO (Lin et al., 2024). Second half cross-attends to the
        // anchor; halves KV cache memory at long-context decode.
        var useYOCO: Bool = false
        // Gradient (activation) checkpointing. Trades ~30% extra
        // compute for a large reduction in activation memory. Each
        // TransformerBlock's forward is wrapped in a CustomFunction
        // whose VJP recomputes the block forward at backward time.
        var useGradCheckpoint: Bool = false
        // Optimiser choice (Lion, Sophia, Muon, Adafactor; default
        // AdamW preserves backward compat). See `Optimizers.swift`.
        var optimizerKind: OptimizerKind = .adamw

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--preset":      preset = args[i+1]; i += 2
            case "--steps":       steps = Int(args[i+1]) ?? steps; i += 2
            case "--corpus":      corpusPath = args[i+1]; i += 2
            case "--out":         outPath = args[i+1]; i += 2
            case "--dtype":       dtype = args[i+1]; i += 2
            case "--batch":       batchSize = Int(args[i+1]); i += 2
            case "--sample-every": sampleEvery = Int(args[i+1]) ?? sampleEvery; i += 2
            case "--resume":      resumePath = args[i+1]; i += 2
            case "--save-every":  saveEvery = Int(args[i+1]); i += 2
            case "--lr-schedule": lrSchedule = args[i+1]; i += 2
            case "--warmup":      warmupSteps = Int(args[i+1]) ?? warmupSteps; i += 2
            case "--max-lr":      maxLR = Float(args[i+1]) ?? maxLR; i += 2
            case "--min-lr":      minLR = Float(args[i+1]) ?? minLR; i += 2
            case "--val-split":   valSplit = Double(args[i+1]) ?? valSplit; i += 2
            case "--val-every":   valEvery = Int(args[i+1]) ?? valEvery; i += 2
            case "--tokenizer":   tokenizerDir = args[i+1]; i += 2
            case "--ctx":         ctxOverride = Int(args[i+1]); i += 2
            case "--accum":       accumSteps = max(1, Int(args[i+1]) ?? 1); i += 2
            case "--grad-clip":   gradClipNorm = Float(args[i+1]) ?? gradClipNorm; i += 2
            case "--moe-experts": moeExperts = max(1, Int(args[i+1]) ?? 1); i += 2
            case "--moe-topk":    moeTopK = max(1, Int(args[i+1]) ?? 1); i += 2
            case "--moe-aux-weight": moeAuxWeight = Float(args[i+1]) ?? moeAuxWeight; i += 2
            case "--mtp-horizons":   mtpHorizons = max(1, Int(args[i+1]) ?? 1); i += 2
            case "--sliding-window": slidingWindow = Int(args[i+1]); i += 2
            case "--alibi":          useALiBi = true; i += 1
            case "--mod":            useMoD = true; i += 1
            case "--diff-attn":      useDiffAttn = true; i += 1
            case "--yoco":           useYOCO = true; i += 1
            case "--grad-checkpoint": useGradCheckpoint = true; i += 1
            case "--optimizer":
                guard let k = parseOptimizerKind(args[i+1]) else {
                    fputs("unknown --optimizer '\(args[i+1])'. Pick adamw|lion|sophia|muon|adafactor.\n", stderr); exit(2)
                }
                optimizerKind = k; i += 2
            case "-h", "--help":  exitUsage()
            default:
                fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
            }
        }

        // Model + config — either fresh from preset, or resumed from .tinygpt.
        // If --tokenizer <dir> is set OR the resumed checkpoint carries one,
        // override vocabSize from the HF tokenizer/config and switch to BPE.
        var cfg: ModelConfig
        let model: TinyGPTModel
        var startStep: Int = 0
        if let r = resumePath {
            let url = URL(fileURLWithPath: r)
            let file: TinyGPTFile
            do { file = try TinyGPTFileReader.read(url) }
            catch { fputs("error reading resume file: \(error)\n", stderr); exit(1) }
            let h = file.header.config
            // Resume restores the tokenizer source the model was trained with;
            // ignore --tokenizer if the resumed checkpoint already pins one,
            // because changing tokenizers mid-training corrupts learned weights.
            let resumedTokenizer = h.tokenizerSource ?? tokenizerDir
            cfg = ModelConfig(
                vocabSize: h.vocabSize ?? 256,
                contextLength: h.ctx ?? 256,
                nLayers: h.layers ?? 12,
                nHeads: h.heads ?? 8,
                dModel: h.dModel ?? 256,
                dMlp: h.dMlp ?? 1024,
                tokenizerSource: resumedTokenizer,
                // MoE: if the resumed file carries MoE metadata, restore
                // the same router/expert layout. CLI MoE flags are ignored
                // on resume — changing architecture mid-run corrupts state.
                nExperts: h.nExperts ?? 1,
                moeTopK: h.moeTopK ?? 1,
                loadBalanceWeight: h.loadBalanceWeight ?? 0.01,
                slidingWindow: h.slidingWindow,
                useMoD: h.useMoD ?? false,
                useDifferentialAttention: h.useDifferentialAttention ?? false,
                useYOCO: h.useYOCO ?? false,
                // Grad-checkpoint travels with the checkpoint so a
                // resumed long run keeps the same memory profile. CLI
                // --grad-checkpoint can ALSO promote a non-checkpointed
                // resume into a checkpointed continuation.
                useGradCheckpoint: (h.useGradCheckpoint ?? false) || useGradCheckpoint
            )
            cfg.dtype = dtype
            model = TinyGPTModel(cfg)
            do { try TinyGPTWeightLoader.load(file, into: model) }
            catch { fputs("error loading weights: \(error)\n", stderr); exit(1) }
            startStep = Int(file.step)
            print("resuming from \(r) at step \(startStep) (Adam state restarts)")
        } else {
            cfg = configFor(preset)
            cfg.dtype = dtype
            // Apply tokenizer override BEFORE building the model — vocabSize
            // determines the token-embedding shape.
            if let tdir = tokenizerDir {
                let hfConfigURL = URL(fileURLWithPath: tdir).appendingPathComponent("config.json")
                if let hfConfig = try? HuggingFaceConfig.read(hfConfigURL) {
                    cfg.vocabSize = hfConfig.vocabSize
                } else {
                    fputs("warning: no config.json in \(tdir) — vocabSize stays at \(cfg.vocabSize)\n", stderr)
                }
                cfg.tokenizerSource = tdir
            }
            // --ctx overrides the preset's context length. Useful when the
            // preset's default is too big for memory or when the user wants
            // longer-range BPE training (Mega default 1024 → 2048 etc).
            if let c = ctxOverride { cfg.contextLength = c }
            // MoE: convert dense MLP blocks into router + expert MLPs. Only
            // honoured on FRESH configs — resumed checkpoints keep whatever
            // structure they were saved with (MoE save/load is a follow-up,
            // see the guard below the model build).
            if moeExperts > 1 {
                cfg.nExperts = moeExperts
                cfg.moeTopK = min(moeTopK, moeExperts)
                cfg.loadBalanceWeight = moeAuxWeight
            }
            // MTP: extra heads materialise inside TinyGPTModel.init when
            // mtpHorizons > 1. They're training-only — see save guard
            // below: manifest entries don't include them, so they're
            // silently dropped on serialise.
            cfg.mtpHorizons = mtpHorizons
            // Sliding window: pure attention-mask change, no extra params.
            // The CausalSelfAttention init reads cfg.slidingWindow.
            if let sw = slidingWindow, sw > 0 { cfg.slidingWindow = sw }
            // ALiBi: when enabled, the model uses NO positional embedding
            // (the position info comes from the attention bias). We still
            // construct the positional embedding table for parameter-name
            // compatibility with the manifest — it's just frozen at init.
            cfg.useALiBi = useALiBi
            // MoD: every block gets a per-token sigmoid gate; manifest
            // gains mod_router.weight/bias per layer.
            cfg.useMoD = useMoD
            // Differential attention: every block gets a diff_attn
            // sibling with 2× Q/K + λ; manifest gains the new
            // q1_proj/k1_proj/q2_proj/k2_proj/v_proj/o_proj/lambda
            // entries per layer (the existing attn entries also stay
            // — see TransformerBlock for the rationale).
            cfg.useDifferentialAttention = useDiffAttn
            // YOCO: half the layers reuse the first half's K, V via
            // cross-attention. Manifest stays identical to the standard
            // dense path; the change is purely in forward orchestration.
            cfg.useYOCO = useYOCO
            // Gradient checkpointing — must be set BEFORE the model is
            // built so each TransformerBlock picks it up at init time.
            cfg.useGradCheckpoint = useGradCheckpoint
            model = TinyGPTModel(cfg)
        }
        // MoE checkpoints now serialise — the manifest gains router +
        // per-expert entries when cfg.isMoE, and the header carries
        // nExperts/moeTopK/loadBalanceWeight so resume + sample can
        // reconstruct the same router/expert layout.
        // bf16 / fp16 training: cast every floating-point parameter to the
        // target dtype. MLX propagates the dtype through all forward / loss
        // / gradient / optimizer ops, so this single cast switches the
        // whole training loop to half precision.
        //
        // bf16 keeps fp32's range (8-bit exponent), so it doesn't need the
        // loss-scaling / master-weights scaffolding fp16 training requires.
        // ~2× memory savings vs fp32 — biggest single lever for fitting
        // larger batches and longer contexts.
        if cfg.mlxDType != .float32 {
            model.apply { $0.dtype.isFloatingPoint ? $0.asType(cfg.mlxDType) : $0 }
            print("model parameters cast to \(cfg.dtype) (memory ~½ of fp32)")
        }

        // Pre-flight memory estimate — runs BEFORE the slow tokenize step so
        // a doomed config can be aborted cheaply. Activations live and die
        // within one micro-batch, so we estimate using the per-micro-batch
        // size (`--batch`), not the effective batch (which is just an
        // accumulator-trick). A >60%-of-RAM projection warns the user.
        let microBatch = batchSize ?? defaultBatch(cfg)
        let memEstimate = OOMGuard.estimate(cfg: cfg, params: model.numParameters(),
                                              batch: microBatch)
        OOMGuard.reportAndWarn(memEstimate)

        // Load the corpus. Two flavours:
        //   - byte-level (vocabSize == 256, no tokenizer): raw bytes →
        //     ByteCorpus. Same shape we've used since day one.
        //   - BPE (vocabSize from HF config): UTF-8 text → HFTokenizer.encode
        //     → TokenizedCorpus. The on-disk size becomes irrelevant; what
        //     matters is the token count.
        //
        // Both expose the same sample-batch closure shape so the training
        // loop below is corpus-flavor-agnostic.
        let sampleTrainBatch: (Int, Int) -> (MLXArray, MLXArray)
        let valSampleBatch: ((Int, Int) -> (MLXArray, MLXArray))?
        let corpusSummary: String
        let trainSummary: String
        let valSummary: String
        if cfg.tokenizerSource != nil {
            let tokDir = URL(fileURLWithPath: cfg.tokenizerSource!)
            print("loading BPE tokenizer from \(tokDir.lastPathComponent)…")
            let tok: HFTokenizer
            do { tok = try HFTokenizer.loadBlocking(from: tokDir) }
            catch { fputs("tokenizer load failed: \(error)\n", stderr); exit(1) }
            guard let p = corpusPath else {
                fputs("--corpus is required when --tokenizer is set\n", stderr); exit(1)
            }
            let corpusURL = URL(fileURLWithPath: p)
            // Persistent token cache — keyed on (corpus, tokenizer, size,
            // mtime, vocab) so any change forces a fresh tokenize. Saves
            // 10-30 min on big corpora across re-runs / --resume cycles.
            let cacheURL = TokenCache.cacheURL(corpus: corpusURL, tokenizerDir: tokDir,
                                                vocabSize: cfg.vocabSize)
            let fileSize = ((try? FileManager.default.attributesOfItem(atPath: p))?[.size]
                            as? NSNumber)?.intValue ?? 0
            let tokens: [Int32]
            if let cu = cacheURL, let cached = TokenCache.read(cu) {
                tokens = cached
                print("loaded \(formatLargeInt(tokens.count)) tokens from cache: \(cu.lastPathComponent)")
            } else {
                let text: String
                do { text = try String(contentsOfFile: p, encoding: .utf8) }
                catch { fputs("error reading corpus: \(error)\n", stderr); exit(1) }
                print("encoding corpus (\(formatBytes(text.utf8.count)))…")
                let ids: [Int]
                do { ids = try tok.encode(text) }
                catch { fputs("tokenize failed: \(error)\n", stderr); exit(1) }
                tokens = ids.map { Int32($0) }
                if let cu = cacheURL {
                    do {
                        try TokenCache.write(tokens, to: cu)
                        print("cached \(formatLargeInt(tokens.count)) tokens → \(cu.lastPathComponent)")
                    } catch {
                        // Non-fatal — next run just re-tokenizes.
                        fputs("warning: cache write failed (\(error))\n", stderr)
                    }
                }
            }
            let full = TokenizedCorpus(tokens: tokens, vocabSize: cfg.vocabSize)
            let (tr, va) = full.split(valSplit: valSplit)
            sampleTrainBatch = { B, T in tr.sampleBatch(batchSize: B, contextLength: T) }
            valSampleBatch = va.map { v in { B, T in v.sampleBatch(batchSize: B, contextLength: T) } }
            corpusSummary = "\(corpusPath ?? "<text>") (\(formatBytes(fileSize)) · \(formatLargeInt(tokens.count)) BPE tokens · vocab=\(cfg.vocabSize))"
            trainSummary = "\(formatLargeInt(tr.tokens.count)) tokens"
            valSummary = va.map { "\(formatLargeInt($0.tokens.count)) tokens" } ?? "—"
        } else {
            let corpusFull: ByteCorpus
            if let p = corpusPath {
                do {
                    corpusFull = try ByteCorpus(contentsOf: URL(fileURLWithPath: p))
                } catch {
                    fputs("error reading corpus: \(error)\n", stderr); exit(1)
                }
            } else {
                print("⚠ no --corpus given, training on random bytes (loss will land at ~ln(256)=5.55)")
                let randomBytes = (0..<1_000_000).map { _ in UInt8.random(in: 0...255) }
                corpusFull = ByteCorpus(Data(randomBytes))
            }
            let (tr, va) = TrainSupport.splitCorpus(corpusFull, valSplit: valSplit)
            sampleTrainBatch = { B, T in tr.sampleBatch(batchSize: B, contextLength: T) }
            valSampleBatch = va.map { v in { B, T in v.sampleBatch(batchSize: B, contextLength: T) } }
            corpusSummary = "\(corpusPath ?? "<random>") (\(formatBytes(corpusFull.bytes.count)) · byte-level)"
            trainSummary = formatBytes(tr.bytes.count)
            valSummary = va.map { formatBytes($0.bytes.count) } ?? "—"
        }

        // Trainer. Compile is only safe when LR is constant AND no
        // gradient accumulation — both would mutate the compiled graph.
        // Disabled otherwise.
        let useSchedule = (lrSchedule == "cosine" || warmupSteps > 0)
        let initialLR: Float = useSchedule ? TrainSupport.lrAt(
            step: startStep, total: steps, warmup: warmupSteps,
            maxLR: maxLR, minLR: minLR
        ) : maxLR
        let B = batchSize ?? defaultBatch(cfg)
        let canCompile = !useSchedule && accumSteps == 1
        let effectiveClip: Float? = gradClipNorm > 0 ? gradClipNorm : nil
        let trainer = Trainer(model: model, learningRate: initialLR,
                              compileStep: canCompile,
                              gradClipNorm: effectiveClip,
                              optimizer: optimizerKind)

        let effB = B * accumSteps
        print("""

        TinyGPT — training run
        ---------------------
        preset:        \(preset) (\(cfg.nLayers)L · d=\(cfg.dModel) · ctx=\(cfg.contextLength))\(cfg.isMoE ? " · MoE(\(cfg.nExperts) experts, top-\(cfg.moeTopK))" : "")\(cfg.mtpHorizons > 1 ? " · MTP(\(cfg.mtpHorizons) horizons)" : "")\(cfg.slidingWindow.map { " · sliding-window=\($0)" } ?? "")\(cfg.useALiBi ? " · ALiBi" : "")
        params:        \(formatLargeInt(model.numParameters()))
        vocab:         \(formatLargeInt(cfg.vocabSize))\(cfg.tokenizerSource != nil ? " (BPE)" : " (byte-level)")
        dtype:         \(cfg.dtype)
        batch size:    \(B)\(accumSteps > 1 ? " × \(accumSteps) accum = \(effB) effective" : "")
        steps:         \(startStep) → \(steps)
        corpus:        \(corpusSummary)
        train/val:     \(trainSummary) / \(valSummary)
        lr schedule:   \(lrSchedule)\(useSchedule ? " (warmup \(warmupSteps), max \(maxLR), min \(minLR))" : " @ \(maxLR)")
        optimizer:     \(optimizerKind.rawValue)
        grad clip:     \(effectiveClip.map { "global L2 ≤ \($0)" } ?? "off")
        grad ckpt:     \(cfg.useGradCheckpoint ? "on (per-block VJP recompute · ~30% slower, ~√L activation mem)" : "off")
        save-every:    \(saveEvery.map { "\($0) steps · atomic" } ?? "end only")
        compile:       \(canCompile ? "on" : (useSchedule ? "off (LR scheduling)" : "off (gradient accumulation)"))
        device:        \(Device.defaultDevice())

        """)
        fflush(stdout)

        // Install SIGINT handler so Ctrl-C flushes a final checkpoint
        // instead of dying mid-step.
        TrainSupport.installSigintHandler()
        TrainSupport.stopRequested.reset()

        // Reset MLX's peak-memory counter at the start of training so
        // the post-run report reflects what training actually consumed
        // (and doesn't include loader/init transients). Always-on; the
        // post-run report uses the same value either way.
        MLX.Memory.peakMemory = 0  // setter triggers mlx_reset_peak_memory

        let t0 = Date()
        var lastLoss: Float = 0
        var lastValLoss: Float? = nil

        // Closure-based sampling — works for both byte and BPE corpora.
        // We've dropped the explicit prefetch pipeline because MLXArray
        // construction blocks anyway; the saved overlap was small.
        var stoppedEarly = false
        var lastStep = startStep
        for step in startStep..<steps {
            // LR schedule update — only meaningful when compile is off.
            if useSchedule {
                trainer.optimizer.learningRate = TrainSupport.lrAt(
                    step: step, total: steps, warmup: warmupSteps,
                    maxLR: maxLR, minLR: minLR
                )
            }

            if accumSteps == 1 {
                let (x, y) = sampleTrainBatch(B, cfg.contextLength)
                lastLoss = trainer.step(inputs: x, targets: y)
            } else {
                // Collect N micro-batches before one optimizer update.
                // Effective batch becomes B × accumSteps with the memory
                // cost of just B.
                var micros: [(MLXArray, MLXArray)] = []
                micros.reserveCapacity(accumSteps)
                for _ in 0..<accumSteps {
                    micros.append(sampleTrainBatch(B, cfg.contextLength))
                }
                lastLoss = trainer.accumulatedStep(microBatches: micros)
            }
            lastStep = step + 1

            if step == 0 || (step + 1) % 50 == 0 || step == steps - 1 {
                let elapsed = -t0.timeIntervalSinceNow
                let done = step - startStep + 1
                let stepsPerSec = Double(done) / elapsed
                let eta = Double(steps - step - 1) / max(stepsPerSec, 1e-6)
                let lrTag = useSchedule ?
                    String(format: "  lr=%.2e", trainer.optimizer.learningRate) : ""
                let valTag = lastValLoss.map { String(format: "  val %.3f", $0) } ?? ""
                fputs(String(format: "  step %5d/%5d  loss %.3f%@%@  · %.1f step/s · eta %.0fs\n",
                             step + 1, steps, lastLoss, lrTag, valTag, stepsPerSec, eta), stderr)
            }
            if (step + 1) % sampleEvery == 0 || step == steps - 1 {
                // Inline sample only meaningful for byte-level — BPE prints
                // would need tokenizer decode; use `tinygpt sample` instead.
                if cfg.tokenizerSource == nil {
                    printSample(model: model, cfg: cfg, tag: "step \(step + 1)")
                }
            }
            // Val loss
            if let vsb = valSampleBatch, (step + 1) % valEvery == 0 {
                var total: Float = 0
                let n = 8
                for _ in 0..<n {
                    let (vx, vy) = vsb(B, cfg.contextLength)
                    let loss = model.loss(vx, vy)
                    MLX.eval(loss)
                    total += loss.item(Float.self)
                }
                lastValLoss = total / Float(n)
                fputs(String(format: "    val loss %.3f\n", lastValLoss!), stderr)
            }
            // Atomic checkpoint
            if let n = saveEvery, let out = outPath, (step + 1) % n == 0 {
                do {
                    try TrainSupport.atomicSave(
                        model: model, cfg: cfg, step: step + 1, finalLoss: lastLoss,
                        weightTranspose: isLinearWeightName,
                        manifestEntries: manifestEntries,
                        to: URL(fileURLWithPath: out)
                    )
                    fputs("    ✓ checkpoint at step \(step + 1) → \(out)\n", stderr)
                } catch {
                    fputs("    ⚠ checkpoint save failed: \(error)\n", stderr)
                }
            }
            // Cooperative cancel
            if TrainSupport.stopRequested.isSet {
                stoppedEarly = true
                fputs("\n[SIGINT] flushing final checkpoint at step \(step + 1)…\n", stderr)
                break
            }
        }
        let elapsed = -t0.timeIntervalSinceNow
        let stepsDone = lastStep - startStep
        let stepsPerSec = elapsed > 0 ? Double(stepsDone) / elapsed : 0
        let summary = stoppedEarly
            ? "interrupted at step \(lastStep) of \(steps) after \(String(format: "%.1f", elapsed))s · loss \(String(format: "%.3f", lastLoss))"
            : "done — \(stepsDone) steps in \(String(format: "%.1f", elapsed))s (\(String(format: "%.1f", stepsPerSec)) step/s) · final loss \(String(format: "%.3f", lastLoss))"
        print("\n\(summary)")

        // Peak GPU-memory report. Always-on at end of training since the
        // counter was reset at start anyway; the line is one of the most
        // useful diagnostics for sizing future runs / verifying that
        // --grad-checkpoint actually reduced activation memory.
        let peak = MLX.Memory.peakMemory
        let snap = MLX.Memory.snapshot()
        print(String(format: "memory:  peak=%@  active=%@  cache=%@%@",
                      formatBytes(peak),
                      formatBytes(snap.activeMemory),
                      formatBytes(snap.cacheMemory),
                      cfg.useGradCheckpoint ? "  · grad-checkpoint=on" : ""))

        // Final save (always — covers both completion and Ctrl-C cases).
        if let out = outPath {
            print("saving to \(out)…")
            do {
                try TrainSupport.atomicSave(
                    model: model, cfg: cfg, step: lastStep, finalLoss: lastLoss,
                    weightTranspose: isLinearWeightName,
                    manifestEntries: manifestEntries,
                    to: URL(fileURLWithPath: out)
                )
                print("✓ wrote \(out)")
            } catch {
                fputs("save failed: \(error)\n", stderr); exit(1)
            }
        }
        if stoppedEarly { exit(130) }  // standard "killed by SIGINT" exit code
    }

    private static func printSample(model: TinyGPTModel, cfg: ModelConfig, tag: String) {
        let promptBytes: [UInt8] = [UInt8]("The ".utf8)
        var idx = MLXArray(promptBytes.map { Int32($0) }, [1, promptBytes.count])
        var bytes = promptBytes
        for _ in 0..<60 {
            let T = idx.shape.last!
            let lo = max(0, T - cfg.contextLength)
            let cond = idx[0..., lo..<T]
            let logits = model(cond)
            let last = logits[0..., logits.shape[1] - 1, 0...]
            let next = argMax(last / MLXArray(Float(0.8)), axis: -1).reshaped([1, 1])
            eval(next)
            let id = Int(next.item(Int32.self))
            bytes.append(UInt8(id & 0xff))
            idx = concatenated([idx, next.asType(idx.dtype)], axis: 1)
        }
        let s = String(bytes: bytes, encoding: .utf8) ?? "<non-utf8>"
        let clipped = s.prefix(120).replacingOccurrences(of: "\n", with: "\\n")
        fputs("    [\(tag) sample] \(clipped)\n", stderr)
    }

    /// Param-name manifest order — must match the existing file format so
    /// saves are interoperable with the browser.
    ///
    /// For dense models the layout is fixed (token + position embedding,
    /// ln_final, then per-block ln1/attn/ln2/mlp). For MoE models the
    /// per-block MLP entries are replaced by router + per-expert MLPs;
    /// non-MoE blocks-of-MoE-models don't exist (the choice is uniform
    /// across blocks for now). The browser doesn't load MoE yet, so the
    /// MoE manifest is a Mac-side extension.
    static func manifestEntries(_ cfg: ModelConfig) -> [TinyGPTHeader.TensorEntry] {
        var entries: [TinyGPTHeader.TensorEntry] = []
        var offset = 0
        let push: (String, [Int]) -> Void = { name, shape in
            let size = shape.reduce(1, *)
            entries.append(.init(name: name, shape: shape, floatOffset: offset))
            offset += size
        }
        let C = cfg.dModel, M = cfg.dMlp
        push("token_embedding.weight", [cfg.vocabSize, C])
        push("position_embedding.weight", [cfg.contextLength, C])
        push("ln_final.weight", [C])
        push("ln_final.bias", [C])
        for i in 0..<cfg.nLayers {
            push("blocks.\(i).ln1.weight", [C])
            push("blocks.\(i).ln1.bias", [C])
            push("blocks.\(i).attn.q_proj.weight", [C, C])
            push("blocks.\(i).attn.q_proj.bias", [C])
            push("blocks.\(i).attn.k_proj.weight", [C, C])
            push("blocks.\(i).attn.k_proj.bias", [C])
            push("blocks.\(i).attn.v_proj.weight", [C, C])
            push("blocks.\(i).attn.v_proj.bias", [C])
            push("blocks.\(i).attn.o_proj.weight", [C, C])
            push("blocks.\(i).attn.o_proj.bias", [C])
            push("blocks.\(i).ln2.weight", [C])
            push("blocks.\(i).ln2.bias", [C])
            if cfg.isMoE {
                // Router: bias-free Linear(d_model → n_experts).
                push("blocks.\(i).moe.router.weight", [cfg.nExperts, C])
                // Each expert is an MLP — same fc_in/fc_out structure
                // as the dense path, replicated N times per block.
                for e in 0..<cfg.nExperts {
                    push("blocks.\(i).moe.experts.\(e).fc_in.weight", [M, C])
                    push("blocks.\(i).moe.experts.\(e).fc_in.bias", [M])
                    push("blocks.\(i).moe.experts.\(e).fc_out.weight", [C, M])
                    push("blocks.\(i).moe.experts.\(e).fc_out.bias", [C])
                }
            } else {
                push("blocks.\(i).mlp.fc_in.weight", [M, C])
                push("blocks.\(i).mlp.fc_in.bias", [M])
                push("blocks.\(i).mlp.fc_out.weight", [C, M])
                push("blocks.\(i).mlp.fc_out.bias", [C])
            }
            // MoD: per-block sigmoid gate. Linear(d_model → 1) with bias.
            // Tiny — adds C + 1 params per layer.
            if cfg.useMoD {
                push("blocks.\(i).mod_router.weight", [1, C])
                push("blocks.\(i).mod_router.bias", [1])
            }
            // Differential attention extras (Ye et al., 2024). The
            // standard attn entries above are still emitted — the
            // diff_attn sibling adds its own. Bias presence follows
            // cfg.attnBias just like the standard path.
            if cfg.useDifferentialAttention {
                push("blocks.\(i).diff_attn.q1_proj.weight", [C, C])
                if cfg.attnBias { push("blocks.\(i).diff_attn.q1_proj.bias", [C]) }
                push("blocks.\(i).diff_attn.k1_proj.weight", [C, C])
                if cfg.attnBias { push("blocks.\(i).diff_attn.k1_proj.bias", [C]) }
                push("blocks.\(i).diff_attn.q2_proj.weight", [C, C])
                if cfg.attnBias { push("blocks.\(i).diff_attn.q2_proj.bias", [C]) }
                push("blocks.\(i).diff_attn.k2_proj.weight", [C, C])
                if cfg.attnBias { push("blocks.\(i).diff_attn.k2_proj.bias", [C]) }
                push("blocks.\(i).diff_attn.v_proj.weight",  [C, C])
                if cfg.attnBias { push("blocks.\(i).diff_attn.v_proj.bias",  [C]) }
                push("blocks.\(i).diff_attn.o_proj.weight",  [C, C])
                if cfg.attnBias { push("blocks.\(i).diff_attn.o_proj.bias",  [C]) }
                push("blocks.\(i).diff_attn.lambda", [])
            }
        }
        return entries
    }

    /// Predicate for Linear-weight transpose on save (PyTorch [out,in] → WASM [in,out]).
    static func isLinearWeightName(_ name: String) -> Bool {
        guard name.hasSuffix(".weight") else { return false }
        if name == "token_embedding.weight" || name == "position_embedding.weight" {
            return false
        }
        if name.hasSuffix(".ln1.weight") || name.hasSuffix(".ln2.weight")
            || name == "ln_final.weight" {
            return false
        }
        return true
    }

    private static func configFor(_ preset: String) -> ModelConfig {
        switch preset.lowercased() {
        case "tiny":     return ModelConfig(vocabSize: 256, contextLength: 128, nLayers: 4,
                                             nHeads: 4, dModel: 128, dMlp: 512)
        case "small":    return ModelConfig(vocabSize: 256, contextLength: 256, nLayers: 6,
                                             nHeads: 6, dModel: 192, dMlp: 768)
        case "huge":     return ModelConfig.huge
        case "mega":     return ModelConfig.mega
        case "behemoth": return ModelConfig.behemoth
        case "titan":    return ModelConfig.titan
        default:
            fputs("unknown preset: \(preset). Choose tiny|small|huge|mega|behemoth|titan.\n", stderr)
            exit(2)
        }
    }

    private static func defaultBatch(_ cfg: ModelConfig) -> Int {
        if cfg.dModel >= 1024 { return 2 }
        if cfg.dModel >= 512 { return 4 }
        if cfg.dModel >= 256 { return 8 }
        return 16
    }

    private static func formatLargeInt(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func formatBytes(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1f MB", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0f KB", Double(n) / 1_000) }
        return "\(n) B"
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt train [options]

        Core:
          --preset tiny|small|huge|mega|behemoth|titan   (default: tiny)
          --steps N                       Training steps (default: 500)
          --corpus path.txt               UTF-8 text file (default: random bytes)
          --out path.tinygpt              Where to save the trained checkpoint
          --dtype float32|float16         Training dtype (default: float32)
          --batch N                       Batch size (default: by preset)
          --sample-every N                Print a sample every N steps (default: 100)
          --tokenizer <hf-dir>            Use BPE/SentencePiece from a HF model dir
                                           (vocab size comes from config.json)
          --ctx N                         Override preset's context length
          --accum N                       Gradient accumulation: N micro-batches per
                                           optimizer step (effective batch = batch × N).
                                           Disables compile.
          --grad-clip F                   Global L2 norm cap for gradients (default 1.0
                                           — standard for transformer LM training).
                                           Pass 0 to disable.
          --moe-experts N                 Mixture-of-Experts mode: N experts per block
                                           (default 1 = dense MLP). 8 is Mixtral-class.
          --moe-topk K                    Experts activated per token (default 1; 2 is
                                           Mixtral-style). Capped at --moe-experts.
          --moe-aux-weight F              Load-balance loss scale (default 0.01).
          --mtp-horizons N                Multi-Token Prediction: predict tokens t+1..t+N
                                           at every position via extra heads. Default 1.
                                           2-4 typical. Heads are training-only — saved
                                           checkpoints stay drop-in compatible.
          --sliding-window N              Restrict attention to the last N positions
                                           (Mistral / GPT-OSS recipe). Default: off
                                           (full causal). Cuts O(T²) attn to O(T·N).
          --alibi                         Use ALiBi position bias (Press et al., 2021)
                                           in lieu of positional embeddings/RoPE. Better
                                           extrapolation beyond train context length.
          --optimizer K                   AdamW (default) | lion | sophia | muon | adafactor.
                                           See docs/optimizers.md for memory + tradeoffs.
                                           Drop-in: same --max-lr / --weight-decay etc.
          --grad-checkpoint               Activation (gradient) checkpointing. Wraps each
                                           TransformerBlock's forward in a CustomFunction
                                           whose VJP recomputes the block forward at
                                           backward time. ~30% step-time overhead in
                                           exchange for dramatically lower activation
                                           memory — unlocks bigger models / batches at
                                           the cost of speed.

        Long-run safety nets:
          --resume <path.tinygpt>         Continue from a saved checkpoint
                                           (Adam state restarts — 100-step warmup)
          --save-every N                  Atomic checkpoint every N steps
          --lr-schedule constant|cosine   (default: constant)
          --warmup N                      Warmup steps (default: 0)
          --max-lr / --min-lr             Cosine endpoints (defaults: 3e-4 / 3e-5)
          --val-split 0.0-0.2             Hold out last fraction of corpus for val
          --val-every N                   Eval val loss every N steps (default: 200)

        Ctrl-C flushes a final checkpoint then exits cleanly.
        """)
        exit(2)
    }
}
