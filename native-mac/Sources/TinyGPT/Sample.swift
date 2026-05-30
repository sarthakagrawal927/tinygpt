import Foundation
import MLX
import MLXNN
import MLXRandom
import TinyGPTIO
import TinyGPTModel

/// `tinygpt sample` — load a browser-trained `.tinygpt` file and generate
/// text. The cross-path interop demo: the model trained in the browser,
/// run here on Metal at native speeds.
enum Sample {
    static func run(args: [String]) {
        var path: String?
        var prompt = "ROMEO:"
        var maxTokens = 200
        var temperature: Float = 0.8
        var useKVCache = true
        var loraPaths: [String] = []
        var loraWeights: [Float] = []
        var quantizeBits: Int? = nil
        var quantizeGroup: Int = 64
        var draftPath: String? = nil
        var speculativeK: Int = 4
        var kvQuantize: String? = nil      // "fp16" | "bf16" → downcast on store
        var prefixCachePath: String? = nil // path to load/save prompt KV cache
        var streamingSink: Int? = nil
        var streamingWindow: Int? = nil
        // Persistent cache: hash-keyed prompt cache. When --prompt-cache-dir is
        // set we auto-save / auto-load by hash(modelName + prompt + cfg). The
        // explicit --cache-prompt path still takes precedence (back-compat).
        var promptCacheDir: String? = nil
        // Pre-allocate the cache buffer at max context so per-step decodes
        // skip the per-step concat allocation. Off by default (matches the
        // historic concat behaviour); enable with --kv-preallocate or when
        // --prompt-cache-dir is set (a fresh-load cache always benefits).
        var preAllocate: Bool = false
        // Speculative-decode HEADS (Medusa / EAGLE-2). When --heads is set,
        // we route generation through the joint-head verification path
        // instead of the standard per-token decode loop. See
        // `MedusaHeads.swift` / `EagleDraft.swift` for the head modules
        // and `TrainHeads.swift` for how a `.heads` sidecar is produced.
        var headsPath: String? = nil
        var headType: String = "medusa"
        // Cold-start optimisations. `--lazy-embedding` defers the embedding
        // tensor (vocab × d_model) until just before the first forward —
        // shaves a meaningful chunk off load time + RAM on large models.
        // `--no-async-load` disables the background-thread load (mostly a
        // debug hatch; the spinner output is identical either way).
        var lazyEmbedding = false
        var asyncLoad = true
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--prompt":
                guard i + 1 < args.count else { exitUsage() }
                prompt = args[i + 1]; i += 2
            case "--tokens":
                guard i + 1 < args.count else { exitUsage() }
                maxTokens = Int(args[i + 1]) ?? maxTokens; i += 2
            case "--temperature", "--temp":
                guard i + 1 < args.count else { exitUsage() }
                temperature = Float(args[i + 1]) ?? temperature; i += 2
            case "--no-cache":
                useKVCache = false; i += 1
            case "--cache":
                useKVCache = true; i += 1
            case "--lora":
                guard i + 1 < args.count else { exitUsage() }
                loraPaths.append(args[i + 1]); i += 2
            case "--lora-weight":
                // Per-adapter mix weight when composing multiple LoRAs.
                // Supply once per --lora, same order. Defaults to 1.0 each.
                guard i + 1 < args.count else { exitUsage() }
                loraWeights.append(Float(args[i + 1]) ?? 1.0); i += 2
            case "--quantize":
                guard i + 1 < args.count else { exitUsage() }
                switch args[i + 1].lowercased() {
                case "int4", "4bit", "4":  quantizeBits = 4
                case "int8", "8bit", "8":  quantizeBits = 8
                default: fputs("--quantize must be int4 or int8\n", stderr); exit(2)
                }
                i += 2
            case "--quantize-group":
                guard i + 1 < args.count else { exitUsage() }
                quantizeGroup = Int(args[i + 1]) ?? quantizeGroup; i += 2
            case "--draft":
                guard i + 1 < args.count else { exitUsage() }
                draftPath = args[i + 1]; i += 2
            case "--speculative-k":
                guard i + 1 < args.count else { exitUsage() }
                speculativeK = max(1, Int(args[i + 1]) ?? speculativeK); i += 2
            case "--kv-quantize":
                guard i + 1 < args.count else { exitUsage() }
                kvQuantize = args[i + 1].lowercased(); i += 2
            case "--cache-prompt":
                guard i + 1 < args.count else { exitUsage() }
                prefixCachePath = args[i + 1]; i += 2
            case "--prompt-cache-dir":
                guard i + 1 < args.count else { exitUsage() }
                promptCacheDir = args[i + 1]; i += 2
            case "--kv-preallocate":
                preAllocate = true; i += 1
            case "--no-kv-preallocate":
                preAllocate = false; i += 1
            case "--streaming-llm-sink":
                guard i + 1 < args.count else { exitUsage() }
                streamingSink = Int(args[i + 1]); i += 2
            case "--streaming-llm-window":
                guard i + 1 < args.count else { exitUsage() }
                streamingWindow = Int(args[i + 1]); i += 2
            case "--heads":
                guard i + 1 < args.count else { exitUsage() }
                headsPath = args[i + 1]; i += 2
            case "--head-type":
                guard i + 1 < args.count else { exitUsage() }
                headType = args[i + 1].lowercased(); i += 2
            case "--lazy-embedding":
                lazyEmbedding = true; i += 1
            case "--no-async-load":
                asyncLoad = false; i += 1
            case "-h", "--help":
                exitUsage()
            default:
                if args[i].hasPrefix("-") {
                    fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
                }
                path = args[i]; i += 1
            }
        }
        guard let path = path else {
            fputs("sample: missing <path> to .tinygpt file\n", stderr)
            exitUsage()
        }
        let url = URL(fileURLWithPath: path)

        // Unified loader — accepts .tinygpt files or HF model dirs.
        //
        // Cold-start path:
        //   - mmap'd reader (default since the cold-start bundle landed)
        //     means the 250 MB file read is a VM map, not a literal read.
        //   - When `asyncLoad` is on (default), the load runs on a
        //     background thread and the foreground prints a spinner.
        //   - `--lazy-embedding` defers the (usually largest) token
        //     embedding tensor until the first forward.
        let load: ModelLoader.LoadResult
        let tLoad = Date()
        do {
            if asyncLoad {
                load = try ColdStart.loadWithSpinner(
                    path: path,
                    deferEmbedding: lazyEmbedding,
                    label: url.lastPathComponent
                )
            } else {
                print("loading \(url.lastPathComponent)…")
                if lazyEmbedding {
                    load = try ModelLoader.loadLazyEmbedding(path)
                } else {
                    load = try ModelLoader.load(path)
                }
            }
        } catch { fputs("error loading: \(error)\n", stderr); exit(1) }
        let cfg = load.config
        let model = load.model
        let loadElapsed = -tLoad.timeIntervalSinceNow
        if let h = load.lazyEmbedding {
            print(String(format: "loaded in %.2fs (lazy embedding: %@ pending)",
                          loadElapsed, formatBytes(h.totalBytes)))
        } else {
            print(String(format: "loaded in %.2fs", loadElapsed))
        }

        // Apply one OR MORE LoRA adapters on top. Adapters carry their
        // base architecture in the header so a from-scratch adapter
        // can't accidentally load on an HF base, and vice versa.
        if !loraPaths.isEmpty {
            do {
                let adapters = try loraPaths.map { try LoraAdapterReader.read(URL(fileURLWithPath: $0)) }
                while loraWeights.count < adapters.count { loraWeights.append(1.0) }
                if adapters.count == 1 {
                    try model.applyLora(adapters[0])
                    print("loaded LoRA: rank=\(adapters[0].header.rank) targets=\(adapters[0].header.targetSuffixes.joined(separator: ","))")
                } else {
                    // Stacked composition — both from-scratch and HF paths.
                    let blend = zip(loraPaths, loraWeights).map {
                        "\($0.0.split(separator: "/").last ?? "") @ \($0.1)"
                    }.joined(separator: " + ")
                    switch model {
                    case .fromScratch(let m):
                        try LoraStackInjection.apply(adapters, weights: loraWeights, to: m)
                    case .huggingFace(let m):
                        try LoraStackInjectionHF.apply(adapters, weights: loraWeights, to: m)
                    }
                    print("composed \(adapters.count) LoRAs: \(blend)")
                }
            } catch {
                fputs("error loading LoRA adapter(s): \(error)\n", stderr)
                exit(1)
            }
        }
        // 4-bit / 8-bit quantization. MLX's `quantize` walks the model's
        // leaf modules and replaces every Linear + Embedding with its
        // QuantizedLinear / QuantizedEmbedding equivalent. ~8× memory
        // savings vs fp32 at int4, lossless-feeling sampling for any
        // model trained at fp16 or above. Skipped when LoRA is loaded
        // because quantizing a LoraLinear (a Linear subclass) loses the
        // LoRA delta — apply quantization to base only, before adapters.
        if let bits = quantizeBits {
            if !loraPaths.isEmpty {
                fputs("warning: --quantize ignored when --lora is in play (would discard the adapter delta). Skipping.\n", stderr)
            } else {
                MLXNN.quantize(model: model.module, groupSize: quantizeGroup, bits: bits)
                print("quantized to int\(bits) (group=\(quantizeGroup)) — ~\(32 / bits)× memory savings vs fp32")
            }
        }
        print("ready — \(formatLargeInt(model.numParameters())) params on \(Device.defaultDevice())")
        print("")

        // Load the tokenizer if the model pins one (either an HF model dir
        // or a from-scratch model trained with `--tokenizer`). Otherwise
        // fall back to byte-level encode/decode (vocab=256).
        let tokenizer: HFTokenizer?
        if let tokDir = load.hfTokenizerDir {
            do {
                tokenizer = try HFTokenizer.loadBlocking(from: tokDir)
                print("tokenizer: BPE from \(tokDir.lastPathComponent) (vocab=\(cfg.vocabSize))")
            } catch {
                fputs("warning: tokenizer load failed (\(error)); falling back to byte-level\n", stderr)
                tokenizer = nil
            }
        } else {
            tokenizer = nil
        }

        // Encode the prompt — either through BPE or as raw bytes.
        let promptIds: MLXArray
        if let tok = tokenizer {
            let ids: [Int]
            do { ids = try tok.encode(prompt) }
            catch { fputs("prompt encode failed: \(error)\n", stderr); exit(1) }
            promptIds = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        } else {
            let promptBytes = [UInt8](prompt.utf8)
            promptIds = MLXArray(promptBytes.map { Int32($0) }, [1, promptBytes.count])
        }

        // Print the prompt first, then stream generated tokens.
        print(prompt, terminator: "")
        fflush(stdout)

        // Materialise any deferred embedding tensor just before the first
        // forward. The `LazyEmbeddingHandle` is idempotent — calling
        // materialize() twice is a no-op — so it's safe even on paths
        // (--heads, --draft) that take alternate routes through the model.
        if let h = load.lazyEmbedding {
            let tEmbed = Date()
            do { try h.materialize() }
            catch {
                fputs("\nembedding materialisation failed: \(error)\n", stderr)
                exit(1)
            }
            let dt = -tEmbed.timeIntervalSinceNow
            fputs(String(format: "[lazy] materialised embedding in %.2fs\n", dt), stderr)
        }

        // Cooperative cancel — Ctrl-C stops generation mid-stream cleanly.
        TrainSupport.installSigintHandler()
        TrainSupport.stopRequested.reset()

        // Optional speculative-decode draft model. Same tokenizer required;
        // we don't validate beyond the architecture sanity ModelLoader does.
        // Spec-decode bypasses the KV cache (the verify pass re-processes
        // the whole tail — KV-cached spec-decode is a follow-up).
        let draftModel: AnyModel? = draftPath.flatMap { p in
            do {
                let d = try ModelLoader.load(p)
                print("draft model loaded: \(formatLargeInt(d.model.numParameters())) params · greedy speculative-k=\(speculativeK)")
                return d.model
            } catch {
                fputs("warning: draft model load failed (\(error)); proceeding without spec-decode\n", stderr)
                return nil
            }
        }

        let t0 = Date()
        // KV cache now works for both from-scratch and HF models — both
        // routes go through `AnyModel.forwardCached` which dispatches to
        // the right concrete forward. HF path applies RoPE with the
        // correct absolute-position offset and respects GQA K/V head
        // counts (see KVCacheHF.swift).
        // Spec-decode disables KV caching for now (mismatched forward shape).
        // Heads decode also bypasses the cache — verify pass re-processes
        // the whole prefix + N proposed tokens in a single forward.
        let useActualCache = useKVCache && draftModel == nil && headsPath == nil
        // Map CLI flag to either MLX DType (fp16/bf16) for cheap downcast
        // storage, or to KIVI config (int8/int4) for the affine-quantised
        // per-channel-K / per-token-V path. Mutually exclusive — KIVI
        // takes over the storage byte layout.
        let kvDType: DType? = {
            switch kvQuantize {
            case "fp16", "float16", "half": return .float16
            case "bf16", "bfloat16":         return .bfloat16
            default:                          return nil
            }
        }()
        let kvKIVI: KVCache.KIVIConfig? = {
            switch kvQuantize {
            case "int8", "8bit", "8":  return .init(bits: 8)
            case "int4", "4bit", "4":  return .init(bits: 4)
            default:                    return nil
            }
        }()
        if kvKIVI != nil && kvDType != nil {
            fputs("--kv-quantize: cannot mix dtype downcast and KIVI int-quantisation\n", stderr)
            exit(2)
        }
        // Build OR load the cache. Three modes, in priority order:
        //
        //   1. --cache-prompt <path> — explicit user-managed path. Wins over
        //      everything else. Same prompt requires same path each time.
        //
        //   2. --prompt-cache-dir <dir> — hash-keyed auto-cache. Filename
        //      derives from SHA(modelName + prompt + cfg), so changing any
        //      of those forces a fresh prefill on next launch. The expected
        //      use is a long-lived agent system prompt: first launch writes,
        //      every subsequent launch loads.
        //
        //   3. No flags — build a fresh cache, never persist.
        //
        // Pre-allocation is independent: when `preAllocate` is true (or
        // when we're in promptCacheDir mode, because the persistent path
        // is going to keep generating after a load) we allocate a full
        // [B, H, ctx, D] buffer per layer up-front and write into it via
        // slice assignment instead of growing with concat. KIVI and
        // StreamingLLM disable pre-allocation in KVCache.init — eviction
        // requires fragmenting the buffer, which is a future patch.
        var cache: KVCache? = nil
        var skipPrefill = false
        // Resolve the effective load/save path. Explicit --cache-prompt
        // wins; otherwise derive from the dir + hash(model + prompt + cfg).
        // Build the hash key once so both load and save sides agree.
        let kvTag: KVCachePersist.KVTag = {
            if let cfg = kvKIVI { return cfg.bits == 4 ? .kiviInt4 : .kiviInt8 }
            switch kvDType {
            case .some(.float16): return .fp16
            case .some(.bfloat16): return .bf16
            default: return .fp32
            }
        }()
        let cacheKey = KVCachePersist.Key(
            modelName: cfg.modelName,
            // (size, mtime) of the model file scopes the cache to the
            // specific checkpoint, not just the architecture preset. Two
            // chat and code models both built from the same preset have
            // the same modelName but different fingerprints → different
            // cache files → no cross-pollution.
            modelFileFingerprint: KVCachePersist.fingerprint(of: path),
            prompt: prompt,
            vocabSize: cfg.vocabSize, nLayers: cfg.nLayers,
            kvTag: kvTag, useYOCO: cfg.useYOCO
        )
        var resolvedCachePath: URL? = prefixCachePath.map { URL(fileURLWithPath: $0) }
        var resolvedMetaPath: URL? = nil
        if resolvedCachePath == nil, let dir = promptCacheDir {
            let dirURL = URL(fileURLWithPath: dir)
            do {
                try KVCachePersist.ensureDir(dirURL)
                let (cacheURL, metaURL) = KVCachePersist.paths(for: cacheKey, in: dirURL)
                resolvedCachePath = cacheURL
                resolvedMetaPath = metaURL
            } catch {
                fputs("warning: could not create prompt cache dir \(dir): \(error). Skipping persistence.\n", stderr)
            }
        }
        if useActualCache {
            if let url = resolvedCachePath,
               FileManager.default.fileExists(atPath: url.path)
            {
                do {
                    cache = try KVCache.load(from: url, nLayers: cfg.nLayers)
                    skipPrefill = true
                    let source = (prefixCachePath != nil) ? "prefix cache" : "auto cache"
                    print("loaded \(source) (\(cache!.currentLength) tokens) — skipping prompt prefill")
                } catch {
                    fputs("warning: prefix cache load failed (\(error)); building fresh\n", stderr)
                }
            }
            if cache == nil {
                // `preAllocCapacity = cfg.contextLength` activates the
                // in-place storage path: one buffer per layer of size
                // [B, H, ctx, D], all subsequent appends write via slice
                // assignment. Skipped automatically by KVCache.init when
                // KIVI or StreamingLLM is on.
                let cap = preAllocate ? cfg.contextLength : nil
                cache = KVCache(nLayers: cfg.nLayers, kvDtype: kvDType,
                                 kivi: kvKIVI,
                                 sink: streamingSink, window: streamingWindow,
                                 preAllocCapacity: cap)
            } else if preAllocate, let c = cache {
                // Disk-loaded cache + pre-alloc requested: promote each
                // layer's buffer to capacity-sized so the post-load decode
                // doesn't drop back to concat mode.
                c.migrateToPreAlloc(capacity: cfg.contextLength)
            }
            if kvDType != nil {
                print("KV cache stored at \(kvQuantize!) (≈½ memory vs fp32)")
            }
            if let cfg = kvKIVI {
                // int8 storage is the literal byte size; for int4 the
                // precision is int4 (16 levels) but storage stays at int8
                // — we report both, and the doc explains the tradeoff.
                let savingsTag = cfg.bits == 4
                    ? "(~½ vs fp32 storage, int4 precision)"
                    : "(~¼ vs fp32 storage)"
                print("KIVI: per-channel K + per-token V, \(cfg.bits)-bit \(savingsTag)")
            }
            if streamingSink != nil || streamingWindow != nil {
                print("StreamingLLM: sink=\(streamingSink ?? 0) window=\(streamingWindow ?? 0) (drop middle on overflow)")
            }
            if let c = cache, c.preAllocCapacity != nil {
                // stderr so it doesn't interleave with the streaming text.
                fputs("KV cache: pre-allocated buffer (\(cfg.contextLength) tokens capacity, in-place writes)\n", stderr)
            }
            if let url = resolvedCachePath, !skipPrefill, prefixCachePath == nil {
                // Auto-cache MISS on a --prompt-cache-dir run. Print so the
                // user can see the first call is paying the prefill cost
                // and the second will skip it. Pre-key reveal helps debug
                // collisions if they ever happen.
                fputs("auto cache miss → will write \(url.lastPathComponent) after prefill\n", stderr)
            }
        }

        // Incremental BPE decode: re-decode the whole accumulated tail
        // each step (multi-byte tokens like " word" only render correctly
        // when neighbours are known) and print the diff vs the previous
        // render. Byte-level path just emits each byte as a scalar.
        var generated: [Int] = []
        func emit(_ id: Int) {
            if let tok = tokenizer {
                generated.append(id)
                let renderedSoFar = tok.decode(generated)
                let prior = tok.decode(Array(generated.dropLast()))
                let piece = String(renderedSoFar.dropFirst(prior.count))
                print(piece, terminator: "")
            } else {
                if let scalar = UnicodeScalar(id) {
                    print(String(scalar), terminator: "")
                }
            }
            fflush(stdout)
        }
        if let headsPath = headsPath {
            // Speculative decoding with JOINT-TRAINED HEADS (Medusa / EAGLE-2).
            // The base model proposes a hidden state; the heads turn that
            // into N additional speculative tokens; the base verifies all
            // N+1 candidates in one forward pass. Lossless wrt the base's
            // own argmax (greedy verify rule); speedup proportional to
            // acceptance rate, which is a function of head quality.
            //
            // The heads path requires direct access to the base's hidden
            // state + LM head, which our HF wrapper doesn't yet expose —
            // restrict to from-scratch models in this first cut. The
            // architecture itself is base-agnostic; extending requires
            // adding `forwardToHidden` / `applyLMHead` to the HF model.
            guard case .fromScratch(let baseModel) = model else {
                fputs("--heads currently only works with from-scratch byte-level models.\n", stderr)
                exit(2)
            }
            if temperature > 0 {
                fputs("note: --heads forces greedy (temperature ignored)\n", stderr)
            }
            // Lift the model's hidden + LM head into closures we can pass
            // into the verify path. Each is a single forward through the
            // already-built model; nothing fancy.
            let baseHidden: (MLXArray) -> MLXArray = { x in baseModel.forwardToHidden(x) }
            let baseLogits: (MLXArray) -> MLXArray = { x in baseModel(x) }
            let baseLMHead: (MLXArray) -> MLXArray = { h in
                if let lmHead = baseModel.lmHead { return lmHead(h) }
                // Tied embeddings — re-use token embedding's transpose.
                return baseModel.tokenEmbedding.asLinear(h)
            }

            // Load + dispatch on head type. Both kinds use the shared
            // SpecHeadsStepResult shape so the metric reporting code below
            // is uniform.
            let kindLower = headType.lowercased()
            var totalAccepted = 0
            var totalProposed = 0
            var stepCount = 0
            var ids: [Int]
            do {
                let arr = promptIds[0, 0...]
                eval(arr)
                ids = arr.asArray(Int32.self).map { Int($0) }
            }
            let startCount = ids.count
            let tHeadsStart = Date()

            switch kindLower {
            case "medusa":
                let stack: MedusaHeadStack
                do { stack = try MedusaHeadsIO.read(URL(fileURLWithPath: headsPath), baseConfig: cfg) }
                catch { fputs("heads load failed: \(error)\n", stderr); exit(1) }
                print("Medusa heads loaded: \(stack.cfg.numHeads) heads · \(formatLargeInt(stack.numParameters())) params")
                while ids.count - startCount < maxTokens {
                    if TrainSupport.stopRequested.isSet { break }
                    let r = MedusaVerify.step(
                        baseHidden: baseHidden, baseLogits: baseLogits,
                        baseLMHead: baseLMHead, heads: stack,
                        ids: &ids, ctxCap: cfg.contextLength
                    )
                    for id in r.acceptedIds { emit(id) }
                    totalAccepted += r.proposalsAccepted
                    totalProposed += r.proposalsTotal
                    stepCount += 1
                    if ids.count >= cfg.contextLength { break }
                }
            case "eagle":
                let draftNet: EagleDraft
                do { draftNet = try EagleDraftIO.read(URL(fileURLWithPath: headsPath), baseConfig: cfg) }
                catch { fputs("heads load failed: \(error)\n", stderr); exit(1) }
                print("EAGLE-2 draft loaded: \(draftNet.numHeads) unroll steps · \(formatLargeInt(draftNet.numParameters())) params")
                while ids.count - startCount < maxTokens {
                    if TrainSupport.stopRequested.isSet { break }
                    let r = EagleVerify.step(
                        baseHidden: baseHidden, baseLogits: baseLogits,
                        baseLMHead: baseLMHead, draft: draftNet,
                        ids: &ids, ctxCap: cfg.contextLength
                    )
                    for id in r.acceptedIds { emit(id) }
                    totalAccepted += r.proposalsAccepted
                    totalProposed += r.proposalsTotal
                    stepCount += 1
                    if ids.count >= cfg.contextLength { break }
                }
            default:
                fputs("--head-type must be 'medusa' or 'eagle', got '\(headType)'\n", stderr)
                exit(2)
            }
            let elapsedHeads = -tHeadsStart.timeIntervalSinceNow
            let acceptRate = totalProposed > 0
                ? Double(totalAccepted) / Double(totalProposed) : 0
            let producedTokens = ids.count - startCount
            let tps = elapsedHeads > 0 ? Double(producedTokens) / elapsedHeads : 0
            fputs(String(format: "\n[heads] steps=%d, proposed=%d, accepted=%d (%.1f%%) · %.0f tok/s · %.2fs\n",
                          stepCount, totalProposed, totalAccepted,
                          acceptRate * 100, tps, elapsedHeads), stderr)
        } else if let draft = draftModel {
            // Speculative decoding (greedy). Generate in bursts of up-to-K
            // tokens at a time. Temperature is ignored — the greedy
            // variant is lossless wrt target's argmax; sampled spec-decode
            // is a follow-up.
            if temperature > 0 {
                fputs("note: --draft forces greedy (temperature ignored)\n", stderr)
            }
            var ids: [Int] = []
            // Initialise ids from the prompt.
            do {
                let arr = promptIds[0, 0...]
                eval(arr)
                let promptInts = arr.asArray(Int32.self).map { Int($0) }
                ids = promptInts
            }
            let startCount = ids.count
            while ids.count - startCount < maxTokens {
                if TrainSupport.stopRequested.isSet { break }
                let remaining = maxTokens - (ids.count - startCount)
                let k = min(speculativeK, remaining)
                let priorCount = ids.count
                let accepted = SpeculativeDecode.step(target: model, draft: draft,
                                                       ids: &ids, k: k,
                                                       ctxCap: cfg.contextLength)
                _ = priorCount
                for id in accepted { emit(id) }
                if ids.count >= cfg.contextLength { break }
            }
        } else if useActualCache, let cache {
            // Prefill the cache from the prompt, UNLESS we loaded a saved
            // prefix cache for this prompt (then we already have the KV
            // state and can jump straight into the per-token decode loop;
            // we still need one forward to get the first logits, but on
            // a one-token input rather than the full prompt).
            var lastLogits: MLXArray
            if skipPrefill {
                // Push the LAST prompt token through forwardCached so the
                // logits we use to predict the next token reflect the saved
                // KV state. We pop the cache's last entry to avoid double-
                // counting that position, then re-add it via forwardCached.
                let lastTok = promptIds[0..., promptIds.shape[1] - 1 ..< promptIds.shape[1]]
                // Rewind cache by one token so we re-feed the prompt's tail.
                // `KVCache.rewind` handles both concat and in-place layouts
                // (in-place just shrinks `validLengths`; the buffer rows
                // are still there but no longer attended to).
                cache.rewind(by: 1)
                let logits = model.forwardCached(lastTok, cache: cache)
                lastLogits = logits[0..., logits.shape[1] - 1, 0...]
            } else {
                let prefillLogits = model.forwardCached(promptIds, cache: cache)
                lastLogits = prefillLogits[0..., prefillLogits.shape[1] - 1, 0...]
                // Save the populated cache if the user requested it AND we
                // built fresh — first cold call pays the prefill cost; later
                // ones for the same prompt skip it. Same path covers both
                // --cache-prompt and the hash-derived auto-cache.
                if let url = resolvedCachePath {
                    do {
                        try cache.saveToDisk(to: url)
                        let (totalBytes, _) = cache.totalBytes(byteWidth: dtypeByteWidth)
                        fputs("saved prefix cache → \(url.path)\n", stderr)
                        if let meta = resolvedMetaPath {
                            KVCachePersist.writeMeta(
                                cacheKey, to: meta,
                                tokens: cache.currentLength, bytes: totalBytes)
                        }
                    } catch {
                        fputs("warning: prefix cache save failed: \(error)\n", stderr)
                    }
                }
            }
            for _ in 0..<maxTokens {
                if TrainSupport.stopRequested.isSet { break }
                let nextId: MLXArray
                if temperature <= 0 {
                    nextId = argMax(lastLogits, axis: -1).reshaped([1, 1])
                } else {
                    let scaled = lastLogits / MLXArray(temperature)
                    nextId = MLXRandomCategorical(scaled).reshaped([1, 1])
                }
                eval(nextId)
                emit(Int(nextId.item(Int32.self)))
                if cache.currentLength >= cfg.contextLength { break }
                let logits = model.forwardCached(nextId.asType(promptIds.dtype), cache: cache)
                lastLogits = logits[0..., 0, 0...]
            }
        } else {
            // Uncached forward — works on either model variant via AnyModel.
            var idx = promptIds
            for _ in 0..<maxTokens {
                if TrainSupport.stopRequested.isSet { break }
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
                    nextId = MLXRandomCategorical(scaled).reshaped([1, 1])
                }
                eval(nextId)
                emit(Int(nextId.item(Int32.self)))
                idx = concatenated([idx, nextId.asType(idx.dtype)], axis: 1)
            }
        }
        let elapsed = -t0.timeIntervalSinceNow
        let tokensPerSec = Double(maxTokens) / elapsed
        print("\n")
        let cacheLabel = useActualCache ? "KV-cached" : "uncached"
        print("(\(maxTokens) tokens in \(String(format: "%.2f", elapsed))s — \(String(format: "%.0f", tokensPerSec)) tok/s · \(cacheLabel))")

        // KV cache size report — useful for verifying YOCO's halving
        // claim and KV-quantize savings. Counts only populated layers
        // (under YOCO the second-half layers leave their slot empty).
        // Routes through `KVCache.totalBytes` so KIVI's quantised storage
        // (int8 K + V plus per-channel/per-token scales) is counted
        // correctly, not via the 0-shape residual.
        if useActualCache, let c = cache {
            let (totalBytes, populatedLayers) = c.totalBytes(byteWidth: dtypeByteWidth)
            let yocoTag = cfg.useYOCO ? "  · YOCO (\(populatedLayers)/\(cfg.nLayers) layers populated)" : ""
            // Stored-token count: under StreamingLLM the cache stays
            // bounded at `sink + window`; `currentLength` is the
            // monotonic generation counter and may exceed it. Probe the
            // first non-empty layer for the actual stored count.
            let storedTokens = c.entries.first(where: { ($0.keysQ?.shape[2] ?? $0.keys.shape[2]) > 0 })
                .map { $0.keysQ?.shape[2] ?? $0.keys.shape[2] } ?? 0
            let streamTag = (streamingSink != nil || streamingWindow != nil)
                && storedTokens != c.currentLength
                ? "  · stored \(storedTokens) (StreamingLLM cap)" : ""
            // Pre-allocated buffer: report the PHYSICAL bytes too so the
            // user can see the upper bound. Logical is what the model
            // attends to; physical is what's pinned in memory.
            var preAllocTag = ""
            if c.preAllocCapacity != nil {
                let phys = c.physicalBytes(byteWidth: dtypeByteWidth)
                preAllocTag = "  · physical \(formatBytes(phys)) (pre-alloc)"
            }
            print(String(format: "KV cache:  %d tokens · %@%@%@%@",
                          c.currentLength, formatBytes(totalBytes), yocoTag, streamTag, preAllocTag))
        }
    }

    private static func dtypeByteWidth(_ dt: DType) -> Int {
        switch dt {
        case .float16, .bfloat16: return 2
        case .float32: return 4
        case .int8, .uint8: return 1
        default: return 4
        }
    }

    private static func formatBytes(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1f MB", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.1f KB", Double(n) / 1_000) }
        return "\(n) B"
    }

    private static func MLXRandomCategorical(_ logits: MLXArray) -> MLXArray {
        // Sample one id per leading row from the unnormalized logits.
        return MLXRandom.categorical(logits)
    }

    private static func formatLargeInt(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt sample <path.tinygpt> [options]

        --prompt "..."        Starting text (default: "ROMEO:")
        --tokens N            Max new tokens (default: 200)
        --temperature F       Sampling temperature (default: 0.8; 0 = greedy)
        --lora <path.lora>    Apply a LoRA adapter (repeatable to compose)
        --lora-weight F       Per-adapter mix weight when composing (default 1.0)
        --quantize int4|int8  Apply MLX `quantize` to base before sampling
        --quantize-group N    Quantize group size (default 64)
        --draft <path>        Greedy speculative decoding with this draft model
        --speculative-k N     Tokens per speculative burst (default 4)
        --no-cache            Disable the KV cache (one forward per token)
        --kv-quantize fp16|bf16|int8|int4
                              fp16/bf16: half-precision dtype cast (≈½ memory).
                              int8/int4: KIVI quantisation (per-channel K,
                              per-token V) — int8 ≈ ¼ memory, int4 ≈ ⅛
                              precision-wise (storage stays at int8 due to
                              MLX's lack of nibble-packing).
        --cache-prompt <path> Save prompt KV cache to <path> on first run;
                              load it on subsequent runs (skip prompt prefill)
        --prompt-cache-dir <dir>
                              Auto-cache the prompt KV by SHA(modelName + prompt
                              + config). First launch writes to <dir>; second
                              launch with the same prompt loads and skips
                              prefill (10×-100× TTFT speedup on long prompts).
        --kv-preallocate      Pre-allocate the KV buffer at max context. Decode
                              writes via slice assignment instead of growing
                              the cache with concat — peak memory stays flat
                              across long generations. (Disabled by KIVI /
                              StreamingLLM since those need eviction.)
        --streaming-llm-sink N
                              Always keep the first N tokens (StreamingLLM anchor)
        --streaming-llm-window M
                              Keep only the last M tokens beyond the sink
        --heads <path>        Speculative decoding with joint-trained heads
                              (`.heads` sidecar from `tinygpt train-heads`).
                              Greedy verify — temperature is ignored.
        --head-type {medusa|eagle}
                              Head architecture used by --heads (default medusa).
                              Must match the sidecar's stored kind.
        --lazy-embedding      Defer loading the token embedding tensor
                              until the first forward (lower cold-start
                              RAM; slightly higher first-token latency)
        --no-async-load       Disable background-thread load (load
                              blocks the main thread, spinner suppressed)
        """)
        exit(2)
    }
}
