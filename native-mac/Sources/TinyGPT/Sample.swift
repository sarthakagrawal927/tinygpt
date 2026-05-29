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
            case "--streaming-llm-sink":
                guard i + 1 < args.count else { exitUsage() }
                streamingSink = Int(args[i + 1]); i += 2
            case "--streaming-llm-window":
                guard i + 1 < args.count else { exitUsage() }
                streamingWindow = Int(args[i + 1]); i += 2
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
        print("loading \(url.lastPathComponent)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(path) }
        catch { fputs("error loading: \(error)\n", stderr); exit(1) }
        let cfg = load.config
        let model = load.model

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
        let useActualCache = useKVCache && draftModel == nil
        // Map CLI flag to MLX DType for KV-quantised storage.
        let kvDType: DType? = {
            switch kvQuantize {
            case "fp16", "float16", "half": return .float16
            case "bf16", "bfloat16":         return .bfloat16
            default:                          return nil
            }
        }()
        // Build OR load the cache:
        //   --cache-prompt <path>: if the file exists, load it (skip prefill);
        //   otherwise build an empty cache, prefill, and write it on exit.
        var cache: KVCache? = nil
        var skipPrefill = false
        if useActualCache {
            if let p = prefixCachePath,
               FileManager.default.fileExists(atPath: p)
            {
                do {
                    cache = try KVCache.load(from: URL(fileURLWithPath: p), nLayers: cfg.nLayers)
                    skipPrefill = true
                    print("loaded prefix cache (\(cache!.currentLength) tokens) — skipping prompt prefill")
                } catch {
                    fputs("warning: prefix cache load failed (\(error)); building fresh\n", stderr)
                }
            }
            if cache == nil {
                cache = KVCache(nLayers: cfg.nLayers, kvDtype: kvDType,
                                 sink: streamingSink, window: streamingWindow)
            }
            if kvDType != nil {
                print("KV cache stored at \(kvQuantize!) (≈½ memory vs fp32)")
            }
            if streamingSink != nil || streamingWindow != nil {
                print("StreamingLLM: sink=\(streamingSink ?? 0) window=\(streamingWindow ?? 0)")
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
        if let draft = draftModel {
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
                let cl = cache.currentLength
                if cl >= 1 {
                    for layer in cache.entries.indices {
                        let k = cache.entries[layer].keys
                        let v = cache.entries[layer].values
                        cache.entries[layer].keys = k[0..., 0..., 0..<(k.shape[2] - 1), 0...]
                        cache.entries[layer].values = v[0..., 0..., 0..<(v.shape[2] - 1), 0...]
                    }
                    cache.currentLength = cl - 1
                }
                let logits = model.forwardCached(lastTok, cache: cache)
                lastLogits = logits[0..., logits.shape[1] - 1, 0...]
            } else {
                let prefillLogits = model.forwardCached(promptIds, cache: cache)
                lastLogits = prefillLogits[0..., prefillLogits.shape[1] - 1, 0...]
                // Save the populated cache if the user requested it AND we
                // built fresh — first cold call pays the prefill cost; later
                // ones for the same prompt skip it.
                if let p = prefixCachePath {
                    do {
                        try cache.saveToDisk(to: URL(fileURLWithPath: p))
                        fputs("saved prefix cache → \(p)\n", stderr)
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
        if useActualCache, let c = cache {
            var totalBytes = 0
            var populatedLayers = 0
            for (i, e) in c.entries.enumerated() {
                // Trust the stored dtype's byte width; defaults to 4 if
                // we somehow loaded an unknown type.
                let kBytes = e.keys.shape.reduce(1, *) * dtypeByteWidth(e.keys.dtype)
                let vBytes = e.values.shape.reduce(1, *) * dtypeByteWidth(e.values.dtype)
                totalBytes += kBytes + vBytes
                if e.keys.shape[2] > 0 { populatedLayers += 1 }
                _ = i
            }
            let yocoTag = cfg.useYOCO ? "  · YOCO (\(populatedLayers)/\(cfg.nLayers) layers populated)" : ""
            print(String(format: "KV cache:  %d tokens · %@%@",
                          c.currentLength, formatBytes(totalBytes), yocoTag))
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
        --kv-quantize fp16|bf16
                              Store KV cache in half precision (≈½ memory)
        --cache-prompt <path> Save prompt KV cache to <path> on first run;
                              load it on subsequent runs (skip prompt prefill)
        --streaming-llm-sink N
                              Always keep the first N tokens (StreamingLLM anchor)
        --streaming-llm-window M
                              Keep only the last M tokens beyond the sink
        """)
        exit(2)
    }
}
