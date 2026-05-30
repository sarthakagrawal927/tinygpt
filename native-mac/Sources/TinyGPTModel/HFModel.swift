import Foundation
import MLX
import MLXNN
import MLXRandom
import TinyGPTIO

/// HF-style top-level transformer model. Mirrors the architecture of
/// Llama 2+ / Mistral / Phi-3 / Qwen / Gemma / LFM: token embedding
/// (no learned positional embedding — RoPE handles position inside
/// attention), N TransformerBlockHF instances, final RMSNorm, untied
/// LM head (Linear from d_model → vocab_size, no bias).
///
/// Param-name layout (matches HF safetensors):
///
///     model.embed_tokens.weight                                  → tokenEmbedding
///     model.layers.N.input_layernorm.weight                      → blocks[N].ln1
///     model.layers.N.self_attn.{q,k,v,o}_proj.weight             → blocks[N].attn.*
///     model.layers.N.post_attention_layernorm.weight             → blocks[N].ln2
///     model.layers.N.mlp.{gate,up,down}_proj.weight              → blocks[N].mlp.*
///     model.norm.weight                                          → lnFinal
///     lm_head.weight                                             → lmHead (untied)
public final class TinyGPTModelHF: Module {
    public let config: ModelConfig

    @ModuleInfo(key: "embed_tokens") public var tokenEmbedding: Embedding
    @ModuleInfo(key: "layers")       public var blocks: [TransformerBlockHF]
    @ModuleInfo(key: "norm")         public var lnFinal: RMSNorm
    @ModuleInfo(key: "lm_head")      public var lmHead: Linear?

    /// Optional RMSNorm on the embedding output. Same role as the
    /// matching slot on `TinyGPTModel`. Populated when
    /// `cfg.useEmbeddingRMSNorm`. Outside the HF safetensors naming
    /// convention (HF doesn't ship this layer), so an HF-loaded
    /// model that ENABLES it via the manifest gets a fresh weight
    /// initialised to ones — the HF load path doesn't touch it.
    @ModuleInfo(key: "embed_norm")   public var embedNorm: RMSNorm?

    /// NEFTune (Jain et al., 2024) — scale of uniform noise added to the
    /// token-embedding output during forward. 0 (default) = off. See the
    /// matching field on `TinyGPTModel`.
    public var nefTuneAlpha: Float = 0

    public init(_ cfg: ModelConfig) {
        self.config = cfg
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: cfg.vocabSize, dimensions: cfg.dModel)
        // YOCO layer split — see `TinyGPTModel.init` for the rationale.
        let yocoAnchorIdx = max(0, (cfg.nLayers / 2) - 1)
        self._blocks.wrappedValue = (0..<cfg.nLayers).map { i in
            let secondHalf = cfg.useYOCO && i > yocoAnchorIdx
            let b = TransformerBlockHF(cfg, yocoSecondHalf: secondHalf)
            b.useGradCheckpoint = cfg.useGradCheckpoint
            return b
        }
        self._lnFinal.wrappedValue = RMSNorm(dimensions: cfg.dModel, eps: 1e-5)
        if cfg.tieEmbeddings {
            self._lmHead.wrappedValue = nil
        } else {
            self._lmHead.wrappedValue = Linear(cfg.dModel, cfg.vocabSize, bias: false)
        }
        if cfg.useEmbeddingRMSNorm {
            self._embedNorm.wrappedValue = RMSNorm(dimensions: cfg.dModel, eps: 1e-5)
        } else {
            self._embedNorm.wrappedValue = nil
        }
        super.init()
    }

    public func callAsFunction(_ idx: MLXArray) -> MLXArray {
        var x = tokenEmbedding(idx)
        if nefTuneAlpha > 0 {
            // NEFTune: small Uniform[-s, s] noise added at the input.
            let T = idx.shape[1]
            let s = nefTuneAlpha / sqrt(Float(T * config.dModel))
            let noise = MLXRandom.uniform(low: -s, high: s, x.shape).asType(x.dtype)
            x = x + noise
        }
        // Optional embedding-output RMSNorm. See the matching slot on
        // TinyGPTModel for the rationale.
        if let en = embedNorm {
            x = en(x)
        }
        if config.useYOCO {
            // YOCO orchestration: first half runs standard self-attention;
            // the LAST first-half block captures its (K, V); second-half
            // blocks cross-attend against that captured pair. Halves the
            // KV cache memory at long-context decode (see KVCacheHF.swift).
            let anchorIdx = max(0, (blocks.count / 2) - 1)
            var yocoK: MLXArray? = nil
            var yocoV: MLXArray? = nil
            for (i, block) in blocks.enumerated() {
                if i < anchorIdx {
                    x = block(x)
                } else if i == anchorIdx {
                    let r = block.callCapturingKV(x)
                    x = r.out; yocoK = r.k; yocoV = r.v
                } else {
                    guard let k = yocoK, let v = yocoV else {
                        preconditionFailure("YOCO anchor missing at HF layer \(i)")
                    }
                    x = block.callWithExternalKV(x, k: k, v: v)
                }
            }
        } else {
            for block in blocks {
                x = block(x)
            }
        }
        x = lnFinal(x)
        if let head = lmHead {
            return head(x)
        }
        return tokenEmbedding.asLinear(x)
    }

    public func numParameters() -> Int {
        var total = 0
        for (_, p) in parameters().flattened() {
            total += p.shape.reduce(1, *)
        }
        return total
    }

    /// Cross-entropy + optional z-loss. Parallel to `TinyGPTModel.loss`
    /// but without the MTP / MoE branches (HF model doesn't host either
    /// yet). Used by `TrainerHF` when a non-zero `zLossWeight` is set.
    public func loss(_ idx: MLXArray, _ targets: MLXArray) -> MLXArray {
        let logits = self(idx)
        let v = logits.shape.last!
        let ce = crossEntropy(
            logits: logits.reshaped([-1, v]),
            targets: targets.reshaped([-1]),
            reduction: .mean
        )
        if config.zLossWeight <= 0 { return ce }
        let flat = logits.reshaped([-1, v])
        let maxLogit = flat.max(axis: -1, keepDims: true)
        let shifted = flat - maxLogit
        let lse = MLX.log(MLX.exp(shifted).sum(axis: -1, keepDims: true)) + maxLogit
        return ce + MLXArray(config.zLossWeight) * (lse * lse).mean()
    }
}

/// Convert an HF config into our ModelConfig. The result has RoPE +
/// RMSNorm + SwiGLU + GQA toggled on (the standard "modern" config),
/// matching what a TinyGPTModelHF expects.
public enum HFConfigConverter {
    public static func toModelConfig(_ hf: HuggingFaceConfig) -> ModelConfig {
        return ModelConfig(
            modelName: hf.architectures.first ?? "hf-loaded",
            vocabSize: hf.vocabSize,
            contextLength: hf.maxPositionEmbeddings,
            nLayers: hf.numHiddenLayers,
            nHeads: hf.numAttentionHeads,
            nKvHeads: hf.numKeyValueHeads,
            dModel: hf.hiddenSize,
            dMlp: hf.intermediateSize,
            dropout: 0.0,
            tieEmbeddings: hf.tieWordEmbeddings,
            dtype: "float32",
            useRoPE: true,
            ropeBase: hf.ropeTheta,
            useRMSNorm: true,
            useSwiGLU: true,
            attnBias: false  // HF Llama-family models don't use attention biases
        )
    }
}

/// Load an HF model from a local directory containing config.json +
/// {model.safetensors, model-XXXXX-of-YYYYY.safetensors}. Returns a
/// fully-loaded TinyGPTModelHF ready to sample from.
///
/// Sharding: large HF models are split across multiple safetensors
/// files (e.g., Llama-3-8B = 4 shards of ~2GB each). The loader walks
/// every `*.safetensors` in the directory and pools their tensors into
/// one name → bytes map. Order doesn't matter; the per-shard headers
/// tell us which tensors live where.
///
/// dtype conversion: HF models ship as bf16 or fp16 on disk; we
/// up-convert to fp32 on the host as we load (MLX-Swift's training +
/// sampling paths run fp32 by default). Total RAM cost is 2× the
/// download size during load, drops back to ~download-size after the
/// MLX arrays settle.
public enum HFModelLoader {
    public enum LoadError: Error, CustomStringConvertible {
        case missingConfig(URL)
        case noSafetensors(URL)
        case unmappedTensor(String)
        case missingTensor(String)
        case shapeMismatch(name: String, expected: [Int], got: [Int])

        public var description: String {
            switch self {
            case .missingConfig(let u): return "no config.json in \(u.path)"
            case .noSafetensors(let u): return "no .safetensors in \(u.path)"
            case .unmappedTensor(let n): return "HF tensor name '\(n)' has no mapping to our model"
            case .missingTensor(let n): return "expected tensor '\(n)' not present in safetensors"
            case .shapeMismatch(let n, let exp, let got): return "\(n) shape mismatch: model wants \(exp), file has \(got)"
            }
        }
    }

    public struct LoadResult {
        public let model: TinyGPTModelHF
        public let config: ModelConfig
        public let hfConfig: HuggingFaceConfig
    }

    /// Load a HF model directory. Throws if architecture isn't supported
    /// (the HFConfig's `unsupportedReason()` is consulted first) or if
    /// any required weight tensor is missing.
    public static func load(from dir: URL) throws -> LoadResult {
        let configURL = dir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw LoadError.missingConfig(dir)
        }
        let hfConfig = try HuggingFaceConfig.read(configURL)

        // Construct the model from the config.
        var cfg = HFConfigConverter.toModelConfig(hfConfig)
        // Peek at the safetensors shards: if NO lm_head.weight exists,
        // the model uses tied embeddings regardless of the config field
        // (Llama family's default is `tie_word_embeddings: true` when the
        // field is absent, and smaller models often omit a separate
        // lm_head to save params). Setting tieEmbeddings=true makes the
        // model use tokenEmbedding.asLinear(x) for the output projection
        // instead of a random fresh Linear that never gets overwritten.
        let allFilesPeek = (try? FileManager.default.contentsOfDirectory(at: dir,
                              includingPropertiesForKeys: nil)) ?? []
        var hasLmHead = false
        for shardURL in allFilesPeek where shardURL.pathExtension == "safetensors" {
            if let f = try? SafetensorsReader.read(shardURL),
               f.tensors["lm_head.weight"] != nil {
                hasLmHead = true; break
            }
        }
        if !hasLmHead { cfg.tieEmbeddings = true }
        let model = TinyGPTModelHF(cfg)

        // Find all safetensors files in the dir.
        let allFiles = (try? FileManager.default.contentsOfDirectory(at: dir,
                          includingPropertiesForKeys: nil)) ?? []
        let shards = allFiles
            .filter { $0.pathExtension == "safetensors" }
            .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
        guard !shards.isEmpty else { throw LoadError.noSafetensors(dir) }

        // Read every shard, building a single name → (info, file) map.
        struct TensorSource {
            let file: SafetensorsReader.File
            let info: SafetensorsReader.TensorInfo
        }
        var sources: [String: TensorSource] = [:]
        for shardURL in shards {
            let file = try SafetensorsReader.read(shardURL)
            for (name, info) in file.tensors {
                sources[name] = TensorSource(file: file, info: info)
            }
        }

        // GPTQ / AWQ detection pass. Both formats fan out a single
        // `{name}.weight` Linear into a packed quartet/triple:
        //   GPTQ: qweight + scales + qzeros (+ optional g_idx)
        //   AWQ : qweight + scales + qzeros
        // We dequantise to dense fp32 here so the rest of the loader
        // pipeline (param-name mapping, update apply) sees a normal
        // `{name}.weight` tensor and Just Works. Memory cost: ~8× the
        // packed payload during load (8-bit nibbles → 32-bit floats).
        // Inference speedup is queued behind a packed-int4 matmul kernel.
        var dequantised: [String: MLXArray] = [:]  // hfName -> dense weight
        var dequantisedBases: Set<String> = []      // base names whose quartet is consumed
        let allNames = Array(sources.keys)
        let gptqBases = GPTQReader.detectGptqBases(in: allNames)
        let awqBases = AWQReader.detectAwqBases(in: allNames)
        // GPTQ first — when both readers detect a base (the format is
        // ambiguous on tensor names alone), prefer GPTQ because the
        // `g_idx` sibling is GPTQ-specific. Pure-AWQ checkpoints lack g_idx.
        for base in gptqBases {
            // Skip if this base looks more like AWQ (has scales but
            // NEVER ships a g_idx in the dir — both formats share the
            // qweight/scales/qzeros triple, so detect via g_idx presence
            // OR config.json's quant_method when present).
            guard let qwSrc = sources[base + ".qweight"],
                  let scSrc = sources[base + ".scales"],
                  let qzSrc = sources[base + ".qzeros"] else { continue }
            let gIdxSrc = sources[base + ".g_idx"]
            // Heuristic: presence of g_idx → definitely GPTQ. Absence of
            // g_idx AND awqBases contains this base AND the user's
            // config.json says quant_method=awq → leave for AWQ pass.
            // For this drop we go GPTQ-first as a conservative default;
            // a future revision can read config.json's quant_config.
            if gIdxSrc == nil && awqBases.contains(base) {
                // Defer to AWQ pass.
                continue
            }
            let qw = GPTQTensor(shape: qwSrc.info.shape, dtype: qwSrc.info.dtype,
                                 bytes: qwSrc.file.tensorData(base + ".qweight")!)
            let sc = GPTQTensor(shape: scSrc.info.shape, dtype: scSrc.info.dtype,
                                 bytes: scSrc.file.tensorData(base + ".scales")!)
            let qz = GPTQTensor(shape: qzSrc.info.shape, dtype: qzSrc.info.dtype,
                                 bytes: qzSrc.file.tensorData(base + ".qzeros")!)
            let g = gIdxSrc.map {
                GPTQTensor(shape: $0.info.shape, dtype: $0.info.dtype,
                            bytes: $0.file.tensorData(base + ".g_idx")!)
            }
            let dense: MLXArray
            do {
                dense = try GPTQReader.dequantize(qweight: qw, scales: sc, qzeros: qz, gIdx: g)
            } catch {
                // Hard fail — the user explicitly downloaded a quantised
                // model; silently falling back would produce garbage.
                fatalError("GPTQ dequant failed for \(base): \(error)")
            }
            dequantised[base + ".weight"] = dense
            dequantisedBases.insert(base)
        }
        // AWQ pass — any base GPTQ skipped (no g_idx + detected by AWQ).
        for base in awqBases where !dequantisedBases.contains(base) {
            guard let qwSrc = sources[base + ".qweight"],
                  let scSrc = sources[base + ".scales"],
                  let qzSrc = sources[base + ".qzeros"] else { continue }
            let qw = AWQTensor(shape: qwSrc.info.shape, dtype: qwSrc.info.dtype,
                                bytes: qwSrc.file.tensorData(base + ".qweight")!)
            let sc = AWQTensor(shape: scSrc.info.shape, dtype: scSrc.info.dtype,
                                bytes: scSrc.file.tensorData(base + ".scales")!)
            let qz = AWQTensor(shape: qzSrc.info.shape, dtype: qzSrc.info.dtype,
                                bytes: qzSrc.file.tensorData(base + ".qzeros")!)
            let dense: MLXArray
            do {
                dense = try AWQReader.dequantize(qweight: qw, scales: sc, qzeros: qz)
            } catch {
                fatalError("AWQ dequant failed for \(base): \(error)")
            }
            dequantised[base + ".weight"] = dense
            dequantisedBases.insert(base)
        }

        // Build the flat update dict using OUR HFModel's parameter names.
        // TinyGPTModelHF's @ModuleInfo keys are HF-native ("embed_tokens",
        // "layers", "norm", "self_attn", "input_layernorm" …) so the only
        // transform needed is stripping the "model." prefix that HF
        // safetensors prepends. lm_head.weight has no prefix.
        var updates: [String: MLXArray] = [:]
        // Quartet tensor suffixes we should skip — they were consumed by
        // the GPTQ/AWQ dequant pass above and the dense replacement lives
        // in `dequantised`.
        let quantSuffixes = [".qweight", ".scales", ".qzeros", ".g_idx"]
        for (hfName, src) in sources {
            // Skip quartet members that have been folded into a dense
            // .weight by the dequant pass.
            let base: String? = quantSuffixes.first(where: { hfName.hasSuffix($0) })
                .map { String(hfName.dropLast($0.count)) }
            if let b = base, dequantisedBases.contains(b) {
                continue
            }
            let key: String
            if hfName.hasPrefix("model.") {
                key = String(hfName.dropFirst("model.".count))
            } else if hfName == "lm_head.weight" {
                key = "lm_head.weight"
            } else {
                // Unknown top-level — likely a rotary inv_freq buffer or
                // similar; HF models often emit those as separate tensors
                // even though we compute them inline. Skip silently.
                continue
            }
            let bytes = src.file.tensorData(hfName)!
            let array = makeMLXArray(bytes: bytes, dtype: src.info.dtype, shape: src.info.shape)
            updates[key] = array
        }
        // Splice in the dequantised dense weights under the .weight key
        // (with the same "model." prefix stripping rule).
        for (hfName, arr) in dequantised {
            let key = hfName.hasPrefix("model.")
                ? String(hfName.dropFirst("model.".count))
                : hfName
            updates[key] = arr
        }

        // Apply the parameter updates to the model.
        let nested = buildNested(updates, model: model)
        try model.update(parameters: nested, verify: [])

        return LoadResult(model: model, config: cfg, hfConfig: hfConfig)
    }

    /// Construct an MLXArray from raw bytes + dtype + shape. Supports
    /// the dtypes HF actually emits: F32, F16, BF16, I8, I32. Up-converts
    /// everything to fp32 for downstream training/sampling.
    private static func makeMLXArray(bytes: Data, dtype: String, shape: [Int]) -> MLXArray {
        let n = shape.reduce(1, *)
        switch dtype {
        case "F32":
            let f32 = bytes.withUnsafeBytes { ptr -> [Float] in
                Array(UnsafeBufferPointer<Float>(
                    start: ptr.baseAddress?.assumingMemoryBound(to: Float.self),
                    count: n))
            }
            return MLXArray(f32, shape)
        case "F16":
            let f16 = bytes.withUnsafeBytes { ptr -> [UInt16] in
                Array(UnsafeBufferPointer<UInt16>(
                    start: ptr.baseAddress?.assumingMemoryBound(to: UInt16.self),
                    count: n))
            }
            var out = [Float](repeating: 0, count: n)
            for i in 0..<n { out[i] = Float(Float16(bitPattern: f16[i])) }
            return MLXArray(out, shape)
        case "BF16":
            // bf16 is the upper half of fp32's bit pattern (the high 16
            // bits of a Float). To convert: shift bf16 left 16 bits and
            // reinterpret as Float.
            let bf16 = bytes.withUnsafeBytes { ptr -> [UInt16] in
                Array(UnsafeBufferPointer<UInt16>(
                    start: ptr.baseAddress?.assumingMemoryBound(to: UInt16.self),
                    count: n))
            }
            var out = [Float](repeating: 0, count: n)
            for i in 0..<n {
                let bits = UInt32(bf16[i]) << 16
                out[i] = Float(bitPattern: bits)
            }
            return MLXArray(out, shape)
        default:
            fatalError("unsupported HF safetensors dtype: \(dtype)")
        }
    }

    /// Build a properly-nested ModuleParameters dict from a flat dotted-key
    /// dict, using the model's existing parameter structure as a template.
    /// Mirrors TinyGPTWeightLoader.rewriteLeaves but adapted for the HF
    /// model's slightly different naming (layers, embed_tokens, norm, etc.).
    private static func buildNested(_ flat: [String: MLXArray],
                                     model: TinyGPTModelHF) -> ModuleParameters {
        // Walk model.parameters() and substitute matching values from flat.
        var result = NestedDictionary<String, MLXArray>()
        let existing = model.parameters()
        for (key, item) in existing {
            result[key] = rewriteItem(item, path: [key], flat: flat)
        }
        return result
    }

    /// Walk an existing ModuleParameters tree and replace each leaf
    /// MLXArray with the value from `flat` keyed by its dotted path.
    /// Since TinyGPTModelHF's @ModuleInfo keys ARE the HF param names
    /// (minus the "model." prefix already stripped above), the dotted
    /// path joins back to exactly the right key.
    private static func rewriteItem(_ item: NestedItem<String, MLXArray>,
                                     path: [String],
                                     flat: [String: MLXArray]) -> NestedItem<String, MLXArray> {
        switch item {
        case .none: return .none
        case .value:
            let key = path.joined(separator: ".")
            if let v = flat[key] {
                return .value(v)
            }
            return item
        case .array(let elements):
            return .array(elements.enumerated().map { (i, e) in
                rewriteItem(e, path: path + [String(i)], flat: flat)
            })
        case .dictionary(let dict):
            var newDict: [String: NestedItem<String, MLXArray>] = [:]
            for (k, v) in dict {
                newDict[k] = rewriteItem(v, path: path + [k], flat: flat)
            }
            return .dictionary(newDict)
        }
    }
}
