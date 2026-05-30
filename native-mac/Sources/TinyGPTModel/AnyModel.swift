import Foundation
import MLX
import MLXNN
import TinyGPTIO

/// A unified wrapper so the CLI commands (sample / finetune / compare /
/// eval) can operate on EITHER a from-scratch TinyGPTModel or an
/// HF-loaded TinyGPTModelHF without branching at every call site.
///
/// The wrapper:
///   - exposes the common interface (callAsFunction, loss, numParameters)
///   - forwards to the underlying concrete model
///   - knows which LoRA injection variant to use
public enum AnyModel {
    case fromScratch(TinyGPTModel)
    case huggingFace(TinyGPTModelHF)

    public var config: ModelConfig {
        switch self {
        case .fromScratch(let m): return m.config
        case .huggingFace(let m): return m.config
        }
    }

    public func callAsFunction(_ idx: MLXArray) -> MLXArray {
        switch self {
        case .fromScratch(let m): return m(idx)
        case .huggingFace(let m): return m(idx)
        }
    }

    public func loss(_ idx: MLXArray, _ targets: MLXArray) -> MLXArray {
        switch self {
        case .fromScratch(let m): return m.loss(idx, targets)
        case .huggingFace(let m):
            // HF model doesn't have a built-in loss helper; compute inline.
            let logits = m(idx)
            let v = logits.shape.last!
            return crossEntropy(
                logits: logits.reshaped([-1, v]),
                targets: targets.reshaped([-1]),
                reduction: .mean
            )
        }
    }

    /// Cross-entropy averaged only over positions where `mask == 1.0`.
    /// Used by SFT to score the model on the RESPONSE tokens only,
    /// ignoring the (instruction, special markers, padding) positions.
    /// Without this masking, the model learns to predict its own
    /// instruction back to itself instead of learning the response.
    public func maskedLoss(_ idx: MLXArray, _ targets: MLXArray, _ mask: MLXArray) -> MLXArray {
        let logits: MLXArray
        switch self {
        case .fromScratch(let m): logits = m(idx)
        case .huggingFace(let m): logits = m(idx)
        }
        let v = logits.shape.last!
        let flatLogits = logits.reshaped([-1, v])
        let flatTargets = targets.reshaped([-1])
        let flatMask = mask.reshaped([-1])
        // Per-token CE, no reduction — multiply by mask, then average over
        // mask.sum() (the count of scored positions). `+ 1` denom guards
        // against an all-masked-out batch (shouldn't happen but defensive).
        let perTok = crossEntropy(
            logits: flatLogits, targets: flatTargets, reduction: .none
        )
        let masked = perTok * flatMask
        let denom = flatMask.sum() + MLXArray(Float(1e-6))
        return masked.sum() / denom
    }

    public func numParameters() -> Int {
        switch self {
        case .fromScratch(let m): return m.numParameters()
        case .huggingFace(let m): return m.numParameters()
        }
    }

    public func parameters() -> ModuleParameters {
        switch self {
        case .fromScratch(let m): return m.parameters()
        case .huggingFace(let m): return m.parameters()
        }
    }

    /// Inject LoRA on the right variant; returns trainable param count.
    @discardableResult
    public func injectLora(config: LoraConfig) -> Int {
        switch self {
        case .fromScratch(let m):
            LoraInjection.inject(m, config: config)
            LoraInjection.freezeBase(m)
            return LoraInjection.trainableParamCount(in: m)
        case .huggingFace(let m):
            LoraInjectionHF.inject(m, config: config)
            LoraInjectionHF.freezeBase(m)
            return LoraInjectionHF.trainableParamCount(in: m)
        }
    }

    /// Apply a saved LoRA adapter to whichever variant we are.
    public func applyLora(_ adapter: LoraAdapter) throws {
        switch self {
        case .fromScratch(let m):
            try LoraAdapterReader.apply(adapter, to: m)
        case .huggingFace(let m):
            try LoraAdapterHFReader.apply(adapter, to: m)
        }
    }

    /// Save a LoRA adapter to disk. The model must have been injected
    /// + trained; this serialises just the A/B matrices.
    public func saveLora(baseConfig: ModelConfig, loraConfig: LoraConfig,
                          finalLoss: Float?, to url: URL) throws {
        switch self {
        case .fromScratch(let m):
            try LoraAdapterWriter.write(model: m, baseConfig: baseConfig,
                                          loraConfig: loraConfig,
                                          finalLoss: finalLoss, to: url)
        case .huggingFace(let m):
            try LoraAdapterHFWriter.write(model: m, baseConfig: baseConfig,
                                            loraConfig: loraConfig,
                                            finalLoss: finalLoss, to: url)
        }
    }

    /// Underlying Module — used by `freeze`/`unfreeze`/optimiser plumbing.
    public var module: Module {
        switch self {
        case .fromScratch(let m): return m
        case .huggingFace(let m): return m
        }
    }

    /// KV-cached forward pass — works for both model variants. First call
    /// with an empty cache processes the full prompt; later calls usually
    /// pass `[B, 1]` for streaming decode. Returns logits of shape
    /// `[B, T_new, vocab_size]`.
    public func forwardCached(_ idx: MLXArray, cache: KVCache) -> MLXArray {
        switch self {
        case .fromScratch(let m): return m.forwardCached(idx, cache: cache)
        case .huggingFace(let m): return m.forwardCached(idx, cache: cache)
        }
    }
}

/// Detect whether a path is a from-scratch `.tinygpt` checkpoint or an
/// HuggingFace model directory. Returns the loaded model + its config.
public enum ModelLoader {
    public struct LoadResult {
        public let model: AnyModel
        public let config: ModelConfig
        /// HF-model variants come with their own tokenizer on disk.
        /// Set to the model directory's URL so callers can load the
        /// tokenizer separately. nil for from-scratch byte-level models.
        public let hfTokenizerDir: URL?
    }

    public static func load(_ path: String) throws -> LoadResult {
        let url = URL(fileURLWithPath: path)
        var isDirectory: ObjCBool = false
        FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)

        if isDirectory.boolValue {
            // HF model directory — expects config.json inside.
            let configURL = url.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: configURL.path) else {
                throw NSError(domain: "TinyGPT", code: 10,
                              userInfo: [NSLocalizedDescriptionKey:
                                "directory \(path) has no config.json — not an HF model dir"])
            }
            let hfResult = try HFModelLoader.load(from: url)
            return LoadResult(model: .huggingFace(hfResult.model),
                              config: hfResult.config, hfTokenizerDir: url)
        }

        // .tinygpt file path. Pick up vocabSize + tokenizerSource from the
        // header — BPE-trained from-scratch models pin their tokenizer dir.
        // MoE fields (nExperts/moeTopK/loadBalanceWeight) are picked up
        // when present so the router + per-expert structure reconstructs
        // exactly; absent → dense MLP (the default).
        let file = try TinyGPTFileReader.read(url)
        let h = file.header.config
        let cfg = ModelConfig(
            vocabSize: h.vocabSize ?? 256,
            contextLength: h.ctx ?? 256,
            nLayers: h.layers ?? 12,
            nHeads: h.heads ?? 8,
            dModel: h.dModel ?? 256,
            dMlp: h.dMlp ?? 1024,
            tokenizerSource: h.tokenizerSource,
            nExperts: h.nExperts ?? 1,
            moeTopK: h.moeTopK ?? 1,
            loadBalanceWeight: h.loadBalanceWeight ?? 0.01,
            slidingWindow: h.slidingWindow,
            useMoD: h.useMoD ?? false,
            useDifferentialAttention: h.useDifferentialAttention ?? false,
            useYOCO: h.useYOCO ?? false,
            useGradCheckpoint: h.useGradCheckpoint ?? false,
            // Tier 2 stability bells round-trip through the manifest.
            // GaLore / z-loss / layer-LR decay are training-only and
            // don't affect the loaded model's forward; the
            // architectural flags (DeepNorm, embedding RMSNorm) DO.
            galoreRank: h.galoreRank,
            galoreUpdateEvery: h.galoreUpdateEvery,
            zLossWeight: h.zLossWeight ?? 0,
            useDeepNorm: h.useDeepNorm ?? false,
            lrLayerDecay: h.lrLayerDecay ?? 1.0,
            useEmbeddingRMSNorm: h.useEmbeddingRMSNorm ?? false
        )
        let m = TinyGPTModel(cfg)
        try TinyGPTWeightLoader.load(file, into: m)
        let tokDir = h.tokenizerSource.map { URL(fileURLWithPath: $0) }
        return LoadResult(model: .fromScratch(m), config: cfg, hfTokenizerDir: tokDir)
    }
}
