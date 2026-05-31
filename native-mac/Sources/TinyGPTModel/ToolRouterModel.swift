import Foundation
import MLX
import MLXNN
import MLXRandom
import TinyGPTIO

/// Tool-call extractor (a.k.a. mini-router).
///
/// A small transformer-encoder classifier that runs BEFORE the full LM
/// forward pass. Given a user query + the active tool catalog, it picks
/// which tool the LM should be steered toward.
///
/// # Architecture
///
/// The model reuses tinygpt's existing `TransformerBlock` stack — same
/// attention, same MLP, same param names — but:
///
///   - DROPs the LM head (no token prediction)
///   - REPLACEs it with a classification head:
///     `mean-pool over T → Linear(d_model → n_classes)`
///   - Trains with cross-entropy over a fixed tool catalog
///
/// The blocks themselves still use *causal* attention (we inherit
/// `CausalSelfAttention` rather than reimplement bidirectional). For a
/// short classification input (≤256 tokens) with mean-pooling the
/// causal mask is a slight loss vs full BERT-style bidirectional, but
/// it's the right tradeoff to keep training infra unchanged. Document
/// the design choice in `docs/tool_call_extractor.md`.
///
/// # Target size
///
/// 4-6 layers, d_model=256, dMlp=1024, ~30M params depending on
/// vocab size + n_classes. CPU inference target: <5 ms for a single
/// 64-token query.
///
/// # Why not just let the LM emit the tool call directly?
///
/// Latency. A 1-3B specialist needs to forward through the full prompt
/// + tool schema (often 1-4 KB of tokens) before it can emit even the
/// first JSON byte. The router runs 4 encoder layers over <100 tokens
/// and outputs softmax over ~20-100 tool classes. That's a 50-100×
/// latency cut for the most common decision the agent makes.
///
/// # Integration with the constrained-decode FSM
///
/// When the router fires with high confidence (>0.7), the agent loop
/// can pin the JSON schema FSM to the matching tool definition —
/// guaranteeing the LM's first emitted byte path is constrained to
/// `{"tool": "<predicted>", "arguments": {...}}`. See
/// `ConstrainedGen.swift` for the FSM. Cline's "force a tool call
/// every turn" enforcement is the inspiration.
public final class ToolRouterModel: Module {

    public let config: ModelConfig
    /// Number of tool classes the head predicts over. The class index
    /// is paired with a `labels` array (stored alongside the
    /// checkpoint) so the inference path can decode `argmax` back to
    /// the tool name.
    public let numClasses: Int
    /// How the per-token hidden states get pooled into a single
    /// classification vector. `.mean` averages over the sequence;
    /// `.firstToken` reads position 0 (BERT [CLS] convention). Mean
    /// is more robust for variable-length queries with no special
    /// classification token.
    public enum Pooling: String, Sendable, Equatable {
        case mean
        case firstToken
    }
    public let pooling: Pooling

    // Shared with TinyGPTModel — same param names so save/load is
    // bit-identical to the LM up to the head.
    @ModuleInfo(key: "token_embedding") public var tokenEmbedding: Embedding
    @ModuleInfo(key: "position_embedding") public var positionEmbedding: Embedding
    @ModuleInfo(key: "blocks") public var blocks: [TransformerBlock]
    @ModuleInfo(key: "ln_final") public var lnFinal: LayerNorm

    /// Classification head. `Linear(d_model → numClasses)`. Lives at
    /// param name `router_head` so the checkpoint loader can recognise
    /// router files vs LM files.
    @ModuleInfo(key: "router_head") public var routerHead: Linear

    public init(_ config: ModelConfig, numClasses: Int, pooling: Pooling = .mean) {
        precondition(numClasses >= 2, "router needs >= 2 classes")
        self.config = config
        self.numClasses = numClasses
        self.pooling = pooling
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.dModel
        )
        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.contextLength, dimensions: config.dModel
        )
        self._blocks.wrappedValue = (0..<config.nLayers).map { _ in
            TransformerBlock(config)
        }
        self._lnFinal.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
        self._routerHead.wrappedValue = Linear(config.dModel, numClasses, bias: true)
        super.init()
    }

    /// `idx: [B, T]` int32 token ids → `[B, numClasses]` logits.
    public func callAsFunction(_ idx: MLXArray) -> MLXArray {
        let h = forwardToHidden(idx)
        return classify(h)
    }

    /// Forward through embedding + blocks + final norm. Returns the
    /// `[B, T, C]` hidden state — same as `TinyGPTModel.forwardToHidden`
    /// minus the YOCO / NEFTune frills (those are LM-specific
    /// regularisers and aren't useful for a classifier).
    public func forwardToHidden(_ idx: MLXArray) -> MLXArray {
        let T = idx.shape[1]
        precondition(T <= config.contextLength,
                     "sequence length \(T) exceeds context \(config.contextLength)")
        let positions = MLXArray((0..<T).map { Int32($0) })
        let posEmb = positionEmbedding(positions).expandedDimensions(axis: 0)
        let tokEmb = tokenEmbedding(idx)
        var x = tokEmb + posEmb
        for block in blocks {
            x = block(x)
        }
        return lnFinal(x)
    }

    /// Pool the per-token hidden state into one classification vector
    /// per batch entry, then project through `routerHead`.
    public func classify(_ hidden: MLXArray) -> MLXArray {
        let pooled: MLXArray
        switch pooling {
        case .mean:
            // Mean across the sequence axis (axis=1). Result: [B, C].
            pooled = hidden.mean(axis: 1)
        case .firstToken:
            // [CLS]-style: read position 0. Result: [B, C].
            pooled = hidden[0..., 0, 0...]
        }
        return routerHead(pooled)
    }

    /// Cross-entropy loss for a batch of `(idx, labels)` pairs.
    /// `labels: [B]` int32 class indices.
    public func loss(_ idx: MLXArray, _ labels: MLXArray) -> MLXArray {
        let logits = self(idx)
        return crossEntropy(
            logits: logits,
            targets: labels,
            reduction: .mean
        )
    }

    /// Softmax over the logits — useful for the inference path which
    /// returns "predicted tool name + confidence".
    public func softmaxProbs(_ idx: MLXArray) -> MLXArray {
        return MLX.softmax(self(idx), axis: -1)
    }

    /// Top-K predicted classes for one query. Returns `(class_idx,
    /// probability)` pairs sorted by probability descending. CPU-side
    /// — caller passes the pre-encoded token ids.
    public func topK(idx: MLXArray, k: Int) -> [(classIdx: Int, prob: Float)] {
        let probs = softmaxProbs(idx)
        eval(probs)
        let vec: [Float] = probs.asArray(Float.self)
        let pairs = vec.enumerated().map { (i, p) in (i, p) }
        let sorted = pairs.sorted { $0.1 > $1.1 }
        return Array(sorted.prefix(max(1, k))).map { (classIdx: $0.0, prob: $0.1) }
    }

    /// Total parameter count.
    public func numParameters() -> Int {
        var total = 0
        for (_, p) in parameters().flattened() {
            total += p.shape.reduce(1, *)
        }
        return total
    }
}

// MARK: - Preset configurations

public extension ToolRouterModel {

    /// Recommended preset: ~30M-class encoder for a small tool catalog.
    /// 4 layers, d_model=256, dMlp=1024, ctx=256. Trains fast on a
    /// few hundred steps of supervised (query, tool) pairs.
    ///
    /// Param count (byte-level vocab=256, numClasses=20):
    ///   - token_emb:   256 × 256       =     65 K
    ///   - pos_emb:     256 × 256       =     65 K
    ///   - 4 × block:   ~1.3 M each     =    5.2 M
    ///   - head:        256 × 20        =      5 K
    ///   total:                              ~5.4 M
    ///
    /// For BPE vocab (e.g. 32 K), embedding alone adds ~16 M, bringing
    /// the model to ~22-30 M which is the actual target.
    static func tinyPreset(vocabSize: Int, contextLength: Int = 256,
                           numClasses: Int) -> (ModelConfig, Int) {
        let cfg = ModelConfig(
            modelName: "tool-router-tiny",
            vocabSize: vocabSize,
            contextLength: contextLength,
            nLayers: 4,
            nHeads: 4,
            dModel: 256,
            dMlp: 1024,
            dropout: 0.0,
            tieEmbeddings: false,
            dtype: "float32"
        )
        return (cfg, numClasses)
    }

    /// Slightly larger preset — 6 layers, ~50-80 M params depending on
    /// vocab. Use this when the tinyPreset overfits or when the tool
    /// catalog is large (>50 tools).
    static func smallPreset(vocabSize: Int, contextLength: Int = 256,
                            numClasses: Int) -> (ModelConfig, Int) {
        let cfg = ModelConfig(
            modelName: "tool-router-small",
            vocabSize: vocabSize,
            contextLength: contextLength,
            nLayers: 6,
            nHeads: 8,
            dModel: 384,
            dMlp: 1536,
            dropout: 0.0,
            tieEmbeddings: false,
            dtype: "float32"
        )
        return (cfg, numClasses)
    }
}

// MARK: - Label table (paired with the .tinygpt checkpoint)

/// The router's softmax outputs are integer class indices; the
/// inference path needs the matching tool names. We store the labels
/// in a sidecar file `<checkpoint>.labels.json` alongside the
/// `.tinygpt` weights. This keeps the checkpoint format unchanged.
///
/// Format:
/// ```json
/// {
///   "version": 1,
///   "kind": "tool-router",
///   "labels": ["read_file", "run_test", "search_web", ...]
/// }
/// ```
public struct ToolRouterLabels: Codable, Equatable, Sendable {
    public var version: Int
    public var kind: String
    public var labels: [String]

    public init(labels: [String]) {
        self.version = 1
        self.kind = "tool-router"
        self.labels = labels
    }

    public static func load(from url: URL) throws -> ToolRouterLabels {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ToolRouterLabels.self, from: data)
    }

    public func save(to url: URL) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try enc.encode(self)
        try data.write(to: url, options: [.atomic])
    }

    /// Default sidecar path: `<checkpoint>.labels.json`.
    public static func sidecarURL(forCheckpoint url: URL) -> URL {
        return url.appendingPathExtension("labels.json")
    }
}

// MARK: - Loader

/// Load a router checkpoint produced by `tinygpt train-extractor`.
/// Mirrors `TinyGPTWeightLoader.load` but targets `ToolRouterModel` —
/// the manifest contains `router_head.{weight,bias}` instead of
/// `lm_head.weight`.
public enum ToolRouterLoader {

    public enum LoaderError: Error, CustomStringConvertible {
        case missingTensor(String)
        case shapeMismatch(name: String, file: [Int], model: [Int])
        public var description: String {
            switch self {
            case .missingTensor(let n):
                return "router checkpoint missing tensor '\(n)'"
            case .shapeMismatch(let n, let f, let m):
                return "router tensor '\(n)' shape file=\(f) model=\(m)"
            }
        }
    }

    /// Read the checkpoint header, infer the model config, build a fresh
    /// `ToolRouterModel`, and patch the weights in. `numClasses` comes
    /// from the labels sidecar — the manifest alone doesn't carry it
    /// (it's a head shape, not a config field).
    public static func load(path: String, numClasses: Int) throws -> ToolRouterModel {
        let url = URL(fileURLWithPath: path)
        let file = try TinyGPTFileReader.readMapped(url)
        let h = file.header.config

        let cfg = ModelConfig(
            modelName: "tool-router",
            vocabSize: h.vocabSize ?? 256,
            contextLength: h.ctx ?? 128,
            nLayers: h.layers ?? 4,
            nHeads: h.heads ?? 4,
            dModel: h.dModel ?? 256,
            dMlp: h.dMlp ?? 1024,
            dropout: 0.0,
            tieEmbeddings: false,
            dtype: "float32"
        )
        let model = ToolRouterModel(cfg, numClasses: numClasses, pooling: .mean)

        var byName: [String: TinyGPTTensor] = [:]
        byName.reserveCapacity(file.tensors.count)
        for t in file.tensors { byName[t.entry.name] = t }

        // For each leaf in the model's parameter tree, look up the
        // matching tensor by name. We DON'T transpose Linear weights
        // here — the trainer (`TrainExtractor.saveCheckpoint`) writes
        // weights from the live MLX-Swift module's `parameters()`, so
        // the on-disk byte order is already MLX-native.
        var flat: [String: MLXArray] = [:]
        for (name, _) in model.parameters().flattened() {
            guard let tensor = byName[name] else {
                throw LoaderError.missingTensor(name)
            }
            flat[name] = mlxArray(from: tensor)
        }

        let nested = rewriteLeaves(model.parameters(), withFlat: flat)
        try model.update(parameters: nested, verify: [.noUnusedKeys])
        eval(model)
        return model
    }

    // MARK: - Helpers (parallel to TinyGPTWeightLoader internals)

    private static func mlxArray(from t: TinyGPTTensor) -> MLXArray {
        switch t.dtype {
        case .fp32:
            return MLXArray(t.weight, t.entry.shape, dtype: .float32)
        case .fp16:
            return MLXArray(t.weight, t.entry.shape, dtype: .float16)
                .asType(.float32)
        }
    }

    private static func rewriteLeaves(
        _ params: ModuleParameters, withFlat flat: [String: MLXArray]
    ) -> ModuleParameters {
        var result = NestedDictionary<String, MLXArray>()
        for (key, item) in params {
            result[key] = rewriteItem(item, path: [key], flat: flat)
        }
        return result
    }

    private static func rewriteItem(
        _ item: NestedItem<String, MLXArray>,
        path: [String],
        flat: [String: MLXArray]
    ) -> NestedItem<String, MLXArray> {
        switch item {
        case .none:
            return .none
        case .value:
            let key = path.joined(separator: ".")
            if let v = flat[key] {
                return .value(v)
            }
            return item
        case .array(let elements):
            return .array(
                elements.enumerated().map { (idx, child) in
                    rewriteItem(child, path: path + [String(idx)], flat: flat)
                }
            )
        case .dictionary(let dict):
            var newDict: [String: NestedItem<String, MLXArray>] = [:]
            for (k, child) in dict {
                newDict[k] = rewriteItem(child, path: path + [k], flat: flat)
            }
            return .dictionary(newDict)
        }
    }
}
