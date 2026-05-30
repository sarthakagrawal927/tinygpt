import Foundation

/// JSON-encoded header for a `.tinygpt` file. Schema matches what the browser
/// (`browser/src/storage.ts`) and the Python reference (`python_ref/load_tinygpt.py`)
/// produce.
///
/// Decoded with relaxed key matching: unknown keys in the JSON header are
/// ignored, so the browser can add new metadata fields without breaking
/// existing readers.
public struct TinyGPTHeader: Codable, Sendable, Equatable {
    /// Model architecture configuration. The browser stores this as a free-form
    /// JSON object; we decode the known fields and stash the rest in `extras`.
    public struct Config: Codable, Sendable, Equatable {
        public var layers: Int?
        public var dModel: Int?
        public var ctx: Int?
        public var heads: Int?
        public var dMlp: Int?
        public var batchSize: Int?
        public var backend: String?
        /// Vocabulary size — required when ≠ 256 (BPE-trained models). Absent
        /// or 256 means byte-level (legacy default).
        public var vocabSize: Int?
        /// HF tokenizer directory the model was trained against. nil means
        /// byte-level. Travels with the checkpoint so sample/finetune can
        /// re-load the matching tokenizer.
        public var tokenizerSource: String?

        /// Mixture-of-Experts metadata. All three are nil for a standard
        /// dense model — backwards-compatible with pre-MoE readers (the
        /// JSON decoder ignores absent optional fields, and older
        /// writers just don't emit them).
        public var nExperts: Int?
        public var moeTopK: Int?
        public var loadBalanceWeight: Float?

        /// Sliding-window attention size. `nil` = full causal (standard).
        /// When set, every query attends only to the last N positions.
        public var slidingWindow: Int?

        /// Mixture-of-Depths flag. `nil` / false = standard dense. When
        /// set, each TransformerBlock has a learned per-token sigmoid
        /// gate that scales the block's contribution.
        public var useMoD: Bool?

        /// Differential attention flag (Ye et al., 2024). `nil` / false =
        /// standard MHA. When set, each block gains a `diff_attn` sibling
        /// with 2× Q/K projections + λ scalar.
        public var useDifferentialAttention: Bool?

        /// YOCO flag (Lin et al., 2024). `nil` / false = full self-
        /// attention at every layer. When set, the model splits in
        /// half and second-half layers cross-attend to the first
        /// half's anchor K, V.
        public var useYOCO: Bool?

        /// Gradient (activation) checkpointing flag. `nil` / false =
        /// standard training (all activations retained). When true,
        /// each TransformerBlock's forward is wrapped in an MLX
        /// `CustomFunction` whose VJP recomputes the forward at
        /// backward time — trading ~30% extra compute for a large
        /// reduction in activation memory. Training-only knob;
        /// inference paths ignore it.
        public var useGradCheckpoint: Bool?

        /// KIVI (Liu et al., 2023) recommended precision for sample-time
        /// KV cache. 4 or 8 = `--kv-quantize int4|int8` is the
        /// recommended default for this checkpoint; absent = no
        /// recommendation. Sample-time CLI flags override. Travels in
        /// the manifest so a deployed gallery checkpoint can ship its
        /// chosen memory profile inline.
        public var kviBits: Int?

        /// StreamingLLM (Xiao et al., 2023) recommended sink + window
        /// sizes for sample-time KV cache. Same semantics as `kviBits`
        /// — inference hint, CLI overrides.
        public var streamingSink: Int?
        public var streamingWindow: Int?

        /// GaLore rank — `nil` means standard AdamW. When set, the
        /// trainer projects each 2-D weight gradient through a rank-R
        /// basis before the optimiser step (Zhao et al., 2024).
        public var galoreRank: Int?
        /// Refresh cadence for the GaLore projection basis (steps).
        public var galoreUpdateEvery: Int?
        /// Z-loss weight (PaLM / GShard). `nil` or 0 = disabled.
        public var zLossWeight: Float?
        /// DeepNorm flag (Wang et al., 2022). `nil` / false = standard
        /// transformer init + residual.
        public var useDeepNorm: Bool?
        /// Layer-wise LR decay factor. `nil` or 1.0 = no decay.
        public var lrLayerDecay: Float?
        /// Embedding-output RMSNorm flag. When true, the model has an
        /// extra `embed_norm.weight` tensor in the manifest right
        /// after `token_embedding.weight`.
        public var useEmbeddingRMSNorm: Bool?

        public init(
            layers: Int? = nil,
            dModel: Int? = nil,
            ctx: Int? = nil,
            heads: Int? = nil,
            dMlp: Int? = nil,
            batchSize: Int? = nil,
            backend: String? = nil,
            vocabSize: Int? = nil,
            tokenizerSource: String? = nil,
            nExperts: Int? = nil,
            moeTopK: Int? = nil,
            loadBalanceWeight: Float? = nil,
            slidingWindow: Int? = nil,
            useMoD: Bool? = nil,
            useDifferentialAttention: Bool? = nil,
            useYOCO: Bool? = nil,
            useGradCheckpoint: Bool? = nil,
            kviBits: Int? = nil,
            streamingSink: Int? = nil,
            streamingWindow: Int? = nil,
            galoreRank: Int? = nil,
            galoreUpdateEvery: Int? = nil,
            zLossWeight: Float? = nil,
            useDeepNorm: Bool? = nil,
            lrLayerDecay: Float? = nil,
            useEmbeddingRMSNorm: Bool? = nil
        ) {
            self.kviBits = kviBits
            self.streamingSink = streamingSink
            self.streamingWindow = streamingWindow
            self.galoreRank = galoreRank
            self.galoreUpdateEvery = galoreUpdateEvery
            self.zLossWeight = zLossWeight
            self.useDeepNorm = useDeepNorm
            self.lrLayerDecay = lrLayerDecay
            self.useEmbeddingRMSNorm = useEmbeddingRMSNorm
            self.layers = layers
            self.dModel = dModel
            self.ctx = ctx
            self.heads = heads
            self.dMlp = dMlp
            self.batchSize = batchSize
            self.backend = backend
            self.vocabSize = vocabSize
            self.tokenizerSource = tokenizerSource
            self.nExperts = nExperts
            self.moeTopK = moeTopK
            self.loadBalanceWeight = loadBalanceWeight
            self.slidingWindow = slidingWindow
            self.useMoD = useMoD
            self.useDifferentialAttention = useDifferentialAttention
            self.useYOCO = useYOCO
            self.useGradCheckpoint = useGradCheckpoint
        }
    }

    /// One entry per tensor stored in the file. `name` matches PyTorch
    /// state-dict naming exactly (`token_embedding.weight`, `blocks.0.attn.q_proj.weight`,
    /// `output.weight`, etc.) so loading into `python_ref/model.py` is a direct map.
    public struct TensorEntry: Codable, Sendable, Equatable {
        public var name: String
        public var shape: [Int]
        /// For the fp16 distribution layout (a single contiguous weight buffer
        /// covering all tensors), this is the entry's offset in float units —
        /// `floatOffset * sizeof(dtype)` is the byte offset. Absent in the
        /// training-resumable fp32 layout.
        public var floatOffset: Int?

        public init(name: String, shape: [Int], floatOffset: Int? = nil) {
            self.name = name
            self.shape = shape
            self.floatOffset = floatOffset
        }

        /// Total scalar count = product of shape dimensions.
        public var elementCount: Int { shape.reduce(1, *) }

        /// Byte count for one fp32 instance of this tensor (the weight, or
        /// each of the two AdamW moments). `weight + m + v` = `3 × this`.
        public var byteLengthFP32: Int { elementCount * MemoryLayout<Float32>.size }

        /// Byte count for one fp16 instance. Half of the fp32 size.
        public var byteLengthFP16: Int { elementCount * 2 }

        // Compatibility alias — older callers read `byteLength` assuming fp32.
        public var byteLength: Int { byteLengthFP32 }
    }

    /// Final-loss summary the browser writes at end-of-run. Present in v2 files.
    public struct FinalLoss: Codable, Sendable, Equatable {
        public var step: Int?
        public var train: Double?
        public var val: Double?

        public init(step: Int? = nil, train: Double? = nil, val: Double? = nil) {
            self.step = step
            self.train = train
            self.val = val
        }
    }

    public var config: Config
    public var manifest: [TensorEntry]
    public var savedAt: String?
    public var finalLoss: FinalLoss?
    public var sample: String?
    /// `"fp32"` for training-resumable exports (default when absent),
    /// `"fp16"` for the distributable gallery format. Other values are
    /// accepted but unsupported by the reader.
    public var weightDtype: String?
    /// `true` for the training-resumable layout (per-tensor `[w, m, v]` fp32
    /// triplets), `false` for the inference fp16 layout (contiguous weights).
    /// Absent → assume true for backwards-compat with older browser exports.
    public var includesOptimizerState: Bool?
    /// Total float count across all tensors. Browser writes this for the
    /// fp16 layout; redundant with summed manifest shapes but useful as a
    /// sanity check.
    public var stateByteLength: Int?
    /// Loss history is stored as `{step, trainLoss, valLoss?}` triplets. We
    /// surface as a raw `[[Double?]]`-shaped JSON to avoid coupling our schema
    /// to the browser's exact field names (which have changed once already).
    public var lossHistoryRaw: Data?

    /// Unstructured-pruning sparsity masks, keyed by tensor name. Each
    /// value is RLE-encoded 0/1 bytes (one per weight). Present only
    /// when `tinygpt prune-unstructured` wrote the file. The
    /// inference path multiplies the loaded weights by the decoded
    /// mask — already-zeroed weights stay zero, so the mask is
    /// purely informational at sample time. Its real value is at
    /// distribution time: a high-sparsity model's mask + zeroed
    /// weights compress significantly tighter than the original
    /// dense weights.
    ///
    /// Wire format: a JSON object `{tensorName: base64String, ...}`
    /// where each base64String decodes to the RLE bytes per
    /// `Pruning.encodeRLE`. Absent on standard checkpoints.
    public var sparsityMasks: [String: String]?

    /// Pruning-recipe metadata: original sparsity, iterations, etc.
    /// Travels alongside `sparsityMasks` for inspectability. The
    /// loader doesn't need this — it's diagnostic only.
    public var pruningInfo: PruningInfo?

    public struct PruningInfo: Codable, Sendable, Equatable {
        public var kind: String          // "unstructured" | "structured-head" | "structured-layer"
        public var sparsity: Float?      // unstructured: fraction zeroed
        public var iterations: Int?      // unstructured: IMP rounds
        public var headsDropped: [Int]?  // structured-head: which heads (per layer or globally)
        public var layersDropped: [Int]? // structured-layer: which layer indices were removed

        public init(kind: String, sparsity: Float? = nil, iterations: Int? = nil,
                    headsDropped: [Int]? = nil, layersDropped: [Int]? = nil) {
            self.kind = kind
            self.sparsity = sparsity
            self.iterations = iterations
            self.headsDropped = headsDropped
            self.layersDropped = layersDropped
        }
    }

    public init(
        config: Config,
        manifest: [TensorEntry],
        savedAt: String? = nil,
        finalLoss: FinalLoss? = nil,
        sample: String? = nil,
        weightDtype: String? = nil,
        includesOptimizerState: Bool? = nil,
        stateByteLength: Int? = nil,
        lossHistoryRaw: Data? = nil,
        sparsityMasks: [String: String]? = nil,
        pruningInfo: PruningInfo? = nil
    ) {
        self.config = config
        self.manifest = manifest
        self.savedAt = savedAt
        self.finalLoss = finalLoss
        self.sample = sample
        self.weightDtype = weightDtype
        self.includesOptimizerState = includesOptimizerState
        self.stateByteLength = stateByteLength
        self.lossHistoryRaw = lossHistoryRaw
        self.sparsityMasks = sparsityMasks
        self.pruningInfo = pruningInfo
    }

    // Manual decode/encode so we can pass `lossHistory` through as raw JSON
    // and so missing keys decode as nil rather than throwing.
    private enum CodingKeys: String, CodingKey {
        case config, manifest, savedAt, finalLoss, sample
        case weightDtype, includesOptimizerState, stateByteLength
        case lossHistory
        case sparsityMasks, pruningInfo
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.config = (try? c.decode(Config.self, forKey: .config)) ?? Config()
        self.manifest = try c.decode([TensorEntry].self, forKey: .manifest)
        self.savedAt = try c.decodeIfPresent(String.self, forKey: .savedAt)
        self.finalLoss = try c.decodeIfPresent(FinalLoss.self, forKey: .finalLoss)
        self.sample = try c.decodeIfPresent(String.self, forKey: .sample)
        self.weightDtype = try c.decodeIfPresent(String.self, forKey: .weightDtype)
        self.includesOptimizerState = try c.decodeIfPresent(Bool.self, forKey: .includesOptimizerState)
        self.stateByteLength = try c.decodeIfPresent(Int.self, forKey: .stateByteLength)
        if c.contains(.lossHistory),
           let raw = try? c.decode(AnyJSON.self, forKey: .lossHistory) {
            self.lossHistoryRaw = try? JSONEncoder().encode(raw)
        } else {
            self.lossHistoryRaw = nil
        }
        self.sparsityMasks = try c.decodeIfPresent([String: String].self, forKey: .sparsityMasks)
        self.pruningInfo = try c.decodeIfPresent(PruningInfo.self, forKey: .pruningInfo)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(config, forKey: .config)
        try c.encode(manifest, forKey: .manifest)
        try c.encodeIfPresent(savedAt, forKey: .savedAt)
        try c.encodeIfPresent(finalLoss, forKey: .finalLoss)
        try c.encodeIfPresent(sample, forKey: .sample)
        try c.encodeIfPresent(weightDtype, forKey: .weightDtype)
        try c.encodeIfPresent(includesOptimizerState, forKey: .includesOptimizerState)
        try c.encodeIfPresent(stateByteLength, forKey: .stateByteLength)
        if let raw = lossHistoryRaw,
           let value = try? JSONDecoder().decode(AnyJSON.self, from: raw) {
            try c.encode(value, forKey: .lossHistory)
        }
        try c.encodeIfPresent(sparsityMasks, forKey: .sparsityMasks)
        try c.encodeIfPresent(pruningInfo, forKey: .pruningInfo)
    }

    /// Convenience: which on-disk body layout the file uses.
    public var bodyLayout: TinyGPTBodyLayout {
        let dtype = weightDtype?.lowercased() ?? "fp32"
        let hasOptim = includesOptimizerState ?? true
        if dtype == "fp16" || !hasOptim { return .inferenceFP16 }
        return .trainingFP32
    }
}

/// Which body layout the `.tinygpt` file uses.
public enum TinyGPTBodyLayout: Sendable {
    /// Per-tensor `[weight_fp32, adam_m_fp32, adam_v_fp32]` triplets. Default
    /// for browser exports — train-continue ready.
    case trainingFP32
    /// Single contiguous fp16 weight buffer indexed by each manifest entry's
    /// `floatOffset`. No optimiser state. Used for gallery distribution.
    case inferenceFP16
}

/// Pass-through container for arbitrary JSON values. Used to preserve the
/// browser's `lossHistory` field without coupling our types to its exact shape.
enum AnyJSON: Codable, Sendable, Equatable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([AnyJSON])
    case object([String: AnyJSON])

    init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null; return }
        if let v = try? c.decode(Bool.self) { self = .bool(v); return }
        if let v = try? c.decode(Double.self) { self = .number(v); return }
        if let v = try? c.decode(String.self) { self = .string(v); return }
        if let v = try? c.decode([AnyJSON].self) { self = .array(v); return }
        if let v = try? c.decode([String: AnyJSON].self) { self = .object(v); return }
        throw DecodingError.dataCorruptedError(
            in: c, debugDescription: "Unrecognised JSON value"
        )
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let v): try c.encode(v)
        case .number(let v): try c.encode(v)
        case .string(let v): try c.encode(v)
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        }
    }
}
