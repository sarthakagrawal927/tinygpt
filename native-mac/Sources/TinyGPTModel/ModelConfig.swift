import Foundation
import MLX

/// Mirror of `python_ref/model.py`'s `ModelConfig` dataclass. Source of truth
/// for shapes; both browser and Mac builds round-trip through this.
public struct ModelConfig: Sendable, Equatable {
    public var modelName: String
    public var vocabSize: Int
    public var contextLength: Int
    public var nLayers: Int
    public var nHeads: Int
    /// Number of K/V heads. Defaults to `nHeads` (= standard multi-head
    /// attention). Set to less than `nHeads` for Grouped Query Attention
    /// (e.g., Llama-3-8B: nHeads=32, nKvHeads=8). Must divide nHeads.
    public var nKvHeads: Int
    public var dModel: Int
    public var dMlp: Int
    public var dropout: Float
    public var tieEmbeddings: Bool
    /// `"float32"` or `"float16"`. Default is float32 for training parity.
    public var dtype: String
    /// Use RoPE rotary embeddings instead of learned absolute positional
    /// embeddings. Required to load HF models (Llama, Mistral, Phi, etc.).
    /// When true, the model SKIPS the position_embedding lookup and
    /// instead has CausalSelfAttention rotate Q, K by the position.
    public var useRoPE: Bool
    /// RoPE base frequency. Standard is 10000; Llama-3 uses 500000 for
    /// extended context.
    public var ropeBase: Float
    /// Use RMSNorm instead of LayerNorm. HF models almost universally
    /// use RMSNorm.
    public var useRMSNorm: Bool
    /// Use SwiGLU MLP instead of plain GELU. HF models almost
    /// universally use SwiGLU.
    public var useSwiGLU: Bool
    /// Attention has bias terms. PyTorch's nn.Linear has bias=True by
    /// default; HF Llama-style models set bias=False to save params and
    /// improve training stability.
    public var attnBias: Bool
    /// HuggingFace model directory whose `tokenizer.json` the model was
    /// trained against (BPE / SentencePiece). `nil` means byte-level
    /// tokenization (vocabSize == 256). Travels with the checkpoint so
    /// sample/finetune can re-load the matching tokenizer.
    public var tokenizerSource: String?

    /// Mixture-of-Experts settings. `nExperts > 1` swaps the standard MLP
    /// for an MoE MLP at every block — a router picks the top-`moeTopK`
    /// experts per token. `loadBalanceWeight` scales the auxiliary load-
    /// balance loss that prevents the router from collapsing onto a
    /// single expert (Switch Transformer recipe: 0.01).
    ///
    /// The first-cut implementation uses DENSE compute (every expert sees
    /// every token, weighted by the router) — the architecture is correct
    /// and trains, but per-token FLOPs don't drop until we ship a sparse
    /// scatter-gather kernel. The parameter-count benefit (more total
    /// capacity per byte of weights loaded) is immediate.
    public var nExperts: Int
    public var moeTopK: Int
    public var loadBalanceWeight: Float
    public var isMoE: Bool { nExperts > 1 }

    /// Multi-Token Prediction horizon count (DeepSeek-V3 / Gloeckle et al.,
    /// 2024). When `mtpHorizons > 1`, the model gets `mtpHorizons - 1`
    /// extra output heads — each one predicts further ahead (h tokens
    /// rather than 1) from the SAME final hidden state. Loss is the mean
    /// across all H horizon CEs. The MTP heads are TRAINING-ONLY: they
    /// aren't serialised, so a saved checkpoint stays drop-in compatible
    /// with the standard sample path. The regulariser usually drops
    /// per-token validation loss by 10-20% on small models — capturing
    /// "what's the local 2-token continuation" disambiguates a noisier
    /// signal than 1-token-ahead alone.
    public var mtpHorizons: Int

    /// Sliding-window attention size. `nil` (default) = standard full
    /// causal attention. When set to W, attention is allowed only on
    /// the last W positions (including self) — i.e. the model can't
    /// look further back than W tokens. Mistral / GPT-OSS standard.
    /// Cuts attention compute from O(T²) to O(T·W) at long context,
    /// and bounds the KV cache at decoding time.
    public var slidingWindow: Int?

    /// ALiBi position bias (Press et al., 2021). Adds a per-head
    /// linear penalty `-slope[h] · (i - j)` to attention scores —
    /// the model "naturally" learns to attend to nearer positions
    /// more strongly, without needing RoPE or learned positional
    /// embeddings. Extrapolates to longer contexts than train. When
    /// `useALiBi` is set, the model SHOULDN'T also use RoPE or learned
    /// positional embeddings; the runner disables them upstream.
    public var useALiBi: Bool

    /// Mixture-of-Depths (Raposo et al., 2024). Each TransformerBlock
    /// gets a per-token router that learns to scale the block's
    /// contribution — tokens the router deems irrelevant pass
    /// through unchanged. This implementation uses SOFT routing
    /// (sigmoid gate, no top-K + STE) so it's trainable end-to-end
    /// without specialised infrastructure. The compute saving of
    /// hard top-K MoD is left to the sparse-dispatch follow-up
    /// (same scatter_add blocker as MoE).
    public var useMoD: Bool

    /// Differential attention (Ye et al., 2024). Each block's attention
    /// has TWO Q/K projections instead of one; outputs are subtracted
    /// to cancel correlated noise. ~1.5-2× per-head compute, often
    /// improves long-context reasoning. Mutually exclusive with the
    /// standard attention path.
    public var useDifferentialAttention: Bool

    /// YOCO — "You Only Cache Once" (Lin et al., 2024). The model is
    /// split in two halves: first half runs standard self-attention,
    /// the LAST first-half layer's K, V are captured as an "anchor",
    /// and every second-half layer skips its own K, V computation —
    /// instead doing cross-attention onto the anchor's K, V. Roughly
    /// halves the KV cache at long-context decode. kProj/vProj
    /// weights ARE still allocated in second-half layers (kept for
    /// manifest-compatibility) but go unused at forward time.
    public var useYOCO: Bool

    /// Gradient checkpointing (a.k.a. activation checkpointing). Trades
    /// ~30% extra compute per training step for a ~√L reduction in
    /// activation memory (where L is the number of transformer layers).
    /// Each TransformerBlock's forward is wrapped in an MLX
    /// `CustomFunction` whose VJP RE-RUNS the block forward at backward
    /// time — so the intermediate activations (K/V projections, MLP
    /// inner state, residual stream) for that block are not retained
    /// across the backward of OTHER blocks. Training-only knob: the
    /// flag travels in the .tinygpt header so a `--resume` keeps the
    /// same memory profile, but inference/sample paths never read it.
    public var useGradCheckpoint: Bool

    public var headDim: Int { dModel / nHeads }

    public var mlxDType: DType {
        switch dtype.lowercased() {
        case "float16", "fp16", "half": return .float16
        case "bfloat16", "bf16": return .bfloat16
        default: return .float32
        }
    }

    public init(
        modelName: String = "byte-tinygpt-v0",
        vocabSize: Int = 256,
        contextLength: Int = 128,
        nLayers: Int = 4,
        nHeads: Int = 4,
        nKvHeads: Int? = nil,
        dModel: Int = 128,
        dMlp: Int = 512,
        dropout: Float = 0.0,
        tieEmbeddings: Bool = true,
        dtype: String = "float32",
        useRoPE: Bool = false,
        ropeBase: Float = 10_000,
        useRMSNorm: Bool = false,
        useSwiGLU: Bool = false,
        attnBias: Bool = true,
        tokenizerSource: String? = nil,
        nExperts: Int = 1,
        moeTopK: Int = 1,
        loadBalanceWeight: Float = 0.01,
        mtpHorizons: Int = 1,
        slidingWindow: Int? = nil,
        useALiBi: Bool = false,
        useMoD: Bool = false,
        useDifferentialAttention: Bool = false,
        useYOCO: Bool = false,
        useGradCheckpoint: Bool = false
    ) {
        self.tokenizerSource = tokenizerSource
        self.nExperts = max(1, nExperts)
        self.moeTopK = max(1, min(moeTopK, max(1, nExperts)))
        self.loadBalanceWeight = loadBalanceWeight
        self.mtpHorizons = max(1, mtpHorizons)
        self.slidingWindow = slidingWindow.flatMap { $0 > 0 ? $0 : nil }
        self.useALiBi = useALiBi
        self.useMoD = useMoD
        self.useDifferentialAttention = useDifferentialAttention
        self.useYOCO = useYOCO
        self.useGradCheckpoint = useGradCheckpoint
        self.modelName = modelName
        self.vocabSize = vocabSize
        self.contextLength = contextLength
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nKvHeads = nKvHeads ?? nHeads
        self.dModel = dModel
        self.dMlp = dMlp
        self.dropout = dropout
        self.tieEmbeddings = tieEmbeddings
        self.dtype = dtype
        self.useRoPE = useRoPE
        self.ropeBase = ropeBase
        self.useRMSNorm = useRMSNorm
        self.useSwiGLU = useSwiGLU
        self.attnBias = attnBias
        precondition(dModel % nHeads == 0, "d_model must be divisible by n_heads")
        precondition(nHeads % self.nKvHeads == 0,
                     "n_heads (\(nHeads)) must be divisible by n_kv_heads (\(self.nKvHeads)) for GQA")
    }

    /// Match the browser's "Huge" preset (12L, d=256, ctx=256, 8 heads, dMlp=1024).
    /// This is the gallery model size — the apples-to-apples comparison target.
    public static let huge = ModelConfig(
        modelName: "byte-tinygpt-huge",
        vocabSize: 256,
        contextLength: 256,
        nLayers: 12,
        nHeads: 8,
        dModel: 256,
        dMlp: 1024,
        dropout: 0.0,
        tieEmbeddings: true
    )

    /// Mega preset (24L, d=512, ctx=1024). Browser can't run this — Mac can.
    /// Context lifted from 512→1024 for Tier 1: long-context BPE training
    /// where dense tokens make 1024 BPE-tokens ~ 4 KB of source text per
    /// window, comparable to a useful paragraph rather than a sentence.
    public static let mega = ModelConfig(
        modelName: "byte-tinygpt-mega",
        vocabSize: 256,
        contextLength: 1024,
        nLayers: 24,
        nHeads: 8,
        dModel: 512,
        dMlp: 2048,
        dropout: 0.0,
        tieEmbeddings: true
    )

    /// Behemoth preset (32L, d=1024, ctx=1024, ~400M params). Pushes the
    /// M5 Pro's 48 GB unified memory hard — fp32 training fits with B=2,
    /// fp16 fits with B=4. Browser absolutely cannot run this.
    public static let behemoth = ModelConfig(
        modelName: "byte-tinygpt-behemoth",
        vocabSize: 256,
        contextLength: 1024,
        nLayers: 32,
        nHeads: 16,
        dModel: 1024,
        dMlp: 4096,
        dropout: 0.0,
        tieEmbeddings: true
    )

    /// Titan preset (48L, d=1536, ctx=1024, ~1.3B params). Most ambitious
    /// preset that still fits on the M5 Pro / 48 GB for fp16 inference;
    /// fp32 training requires careful batching.
    public static let titan = ModelConfig(
        modelName: "byte-tinygpt-titan",
        vocabSize: 256,
        contextLength: 1024,
        nLayers: 48,
        nHeads: 24,
        dModel: 1536,
        dMlp: 6144,
        dropout: 0.0,
        tieEmbeddings: true
    )
}
