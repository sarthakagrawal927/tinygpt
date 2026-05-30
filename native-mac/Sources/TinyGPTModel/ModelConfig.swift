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

    /// Default KIVI precision for sample-time KV cache. `nil` = sample
    /// uses fp32/fp16/bf16 KV cache (the historical default). Set to 4
    /// or 8 to recommend KIVI quantisation in the checkpoint manifest;
    /// `tinygpt sample --kv-quantize ...` overrides. Inference-time
    /// hint only — has no effect on training.
    public var kviBits: Int?

    /// Default StreamingLLM sink + window for sample-time KV cache.
    /// `nil` = unbounded growth (historical default). Inference-time
    /// hint only; `tinygpt sample` flags override.
    public var streamingSink: Int?
    public var streamingWindow: Int?

    /// Speculative-decode head configuration (Medusa / EAGLE-2). When a
    /// `.heads` sidecar is loaded at sample time this carries the head
    /// architecture so the verify path can rebuild it. The fields live
    /// only in memory — the .tinygpt manifest stays untouched (heads
    /// are a SIDECAR, not baked into the base checkpoint), so older
    /// readers round-trip unchanged. See `MedusaHeads.swift` /
    /// `EagleDraft.swift` for the head modules themselves.
    public struct SpeculativeHeadConfig: Sendable, Equatable {
        public enum Kind: String, Sendable, Equatable { case medusa, eagle }
        public var kind: Kind
        /// Number of look-ahead steps the heads cover. Medusa: N independent
        /// heads predicting offsets 1..N. EAGLE-2: 1 auto-regressive draft
        /// net unrolled N times.
        public var numHeads: Int
        /// Hidden width for the head's internal projection. Defaults to
        /// `dModel` so the head's residual block stays well-sized.
        public var hiddenDim: Int
        public init(kind: Kind, numHeads: Int, hiddenDim: Int) {
            self.kind = kind
            self.numHeads = max(1, numHeads)
            self.hiddenDim = max(8, hiddenDim)
        }
    }

    // MARK: - Training-stability bells (Tier 2)
    //
    // The five knobs below are TRAINING-TIME features; they don't
    // change the saved parameter layout. They survive a save/load
    // round-trip via the manifest so a `--resume` continues with
    // the same regimen, but they don't appear as tensors.

    /// GaLore rank — when > 0, the trainer wraps each 2-D weight
    /// matrix's gradient in a rank-R projection before the optimiser
    /// step (Zhao et al., 2024). `nil` / 0 = disabled.
    public var galoreRank: Int?

    /// GaLore basis refresh cadence (steps). nil = default 200.
    public var galoreUpdateEvery: Int?

    /// Z-loss weight (PaLM / GShard). When > 0, the loss adds
    /// `weight · (log Σ exp(logit))^2` so logits don't drift to
    /// magnitudes that destabilise the softmax. 1e-4 is the PaLM
    /// recipe.
    public var zLossWeight: Float

    /// DeepNorm residual scaling flag (Wang et al., 2022). When set,
    /// the residual path scales by α = (2L)^(1/4) and specific
    /// projections init by β = (8L)^(-1/4). Stabilises training of
    /// VERY deep (>100 layer) transformers. Visible at init time
    /// only — no runtime change once weights are trained.
    public var useDeepNorm: Bool

    /// Layer-wise LR decay factor. When `< 1.0`, each block's
    /// gradient is scaled by `factor^(nLayers - 1 - i)` so shallow
    /// blocks update slower than deep blocks. Standard fine-tuning
    /// lever; 0.85-0.95 typical.
    public var lrLayerDecay: Float

    /// Apply an RMSNorm right after the token embedding lookup.
    /// Recent (2025) papers show this stabilises early-training
    /// loss on long-context transformers. Adds one `d_model`-shaped
    /// weight to the model — small, but it DOES land in the manifest
    /// when set, so a checkpoint trained without it can't be resumed
    /// with it on (the manifest entries would mismatch).
    public var useEmbeddingRMSNorm: Bool

    public var headDim: Int { dModel / nHeads }

    /// DeepNorm α — multiplier on the *residual* (the running x) at
    /// every sub-layer (Wang et al., 2022). For an encoder-only or
    /// decoder-only stack of N layers, α = (2N)^(1/4). Off-by-default;
    /// active when `useDeepNorm == true`.
    public var deepNormAlpha: Float {
        useDeepNorm ? Foundation.pow(2.0 * Float(nLayers), 0.25) : 1.0
    }

    /// DeepNorm β — multiplier on the INIT of certain projections
    /// (v_proj, o_proj, and the MLP down-projection). For a decoder-
    /// only stack, β = (8N)^(-1/4). The init pulls the variance of
    /// the residual stream contributions back into the right range
    /// to balance α. Inactive when `useDeepNorm == false`.
    public var deepNormBeta: Float {
        useDeepNorm ? Foundation.pow(8.0 * Float(nLayers), -0.25) : 1.0
    }

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
        useGradCheckpoint: Bool = false,
        kviBits: Int? = nil,
        streamingSink: Int? = nil,
        streamingWindow: Int? = nil,
        galoreRank: Int? = nil,
        galoreUpdateEvery: Int? = nil,
        zLossWeight: Float = 0,
        useDeepNorm: Bool = false,
        lrLayerDecay: Float = 1.0,
        useEmbeddingRMSNorm: Bool = false
    ) {
        self.kviBits = kviBits
        self.streamingSink = streamingSink
        self.streamingWindow = streamingWindow
        self.galoreRank = (galoreRank ?? 0) > 0 ? galoreRank : nil
        self.galoreUpdateEvery = galoreUpdateEvery
        self.zLossWeight = max(0, zLossWeight)
        self.useDeepNorm = useDeepNorm
        self.lrLayerDecay = lrLayerDecay
        self.useEmbeddingRMSNorm = useEmbeddingRMSNorm
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
