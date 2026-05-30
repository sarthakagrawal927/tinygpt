import Foundation
import MLX
import MLXNN
import MLXRandom

/// EAGLE-2 draft network (Li et al., 2024 — "EAGLE-2: Faster Inference of
/// Language Models with Dynamic Draft Trees").
///
/// In the paper, EAGLE replaces the Medusa "N independent heads" design
/// with a small AUTO-REGRESSIVE draft net that takes the base's hidden
/// state as input (rather than just the embedding) and unrolls forward
/// for `numHeads` steps. The intuition is that the base's hidden state
/// carries far richer next-token information than a token-id embedding
/// alone — so a tiny draft net conditioned on it can predict ahead
/// much more accurately than N parallel projections from a single
/// frozen hidden.
///
/// EAGLE-2's full architecture is a 1-layer transformer (norm + attn +
/// norm + MLP) that maintains its own KV state across the unroll. **This
/// first cut implements a deliberately simplified draft network** to
/// keep the surface area honest:
///
///   step k input:  concat(hidden_k, token_embed(t_k))  →  [d + d]
///   step k body:   Linear(2d → d) → SiLU → Linear(d → d) → SiLU → residual
///                  (hidden_k+1 = body + hidden_k)
///   step k output: base_lm_head(hidden_k+1)  → vocab logits
///
/// `hidden_0` is the base's last-position hidden state for the input
/// sequence; `t_0` is the base's argmax (i.e. EAGLE always commits the
/// base's own first token, then drafts from there). Each subsequent
/// step uses the previous step's predicted hidden + its argmax token.
///
/// **Differences from real EAGLE-2 (acknowledged):**
///   - No attention layer inside the draft net (we use MLP only) — full
///     paper uses 1 transformer block with self-attention onto past
///     draft states. Cost: lower acceptance rate, especially for the
///     longer draft tails (head 3+).
///   - No dynamic tree pruning. Real EAGLE-2 expands a TREE of draft
///     candidates and prunes branches by a confidence prior; we draft
///     a single linear chain (tree width 1). Trade: simpler verify path,
///     gives up the multi-path acceptance boost.
///   - The base LM head is RE-USED (tied) for vocab projection — this
///     matches EAGLE's "we use the base's lm_head; only the draft
///     transformer is trained" philosophy, but we don't currently
///     expose it as the literal base.lmHead reference — we copy weights
///     into a sibling Linear at load time. Functionally equivalent
///     when the base is frozen; loses ability to fine-tune the head.
///
/// Training: same surface as Medusa. Base frozen, only the draft net
/// updates. Loss is the per-step CE between draft logits at offset k
/// and `targets[:, k:]` (the right shifted ground truth).
///
/// Verification reuses Medusa's linear (width-1) verify path —
/// `MedusaVerify.step(…)` analogue specialised for the auto-regressive
/// draft. The mismatch-then-accept-base rule preserves greedy
/// correctness wrt the base's argmax, regardless of the draft net's
/// quality.

// MARK: - Draft network

/// The EAGLE-2 draft net itself. Takes hidden_t + embed(t_token) and
/// produces a candidate next-step hidden. Tied vocab projection via a
/// sibling Linear that the loader fills in from the base LM head.
public final class EagleDraft: Module {
    @ModuleInfo(key: "in_proj") public var inProj: Linear
    @ModuleInfo(key: "hidden_proj") public var hiddenProj: Linear
    @ModuleInfo(key: "out_norm") public var outNorm: LayerNorm
    /// Vocab projection — initialised from the base LM head's weights at
    /// load time. Owning a copy (rather than referencing
    /// `model.lmHead`) keeps the draft net's `parameters()` tree clean
    /// for training (we only need to autograd against THIS module). The
    /// copy is frozen after init in the training loop via the
    /// gradient-mask trick used elsewhere.
    @ModuleInfo(key: "vocab_proj") public var vocabProj: Linear
    /// Token embedding: same shape as the base's token embedding. We
    /// copy the base's embedding weights at load time and freeze them
    /// — same logic as `vocab_proj`.
    @ModuleInfo(key: "token_embed") public var tokenEmbed: Embedding
    public let dModel: Int
    public let vocabSize: Int
    public let numHeads: Int   // how many auto-regressive draft steps

    public init(dModel: Int, vocabSize: Int, numHeads: Int) {
        self.dModel = dModel
        self.vocabSize = vocabSize
        self.numHeads = max(1, numHeads)
        self._inProj.wrappedValue = Linear(dModel * 2, dModel, bias: true)
        self._hiddenProj.wrappedValue = Linear(dModel, dModel, bias: true)
        self._outNorm.wrappedValue = LayerNorm(dimensions: dModel, eps: 1e-5)
        self._vocabProj.wrappedValue = Linear(dModel, vocabSize, bias: false)
        self._tokenEmbed.wrappedValue = Embedding(
            embeddingCount: vocabSize, dimensions: dModel
        )
        super.init()
    }

    /// One draft step: given (hidden, prevTokenId) → (nextHidden, nextLogits).
    ///
    /// `hidden`: `[B, T, d]` — typically `[B, 1, d]` at inference.
    /// `prevTokenId`: `[B, T]` int — token whose embedding we splice in.
    public func step(hidden: MLXArray, prevTokenId: MLXArray) -> (MLXArray, MLXArray) {
        let tEmb = tokenEmbed(prevTokenId)               // [B, T, d]
        let combined = concatenated([hidden, tEmb], axis: -1)  // [B, T, 2d]
        let h1 = silu(inProj(combined))                  // [B, T, d]
        let h2 = silu(hiddenProj(h1))                    // [B, T, d]
        let next = outNorm(h2 + hidden)                  // residual + norm
        let logits = vocabProj(next)                     // [B, T, vocab]
        return (next, logits)
    }

    public func numParameters() -> Int {
        var total = 0
        for (_, p) in parameters().flattened() {
            total += p.shape.reduce(1, *)
        }
        return total
    }
}

// MARK: - Training-time forward

/// Run the EAGLE draft net auto-regressively at every position of a
/// `[B, T, d]` hidden-state sequence, "teacher-forced" on the ground-
/// truth tokens. Returns a list of `[B, T, vocab]` logits arrays —
/// one per draft step k ∈ [0, numHeads), where step k predicts the
/// token at offset (k+1) from each position.
///
/// Teacher-forcing keeps training cheap and stable: each step's hidden
/// input is the BASE's hidden at that position (not the draft's
/// previously predicted hidden). Inference-time the draft IS the only
/// source of hidden states from step 2 onward, so there's a small
/// training/inference gap — same gap that vanilla auto-regressive
/// LM training has.
public func eagleTrainingForward(
    draft: EagleDraft,
    baseHidden: MLXArray,    // [B, T, d]
    tokens: MLXArray         // [B, T] int — input token ids (the position-t base inputs)
) -> [MLXArray] {
    var out: [MLXArray] = []
    out.reserveCapacity(draft.numHeads)
    // Step k=0: use base hidden as input, token = `tokens[t]` (the token
    // AT position t, which the head's logits should predict the t+1 token
    // for — same convention as the rest of the codebase).
    let T = tokens.shape[1]
    // First step: draft uses (baseHidden_t, embed(tokens_t)) → predicts t+1.
    var hidden = baseHidden
    var tokenIn = tokens
    let (h0, logits0) = draft.step(hidden: hidden, prevTokenId: tokenIn)
    out.append(logits0)
    hidden = h0
    // Subsequent steps: predicted hidden + argmax token (teacher-forced).
    // We could either teacher-force on `tokens[:, k:]` (true ground truth)
    // or use the draft's own argmax. We use the SHIFTED ground truth to
    // avoid scale-bias drift early in training — the draft's argmax
    // is near-random at step 0, which makes step 1's hidden meaningless.
    for k in 1..<draft.numHeads {
        if k >= T { break }
        // Slide tokens left by 1 — pad the rightmost position with the
        // input's last token (it gets masked out by the CE shift anyway).
        // Implementation: use tokens shifted left by k.
        let valid = T - k
        let shiftedTokens = tokens[0..., k..<T]    // [B, valid]
        let h = hidden[0..., 0..<valid, 0...]      // [B, valid, d]
        let (hNext, logitsK) = draft.step(hidden: h, prevTokenId: shiftedTokens)
        // Pad logitsK to full T on the right with zeros so the loss
        // function below can slice it uniformly. Zero rows are masked out
        // by the same valid-window trick used in medusaHeadsLoss.
        let padShape = [logitsK.shape[0], T - valid, logitsK.shape[2]]
        let pad = MLXArray.zeros(padShape).asType(logitsK.dtype)
        let logitsFull = concatenated([logitsK, pad], axis: 1)
        out.append(logitsFull)
        // Same padding for hidden.
        let hidPad = MLXArray.zeros([hNext.shape[0], T - valid, hNext.shape[2]]).asType(hNext.dtype)
        hidden = concatenated([hNext, hidPad], axis: 1)
        _ = tokenIn
    }
    return out
}

/// Per-step next-token CE, identical loss shape to `medusaHeadsLoss`.
/// Step k's target is `targets[:, k:]`. The last k positions of the k-th
/// logits tensor are unscored (we pad them above so the slicing works).
public func eagleDraftLoss(stepLogits: [MLXArray], targets: MLXArray) -> MLXArray {
    precondition(!stepLogits.isEmpty, "eagleDraftLoss needs ≥1 step")
    let T = targets.shape[1]
    var total = MLXArray(Float(0))
    var scored = 0
    for (k, logitsK) in stepLogits.enumerated() {
        let valid = T - k
        if valid <= 0 { continue }
        let logitsSlice = logitsK[0..., 0..<valid, 0...]
        let targetsSlice = targets[0..., k..<T]
        let v = logitsSlice.shape.last!
        let ce = crossEntropy(
            logits: logitsSlice.reshaped([-1, v]),
            targets: targetsSlice.reshaped([-1]),
            reduction: .mean
        )
        total = total + ce
        scored += 1
    }
    return total / MLXArray(Float(max(1, scored)))
}

// MARK: - Inference-time auto-regressive draft

/// One EAGLE-style speculative step.
///
/// Cost model is identical to Medusa: 2 base forwards per step. The
/// draft net adds a tiny per-step cost (one MLP forward at batch=1,
/// negligible vs. the base). The expected speedup distribution
/// reflects EAGLE-2's published acceptance-rate gains over Medusa: a
/// well-trained draft net beats independent heads on the longer
/// draft tail (head 3+). At 50 training steps, neither will look good.
public enum EagleVerify {
    public static func step(
        baseHidden: (MLXArray) -> MLXArray,
        baseLogits: (MLXArray) -> MLXArray,
        baseLMHead: (MLXArray) -> MLXArray,
        draft: EagleDraft,
        ids: inout [Int],
        ctxCap: Int
    ) -> SpecHeadsStepResult {
        // 1. Run base on `ids` to get the last hidden + base's argmax.
        let tail = ids.suffix(ctxCap)
        let arr = MLXArray(tail.map { Int32($0) }, [1, tail.count])
        let hiddenAll = baseHidden(arr)
        let lastHidden = hiddenAll[0..., hiddenAll.shape[1] - 1 ..< hiddenAll.shape[1], 0...] // [1, 1, d]
        let baseLast = baseLMHead(lastHidden)            // [1, 1, vocab]
        let baseArg = argMax(baseLast[0..., 0, 0...], axis: -1).reshaped([1])
        eval(baseArg)
        var proposals: [Int] = [Int(baseArg.item(Int32.self))]

        // 2. Auto-regressively unroll the draft net for numHeads steps.
        //    Start with (lastHidden, baseArg) → (nextHidden, nextToken).
        var prevHidden = lastHidden
        var prevTok = baseArg.reshaped([1, 1])
        for _ in 0..<draft.numHeads {
            let (h, l) = draft.step(hidden: prevHidden, prevTokenId: prevTok)
            let argK = argMax(l[0..., 0, 0...], axis: -1).reshaped([1])
            eval(argK)
            let tok = Int(argK.item(Int32.self))
            proposals.append(tok)
            prevHidden = h
            prevTok = MLXArray([Int32(tok)], [1, 1])
        }
        // proposals: [baseArg, draftArg1, draftArg2, ..., draftArgN] — length N+1

        // 3. Verify in ONE base forward — same accept rule as Medusa.
        let withProposals = (ids + proposals).suffix(ctxCap)
        let inputArr = MLXArray(withProposals.map { Int32($0) }, [1, withProposals.count])
        let tLogits = baseLogits(inputArr)
        let promptLen = withProposals.count - proposals.count
        var accepted: [Int] = []
        var acceptedProposals = 0
        let totalProposals = proposals.count
        for i in 0..<proposals.count {
            let pos = promptLen - 1 + i
            let row = tLogits[0..., pos, 0...]
            let argT = argMax(row, axis: -1).reshaped([1])
            eval(argT)
            let tTok = Int(argT.item(Int32.self))
            if tTok == proposals[i] {
                accepted.append(tTok)
                acceptedProposals += 1
            } else {
                accepted.append(tTok)
                ids.append(contentsOf: accepted)
                return SpecHeadsStepResult(
                    acceptedIds: accepted,
                    proposalsAccepted: acceptedProposals,
                    proposalsTotal: totalProposals
                )
            }
        }
        let bonusPos = promptLen - 1 + proposals.count
        if bonusPos < tLogits.shape[1] {
            let bonusRow = tLogits[0..., bonusPos, 0...]
            let bonusArg = argMax(bonusRow, axis: -1).reshaped([1])
            eval(bonusArg)
            accepted.append(Int(bonusArg.item(Int32.self)))
        }
        ids.append(contentsOf: accepted)
        return SpecHeadsStepResult(
            acceptedIds: accepted,
            proposalsAccepted: acceptedProposals,
            proposalsTotal: totalProposals
        )
    }
}

// MARK: - Serialisation

/// File format reuses `SpecHeadsFileHeader` / `SpecHeadsFormat` from
/// `MedusaHeads.swift` with `kind == "eagle"`. Entries are the draft
/// net's flattened parameter names + shapes; on-disk bytes are the fp32
/// blobs in `parameters().flattened()` order.
public enum EagleDraftIO {
    public static func write(
        draft: EagleDraft,
        baseConfig: ModelConfig,
        hiddenDim: Int,
        finalLoss: Float?,
        to url: URL
    ) throws {
        let params = draft.parameters().flattened()
        var entries: [SpecHeadsTensorEntry] = []
        var blobs: [Data] = []
        for (name, p) in params {
            eval(p)
            let floats: [Float] = p.asArray(Float.self)
            let bytes = floats.withUnsafeBufferPointer { Data(buffer: $0) }
            entries.append(.init(name: name, shape: p.shape))
            blobs.append(bytes)
        }
        let header = SpecHeadsFileHeader(
            kind: "eagle",
            numHeads: draft.numHeads,
            hiddenDim: hiddenDim,
            dModel: draft.dModel,
            vocabSize: draft.vocabSize,
            baseLayers: baseConfig.nLayers,
            baseDModel: baseConfig.dModel,
            baseHeads: baseConfig.nHeads,
            baseCtx: baseConfig.contextLength,
            savedAt: ISO8601DateFormatter().string(from: Date()),
            finalLoss: finalLoss,
            entries: entries
        )
        try writeHeaderAndBlobs(header: header, blobs: blobs, to: url)
    }

    public static func read(_ url: URL, baseConfig: ModelConfig) throws -> EagleDraft {
        let (header, blobs) = try readHeaderAndBlobs(url)
        guard header.kind == "eagle" else {
            throw NSError(domain: "TinyGPTHeads", code: 5,
                          userInfo: [NSLocalizedDescriptionKey:
                            "expected kind=eagle, got \(header.kind)"])
        }
        try validateBaseConfig(header: header, baseConfig: baseConfig)
        let draft = EagleDraft(dModel: header.dModel,
                                vocabSize: header.vocabSize,
                                numHeads: header.numHeads)
        try restoreParameters(into: draft, header: header, blobs: blobs)
        return draft
    }
}

// MARK: - Helper: warm-start the draft from the base

/// Copy the base model's token embedding + LM head weights into the
/// draft net. This is the EAGLE recipe: the draft net's vocab
/// projection is a copy of the base's LM head (tied embedding case:
/// copy from the token embedding). Warm-starting them saves the draft
/// having to relearn the vocab-space mapping from scratch.
///
/// The copy is one-way — once trained, the draft owns its own version
/// and the base stays untouched.
public enum EagleWarmStart {
    public static func fromBase(_ base: TinyGPTModel, into draft: EagleDraft) throws {
        // token_embed.weight ← base.token_embedding.weight
        let tokWeight = base.tokenEmbedding.weight
        eval(tokWeight)
        // vocab_proj.weight ← base.lmHead.weight (untied) or
        //                     base.token_embedding.weight (tied)
        let vocabWeight: MLXArray
        if let head = base.lmHead {
            vocabWeight = head.weight
        } else {
            vocabWeight = tokWeight
        }
        eval(vocabWeight)
        // Build a NestedDictionary update.
        var root: [String: NestedItem<String, MLXArray>] = [:]
        root["token_embed"] = .dictionary([
            "weight": .value(tokWeight)
        ])
        root["vocab_proj"] = .dictionary([
            "weight": .value(vocabWeight)
        ])
        try draft.update(parameters: NestedDictionary(values: root), verify: [])
    }
}
