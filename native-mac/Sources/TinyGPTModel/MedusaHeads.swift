import Foundation
import MLX
import MLXNN
import MLXRandom

/// Medusa speculative-decoding heads (Cai et al., 2024 — "Medusa: Simple
/// LLM Inference Acceleration Framework with Multiple Decoding Heads").
///
/// The idea: bolt N tiny prediction heads onto a frozen base LM. Head `k`
/// takes the base's hidden state at position t and predicts the token at
/// position t + k + 1 — i.e. the (k+1)-step-ahead token. At inference,
/// after one base forward we already have N+1 candidate tokens for the
/// next N+1 positions (the base's own argmax + each head's argmax). One
/// more base forward over `prompt + candidates` then VERIFIES which of
/// the head-predicted tokens match what the base WOULD have produced at
/// each position — accept the longest matching prefix.
///
/// Training: base is frozen. Only the heads update. Loss is the per-head
/// next-token CE at the head's offset — head k scores against
/// `targets[:, k+1:]`. With heads tiny (one ResBlock + linear projection
/// to vocab), training is cheap.
///
/// Tree attention (Cai 2024 §3.3): instead of a single 1-D candidate
/// continuation, Medusa proposes a TREE of candidates (top-k per head,
/// expand combinations under a learned acceptance prior). The base
/// verifies all tree paths in ONE forward with a custom block-diagonal
/// causal mask. **This first cut implements the simpler LINEAR variant**:
/// take each head's argmax as a single proposed token at its offset.
/// Mathematically equivalent to "tree of width 1"; preserves correctness
/// (greedy verify wrt the base's argmax), gives up the wider-search
/// acceptance-rate boost that the full tree attention buys.
///
/// File format for a saved head set is a sidecar `.heads`, similar in
/// spirit to the LoRA adapter / tuned-lens formats:
///   magic "TGMH" (4 bytes)     — TinyGPT Medusa/EAGLE Heads
///   version u32 (currently 1)
///   header_len u32
///   JSON header  { kind, numHeads, dModel, vocabSize, ... }
///   raw fp32 weights, one head at a time, in the order described by
///                                  the head module's `parameterEntries`
///
/// Kind is `"medusa"` here; the same container is reused by EAGLE-2 (see
/// `EagleDraft.swift`) — distinguished via the JSON header's `kind` field.

// MARK: - A single Medusa head

/// One Medusa head:
///   y = SiLU(W1 · x) + x       (residual block)
///   logits = W2 · y            (projection to vocab, untied — owning its
///                                own copy is cheaper to train than tying
///                                via the base LM head, which would
///                                require the base to be unfrozen)
public final class MedusaHead: Module {
    @ModuleInfo(key: "res_proj") public var resProj: Linear
    @ModuleInfo(key: "vocab_proj") public var vocabProj: Linear
    public let dModel: Int
    public let vocabSize: Int

    public init(dModel: Int, vocabSize: Int) {
        self.dModel = dModel
        self.vocabSize = vocabSize
        // Residual projection — keep the bias to give the head a free
        // scalar to adjust toward an offset target that may be far from
        // the base's argmax (think: long-tail token frequency).
        self._resProj.wrappedValue = Linear(dModel, dModel, bias: true)
        // Vocab projection — bias-free to mirror the standard LM head.
        self._vocabProj.wrappedValue = Linear(dModel, vocabSize, bias: false)
        super.init()
    }

    /// `[B, T, dModel]` hidden state → `[B, T, vocabSize]` head logits.
    public func callAsFunction(_ h: MLXArray) -> MLXArray {
        let r = silu(resProj(h)) + h
        return vocabProj(r)
    }
}

// MARK: - Stack of N Medusa heads

/// `MedusaHeadStack` — `numHeads` independent Medusa heads. Head index `k`
/// predicts offset `k + 1`. The stack is a standalone `Module` so we can
/// run autograd against it directly (the base model stays a captured
/// constant during head training, just like the tuned-lens probes setup).
public final class MedusaHeadStack: Module {
    @ModuleInfo(key: "heads") public var heads: [MedusaHead]
    public let cfg: ModelConfig.SpeculativeHeadConfig
    public let dModel: Int
    public let vocabSize: Int

    public init(cfg: ModelConfig.SpeculativeHeadConfig, dModel: Int, vocabSize: Int) {
        precondition(cfg.kind == .medusa, "MedusaHeadStack needs cfg.kind == .medusa")
        self.cfg = cfg
        self.dModel = dModel
        self.vocabSize = vocabSize
        self._heads.wrappedValue = (0..<cfg.numHeads).map { _ in
            MedusaHead(dModel: dModel, vocabSize: vocabSize)
        }
        super.init()
    }

    /// Per-head logits at every position. Used during head training so
    /// we can backprop one CE per head against the right shifted target.
    public func callAsFunction(_ hidden: MLXArray) -> [MLXArray] {
        return heads.map { $0(hidden) }
    }

    /// Total parameter count across all heads. The head set is meant to
    /// be small (1-2% of base model params) so this is mostly a diagnostic.
    public func numParameters() -> Int {
        var total = 0
        for (_, p) in parameters().flattened() {
            total += p.shape.reduce(1, *)
        }
        return total
    }
}

// MARK: - Training-time loss

/// Per-head next-token CE for Medusa training.
///
/// Head `k` at position `t` predicts `targets[t + k + 1]` (in the
/// standard shifted-by-one convention used by the rest of the codebase,
/// `targets[t]` IS already the t+1 ground-truth — so head k's target at
/// position t is `targets[t + k]`). The last `k` positions can't be
/// scored at head k because we run out of look-ahead; we slice them off
/// from both logits and targets symmetrically (same trick as the MTP
/// loss in `TinyGPTModel.swift`).
///
/// Returns the mean of per-head means — so each head contributes
/// equally regardless of how many valid positions it had.
public func medusaHeadsLoss(headLogits: [MLXArray], targets: MLXArray) -> MLXArray {
    precondition(!headLogits.isEmpty, "medusaHeadsLoss needs ≥1 head logits")
    let T = targets.shape[1]
    var total = MLXArray(Float(0))
    var scored = 0
    for (k, logitsK) in headLogits.enumerated() {
        let valid = T - k
        if valid <= 0 { continue }
        // Slice logits' time-axis to the valid window; shift targets left by k.
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

// MARK: - Greedy linear verification

/// Result of one speculative step: the set of newly accepted token ids
/// (≥ 1, ≤ proposals.count + 1) and the number of proposals that got
/// accepted (so the caller can track an acceptance-rate metric).
public struct SpecHeadsStepResult {
    public let acceptedIds: [Int]
    public let proposalsAccepted: Int
    public let proposalsTotal: Int
}

/// Greedy "linear" Medusa step — width-1 tree path.
///
/// Cost model: 2 base forwards per step (one to get the hidden states +
/// head proposals, one to verify the proposed prefix). On the dense
/// fp32 path the verify forward is the same shape as standard decode,
/// so when acceptance rate is high (say, 60%+) we get k+1 accepted
/// tokens for 2 base forwards instead of k+1 base forwards — speedup
/// approaches (k+1)/2.
///
/// When acceptance rate is near zero (early in head training, or for a
/// near-random head), we always accept just 1 token (the base's own
/// argmax at the first position) and end up doing 2 base forwards per
/// 1 token — a 0.5× slowdown. The honest training-from-50-steps case
/// will look like the latter; that's expected.
///
/// `model` must support the `AnyModel` interface used elsewhere in the
/// CLI so this works for both from-scratch + HF models.
public enum MedusaVerify {
    /// One Medusa-style verify step. Mutates `ids` by appending the
    /// newly accepted tokens (≥ 1).
    public static func step(
        baseHidden: (MLXArray) -> MLXArray,   // [B,T] → [B,T,d]
        baseLogits: (MLXArray) -> MLXArray,   // [B,T] → [B,T,vocab]
        baseLMHead: (MLXArray) -> MLXArray,   // hidden [B,T,d] → logits [B,T,vocab]
        heads: MedusaHeadStack,
        ids: inout [Int],
        ctxCap: Int
    ) -> SpecHeadsStepResult {
        // 1. Forward base on `ids` → grab the last-position hidden state,
        //    feed it through each head to get proposed tokens [k1, k2, ...].
        let tail = ids.suffix(ctxCap)
        let arr = MLXArray(tail.map { Int32($0) }, [1, tail.count])
        let hiddenAll = baseHidden(arr)                  // [1, T, d]
        let lastHidden = hiddenAll[0..., hiddenAll.shape[1] - 1 ..< hiddenAll.shape[1], 0...] // [1, 1, d]
        // Base's own next-token argmax (the "head 0" proposal that doesn't
        // need a Medusa head — just the existing LM head). This becomes the
        // FIRST proposal in our verify burst.
        let baseLastLogits = baseLMHead(lastHidden)      // [1, 1, vocab]
        let baseArg = argMax(baseLastLogits[0..., 0, 0...], axis: -1).reshaped([1])
        eval(baseArg)
        var proposals: [Int] = [Int(baseArg.item(Int32.self))]

        // Heads predict offsets 1..N from the same lastHidden.
        for head in heads.heads {
            let logitsK = head(lastHidden)                // [1, 1, vocab]
            let arg = argMax(logitsK[0..., 0, 0...], axis: -1).reshaped([1])
            eval(arg)
            proposals.append(Int(arg.item(Int32.self)))
        }
        // proposals now has length N+1 (the base + the N heads).

        // 2. Verify in one forward: feed (ids + proposals) and read off
        //    the base's argmax at the positions that PREDICT each
        //    proposal. Same accept rule as vanilla spec-decode: longest
        //    prefix of matching argmax, then on the first mismatch
        //    substitute the base's own argmax and stop.
        let withProposals = (ids + proposals).suffix(ctxCap)
        let inputArr = MLXArray(withProposals.map { Int32($0) }, [1, withProposals.count])
        let tLogits = baseLogits(inputArr)               // [1, T+P, vocab]
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
                // First mismatch: take base's argmax, end the burst.
                accepted.append(tTok)
                ids.append(contentsOf: accepted)
                return SpecHeadsStepResult(
                    acceptedIds: accepted,
                    proposalsAccepted: acceptedProposals,
                    proposalsTotal: totalProposals
                )
            }
        }
        // Every proposal accepted. Bonus: append the base's argmax at the
        // position AFTER the last proposal (predicts the next token for
        // free, same logic as the vanilla speculative path).
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

// MARK: - On-disk format

/// JSON header for a `.heads` sidecar — shared by Medusa + EAGLE-2.
public struct SpecHeadsFileHeader: Codable, Sendable {
    public var kind: String           // "medusa" | "eagle"
    public var numHeads: Int
    public var hiddenDim: Int
    public var dModel: Int
    public var vocabSize: Int
    public var baseLayers: Int
    public var baseDModel: Int
    public var baseHeads: Int
    public var baseCtx: Int
    public var savedAt: String?
    public var finalLoss: Float?
    /// Per-parameter tensor entries in serialise order. Each entry records
    /// the parameter's flattened key path and shape; the on-disk bytes
    /// after the header are these tensors concatenated in this order.
    public var entries: [SpecHeadsTensorEntry]
}

public struct SpecHeadsTensorEntry: Codable, Sendable {
    public var name: String      // e.g. "heads.0.res_proj.weight"
    public var shape: [Int]
}

public enum SpecHeadsFormat {
    public static let magic: [UInt8] = Array("TGMH".utf8)   // TinyGPT Medusa/EAGLE Heads
    public static let currentVersion: UInt32 = 1
}

// MARK: - Serialisation

/// Write a Medusa head stack to a `.heads` sidecar.
public enum MedusaHeadsIO {
    public static func write(
        stack: MedusaHeadStack,
        baseConfig: ModelConfig,
        finalLoss: Float?,
        to url: URL
    ) throws {
        let params = stack.parameters().flattened()
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
            kind: stack.cfg.kind.rawValue,
            numHeads: stack.cfg.numHeads,
            hiddenDim: stack.cfg.hiddenDim,
            dModel: stack.dModel,
            vocabSize: stack.vocabSize,
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

    /// Load a Medusa head stack from disk. Throws if the header's `kind`
    /// isn't `"medusa"` or the base config doesn't match.
    public static func read(_ url: URL, baseConfig: ModelConfig) throws -> MedusaHeadStack {
        let (header, blobs) = try readHeaderAndBlobs(url)
        guard header.kind == "medusa" else {
            throw NSError(domain: "TinyGPTHeads", code: 5,
                          userInfo: [NSLocalizedDescriptionKey:
                            "expected kind=medusa, got \(header.kind)"])
        }
        try validateBaseConfig(header: header, baseConfig: baseConfig)
        let cfg = ModelConfig.SpeculativeHeadConfig(
            kind: .medusa, numHeads: header.numHeads, hiddenDim: header.hiddenDim
        )
        let stack = MedusaHeadStack(cfg: cfg, dModel: header.dModel, vocabSize: header.vocabSize)
        try restoreParameters(into: stack, header: header, blobs: blobs)
        return stack
    }
}

// MARK: - Shared header / blob plumbing (used by EAGLE too)

func writeHeaderAndBlobs(header: SpecHeadsFileHeader, blobs: [Data], to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let headerData = try encoder.encode(header)
    var out = Data()
    out.append(contentsOf: SpecHeadsFormat.magic)
    var ver = SpecHeadsFormat.currentVersion.littleEndian
    withUnsafeBytes(of: &ver) { out.append(contentsOf: $0) }
    var headerLen = UInt32(headerData.count).littleEndian
    withUnsafeBytes(of: &headerLen) { out.append(contentsOf: $0) }
    out.append(headerData)
    for blob in blobs { out.append(blob) }
    try out.write(to: url, options: .atomic)
}

func readHeaderAndBlobs(_ url: URL) throws -> (SpecHeadsFileHeader, [Data]) {
    let data = try Data(contentsOf: url)
    guard data.count >= 12 else {
        throw NSError(domain: "TinyGPTHeads", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "file too small"])
    }
    guard Array(data[0..<4]) == SpecHeadsFormat.magic else {
        throw NSError(domain: "TinyGPTHeads", code: 2,
                      userInfo: [NSLocalizedDescriptionKey: "bad magic, expected 'TGMH'"])
    }
    let version = data[4..<8].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian }
    guard version == SpecHeadsFormat.currentVersion else {
        throw NSError(domain: "TinyGPTHeads", code: 3,
                      userInfo: [NSLocalizedDescriptionKey: "unsupported version \(version)"])
    }
    let headerLen = Int(data[8..<12].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian })
    let headerData = data.subdata(in: 12..<(12 + headerLen))
    let header = try JSONDecoder().decode(SpecHeadsFileHeader.self, from: headerData)
    var cursor = 12 + headerLen
    var blobs: [Data] = []
    for entry in header.entries {
        let nFloats = entry.shape.reduce(1, *)
        let nBytes = nFloats * 4
        let blob = data.subdata(in: cursor..<(cursor + nBytes))
        cursor += nBytes
        blobs.append(blob)
    }
    return (header, blobs)
}

func validateBaseConfig(header: SpecHeadsFileHeader, baseConfig: ModelConfig) throws {
    guard header.baseLayers == baseConfig.nLayers,
          header.baseDModel == baseConfig.dModel,
          header.baseHeads == baseConfig.nHeads,
          header.baseCtx == baseConfig.contextLength,
          header.dModel == baseConfig.dModel,
          header.vocabSize == baseConfig.vocabSize else {
        throw NSError(domain: "TinyGPTHeads", code: 4,
                      userInfo: [NSLocalizedDescriptionKey:
                        "head sidecar base config doesn't match loaded model"])
    }
}

/// Copy serialised fp32 blobs into the head stack's parameter tree by
/// name. Same trick the LoRA adapter loader uses — build a
/// NestedDictionary `update(...)` payload.
func restoreParameters(into module: Module, header: SpecHeadsFileHeader, blobs: [Data]) throws {
    // Build a name → MLXArray map.
    var named: [String: MLXArray] = [:]
    for (entry, blob) in zip(header.entries, blobs) {
        let n = entry.shape.reduce(1, *)
        let floats = blob.withUnsafeBytes { Array(UnsafeBufferPointer<Float>(
            start: $0.baseAddress?.assumingMemoryBound(to: Float.self), count: n
        )) }
        named[entry.name] = MLXArray(floats, entry.shape)
    }
    // Convert flat dotted-name map to MLX's NestedDictionary update payload.
    let nested = buildNestedDictionary(named)
    try module.update(parameters: nested, verify: [])
}

/// Build a NestedDictionary<String, MLXArray> from a flat dotted-name map.
/// Walks each dotted name, growing the tree, and stamps `.value(arr)` at
/// the leaf. Integer path components (from `[ModuleInfo]` arrays) become
/// `.array` children, mirroring how `Module.parameters()` flattens.
func buildNestedDictionary(_ flat: [String: MLXArray]) -> ModuleParameters {
    // We assemble dictionaries/arrays bottom-up by sorting keys and
    // grouping by their path prefixes. Simpler approach: build a recursive
    // [String: Any] then convert. Use NestedItem directly via mergeKey.
    var root: [String: NestedItem<String, MLXArray>] = [:]
    for (path, arr) in flat {
        insertPath(&root, components: path.split(separator: ".").map(String.init), value: arr)
    }
    return NestedDictionary(values: root)
}

private func insertPath(_ container: inout [String: NestedItem<String, MLXArray>],
                        components: [String], value: MLXArray) {
    guard let head = components.first else { return }
    let rest = Array(components.dropFirst())
    if rest.isEmpty {
        container[head] = .value(value)
        return
    }
    // If `rest[0]` looks like an integer, the current head should map to
    // an `.array` of children; otherwise it's a dictionary.
    if let _ = Int(rest[0]) {
        // head → .array(...)
        var existing: [NestedItem<String, MLXArray>]
        if case .array(let a)? = container[head] { existing = a } else { existing = [] }
        let idx = Int(rest[0])!
        while existing.count <= idx { existing.append(.none) }
        let restRest = Array(rest.dropFirst())
        if restRest.isEmpty {
            existing[idx] = .value(value)
        } else {
            // Grab dict at index, recurse, put back.
            var inner: [String: NestedItem<String, MLXArray>]
            if case .dictionary(let d) = existing[idx] { inner = d } else { inner = [:] }
            insertPath(&inner, components: restRest, value: value)
            existing[idx] = .dictionary(inner)
        }
        container[head] = .array(existing)
    } else {
        // head → .dictionary(...)
        var inner: [String: NestedItem<String, MLXArray>]
        if case .dictionary(let d)? = container[head] { inner = d } else { inner = [:] }
        insertPath(&inner, components: rest, value: value)
        container[head] = .dictionary(inner)
    }
}
