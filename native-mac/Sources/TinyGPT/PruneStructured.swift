import Foundation
import MLX
import MLXNN
import TinyGPTIO
import TinyGPTModel

/// `tinygpt prune-structured` — drop entire attention heads or whole
/// transformer layers. Unlike unstructured pruning (which leaves
/// tensor shapes intact), this DOES change the model topology:
///
///   - `--layers-to-drop M`: physically removes M transformer blocks
///     from the model. The output `.tinygpt` has `nLayers - M` layers
///     and is genuinely smaller (fewer parameters, fewer FLOPs, real
///     wallclock win on the GPU).
///   - `--heads-to-drop N`: zeros N attention heads' worth of weights
///     in EVERY layer. Shape-preserving (still loads with the original
///     `nHeads`); inference contribution is zero. No wallclock win in
///     this iteration — see docs/pruning.md for the rationale.
///
/// Importance scoring:
///   - Heads: Frobenius-norm sum across each head's Q/K/V/O slabs.
///     A head whose projections have collapsed near zero gets a low
///     score and is a safe drop. See `Pruning.headImportance`.
///   - Layers: block angular distance — for each layer L, we measure
///     the cosine angle between the residual stream entering L and
///     exiting L. Layers whose output is nearly identical to their
///     input (angle ≈ 0) are contributing little. From Gromov et al.,
///     2024, "The Unreasonable Ineffectiveness of the Deeper Layers".
///
/// USAGE
///   tinygpt prune-structured <model.tinygpt> \
///       --heads-to-drop 4 --calibration calib.txt --out reduced.tinygpt
///   tinygpt prune-structured <model.tinygpt> \
///       --layers-to-drop 2 --calibration calib.txt --out reduced.tinygpt
enum PruneStructured {
    static func run(args: [String]) {
        var inPath: String? = nil
        var outPath: String? = nil
        var headsToDrop: Int = 0
        var layersToDrop: Int = 0
        var calibrationPath: String? = nil
        var calibBatches: Int = 4
        var calibBatchSize: Int = 4

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--out":             outPath = args[i+1]; i += 2
            case "--heads-to-drop":   headsToDrop = Int(args[i+1]) ?? headsToDrop; i += 2
            case "--layers-to-drop":  layersToDrop = Int(args[i+1]) ?? layersToDrop; i += 2
            case "--calibration":     calibrationPath = args[i+1]; i += 2
            case "--calib-batches":   calibBatches = Int(args[i+1]) ?? calibBatches; i += 2
            case "--calib-batch":     calibBatchSize = Int(args[i+1]) ?? calibBatchSize; i += 2
            case "-h", "--help":      exitUsage()
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                inPath = args[i]; i += 1
            }
        }
        guard let inPath = inPath else { fputs("missing <model>\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out <path> required\n", stderr); exitUsage() }
        if headsToDrop <= 0 && layersToDrop <= 0 {
            fputs("at least one of --heads-to-drop or --layers-to-drop must be > 0\n", stderr)
            exitUsage()
        }

        print("loading \(inPath)…")
        let inputURL = URL(fileURLWithPath: inPath)
        var file: TinyGPTFile
        do { file = try TinyGPTFileReader.read(inputURL) }
        catch { fputs("read failed: \(error)\n", stderr); exit(1) }

        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(inPath) }
        catch { fputs("model load failed: \(error)\n", stderr); exit(1) }
        guard case .fromScratch(let model) = load.model else {
            fputs("structured pruning currently supports only from-scratch (.tinygpt) models — got HF\n", stderr)
            exit(1)
        }
        let cfg = load.config

        print("""

        TinyGPT — Structured pruning
        ----------------------------
        input:           \(inPath)
        nLayers:         \(cfg.nLayers)    drop: \(layersToDrop)
        nHeads:          \(cfg.nHeads)    drop: \(headsToDrop)
        calibration:     \(calibrationPath ?? "—")
        output:          \(outPath)

        """)

        // 1. HEAD pruning (shape-preserving zeros). Score across the
        // CALIBRATION corpus is omitted — head importance via weight
        // norms is purely a function of the loaded weights, no
        // forward pass needed.
        var headsDroppedReport: [Int] = []
        if headsToDrop > 0 {
            headsDroppedReport = pruneHeads(file: &file, cfg: cfg, headsToDrop: headsToDrop)
        }

        // 2. LAYER pruning (topology-changing). Needs calibration
        // text to compute per-layer angular distance. If no
        // calibration was supplied, fall back to dropping the
        // middle-most layers (Gromov et al.'s empirical observation
        // is that mid-late layers are the safest drops).
        var layersDroppedReport: [Int] = []
        if layersToDrop > 0 {
            let layerScores: [Float]
            if let cp = calibrationPath {
                do {
                    let corpus = try ByteCorpus(contentsOf: URL(fileURLWithPath: cp))
                    layerScores = computeLayerScores(model: model, cfg: cfg,
                                                      corpus: corpus,
                                                      nBatches: calibBatches,
                                                      batchSize: calibBatchSize)
                } catch {
                    fputs("calibration load failed (\(error)); falling back to mid-layer drop\n", stderr)
                    layerScores = midDropScores(nLayers: cfg.nLayers)
                }
            } else {
                fputs("warning: no --calibration given; falling back to mid-layer drop\n", stderr)
                layerScores = midDropScores(nLayers: cfg.nLayers)
            }
            // Drop the K layers with the LOWEST importance (= lowest
            // angular distance = output ≈ input).
            let drops = layerScores.enumerated()
                .sorted(by: { $0.element < $1.element })
                .prefix(layersToDrop)
                .map { $0.offset }
                .sorted()
            layersDroppedReport = drops
            print("\nlayer importance (angular distance — lower = drop me):")
            for (idx, s) in layerScores.enumerated() {
                let tag = drops.contains(idx) ? " ← DROP" : ""
                print(String(format: "  block %2d  %.4f%@", idx, s, tag))
            }
            file = rebuildFileWithDroppedLayers(file: file, drops: Set(drops))
        }

        // Write the new file. Header gets pruning metadata; for
        // layer-drop we also reduce the `layers` config field so
        // load-back constructs a smaller model.
        var info = TinyGPTHeader.PruningInfo(kind: layersToDrop > 0 && headsToDrop > 0
            ? "structured-mixed"
            : (layersToDrop > 0 ? "structured-layer" : "structured-head"))
        if !headsDroppedReport.isEmpty { info.headsDropped = headsDroppedReport }
        if !layersDroppedReport.isEmpty { info.layersDropped = layersDroppedReport }
        file.header.pruningInfo = info
        // Update stateByteLength for layer-drop (manifest changed).
        file.header.stateByteLength = computeStateByteLength(file)
        do {
            try TinyGPTFileWriter.write(file, to: URL(fileURLWithPath: outPath))
            print("\n✓ wrote \(outPath)")
        } catch {
            fputs("write failed: \(error)\n", stderr); exit(1)
        }
    }

    // MARK: - Head pruning

    /// Score every head and drop the K lowest-importance PER LAYER.
    /// `headsToDrop` is interpreted as "drop the K weakest heads in
    /// EVERY transformer block" (Michel et al. 2019's standard
    /// convention). With 12 layers and `--heads-to-drop 4`, every
    /// layer loses 4 of 8 heads = a 50% head pruning rate. Returns
    /// the flat indices that were zeroed, `layer * nHeads + head`.
    private static func pruneHeads(file: inout TinyGPTFile, cfg: ModelConfig, headsToDrop: Int) -> [Int] {
        precondition(headsToDrop < cfg.nHeads,
                     "--heads-to-drop \(headsToDrop) must be < nHeads (\(cfg.nHeads)) — a layer can't have zero heads")
        var byLayer: [Int: Set<Int>] = [:]
        var droppedFlat: [Int] = []
        for L in 0..<cfg.nLayers {
            guard let q = tensorByName(file, "blocks.\(L).attn.q_proj.weight"),
                  let k = tensorByName(file, "blocks.\(L).attn.k_proj.weight"),
                  let v = tensorByName(file, "blocks.\(L).attn.v_proj.weight"),
                  let o = tensorByName(file, "blocks.\(L).attn.o_proj.weight")
            else { continue }
            let qf = floatsFromTensor(q)
            let kf = floatsFromTensor(k)
            let vf = floatsFromTensor(v)
            let of = floatsFromTensor(o)
            // The on-disk weight order is WASM [in, out]; PyTorch
            // is [out, in]. The Pruning helpers assume PyTorch
            // [dModel_out × dModel_in] row-major. Transpose them
            // here in plain Swift before scoring.
            let qF = transposeRowMajor(qf, rows: cfg.dModel, cols: cfg.dModel)
            let kF = transposeRowMajor(kf, rows: cfg.dModel, cols: cfg.dModel)
            let vF = transposeRowMajor(vf, rows: cfg.dModel, cols: cfg.dModel)
            let oF = transposeRowMajor(of, rows: cfg.dModel, cols: cfg.dModel)
            let scores = Pruning.headImportance(
                qProj: qF, kProj: kF, vProj: vF, oProj: oF,
                dModel: cfg.dModel, nHeads: cfg.nHeads, nKvHeads: cfg.nKvHeads
            )
            // Drop the K lowest-scoring heads in this layer.
            let drops = scores.enumerated().sorted(by: { $0.element < $1.element })
                .prefix(headsToDrop).map { $0.offset }
            byLayer[L] = Set(drops)
            for h in drops { droppedFlat.append(L * cfg.nHeads + h) }
        }
        print("\nheads dropped: \(droppedFlat.count) (across \(byLayer.count) layers)")
        for L in byLayer.keys.sorted() {
            let hs = byLayer[L]!.sorted()
            print("  block \(L): heads \(hs)")
        }
        // Zero in place. We touch q, k, v, o weights (and biases if
        // present). The on-disk WASM [in, out] order means we need to
        // transpose into PyTorch, edit, then transpose back.
        for (L, hs) in byLayer {
            zeroHeadsInTensorPair(file: &file, cfg: cfg, layer: L, headsToDrop: hs)
        }
        return droppedFlat.sorted()
    }

    /// Apply head-zeroing in place to one layer's q/k/v/o (weights +
    /// optional biases) by going through the PyTorch [out, in] view.
    private static func zeroHeadsInTensorPair(file: inout TinyGPTFile, cfg: ModelConfig,
                                                layer L: Int, headsToDrop: Set<Int>)
    {
        let dM = cfg.dModel
        for tag in ["q", "k", "v", "o"] {
            let wName = "blocks.\(L).attn.\(tag)_proj.weight"
            guard let wi = file.tensors.firstIndex(where: { $0.entry.name == wName }) else { continue }
            var w = floatsFromTensor(file.tensors[wi])
            // file stores [in, out]; transpose to [out, in] for the
            // per-head row/col operation.
            var wPT = transposeRowMajor(w, rows: dM, cols: dM)
            // Apply per-tensor zeroing.
            let headDim = cfg.headDim
            let nH = cfg.nHeads
            if tag == "q" {
                for h in headsToDrop {
                    let rStart = h * headDim
                    let rEnd = rStart + headDim
                    for r in rStart..<rEnd {
                        let rowBase = r * dM
                        for c in 0..<dM { wPT[rowBase + c] = 0 }
                    }
                }
            } else if tag == "k" || tag == "v" {
                // KV head dropped only when ALL grouped query heads
                // are dropped. With nKvHeads == nHeads (no GQA), one
                // query head = one KV head — same drop set.
                let groupSize = max(1, nH / cfg.nKvHeads)
                var dropKV = Set<Int>()
                for kvh in 0..<cfg.nKvHeads {
                    let group = Set((0..<groupSize).map { kvh * groupSize + $0 })
                    if group.isSubset(of: headsToDrop) { dropKV.insert(kvh) }
                }
                for kvh in dropKV {
                    let rStart = kvh * headDim
                    let rEnd = rStart + headDim
                    for r in rStart..<rEnd {
                        let rowBase = r * dM
                        for c in 0..<dM { wPT[rowBase + c] = 0 }
                    }
                }
            } else { // o
                for h in headsToDrop {
                    let cStart = h * headDim
                    let cEnd = cStart + headDim
                    for r in 0..<dM {
                        let rowBase = r * dM
                        for c in cStart..<cEnd { wPT[rowBase + c] = 0 }
                    }
                }
            }
            // Transpose back to [in, out] and write.
            w = transposeRowMajor(wPT, rows: dM, cols: dM)
            file.tensors[wi].weight = packFloats(w, dtype: file.tensors[wi].dtype)
            // Bias (1-D, length dM) — same head-block layout.
            let bName = "blocks.\(L).attn.\(tag)_proj.bias"
            if let bi = file.tensors.firstIndex(where: { $0.entry.name == bName }) {
                var b = floatsFromTensor(file.tensors[bi])
                if tag == "q" {
                    for h in headsToDrop {
                        let s = h * cfg.headDim
                        for r in s..<(s + cfg.headDim) { b[r] = 0 }
                    }
                } else if tag == "k" || tag == "v" {
                    let groupSize = max(1, nH / cfg.nKvHeads)
                    var dropKV = Set<Int>()
                    for kvh in 0..<cfg.nKvHeads {
                        let group = Set((0..<groupSize).map { kvh * groupSize + $0 })
                        if group.isSubset(of: headsToDrop) { dropKV.insert(kvh) }
                    }
                    for kvh in dropKV {
                        let s = kvh * cfg.headDim
                        for r in s..<(s + cfg.headDim) { b[r] = 0 }
                    }
                }
                // o_proj bias has shape [dModel] — same as the output side,
                // shared across heads, so we leave it intact.
                file.tensors[bi].weight = packFloats(b, dtype: file.tensors[bi].dtype)
            }
        }
    }

    // MARK: - Layer pruning

    /// Run a small calibration forward, capture per-layer
    /// hidden-state vectors, score by angular distance from the
    /// PREVIOUS layer's output. Returns one score per layer
    /// (`layerScores[L]` = how MUCH layer L moves the residual).
    /// Lower = safer to drop.
    private static func computeLayerScores(model: TinyGPTModel, cfg: ModelConfig,
                                            corpus: ByteCorpus,
                                            nBatches: Int, batchSize: Int) -> [Float]
    {
        var scores = [Float](repeating: 0, count: cfg.nLayers)
        var nObs = 0
        for _ in 0..<nBatches {
            // Use a SHORTER context to keep this cheap — the layer's
            // angular-distance score is a per-position average, so
            // we don't need full context to estimate it stably.
            let T = min(cfg.contextLength, 64)
            let (x, _) = corpus.sampleBatch(batchSize: batchSize, contextLength: T)
            let layerwise = model.forwardLayerwise(x)
            // layerwise[L] = residual AFTER block L. To compute the
            // angular distance "this layer's input vs output", we
            // need block-L's INPUT, which is the previous output
            // (or the embedding for L == 0). We can synthesise that
            // — for L == 0 we use a fresh embedding lookup; for L >
            // 0 we use layerwise[L-1].
            // For simplicity (and following the spirit of Gromov et
            // al., who score "blocks of layers"), we compute the
            // angular distance between layerwise[L-1] and
            // layerwise[L]. We score L = 0 by comparing the
            // embedding to layerwise[0] — same idea.
            // Get the embedding output. We can read it via the
            // model's helpers; cheap, no autograd.
            var prev: MLXArray = model.tokenEmbedding(x)
            for L in 0..<cfg.nLayers {
                let cur = layerwise[L]
                let pf: [Float] = prev.asArray(Float.self)
                let cf: [Float] = cur.asArray(Float.self)
                scores[L] += Pruning.angularDistance(pf, cf)
                prev = cur
            }
            nObs += 1
        }
        if nObs > 0 {
            for L in 0..<cfg.nLayers { scores[L] /= Float(nObs) }
        }
        return scores
    }

    /// Fallback layer scoring when no calibration is supplied:
    /// drop the MIDDLE layers (give them the lowest score). The
    /// first and last layers are protected because they handle
    /// embedding-to-residual mapping and final pre-readout shaping.
    private static func midDropScores(nLayers: Int) -> [Float] {
        var s = [Float](repeating: 1.0, count: nLayers)
        // Make middle layers small (so they get dropped first).
        for i in 0..<nLayers {
            let center = Float(nLayers) / 2.0
            // Score = distance from center; lowest at center.
            s[i] = abs(Float(i) - center + 0.5)
        }
        // Keep first and last very high so they survive.
        if nLayers > 0 { s[0] = 1e9 }
        if nLayers > 1 { s[nLayers - 1] = 1e9 }
        return s
    }

    /// Rebuild the file's manifest + tensor list with the given
    /// layer indices physically removed. Block indices are
    /// re-numbered contiguously so the result is a valid model
    /// with fewer layers. Updates header.config.layers.
    private static func rebuildFileWithDroppedLayers(file: TinyGPTFile, drops: Set<Int>) -> TinyGPTFile
    {
        let oldNLayers = file.header.config.layers ?? 0
        let surviving = (0..<oldNLayers).filter { !drops.contains($0) }
        // Map old block index → new block index.
        var remap: [Int: Int] = [:]
        for (newIdx, oldIdx) in surviving.enumerated() {
            remap[oldIdx] = newIdx
        }

        // Walk tensors. For each tensor whose name has a "blocks.N."
        // prefix where N is dropped, skip. Otherwise rename N → new N
        // (a no-op when no remap is needed). Non-block tensors
        // (token_embedding, ln_final, etc.) pass through unchanged.
        var newTensors: [TinyGPTTensor] = []
        var newEntries: [TinyGPTHeader.TensorEntry] = []
        var offset = 0
        for t in file.tensors {
            let name = t.entry.name
            let parsed = parseBlockIndex(name)
            if let L = parsed.layer {
                if drops.contains(L) { continue }
                let newL = remap[L] ?? L
                let newName = "blocks.\(newL).\(parsed.suffix)"
                var entry = t.entry
                entry.name = newName
                entry.floatOffset = offset
                newEntries.append(entry)
                offset += entry.elementCount
                var newT = t
                newT.entry = entry
                newTensors.append(newT)
            } else {
                var entry = t.entry
                entry.floatOffset = offset
                newEntries.append(entry)
                offset += entry.elementCount
                var newT = t
                newT.entry = entry
                newTensors.append(newT)
            }
        }

        // Update header.config.layers.
        var newHeader = file.header
        newHeader.config.layers = surviving.count
        newHeader.manifest = newEntries
        return TinyGPTFile(version: file.version, header: newHeader,
                            step: file.step, tensors: newTensors)
    }

    /// Split a tensor name like "blocks.7.attn.q_proj.weight" into
    /// (layer: 7, suffix: "attn.q_proj.weight"). Returns nil layer
    /// for non-block tensors.
    private static func parseBlockIndex(_ name: String) -> (layer: Int?, suffix: String) {
        let parts = name.split(separator: ".", maxSplits: 2, omittingEmptySubsequences: false)
        guard parts.count >= 3, parts[0] == "blocks", let L = Int(parts[1]) else {
            return (nil, name)
        }
        return (L, String(parts[2]))
    }

    // MARK: - Plumbing

    private static func tensorByName(_ file: TinyGPTFile, _ name: String) -> TinyGPTTensor? {
        file.tensors.first { $0.entry.name == name }
    }

    private static func floatsFromTensor(_ tensor: TinyGPTTensor) -> [Float] {
        switch tensor.dtype {
        case .fp32: return tensor.weightFloats
        case .fp16: return tensor.weightFP16AsFloat32()
        }
    }

    private static func packFloats(_ floats: [Float], dtype: TinyGPTDtype) -> Data {
        switch dtype {
        case .fp32:
            return floats.withUnsafeBufferPointer { Data(buffer: $0) }
        case .fp16:
            var halves = [UInt16](repeating: 0, count: floats.count)
            for i in 0..<floats.count { halves[i] = Float16(floats[i]).bitPattern }
            return halves.withUnsafeBufferPointer { Data(buffer: $0) }
        }
    }

    private static func transposeRowMajor(_ flat: [Float], rows: Int, cols: Int) -> [Float] {
        precondition(flat.count == rows * cols, "transpose shape mismatch")
        var out = [Float](repeating: 0, count: rows * cols)
        for r in 0..<rows {
            for c in 0..<cols {
                out[c * rows + r] = flat[r * cols + c]
            }
        }
        return out
    }

    private static func computeStateByteLength(_ file: TinyGPTFile) -> Int {
        switch file.header.bodyLayout {
        case .trainingFP32:
            return 4 + file.tensors.reduce(0) { $0 + 3 * $1.weight.count }
        case .inferenceFP16:
            return file.tensors.reduce(0) { $0 + $1.weight.count }
        }
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt prune-structured <model.tinygpt> [options]

        --out <path>          Where to save the pruned model — required
        --heads-to-drop N     Number of attention heads to zero PER LAYER (Michel et al.
                              2019 convention). Ranked by Frobenius norm of each head's
                              Q/K/V/O slabs; lowest scores get dropped first. Must be < nHeads.
        --layers-to-drop M    Number of transformer blocks to physically remove
                              (ranked by block angular distance — Gromov et al., 2024)
        --calibration <path>  Text corpus for layer importance scoring (required
                              for --layers-to-drop; ignored when only dropping heads)
        --calib-batches N     Calibration batches (default 4)
        --calib-batch N       Calibration batch size (default 4)

        Heads are zeroed in place (shape-preserving, no wallclock speedup —
        Metal has no head-aware sparse matmul). Layers are PHYSICALLY removed,
        producing a smaller dense model with real wallclock + memory wins.
        See docs/pruning.md for the design rationale.
        """)
        exit(2)
    }
}
