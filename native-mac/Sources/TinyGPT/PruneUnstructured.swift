import Foundation
import MLX
import MLXNN
import MLXOptimizers
import TinyGPTIO
import TinyGPTModel

/// `tinygpt prune-unstructured` — drop weights below a magnitude
/// threshold. Each Linear-weight tensor gets a 0/1 mask; weights at
/// mask-0 positions are zeroed.
///
/// Two modes:
///
///   - **Single shot** (default `--iterations 1`): one prune pass,
///     write out. Simplest; the model takes a small quality hit but
///     samples cleanly.
///   - **Iterative Magnitude Pruning** (`--iterations N`): N rounds
///     of (prune K%, fine-tune for `--ft-steps` steps, prune another
///     K%, ...). Each round prunes the SAME fraction of remaining
///     weights, so total sparsity is `1 - (1 - K)^N` (e.g. 50% over
///     2 rounds = 75% total sparsity). IMP recovers most of the
///     quality lost to one-shot pruning, at the cost of fine-tuning
///     cycles. From Frankle & Carbin, 2019, "The Lottery Ticket
///     Hypothesis".
///
/// USAGE
///   tinygpt prune-unstructured <model.tinygpt> --sparsity 0.5 --out pruned.tinygpt
///   tinygpt prune-unstructured <model.tinygpt> --sparsity 0.3 --iterations 3 \
///       --corpus train.txt --ft-steps 100 --out pruned.tinygpt
enum PruneUnstructured {
    static func run(args: [String]) {
        var inPath: String? = nil
        var outPath: String? = nil
        var sparsity: Float = 0.5
        var iterations: Int = 1
        var corpusPath: String? = nil
        var ftSteps: Int = 100
        var ftBatch: Int = 8
        var ftLR: Float = 1e-4
        var includeEmbeddings = false
        var emitMask = true

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--out":          outPath = args[i+1]; i += 2
            case "--sparsity":     sparsity = Float(args[i+1]) ?? sparsity; i += 2
            case "--iterations":   iterations = Int(args[i+1]) ?? iterations; i += 2
            case "--corpus":       corpusPath = args[i+1]; i += 2
            case "--ft-steps":     ftSteps = Int(args[i+1]) ?? ftSteps; i += 2
            case "--ft-batch":     ftBatch = Int(args[i+1]) ?? ftBatch; i += 2
            case "--ft-lr":        ftLR = Float(args[i+1]) ?? ftLR; i += 2
            case "--include-embeddings": includeEmbeddings = true; i += 1
            case "--no-mask": emitMask = false; i += 1
            case "-h", "--help":   exitUsage()
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                inPath = args[i]; i += 1
            }
        }
        guard let inPath = inPath else { fputs("missing <model>\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out <path> required\n", stderr); exitUsage() }
        precondition(sparsity >= 0 && sparsity < 1, "--sparsity must be in [0, 1)")
        precondition(iterations >= 1, "--iterations must be ≥ 1")
        if iterations > 1 && corpusPath == nil {
            fputs("--iterations > 1 requires --corpus (fine-tune between prune steps)\n", stderr)
            exitUsage()
        }

        print("loading \(inPath)…")
        let inputURL = URL(fileURLWithPath: inPath)
        var file: TinyGPTFile
        do { file = try TinyGPTFileReader.read(inputURL) }
        catch { fputs("read failed: \(error)\n", stderr); exit(1) }

        print("""

        TinyGPT — Unstructured magnitude pruning
        ----------------------------------------
        input:         \(inPath)
        sparsity:      \(sparsity)  (per round)
        iterations:    \(iterations)  \(iterations > 1 ? "(IMP)" : "(one-shot)")
        target dtype:  \(file.header.weightDtype ?? "fp32")
        ft corpus:     \(corpusPath ?? "—")
        ft per round:  \(iterations > 1 ? "\(ftSteps) steps · batch \(ftBatch) · lr \(ftLR)" : "—")
        output:        \(outPath)

        """)

        // Build the list of tensor names that are candidates for
        // pruning. By default: everything that's a 2-D Linear weight
        // (q/k/v/o_proj, fc_in/fc_out). Embeddings are excluded by
        // default — pruning them tends to silently kill rare tokens.
        let targetNames = pruneCandidates(file: file, includeEmbeddings: includeEmbeddings)
        print("pruning candidates: \(targetNames.count) tensors")

        // Single-shot path: don't bother loading the model into MLX,
        // just walk the float buffers and zero them.
        if iterations == 1 {
            let (masks, totals) = pruneOneShot(file: &file, targets: targetNames, sparsity: sparsity)
            print("\nsingle-shot prune complete:")
            print("  zeroed weights: \(formatLargeInt(totals.zeroed)) / \(formatLargeInt(totals.total)) (\(String(format: "%.2f%%", 100 * Float(totals.zeroed) / Float(totals.total))))")
            if emitMask {
                file.header.sparsityMasks = encodeMasks(masks)
            }
            file.header.pruningInfo = TinyGPTHeader.PruningInfo(
                kind: "unstructured", sparsity: sparsity, iterations: 1
            )
            // Re-derive stateByteLength under the inference layout — the
            // total weight byte count doesn't change (zeros take the
            // same space as nonzeros), but the .tinygpt header writer
            // wants this kept consistent.
            file.header.stateByteLength = computeStateByteLength(file)
            do {
                try TinyGPTFileWriter.write(file, to: URL(fileURLWithPath: outPath))
            } catch { fputs("write failed: \(error)\n", stderr); exit(1) }
            reportFileSizes(input: inputURL, output: URL(fileURLWithPath: outPath))
            return
        }

        // Iterative Magnitude Pruning. Each round: prune the same
        // fraction K of REMAINING (non-zero) weights, then fine-tune
        // for `ftSteps` steps with the mask kept fixed (gradients
        // masked out so already-pruned weights stay zero).
        guard let corpusPath = corpusPath else {
            fputs("internal: --iterations > 1 without corpus\n", stderr); exit(1)
        }
        // Load the model into MLX so we can fine-tune. We currently
        // only support from-scratch TinyGPTModel for IMP — HF models
        // would require a parallel mask-application path; documented
        // in docs/pruning.md.
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(inPath) }
        catch { fputs("model load failed: \(error)\n", stderr); exit(1) }
        guard case .fromScratch(let model) = load.model else {
            fputs("IMP currently supports only from-scratch (.tinygpt) models — got HF\n", stderr)
            exit(1)
        }
        let cfg = load.config

        let corpus: ByteCorpus
        do { corpus = try ByteCorpus(contentsOf: URL(fileURLWithPath: corpusPath)) }
        catch { fputs("corpus read failed: \(error)\n", stderr); exit(1) }

        var maskByName: [String: [UInt8]] = [:]
        let perRoundSparsity = sparsity
        // For the printed final-sparsity estimate when fractions are
        // applied per-round to the *remaining* weights.
        var cumulativeKeep: Float = 1.0
        for round in 1...iterations {
            print("\n— round \(round)/\(iterations) (sparsity this round: \(perRoundSparsity)) —")
            // Materialise the current model's parameters into the
            // file's tensor buffers, then compute new masks (using
            // ABSOLUTE magnitudes from the just-fine-tuned weights).
            captureModelIntoFile(model: model, file: &file)
            for name in targetNames {
                guard let idx = file.tensors.firstIndex(where: { $0.entry.name == name }) else { continue }
                var floats = floatsFromTensor(file.tensors[idx])
                let currentMask = maskByName[name] ?? [UInt8](repeating: 1, count: floats.count)
                // We prune `perRoundSparsity` of the CURRENTLY NON-ZERO
                // weights. So we collect the live indices, compute a
                // magnitude mask on those, and OR back into the
                // existing mask.
                var liveIdx: [Int] = []
                liveIdx.reserveCapacity(floats.count)
                for i in 0..<floats.count where currentMask[i] == 1 { liveIdx.append(i) }
                if liveIdx.isEmpty { continue }
                var liveFloats = [Float](); liveFloats.reserveCapacity(liveIdx.count)
                for i in liveIdx { liveFloats.append(floats[i]) }
                let liveMask = Pruning.magnitudeMask(liveFloats, sparsity: perRoundSparsity)
                var newMask = currentMask
                for (j, i) in liveIdx.enumerated() {
                    if liveMask[j] == 0 { newMask[i] = 0 }
                }
                maskByName[name] = newMask
                Pruning.applyMask(&floats, mask: newMask)
                // Pack back to the file's actual on-disk dtype — must
                // not corrupt the byte layout (fp16 files expect 2
                // bytes per element, not 4).
                file.tensors[idx].weight = packFloats(floats, dtype: file.tensors[idx].dtype)
            }
            cumulativeKeep *= (1 - perRoundSparsity)
            print("  cumulative sparsity ≈ \(String(format: "%.1f%%", 100 * (1 - cumulativeKeep)))")
            fflush(stdout)
            // Apply the new masks back into the live model and
            // fine-tune for ftSteps. The masks are kept fixed; we
            // re-apply them after each optimizer step so the dead
            // weights don't drift back to nonzero.
            do { try writeMasksIntoModel(file: &file, model: model) }
            catch { fputs("warning: writeMasksIntoModel: \(error)\n", stderr) }
            if round < iterations {
                impFineTune(model: model, cfg: cfg, corpus: corpus,
                             masks: maskByName, batchSize: ftBatch,
                             steps: ftSteps, lr: ftLR)
            }
        }
        // Final snapshot of model → file, then write.
        captureModelIntoFile(model: model, file: &file)
        if emitMask {
            file.header.sparsityMasks = encodeMasks(maskByName)
        }
        file.header.pruningInfo = TinyGPTHeader.PruningInfo(
            kind: "unstructured", sparsity: 1 - cumulativeKeep, iterations: iterations
        )
        file.header.stateByteLength = computeStateByteLength(file)
        do {
            try TinyGPTFileWriter.write(file, to: URL(fileURLWithPath: outPath))
        } catch { fputs("write failed: \(error)\n", stderr); exit(1) }
        reportFileSizes(input: inputURL, output: URL(fileURLWithPath: outPath))
    }

    // MARK: - Helpers

    /// Names of the tensors we'll mask. Skips 1-D params (biases,
    /// LayerNorm gain) and embeddings by default.
    private static func pruneCandidates(file: TinyGPTFile, includeEmbeddings: Bool) -> [String] {
        var out: [String] = []
        for t in file.tensors {
            let name = t.entry.name
            if t.entry.shape.count != 2 { continue }
            if !includeEmbeddings {
                if name == "token_embedding.weight" || name == "position_embedding.weight" { continue }
            }
            // Skip lm_head (untied output) by default — it's tied
            // with token_embedding for our models. If it's separately
            // serialised we keep it for now (rare).
            out.append(name)
        }
        return out
    }

    /// One-shot magnitude prune across all targeted tensors. Returns
    /// (per-tensor masks, totals) for the run summary.
    private static func pruneOneShot(file: inout TinyGPTFile, targets: [String], sparsity: Float)
        -> (masks: [String: [UInt8]], totals: (zeroed: Int, total: Int))
    {
        var masks: [String: [UInt8]] = [:]
        var zeroed = 0
        var total = 0
        for name in targets {
            guard let idx = file.tensors.firstIndex(where: { $0.entry.name == name }) else { continue }
            var floats = floatsFromTensor(file.tensors[idx])
            let mask = Pruning.magnitudeMask(floats, sparsity: sparsity)
            Pruning.applyMask(&floats, mask: mask)
            // Write back, respecting on-disk dtype.
            file.tensors[idx].weight = packFloats(floats, dtype: file.tensors[idx].dtype)
            masks[name] = mask
            total += mask.count
            for b in mask where b == 0 { zeroed += 1 }
        }
        return (masks, (zeroed, total))
    }

    /// Encode the per-tensor masks to a JSON-friendly dict of
    /// `tensorName → base64(RLE bytes)`. Matches Manifest.swift's
    /// `sparsityMasks` field type.
    private static func encodeMasks(_ masks: [String: [UInt8]]) -> [String: String] {
        var out: [String: String] = [:]
        for (k, v) in masks {
            let rle = Pruning.encodeRLE(v)
            out[k] = rle.base64EncodedString()
        }
        return out
    }

    /// Read a tensor's raw weight bytes back as fp32 floats,
    /// expanding fp16 if necessary.
    private static func floatsFromTensor(_ tensor: TinyGPTTensor) -> [Float] {
        switch tensor.dtype {
        case .fp32:
            return tensor.weightFloats
        case .fp16:
            return tensor.weightFP16AsFloat32()
        }
    }

    /// Inverse of `floatsFromTensor` — pack a Float array back to
    /// the tensor's on-disk dtype.
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

    /// Snapshot the current model's parameter tree into the file's
    /// tensor list, preserving the file's on-disk dtype. This is the
    /// inverse of weight loading — note that the loader transposed
    /// Linear weights from WASM order to PyTorch order, so the save
    /// path must transpose back.
    private static func captureModelIntoFile(model: TinyGPTModel, file: inout TinyGPTFile) {
        let params = model.parameters().flattened()
        let paramMap: [String: MLXArray] = Dictionary(uniqueKeysWithValues: params)
        for (idx, tensor) in file.tensors.enumerated() {
            guard let mlx = paramMap[tensor.entry.name] else { continue }
            var arr = mlx
            // The file stores Linear weights in WASM `[in, out]` order;
            // the model's parameters are PyTorch `[out, in]`. So we
            // transpose 2-D Linear weights on the way out.
            if Train.isLinearWeightName(tensor.entry.name) && arr.shape.count == 2 {
                arr = arr.transposed()
            }
            eval(arr)
            let floats: [Float] = arr.asArray(Float.self)
            file.tensors[idx].weight = packFloats(floats, dtype: tensor.dtype)
        }
    }

    /// After updating the on-file tensor buffers with new masked
    /// weights, push those back into the model so the next forward
    /// pass sees the masked values. Used between IMP rounds.
    private static func writeMasksIntoModel(file: inout TinyGPTFile, model: TinyGPTModel) throws {
        // Easiest correct path: serialise the file to a temp .tinygpt
        // and reload via the canonical loader. Avoids re-implementing
        // the WASM→PyTorch transpose / nested-parameter merge.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("tinygpt-imp-\(UUID().uuidString).tinygpt")
        try TinyGPTFileWriter.write(file, to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        try TinyGPTWeightLoader.load(tmp, into: model)
    }

    /// Fine-tune for `steps` steps with masks applied after every
    /// optimizer step. Standard cross-entropy LM loss on the
    /// supplied byte corpus.
    private static func impFineTune(model: TinyGPTModel, cfg: ModelConfig,
                                     corpus: ByteCorpus,
                                     masks: [String: [UInt8]],
                                     batchSize: Int, steps: Int, lr: Float)
    {
        print("  fine-tune: \(steps) steps · batch \(batchSize) · lr \(lr)")
        let opt = AdamW(learningRate: lr, weightDecay: 0)
        let lossFn = { (m: TinyGPTModel, x: MLXArray, y: MLXArray) -> MLXArray in
            return m.loss(x, y)
        }
        let gradFn = valueAndGrad(model: model, lossFn)
        // Precompute MLX mask arrays per tensor (shape = model
        // parameter shape after the WASM→PyTorch transpose).
        let maskArrays: [String: MLXArray] = computeMaskArrays(model: model, masks: masks)

        var lossSum: Float = 0
        let logEvery = max(10, steps / 5)
        for step in 0..<steps {
            let (x, y) = corpus.sampleBatch(batchSize: batchSize, contextLength: cfg.contextLength)
            let (loss, grads) = gradFn(model, x, y)
            opt.update(model: model, gradients: grads)
            // Re-apply the masks: walk the model params, multiply each
            // masked param by the mask array. Important: we mutate
            // the parameters in place.
            applyMaskArraysToModel(model: model, masks: maskArrays)
            MLX.eval(loss, model, opt)
            lossSum += loss.item(Float.self)
            if (step + 1) % logEvery == 0 || step == steps - 1 {
                fputs("    step \(step + 1)/\(steps)  loss \(String(format: "%.3f", lossSum / Float(step + 1)))\n", stderr)
            }
        }
    }

    /// Build MLXArray masks matching each parameter's actual SHAPE
    /// (post WASM→PyTorch transpose). The on-disk masks are in WASM
    /// `[in, out]` order; the parameter we mask in-place is in
    /// PyTorch `[out, in]` order. So we transpose 2-D masks here.
    private static func computeMaskArrays(model: TinyGPTModel, masks: [String: [UInt8]]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        let params = model.parameters().flattened()
        for (name, value) in params {
            guard let mask = masks[name] else { continue }
            let shape = value.shape
            let floatMask: [Float] = mask.map { Float($0) }
            var arr = MLXArray(floatMask, shape.reversed().count == 2 ? [shape[1], shape[0]] : shape)
            if shape.count == 2 {
                // The on-disk mask order matches the on-disk weight
                // order ([in, out] for Linear). The model parameter
                // is [out, in], so transpose.
                arr = arr.transposed()
            }
            out[name] = arr.asType(value.dtype)
        }
        return out
    }

    /// Multiply each parameter by its mask in place. MLX-Swift
    /// modifies modules through `update(parameters:)` — we walk the
    /// flat parameter list, build a fresh nested dict where each
    /// masked leaf has been multiplied by its mask, and apply.
    private static func applyMaskArraysToModel(model: TinyGPTModel, masks: [String: MLXArray]) {
        // Build the masked replacement values by reading the current
        // (post-update) parameters and multiplying by the precomputed
        // mask arrays. Unmasked tensors are passed through untouched.
        var updates: [String: MLXArray] = [:]
        for (name, value) in model.parameters().flattened() {
            if let m = masks[name] {
                updates[name] = value * m
            }
        }
        // Walk the model's existing parameter tree and substitute
        // the masked values in place — same mechanism as the
        // weight loader uses to keep the nested array-vs-dict
        // structure correct.
        let nested = rewriteLeaves(model.parameters(), withFlat: updates)
        do {
            try model.update(parameters: nested, verify: [])
        } catch {
            fputs("warning: mask reapply failed: \(error)\n", stderr)
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
            if let v = flat[key] { return .value(v) }
            return item
        case .array(let elements):
            return .array(elements.enumerated().map { (idx, child) in
                rewriteItem(child, path: path + [String(idx)], flat: flat)
            })
        case .dictionary(let dict):
            var newDict: [String: NestedItem<String, MLXArray>] = [:]
            for (k, child) in dict {
                newDict[k] = rewriteItem(child, path: path + [k], flat: flat)
            }
            return .dictionary(newDict)
        }
    }

    private static func computeStateByteLength(_ file: TinyGPTFile) -> Int {
        switch file.header.bodyLayout {
        case .trainingFP32:
            return 4 + file.tensors.reduce(0) { $0 + 3 * $1.weight.count }
        case .inferenceFP16:
            return file.tensors.reduce(0) { $0 + $1.weight.count }
        }
    }

    private static func reportFileSizes(input: URL, output: URL) {
        let inSize = ((try? FileManager.default.attributesOfItem(atPath: input.path))?[.size] as? NSNumber)?.intValue ?? 0
        let outSize = ((try? FileManager.default.attributesOfItem(atPath: output.path))?[.size] as? NSNumber)?.intValue ?? 0
        let delta = outSize - inSize
        let pct = inSize > 0 ? Double(delta) / Double(inSize) * 100 : 0
        print("""

        FILE SIZE
          input:   \(formatBytes(inSize))
          output:  \(formatBytes(outSize))  (\(String(format: "%+.1f%%", pct)))

        Note: the .tinygpt format stores weights densely. Zeros take
        the same space as nonzeros, so a pruned file is the same size
        as the original PLUS the RLE mask in the header. The compression
        win only materialises when the file is then run through a
        general-purpose compressor (gzip/zstd) — which collapses runs of
        zeros aggressively. See docs/pruning.md for measured numbers.
        """)
        // Bonus: try gzipping both files and report the compressed
        // sizes — the realistic distribution-time number.
        if let (gzIn, gzOut) = gzipSizeBoth(input: input, output: output) {
            let pctC = gzIn > 0 ? Double(gzOut - gzIn) / Double(gzIn) * 100 : 0
            print("""

          gzipped:
            input:  \(formatBytes(gzIn))
            output: \(formatBytes(gzOut))  (\(String(format: "%+.1f%%", pctC)))
        """)
        }
    }

    /// Best-effort gzip of both files for the report. Shells out to
    /// /usr/bin/gzip into a tempdir. Returns nil on any failure.
    private static func gzipSizeBoth(input: URL, output: URL) -> (Int, Int)? {
        func gzSize(_ src: URL) -> Int? {
            let tmp = FileManager.default.temporaryDirectory
                .appendingPathComponent("\(src.lastPathComponent).\(UUID().uuidString).gz")
            defer { try? FileManager.default.removeItem(at: tmp) }
            let proc = Process()
            proc.launchPath = "/usr/bin/gzip"
            // -c writes to stdout; we redirect into the temp file.
            proc.arguments = ["-c", "-9", src.path]
            let pipe = Pipe()
            proc.standardOutput = pipe
            do { try proc.run() } catch { return nil }
            let handle = pipe.fileHandleForReading
            // Drain into the temp file. We do this synchronously so
            // small files (.tinygpt under 100MB) don't bottleneck.
            do {
                FileManager.default.createFile(atPath: tmp.path, contents: nil)
                let fh = try FileHandle(forWritingTo: tmp)
                while true {
                    let chunk = handle.availableData
                    if chunk.isEmpty { break }
                    fh.write(chunk)
                }
                try? fh.close()
            } catch { return nil }
            proc.waitUntilExit()
            guard proc.terminationStatus == 0 else { return nil }
            return ((try? FileManager.default.attributesOfItem(atPath: tmp.path))?[.size] as? NSNumber)?.intValue
        }
        guard let a = gzSize(input), let b = gzSize(output) else { return nil }
        return (a, b)
    }

    private static func formatBytes(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.2f MB", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.1f KB", Double(n) / 1_000) }
        return "\(n) B"
    }

    private static func formatLargeInt(_ n: Int) -> String {
        let f = NumberFormatter(); f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt prune-unstructured <model.tinygpt> [options]

        --out <path>            Where to save the pruned model — required
        --sparsity F            Fraction of weights to zero in each round
                                  (default 0.5). Final sparsity after N
                                  iterations is 1 - (1 - F)^N.
        --iterations N          IMP rounds (default 1). N > 1 needs --corpus.
        --corpus <path>         Text corpus for fine-tuning between rounds
        --ft-steps N            Steps per fine-tune round (default 100)
        --ft-batch N            Fine-tune batch size (default 8)
        --ft-lr F               Fine-tune learning rate (default 1e-4)
        --include-embeddings    Also prune token/position embeddings
                                  (default: off — pruning embeddings
                                  silently kills rare tokens)
        --no-mask               Don't store the mask in the header.
                                  Useful for distribution: zeros still
                                  compress, and dropping the mask
                                  saves the 1/8th-mask-storage cost.

        The output model has the SAME shapes as the input; weights below
        the magnitude threshold are zeroed and a 0/1 mask per tensor is
        stored in the header. Inference works unchanged (zeros multiply
        to zero) — Metal has no sparse matmul, so wallclock is unaffected.
        The compression win shows up after gzipping the file.
        """)
        exit(2)
    }
}
