import Foundation
import MLX
import MLXNN
import TinyGPTIO
import TinyGPTModel

/// `tinygpt extract` — run a trained tool-call extractor on a query.
///
/// Loads a router checkpoint + its `.labels.json` sidecar, encodes the
/// query, and prints the top-K predicted tool names with softmax
/// confidences.
///
/// FLAGS
///   <model.tinygpt>       Router checkpoint (positional, required)
///   --query "<text>"      Query to classify (required unless --stdin)
///   --stdin               Read query lines from stdin (one per line)
///   --top-k N             Print top-K predictions (default: 3)
///   --json                Print predictions as one JSON object per line
///   --threshold F         Only print predictions with prob >= F
///
/// EXAMPLES
///   tinygpt extract router.tinygpt --query "open foo.py and read it"
///   echo "find the bug" | tinygpt extract router.tinygpt --stdin
///   tinygpt extract router.tinygpt --query "..." --json --top-k 5
enum Extract {

    static func run(args: [String]) {
        var modelPath: String? = nil
        var query: String? = nil
        var stdin = false
        var topK = 3
        var jsonOut = false
        var threshold: Float = 0.0

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--query":
                guard i + 1 < args.count else { exitUsage() }
                query = args[i + 1]; i += 2
            case "--stdin":
                stdin = true; i += 1
            case "--top-k":
                guard i + 1 < args.count else { exitUsage() }
                topK = Int(args[i + 1]) ?? topK; i += 2
            case "--json":
                jsonOut = true; i += 1
            case "--threshold":
                guard i + 1 < args.count else { exitUsage() }
                threshold = Float(args[i + 1]) ?? threshold; i += 2
            case "-h", "--help":
                exitUsage()
            default:
                if args[i].hasPrefix("-") {
                    fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
                }
                modelPath = args[i]; i += 1
            }
        }
        guard let modelPath = modelPath else {
            fputs("extract: missing <model.tinygpt>\n", stderr); exitUsage()
        }
        if query == nil && !stdin {
            fputs("extract: pass --query or --stdin\n", stderr); exitUsage()
        }

        // Load.
        let url = URL(fileURLWithPath: modelPath)
        let labelsURL = ToolRouterLabels.sidecarURL(forCheckpoint: url)
        let router: ToolRouterModel
        let labels: ToolRouterLabels
        do {
            labels = try ToolRouterLabels.load(from: labelsURL)
        } catch {
            fputs("extract: could not load labels sidecar at \(labelsURL.path): \(error)\n", stderr)
            exit(1)
        }
        do {
            router = try ToolRouterLoader.load(path: modelPath, numClasses: labels.labels.count)
        } catch {
            fputs("extract: model load failed: \(error)\n", stderr); exit(1)
        }
        fputs("extract: \(modelPath)  ·  \(labels.labels.count) classes  ·  ctx \(router.config.contextLength)\n", stderr)

        // Predict.
        func predictOne(_ q: String) {
            let t0 = Date()
            let ids = encode(q, contextLength: router.config.contextLength,
                              vocabSize: router.config.vocabSize)
            let x = MLXArray(ids, [1, ids.count])
            let preds = router.topK(idx: x, k: topK)
            let elapsedMs = -t0.timeIntervalSinceNow * 1000.0
            if jsonOut {
                var arr: [[String: Any]] = []
                for p in preds where p.prob >= threshold {
                    let name = labels.labels[safe: p.classIdx] ?? "?"
                    arr.append([
                        "tool": name,
                        "prob": p.prob,
                        "class_idx": p.classIdx,
                    ])
                }
                let obj: [String: Any] = [
                    "query": q,
                    "predictions": arr,
                    "latency_ms": elapsedMs,
                ]
                if let d = try? JSONSerialization.data(withJSONObject: obj),
                   let s = String(data: d, encoding: .utf8) {
                    print(s)
                }
            } else {
                print("query: \(q)")
                print(String(format: "  latency: %.2f ms", elapsedMs))
                for p in preds where p.prob >= threshold {
                    let name = labels.labels[safe: p.classIdx] ?? "?"
                    print(String(format: "  %.4f  %@", p.prob, name))
                }
            }
        }

        if let q = query { predictOne(q) }
        if stdin {
            while let line = readLine(strippingNewline: true) {
                let t = line.trimmingCharacters(in: .whitespaces)
                if t.isEmpty { continue }
                predictOne(t)
            }
        }
    }

    // MARK: - Encoding

    /// Byte-level encode + truncate / pad to `contextLength`. Mirrors the
    /// trainer's encode function exactly — any mismatch produces gibberish
    /// classifications.
    static func encode(_ s: String, contextLength: Int, vocabSize: Int) -> [Int32] {
        var ids = [UInt8](s.utf8).prefix(contextLength).map { Int32($0) }
        for i in 0..<ids.count {
            if ids[i] < 0 || ids[i] >= Int32(vocabSize) {
                ids[i] = 0
            }
        }
        while ids.count < contextLength { ids.append(0) }
        return ids
    }

    static func exitUsage() -> Never {
        print("""
        usage: tinygpt extract <router.tinygpt> [--query "<text>" | --stdin] [flags]

          --query "<text>"     query to classify
          --stdin              read queries from stdin (one per line)
          --top-k N            print top-K (default: 3)
          --json               emit JSON, one object per line
          --threshold F        only print predictions with prob >= F

        Companion sidecar (auto-loaded): <router.tinygpt>.labels.json
        """)
        exit(2)
    }
}

private extension Array {
    subscript(safe idx: Int) -> Element? {
        return (idx >= 0 && idx < count) ? self[idx] : nil
    }
}
