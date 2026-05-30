import Foundation
import TinyGPTIO
import TinyGPTBench
import TinyGPTServe

/// CLI entry point. Mirrors `python_ref/load_tinygpt.py --inspect`.
/// Subcommands:
///   tinygpt inspect <path>     — print the file's manifest + metadata
///   tinygpt validate <path>    — read and re-encode, exit 0 iff round-trips
///                                bit-identically (sanity check for writers)
///
/// Once the model + training milestones land, this entry point grows
/// `train` and `sample` subcommands. For M1 it's read-only.
@main
struct TinyGPT {
    static func main() {
        let args = Array(CommandLine.arguments.dropFirst())
        guard let cmd = args.first else {
            printUsage()
            exit(2)
        }
        // Pre-switch shim for the score-bench subcommand: dispatched here
        // (rather than as a `case` inside the main switch) so the formal
        // case-dispatch list stays unchanged until the merge PR — see
        // // TODO(score-bench-merge) below.
        if cmd == "score-bench" {
            Score.run(args: Array(args.dropFirst()))
            return
        }
        // Same pre-switch shim pattern for `train-heads` — see the merge
        // TODO note next to `tuned-lens` in the switch below. Other agents
        // are concurrently touching this switch; we ship the shim to keep
        // the surface area testable until the speculative-heads PR lands.
        if cmd == "train-heads" {
            TrainHeads.run(args: Array(args.dropFirst()))
            return
        }
        // TODO(score-bench-merge): once review is happy, move the dispatch
        // for `score-bench` into the case below (next to `case "eval":`)
        // and delete the pre-switch shim above.
        //
        // Same pre-switch shim treatment for `gptq` — the from-scratch
        // GPTQ worker (Sources/TinyGPT/GPTQ.swift::GPTQWorker.run) is
        // still smoke-testing, so we dispatch it here rather than wire
        // a `case "gptq":` inside the switch. See TODO(gptq-cli) below.
        if cmd == "gptq" {
            GPTQWorker.run(args: Array(args.dropFirst()))
            return
        }
        switch cmd {
        case "inspect":
            guard let path = args.dropFirst().first else {
                fputs("inspect: missing <path>\n", stderr); exit(2)
            }
            run { try inspect(path: path) }
        case "validate":
            guard let path = args.dropFirst().first else {
                fputs("validate: missing <path>\n", stderr); exit(2)
            }
            run { try validate(path: path) }
        case "bench":
            // Inference-side LLM benchmark harness (Bench360-modelled).
            // See docs/benchmark_harness_design.md.
            Benchmark.run(args: Array(args.dropFirst()))
        case "bench-train":
            // Legacy training-throughput benchmark vs the WebGPU browser
            // baseline. Used to be `tinygpt bench` before the inference
            // harness shipped — preserved under a new name.
            Bench.run(args: Array(args.dropFirst()))
        case "train":
            Train.run(args: Array(args.dropFirst()))
        case "eval":
            Eval.run(args: Array(args.dropFirst()))
        case "finetune":
            Finetune.run(args: Array(args.dropFirst()))
        case "sft":
            SFT.run(args: Array(args.dropFirst()))
        case "dpo":
            DPO.run(args: Array(args.dropFirst()))
        case "distill":
            Distill.run(args: Array(args.dropFirst()))
        case "es":
            ES.run(args: Array(args.dropFirst()))
        case "laser":
            LASER.run(args: Array(args.dropFirst()))
        case "hqq":
            HQQ.run(args: Array(args.dropFirst()))
        // TODO(gptq-cli): wire `case "gptq":` here once the from-scratch
        // GPTQ worker (Sources/TinyGPT/GPTQ.swift::GPTQWorker.run) finishes
        // its smoke-test cycle. Dispatch shape would be:
        //     case "gptq":
        //         GPTQWorker.run(args: Array(args.dropFirst()))
        // Marker left deliberately so the build still surfaces unused-
        // file warnings (forcing us to either ship or remove) and so
        // the help text gets a matching entry in `printUsage`.
        case "magpie":
            Magpie.run(args: Array(args.dropFirst()))
        case "tuned-lens":
            TunedLens.run(args: Array(args.dropFirst()))
        // TODO(train-heads-merge): wire up `tinygpt train-heads` once the
        // speculative-heads PR (MedusaHeads.swift / EagleDraft.swift /
        // TrainHeads.swift) is merged. Until then a manual case here would
        // race other agents touching this switch — see HANDOFF.md.
        //   case "train-heads":
        //       TrainHeads.run(args: Array(args.dropFirst()))
        case "compare":
            Compare.run(args: Array(args.dropFirst()))
        case "hf-inspect":
            HFInspect.run(args: Array(args.dropFirst()))
        case "hf-load":
            HFLoad.run(args: Array(args.dropFirst()))
        case "sample":
            Sample.run(args: Array(args.dropFirst()))
        case "serve":
            Serve.run(args: Array(args.dropFirst()))
        case "debug-names":
            DebugNames.run(args: Array(args.dropFirst()))
        case "debug-load":
            DebugNames.compareLoaded(args: Array(args.dropFirst()))
        case "debug-logits":
            DebugNames.logits(args: Array(args.dropFirst()))
        case "debug-dtypes":
            DebugNames.dtypes(args: Array(args.dropFirst()))
        case "debug-loss":
            DebugNames.sanityLoss(args: Array(args.dropFirst()))
        case "-h", "--help":
            printUsage()
        default:
            fputs("unknown subcommand: \(cmd)\n\n", stderr)
            printUsage()
            exit(2)
        }
    }

    private static func run(_ body: () throws -> Void) {
        do {
            try body()
        } catch {
            fputs("error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func printUsage() {
        print("""
        tinygpt — native-side CLI for the .tinygpt file format and training

        usage:
          tinygpt inspect <path>     print manifest + metadata for a .tinygpt file
          tinygpt validate <path>    round-trip check: read → encode → byte-compare
          tinygpt bench [flags]      inference-side LLM benchmark harness (Bench360-modelled)
          tinygpt bench-train [flags] training-throughput benchmark vs. WebGPU baseline

        file format documented in Sources/TinyGPTIO/TinyGPTFile.swift.
        bench flags documented in `tinygpt bench --help`.
        bench harness design documented in docs/benchmark_harness_design.md.
        """)
    }

    static func inspect(path: String) throws {
        let url = URL(fileURLWithPath: path)
        let file = try TinyGPTFileReader.read(url)
        let header = file.header

        print("\nFile: \(url.path)")
        print(String(repeating: "-", count: 64))
        print("Version: \(file.version)")
        print("Step:    \(file.step)")

        print("\nConfig:")
        if let v = header.config.layers   { print("  layers     \(v)") }
        if let v = header.config.dModel   { print("  dModel     \(v)") }
        if let v = header.config.ctx      { print("  ctx        \(v)") }
        if let v = header.config.heads    { print("  heads      \(v)") }
        if let v = header.config.dMlp     { print("  dMlp       \(v)") }
        if let v = header.config.batchSize { print("  batchSize  \(v)") }
        if let v = header.config.backend  { print("  backend    \(v)") }

        if let fl = header.finalLoss {
            var line = "  final loss"
            if let t = fl.train { line += "  train \(format(t))" }
            if let v = fl.val   { line += ", val \(format(v))" }
            if let s = fl.step  { line += " @ step \(s)" }
            print(line)
        }

        if let sample = header.sample, !sample.isEmpty {
            let snippet = sample.prefix(80)
            print("\n  sample: \(snippet.debugDescription)")
        }

        print("\nTensors (\(file.tensors.count)):")
        var total = 0
        for tensor in file.tensors {
            let n = tensor.entry.elementCount
            total += n
            let shape = "\(tensor.entry.shape)"
            print("  \(pad(tensor.entry.name, 40))  \(pad(shape, 20))  \(formatCount(n))")
        }
        print(String(repeating: "-", count: 64))
        print("  total parameters: \(formatCount(total))")
    }

    static func validate(path: String) throws {
        let url = URL(fileURLWithPath: path)
        let original = try Data(contentsOf: url)
        let file = try TinyGPTFileReader.decode(original, source: url)
        let reencoded = try TinyGPTFileWriter.encode(file)

        if original == reencoded {
            print("OK: \(url.lastPathComponent) round-trips bit-identically (\(original.count) bytes)")
            return
        }

        // If the bytes differ, the most common causes are JSON key ordering
        // (the browser doesn't sort keys) and re-encoded loss-history (passed
        // through as raw JSON, whose nested objects also re-order). Fall back
        // to a semantic comparison of the modeled fields.
        let reDecoded = try TinyGPTFileReader.decode(reencoded, source: url)
        let headersMatch =
            file.header.config == reDecoded.header.config
            && file.header.manifest == reDecoded.header.manifest
            && file.header.savedAt == reDecoded.header.savedAt
            && file.header.finalLoss == reDecoded.header.finalLoss
            && file.header.sample == reDecoded.header.sample
            && file.header.weightDtype == reDecoded.header.weightDtype
            && file.header.includesOptimizerState == reDecoded.header.includesOptimizerState
            && file.header.stateByteLength == reDecoded.header.stateByteLength
        let tensorsMatch =
            file.step == reDecoded.step
            && file.tensors.count == reDecoded.tensors.count
            && zip(file.tensors, reDecoded.tensors).allSatisfy { a, b in
                a.weight == b.weight && a.adamM == b.adamM && a.adamV == b.adamV
            }
        if headersMatch && tensorsMatch {
            print("OK (semantic): \(url.lastPathComponent) round-trips with reordered JSON keys")
            print("    original \(original.count) bytes, re-encoded \(reencoded.count) bytes")
            return
        }
        fputs("FAIL: \(url.lastPathComponent) does not round-trip\n", stderr)
        if !headersMatch { fputs("  reason: header field mismatch\n", stderr) }
        if !tensorsMatch { fputs("  reason: tensor body mismatch\n", stderr) }
        exit(1)
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.3f", value)
    }

    private static func formatCount(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }
}
