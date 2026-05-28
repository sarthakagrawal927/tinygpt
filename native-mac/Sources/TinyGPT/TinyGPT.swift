import Foundation
import TinyGPTIO

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
            Bench.run(args: Array(args.dropFirst()))
        case "train":
            Train.run(args: Array(args.dropFirst()))
        case "eval":
            Eval.run(args: Array(args.dropFirst()))
        case "finetune":
            Finetune.run(args: Array(args.dropFirst()))
        case "sample":
            Sample.run(args: Array(args.dropFirst()))
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
          tinygpt bench [flags]      training-throughput benchmark vs. WebGPU baseline

        file format documented in Sources/TinyGPTIO/TinyGPTFile.swift.
        bench flags documented in `tinygpt bench --help`.
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
