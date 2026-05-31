import Foundation
import TinyGPTScreen
#if canImport(ImageIO)
import ImageIO
#endif

/// `tinygpt screen <capture|tree|both>` — Mac screen-reading capabilities.
///
/// Wave 2.6 scaffold for the demonstration-specialist screen reader. We
/// ship the *data-capture* half here:
///
///   tinygpt screen capture --out window.png
///       Snapshot the active window via ScreenCaptureKit (macOS 14+).
///
///   tinygpt screen tree [--out tree.json]
///       Dump the focused window's macOS Accessibility (AX) tree as JSON.
///       Prints to stdout if --out is omitted.
///
///   tinygpt screen both --out-dir /tmp/snap
///       Both of the above, side-by-side, into <out-dir>/window.png +
///       <out-dir>/tree.json.
///
/// The vision-encoder → tinygpt-decoder half (consuming the PNG and
/// emitting tokens) is intentionally *not* in this commit — that's
/// research-grade work tracked separately in the roadmap. The AX tree
/// is the more useful half for tool-calling SLMs anyway.
enum Screen {
    static func run(args: [String]) {
        guard let sub = args.first else { exitUsage() }
        let rest = Array(args.dropFirst())
        switch sub {
        case "capture":  runCapture(args: rest)
        case "tree":     runTree(args: rest)
        case "both":     runBoth(args: rest)
        case "-h", "--help": exitUsage()
        default:
            fputs("screen: unknown subcommand '\(sub)'\n", stderr)
            exitUsage()
        }
    }

    // MARK: - capture

    private static func runCapture(args: [String]) {
        var outPath: String?
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--out": outPath = args[i+1]; i += 2
            case "-h", "--help": exitUsage()
            default:
                fputs("capture: unknown flag '\(args[i])'\n", stderr); exitUsage()
            }
        }
        guard let outPath = outPath else {
            fputs("screen capture: --out <path.png> required\n", stderr); exit(2)
        }
        do {
            let img = try ScreenCapture.captureActiveWindow()
            let bytes = try ScreenCapture.writePNG(img, to: outPath)
            print("wrote \(bytes) bytes to \(outPath) (\(img.width)×\(img.height))")
        } catch {
            fputs("\(error)\n", stderr)
            exit(1)
        }
    }

    // MARK: - tree

    private static func runTree(args: [String]) {
        var outPath: String?
        var maxDepth = AccessibilityTree.defaultMaxDepth
        var maxChildren = AccessibilityTree.defaultMaxChildrenPerNode
        var pretty = true
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--out": outPath = args[i+1]; i += 2
            case "--max-depth": maxDepth = Int(args[i+1]) ?? maxDepth; i += 2
            case "--max-children": maxChildren = Int(args[i+1]) ?? maxChildren; i += 2
            case "--compact": pretty = false; i += 1
            case "-h", "--help": exitUsage()
            default:
                fputs("tree: unknown flag '\(args[i])'\n", stderr); exitUsage()
            }
        }
        do {
            let node = try AccessibilityTree.readFocused(
                maxDepth: maxDepth,
                maxChildrenPerNode: maxChildren
            )
            let json = try encode(node, pretty: pretty)
            if let outPath = outPath {
                try json.write(to: URL(fileURLWithPath: outPath))
                print("wrote AX tree to \(outPath) (\(json.count) bytes)")
            } else {
                FileHandle.standardOutput.write(json)
                if pretty { FileHandle.standardOutput.write(Data([0x0a])) }
            }
        } catch {
            fputs("\(error)\n", stderr)
            exit(1)
        }
    }

    // MARK: - both

    private static func runBoth(args: [String]) {
        var outDir: String?
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--out-dir": outDir = args[i+1]; i += 2
            case "-h", "--help": exitUsage()
            default:
                fputs("both: unknown flag '\(args[i])'\n", stderr); exitUsage()
            }
        }
        guard let outDir = outDir else {
            fputs("screen both: --out-dir <path> required\n", stderr); exit(2)
        }
        do {
            try FileManager.default.createDirectory(
                atPath: outDir, withIntermediateDirectories: true
            )
        } catch {
            fputs("create dir failed: \(error)\n", stderr); exit(1)
        }

        let pngPath = (outDir as NSString).appendingPathComponent("window.png")
        let jsonPath = (outDir as NSString).appendingPathComponent("tree.json")

        // We capture image then tree — but if either half fails (e.g.
        // missing Screen Recording permission) we still try the other,
        // since the AX tree is independently useful and uses a different
        // permission. Failures are surfaced after both runs.
        var captureError: Error?
        var treeError: Error?
        var capturedImage: (w: Int, h: Int, bytes: Int)?
        var treeBytes: Int?

        do {
            let img = try ScreenCapture.captureActiveWindow()
            let bytes = try ScreenCapture.writePNG(img, to: pngPath)
            capturedImage = (img.width, img.height, bytes)
        } catch {
            captureError = error
        }
        do {
            let node = try AccessibilityTree.readFocused()
            let json = try encode(node, pretty: true)
            try json.write(to: URL(fileURLWithPath: jsonPath))
            treeBytes = json.count
        } catch {
            treeError = error
        }

        if let info = capturedImage {
            print("capture: wrote \(info.bytes) bytes to \(pngPath) (\(info.w)×\(info.h))")
        } else if let e = captureError {
            fputs("capture failed: \(e)\n", stderr)
        }
        if let n = treeBytes {
            print("tree:    wrote \(n) bytes to \(jsonPath)")
        } else if let e = treeError {
            fputs("tree failed: \(e)\n", stderr)
        }
        if captureError != nil && treeError != nil { exit(1) }
    }

    // MARK: - helpers

    private static func encode(_ node: AXNode, pretty: Bool) throws -> Data {
        let enc = JSONEncoder()
        if pretty {
            enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        } else {
            enc.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        }
        return try enc.encode(node)
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt screen <capture|tree|both> [flags]

        Screen-reading capabilities for the demonstration specialist.
        Wave 2.6 scaffold — data capture only; vision-encoder integration
        is research-grade work tracked separately.

        Subcommands:
          capture --out <path.png>
            Snapshot the active window via ScreenCaptureKit and write PNG.

          tree [--out <path.json>] [--max-depth N] [--max-children N] [--compact]
            Dump the focused window's macOS Accessibility (AX) tree as
            JSON. Prints to stdout if --out is omitted.

          both --out-dir <dir>
            Capture both image + AX tree side-by-side into <dir>/window.png
            and <dir>/tree.json.

        Permissions (one-time setup; macOS will prompt on first use):
          • capture / both — System Settings → Privacy & Security →
                             Screen Recording → enable for your terminal
                             (or the tinygpt binary).
          • tree    / both — System Settings → Privacy & Security →
                             Accessibility → enable for your terminal
                             (or the tinygpt binary).
        Both permissions are tied to the *signed bundle identifier* of
        the calling process; the simplest workflow is to launch tinygpt
        from a terminal that already has the relevant grants.
        """)
        exit(2)
    }
}
