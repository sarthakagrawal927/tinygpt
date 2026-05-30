import Foundation
import TinyGPTData

/// `tinygpt list-datasets [--specialist <kind>]` —
/// pretty-print the curated dataset registry. The registry lives in
/// TinyGPTData/DatasetRegistry.swift; this command is just the lens.
///
/// USAGE
///   tinygpt list-datasets                          # everything
///   tinygpt list-datasets --specialist tool-calling
///   tinygpt list-datasets --specialist math
///   tinygpt list-datasets --info Salesforce/xlam-function-calling-60k
enum ListDatasets {

    static func run(args: [String]) {
        var specialist: DatasetSpecialist?
        var infoId: String?
        var format: CorpusFormat?
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--specialist":
                guard i+1 < args.count, let s = DatasetSpecialist(parsing: args[i+1]) else {
                    let opts = DatasetSpecialist.allCases.map(\.rawValue).joined(separator: ", ")
                    fputs("--specialist requires one of: \(opts)\n", stderr)
                    exit(2)
                }
                specialist = s; i += 2
            case "--info":
                infoId = args[i+1]; i += 2
            case "--format":
                guard i+1 < args.count, let f = CorpusFormat(parsing: args[i+1]) else {
                    fputs("--format requires sft|dpo|plain\n", stderr); exit(2)
                }
                format = f; i += 2
            case "-h", "--help":
                printUsage(); return
            default:
                fputs("unknown arg: \(args[i])\n", stderr); printUsage(); exit(2)
            }
        }

        if let id = infoId {
            guard let e = DatasetRegistry.entry(id: id) else {
                fputs("not in registry: \(id)\n", stderr); exit(1)
            }
            printEntryDetailed(e)
            return
        }

        var entries = DatasetRegistry.entries(for: specialist)
        if let f = format { entries = entries.filter { $0.format == f } }

        if entries.isEmpty {
            print("(no matching datasets in registry)"); return
        }

        // Group by specialist for the all-list view.
        if specialist == nil && format == nil {
            print("tinygpt curated dataset registry  (\(entries.count) entries)")
            print(String(repeating: "=", count: 72))
            for spec in DatasetSpecialist.allCases {
                let group = DatasetRegistry.entries(for: spec)
                if group.isEmpty { continue }
                print("\n[\(spec.rawValue)]")
                for e in group { printEntryShort(e) }
            }
            print("\nuse --specialist <kind> to filter; --info <id> for full notes.")
        } else {
            let header = specialist.map { "specialist: \($0.rawValue)" }
                ?? format.map { "format: \($0.rawValue)" }
                ?? "all"
            print("tinygpt datasets — \(header)  (\(entries.count) entries)")
            print(String(repeating: "=", count: 72))
            for e in entries { printEntryShort(e) }
            print("")
            print("download with:  tinygpt download-dataset hf://datasets/<id>")
            // For specialists with GitHub recipes (debugger / code), also
            // list the curated repos. The HF datasets are pre-packaged;
            // the GitHub recipes are live signal — both belong in the
            // user's mental model of "where do I get training data for
            // this specialist".
            if let spec = specialist {
                let recipes = GitHubRecipes.entries(for: spec)
                if !recipes.isEmpty {
                    print("")
                    print("github recipes — \(spec.rawValue)  (\(recipes.count) repos)")
                    print(String(repeating: "-", count: 72))
                    for r in recipes { printRecipeShort(r) }
                    print("")
                    print("fetch with:     tinygpt fetch-github <owner/repo> --kind issues-prs")
                }
            }
        }
    }

    private static func printRecipeShort(_ r: GitHubRecipe) {
        let kinds = r.recommendedKinds.joined(separator: ",")
        print("  \(pad(r.repo, 32))  \(pad(r.language, 16))  \(pad(r.approxBugIssues, 26))  \(kinds)")
    }

    private static func printEntryShort(_ e: RegistryEntry) {
        let gatedTag = e.gated ? " [gated]" : ""
        let specs = e.specialists.map(\.rawValue).joined(separator: ",")
        print("  \(pad(e.id, 50))  \(pad(e.format.rawValue, 5))  \(pad(e.approxSize, 14))  \(specs)\(gatedTag)")
    }

    private static func printEntryDetailed(_ e: RegistryEntry) {
        print("""

        \(e.id)
        \(String(repeating: "-", count: e.id.count))
        format:        \(e.format.rawValue)
        specialists:   \(e.specialists.map(\.rawValue).joined(separator: ", "))
        approx size:   \(e.approxSize)
        license:       \(e.license)
        gated:         \(e.gated ? "yes (HF_TOKEN required)" : "no")

        notes:
        \(e.notes)

        download:
          tinygpt download-dataset hf://datasets/\(e.id) --format \(e.format.rawValue)

        """)
    }

    private static func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }

    private static func printUsage() {
        let specs = DatasetSpecialist.allCases.map(\.rawValue).joined(separator: " | ")
        print("""
        usage:
          tinygpt list-datasets                          show full registry
          tinygpt list-datasets --specialist <kind>      filter by specialist
          tinygpt list-datasets --format sft|dpo|plain   filter by target format
          tinygpt list-datasets --info <id>              detailed info for one entry

        specialists: \(specs)
        """)
    }
}
