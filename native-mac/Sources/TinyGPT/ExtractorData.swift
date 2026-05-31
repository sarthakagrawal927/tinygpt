import Foundation
import TinyGPTData
import TinyGPTServe

/// `tinygpt extractor-data` — build a (query, tool_name) JSONL corpus
/// for training the tool-call extractor / mini-router.
///
/// The router is a tiny BERT-class classifier; its training signal is a
/// flat list of `{"query": "<user text>", "tool": "<tool_name>"}` pairs.
/// This subcommand sources those pairs from three places:
///
///   1. **BFCL** — Berkeley Function-Calling Leaderboard, the canonical
///      function-calling eval. Distributed on HF Hub as
///      `gorilla-llm/Berkeley-Function-Calling-Leaderboard`. Each row
///      has a user `question` + an oracle `function` call (we extract
///      the function name).
///
///   2. **τ-bench** (tau-bench) — Sierra Research's tool-use benchmark
///      with two domains (retail, airline). Cloned from
///      https://github.com/sierra-research/tau-bench. Each task has a
///      `user_instruction` + an expected `tools_used[0]` (we take the
///      first one as the supervised label — multi-tool sequences get
///      pinned to the first call).
///
///   3. **Synthetic** (`--synth`) — for low-resource tools (≤ 5 real
///      examples after BFCL+τ-bench), the CLI queries CloudEscalate
///      (Claude/GPT) to generate 30-50 plausible user queries that
///      would call that tool. Bootstraps the long tail.
///
/// Output: one `{"query": "...", "tool": "name"}` per line.
///
/// FLAGS:
///   --bfcl <path>          Path to a downloaded BFCL JSONL file
///                          (see `tinygpt download-dataset` to grab it)
///   --tau-bench <dir>      Path to a tau-bench checkout's `data/`
///                          directory (the JSON task files)
///   --tools <schema.json>  Tool catalog (OpenAI tool schema). Used
///                          to determine the label set + which tools
///                          need --synth backfill.
///   --synth                Generate synthetic examples via cloud for
///                          underrepresented tools.
///   --synth-provider P     anthropic | openai (default: anthropic)
///   --synth-model M        Provider-specific model id
///   --synth-per-tool N     Synthetic examples per low-resource tool
///                          (default: 40)
///   --min-examples N       Threshold below which a tool is "low
///                          resource" and gets --synth backfill
///                          (default: 5)
///   --out <path>           Output JSONL path (default:
///                          ~/.cache/tinygpt/router/router_data.jsonl)
///   --dry-run              Parse + report stats, don't write
///
/// EXAMPLES
///   # Step 1: fetch BFCL via the existing dataset downloader
///   tinygpt download-dataset hf://datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
///
///   # Step 2: clone tau-bench (small repo, do this manually)
///   git clone https://github.com/sierra-research/tau-bench ~/code/tau-bench
///
///   # Step 3: build the training corpus
///   tinygpt extractor-data \
///     --bfcl ~/.cache/tinygpt/datasets/gorilla-llm/.../corpus.jsonl \
///     --tau-bench ~/code/tau-bench/tau_bench/envs \
///     --tools my_agent_tools.json \
///     --out router_data.jsonl
///
///   # Step 4 (optional): add synthetic examples for rare tools
///   tinygpt extractor-data --tools my_agent_tools.json --synth \
///     --bfcl ... --out router_data.jsonl
enum ExtractorData {

    static func run(args: [String]) {
        var bfclPath: String? = nil
        var tauBenchDir: String? = nil
        var toolsPath: String? = nil
        var synth = false
        var synthProvider = "anthropic"
        var synthModel: String? = nil
        var synthPerTool = 40
        var minExamples = 5
        var outPath: String? = nil
        var dryRun = false

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--bfcl":
                guard i + 1 < args.count else { exitUsage() }
                bfclPath = args[i + 1]; i += 2
            case "--tau-bench":
                guard i + 1 < args.count else { exitUsage() }
                tauBenchDir = args[i + 1]; i += 2
            case "--tools":
                guard i + 1 < args.count else { exitUsage() }
                toolsPath = args[i + 1]; i += 2
            case "--synth":
                synth = true; i += 1
            case "--synth-provider":
                guard i + 1 < args.count else { exitUsage() }
                synthProvider = args[i + 1]; i += 2
            case "--synth-model":
                guard i + 1 < args.count else { exitUsage() }
                synthModel = args[i + 1]; i += 2
            case "--synth-per-tool":
                guard i + 1 < args.count else { exitUsage() }
                synthPerTool = Int(args[i + 1]) ?? synthPerTool; i += 2
            case "--min-examples":
                guard i + 1 < args.count else { exitUsage() }
                minExamples = Int(args[i + 1]) ?? minExamples; i += 2
            case "--out":
                guard i + 1 < args.count else { exitUsage() }
                outPath = args[i + 1]; i += 2
            case "--dry-run":
                dryRun = true; i += 1
            case "-h", "--help":
                exitUsage()
            default:
                fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
            }
        }

        // Load the tool catalog (defines the label set).
        let toolNames: [String]
        if let tools = toolsPath {
            do {
                let schema = try ToolSchema.load(from: URL(fileURLWithPath: tools))
                toolNames = schema.tools.map { $0.name }
                fputs("extractor-data: loaded \(toolNames.count) tools from \(tools)\n", stderr)
            } catch {
                fputs("extractor-data: failed to load tools schema: \(error)\n", stderr)
                exit(1)
            }
        } else if synth {
            fputs("extractor-data: --synth requires --tools (need the catalog to know what to synthesise)\n", stderr)
            exit(2)
        } else {
            toolNames = []
            fputs("extractor-data: no --tools schema given; accepting all tool names found in inputs\n", stderr)
        }

        // Resolve output path.
        let outURL: URL = {
            if let p = outPath { return URL(fileURLWithPath: p) }
            let home = FileManager.default.homeDirectoryForCurrentUser
            let dir = home
                .appendingPathComponent(".cache")
                .appendingPathComponent("tinygpt")
                .appendingPathComponent("router")
            try? FileManager.default.createDirectory(at: dir,
                                                     withIntermediateDirectories: true)
            return dir.appendingPathComponent("router_data.jsonl")
        }()

        // Collected (query, tool) pairs. Counts per-tool so we know who
        // needs synth backfill.
        var pairs: [(query: String, tool: String)] = []
        var counts: [String: Int] = [:]
        let allowed = toolNames.isEmpty ? nil : Set(toolNames)

        // ---- BFCL ----
        if let path = bfclPath {
            let url = URL(fileURLWithPath: path)
            do {
                let added = try ingestBFCL(url: url, allowed: allowed,
                                            pairs: &pairs, counts: &counts)
                fputs("extractor-data: BFCL → \(added) pairs from \(path)\n", stderr)
            } catch {
                fputs("extractor-data: BFCL ingest failed: \(error)\n", stderr)
            }
        }

        // ---- τ-bench ----
        if let dir = tauBenchDir {
            let url = URL(fileURLWithPath: dir)
            do {
                let added = try ingestTauBench(dir: url, allowed: allowed,
                                                pairs: &pairs, counts: &counts)
                fputs("extractor-data: τ-bench → \(added) pairs from \(dir)\n", stderr)
            } catch {
                fputs("extractor-data: τ-bench ingest failed: \(error)\n", stderr)
            }
        }

        // ---- Synth ----
        if synth {
            guard !toolNames.isEmpty else {
                fputs("extractor-data: --synth requires --tools\n", stderr)
                exit(2)
            }
            guard let provider = CloudEscalate.Provider(rawValue: synthProvider.lowercased()) else {
                fputs("extractor-data: --synth-provider must be anthropic|openai\n", stderr)
                exit(2)
            }
            let low = toolNames.filter { (counts[$0] ?? 0) < minExamples }
            fputs("extractor-data: \(low.count)/\(toolNames.count) tools below threshold (\(minExamples)) — generating synthetic\n", stderr)
            for name in low {
                do {
                    // Look up the tool description from the schema so
                    // the cloud prompt is grounded in the actual tool
                    // semantics. Re-open the schema (cheap).
                    let descr = try toolDescription(name: name, toolsPath: toolsPath)
                    let added = try synthesise(
                        toolName: name, description: descr,
                        n: synthPerTool, provider: provider, model: synthModel,
                        pairs: &pairs, counts: &counts)
                    fputs("extractor-data: synth \(name) → \(added) queries\n", stderr)
                } catch {
                    fputs("extractor-data: synth(\(name)) failed: \(error)\n", stderr)
                }
            }
        }

        // ---- Report ----
        fputs("\nextractor-data: corpus stats\n", stderr)
        let sortedCounts = counts.sorted { $0.value > $1.value }
        for (name, c) in sortedCounts.prefix(20) {
            fputs("  \(name.padding(toLength: 32, withPad: " ", startingAt: 0))  \(c)\n", stderr)
        }
        if sortedCounts.count > 20 {
            fputs("  ... (+\(sortedCounts.count - 20) more)\n", stderr)
        }
        fputs("  total: \(pairs.count) pairs across \(counts.count) tools\n", stderr)

        // ---- Write ----
        if dryRun {
            fputs("--dry-run: skipping write phase\n", stderr)
            return
        }
        do {
            try writeJSONL(pairs: pairs, to: outURL)
            fputs("extractor-data: wrote \(pairs.count) pairs → \(outURL.path)\n", stderr)
        } catch {
            fputs("extractor-data: write failed: \(error)\n", stderr)
            exit(1)
        }
    }

    // MARK: - BFCL ingest

    /// BFCL rows have varied shapes across the leaderboard (single-call,
    /// multi-turn, parallel, irrelevance). For the router we only need
    /// (query, tool_name); we pull from the most common fields and skip
    /// rows that don't fit. The dataset typically ships as JSONL.
    static func ingestBFCL(url: URL, allowed: Set<String>?,
                            pairs: inout [(query: String, tool: String)],
                            counts: inout [String: Int]) throws -> Int {
        var added = 0
        _ = try RowReader.readRows(url: url) { row in
            // Field aliases — BFCL uses different keys across categories:
            //   "question" or "user_query" or "prompt" or "task"
            //   "function" (object or array) or "tool_name" or "answer"
            let query: String? = stringFrom(row, keys: ["question", "user_query", "prompt", "task", "user_input"])
                ?? extractFirstUserMessage(row)
            let toolName: String? = extractToolName(row)

            guard let q = query, let t = toolName, !q.isEmpty, !t.isEmpty else {
                return true
            }
            if let allow = allowed, !allow.contains(t) { return true }
            pairs.append((query: q, tool: t))
            counts[t, default: 0] += 1
            added += 1
            return true
        }
        return added
    }

    /// Find a tool/function name in a BFCL-style row. The leaderboard
    /// stores oracle calls in any of:
    ///   - top-level "function" or "tool_name" as a String
    ///   - "function": {"name": "..."}
    ///   - "function": [ {"name": "..."} ]
    ///   - "answer": [ {"name": "..."} ]
    ///   - "ground_truth": same shapes
    static func extractToolName(_ row: [String: Any]) -> String? {
        for key in ["function", "tool_name", "answer", "ground_truth", "expected_function"] {
            if let s = row[key] as? String, !s.isEmpty { return s }
            if let obj = row[key] as? [String: Any],
               let n = obj["name"] as? String, !n.isEmpty { return n }
            if let arr = row[key] as? [Any],
               let first = arr.first as? [String: Any],
               let n = first["name"] as? String, !n.isEmpty { return n }
            // Sometimes the array element is itself a string like
            // "fn(arg=1)" — strip to the part before "(".
            if let arr = row[key] as? [Any],
               let first = arr.first as? String, !first.isEmpty {
                if let paren = first.firstIndex(of: "(") {
                    return String(first[..<paren])
                }
                return first
            }
        }
        return nil
    }

    /// BFCL multi-turn rows put the query in `messages[*]` (ChatML-ish).
    /// Find the first user message.
    static func extractFirstUserMessage(_ row: [String: Any]) -> String? {
        guard let msgs = row["messages"] as? [[String: Any]] else { return nil }
        for m in msgs {
            if (m["role"] as? String) == "user", let c = m["content"] as? String {
                return c
            }
        }
        return nil
    }

    // MARK: - tau-bench ingest

    /// τ-bench's task files live as JSON under
    /// `tau_bench/envs/<domain>/tasks*.py` (literal Python) and
    /// `tau_bench/envs/<domain>/data/*.json`. We scan for `*.json` under
    /// the provided directory and extract `(user_instruction, tools[0])`
    /// from each task. The repo also exports tasks as JSON dumps which
    /// are simpler to parse.
    static func ingestTauBench(dir: URL, allowed: Set<String>?,
                                pairs: inout [(query: String, tool: String)],
                                counts: inout [String: Int]) throws -> Int {
        var added = 0
        guard let walker = FileManager.default.enumerator(at: dir,
                                                            includingPropertiesForKeys: nil) else {
            throw NSError(domain: "extractor-data", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "could not enumerate \(dir.path)"])
        }
        for case let file as URL in walker {
            guard file.pathExtension.lowercased() == "json" else { continue }
            guard let data = try? Data(contentsOf: file) else { continue }
            guard let any = try? JSONSerialization.jsonObject(with: data) else { continue }
            let rows: [[String: Any]]
            if let arr = any as? [[String: Any]] {
                rows = arr
            } else if let obj = any as? [String: Any] {
                rows = [obj]
            } else {
                continue
            }
            for r in rows {
                let query: String? = stringFrom(r, keys: [
                    "user_instruction", "instruction", "user_query", "query", "question"
                ])
                let toolName: String? = extractTauTool(r)
                guard let q = query, let t = toolName, !q.isEmpty, !t.isEmpty else { continue }
                if let allow = allowed, !allow.contains(t) { continue }
                pairs.append((query: q, tool: t))
                counts[t, default: 0] += 1
                added += 1
            }
        }
        return added
    }

    static func extractTauTool(_ row: [String: Any]) -> String? {
        // τ-bench tasks list expected tool sequences; we take the first.
        if let arr = row["tools_used"] as? [String], let first = arr.first { return first }
        if let arr = row["actions"] as? [[String: Any]],
           let first = arr.first,
           let n = (first["name"] as? String) ?? (first["tool"] as? String) {
            return n
        }
        if let arr = row["expected_tools"] as? [String], let first = arr.first { return first }
        return nil
    }

    // MARK: - Synth via cloud

    /// Ask the cloud model to generate `n` plausible user queries that
    /// would invoke the named tool. Parse the response as a JSON array
    /// of strings; tolerate the common "numbered list" output too.
    static func synthesise(toolName: String, description: String,
                            n: Int, provider: CloudEscalate.Provider,
                            model: String?,
                            pairs: inout [(query: String, tool: String)],
                            counts: inout [String: Int]) throws -> Int {
        let prompt = """
        You are generating training data for a small tool-routing model.

        Tool name: \(toolName)
        Tool description: \(description)

        Produce a JSON array of exactly \(n) distinct, realistic user
        queries that would call this tool. Vary phrasing (questions,
        commands, conversational), length (5-30 words), and context.
        Do NOT include the tool name in the query itself. Output ONLY
        the JSON array, no prose.
        """
        let text = try CloudEscalate.complete(
            provider: provider, model: model,
            messages: [CloudEscalate.Message(role: "user", content: prompt)],
            maxTokens: 2048)
        let queries = parseStringArray(from: text)
        var added = 0
        for q in queries {
            let trimmed = q.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            pairs.append((query: trimmed, tool: toolName))
            counts[toolName, default: 0] += 1
            added += 1
        }
        return added
    }

    /// Extract a JSON array of strings from the cloud's reply, with
    /// fallback for numbered/markdown lists.
    static func parseStringArray(from text: String) -> [String] {
        // Find the first `[` and matching `]`.
        if let s = text.firstIndex(of: "["),
           let e = text.lastIndex(of: "]"),
           s < e {
            let slice = String(text[s...e])
            if let data = slice.data(using: .utf8),
               let arr = try? JSONSerialization.jsonObject(with: data) as? [String] {
                return arr
            }
        }
        // Fallback: split on lines, strip leading numerals / bullets.
        var out: [String] = []
        for raw in text.split(whereSeparator: { $0.isNewline }) {
            var s = String(raw).trimmingCharacters(in: .whitespaces)
            // Drop "1. ", "12) ", "- ", "* ".
            while let first = s.first {
                if first.isNumber || first == "." || first == ")"
                    || first == "-" || first == "*" || first == " " {
                    s.removeFirst()
                } else {
                    break
                }
            }
            // Drop surrounding quotes if the model wrapped each line.
            if (s.hasPrefix("\"") && s.hasSuffix("\""))
                || (s.hasPrefix("'") && s.hasSuffix("'")) {
                s = String(s.dropFirst().dropLast())
            }
            if !s.isEmpty { out.append(s) }
        }
        return out
    }

    // MARK: - Helpers

    static func stringFrom(_ row: [String: Any], keys: [String]) -> String? {
        for k in keys {
            if let s = row[k] as? String, !s.isEmpty { return s }
        }
        return nil
    }

    static func toolDescription(name: String, toolsPath: String?) throws -> String {
        guard let p = toolsPath else { return "(no description provided)" }
        let schema = try ToolSchema.load(from: URL(fileURLWithPath: p))
        if let t = schema.tools.first(where: { $0.name == name }) {
            return t.description.isEmpty ? "(no description)" : t.description
        }
        return "(unknown tool)"
    }

    static func writeJSONL(pairs: [(query: String, tool: String)], to url: URL) throws {
        let dir = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        guard let fh = try? FileHandle(forWritingTo: url) else {
            throw NSError(domain: "extractor-data", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "could not open \(url.path) for write"])
        }
        defer { try? fh.close() }
        for p in pairs {
            let obj: [String: Any] = ["query": p.query, "tool": p.tool]
            guard let data = try? JSONSerialization.data(withJSONObject: obj, options: []) else { continue }
            fh.write(data)
            fh.write(Data([0x0A]))
        }
    }

    static func exitUsage() -> Never {
        print("""
        usage: tinygpt extractor-data [flags]

        Builds (query, tool) pairs for training the tool-call extractor.

        sources (any combination):
          --bfcl <path>           Path to a BFCL JSONL file
          --tau-bench <dir>       Path to a tau-bench tasks directory
          --synth                 Generate synthetic queries via cloud
                                  for low-resource tools (requires --tools)

        config:
          --tools <schema.json>   Tool catalog (OpenAI schema). Defines
                                  the label set; required with --synth
          --synth-provider P      anthropic | openai (default: anthropic)
          --synth-model M         provider-specific model id
          --synth-per-tool N      synthetic examples per rare tool (default: 40)
          --min-examples N        threshold for "low resource" (default: 5)

        output:
          --out <path>            JSONL output (default:
                                  ~/.cache/tinygpt/router/router_data.jsonl)
          --dry-run               parse + report stats, don't write

        examples:
          tinygpt extractor-data --bfcl ~/.cache/tinygpt/datasets/.../corpus.jsonl --out router_data.jsonl
          tinygpt extractor-data --tools tools.json --synth --bfcl ... --out router_data.jsonl
        """)
        exit(2)
    }
}
