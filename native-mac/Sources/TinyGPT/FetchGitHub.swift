import Foundation
import TinyGPTData

/// `tinygpt fetch-github <owner/repo> [flags]` — pull structured
/// training records from GitHub.
///
/// This is the GitHub-side counterpart to `tinygpt download-dataset`.
/// HuggingFace gives you pre-curated datasets; GitHub gives you the
/// raw, in-the-wild signal for a code-specialist agent:
///
///   * issue→PR pairs   — "user reported bug X, here's the diff that
///                        fixed it" — the canonical debugger training
///                        signal.
///   * PR reviews       — "here's a hunk + the reviewer's comment" —
///                        trains a reviewer agent / refinement signal.
///   * commits          — "diff → commit message" — commit-message
///                        generator training data.
///
/// FLAGS
///   <repo>                 owner/repo (positional)
///   --kind <kind>          issues-prs | reviews | commits (default issues-prs)
///   --label <name>         GitHub label filter (default: bug for issues-prs)
///   --state <state>        open|closed|all (default closed for issues-prs)
///   --since <iso8601>      created/updated after this date
///   --max-diff-bytes N     truncate diffs above this size (default 10000)
///   --limit N              max records to emit (default 1000)
///   --out <path>           output JSONL path (default ./<repo>.<kind>.jsonl)
///   --multi-repo <file>    read newline-separated owner/repo list from <file>
///                          and aggregate
///   --resume               skip ids already present in <out> (if it exists)
///   --dry-run              estimate count, no fetch
///
/// EXAMPLES
///   tinygpt fetch-github pytorch/pytorch --kind issues-prs --limit 200
///   tinygpt fetch-github rust-lang/rust --kind commits --limit 5000 --out rust.commits.jsonl
///   tinygpt fetch-github huggingface/transformers --kind reviews --limit 500
///   tinygpt fetch-github --multi-repo bug_repos.txt --kind issues-prs --out combined.jsonl
///
/// ENV
///   GITHUB_TOKEN           bearer token (5000 req/h with, 60 without)
///   TINYGPT_GITHUB_CACHE   override cache root (default ~/.cache/tinygpt/github)
enum FetchGitHub {

    static func run(args: [String]) {
        var repoArg: String?
        var kind: GitHubCorpus.Kind = .issuesPRs
        var label: String?
        var state: String = "closed"
        var since: String?
        var maxDiffBytes: Int = 10_000
        var limit: Int = 1000
        var outPath: String?
        var multiRepoFile: String?
        var resume: Bool = false
        var dryRun: Bool = false

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--kind":
                guard i+1 < args.count, let k = GitHubCorpus.Kind(parsing: args[i+1]) else {
                    let opts = GitHubCorpus.Kind.allCases.map(\.rawValue).joined(separator: "|")
                    fputs("--kind requires one of: \(opts)\n", stderr); exit(2)
                }
                kind = k; i += 2
            case "--label":
                guard i+1 < args.count else { fputs("--label needs value\n", stderr); exit(2) }
                label = args[i+1]; i += 2
            case "--state":
                guard i+1 < args.count else { fputs("--state needs value\n", stderr); exit(2) }
                state = args[i+1]; i += 2
            case "--since":
                guard i+1 < args.count else { fputs("--since needs ISO 8601 date\n", stderr); exit(2) }
                since = args[i+1]; i += 2
            case "--max-diff-bytes":
                guard i+1 < args.count, let n = Int(args[i+1]) else { fputs("--max-diff-bytes needs int\n", stderr); exit(2) }
                maxDiffBytes = n; i += 2
            case "--limit":
                guard i+1 < args.count, let n = Int(args[i+1]) else { fputs("--limit needs int\n", stderr); exit(2) }
                limit = n; i += 2
            case "--out":
                guard i+1 < args.count else { fputs("--out needs path\n", stderr); exit(2) }
                outPath = args[i+1]; i += 2
            case "--multi-repo":
                guard i+1 < args.count else { fputs("--multi-repo needs file path\n", stderr); exit(2) }
                multiRepoFile = args[i+1]; i += 2
            case "--resume":   resume = true; i += 1
            case "--dry-run":  dryRun = true; i += 1
            case "-h", "--help": printUsage(); return
            default:
                if args[i].hasPrefix("-") {
                    fputs("unknown flag: \(args[i])\n", stderr); printUsage(); exit(2)
                }
                repoArg = args[i]; i += 1
            }
        }

        // Resolve list of (owner, repo) targets.
        var targets: [(String, String)] = []
        if let file = multiRepoFile {
            do {
                let data = try String(contentsOfFile: file, encoding: .utf8)
                for raw in data.split(whereSeparator: { $0.isNewline }) {
                    let line = raw.trimmingCharacters(in: .whitespaces)
                    if line.isEmpty || line.hasPrefix("#") { continue }
                    guard let pair = parseRepo(line) else {
                        fputs("warn: skipping unparseable repo: \(line)\n", stderr)
                        continue
                    }
                    targets.append(pair)
                }
            } catch {
                fputs("error: could not read --multi-repo file: \(error)\n", stderr); exit(1)
            }
        }
        if let raw = repoArg {
            guard let pair = parseRepo(raw) else {
                fputs("error: --repo must look like owner/repo (got \(raw))\n", stderr); exit(2)
            }
            targets.append(pair)
        }
        guard !targets.isEmpty else {
            fputs("error: pass <owner/repo> or --multi-repo <file>\n", stderr)
            printUsage(); exit(2)
        }

        // Warn if no token.
        if (ProcessInfo.processInfo.environment["GITHUB_TOKEN"] ?? "").isEmpty {
            fputs("""
            warn: GITHUB_TOKEN is not set. Without a token the GitHub REST API
                  allows only 60 requests / hour, which is rarely enough for
                  even a single repo. Set:
                      export GITHUB_TOKEN=ghp_xxx
                  to raise the limit to 5000 req/h.

            """, stderr)
        }

        // Build filters.
        var filters = GitHubCorpus.Filters()
        filters.label = label ?? (kind == .issuesPRs ? "bug" : nil)
        filters.state = state
        filters.since = since
        filters.maxDiffBytes = maxDiffBytes
        filters.limit = limit

        // Resolve output path.
        let firstRepoTag = targets.count == 1
            ? "\(targets[0].0)__\(targets[0].1)"
            : "multi-repo"
        let outURL = URL(fileURLWithPath: outPath ?? "./\(firstRepoTag).\(kind.rawValue).jsonl")

        if dryRun {
            print("==> dry-run: would fetch \(kind.rawValue) from \(targets.count) repo(s) with limit \(limit)")
            for (o, r) in targets {
                print("    - \(o)/\(r)")
            }
            print("    out: \(outURL.path)")
            return
        }

        // Build resume set if requested.
        let seen: Set<String> = resume ? loadResumeSet(url: outURL) : []
        if resume && !seen.isEmpty {
            print("==> resume: \(seen.count) records already in \(outURL.path)")
        }

        // Open writer (append if resuming, else fresh).
        let writer: JSONLBytesWriter
        do {
            writer = try JSONLBytesWriter(url: outURL, append: resume)
        } catch {
            fputs("error: could not open output: \(error)\n", stderr); exit(1)
        }
        defer { writer.close() }

        var totals = GitHubCorpus.Stats()
        for (owner, repo) in targets {
            print("==> \(owner)/\(repo) — fetching \(kind.rawValue)")
            do {
                let stats = try dispatch(kind: kind, owner: owner, repo: repo, filters: filters) { record in
                    // Resume dedupe: build an id for the record and skip
                    // if we've already written it.
                    let id = recordID(record)
                    if seen.contains(id) { return }
                    do { try writer.write(record) } catch {
                        fputs("warn: write failed: \(error)\n", stderr)
                    }
                }
                totals.scanned += stats.scanned
                totals.emitted += stats.emitted
                totals.skipped += stats.skipped
                totals.skippedNoPR += stats.skippedNoPR
                print("    scanned: \(stats.scanned)  emitted: \(stats.emitted)  skipped: \(stats.skipped)  no-PR: \(stats.skippedNoPR)")
            } catch let err as GitHubAPI.GHError {
                fputs("error fetching \(owner)/\(repo): \(err.description)\n", stderr)
                // Don't bail on a single failure when iterating multi-repo.
                if targets.count == 1 { exit(1) }
            } catch {
                fputs("error fetching \(owner)/\(repo): \(error)\n", stderr)
                if targets.count == 1 { exit(1) }
            }
        }

        print("")
        print("==> done")
        print("    repos:    \(targets.count)")
        print("    scanned:  \(totals.scanned)")
        print("    emitted:  \(totals.emitted)")
        print("    skipped:  \(totals.skipped) (\(totals.skippedNoPR) had no linked PR)")
        print("    output:   \(outURL.path)")
        switch kind {
        case .issuesPRs:
            print("    next: tinygpt sft <base> --data \(outURL.path) --template chatml --out debugger.tinygpt")
        case .reviews:
            print("    next: tinygpt sft <base> --data \(outURL.path) --template chatml --out reviewer.tinygpt")
        case .commits:
            print("    next: tinygpt sft <base> --data \(outURL.path) --template chatml --out commit-msg.tinygpt")
        }
    }

    // MARK: - Helpers

    static func dispatch(
        kind: GitHubCorpus.Kind,
        owner: String,
        repo: String,
        filters: GitHubCorpus.Filters,
        emit: (GitHubCorpus.Record) -> Void
    ) throws -> GitHubCorpus.Stats {
        switch kind {
        case .issuesPRs:
            return try GitHubCorpus.fetchIssuesPRs(owner: owner, repo: repo, filters: filters) { rec in
                emit(rec)
            }
        case .reviews:
            return try GitHubCorpus.fetchReviews(owner: owner, repo: repo, filters: filters) { rec in
                emit(rec)
            }
        case .commits:
            return try GitHubCorpus.fetchCommits(owner: owner, repo: repo, filters: filters) { rec in
                emit(rec)
            }
        }
    }

    /// Parse `owner/repo`, optionally with a leading "github.com/" or
    /// "https://github.com/" prefix, into a tuple.
    static func parseRepo(_ raw: String) -> (String, String)? {
        var s = raw
        let prefixes = ["https://github.com/", "http://github.com/", "github.com/"]
        for p in prefixes where s.hasPrefix(p) { s.removeFirst(p.count) }
        if s.hasSuffix("/") { s.removeLast() }
        let parts = s.split(separator: "/")
        guard parts.count == 2 else { return nil }
        return (String(parts[0]), String(parts[1]))
    }

    /// Build a stable id for a record from its metadata block. Used for
    /// resume dedupe.
    static func recordID(_ r: GitHubCorpus.Record) -> String {
        let md = r.metadata
        let repo = (md["repo"] as? String) ?? "?"
        if let sha = md["sha"] as? String { return "commit:\(repo)@\(sha)" }
        if let pr = md["pr_number"] as? Int, let cid = md["comment_id"] as? Int, cid != 0 {
            return "review:\(repo)#\(pr)c\(cid)"
        }
        if let iss = md["issue_number"] as? Int {
            return "issuepr:\(repo)#\(iss)"
        }
        // Fallback: instruction prefix hash.
        return "anon:\(r.instruction.prefix(64))"
    }

    /// Read the output file (if any) and rebuild the set of already-
    /// emitted ids by re-keying their metadata.
    static func loadResumeSet(url: URL) -> Set<String> {
        guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        var seen = Set<String>()
        guard let data = try? Data(contentsOf: url) else { return seen }
        guard let str = String(data: data, encoding: .utf8) else { return seen }
        for line in str.split(whereSeparator: { $0.isNewline }) {
            guard let lineData = String(line).data(using: .utf8) else { continue }
            guard let obj = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any] else { continue }
            let md = (obj["metadata"] as? [String: Any]) ?? [:]
            let repo = (md["repo"] as? String) ?? "?"
            if let sha = md["sha"] as? String {
                seen.insert("commit:\(repo)@\(sha)"); continue
            }
            if let pr = (md["pr_number"] as? NSNumber)?.intValue,
               let cid = (md["comment_id"] as? NSNumber)?.intValue, cid != 0 {
                seen.insert("review:\(repo)#\(pr)c\(cid)"); continue
            }
            if let iss = (md["issue_number"] as? NSNumber)?.intValue {
                seen.insert("issuepr:\(repo)#\(iss)"); continue
            }
        }
        return seen
    }

    static func printUsage() {
        let kinds = GitHubCorpus.Kind.allCases.map(\.rawValue).joined(separator: " | ")
        print("""
        usage: tinygpt fetch-github <owner/repo> [flags]

        flags:
          --kind <\(kinds)>
                                  what to extract (default: issues-prs)
          --label <name>          GitHub label filter (default for issues-prs: bug)
          --state <s>             open|closed|all (default: closed)
          --since <iso8601>       updated-after date
          --max-diff-bytes <n>    truncate diffs above this size (default 10000)
          --limit <n>             max records to emit (default 1000)
          --out <path>            output JSONL (default ./<repo>.<kind>.jsonl)
          --multi-repo <file>     repo-per-line file, aggregate results
          --resume                skip records already in <out>
          --dry-run               print the plan, no fetch

        env:
          GITHUB_TOKEN            bearer token; without it 60 req/h hard cap
          TINYGPT_GITHUB_CACHE    override cache root (~/.cache/tinygpt/github)
        """)
    }
}

// MARK: - JSONL writer for GitHubCorpus.Record

/// Append-mode JSONL writer that takes our metadata-bearing record.
/// `JSONLWriter` over in CorpusFormat.swift is for SFT/DPO/plain
/// CorpusRecord variants — those drop metadata. We keep a separate
/// writer here so the GitHub records preserve the `metadata` block
/// downstream filtering depends on.
final class JSONLBytesWriter {
    let url: URL
    private let handle: FileHandle
    private(set) var count: Int = 0

    init(url: URL, append: Bool) throws {
        self.url = url
        let fm = FileManager.default
        try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !append || !fm.fileExists(atPath: url.path) {
            fm.createFile(atPath: url.path, contents: nil)
        }
        guard let h = try? FileHandle(forWritingTo: url) else {
            throw NSError(domain: "fetch-github", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "open failed: \(url.path)"])
        }
        self.handle = h
        if append { try h.seekToEnd() }
    }

    func write(_ record: GitHubCorpus.Record) throws {
        let data = try record.toJSONLine()
        handle.write(data)
        count += 1
    }

    func close() { try? handle.close() }
}
