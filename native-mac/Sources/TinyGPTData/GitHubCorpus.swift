import Foundation

/// GitHub → tinygpt training corpus extractor.
///
/// Three kinds of records, all written as JSONL with the same shape as
/// the SFT format (instruction + response + metadata) so they drop
/// straight into `tinygpt sft`:
///
///   * issue→PR pairs  — bug fix training signal
///   * PR reviews      — refinement / reviewer-agent training signal
///   * commits         — commit-message generation training signal
///
/// The hard problem here is the issue→PR linkage. GitHub doesn't have a
/// first-class "this PR closes that issue" pointer; the linkage lives
/// in:
///
///   a) The issue's timeline events (`closed`, `cross-referenced`),
///      where a referenced PR shows up as an event with `source.type ==
///      "issue"` and a `pull_request` field on the source issue.
///   b) "Closes #N" / "Fixes #N" mentions in the PR body or commits.
///   c) The PR's own `closed_issues` connection (GraphQL only — we
///      don't use it because we're on REST).
///
/// We use the timeline-events path (most reliable) with body-regex as a
/// fallback. The pair is dropped if neither finds a PR — bug-tracker
/// issues that were closed by a commit (not a PR) just aren't useful
/// training signal for a PR-writer agent.
///
/// All records carry a `metadata` block so downstream filtering can
/// dedupe / weight by repo, labels, language, etc. Output is JSONL.
public enum GitHubCorpus {

    // MARK: - Output record

    /// One record. We serialize this directly as JSON; the structure
    /// matches the SFT JSONL convention (`instruction` / `response`)
    /// plus a `metadata` block. Downstream `tinygpt sft` consumes the
    /// instruction+response pair and ignores metadata.
    ///
    /// We intentionally do NOT mark this `Sendable`: `metadata`'s
    /// `[String: Any]` representation isn't expressible as Sendable
    /// without wrapping every value, and these records never cross an
    /// actor / concurrency boundary in practice (the corpus extractor
    /// is single-threaded synchronous).
    public struct Record {
        public let instruction: String
        public let response: String
        public let metadata: [String: Any]
        public init(instruction: String, response: String, metadata: [String: Any]) {
            self.instruction = instruction
            self.response = response
            self.metadata = metadata
        }

        public func toJSONLine() throws -> Data {
            let obj: [String: Any] = [
                "instruction": instruction,
                "response": response,
                "metadata": metadata
            ]
            var data = try JSONSerialization.data(
                withJSONObject: obj,
                options: [.sortedKeys, .withoutEscapingSlashes]
            )
            data.append(0x0a) // '\n'
            return data
        }
    }

    // MARK: - Filters

    public struct Filters: Sendable {
        public var label: String?
        public var state: String = "closed"
        public var since: String?            // ISO 8601 e.g. "2024-01-01T00:00:00Z"
        public var maxDiffBytes: Int = 10_000
        public var limit: Int = 1000
        public init() {}
    }

    public enum Kind: String, Sendable, CaseIterable {
        case issuesPRs = "issues-prs"
        case reviews   = "reviews"
        case commits   = "commits"

        public init?(parsing s: String) {
            switch s.lowercased().replacingOccurrences(of: "_", with: "-") {
            case "issues-prs", "issues", "issue-prs", "bugs": self = .issuesPRs
            case "reviews", "review", "pr-reviews": self = .reviews
            case "commits", "commit", "commit-log": self = .commits
            default: return nil
            }
        }
    }

    // MARK: - Cache root

    /// Per-repo cache root: `~/.cache/tinygpt/github/<owner>/<repo>/`
    /// (parallels HFDatasets.cacheRoot). Cache files are JSON dumps of
    /// previously-fetched records, keyed by issue / PR / commit
    /// identifier — resume just skips ids that have already been
    /// written.
    public static func cacheRoot() -> URL {
        let env = ProcessInfo.processInfo.environment
        if let override = env["TINYGPT_GITHUB_CACHE"], !override.isEmpty {
            return URL(fileURLWithPath: (override as NSString).expandingTildeInPath, isDirectory: true)
        }
        let home = env["HOME"].map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".cache/tinygpt/github", isDirectory: true)
    }

    public static func cacheDir(owner: String, repo: String) throws -> URL {
        let dir = cacheRoot()
            .appendingPathComponent(owner, isDirectory: true)
            .appendingPathComponent(repo, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - Public extraction entrypoints

    /// Issue → PR pairs.
    ///
    /// For each closed issue matching `filters`, we:
    ///   1. Read the issue body + title.
    ///   2. Walk the issue's `/timeline` to find the PR that closed it.
    ///      If no timeline hit, fall back to scanning the issue body
    ///      for `Closes #N` / `Fixes #N` (PRs are issues too, so this
    ///      resolves the same way).
    ///   3. Fetch the PR diff (via `Accept: application/vnd.github.diff`).
    ///   4. Truncate the diff if it's larger than `filters.maxDiffBytes`.
    ///   5. Emit a Record with instruction = issue title+body and
    ///      response = PR description + truncated diff.
    @discardableResult
    public static func fetchIssuesPRs(
        owner: String,
        repo: String,
        filters: Filters,
        emit: (Record) throws -> Void
    ) throws -> Stats {
        var stats = Stats()
        let label = filters.label ?? "bug"
        var baseQuery: [(String, String)] = [
            ("state", filters.state),
            ("labels", label),
            ("sort", "updated"),
            ("direction", "desc"),
        ]
        if let since = filters.since { baseQuery.append(("since", since)) }

        var emitted = 0
        let maxPages = max(1, (filters.limit + 99) / 100)
        try GitHubAPI.paginate(
            path: "/repos/\(owner)/\(repo)/issues",
            query: baseQuery,
            maxPages: maxPages
        ) { page, _ in
            for issue in page {
                if emitted >= filters.limit { return false }
                stats.scanned += 1

                // The `/repos/{owner}/{repo}/issues` endpoint also lists
                // PRs (PRs are issues with a `pull_request` member). For
                // issue→PR pairing we want the *issue* side; skip PRs.
                if issue["pull_request"] != nil { continue }

                let issueNumber = (issue["number"] as? NSNumber)?.intValue ?? -1
                guard issueNumber > 0 else { continue }
                let title = (issue["title"] as? String) ?? ""
                let body = (issue["body"] as? String) ?? ""
                let labels: [String] = (issue["labels"] as? [[String: Any]] ?? [])
                    .compactMap { $0["name"] as? String }
                if title.isEmpty && body.isEmpty {
                    stats.skipped += 1
                    continue
                }

                // Look for the PR that closed this issue.
                let prNumber: Int? = try findLinkingPR(owner: owner, repo: repo, issueNumber: issueNumber, issueBody: body)
                guard let pr = prNumber else {
                    stats.skippedNoPR += 1
                    continue
                }

                // Fetch PR description + diff.
                let prInfo: (body: String, diff: String, files: Int)
                do {
                    prInfo = try fetchPRBundle(owner: owner, repo: repo, prNumber: pr,
                                               maxDiffBytes: filters.maxDiffBytes)
                } catch GitHubAPI.GHError.notFound {
                    stats.skipped += 1
                    continue
                }
                if prInfo.diff.isEmpty {
                    stats.skipped += 1
                    continue
                }

                let instruction = title + (body.isEmpty ? "" : "\n\n" + body)
                var response = ""
                if !prInfo.body.isEmpty {
                    response += prInfo.body + "\n\n"
                }
                response += "--- diff ---\n" + prInfo.diff

                let metadata: [String: Any] = [
                    "repo": "\(owner)/\(repo)",
                    "issue_number": issueNumber,
                    "pr_number": pr,
                    "labels": labels,
                    "kind": "issue-pr",
                    "files_changed": prInfo.files,
                ]
                try emit(Record(instruction: instruction, response: response, metadata: metadata))
                emitted += 1
                stats.emitted += 1
            }
            return emitted < filters.limit
        }
        return stats
    }

    /// PR review pairs. For each merged PR we emit one record per
    /// review-comment chain: instruction = the hunk that was commented
    /// on, response = the review comment. This is the "PR-reviewer
    /// specialist" signal.
    @discardableResult
    public static func fetchReviews(
        owner: String,
        repo: String,
        filters: Filters,
        emit: (Record) throws -> Void
    ) throws -> Stats {
        var stats = Stats()
        var baseQuery: [(String, String)] = [
            ("state", filters.state == "closed" ? "closed" : "all"),
            ("sort", "updated"),
            ("direction", "desc"),
        ]
        if let since = filters.since { baseQuery.append(("since", since)) }

        var emitted = 0
        let maxPages = max(1, (filters.limit + 99) / 100)
        try GitHubAPI.paginate(
            path: "/repos/\(owner)/\(repo)/pulls",
            query: baseQuery,
            maxPages: maxPages
        ) { page, _ in
            for pr in page {
                if emitted >= filters.limit { return false }
                stats.scanned += 1
                let prNumber = (pr["number"] as? NSNumber)?.intValue ?? -1
                guard prNumber > 0 else { continue }
                // Skip unmerged PRs — review on a closed-without-merge
                // PR is noisy training signal.
                let merged = (pr["merged_at"] as? String) != nil
                if !merged { continue }

                // /pulls/{n}/comments returns review-line comments
                // (the ones attached to a diff hunk).
                let path = "/repos/\(owner)/\(repo)/pulls/\(prNumber)/comments"
                let (json, _, _) = try GitHubAPI.get(path, query: [("per_page", "100")])
                guard let comments = json as? [[String: Any]] else { continue }
                for c in comments {
                    if emitted >= filters.limit { break }
                    let hunk = (c["diff_hunk"] as? String) ?? ""
                    let body = (c["body"] as? String) ?? ""
                    if hunk.isEmpty || body.isEmpty {
                        stats.skipped += 1
                        continue
                    }
                    // Truncate hunk to avoid mega-records.
                    let truncatedHunk = truncate(hunk, to: filters.maxDiffBytes)
                    let path = (c["path"] as? String) ?? ""
                    let metadata: [String: Any] = [
                        "repo": "\(owner)/\(repo)",
                        "pr_number": prNumber,
                        "comment_id": (c["id"] as? NSNumber)?.intValue ?? 0,
                        "file": path,
                        "kind": "review",
                    ]
                    try emit(Record(
                        instruction: "review this code:\n\(truncatedHunk)",
                        response: body,
                        metadata: metadata
                    ))
                    emitted += 1
                    stats.emitted += 1
                }
            }
            return emitted < filters.limit
        }
        return stats
    }

    /// Commit log pairs. instruction = the diff, response = the commit
    /// message. The "commit-message-writer" training signal.
    @discardableResult
    public static func fetchCommits(
        owner: String,
        repo: String,
        filters: Filters,
        emit: (Record) throws -> Void
    ) throws -> Stats {
        var stats = Stats()
        var baseQuery: [(String, String)] = []
        if let since = filters.since { baseQuery.append(("since", since)) }

        var emitted = 0
        let maxPages = max(1, (filters.limit + 99) / 100)
        try GitHubAPI.paginate(
            path: "/repos/\(owner)/\(repo)/commits",
            query: baseQuery,
            maxPages: maxPages
        ) { page, _ in
            for commit in page {
                if emitted >= filters.limit { return false }
                stats.scanned += 1
                guard let sha = commit["sha"] as? String else { continue }
                let commitObj = commit["commit"] as? [String: Any] ?? [:]
                let message = (commitObj["message"] as? String) ?? ""
                if message.isEmpty {
                    stats.skipped += 1
                    continue
                }
                // Fetch the full commit (includes file diffs).
                let detail: [String: Any]
                do {
                    let (json, _, _) = try GitHubAPI.get("/repos/\(owner)/\(repo)/commits/\(sha)")
                    guard let obj = json as? [String: Any] else { continue }
                    detail = obj
                } catch GitHubAPI.GHError.notFound {
                    stats.skipped += 1
                    continue
                }
                let files = detail["files"] as? [[String: Any]] ?? []
                let diff = assembleDiff(from: files, maxBytes: filters.maxDiffBytes)
                if diff.isEmpty {
                    stats.skipped += 1
                    continue
                }
                let metadata: [String: Any] = [
                    "repo": "\(owner)/\(repo)",
                    "sha": sha,
                    "files_changed": files.count,
                    "kind": "commit",
                ]
                try emit(Record(
                    instruction: diff,
                    response: message,
                    metadata: metadata
                ))
                emitted += 1
                stats.emitted += 1
            }
            return emitted < filters.limit
        }
        return stats
    }

    // MARK: - Internals

    /// Find the PR that closed an issue. Order:
    ///   1. timeline event `closed` whose `commit_id` was authored by
    ///      a PR (we read `source` if present);
    ///   2. timeline event `cross-referenced` where source is a PR;
    ///   3. `Closes #N` / `Fixes #N` in the issue body.
    static func findLinkingPR(owner: String, repo: String, issueNumber: Int, issueBody: String) throws -> Int? {
        // Timeline endpoint.
        do {
            let (json, _, _) = try GitHubAPI.get(
                "/repos/\(owner)/\(repo)/issues/\(issueNumber)/timeline",
                query: [("per_page", "100")]
            )
            if let events = json as? [[String: Any]] {
                for ev in events {
                    let kind = (ev["event"] as? String) ?? ""
                    // 'cross-referenced' is the most reliable: it fires
                    // when a PR description / commit mentions the issue.
                    if kind == "cross-referenced" {
                        if let source = ev["source"] as? [String: Any],
                           let issueLike = source["issue"] as? [String: Any],
                           let prLink = issueLike["pull_request"] as? [String: Any],
                           let url = prLink["url"] as? String,
                           let n = extractPRNumber(fromURL: url) {
                            return n
                        }
                    }
                    // 'closed' events sometimes carry a commit_id only;
                    // a commit_id alone doesn't give us a PR number, so
                    // we skip unless a PR is also referenced.
                }
            }
        } catch GitHubAPI.GHError.http {
            // Some repos disable timeline; fall through to body regex.
        }

        // Body regex fallback. Matches "Closes #123", "Fixes: #45",
        // "resolves owner/repo#67". Case-insensitive.
        if let prNumber = parseClosingKeywords(in: issueBody, currentRepo: "\(owner)/\(repo)") {
            return prNumber
        }
        return nil
    }

    /// Fetch a PR's body + unified diff. The diff is fetched via the
    /// `application/vnd.github.diff` media type — REST endpoint, raw
    /// body. The PR body is fetched separately via JSON.
    static func fetchPRBundle(owner: String, repo: String, prNumber: Int, maxDiffBytes: Int) throws -> (body: String, diff: String, files: Int) {
        let (json, _, _) = try GitHubAPI.get("/repos/\(owner)/\(repo)/pulls/\(prNumber)")
        let pr = json as? [String: Any] ?? [:]
        let body = (pr["body"] as? String) ?? ""
        let files = (pr["changed_files"] as? NSNumber)?.intValue ?? 0

        let (diffData, status) = try GitHubAPI.getRaw(
            "/repos/\(owner)/\(repo)/pulls/\(prNumber)",
            accept: "application/vnd.github.diff"
        )
        if status == 406 {
            // Some old PRs return 406 if the diff is too large to be
            // served inline. Fall back to files-listing.
            let (filesJson, _, _) = try GitHubAPI.get("/repos/\(owner)/\(repo)/pulls/\(prNumber)/files",
                                                     query: [("per_page", "100")])
            if let arr = filesJson as? [[String: Any]] {
                return (body, assembleDiff(from: arr, maxBytes: maxDiffBytes), files)
            }
            return (body, "", files)
        }
        if !(200..<300).contains(status) {
            return (body, "", files)
        }
        let diffStr = String(data: diffData, encoding: .utf8) ?? ""
        return (body, truncate(diffStr, to: maxDiffBytes), files)
    }

    /// Concatenate per-file patch hunks (from a /files endpoint
    /// response) into a single string, capped at `maxBytes`.
    static func assembleDiff(from files: [[String: Any]], maxBytes: Int) -> String {
        var out = ""
        for f in files {
            let filename = (f["filename"] as? String) ?? "?"
            // `patch` is absent for binary files / large diffs.
            let patch = (f["patch"] as? String) ?? ""
            if patch.isEmpty { continue }
            out += "--- a/\(filename)\n+++ b/\(filename)\n"
            out += patch
            out += "\n"
            if out.count >= maxBytes {
                return truncate(out, to: maxBytes)
            }
        }
        return out
    }

    /// Hard truncate, marking the cut so it's obvious downstream that
    /// the record is a snippet.
    static func truncate(_ s: String, to maxBytes: Int) -> String {
        if s.count <= maxBytes { return s }
        let prefix = s.prefix(maxBytes)
        return String(prefix) + "\n\n[…truncated \(s.count - maxBytes) chars…]\n"
    }

    /// Extract a PR number from a URL like
    /// `https://api.github.com/repos/foo/bar/pulls/123`.
    static func extractPRNumber(fromURL url: String) -> Int? {
        guard let range = url.range(of: "/pulls/") else { return nil }
        let tail = url[range.upperBound...]
        let digits = tail.prefix(while: { $0.isNumber })
        return Int(digits)
    }

    /// Parse GitHub's closing-keyword syntax in an issue body. We
    /// accept the canonical set (closes, fixes, resolves, close, fix,
    /// resolve) followed by `#N` or `owner/repo#N`. Returns the *first*
    /// PR number from the *current* repo only — refusing
    /// cross-repo links keeps the dataset clean.
    static func parseClosingKeywords(in body: String, currentRepo: String) -> Int? {
        if body.isEmpty { return nil }
        let keywords = ["closes", "closed", "close",
                        "fixes", "fixed", "fix",
                        "resolves", "resolved", "resolve"]
        let lower = body.lowercased()
        for kw in keywords {
            var search = Substring(lower)
            while let range = search.range(of: kw) {
                // Must be word-boundary on the left.
                if range.lowerBound != search.startIndex {
                    let prev = search[search.index(before: range.lowerBound)]
                    if prev.isLetter || prev.isNumber || prev == "_" {
                        search = search[range.upperBound...]; continue
                    }
                }
                var idx = range.upperBound
                // Skip ':', spaces.
                while idx < search.endIndex, [":", " ", "\t"].contains(search[idx]) {
                    idx = search.index(after: idx)
                }
                // Optional owner/repo prefix.
                let remaining = search[idx...]
                if remaining.hasPrefix(currentRepo.lowercased() + "#") {
                    idx = search.index(idx, offsetBy: currentRepo.count + 1)
                } else if remaining.hasPrefix("#") {
                    idx = search.index(after: idx)
                } else {
                    search = search[range.upperBound...]; continue
                }
                let digits = search[idx...].prefix(while: { $0.isNumber })
                if let n = Int(digits), n > 0 { return n }
                search = search[range.upperBound...]
            }
        }
        return nil
    }

    // MARK: - Stats

    public struct Stats: Sendable {
        public var scanned: Int = 0
        public var emitted: Int = 0
        public var skipped: Int = 0
        public var skippedNoPR: Int = 0
        public init() {}
    }
}
