import Foundation

/// GitHub REST API v3 client used by the `tinygpt fetch-github` data
/// pipeline.
///
/// Why this exists: tinygpt is an on-device SLM factory and the
/// code-debugger specialist needs a very specific training signal —
/// "here is the issue someone filed, here is the PR that fixed it"
/// pairs. The HuggingFace Datasets integration (see HFDatasets.swift)
/// already ships the pre-curated read path; this is the equivalent
/// read path for GitHub, which is the largest single source of
/// real-world issue→fix pairs anywhere.
///
/// Design notes
/// ------------
/// * Pure Foundation. URLSession-backed. No 3rd-party HTTP layer so
///   the CLI subcommand boots fast and so we can ship this without
///   any new package dependency.
/// * Synchronous wrapper around dataTask. The CLI flow reads
///   top-to-bottom; if/when this code moves into a SwiftUI surface
///   the wrapper can be lifted to async.
/// * Auth: GITHUB_TOKEN environment variable. With a token the rate
///   limit is 5000 req/h, without one it's 60 req/h — too low to be
///   useful in practice but we surface that clearly rather than
///   silently bailing.
/// * Pagination: REST returns 100 items max per page. The high-level
///   helpers in GitHubCorpus walk pages until the caller's `limit` is
///   satisfied or the server returns an empty page.
/// * Rate-limit handling: we read the `X-RateLimit-Remaining` and
///   `Retry-After` headers and sleep when the remaining count drops
///   below a safety threshold. 403 with the
///   "API rate limit exceeded" body is treated the same as 429.
///
/// References
/// ----------
/// * REST API:        https://docs.github.com/en/rest
/// * Rate limits:     https://docs.github.com/en/rest/rate-limit
/// * Issues:          https://docs.github.com/en/rest/issues/issues
/// * Pulls:           https://docs.github.com/en/rest/pulls/pulls
/// * Timeline events: https://docs.github.com/en/rest/issues/timeline
public enum GitHubAPI {

    // MARK: - Errors

    public enum GHError: Error, CustomStringConvertible {
        case http(status: Int, url: String, body: String)
        case malformedResponse(String)
        case network(String)
        case rateLimited(resetUnix: Int?, url: String)
        case unauthorized(url: String)
        case notFound(url: String)
        case ioError(String)
        public var description: String {
            switch self {
            case .http(let s, let u, let b):
                let snippet = b.count > 200 ? String(b.prefix(200)) + "..." : b
                return "HTTP \(s) for \(u)\n  body: \(snippet)"
            case .malformedResponse(let s): return "malformed GitHub response: \(s)"
            case .network(let s): return "network error: \(s)"
            case .rateLimited(let reset, let u):
                let when = reset.map { " (resets at unix \($0))" } ?? ""
                return """
                GitHub API rate limit hit\(when) for \(u).
                Without GITHUB_TOKEN the limit is 60 requests/hour — too low
                for `tinygpt fetch-github`. Set:
                    export GITHUB_TOKEN=ghp_xxx
                A fine-grained personal access token with `public_repo` scope
                (or `repo` for private repos) is enough. Tokens raise the
                limit to 5000 req/h.
                """
            case .unauthorized(let u):
                return """
                GitHub returned 401 for \(u).
                The GITHUB_TOKEN environment variable is either missing,
                expired, or doesn't have the required scope. For public
                repos the `public_repo` scope is sufficient; for private
                repos use `repo`. Create a token at:
                    https://github.com/settings/tokens
                """
            case .notFound(let u):
                return "GitHub returned 404 for \(u) (repo / issue / PR does not exist or is private)"
            case .ioError(let s): return "I/O error: \(s)"
            }
        }
    }

    // MARK: - Public knobs

    /// Default base URL — overridable for Enterprise Server (GHES).
    public static let defaultBaseURL = "https://api.github.com"

    /// Internal safety knob: if the response carries
    /// `X-RateLimit-Remaining` ≤ this number, we proactively sleep
    /// until reset rather than risk a 403.
    public static let rateLimitSafetyFloor = 5

    // MARK: - Low-level GET

    /// Synchronous GET. Returns `(decoded JSON value, response headers,
    /// HTTP status)`. Throws on transport errors / non-2xx / rate limits.
    ///
    /// We intentionally decode to `Any` (Dictionary or Array) here rather
    /// than to a typed struct: the GitHub REST API has dozens of
    /// endpoints and we only want a handful of fields from each — Codable
    /// boilerplate per endpoint would dwarf the actual data extraction.
    public static func get(
        _ path: String,
        query: [(String, String)] = [],
        baseURL: String = defaultBaseURL,
        previewMedia: String? = nil
    ) throws -> (json: Any, headers: [String: String], status: Int) {
        let url = try buildURL(base: baseURL, path: path, query: query)
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        // Recommended Accept header per docs. Some endpoints want a
        // preview media type (e.g. timeline events used to need
        // "application/vnd.github.mockingbird-preview+json"); we set it
        // when the caller asks.
        req.setValue(previewMedia ?? "application/vnd.github+json", forHTTPHeaderField: "Accept")
        req.setValue("2022-11-28", forHTTPHeaderField: "X-GitHub-Api-Version")
        if let token = ProcessInfo.processInfo.environment["GITHUB_TOKEN"], !token.isEmpty {
            // GitHub accepts both `token <pat>` and `Bearer <pat>`; the
            // newer fine-grained tokens prefer Bearer.
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        req.setValue("tinygpt/0.1 (+https://github.com/sarthak/tinygpt)", forHTTPHeaderField: "User-Agent")

        let box = HTTPResultBox()
        let sema = DispatchSemaphore(value: 0)
        let task = URLSession.shared.dataTask(with: req) { data, resp, err in
            defer { sema.signal() }
            if let err = err { box.error = err; return }
            box.data = data ?? Data()
            if let http = resp as? HTTPURLResponse {
                box.status = http.statusCode
                // HTTPURLResponse.allHeaderFields uses [AnyHashable:Any]
                // — flatten to a string-keyed dictionary with the
                // canonical lowercase keys the rest of the code expects.
                var hdrs: [String: String] = [:]
                for (k, v) in http.allHeaderFields {
                    if let ks = (k as? String), let vs = (v as? String) {
                        hdrs[ks.lowercased()] = vs
                    }
                }
                box.headers = hdrs
            }
        }
        task.resume()
        sema.wait()
        if let error = box.error { throw GHError.network("\(error)") }
        let data = box.data ?? Data()
        let status = box.status
        let headers = box.headers

        // Rate limit handling. GitHub returns 403 (not 429) with the
        // body string "API rate limit exceeded" — we coerce that to a
        // dedicated case so the CLI can render a useful message.
        if status == 403, let body = String(data: data, encoding: .utf8),
           body.contains("rate limit") || body.contains("rate-limit") {
            let reset = (headers["x-ratelimit-reset"]).flatMap(Int.init)
            throw GHError.rateLimited(resetUnix: reset, url: url.absoluteString)
        }
        if status == 429 {
            let reset = (headers["x-ratelimit-reset"]).flatMap(Int.init)
            throw GHError.rateLimited(resetUnix: reset, url: url.absoluteString)
        }
        if status == 401 { throw GHError.unauthorized(url: url.absoluteString) }
        if status == 404 { throw GHError.notFound(url: url.absoluteString) }
        guard (200..<300).contains(status) else {
            throw GHError.http(status: status, url: url.absoluteString,
                               body: String(data: data, encoding: .utf8) ?? "")
        }
        let json: Any
        do {
            json = try JSONSerialization.jsonObject(with: data, options: [.allowFragments])
        } catch {
            throw GHError.malformedResponse("\(error)")
        }
        return (json, headers, status)
    }

    /// Same as `get` but the response body is returned as `Data`. Used
    /// for the `.diff` media type from the PR endpoint, which is not
    /// JSON.
    public static func getRaw(
        _ path: String,
        accept: String,
        baseURL: String = defaultBaseURL
    ) throws -> (data: Data, status: Int) {
        let url = try buildURL(base: baseURL, path: path, query: [])
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        req.setValue(accept, forHTTPHeaderField: "Accept")
        req.setValue("2022-11-28", forHTTPHeaderField: "X-GitHub-Api-Version")
        if let token = ProcessInfo.processInfo.environment["GITHUB_TOKEN"], !token.isEmpty {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        req.setValue("tinygpt/0.1", forHTTPHeaderField: "User-Agent")

        let box = HTTPResultBox()
        let sema = DispatchSemaphore(value: 0)
        let task = URLSession.shared.dataTask(with: req) { data, resp, err in
            defer { sema.signal() }
            if let err = err { box.error = err; return }
            box.data = data ?? Data()
            box.status = (resp as? HTTPURLResponse)?.statusCode ?? 0
        }
        task.resume()
        sema.wait()
        if let error = box.error { throw GHError.network("\(error)") }
        return (box.data ?? Data(), box.status)
    }

    // MARK: - Pagination helpers

    /// Walk a paginated REST list endpoint until `accept` returns false
    /// for a page or the server returns an empty page. The `accept`
    /// closure receives one page (an array of dictionaries) and returns
    /// `true` to continue.
    ///
    /// Per-page size is fixed at 100 (REST API max). Page numbers are
    /// 1-based per GitHub convention.
    public static func paginate(
        path: String,
        query baseQuery: [(String, String)],
        maxPages: Int = Int.max,
        sleepOnLowRemaining: Bool = true,
        accept: (_ page: [[String: Any]], _ pageIndex: Int) throws -> Bool
    ) throws {
        var page = 1
        while page <= maxPages {
            var q = baseQuery
            // Append per_page and page (without clobbering caller-provided
            // ones — caller might want to override per_page).
            if !q.contains(where: { $0.0 == "per_page" }) { q.append(("per_page", "100")) }
            q.append(("page", String(page)))
            let (json, headers, _) = try get(path, query: q)

            if sleepOnLowRemaining {
                respectRateLimit(headers: headers)
            }

            guard let arr = json as? [[String: Any]] else {
                // Some list endpoints (e.g. /search/*) wrap items in
                // `{ total_count, items: [...] }`. Handle that here so
                // callers can use a uniform interface.
                if let obj = json as? [String: Any], let items = obj["items"] as? [[String: Any]] {
                    if items.isEmpty { return }
                    if !(try accept(items, page)) { return }
                    if items.count < 100 { return }
                    page += 1
                    continue
                }
                return
            }
            if arr.isEmpty { return }
            if !(try accept(arr, page)) { return }
            if arr.count < 100 { return }     // last page
            page += 1
        }
    }

    /// Read `X-RateLimit-Remaining` and sleep until reset if we're below
    /// the safety floor. Best-effort: missing headers → no-op.
    public static func respectRateLimit(headers: [String: String]) {
        guard let remainingStr = headers["x-ratelimit-remaining"],
              let remaining = Int(remainingStr) else { return }
        if remaining > rateLimitSafetyFloor { return }
        guard let resetStr = headers["x-ratelimit-reset"],
              let reset = Int(resetStr) else { return }
        let now = Int(Date().timeIntervalSince1970)
        let wait = max(0, reset - now) + 1
        if wait > 0 && wait < 3600 {
            fputs("note: rate-limit remaining=\(remaining); sleeping \(wait)s until window resets…\n", stderr)
            Thread.sleep(forTimeInterval: TimeInterval(wait))
        }
    }

    // MARK: - URL builder

    private static func buildURL(base: String, path: String, query: [(String, String)]) throws -> URL {
        let normalizedPath = path.hasPrefix("/") ? path : "/" + path
        let normalizedBase = base.hasSuffix("/") ? String(base.dropLast()) : base
        var comps = URLComponents(string: normalizedBase + normalizedPath)
        if !query.isEmpty {
            comps?.queryItems = query.map { URLQueryItem(name: $0.0, value: $0.1) }
        }
        guard let url = comps?.url else {
            throw GHError.malformedResponse("bad URL components for \(base)\(path)")
        }
        return url
    }
}

// MARK: - Reference cells

/// Mutable reference cell shared with URLSession closures. Same pattern
/// as HFDatasets.ResultBox — Swift 6 strict concurrency disallows
/// closure capture of `var`, so we route through a class.
private final class HTTPResultBox: @unchecked Sendable {
    var data: Data?
    var headers: [String: String] = [:]
    var status: Int = 0
    var error: Error?
}
