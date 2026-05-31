import Foundation

/// Cloud-escalation client — calls a larger remote model when the
/// on-device specialist defers.
///
/// Why this exists: the north-star architecture is a small on-device
/// specialist that handles ~80% of requests + escalates the rest to a
/// stronger cloud model. This is the cloud half.
///
/// Why shell-out to `curl`: same rationale as R2Client + aws CLI.
/// curl is everywhere, handles HTTP/2 + retries + TLS correctly, no
/// dependency on URLSession-async-Sendable headaches. Cost is one
/// dep that's already on every Mac.
///
/// AUTH: env vars (never logged):
///   ANTHROPIC_API_KEY    — required for Anthropic Messages API
///   OPENAI_API_KEY       — required for OpenAI Chat Completions API
///
/// Endpoints picked at call-time based on `provider` arg.
public enum CloudEscalate {

    public enum Provider: String, CaseIterable {
        case anthropic, openai
    }

    public enum EscalateError: Error, CustomStringConvertible {
        case missingApiKey(Provider)
        case curlNotFound
        case httpError(status: Int, body: String)
        case parseError(String)

        public var description: String {
            switch self {
            case .missingApiKey(let p):
                let envName = p == .anthropic ? "ANTHROPIC_API_KEY" : "OPENAI_API_KEY"
                return "cloud escalation requires \(envName) in env"
            case .curlNotFound:
                return "`curl` not found in PATH"
            case .httpError(let status, let body):
                return "HTTP \(status): \(body.prefix(500))"
            case .parseError(let msg):
                return "response parse failed: \(msg)"
            }
        }
    }

    public struct Message {
        public let role: String   // "user" or "assistant" or "system"
        public let content: String
        public init(role: String, content: String) {
            self.role = role; self.content = content
        }
    }

    /// Call the cloud model. Returns the assistant's text response.
    ///
    /// - provider: which API to talk to
    /// - model: provider-specific model name. Defaults to a sensible
    ///   pick per provider; the caller can override (e.g.,
    ///   "claude-sonnet-4-5" or "gpt-4o-mini").
    /// - messages: conversation history including the latest user turn
    /// - maxTokens: hard cap on response length
    /// - systemPrompt: optional system message (Anthropic prefers
    ///   top-level `system`; OpenAI prefers `messages[0].role=system`).
    public static func complete(provider: Provider,
                                 model: String? = nil,
                                 messages: [Message],
                                 maxTokens: Int = 1024,
                                 systemPrompt: String? = nil,
                                 timeoutSeconds: Int = 60) throws -> String {
        switch provider {
        case .anthropic:
            return try callAnthropic(
                model: model ?? "claude-sonnet-4-5",
                messages: messages,
                maxTokens: maxTokens,
                systemPrompt: systemPrompt,
                timeoutSeconds: timeoutSeconds
            )
        case .openai:
            return try callOpenAI(
                model: model ?? "gpt-4o-mini",
                messages: messages,
                maxTokens: maxTokens,
                systemPrompt: systemPrompt,
                timeoutSeconds: timeoutSeconds
            )
        }
    }

    // MARK: - Anthropic Messages API

    private static func callAnthropic(model: String,
                                       messages: [Message],
                                       maxTokens: Int,
                                       systemPrompt: String?,
                                       timeoutSeconds: Int) throws -> String {
        guard let apiKey = ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"],
              !apiKey.isEmpty else { throw EscalateError.missingApiKey(.anthropic) }

        // Build the body. Anthropic uses top-level `system` (not in
        // messages array) for system prompts.
        var body: [String: Any] = [
            "model": model,
            "max_tokens": maxTokens,
            "messages": messages
                .filter { $0.role != "system" }
                .map { ["role": $0.role, "content": $0.content] }
        ]
        // Either the explicit systemPrompt or a system message from the array.
        let sys = systemPrompt ?? messages.first(where: { $0.role == "system" })?.content
        if let s = sys { body["system"] = s }

        let bodyData = try JSONSerialization.data(withJSONObject: body)
        let response = try curlPost(
            url: "https://api.anthropic.com/v1/messages",
            headers: [
                "x-api-key": apiKey,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            ],
            body: bodyData,
            timeoutSeconds: timeoutSeconds
        )

        // Anthropic response shape:
        //   { content: [{ type: "text", text: "..." }, ...], ... }
        guard let json = try? JSONSerialization.jsonObject(with: response) as? [String: Any],
              let content = json["content"] as? [[String: Any]] else {
            throw EscalateError.parseError("unexpected response shape: " +
                                            (String(data: response, encoding: .utf8) ?? "<binary>"))
        }
        // Concatenate all text blocks (Anthropic can return multiple).
        let texts = content.compactMap { block -> String? in
            guard let type = block["type"] as? String, type == "text",
                  let text = block["text"] as? String else { return nil }
            return text
        }
        return texts.joined(separator: "\n\n")
    }

    // MARK: - OpenAI Chat Completions API

    private static func callOpenAI(model: String,
                                    messages: [Message],
                                    maxTokens: Int,
                                    systemPrompt: String?,
                                    timeoutSeconds: Int) throws -> String {
        guard let apiKey = ProcessInfo.processInfo.environment["OPENAI_API_KEY"],
              !apiKey.isEmpty else { throw EscalateError.missingApiKey(.openai) }

        // OpenAI uses messages[0].role="system" — splice in if needed.
        var msgs: [[String: Any]] = []
        if let sys = systemPrompt ?? messages.first(where: { $0.role == "system" })?.content {
            msgs.append(["role": "system", "content": sys])
        }
        for m in messages where m.role != "system" {
            msgs.append(["role": m.role, "content": m.content])
        }
        let body: [String: Any] = [
            "model": model,
            "max_tokens": maxTokens,
            "messages": msgs
        ]
        let bodyData = try JSONSerialization.data(withJSONObject: body)
        let response = try curlPost(
            url: "https://api.openai.com/v1/chat/completions",
            headers: [
                "Authorization": "Bearer \(apiKey)",
                "Content-Type": "application/json"
            ],
            body: bodyData,
            timeoutSeconds: timeoutSeconds
        )

        // OpenAI response:
        //   { choices: [{ message: { content: "..." } }] }
        guard let json = try? JSONSerialization.jsonObject(with: response) as? [String: Any],
              let choices = json["choices"] as? [[String: Any]],
              let first = choices.first,
              let message = first["message"] as? [String: Any],
              let content = message["content"] as? String else {
            throw EscalateError.parseError("unexpected response shape: " +
                                            (String(data: response, encoding: .utf8) ?? "<binary>"))
        }
        return content
    }

    // MARK: - curl shell-out

    private static func curlPost(url: String,
                                   headers: [String: String],
                                   body: Data,
                                   timeoutSeconds: Int) throws -> Data {
        let p = Process()
        p.launchPath = "/usr/bin/env"

        var args = ["curl", "-sS", "--max-time", "\(timeoutSeconds)",
                    "-X", "POST", url]
        for (k, v) in headers {
            args.append("-H")
            args.append("\(k): \(v)")
        }
        // Body via stdin to avoid command-line length limits + leakage in ps.
        args.append("--data-binary")
        args.append("@-")
        args.append("-w")
        args.append("\n%{http_code}")  // append status code after body, on a fresh line
        p.arguments = args

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        p.standardInput = stdinPipe
        p.standardOutput = stdoutPipe
        p.standardError = stderrPipe

        do { try p.run() } catch { throw EscalateError.curlNotFound }

        stdinPipe.fileHandleForWriting.write(body)
        stdinPipe.fileHandleForWriting.closeFile()

        let output = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stderr = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()

        if p.terminationStatus != 0 {
            let msg = String(data: stderr, encoding: .utf8) ?? "<curl error>"
            throw EscalateError.httpError(status: -1, body: msg)
        }

        // Split body / status code. Status is the last line.
        let raw = String(data: output, encoding: .utf8) ?? ""
        guard let lastNewline = raw.lastIndex(of: "\n") else {
            // No newline — entire output is the response, no status separator
            return output
        }
        let bodyStr = String(raw[..<lastNewline])
        let statusStr = String(raw[raw.index(after: lastNewline)...]).trimmingCharacters(in: .whitespaces)
        if let status = Int(statusStr), status >= 400 {
            throw EscalateError.httpError(status: status, body: bodyStr)
        }
        return bodyStr.data(using: .utf8) ?? Data()
    }
}
