import Foundation

/// Cloudflare R2 client — shell-out wrapper around the `aws s3` CLI with
/// R2's S3-compatible endpoint.
///
/// Why shell-out rather than native Swift HTTP + SigV4: SigV4 signing is
/// ~200 lines of HMAC chains that we'd have to maintain forever; `aws`
/// CLI handles it correctly + transparently (refreshes auth, handles
/// retries, multipart uploads, etc.). The cost is one extra dep
/// (`brew install awscli`) which most ML devs already have.
///
/// AUTH: reads four values from env first, falls back to
/// `~/.config/tinygpt/r2.env`:
///   R2_ACCOUNT_ID       — your Cloudflare account ID
///   R2_ACCESS_KEY_ID    — R2 API token
///   R2_SECRET_ACCESS_KEY — R2 API token secret
///   R2_BUCKET           — bucket name
///
/// Endpoint is derived from account ID:
///   https://<ACCOUNT_ID>.r2.cloudflarestorage.com
///
/// PRICING (current as of 2026-05): $0.015 / GB-month storage, $0 egress
/// (the killer feature), $4.50/M class-A ops (uploads), $0.36/M class-B
/// (downloads). Free tier: 10 GB. A 100 GB cache costs ~$1.50/month.
public enum R2Client {

    public struct Credentials {
        public let accountID: String
        public let accessKeyID: String
        public let secretAccessKey: String
        public let bucket: String
        public var endpoint: String { "https://\(accountID).r2.cloudflarestorage.com" }
    }

    public enum R2Error: Error, CustomStringConvertible {
        case missingCredentials(String)
        case awsCliNotFound
        case commandFailed(exit: Int32, stderr: String)

        public var description: String {
            switch self {
            case .missingCredentials(let which):
                return "R2 credentials missing: \(which). Run `tinygpt cloud setup` or set R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_BUCKET."
            case .awsCliNotFound:
                return "`aws` CLI not found in PATH. Install via `brew install awscli`."
            case .commandFailed(let exit, let stderr):
                return "aws command failed (exit \(exit)): \(stderr)"
            }
        }
    }

    /// Resolve credentials: env vars first, then ~/.config/tinygpt/r2.env.
    /// Returns nil if any of the four required values are missing.
    public static func resolveCredentials() throws -> Credentials {
        let env = ProcessInfo.processInfo.environment
        var values: [String: String] = [:]

        // Layer 1: env
        for key in ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"] {
            if let v = env[key], !v.isEmpty { values[key] = v }
        }

        // Layer 2: ~/.config/tinygpt/r2.env (only fills missing keys)
        let home = FileManager.default.homeDirectoryForCurrentUser
        let confURL = home.appendingPathComponent(".config/tinygpt/r2.env")
        if let data = try? String(contentsOf: confURL, encoding: .utf8) {
            for line in data.split(separator: "\n") {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else { continue }
                let parts = trimmed.split(separator: "=", maxSplits: 1)
                guard parts.count == 2 else { continue }
                let k = String(parts[0]).trimmingCharacters(in: .whitespaces)
                var v = String(parts[1]).trimmingCharacters(in: .whitespaces)
                // Strip surrounding quotes if present
                if v.hasPrefix("\"") && v.hasSuffix("\"") { v = String(v.dropFirst().dropLast()) }
                if values[k] == nil { values[k] = v }
            }
        }

        let missing = ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"]
            .filter { values[$0] == nil }
        if !missing.isEmpty {
            throw R2Error.missingCredentials(missing.joined(separator: ", "))
        }

        return Credentials(
            accountID: values["R2_ACCOUNT_ID"]!,
            accessKeyID: values["R2_ACCESS_KEY_ID"]!,
            secretAccessKey: values["R2_SECRET_ACCESS_KEY"]!,
            bucket: values["R2_BUCKET"]!
        )
    }

    /// Verify `aws` CLI is reachable.
    public static func verifyAwsCli() throws {
        let p = Process()
        p.launchPath = "/usr/bin/env"
        p.arguments = ["which", "aws"]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = Pipe()
        do { try p.run() } catch { throw R2Error.awsCliNotFound }
        p.waitUntilExit()
        if p.terminationStatus != 0 {
            throw R2Error.awsCliNotFound
        }
    }

    /// Run `aws s3 cp/ls/rm` with R2 endpoint + credentials injected via env.
    /// `dryRun=true` prints the command instead of executing — useful when
    /// the user hasn't set up credentials yet.
    public static func runS3(_ args: [String],
                              creds: Credentials,
                              dryRun: Bool = false) throws -> String {
        var fullArgs = ["s3"] + args + ["--endpoint-url", creds.endpoint]

        if dryRun {
            // Show the command WITHOUT the secret values. Just enough for
            // the user to understand what would happen.
            let pretty = "aws " + fullArgs.joined(separator: " ")
            print("[dry-run] would execute:")
            print("  \(pretty)")
            print("  (with AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY from R2 credentials in env)")
            return ""
        }

        let p = Process()
        p.launchPath = "/usr/bin/env"
        // We pass credentials via env (not args) so they don't show up in
        // `ps` output. The aws CLI reads these standardly.
        var env = ProcessInfo.processInfo.environment
        env["AWS_ACCESS_KEY_ID"] = creds.accessKeyID
        env["AWS_SECRET_ACCESS_KEY"] = creds.secretAccessKey
        env["AWS_DEFAULT_REGION"] = "auto"  // R2 doesn't use regions
        p.environment = env
        fullArgs.insert("aws", at: 0)
        p.arguments = fullArgs

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        p.standardOutput = stdoutPipe
        p.standardError = stderrPipe

        try p.run()
        p.waitUntilExit()

        let stdout = String(data: stdoutPipe.fileHandleForReading.readDataToEndOfFile(),
                            encoding: .utf8) ?? ""
        let stderr = String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(),
                            encoding: .utf8) ?? ""

        if p.terminationStatus != 0 {
            throw R2Error.commandFailed(exit: p.terminationStatus, stderr: stderr)
        }
        return stdout
    }

    /// Upload a file to R2.
    public static func push(localPath: String,
                             remoteKey: String,
                             creds: Credentials,
                             dryRun: Bool = false) throws {
        let dst = "s3://\(creds.bucket)/\(remoteKey)"
        _ = try runS3(["cp", localPath, dst], creds: creds, dryRun: dryRun)
    }

    /// Download a file from R2 to a local path.
    public static func pull(remoteKey: String,
                             localPath: String,
                             creds: Credentials,
                             dryRun: Bool = false) throws {
        let src = "s3://\(creds.bucket)/\(remoteKey)"
        _ = try runS3(["cp", src, localPath], creds: creds, dryRun: dryRun)
    }

    /// List bucket contents at the given prefix (empty = root).
    public static func list(prefix: String = "",
                             creds: Credentials,
                             dryRun: Bool = false) throws -> String {
        let target = prefix.isEmpty
            ? "s3://\(creds.bucket)/"
            : "s3://\(creds.bucket)/\(prefix)"
        return try runS3(["ls", target, "--recursive"], creds: creds, dryRun: dryRun)
    }

    /// Delete a key from R2.
    public static func delete(remoteKey: String,
                               creds: Credentials,
                               dryRun: Bool = false) throws {
        let target = "s3://\(creds.bucket)/\(remoteKey)"
        _ = try runS3(["rm", target], creds: creds, dryRun: dryRun)
    }
}
