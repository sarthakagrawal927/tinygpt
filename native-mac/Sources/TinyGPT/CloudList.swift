import Foundation
import TinyGPTIO

/// `tinygpt cloud list|delete|setup` — manage R2 cloud storage.
enum CloudList {
    static func run(args: [String]) {
        guard let sub = args.first else { exitUsage() }
        switch sub {
        case "list":   runList(Array(args.dropFirst()))
        case "delete": runDelete(Array(args.dropFirst()))
        case "setup":  runSetup(Array(args.dropFirst()))
        case "-h", "--help": exitUsage()
        default:
            fputs("unknown cloud subcommand: \(sub)\n", stderr); exitUsage()
        }
    }

    static func runList(_ args: [String]) {
        var prefix = ""
        var dryRun = false
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--prefix":  prefix = args[i+1]; i += 2
            case "--dry-run": dryRun = true; i += 1
            default: fputs("unknown flag: \(args[i])\n", stderr); exit(2)
            }
        }
        do {
            try R2Client.verifyAwsCli()
            let creds = try R2Client.resolveCredentials()
            let out = try R2Client.list(prefix: prefix, creds: creds, dryRun: dryRun)
            if !dryRun {
                if out.isEmpty {
                    print("(bucket is empty)")
                } else {
                    print(out)
                }
            }
        } catch {
            fputs("\(error)\n", stderr); exit(1)
        }
    }

    static func runDelete(_ args: [String]) {
        var tag: String?
        var dryRun = false
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--tag":     tag = args[i+1]; i += 2
            case "--dry-run": dryRun = true; i += 1
            default: fputs("unknown flag: \(args[i])\n", stderr); exit(2)
            }
        }
        guard let tag = tag else {
            fputs("cloud delete: --tag required\n", stderr); exit(2)
        }
        let remoteKey = tag.hasSuffix(".tinygpt") ? tag : "\(tag).tinygpt"
        do {
            try R2Client.verifyAwsCli()
            let creds = try R2Client.resolveCredentials()
            print("→ deleting s3://\(creds.bucket)/\(remoteKey)")
            try R2Client.delete(remoteKey: remoteKey, creds: creds, dryRun: dryRun)
            if !dryRun { print("✓ deleted") }
        } catch {
            fputs("\(error)\n", stderr); exit(1)
        }
    }

    /// Interactive credential setup — writes ~/.config/tinygpt/r2.env mode 600.
    static func runSetup(_ args: [String]) {
        _ = args  // no flags for now

        print("""
        Cloudflare R2 setup
        -------------------
        We'll prompt for the four values needed to talk to R2 and write
        them to ~/.config/tinygpt/r2.env (mode 600 — owner-readable only).

        If you don't have an R2 bucket + API token yet:
          https://dash.cloudflare.com → R2 → Create bucket → Manage API
          tokens → Create API token (Object Read & Write, bucket-scoped)

        """)
        func prompt(_ label: String, secret: Bool = false) -> String {
            print("\(label): ", terminator: "")
            if secret {
                // Best-effort: no TTY mode toggle. Tell the user the input
                // is visible; for true secret entry use `read -s`.
                print("(visible — set via env if you'd prefer hidden) ", terminator: "")
            }
            return readLine() ?? ""
        }
        let accountID = prompt("R2_ACCOUNT_ID")
        let accessKey = prompt("R2_ACCESS_KEY_ID")
        let secret    = prompt("R2_SECRET_ACCESS_KEY", secret: true)
        let bucket    = prompt("R2_BUCKET")

        let home = FileManager.default.homeDirectoryForCurrentUser
        let confDir = home.appendingPathComponent(".config/tinygpt")
        let confURL = confDir.appendingPathComponent("r2.env")
        do {
            try FileManager.default.createDirectory(at: confDir,
                                                     withIntermediateDirectories: true)
            let body = """
            # tinygpt R2 credentials — keep secret.
            # Created by `tinygpt cloud setup` on \(ISO8601DateFormatter().string(from: Date())).
            R2_ACCOUNT_ID=\(accountID)
            R2_ACCESS_KEY_ID=\(accessKey)
            R2_SECRET_ACCESS_KEY=\(secret)
            R2_BUCKET=\(bucket)
            """
            try body.write(to: confURL, atomically: true, encoding: .utf8)
            // Mode 600
            try FileManager.default.setAttributes(
                [.posixPermissions: NSNumber(value: 0o600)],
                ofItemAtPath: confURL.path
            )
            print("\n✓ wrote \(confURL.path) (mode 600)")
            print("verify with: tinygpt cloud list")
        } catch {
            fputs("setup failed: \(error)\n", stderr); exit(1)
        }
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt cloud <subcommand>

        Subcommands:
          list [--prefix <p>]            List bucket contents
          delete --tag <name>            Remove a checkpoint
          setup                          Interactive credential setup
        """)
        exit(2)
    }
}
