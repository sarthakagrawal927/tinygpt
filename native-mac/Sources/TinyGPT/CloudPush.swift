import Foundation
import TinyGPTIO

/// `tinygpt push <local.tinygpt> --tag <name>` — upload a checkpoint to
/// Cloudflare R2 (S3-compatible, zero egress, cheap).
///
/// See R2Client.swift for the auth + CLI plumbing. This file is just the
/// CLI handler that maps user args to R2Client calls.
enum CloudPush {
    static func run(args: [String]) {
        var localPath: String?
        var tag: String?
        var dryRun = false
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--tag":     tag = args[i+1]; i += 2
            case "--dry-run": dryRun = true; i += 1
            case "-h", "--help": exitUsage()
            default:
                if args[i].hasPrefix("-") {
                    fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
                }
                if localPath == nil { localPath = args[i]; i += 1 }
                else { fputs("unexpected positional: \(args[i])\n", stderr); exitUsage() }
            }
        }
        guard let localPath = localPath else {
            fputs("push: missing <local.tinygpt>\n", stderr); exitUsage()
        }
        guard let tag = tag else {
            fputs("push: --tag <name> required\n", stderr); exitUsage()
        }

        // Derive a remote key. Convention: <tag>.tinygpt at bucket root.
        // The user can override the key by passing tag with a slash:
        // `--tag flagship/v1` → flagship/v1.tinygpt
        let remoteKey = tag.hasSuffix(".tinygpt") ? tag : "\(tag).tinygpt"

        do {
            try R2Client.verifyAwsCli()
            let creds = try R2Client.resolveCredentials()
            print("→ pushing \(localPath) → s3://\(creds.bucket)/\(remoteKey)")
            try R2Client.push(localPath: localPath, remoteKey: remoteKey,
                              creds: creds, dryRun: dryRun)
            if !dryRun {
                print("✓ uploaded")
            }
        } catch {
            fputs("\(error)\n", stderr)
            exit(1)
        }
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt push <local.tinygpt> --tag <name> [--dry-run]

        Upload a checkpoint to Cloudflare R2. Uses the `aws` CLI under
        the hood with R2's S3-compatible endpoint.

        Credentials: set in env (R2_ACCOUNT_ID, R2_ACCESS_KEY_ID,
        R2_SECRET_ACCESS_KEY, R2_BUCKET) or in ~/.config/tinygpt/r2.env.
        """)
        exit(2)
    }
}
