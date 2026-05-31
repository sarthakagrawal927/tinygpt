import Foundation
import TinyGPTIO

/// `tinygpt pull --tag <name> [--out path]` — download a checkpoint from R2.
enum CloudPull {
    static func run(args: [String]) {
        var tag: String?
        var out: String?
        var dryRun = false
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--tag":     tag = args[i+1]; i += 2
            case "--out":     out = args[i+1]; i += 2
            case "--dry-run": dryRun = true; i += 1
            case "-h", "--help": exitUsage()
            default:
                fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
            }
        }
        guard let tag = tag else {
            fputs("pull: --tag <name> required\n", stderr); exitUsage()
        }
        let remoteKey = tag.hasSuffix(".tinygpt") ? tag : "\(tag).tinygpt"
        // Default output: same filename in current dir
        let localPath = out ?? URL(fileURLWithPath: remoteKey).lastPathComponent

        do {
            try R2Client.verifyAwsCli()
            let creds = try R2Client.resolveCredentials()
            print("→ pulling s3://\(creds.bucket)/\(remoteKey) → \(localPath)")
            try R2Client.pull(remoteKey: remoteKey, localPath: localPath,
                              creds: creds, dryRun: dryRun)
            if !dryRun {
                print("✓ downloaded")
            }
        } catch {
            fputs("\(error)\n", stderr)
            exit(1)
        }
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt pull --tag <name> [--out path] [--dry-run]

        Download a checkpoint from Cloudflare R2. Same credential
        resolution as `tinygpt push`.
        """)
        exit(2)
    }
}
