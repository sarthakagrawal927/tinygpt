import Foundation
import TinyGPTServe

/// `tinygpt escalate --provider <p> --prompt "..."` — direct cloud
/// escalation entry point. Calls the configured cloud provider with
/// the prompt; returns the assistant's response.
///
/// This is the explicit/manual half of the cloud-escalation
/// architecture. The implicit/agent-driven half — when the on-device
/// specialist decides it doesn't know and routes to cloud — lives in
/// AgentLoop and uses `CloudEscalate.complete(...)` programmatically.
///
/// USAGE
///   tinygpt escalate --provider anthropic --prompt "Explain RoPE"
///   tinygpt escalate --provider openai --model gpt-4o --system "..." --prompt "..."
enum Escalate {
    static func run(args: [String]) {
        var providerName = "anthropic"
        var model: String? = nil
        var prompt: String? = nil
        var systemPrompt: String? = nil
        var maxTokens: Int = 1024
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--provider": providerName = args[i+1]; i += 2
            case "--model":    model = args[i+1]; i += 2
            case "--prompt":   prompt = args[i+1]; i += 2
            case "--system":   systemPrompt = args[i+1]; i += 2
            case "--max-tokens": maxTokens = Int(args[i+1]) ?? 1024; i += 2
            case "-h", "--help": exitUsage()
            default:
                fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
            }
        }
        guard let prompt = prompt else {
            fputs("escalate: --prompt required\n", stderr); exitUsage()
        }
        guard let provider = CloudEscalate.Provider(rawValue: providerName.lowercased()) else {
            fputs("escalate: unknown provider '\(providerName)'. Pick anthropic|openai.\n", stderr)
            exit(2)
        }

        do {
            let response = try CloudEscalate.complete(
                provider: provider,
                model: model,
                messages: [CloudEscalate.Message(role: "user", content: prompt)],
                maxTokens: maxTokens,
                systemPrompt: systemPrompt
            )
            print(response)
        } catch {
            fputs("\(error)\n", stderr)
            exit(1)
        }
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt escalate --provider {anthropic|openai} --prompt "..."

        Direct cloud escalation — call a larger remote model when the
        on-device specialist defers (or for testing the cloud path).

        Auth (env vars; never logged):
          ANTHROPIC_API_KEY   for --provider anthropic
          OPENAI_API_KEY      for --provider openai

        Options:
          --provider {anthropic|openai}     Default: anthropic
          --model <name>                    Provider-specific. Defaults:
                                              anthropic: claude-sonnet-4-5
                                              openai: gpt-4o-mini
          --system "..."                    Optional system prompt
          --max-tokens N                    Response length cap (default 1024)
        """)
        exit(2)
    }
}
