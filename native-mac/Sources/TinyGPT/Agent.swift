import Foundation
import MLX
import MLXNN
import TinyGPTModel
import TinyGPTServe

/// `tinygpt agent <model> --tools tools.json [...]` — the agent runtime CLI.
///
/// Wires together:
///   - `ColdStart.loadWithSpinner` for fast model load + spinner UX
///   - `KVCachePersist` for cross-launch system-prompt prefix caching
///   - `ToolSchema.load` for the tool definitions file
///   - `AgentLoop` for the actual conversation loop
///
/// Modes:
///   - default (interactive): REPL — user types lines, agent replies
///   - `--single "<prompt>"`: one shot, print, exit
///   - `--json-out`: every event (assistant turn, tool call, tool result,
///                   answer) printed as one JSON object per line on stdout.
///                   Useful for piping the agent into another process.
enum Agent {
    static func run(args: [String]) {
        var path: String?
        var toolsPath: String?
        var systemPrompt = ""
        var singlePrompt: String? = nil
        var promptCacheDir: String? = nil
        var transcriptPath: String? = nil
        var jsonOut = false
        var temperature: Float = 0.0
        var maxTokensPerTurn = 256
        var maxSteps = 8
        var toolTimeout = 30.0
        var lazyEmbedding = false
        var asyncLoad = true
        var preAllocate = true
        var cloudEscalate: String? = nil    // "anthropic" or "openai"
        var cloudEscalateModel: String? = nil
        // Tool-call extractor (mini-router) — Wave 2.6 scaffold.
        // When set, the agent loads the router checkpoint at startup
        // and, on each user turn, predicts the most likely tool BEFORE
        // the LM is asked to choose. If confidence ≥ routerThreshold,
        // the prediction is fed to the LM as a constrained-decode hint.
        var routerPath: String? = nil
        var routerThreshold: Float = 0.7
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--tools":
                guard i + 1 < args.count else { exitUsage() }
                toolsPath = args[i + 1]; i += 2
            case "--system":
                guard i + 1 < args.count else { exitUsage() }
                systemPrompt = args[i + 1]; i += 2
            case "--single":
                guard i + 1 < args.count else { exitUsage() }
                singlePrompt = args[i + 1]; i += 2
            case "--prompt-cache-dir":
                guard i + 1 < args.count else { exitUsage() }
                promptCacheDir = args[i + 1]; i += 2
            case "--transcript":
                guard i + 1 < args.count else { exitUsage() }
                transcriptPath = args[i + 1]; i += 2
            case "--json-out":
                jsonOut = true; i += 1
            case "--temperature", "--temp":
                guard i + 1 < args.count else { exitUsage() }
                temperature = Float(args[i + 1]) ?? temperature; i += 2
            case "--max-tokens":
                guard i + 1 < args.count else { exitUsage() }
                maxTokensPerTurn = Int(args[i + 1]) ?? maxTokensPerTurn; i += 2
            case "--max-steps":
                guard i + 1 < args.count else { exitUsage() }
                maxSteps = Int(args[i + 1]) ?? maxSteps; i += 2
            case "--tool-timeout":
                guard i + 1 < args.count else { exitUsage() }
                toolTimeout = Double(args[i + 1]) ?? toolTimeout; i += 2
            case "--lazy-embedding":
                lazyEmbedding = true; i += 1
            case "--no-async-load":
                asyncLoad = false; i += 1
            case "--no-kv-preallocate":
                preAllocate = false; i += 1
            case "--cloud-escalate":
                guard i + 1 < args.count else { exitUsage() }
                cloudEscalate = args[i + 1]; i += 2
            case "--cloud-escalate-model":
                guard i + 1 < args.count else { exitUsage() }
                cloudEscalateModel = args[i + 1]; i += 2
            case "--router":
                guard i + 1 < args.count else { exitUsage() }
                routerPath = args[i + 1]; i += 2
            case "--router-threshold":
                guard i + 1 < args.count else { exitUsage() }
                routerThreshold = Float(args[i + 1]) ?? routerThreshold; i += 2
            case "-h", "--help":
                exitUsage()
            default:
                if args[i].hasPrefix("-") {
                    fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
                }
                path = args[i]; i += 1
            }
        }
        guard let path = path else {
            fputs("agent: missing <model> path\n", stderr); exitUsage()
        }
        guard let toolsPath = toolsPath else {
            fputs("agent: --tools <path> is required\n", stderr); exitUsage()
        }
        let url = URL(fileURLWithPath: path)
        let schema: ToolSchema
        do {
            schema = try ToolSchema.load(from: URL(fileURLWithPath: toolsPath))
        } catch {
            fputs("error loading tools schema: \(error)\n", stderr); exit(1)
        }

        // Model load — cold-start spinner path.
        let load: ModelLoader.LoadResult
        do {
            if asyncLoad {
                load = try ColdStart.loadWithSpinner(
                    path: path,
                    deferEmbedding: lazyEmbedding,
                    label: url.lastPathComponent)
            } else {
                print("loading \(url.lastPathComponent)…")
                if lazyEmbedding {
                    load = try ModelLoader.loadLazyEmbedding(path)
                } else {
                    load = try ModelLoader.load(path)
                }
            }
        } catch {
            fputs("error loading model: \(error)\n", stderr); exit(1)
        }
        let cfg = load.config
        let model = load.model
        // Materialise any deferred embedding before the first forward.
        if let h = load.lazyEmbedding {
            do { try h.materialize() }
            catch { fputs("embedding materialisation failed: \(error)\n", stderr); exit(1) }
        }
        fputs("agent: \(formatLargeInt(model.numParameters())) params · ctx \(cfg.contextLength)\n", stderr)

        // Tokenizer — HF if pinned, otherwise byte-level.
        let tokenizer: HFTokenizer?
        if let tokDir = load.hfTokenizerDir {
            do {
                tokenizer = try HFTokenizer.loadBlocking(from: tokDir)
            } catch {
                fputs("warning: tokenizer load failed (\(error)); falling back to byte-level\n", stderr)
                tokenizer = nil
            }
        } else {
            tokenizer = nil
        }

        // Build the system prompt and resolve the persistent KV cache.
        let sysPrompt = AgentLoop.systemPrompt(userSystem: systemPrompt, schema: schema)
        let cacheKey = KVCachePersist.Key(
            modelName: cfg.modelName,
            modelFileFingerprint: KVCachePersist.fingerprint(of: path),
            prompt: sysPrompt,
            vocabSize: cfg.vocabSize, nLayers: cfg.nLayers,
            kvTag: .fp32, useYOCO: cfg.useYOCO
        )
        var resolvedCachePath: URL? = nil
        var resolvedMetaPath: URL? = nil
        if let dir = promptCacheDir {
            let dirURL = URL(fileURLWithPath: (dir as NSString).expandingTildeInPath)
            do {
                try KVCachePersist.ensureDir(dirURL)
                let (cacheURL, metaURL) = KVCachePersist.paths(for: cacheKey, in: dirURL)
                resolvedCachePath = cacheURL
                resolvedMetaPath = metaURL
            } catch {
                fputs("warning: could not create prompt cache dir \(dir): \(error)\n", stderr)
            }
        }
        var cache: KVCache? = nil
        var alreadyPrefilled = false
        if let cacheURL = resolvedCachePath, FileManager.default.fileExists(atPath: cacheURL.path) {
            do {
                cache = try KVCache.load(from: cacheURL, nLayers: cfg.nLayers)
                alreadyPrefilled = true
                fputs("agent: loaded system-prompt cache (\(cache!.currentLength) tokens) — skipping prefill\n", stderr)
                if preAllocate {
                    cache!.migrateToPreAlloc(capacity: cfg.contextLength)
                }
            } catch {
                fputs("warning: prefix cache load failed (\(error)); building fresh\n", stderr)
            }
        }
        if cache == nil {
            let cap = preAllocate ? cfg.contextLength : nil
            cache = KVCache(nLayers: cfg.nLayers, preAllocCapacity: cap)
        }

        // Transcript file (one event per line).
        let transcriptURL: URL? = transcriptPath.map { URL(fileURLWithPath: $0) }
        if let url = transcriptURL {
            try? "".write(to: url, atomically: true, encoding: .utf8)
        }

        // Resolve the cloud-escalate provider, if requested.
        var cloudProvider: CloudEscalate.Provider? = nil
        if let cloud = cloudEscalate {
            guard let resolved = CloudEscalate.Provider(rawValue: cloud.lowercased()) else {
                fputs("agent: --cloud-escalate must be one of: anthropic|openai (got \"\(cloud)\")\n", stderr)
                exit(2)
            }
            cloudProvider = resolved
            fputs("agent: cloud escalation enabled — provider=\(resolved.rawValue)" +
                   (cloudEscalateModel.map { ", model=\($0)" } ?? "") + "\n", stderr)
        }

        // Tool-call extractor (mini-router) — optional pre-step.
        // When `--router <path>` is set, load the checkpoint + label
        // sidecar. The loop uses it to predict the most likely tool
        // before the LM forward pass; ≥ threshold → constrained-decode
        // hint, else fall through to normal LM-chooses behaviour.
        var routerHook: AgentLoop.RouterHook? = nil
        if let routerPath = routerPath {
            do {
                let labelsURL = ToolRouterLabels.sidecarURL(
                    forCheckpoint: URL(fileURLWithPath: routerPath))
                let labels = try ToolRouterLabels.load(from: labelsURL)
                let router = try ToolRouterLoader.load(
                    path: routerPath, numClasses: labels.labels.count)
                routerHook = AgentLoop.RouterHook(
                    router: router,
                    labels: labels.labels,
                    threshold: routerThreshold)
                fputs("agent: router loaded — \(labels.labels.count) classes" +
                      String(format: ", threshold=%.2f\n", routerThreshold), stderr)
            } catch {
                fputs("agent: --router load failed (\(error)); continuing without router\n", stderr)
            }
        }

        let loop = AgentLoop(
            model: model, cfg: cfg, tokenizer: tokenizer,
            schema: schema, cache: cache!,
            temperature: temperature, maxTokensPerTurn: maxTokensPerTurn,
            toolTimeoutSec: toolTimeout,
            transcriptURL: transcriptURL, jsonOut: jsonOut,
            maxAgentSteps: maxSteps,
            cloudEscalateProvider: cloudProvider,
            cloudEscalateModel: cloudEscalateModel,
            routerHook: routerHook
        )

        // Prefill (no-op if loaded from disk).
        TrainSupport.installSigintHandler()
        TrainSupport.stopRequested.reset()
        let tPrefill = Date()
        let prefillTokens = loop.prefillSystemPrompt(sysPrompt, alreadyPrefilled: alreadyPrefilled)
        let prefillElapsed = -tPrefill.timeIntervalSinceNow
        if !alreadyPrefilled {
            fputs(String(format: "agent: prefill %d tokens in %.2fs\n",
                          prefillTokens, prefillElapsed), stderr)
            if let cacheURL = resolvedCachePath, let cache = cache {
                do {
                    try cache.saveToDisk(to: cacheURL)
                    fputs("agent: saved system-prompt cache → \(cacheURL.lastPathComponent)\n", stderr)
                    if let meta = resolvedMetaPath {
                        KVCachePersist.writeMeta(
                            cacheKey, to: meta,
                            tokens: cache.currentLength,
                            bytes: 0)
                    }
                } catch {
                    fputs("warning: prompt cache save failed: \(error)\n", stderr)
                }
            }
        }

        // Single-shot vs REPL.
        if let prompt = singlePrompt {
            let answer = loop.runTurn(userText: prompt)
            if !jsonOut {
                print("")
                print(answer)
            }
            return
        }
        runREPL(loop: loop, jsonOut: jsonOut)
    }

    private static func runREPL(loop: AgentLoop, jsonOut: Bool) {
        if !jsonOut {
            fputs("tinygpt agent — interactive. Ctrl-D or `:quit` to exit.\n", stderr)
        }
        while true {
            if !jsonOut {
                fputs("\nyou> ", stderr)
            }
            guard let line = readLine(strippingNewline: true) else { break }
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }
            if trimmed == ":quit" || trimmed == ":q" { break }
            let answer = loop.runTurn(userText: trimmed)
            if !jsonOut {
                print("\nagent> \(answer)")
            }
        }
    }

    private static func formatLargeInt(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt agent <path.tinygpt|model_dir> --tools tools.json [options]

        --tools <path>           OpenAI-compatible tool schema (required)
        --system "..."           Extra system-prompt prefix above the tool block
        --single "<prompt>"      One-shot mode — process this prompt and exit
        --prompt-cache-dir <dir> Persist the system-prompt KV cache by hash so
                                 the next launch with the same prompt skips
                                 prefill. Recommended for production agents.
        --transcript <path>      Append one JSON event per line: user input,
                                 raw assistant text, tool call, tool result,
                                 final answer. Useful for eval + debugging.
        --json-out               Print every event as JSON on stdout instead
                                 of free-form text. Useful for piping into
                                 another process.
        --temperature F          Sampling temperature (default 0 = greedy)
        --max-tokens N           Max tokens generated per assistant step
                                 (default 256)
        --max-steps N            Max tool-call rounds per user turn (default 8)
        --tool-timeout F         Subprocess timeout per tool call (default 30s)
        --lazy-embedding         Defer the token embedding load
        --no-async-load          Disable background-thread model load
        --no-kv-preallocate      Disable KV pre-allocation at max context

        Cloud escalation (north-star: small on-device specialist + cloud fallback):
        --cloud-escalate <p>     Add a synthetic `escalate` tool dispatched
                                 against the named cloud provider when the
                                 on-device model defers. Provider: anthropic|openai.
                                 Requires ANTHROPIC_API_KEY / OPENAI_API_KEY in env.
        --cloud-escalate-model M Override the cloud model name. Defaults:
                                 anthropic→claude-sonnet-4-5, openai→gpt-4o-mini.

        Tool-call extractor (mini-router) — Wave 2.6 scaffold:
        --router <path>          Load a router checkpoint trained via
                                 `tinygpt train-extractor`. Before the LM
                                 forward pass, the router predicts which
                                 tool best matches the user query; high-
                                 confidence picks are fed to the LM as a
                                 constrained-decode hint.
        --router-threshold F     Minimum softmax confidence to fire the
                                 router (default: 0.7).

        examples:
          tinygpt agent specialist.tinygpt --tools tools.json
          tinygpt agent specialist.tinygpt --tools tools.json --single "Debug this error"
          tinygpt agent specialist.tinygpt --tools tools.json --json-out
          tinygpt agent specialist.tinygpt --tools tools.json --cloud-escalate anthropic
        """)
        exit(2)
    }
}
