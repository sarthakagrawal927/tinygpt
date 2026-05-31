import Foundation
import TinyGPTIO
import TinyGPTModel

/// `tinygpt hf-inspect <dir>` — point at a downloaded HuggingFace
/// model directory (the `config.json`, `tokenizer.json`,
/// `model.safetensors` triple) and print what we find. This is the
/// diagnostic step that says "yes we can load this" or "no, here's
/// what's unsupported."
///
/// Once the loader is wired (next commit), `tinygpt hf-load` will
/// actually instantiate a TinyGPTModel from the HF weights.
enum HFInspect {
    static func run(args: [String]) {
        guard let dirPath = args.first else {
            fputs("usage: tinygpt hf-inspect <dir-with-config.json-and-safetensors>\n", stderr)
            exit(2)
        }
        if dirPath == "-h" || dirPath == "--help" {
            print("""
            usage: tinygpt hf-inspect <model-dir>

            Inspect a downloaded HuggingFace model directory: prints
            architecture, vocab size, layer counts, GQA layout, context
            length, RoPE theta, tied-embedding flag, and confirms
            whether tinygpt can load it.

            The directory must contain at minimum config.json plus
            either model.safetensors or a pytorch_model.bin shard set.
            tokenizer.json is optional but recommended.
            """)
            exit(0)
        }
        let dir = URL(fileURLWithPath: dirPath)
        let configURL = dir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            fputs("no config.json in \(dirPath)\n", stderr)
            exit(1)
        }

        let cfg: HuggingFaceConfig
        do { cfg = try HuggingFaceConfig.read(configURL) }
        catch { fputs("config error: \(error)\n", stderr); exit(1) }

        print("""

        HuggingFace model inventory: \(dir.path)
        ----------------------------------------
        architecture:     \(cfg.architectures.joined(separator: ", "))
        vocab_size:       \(cfg.vocabSize)
        hidden_size:      \(cfg.hiddenSize)
        intermediate_size:\(cfg.intermediateSize)
        n_layers:         \(cfg.numHiddenLayers)
        n_heads:          \(cfg.numAttentionHeads)
        n_kv_heads:       \(cfg.numKeyValueHeads)\(cfg.numAttentionHeads != cfg.numKeyValueHeads ? "  (GQA)" : "")
        ctx_len:          \(cfg.maxPositionEmbeddings)
        hidden_act:       \(cfg.hiddenAct)
        norm_eps:         \(cfg.rmsNormEps)
        rope_theta:       \(cfg.ropeTheta)
        tied_embeddings:  \(cfg.tieWordEmbeddings)
        rope_scaling:     \(cfg.ropeScaling.map { "\($0)" } ?? "(none)")

        """)
        let approxParams = estimateParams(cfg: cfg)
        print("  approximate params: \(formatLargeInt(approxParams))")

        let missing = HFWeightMapping.missingCapabilities(for: cfg)
        if missing.isEmpty {
            print("\n✓ TinyGPTModel can load this architecture as-is.")
        } else {
            print("\n⚠ TinyGPTModel needs \(missing.count) capability(ies) added before loading this model:")
            for item in missing {
                print("    • \(item)")
            }
            print("\n  Each is a discrete piece of engineering. Once they land, the same")
            print("  safetensors + config inspector above will say ✓.")
        }

        // Look for safetensors files
        let candidates = ["model.safetensors", "model-00001-of-00001.safetensors"]
        let multiShard = (try? FileManager.default.contentsOfDirectory(at: dir,
                          includingPropertiesForKeys: [.fileSizeKey]))?
            .filter { $0.lastPathComponent.hasSuffix(".safetensors") } ?? []

        print("\nsafetensors files:")
        var totalBytes = 0
        for url in multiShard.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            if let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) {
                totalBytes += size
                print("  \(url.lastPathComponent)  \(formatBytes(size))")
            }
        }
        if !multiShard.isEmpty {
            print("  total: \(formatBytes(totalBytes))")
        } else {
            print("  (none found)")
        }
        _ = candidates  // silence "unused"

        // Try reading the first shard's header so we see what's in it.
        if let first = multiShard.first {
            print("\nfirst shard tensors:")
            do {
                let file = try SafetensorsReader.read(first)
                print("  metadata: \(file.metadata)")
                print("  \(file.tensors.count) tensors")
                let sorted = file.tensors.sorted(by: { $0.key < $1.key }).prefix(8)
                for (name, info) in sorted {
                    print("    \(name)  \(info.dtype)  shape=\(info.shape)")
                }
                if file.tensors.count > 8 {
                    print("    ... (+\(file.tensors.count - 8) more)")
                }
            } catch {
                print("  read failed: \(error)")
            }
        }
    }

    private static func estimateParams(cfg: HuggingFaceConfig) -> Int {
        // Rough estimate: embeddings + L * (4·d² + 3·d·d_ff) + final norm + lm_head
        let d = cfg.hiddenSize
        let dff = cfg.intermediateSize
        let L = cfg.numHiddenLayers
        let v = cfg.vocabSize
        let embed = v * d
        let perLayer = 4 * d * d + 3 * d * dff + 2 * d  // attn (q,k,v,o) + mlp (gate,up,down) + 2 norms
        let final = d
        let lmHead = cfg.tieWordEmbeddings ? 0 : v * d
        return embed + L * perLayer + final + lmHead
    }

    private static func formatLargeInt(_ n: Int) -> String {
        if n >= 1_000_000_000 { return String(format: "%.1f B", Double(n) / 1_000_000_000) }
        if n >= 1_000_000 { return String(format: "%.1f M", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0f K", Double(n) / 1_000) }
        return "\(n)"
    }

    private static func formatBytes(_ n: Int) -> String {
        if n >= 1_000_000_000 { return String(format: "%.1f GB", Double(n) / 1_000_000_000) }
        if n >= 1_000_000 { return String(format: "%.0f MB", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0f KB", Double(n) / 1_000) }
        return "\(n) B"
    }
}
