import Foundation
import CryptoKit

/// Persistent KV cache by hash of (model identity + prompt + cache-affecting
/// config) — the "system prompt cache" that lets a `tinygpt sample` invocation
/// skip the prefill on its second run with the same prompt.
///
/// # Why this exists
///
/// An agent specialist's first few turns are a fixed system prompt followed
/// by the live tool-call history. The system prompt is identical every
/// session; recomputing its K, V at every launch is wall-clock waste — on a
/// titan-ish ~1B model with a 1 KB system prompt we measured ~1.5 s of
/// prefill, the bulk of TTFT. Caching it to disk and mmap-loading on the
/// next launch drops that to a single small forward (the rewind step in
/// Sample.swift's `skipPrefill` branch) — 10×-100× TTFT improvement
/// depending on prompt length, model depth, and dtype.
///
/// # Cache key
///
/// SHA-256 over the bytes of:
///   - `modelName` (config field — uniquely identifies architecture +
///     trained checkpoint for our presets, and the HF model id for HF
///     loads). Two models that diverged in training but share modelName
///     would collide silently; we treat that as a user error rather than
///     try to fingerprint weight tensors (a fingerprint over a multi-GB
///     model is itself a 100 ms+ cost we don't want to pay every launch).
///   - The raw prompt UTF-8 bytes (NOT the token ids — a tokenizer swap
///     between launches must invalidate the cache, and we can't trust the
///     caller to remember to bump some version field).
///   - The vocab size (catches a tokenizer change that didn't change the
///     prompt text — e.g., growing the vocab from 256 → 32000).
///   - `nLayers` (catches a pruning-induced architecture change).
///   - The KV dtype tag (string "fp32" / "fp16" / "bf16" / "kivi-int8" /
///     "kivi-int4"). Same prompt, different storage format → different
///     cache file.
///   - `useYOCO` (1 byte). Affects which layers get K, V written.
///
/// Truncated to 12 hex chars for the filename. Collision probability at
/// 12 hex = 48 bits is ~ 1 in 2^48 ≈ 0.0000000004% over a working set of
/// a few hundred prompts — fine.
///
/// # On-disk format
///
/// `KVCache.saveToDisk` already serialises K, V tensors + currentLength.
/// This module just chooses the file PATH from the hash and exposes a
/// sidecar `.meta.json` so a human poking at the cache directory can see
/// which prompt and config a cache file belongs to (with the prompt
/// truncated to the first 200 chars for sanity).
public enum KVCachePersist {

    /// Tag for the cache's KV storage format. Lives in the hash so a
    /// `--kv-quantize fp16` cache and an `fp32` cache for the same prompt
    /// don't collide (they have different layouts and the load path would
    /// crash on a dtype mismatch).
    public enum KVTag: String, Sendable, Equatable {
        case fp32, fp16, bf16
        case kiviInt8 = "kivi-int8"
        case kiviInt4 = "kivi-int4"
    }

    /// All inputs that affect the cache's binary layout. Bundled so we
    /// can pass one struct around instead of N parameters.
    ///
    /// `modelFileFingerprint` is the (size, mtime) pair of the model file
    /// — cheap to compute (one `stat`) and good enough to invalidate the
    /// cache when the user retrains and overwrites the same path. Two
    /// distinct checkpoints with the same modelName + same size + same
    /// mtime would collide, but that requires deliberately copying the
    /// mtime forward, which is well outside "user error" territory.
    public struct Key: Sendable {
        public let modelName: String
        public let modelFileFingerprint: String   // empty for in-memory models
        public let prompt: String
        public let vocabSize: Int
        public let nLayers: Int
        public let kvTag: KVTag
        public let useYOCO: Bool
        public init(modelName: String, modelFileFingerprint: String = "",
                    prompt: String, vocabSize: Int,
                    nLayers: Int, kvTag: KVTag, useYOCO: Bool) {
            self.modelName = modelName
            self.modelFileFingerprint = modelFileFingerprint
            self.prompt = prompt
            self.vocabSize = vocabSize
            self.nLayers = nLayers
            self.kvTag = kvTag
            self.useYOCO = useYOCO
        }
    }

    /// Compute a cheap fingerprint of the model file: "size:mtime" in
    /// decimal. Returns an empty string when the file isn't stat'able
    /// (the path is bogus, HF model dir, etc.) — in that case the key
    /// falls back to modelName-only collision risk.
    public static func fingerprint(of path: String) -> String {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attrs[.size] as? NSNumber,
              let mtime = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970
        else { return "" }
        return "\(size.int64Value):\(Int(mtime * 1000))"
    }

    /// Compute the cache filename inside `dir`. Pure function — does NOT
    /// hit disk. Returns the `.kvcache` file and the sibling `.meta.json`
    /// path as a pair.
    public static func paths(for key: Key, in dir: URL) -> (cache: URL, meta: URL) {
        var hasher = SHA256()
        hasher.update(data: Data(key.modelName.utf8))
        hasher.update(data: Data(key.modelFileFingerprint.utf8))
        hasher.update(data: Data(key.prompt.utf8))
        var vs = Int32(key.vocabSize)
        hasher.update(data: Data(bytes: &vs, count: MemoryLayout<Int32>.size))
        var nl = Int32(key.nLayers)
        hasher.update(data: Data(bytes: &nl, count: MemoryLayout<Int32>.size))
        hasher.update(data: Data(key.kvTag.rawValue.utf8))
        var yoco: UInt8 = key.useYOCO ? 1 : 0
        hasher.update(data: Data(bytes: &yoco, count: MemoryLayout<UInt8>.size))
        let digest = hasher.finalize()
        let hex = digest.prefix(6).map { String(format: "%02x", $0) }.joined()
        let base = sanitize(key.modelName)
        // `<safe-modelname>-<hex>` keeps cache files browseable: when the
        // user has a few caches they can guess at filenames. Falls back
        // to pure hex when modelName has nothing alphanumeric.
        let stem = base.isEmpty ? hex : "\(base)-\(hex)"
        return (
            cache: dir.appendingPathComponent("\(stem).kvcache"),
            meta:  dir.appendingPathComponent("\(stem).meta.json")
        )
    }

    /// Ensure the cache directory exists. Idempotent — calls
    /// `createDirectory` with `withIntermediateDirectories: true`.
    public static func ensureDir(_ dir: URL) throws {
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
    }

    /// Write the sidecar metadata. Best-effort — failure here is not
    /// fatal (the cache file itself is what matters for correctness).
    public static func writeMeta(_ key: Key, to url: URL,
                                  tokens: Int, bytes: Int) {
        // Truncate prompt to keep the metadata file small. Anyone wanting
        // the full prompt can re-derive it from their shell history.
        let promptPreview = String(key.prompt.prefix(200))
        let obj: [String: Any] = [
            "modelName": key.modelName,
            "promptPreview": promptPreview,
            "promptLength": key.prompt.count,
            "vocabSize": key.vocabSize,
            "nLayers": key.nLayers,
            "kvTag": key.kvTag.rawValue,
            "useYOCO": key.useYOCO,
            "tokens": tokens,
            "bytes": bytes,
            "createdAtEpoch": Int(Date().timeIntervalSince1970),
        ]
        // `JSONSerialization` keeps the impl dependency-free — we don't
        // need Codable for a 9-field record. Any failure here drops on the
        // floor; the .kvcache file is the source of truth.
        if let data = try? JSONSerialization.data(withJSONObject: obj,
                                                   options: [.prettyPrinted]) {
            try? data.write(to: url, options: .atomic)
        }
    }

    /// Strip non-alphanumeric / non-dash characters from a model name so
    /// it's safe inside a filename. Path-safe across macOS, Linux, and
    /// the browser-side blob store that the Mac sometimes shares with.
    private static func sanitize(_ s: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(.init(charactersIn: "-_."))
        let scalars = s.unicodeScalars.filter { allowed.contains($0) }
        return String(String.UnicodeScalarView(scalars))
    }
}
