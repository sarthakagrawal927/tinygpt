import Foundation
import MLX
import MLXNN

/// On-disk format for a LoRA adapter — separate from `.tinygpt`. A small
/// header + the A/B matrices for each wrapped Linear. Bytes:
///
///     0    4    magic = "TGLA"  (TinyGPT LoRA Adapter)
///     4    4    version (u32, currently 1)
///     8    4    header_len (u32)
///     12   N    JSON header — { config, modelConfig, entries: [{name, shape}, ...] }
///     12+N      raw fp32 matrices in manifest order, each [in,r] then [r,out]
///
/// The header carries the BASE model's config so the adapter refuses to
/// load against an architecture-mismatched checkpoint.
public enum LoraAdapterFormat {
    public static let magic: [UInt8] = Array("TGLA".utf8)
    public static let currentVersion: UInt32 = 1
}

public struct LoraAdapter {
    public struct Entry: Codable, Equatable, Sendable {
        public var name: String      // "blocks.0.attn.q_proj"
        public var loraAShape: [Int] // [in, r]
        public var loraBShape: [Int] // [r, out]
        public init(name: String, loraAShape: [Int], loraBShape: [Int]) {
            self.name = name; self.loraAShape = loraAShape; self.loraBShape = loraBShape
        }
    }
    public struct Header: Codable, Sendable {
        public var rank: Int
        public var alpha: Float
        public var targetSuffixes: [String]
        public var baseLayers: Int
        public var baseDModel: Int
        public var baseCtx: Int
        public var baseHeads: Int
        public var baseDMlp: Int
        public var entries: [Entry]
        public var savedAt: String?
        public var finalLoss: Float?
    }

    public var header: Header
    public var matrices: [(loraA: [Float], loraB: [Float])]  // one pair per entry, in header.entries order
}

public enum LoraAdapterWriter {
    public static func write(model: TinyGPTModel, baseConfig: ModelConfig,
                              loraConfig: LoraConfig, finalLoss: Float?,
                              to url: URL) throws {
        var entries: [LoraAdapter.Entry] = []
        var matrices: [(loraA: [Float], loraB: [Float])] = []

        let projections: [(String, KeyPath<TransformerBlock, Linear>)] = [
            ("attn.q_proj", \.attn.qProj),
            ("attn.k_proj", \.attn.kProj),
            ("attn.v_proj", \.attn.vProj),
            ("attn.o_proj", \.attn.oProj),
            ("mlp.fc_in",   \.mlp.fcIn),
            ("mlp.fc_out",  \.mlp.fcOut),
        ]
        for (i, block) in model.blocks.enumerated() {
            for (suffix, kp) in projections {
                guard let lora = block[keyPath: kp] as? LoraLinear else { continue }
                eval(lora.loraA, lora.loraB)
                let aFloats = lora.loraA.asArray(Float.self)
                let bFloats = lora.loraB.asArray(Float.self)
                entries.append(.init(
                    name: "blocks.\(i).\(suffix)",
                    loraAShape: lora.loraA.shape,
                    loraBShape: lora.loraB.shape
                ))
                matrices.append((loraA: aFloats, loraB: bFloats))
            }
        }
        let header = LoraAdapter.Header(
            rank: loraConfig.rank, alpha: loraConfig.alpha,
            targetSuffixes: loraConfig.targetSuffixes,
            baseLayers: baseConfig.nLayers, baseDModel: baseConfig.dModel,
            baseCtx: baseConfig.contextLength, baseHeads: baseConfig.nHeads,
            baseDMlp: baseConfig.dMlp,
            entries: entries,
            savedAt: ISO8601DateFormatter().string(from: Date()),
            finalLoss: finalLoss
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let headerData = try encoder.encode(header)
        var out = Data()
        out.append(contentsOf: LoraAdapterFormat.magic)
        appendU32(&out, LoraAdapterFormat.currentVersion)
        appendU32(&out, UInt32(headerData.count))
        out.append(headerData)
        for (a, b) in matrices {
            a.withUnsafeBufferPointer { out.append(Data(buffer: $0)) }
            b.withUnsafeBufferPointer { out.append(Data(buffer: $0)) }
        }
        try out.write(to: url, options: .atomic)
    }

    private static func appendU32(_ data: inout Data, _ value: UInt32) {
        var v = value.littleEndian
        withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
    }
}

public enum LoraAdapterReader {
    public static func read(_ url: URL) throws -> LoraAdapter {
        let data = try Data(contentsOf: url)
        guard data.count >= 12 else {
            throw NSError(domain: "TinyGPTLoRA", code: 1, userInfo: [NSLocalizedDescriptionKey: "file too small"])
        }
        let magicBytes = Array(data[0..<4])
        guard magicBytes == LoraAdapterFormat.magic else {
            throw NSError(domain: "TinyGPTLoRA", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "bad magic, expected 'TGLA'"])
        }
        let version = data[4..<8].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian }
        guard version == LoraAdapterFormat.currentVersion else {
            throw NSError(domain: "TinyGPTLoRA", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "unsupported version \(version)"])
        }
        let headerLen = Int(data[8..<12].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian })
        let header = try JSONDecoder().decode(LoraAdapter.Header.self,
                                              from: data.subdata(in: 12..<(12 + headerLen)))
        var cursor = 12 + headerLen
        var matrices: [(loraA: [Float], loraB: [Float])] = []
        for entry in header.entries {
            let aSize = entry.loraAShape.reduce(1, *) * 4
            let bSize = entry.loraBShape.reduce(1, *) * 4
            let aData = data.subdata(in: cursor..<(cursor + aSize)); cursor += aSize
            let bData = data.subdata(in: cursor..<(cursor + bSize)); cursor += bSize
            let aFloats = aData.withUnsafeBytes { Array(UnsafeBufferPointer<Float>(
                start: $0.baseAddress?.assumingMemoryBound(to: Float.self),
                count: aSize / 4)) }
            let bFloats = bData.withUnsafeBytes { Array(UnsafeBufferPointer<Float>(
                start: $0.baseAddress?.assumingMemoryBound(to: Float.self),
                count: bSize / 4)) }
            matrices.append((loraA: aFloats, loraB: bFloats))
        }
        return LoraAdapter(header: header, matrices: matrices)
    }

    /// Inject and load a saved adapter onto a model. Throws if the
    /// adapter's recorded base config doesn't match the model.
    public static func apply(_ adapter: LoraAdapter, to model: TinyGPTModel) throws {
        let h = adapter.header
        let cfg = model.config
        guard h.baseLayers == cfg.nLayers,
              h.baseDModel == cfg.dModel,
              h.baseCtx == cfg.contextLength,
              h.baseHeads == cfg.nHeads,
              h.baseDMlp == cfg.dMlp else {
            throw NSError(domain: "TinyGPTLoRA", code: 4,
                          userInfo: [NSLocalizedDescriptionKey:
                            "adapter base config doesn't match loaded model"])
        }
        let loraCfg = LoraConfig(rank: h.rank, alpha: h.alpha, targetSuffixes: h.targetSuffixes)
        LoraInjection.inject(model, config: loraCfg)
        // Now overwrite each LoraLinear's A, B with the saved values.
        // Build a NestedDictionary update of just the LoRA params.
        var blocksList: [NestedItem<String, MLXArray>] = []
        var idx = 0
        for (i, block) in model.blocks.enumerated() {
            _ = i
            var attn: [String: NestedItem<String, MLXArray>] = [:]
            var mlp: [String: NestedItem<String, MLXArray>] = [:]
            let projs: [(String, Linear, Bool)] = [
                ("q_proj", block.attn.qProj, h.targetSuffixes.contains("q_proj")),
                ("k_proj", block.attn.kProj, h.targetSuffixes.contains("k_proj")),
                ("v_proj", block.attn.vProj, h.targetSuffixes.contains("v_proj")),
                ("o_proj", block.attn.oProj, h.targetSuffixes.contains("o_proj")),
            ]
            for (name, _, isTarget) in projs where isTarget {
                let entry = adapter.matrices[idx]
                let aShape = h.entries[idx].loraAShape
                let bShape = h.entries[idx].loraBShape
                attn[name] = .dictionary([
                    "loraA": .value(MLXArray(entry.loraA, aShape)),
                    "loraB": .value(MLXArray(entry.loraB, bShape)),
                ])
                idx += 1
            }
            let mProjs: [(String, Linear, Bool)] = [
                ("fc_in",  block.mlp.fcIn,  h.targetSuffixes.contains("fc_in")),
                ("fc_out", block.mlp.fcOut, h.targetSuffixes.contains("fc_out")),
            ]
            for (name, _, isTarget) in mProjs where isTarget {
                let entry = adapter.matrices[idx]
                let aShape = h.entries[idx].loraAShape
                let bShape = h.entries[idx].loraBShape
                mlp[name] = .dictionary([
                    "loraA": .value(MLXArray(entry.loraA, aShape)),
                    "loraB": .value(MLXArray(entry.loraB, bShape)),
                ])
                idx += 1
            }
            var blockEntries: [String: NestedItem<String, MLXArray>] = [:]
            if !attn.isEmpty { blockEntries["attn"] = .dictionary(attn) }
            if !mlp.isEmpty { blockEntries["mlp"] = .dictionary(mlp) }
            blocksList.append(.dictionary(blockEntries))
        }
        var root = NestedDictionary<String, MLXArray>()
        root["blocks"] = .array(blocksList)
        try model.update(parameters: root, verify: [])
    }
}
