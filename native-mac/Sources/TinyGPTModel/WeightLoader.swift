import Foundation
import MLX
import MLXNN
import TinyGPTIO

/// Load a `.tinygpt` file into a `TinyGPTModel`. Bridges the file-format
/// reader (`TinyGPTIO`) and MLX-Swift's module-update API.
///
/// Browser weight names match PyTorch state-dict naming exactly
/// (`blocks.0.attn.q_proj.weight`, `ln_final.bias`, etc.); the MLX-Swift
/// model uses the same names via `@ModuleInfo(key:)`. No remapping needed.
public enum TinyGPTWeightLoader {
    /// Load weights from a `.tinygpt` URL into the given model. The model's
    /// architecture (layers / d_model / heads / etc.) must already match the
    /// file's header `config`. Throws if the config doesn't match — silently
    /// loading mismatched shapes would create a model that runs but produces
    /// garbage.
    public static func load(
        _ url: URL, into model: TinyGPTModel
    ) throws {
        let file = try TinyGPTFileReader.read(url)
        try load(file, into: model)
    }

    public static func load(
        _ file: TinyGPTFile, into model: TinyGPTModel
    ) throws {
        try checkConfigMatches(file: file, model: model)
        var updates: [String: MLXArray] = [:]
        for tensor in file.tensors {
            // The browser dumps raw WASM bytes. WASM stores Linear weights
            // in `[in, out]` order (so its matmul is `y = x @ W` directly).
            // PyTorch and MLX-Swift use `[out, in]` and compute `y = x @ W.T`.
            // The browser's manifest claims PyTorch shape, but the BYTE
            // LAYOUT is WASM's order — read with WASM shape, then transpose
            // to get the matrix that produces correct PyTorch-convention math.
            // (For 1D weights and Embedding weights, no transpose is needed.)
            let array: MLXArray
            if isLinearWeightName(tensor.entry.name) && tensor.entry.shape.count == 2 {
                let wasmShape = [tensor.entry.shape[1], tensor.entry.shape[0]]
                array = arrayFromTensor(tensor, withShape: wasmShape).transposed()
            } else {
                array = arrayFromTensor(tensor, withShape: tensor.entry.shape)
            }
            updates[tensor.entry.name] = array
        }
        // Take the model's existing parameter structure (which knows
        // arrays-vs-dicts correctly), then swap each leaf with our values
        // by matching on dotted-key path.
        let nested = rewriteLeaves(model.parameters(), withFlat: updates)
        try model.update(parameters: nested, verify: [.noUnusedKeys])
    }

    /// Walk an existing ModuleParameters tree and replace each leaf MLXArray
    /// with the value from `flat` keyed by its dotted path. This avoids
    /// reconstructing the array-vs-dict shape from scratch (which got the
    /// numeric-vs-named key distinction wrong on `blocks.0..blocks.11`).
    private static func rewriteLeaves(
        _ params: ModuleParameters, withFlat flat: [String: MLXArray]
    ) -> ModuleParameters {
        var result = NestedDictionary<String, MLXArray>()
        for (key, item) in params {
            result[key] = rewriteItem(item, path: [key], flat: flat)
        }
        return result
    }

    private static func rewriteItem(
        _ item: NestedItem<String, MLXArray>,
        path: [String],
        flat: [String: MLXArray]
    ) -> NestedItem<String, MLXArray> {
        switch item {
        case .none:
            return .none
        case .value:
            let key = path.joined(separator: ".")
            if let v = flat[key] {
                return .value(v)
            }
            return item
        case .array(let elements):
            return .array(
                elements.enumerated().map { (idx, child) in
                    rewriteItem(child, path: path + [String(idx)], flat: flat)
                }
            )
        case .dictionary(let dict):
            var newDict: [String: NestedItem<String, MLXArray>] = [:]
            for (k, child) in dict {
                newDict[k] = rewriteItem(child, path: path + [k], flat: flat)
            }
            return .dictionary(newDict)
        }
    }

    private static func checkConfigMatches(
        file: TinyGPTFile, model: TinyGPTModel
    ) throws {
        let cfg = model.config
        let h = file.header.config
        var mismatches: [String] = []
        if let v = h.layers, v != cfg.nLayers { mismatches.append("layers: file=\(v) model=\(cfg.nLayers)") }
        if let v = h.dModel, v != cfg.dModel { mismatches.append("dModel: file=\(v) model=\(cfg.dModel)") }
        if let v = h.ctx, v != cfg.contextLength { mismatches.append("ctx: file=\(v) model=\(cfg.contextLength)") }
        if let v = h.heads, v != cfg.nHeads { mismatches.append("heads: file=\(v) model=\(cfg.nHeads)") }
        if let v = h.dMlp, v != cfg.dMlp { mismatches.append("dMlp: file=\(v) model=\(cfg.dMlp)") }
        if !mismatches.isEmpty {
            throw WeightLoaderError.configMismatch(mismatches.joined(separator: ", "))
        }
    }

    /// Recognise tensor names that belong to MLX-Swift `Linear` modules
    /// (their `.weight` needs the WASM→PyTorch transpose). Excludes
    /// `Embedding.weight` and `LayerNorm.weight`, which don't.
    private static func isLinearWeightName(_ name: String) -> Bool {
        guard name.hasSuffix(".weight") else { return false }
        // Embedding weights (no transpose):
        if name == "token_embedding.weight" || name == "position_embedding.weight" {
            return false
        }
        // LayerNorm weights (1D, no transpose; skip even though they end in .weight):
        if name.hasSuffix(".ln1.weight") || name.hasSuffix(".ln2.weight")
            || name == "ln_final.weight" {
            return false
        }
        // The remaining `.weight` names — q_proj, k_proj, v_proj, o_proj,
        // fc_in, fc_out, lm_head — are all Linear-module weights.
        return true
    }

    private static func arrayFromTensor(_ tensor: TinyGPTTensor, withShape shape: [Int]) -> MLXArray {
        switch tensor.dtype {
        case .fp32:
            return MLXArray(tensor.weightFloats, shape)
        case .fp16:
            // The on-disk fp16 layout: expand to fp32 on the host. MLX
            // can later cast back to .float16 if the model is fp16-mode.
            return MLXArray(tensor.weightFP16AsFloat32(), shape)
        }
    }

    /// Turn a flat `["blocks.0.attn.q_proj.weight": …, "ln_final.bias": …]`
    /// dictionary into the nested structure MLX-Swift's `Module.update` wants.
    private static func buildNested(_ flat: [String: MLXArray]) -> ModuleParameters {
        var root = NestedDictionary<String, MLXArray>()
        for (key, value) in flat {
            let parts = key.split(separator: ".").map(String.init)
            insert(into: &root, path: parts, value: value)
        }
        return root
    }

    private static func insert(
        into nested: inout NestedDictionary<String, MLXArray>,
        path: [String], value: MLXArray
    ) {
        guard !path.isEmpty else { return }
        if path.count == 1 {
            nested[path[0]] = .value(value)
            return
        }
        let head = path[0]
        var child: NestedItem<String, MLXArray>
        if let existing = nested[head] {
            child = existing
        } else if Int(head) != nil {
            // Numeric index → array child
            child = .array([])
        } else {
            // Named child → dictionary
            child = .dictionary([:])
        }
        insert(child: &child, path: Array(path.dropFirst()), value: value)
        nested[head] = child
    }

    private static func insert(
        child: inout NestedItem<String, MLXArray>,
        path: [String], value: MLXArray
    ) {
        if path.isEmpty {
            child = .value(value)
            return
        }
        switch child {
        case .none, .value:
            // Determine new kind based on whether path[0] is numeric.
            if Int(path[0]) != nil {
                child = .array([])
            } else {
                child = .dictionary([:])
            }
            insert(child: &child, path: path, value: value)
        case .array(var elements):
            guard let idx = Int(path[0]) else {
                fatalError("expected numeric index, got \(path[0])")
            }
            while elements.count <= idx {
                elements.append(.dictionary([:]))
            }
            var item = elements[idx]
            insert(child: &item, path: Array(path.dropFirst()), value: value)
            elements[idx] = item
            child = .array(elements)
        case .dictionary(var dict):
            let head = path[0]
            var sub = dict[head] ?? .dictionary([:])
            insert(child: &sub, path: Array(path.dropFirst()), value: value)
            dict[head] = sub
            child = .dictionary(dict)
        }
    }
}

public enum WeightLoaderError: Error, CustomStringConvertible {
    case configMismatch(String)
    case missingKey(String)
    case shapeMismatch(name: String, expected: [Int], got: [Int])

    public var description: String {
        switch self {
        case .configMismatch(let why):
            return "model config doesn't match file header: \(why)"
        case .missingKey(let key):
            return "checkpoint has no entry for parameter \(key)"
        case .shapeMismatch(let name, let expected, let got):
            return "shape mismatch on \(name): model expects \(expected), file has \(got)"
        }
    }
}
