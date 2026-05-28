import Foundation
import MLX
import MLXNN

/// Multi-LoRA composition. A LoraStack wraps multiple sets of A/B
/// matrices (one per loaded adapter) plus a per-adapter weight. The
/// forward pass sums all deltas:
///
///     y = base(x) + Σ_k w_k * (x @ A_k @ B_k * scale_k)
///
/// Practical use:
///   - "Shakespeare-trained base" + adapter_A (Austen voice, weight 0.6)
///     + adapter_B (sci-fi vocab, weight 0.4) → blended output
///   - Hot-swap adapters at runtime without reloading the base
///   - Mix two "personas" with a slider
///
/// Limitation: all stacked adapters must target the SAME Linear modules
/// with compatible ranks (we don't currently support adapter_A targeting
/// q_proj while adapter_B targets only fc_in). Mixing target sets needs
/// a more elaborate per-adapter map; defer until someone asks.
public final class StackedLoraLinear: Linear {
    public struct Slot {
        public var loraA: MLXArray  // [in, r]
        public var loraB: MLXArray  // [r, out]
        public var scale: Float     // alpha / r
        public var weight: Float    // user-facing per-adapter mix weight
    }

    public var slots: [Slot]

    public init(wrapping base: Linear, slots: [Slot]) {
        self.slots = slots
        super.init(weight: base.weight, bias: base.bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = super.callAsFunction(x)
        for slot in slots {
            let delta = matmul(matmul(x, slot.loraA), slot.loraB)
            out = out + delta * MLXArray(slot.scale * slot.weight)
        }
        return out
    }
}

/// Apply N LoRA adapters with optional per-adapter weights. All
/// adapters must share the same target suffixes (otherwise the user
/// gets a clear error explaining which targets diverge).
public enum LoraStackInjection {
    public static func apply(_ adapters: [LoraAdapter], weights: [Float],
                              to model: TinyGPTModel) throws {
        precondition(adapters.count == weights.count, "weights array length must match adapters")
        guard !adapters.isEmpty else { return }

        // Verify target consistency.
        let firstTargets = Set(adapters[0].header.targetSuffixes)
        for (i, a) in adapters.enumerated() {
            if Set(a.header.targetSuffixes) != firstTargets {
                throw NSError(domain: "TinyGPTLoRA", code: 5,
                              userInfo: [NSLocalizedDescriptionKey:
                                "adapter \(i) targets \(a.header.targetSuffixes) which differs from adapter 0's \(adapters[0].header.targetSuffixes)"])
            }
        }

        // Verify base config consistency.
        let cfg = model.config
        for (i, a) in adapters.enumerated() {
            let h = a.header
            guard h.baseLayers == cfg.nLayers,
                  h.baseDModel == cfg.dModel,
                  h.baseCtx == cfg.contextLength,
                  h.baseHeads == cfg.nHeads,
                  h.baseDMlp == cfg.dMlp else {
                throw NSError(domain: "TinyGPTLoRA", code: 4,
                              userInfo: [NSLocalizedDescriptionKey:
                                "adapter \(i) base config doesn't match loaded model"])
            }
        }

        // Build the replacement: for each target Linear in each block,
        // create a StackedLoraLinear with one slot per adapter.
        let targetSuffixes = firstTargets
        var blocksList: [NestedItem<String, Module>] = []
        // Per-adapter index into matrices (incremented per Linear-target encountered)
        var matIdx: [Int] = Array(repeating: 0, count: adapters.count)
        for block in model.blocks {
            var attn: [String: NestedItem<String, Module>] = [:]
            var mlp:  [String: NestedItem<String, Module>] = [:]
            let projs: [(String, Linear, Bool, Bool)] = [
                // (name, linear, isAttn, isTarget)
                ("q_proj", block.attn.qProj, true,  targetSuffixes.contains("q_proj")),
                ("k_proj", block.attn.kProj, true,  targetSuffixes.contains("k_proj")),
                ("v_proj", block.attn.vProj, true,  targetSuffixes.contains("v_proj")),
                ("o_proj", block.attn.oProj, true,  targetSuffixes.contains("o_proj")),
                ("fc_in",  block.mlp.fcIn,   false, targetSuffixes.contains("fc_in")),
                ("fc_out", block.mlp.fcOut,  false, targetSuffixes.contains("fc_out")),
            ]
            for (name, lin, isAttn, isTarget) in projs where isTarget {
                var slots: [StackedLoraLinear.Slot] = []
                for (k, adapter) in adapters.enumerated() {
                    let mi = matIdx[k]
                    let aShape = adapter.header.entries[mi].loraAShape
                    let bShape = adapter.header.entries[mi].loraBShape
                    let entry = adapter.matrices[mi]
                    let scale = adapter.header.alpha / Float(adapter.header.rank)
                    slots.append(.init(
                        loraA: MLXArray(entry.loraA, aShape),
                        loraB: MLXArray(entry.loraB, bShape),
                        scale: scale,
                        weight: weights[k]
                    ))
                    matIdx[k] += 1
                }
                let stacked = StackedLoraLinear(wrapping: lin, slots: slots)
                if isAttn {
                    attn[name] = .value(stacked)
                } else {
                    mlp[name] = .value(stacked)
                }
            }
            var entries: [String: NestedItem<String, Module>] = [:]
            if !attn.isEmpty { entries["attn"] = .dictionary(attn) }
            if !mlp.isEmpty { entries["mlp"] = .dictionary(mlp) }
            blocksList.append(.dictionary(entries))
        }
        var root = NestedDictionary<String, Module>()
        root["blocks"] = .array(blocksList)
        model.update(modules: root)
    }
}
