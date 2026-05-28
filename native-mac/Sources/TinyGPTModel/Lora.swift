import Foundation
import MLX
import MLXNN
import MLXRandom

/// LoRA (Low-Rank Adaptation): a Linear subclass that adds a low-rank
/// delta on top of the frozen base weight.
///
///     y = base_linear(x) + (x @ A) @ B * (alpha / r)
///
///   - `weight` and `bias` (inherited from Linear): FROZEN at fine-tune time
///   - `loraA: [in, r]` trainable, gaussian-init std=0.02
///   - `loraB: [r, out]` trainable, ZERO-init
///   - `rank r`: typically 4-16
///   - `alpha`: scaling, typically 2× r
///
/// B starts at zero → initial output exactly equals base output. Training
/// is purely additive — never destructive.
///
/// Param math: Huge has ~9.6M base params; LoRA-on-QV at r=4 across 12
/// blocks adds 12 × 2 × (256·4 + 4·256) = 49 152 trainable params (~200×
/// fewer). Training fits in minutes; adapter files are 100KB-1MB instead
/// of tens of MB, so multiple "voices" (legal text, code style, lyrics
/// register) share the same base cheaply.
public final class LoraLinear: Linear {
    public let loraA: MLXArray  // [in, r]
    public let loraB: MLXArray  // [r, out]
    public let rank: Int
    public let alpha: Float
    public var scale: Float { alpha / Float(rank) }

    /// Wrap an existing Linear with LoRA adapters. The base weight + bias
    /// are reused verbatim (no copy); the adapter matrices are fresh.
    public init(wrapping base: Linear, rank: Int = 4, alpha: Float = 8.0) {
        precondition(rank > 0, "LoRA rank must be > 0")
        self.rank = rank
        self.alpha = alpha
        let outFeatures = base.weight.shape[0]
        let inFeatures = base.weight.shape[1]
        self.loraA = MLXRandom.normal([inFeatures, rank], scale: 0.02)
        self.loraB = MLXArray.zeros([rank, outFeatures])
        super.init(weight: base.weight, bias: base.bias)
    }

    /// Build with explicit A, B (used by the adapter-loader to restore
    /// a saved fine-tune).
    public init(wrapping base: Linear, loraA: MLXArray, loraB: MLXArray,
                rank: Int, alpha: Float) {
        self.rank = rank
        self.alpha = alpha
        self.loraA = loraA
        self.loraB = loraB
        super.init(weight: base.weight, bias: base.bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // base(x): inherited Linear forward — x @ weight.T + bias
        // delta:   x @ loraA @ loraB * scale
        let baseOut = super.callAsFunction(x)
        let delta = matmul(matmul(x, loraA), loraB) * MLXArray(scale)
        return baseOut + delta
    }
}

/// LoRA wiring config. The default ("QV at rank 4, alpha 8") is the
/// recipe from the original LoRA paper that captures most of the
/// fine-tuning benefit on small adapter sizes.
public struct LoraConfig: Sendable {
    public var rank: Int
    public var alpha: Float
    /// Suffixes of fully-qualified parameter names to wrap (matched at
    /// the Linear leaf level — "q_proj", "fc_in", etc.).
    public var targetSuffixes: [String]

    public init(rank: Int = 4, alpha: Float = 8.0,
                targetSuffixes: [String] = ["q_proj", "v_proj"]) {
        self.rank = rank
        self.alpha = alpha
        self.targetSuffixes = targetSuffixes
    }

    /// Conservative: just QV. Smaller adapter, faster training.
    public static let qv = LoraConfig(rank: 4, alpha: 8.0,
                                       targetSuffixes: ["q_proj", "v_proj"])
    /// More expressive: all attention projections.
    public static let attention = LoraConfig(rank: 8, alpha: 16.0,
                                              targetSuffixes: ["q_proj", "k_proj", "v_proj", "o_proj"])
    /// Maximum: every Linear in the network.
    public static let full = LoraConfig(rank: 8, alpha: 16.0,
                                         targetSuffixes: ["q_proj", "k_proj", "v_proj", "o_proj",
                                                           "fc_in", "fc_out"])
}

/// Inject LoRA adapters into a TinyGPTModel via `Module.update(modules:)`.
/// The @ModuleInfo storage for q/k/v/o/fc_in/fc_out is private, so we
/// build a ModuleChildren nested dict naming the targets and ask the
/// framework to swap them in.
public enum LoraInjection {
    @discardableResult
    public static func inject(_ model: TinyGPTModel, config: LoraConfig) -> TinyGPTModel {
        let suffixes = Set(config.targetSuffixes)

        // Build the replacement tree:
        //   blocks: [
        //     0: { attn: { q_proj: LoraLinear, v_proj: LoraLinear }, mlp: { ... } },
        //     1: { ... },
        //     ...
        //   ]
        var blocksList: [NestedItem<String, Module>] = []
        for block in model.blocks {
            var attnEntries: [String: NestedItem<String, Module>] = [:]
            var mlpEntries: [String: NestedItem<String, Module>] = [:]
            if suffixes.contains("q_proj") {
                attnEntries["q_proj"] = .value(LoraLinear(wrapping: block.attn.qProj,
                                                            rank: config.rank, alpha: config.alpha))
            }
            if suffixes.contains("k_proj") {
                attnEntries["k_proj"] = .value(LoraLinear(wrapping: block.attn.kProj,
                                                            rank: config.rank, alpha: config.alpha))
            }
            if suffixes.contains("v_proj") {
                attnEntries["v_proj"] = .value(LoraLinear(wrapping: block.attn.vProj,
                                                            rank: config.rank, alpha: config.alpha))
            }
            if suffixes.contains("o_proj") {
                attnEntries["o_proj"] = .value(LoraLinear(wrapping: block.attn.oProj,
                                                            rank: config.rank, alpha: config.alpha))
            }
            if suffixes.contains("fc_in") {
                mlpEntries["fc_in"] = .value(LoraLinear(wrapping: block.mlp.fcIn,
                                                          rank: config.rank, alpha: config.alpha))
            }
            if suffixes.contains("fc_out") {
                mlpEntries["fc_out"] = .value(LoraLinear(wrapping: block.mlp.fcOut,
                                                           rank: config.rank, alpha: config.alpha))
            }
            var blockChildren: [String: NestedItem<String, Module>] = [:]
            if !attnEntries.isEmpty { blockChildren["attn"] = .dictionary(attnEntries) }
            if !mlpEntries.isEmpty { blockChildren["mlp"] = .dictionary(mlpEntries) }
            blocksList.append(.dictionary(blockChildren))
        }
        var root = NestedDictionary<String, Module>()
        root["blocks"] = .array(blocksList)
        model.update(modules: root)
        return model
    }

    /// Count the trainable parameters once LoRA is injected.
    public static func trainableParamCount(in model: TinyGPTModel) -> Int {
        var n = 0
        for block in model.blocks {
            for layer in [block.attn.qProj, block.attn.kProj, block.attn.vProj, block.attn.oProj,
                          block.mlp.fcIn, block.mlp.fcOut] {
                if let lora = layer as? LoraLinear {
                    n += lora.loraA.shape.reduce(1, *) + lora.loraB.shape.reduce(1, *)
                }
            }
        }
        return n
    }

    /// Freeze base weights; only LoRA's A, B receive gradients.
    public static func freezeBase(_ model: TinyGPTModel) {
        model.freeze(recursive: true)
        for block in model.blocks {
            for layer in [block.attn.qProj, block.attn.kProj,
                          block.attn.vProj, block.attn.oProj,
                          block.mlp.fcIn, block.mlp.fcOut] {
                if let lora = layer as? LoraLinear {
                    lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
                }
            }
        }
    }
}
