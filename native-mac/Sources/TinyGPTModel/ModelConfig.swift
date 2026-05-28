import Foundation
import MLX

/// Mirror of `python_ref/model.py`'s `ModelConfig` dataclass. Source of truth
/// for shapes; both browser and Mac builds round-trip through this.
public struct ModelConfig: Sendable, Equatable {
    public var modelName: String
    public var vocabSize: Int
    public var contextLength: Int
    public var nLayers: Int
    public var nHeads: Int
    public var dModel: Int
    public var dMlp: Int
    public var dropout: Float
    public var tieEmbeddings: Bool
    /// `"float32"` or `"float16"`. Default is float32 for training parity.
    public var dtype: String

    public var headDim: Int { dModel / nHeads }

    public var mlxDType: DType {
        switch dtype.lowercased() {
        case "float16", "fp16", "half": return .float16
        case "bfloat16", "bf16": return .bfloat16
        default: return .float32
        }
    }

    public init(
        modelName: String = "byte-tinygpt-v0",
        vocabSize: Int = 256,
        contextLength: Int = 128,
        nLayers: Int = 4,
        nHeads: Int = 4,
        dModel: Int = 128,
        dMlp: Int = 512,
        dropout: Float = 0.0,
        tieEmbeddings: Bool = true,
        dtype: String = "float32"
    ) {
        self.modelName = modelName
        self.vocabSize = vocabSize
        self.contextLength = contextLength
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.dModel = dModel
        self.dMlp = dMlp
        self.dropout = dropout
        self.tieEmbeddings = tieEmbeddings
        self.dtype = dtype
        precondition(dModel % nHeads == 0, "d_model must be divisible by n_heads")
    }

    /// Match the browser's "Huge" preset (12L, d=256, ctx=256, 8 heads, dMlp=1024).
    /// This is the gallery model size — the apples-to-apples comparison target.
    public static let huge = ModelConfig(
        modelName: "byte-tinygpt-huge",
        vocabSize: 256,
        contextLength: 256,
        nLayers: 12,
        nHeads: 8,
        dModel: 256,
        dMlp: 1024,
        dropout: 0.0,
        tieEmbeddings: true
    )

    /// Mega preset (24L, d=512, ctx=512). Browser can't run this — Mac can.
    public static let mega = ModelConfig(
        modelName: "byte-tinygpt-mega",
        vocabSize: 256,
        contextLength: 512,
        nLayers: 24,
        nHeads: 8,
        dModel: 512,
        dMlp: 2048,
        dropout: 0.0,
        tieEmbeddings: true
    )
}
