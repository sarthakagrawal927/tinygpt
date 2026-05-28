import Foundation
#if canImport(CoreML)
@preconcurrency import CoreML

/// Inference path that routes through Core ML, which dispatches eligible
/// ops to the Apple Neural Engine (16-core, ~38 TOPS on M3+). Sampling
/// throughput is expected to be 3-10× the MLX-Swift Metal path for the
/// same model.
///
/// Usage:
///     let ane = try await TinyGPTANE.load(url: someMLPackageURL)
///     let tokens = try ane.predict(tokens: [82, 79, 77, 69, 79, 58])
///
/// Generating the .mlpackage from a .tinygpt checkpoint is a Python step;
/// see `python_ref/export_to_coreml.py`. The exported package has fixed
/// context length — match it when calling `predict`.
@available(macOS 14.0, *)
public final class TinyGPTANE {
    private let model: MLModel
    public let contextLength: Int

    public init(model: MLModel, contextLength: Int) {
        self.model = model
        self.contextLength = contextLength
    }

    /// Load a compiled .mlpackage / .mlmodelc and configure it for ANE.
    public static func load(url: URL) async throws -> TinyGPTANE {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all  // .all = CPU + GPU + ANE; runtime picks per op
        let compiled: URL
        if url.pathExtension == "mlmodelc" {
            compiled = url
        } else {
            compiled = try await MLModel.compileModel(at: url)
        }
        let model = try await MLModel.load(contentsOf: compiled, configuration: cfg)
        // Pull context length from the input description ("tokens" shape).
        let inputDesc = model.modelDescription.inputDescriptionsByName["tokens"]
        let ctx = inputDesc?.multiArrayConstraint?.shape.last?.intValue ?? 256
        return TinyGPTANE(model: model, contextLength: ctx)
    }

    /// Run one forward pass on a token sequence (left-padded with 0s to
    /// `contextLength`). Returns logits over the last position — `[256]`
    /// next-byte distribution. Caller does sampling.
    public func predict(tokens: [UInt8]) throws -> [Float] {
        // Build a fixed-length input. Truncate from the left if longer.
        let T = contextLength
        var ids = [Int32](repeating: 0, count: T)
        let src = Array(tokens.suffix(T))
        for (i, b) in src.enumerated() {
            ids[T - src.count + i] = Int32(b)
        }
        let arr = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
        for i in 0..<T {
            arr[i] = NSNumber(value: ids[i])
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: ["tokens": arr])
        let out = try model.prediction(from: provider)
        guard let logits = out.featureValue(for: "logits")?.multiArrayValue else {
            throw NSError(domain: "TinyGPTANE", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "no logits output"])
        }
        // logits shape: [1, T, 256]. We want logits[0, T-1, :].
        let V = 256
        let base = (T - 1) * V
        var result = [Float](repeating: 0, count: V)
        for i in 0..<V {
            result[i] = Float(truncating: logits[base + i])
        }
        return result
    }

}
#endif
