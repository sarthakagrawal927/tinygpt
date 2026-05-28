import Foundation
import XCTest
import MLX
@testable import TinyGPTModel

/// MLX-Swift's SPM build does NOT compile the Metal shader library
/// (`default.metallib`) — Xcode compiles `.metal` files automatically as part
/// of its build system, but `swift build` doesn't run the Metal toolchain.
///
/// The result: running these tests via `swift test` (Command Line) fails with
/// "Failed to load the default metallib." even when we force the CPU stream,
/// because MLX's C runtime tries to init both streams at module load.
///
/// **Workaround**: run these tests inside Xcode (Product → Test) or via
/// `xcodebuild test -scheme TinyGPT`. The compiled metallib ends up in the
/// product's resources directory and the C runtime finds it.
///
/// We keep one trivial test that doesn't touch MLX at runtime so `swift test`
/// still surfaces source-level breakage; the real numerics tests live behind
/// the Xcode-only flag.
final class TinyGPTModelTests: XCTestCase {

    /// Compile-only test — proves `TinyGPTModel.swift` and the public surface
    /// still build without MLX runtime calls. The full numerics suite is in
    /// `TinyGPTModelNumericsTests` (Xcode-only).
    func test_modelConfigConstructs() {
        let cfg = ModelConfig.huge
        XCTAssertEqual(cfg.nLayers, 12)
        XCTAssertEqual(cfg.dModel, 256)
        XCTAssertEqual(cfg.contextLength, 256)
        XCTAssertEqual(cfg.headDim, 32)
    }

    func test_megaConfigIsBiggerThanHuge() {
        XCTAssertGreaterThan(ModelConfig.mega.dModel, ModelConfig.huge.dModel)
        XCTAssertGreaterThan(ModelConfig.mega.contextLength, ModelConfig.huge.contextLength)
        XCTAssertGreaterThan(ModelConfig.mega.nLayers, ModelConfig.huge.nLayers)
    }
}
