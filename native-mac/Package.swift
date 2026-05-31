// swift-tools-version: 6.1
//
// TinyGPT — native macOS app.
//
// The package is split into a library target (`TinyGPTIO`) that owns the
// `.tinygpt` file-format reader/writer and an executable (`tinygpt`) that
// exposes a CLI today (`inspect`, `validate`) and will host the SwiftUI app
// once the file-format + model parity milestones land.
//
// MLX-Swift gets wired in once the M2 (forward-pass numerics parity)
// milestone starts; M1 (file format) is pure Foundation + Swift stdlib.

import PackageDescription

let package = Package(
    name: "TinyGPT",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "TinyGPTIO", targets: ["TinyGPTIO"]),
        .library(name: "TinyGPTModel", targets: ["TinyGPTModel"]),
        .library(name: "TinyGPTBench", targets: ["TinyGPTBench"]),
        .library(name: "TinyGPTServe", targets: ["TinyGPTServe"]),
        // TinyGPTData — HuggingFace Datasets Hub client + format
        // converters. Pure Foundation; deliberately depends on nothing
        // else so the CLI `download-dataset` subcommand boots fast and
        // we can unit-test the registry / format detector without
        // pulling MLX. See docs/hf_datasets_integration.md.
        .library(name: "TinyGPTData", targets: ["TinyGPTData"]),
        // TinyGPTScreen — Mac screen-reading scaffold (Wave 2.6).
        // ScreenCaptureKit window capture + macOS Accessibility (AX) tree
        // reader. Pure Foundation + ScreenCaptureKit + ApplicationServices;
        // deliberately holds no model dependencies so it stays linkable
        // from any CLI subcommand (and so the agent can call it before
        // any model is loaded). The vision-encoder/ViT half (consuming
        // PNG → tokens) is intentionally NOT in this target.
        .library(name: "TinyGPTScreen", targets: ["TinyGPTScreen"]),
        .executable(name: "tinygpt", targets: ["TinyGPT"]),
        .executable(name: "TinyGPTApp", targets: ["TinyGPTApp"]),
    ],
    dependencies: [
        // MLX-Swift — Apple ML primitives for Apple Silicon. Pinned to a
        // recent stable; bump the lower bound as the API stabilises.
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.0"),
        // HuggingFace's canonical Swift tokenizer library. Supports BPE,
        // SentencePiece, tiktoken-style; used by mlx-swift-examples and
        // every production Swift LLM project. We use it for the HF model
        // loading path; our from-scratch byte-level path doesn't need it.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "TinyGPTIO"
        ),
        // Pure-Foundation HF Datasets Hub client. Lives outside
        // TinyGPTModel so it does not drag MLX into CLI subcommands
        // that only need to fetch + convert text (download-dataset,
        // list-datasets).
        .target(
            name: "TinyGPTData"
        ),
        // See the library declaration above for rationale. This target is
        // pure Foundation + ScreenCaptureKit + ApplicationServices (linked
        // via the AppKit umbrella through Foundation autolinking) — no MLX,
        // no model code. Build constraint: macOS 14 (already the package
        // platform floor), which has the SCContentFilter(desktopIndependent
        // Window:) initialiser we use.
        .target(
            name: "TinyGPTScreen"
        ),
        .target(
            name: "TinyGPTModel",
            dependencies: [
                "TinyGPTIO",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "TinyGPTBench",
            dependencies: [
                "TinyGPTIO",
                "TinyGPTModel",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        // `TinyGPTServe` exposes an OpenAI-compatible HTTP endpoint over a
        // loaded tinygpt / HF model. This is the adapter that lets
        // `lm-evaluation-harness` (HellaSwag, MMLU-Pro, GSM8K, IFEval, …)
        // evaluate any tinygpt-loaded model. Lives in its own library so the
        // executable target stays a thin CLI shim AND so we can unit-test
        // the server by calling `Serve.start()` directly from XCTest.
        // See docs/lm_eval_integration.md.
        .target(
            name: "TinyGPTServe",
            dependencies: [
                "TinyGPTIO",
                "TinyGPTModel",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "TinyGPT",
            dependencies: [
                "TinyGPTIO",
                "TinyGPTModel",
                "TinyGPTBench",
                "TinyGPTServe",
                "TinyGPTData",
                "TinyGPTScreen",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "TinyGPTApp",
            dependencies: [
                "TinyGPTIO",
                "TinyGPTModel",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "TinyGPTIOTests",
            dependencies: ["TinyGPTIO"]
        ),
        .testTarget(
            name: "TinyGPTModelTests",
            dependencies: ["TinyGPTModel"]
        ),
        // Exercises the OpenAI-compatible HTTP server by calling
        // `Serve.start()` directly (no subprocess, no curl). Same caveat as
        // TinyGPTModelTests — the MLX runtime needs the Metal library, so
        // these tests must be run via `xcodebuild test` (or Xcode UI),
        // NOT `swift test`.
        .testTarget(
            name: "TinyGPTServeTests",
            dependencies: ["TinyGPTServe", "TinyGPTModel"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
