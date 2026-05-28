// swift-tools-version: 6.3
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
        .executable(name: "tinygpt", targets: ["TinyGPT"]),
        .executable(name: "TinyGPTApp", targets: ["TinyGPTApp"]),
    ],
    dependencies: [
        // MLX-Swift — Apple ML primitives for Apple Silicon. Pinned to a
        // recent stable; bump the lower bound as the API stabilises.
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.0"),
    ],
    targets: [
        .target(
            name: "TinyGPTIO"
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
            ]
        ),
        .executableTarget(
            name: "TinyGPT",
            dependencies: [
                "TinyGPTIO",
                "TinyGPTModel",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
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
    ],
    swiftLanguageModes: [.v6]
)
