import SwiftUI

/// Colors and typography matching the browser playground's visual language.
/// The web side uses these as CSS variables; we mirror them here as `Color`
/// extensions so SwiftUI views reach for them by name.
enum Theme {
    /// Mint accent — the "this is alive" color used for the loss curve,
    /// active controls, and the GPU-active dot.
    static let accent = Color(red: 72/255, green: 229/255, blue: 194/255)
    static let accentDim = Color(red: 31/255, green: 111/255, blue: 95/255)
    static let accentGlow = Color(red: 72/255, green: 229/255, blue: 194/255, opacity: 0.20)

    /// Surface colors — three depths matching the browser's `--base`, `--panel`,
    /// `--panel-2`.
    static let base = Color(red: 8/255, green: 9/255, blue: 10/255)
    static let panel = Color(red: 13/255, green: 14/255, blue: 16/255)
    static let panel2 = Color(red: 20/255, green: 21/255, blue: 24/255)
    static let line = Color(red: 29/255, green: 31/255, blue: 35/255)
    static let lineStrong = Color(red: 44/255, green: 47/255, blue: 53/255)

    /// Foreground hierarchy.
    static let fg = Color(red: 231/255, green: 232/255, blue: 234/255)
    static let muted = Color(red: 146/255, green: 150/255, blue: 160/255)
    static let faint = Color(red: 120/255, green: 125/255, blue: 136/255)

    static let warn = Color(red: 245/255, green: 177/255, blue: 74/255)
    static let danger = Color(red: 255/255, green: 104/255, blue: 104/255)
}

extension Font {
    /// Tight monospace for numbers + code samples.
    static let tgMono = Font.system(.body, design: .monospaced)
    /// Display heading.
    static let tgDisplay = Font.system(size: 24, weight: .semibold)
}
