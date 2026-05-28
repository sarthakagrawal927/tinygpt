import Foundation
import TinyGPTIO

/// One row in the sidebar gallery — a `.tinygpt` checkpoint that the user
/// can load and generate from.
struct GalleryItem: Identifiable, Hashable {
    let id: String              // filename stem ("shakespeare")
    let displayName: String     // "Shakespeare"
    let icon: String            // emoji
    let url: URL                // path to the .tinygpt / .bin file
    let prompt: String          // suggested starting text

    static func == (a: GalleryItem, b: GalleryItem) -> Bool { a.url == b.url }
    func hash(into hasher: inout Hasher) { hasher.combine(url) }
}

/// Look in the standard locations the browser playground uses and return the
/// gallery cards we find. Empty list is fine — the app falls back to "open
/// a checkpoint" via NSOpenPanel.
enum GalleryDiscovery {
    /// Browser ships these four canonical gallery slots. Display order +
    /// pretty names + suggested prompts come from this static map (vs. the
    /// browser's `manifest.json` which duplicates the same info — keeping
    /// the Mac side independent so it doesn't break if the browser changes
    /// the manifest shape).
    private static let slots: [(id: String, name: String, icon: String, prompt: String)] = [
        ("shakespeare", "Shakespeare", "🎭", "MENENIUS:\n"),
        ("tinystories", "TinyStories", "📖", "Once upon a time"),
        ("code",        "Python code", "⌨️", "def "),
        ("chat",        "Q&A chat",    "💬", "User: "),
    ]

    static func discover() -> [GalleryItem] {
        var found: [GalleryItem] = []
        let candidates = candidatePaths()
        for (id, name, icon, prompt) in slots {
            // Try `.bin` first (gallery distribution format), then `.tinygpt`.
            for ext in ["bin", "tinygpt"] {
                for base in candidates {
                    let url = base.appendingPathComponent("\(id).\(ext)")
                    if FileManager.default.fileExists(atPath: url.path) {
                        found.append(GalleryItem(
                            id: id, displayName: name, icon: icon,
                            url: url, prompt: prompt
                        ))
                        break
                    }
                }
                if found.last?.id == id { break }
            }
        }
        return found
    }

    /// Where to look. The first matching file wins.
    private static func candidatePaths() -> [URL] {
        var paths: [URL] = []
        let fm = FileManager.default
        // 1. Bundle Resources/gallery (production install location).
        if let resourceURL = Bundle.main.resourceURL {
            paths.append(resourceURL.appendingPathComponent("gallery"))
        }
        // 2. The browser's public/gallery dir relative to the repo root,
        //    discovered by walking up from the executable.
        if let exec = Bundle.main.executableURL {
            var dir = exec.deletingLastPathComponent()
            for _ in 0..<6 {
                let try1 = dir.appendingPathComponent("browser/public/gallery")
                if fm.fileExists(atPath: try1.path) { paths.append(try1); break }
                let try2 = dir.appendingPathComponent("public/gallery")
                if fm.fileExists(atPath: try2.path) { paths.append(try2); break }
                dir = dir.deletingLastPathComponent()
            }
        }
        // 3. ~/Library/Application Support/TinyGPT/gallery — user cache.
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            paths.append(appSupport.appendingPathComponent("TinyGPT/gallery"))
        }
        return paths
    }
}
