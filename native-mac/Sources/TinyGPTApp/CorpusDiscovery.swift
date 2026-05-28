import Foundation

/// A corpus the user can train on with one click — the Project Gutenberg
/// classics fetched by `scripts/fetch_corpora.sh`, plus the gallery
/// originals if they're around.
struct CorpusItem: Identifiable, Hashable {
    let id: String
    let displayName: String
    let icon: String
    let url: URL
    let size: Int

    static func == (a: CorpusItem, b: CorpusItem) -> Bool { a.url == b.url }
    func hash(into hasher: inout Hasher) { hasher.combine(url) }
}

enum CorpusDiscovery {
    /// Curated list of corpus filenames the app knows about — display
    /// name + icon + filename. Discovery scans the candidate paths for
    /// any of these and shows what it finds. Unknown corpora can still
    /// be loaded via the "Load file..." button.
    private static let known: [(file: String, name: String, icon: String)] = [
        ("shakespeare-complete.txt",   "Shakespeare (Complete)",   "🎭"),
        ("war-and-peace.txt",          "War and Peace",             "📚"),
        ("monte-cristo.txt",           "Monte Cristo",              "⚔️"),
        ("don-quixote.txt",            "Don Quixote",               "🐎"),
        ("middlemarch.txt",            "Middlemarch",               "📖"),
        ("moby-dick.txt",              "Moby Dick",                 "🐳"),
        ("pride-prejudice.txt",        "Pride and Prejudice",       "💍"),
        ("huck-finn.txt",              "Huckleberry Finn",          "🚣"),
        ("frankenstein.txt",           "Frankenstein",              "⚡"),
        ("heart-of-darkness.txt",      "Heart of Darkness",         "🌑"),
        ("alice.txt",                  "Alice in Wonderland",       "🐇"),
        ("sherlock-holmes.txt",        "Sherlock Holmes",           "🔍"),
        ("modest-proposal.txt",        "A Modest Proposal",         "✒️"),
        // Browser gallery originals
        ("tinystories.txt",            "TinyStories",               "🧒"),
        ("recipes.txt",                "Recipes",                   "🍳"),
        ("code.txt",                   "Python code",               "⌨️"),
        ("chat.txt",                   "Q&A chat",                  "💬"),
        // Combined themed corpora the user may have built with cat
        ("western-canon.txt",          "Western Canon (combined)",  "🏛️"),
        ("everything.txt",             "Everything (combined)",     "🌍"),
    ]

    static func discover() -> [CorpusItem] {
        var found: [CorpusItem] = []
        let candidates = candidatePaths()
        for (file, name, icon) in known {
            for base in candidates {
                let url = base.appendingPathComponent(file)
                if let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
                   let size = attrs[.size] as? Int, size > 0 {
                    found.append(CorpusItem(
                        id: file, displayName: name, icon: icon,
                        url: url, size: size
                    ))
                    break
                }
            }
        }
        return found.sorted { $0.size > $1.size }  // biggest first
    }

    private static func candidatePaths() -> [URL] {
        var paths: [URL] = []
        let fm = FileManager.default
        // The fetch_corpora.sh default
        paths.append(URL(fileURLWithPath: "/tmp/tinygpt-corpora"))
        // Bundled resources/corpora (production install)
        if let resourceURL = Bundle.main.resourceURL {
            paths.append(resourceURL.appendingPathComponent("corpora"))
        }
        // ~/Library/Application Support/TinyGPT/corpora
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            paths.append(appSupport.appendingPathComponent("TinyGPT/corpora"))
        }
        // Repo-local data/corpora (when launched from .xcode-build during dev)
        if let exec = Bundle.main.executableURL {
            var dir = exec.deletingLastPathComponent()
            for _ in 0..<6 {
                let try1 = dir.appendingPathComponent("data/corpora")
                if fm.fileExists(atPath: try1.path) { paths.append(try1); break }
                dir = dir.deletingLastPathComponent()
            }
        }
        return paths
    }
}
