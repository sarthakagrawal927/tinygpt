import Foundation

/// Curated catalog of HF datasets, grouped by the specialist they train.
///
/// This is the agent-factory data moat: rather than ask users to wade
/// through 100k+ datasets on HF Hub, we maintain a vetted shortlist for
/// each capability (tool-calling, code, math, reasoning, instruct,
/// preference, debugging). Entries here have been hand-picked for size,
/// license, schema cleanliness, and downstream signal.
///
/// API:
///   - DatasetRegistry.all                — every curated entry
///   - DatasetRegistry.entries(for: ...)  — filter by specialist
///   - `tinygpt list-datasets [--specialist <k>]`  → uses these
///
/// When you add a new entry, fill in:
///   - id: canonical HF id ("owner/name")
///   - specialists: one or more capabilities it trains
///   - format: the natural target (sft|dpo|plain)
///   - approxSize: rough total bytes (for sanity-check / cache warnings)
///   - notes: license, gated?, schema oddities
public enum DatasetSpecialist: String, Sendable, CaseIterable {
    case toolCalling   = "tool-calling"
    case debugger      = "debugger"
    case code          = "code"
    case math          = "math"
    case reasoning     = "reasoning"
    case instruct      = "instruct"
    case preference    = "preference"
    case general       = "general"

    public init?(parsing s: String) {
        // Accept "tool-calling", "tools", "tool_call" etc.
        let norm = s.lowercased().replacingOccurrences(of: "_", with: "-")
        if let m = Self(rawValue: norm) { self = m; return }
        switch norm {
        case "tools", "tool", "function-calling", "functions": self = .toolCalling
        case "debug", "bugs", "swe": self = .debugger
        case "coding", "program": self = .code
        case "maths", "arithmetic", "reason-math": self = .math
        case "think", "thinking", "cot": self = .reasoning
        case "sft", "instruction": self = .instruct
        case "dpo", "rlhf", "simpo": self = .preference
        default: return nil
        }
    }
}

public struct RegistryEntry: Sendable {
    public let id: String                    // HF "owner/name"
    public let specialists: [DatasetSpecialist]
    public let format: CorpusFormat
    public let approxSize: String            // human-readable; rough estimate
    public let license: String
    public let gated: Bool
    public let notes: String
    public init(id: String, specialists: [DatasetSpecialist], format: CorpusFormat,
                approxSize: String, license: String, gated: Bool = false, notes: String) {
        self.id = id; self.specialists = specialists; self.format = format
        self.approxSize = approxSize; self.license = license; self.gated = gated; self.notes = notes
    }
}

public enum DatasetRegistry {

    /// The curated catalog. Hand-maintained; small enough to inline.
    /// Bias is "what would a tinygpt user actually want to train on?"
    /// — every entry should have a clear hypothesis for what it improves.
    public static let all: [RegistryEntry] = [
        // ── Tool / function calling ─────────────────────────────────
        RegistryEntry(
            id: "Salesforce/xlam-function-calling-60k",
            specialists: [.toolCalling],
            format: .sft,
            approxSize: "~80 MB",
            license: "CC BY-NC 4.0",
            notes: "60k single-turn function-calling examples. Schema: query + tools (JSON) + answers (JSON tool calls). Adapt: serialize tools+answer into response."
        ),
        RegistryEntry(
            id: "NousResearch/hermes-function-calling-v1",
            specialists: [.toolCalling, .instruct],
            format: .sft,
            approxSize: "~200 MB",
            license: "apache-2.0",
            notes: "ShareGPT-style chat array with tool_calls in assistant turns. Multi-turn."
        ),
        RegistryEntry(
            id: "Locutusque/function-calling-chatml",
            specialists: [.toolCalling],
            format: .sft,
            approxSize: "~60 MB",
            license: "apache-2.0",
            notes: "ChatML-formatted function-calling traces. Already templated."
        ),

        // ── Debugger / SWE ───────────────────────────────────────────
        RegistryEntry(
            id: "princeton-nlp/SWE-bench_Verified",
            specialists: [.debugger],
            format: .plain,
            approxSize: "~50 MB",
            license: "MIT",
            notes: "EVAL ONLY. 500 verified GitHub issue/patch pairs. Used for benchmark, not training. Run via `lm-evaluation-harness`."
        ),
        RegistryEntry(
            id: "princeton-nlp/SWE-bench",
            specialists: [.debugger],
            format: .sft,
            approxSize: "~3 GB",
            license: "MIT",
            notes: "Full SWE-bench. issue → patch pairs. Use 'problem_statement' as instruction, 'patch' as response."
        ),
        RegistryEntry(
            id: "bigcode/commitpack",
            specialists: [.debugger, .code],
            format: .sft,
            approxSize: "~4 TB (subset recommended)",
            license: "MIT",
            notes: "Git commits across 4TB of code. Use small language subset (e.g. python/javascript). instruction=commit message, response=diff."
        ),

        // ── Code ─────────────────────────────────────────────────────
        RegistryEntry(
            id: "bigcode/the-stack-smol",
            specialists: [.code],
            format: .plain,
            approxSize: "~250 MB",
            license: "Other (per-file)",
            notes: "Smol subset of The Stack. Plain code for pretraining. License: per-file (mostly permissive)."
        ),
        RegistryEntry(
            id: "open-r1/codeforces-cots",
            specialists: [.code, .reasoning],
            format: .sft,
            approxSize: "~1.5 GB",
            license: "apache-2.0",
            notes: "Codeforces problems with chain-of-thought reasoning. instruction=problem, response=CoT+solution."
        ),
        RegistryEntry(
            id: "iamtarun/python_code_instructions_18k_alpaca",
            specialists: [.code, .instruct],
            format: .sft,
            approxSize: "~12 MB",
            license: "apache-2.0",
            notes: "Alpaca-format python instruction-following. Small + clean — good smoke-test target."
        ),

        // ── Math ─────────────────────────────────────────────────────
        RegistryEntry(
            id: "nvidia/OpenMathReasoning",
            specialists: [.math, .reasoning],
            format: .sft,
            approxSize: "~1 GB",
            license: "CC BY 4.0",
            notes: "Multi-step math reasoning traces from Nvidia. problem + generated_solution."
        ),
        RegistryEntry(
            id: "AI-MO/NuminaMath-CoT",
            specialists: [.math, .reasoning],
            format: .sft,
            approxSize: "~800 MB",
            license: "apache-2.0",
            notes: "Numina's CoT math dataset. problem + solution with chain-of-thought."
        ),
        RegistryEntry(
            id: "meta-math/MetaMathQA",
            specialists: [.math],
            format: .sft,
            approxSize: "~200 MB",
            license: "MIT",
            notes: "MetaMath augmented GSM8K + MATH. query + response. Standard SFT shape."
        ),

        // ── Reasoning ─────────────────────────────────────────────────
        RegistryEntry(
            id: "open-thoughts/OpenThoughts-114k",
            specialists: [.reasoning],
            format: .sft,
            approxSize: "~3 GB",
            license: "apache-2.0",
            notes: "114k chain-of-thought reasoning traces across math/code/science. ShareGPT format (conversations)."
        ),
        RegistryEntry(
            id: "open-thoughts/OpenThoughts2-1M",
            specialists: [.reasoning],
            format: .sft,
            approxSize: "~30 GB",
            license: "apache-2.0",
            notes: "Million-row reasoning dataset. Hefty. Stream parquet — don't full-download."
        ),

        // ── Instruct ─────────────────────────────────────────────────
        RegistryEntry(
            id: "teknium/OpenHermes-2.5",
            specialists: [.instruct],
            format: .sft,
            approxSize: "~1.6 GB",
            license: "Other",
            notes: "1M instruction-following examples (cleaned subset of GPT-4 + Claude generations). ShareGPT chat array."
        ),
        RegistryEntry(
            id: "HuggingFaceH4/ultrachat_200k",
            specialists: [.instruct],
            format: .sft,
            approxSize: "~1.2 GB",
            license: "MIT",
            notes: "200k multi-turn cleanly-filtered UltraChat conversations. Used to fine-tune Zephyr."
        ),
        RegistryEntry(
            id: "yahma/alpaca-cleaned",
            specialists: [.instruct],
            format: .sft,
            approxSize: "~25 MB",
            license: "CC BY-NC 4.0",
            notes: "Alpaca instruction-following with major errors removed. Tiny — perfect for smoke tests."
        ),

        // ── Preference ───────────────────────────────────────────────
        RegistryEntry(
            id: "argilla/ultrafeedback-binarized-preferences-cleaned",
            specialists: [.preference],
            format: .dpo,
            approxSize: "~200 MB",
            license: "MIT",
            notes: "Cleaned UltraFeedback binarized preferences. {prompt, chosen[chat], rejected[chat]}."
        ),
        RegistryEntry(
            id: "HuggingFaceH4/ultrafeedback_binarized",
            specialists: [.preference],
            format: .dpo,
            approxSize: "~250 MB",
            license: "MIT",
            notes: "Original H4 UltraFeedback binarized. Used to train Zephyr-DPO."
        ),
        RegistryEntry(
            id: "Intel/orca_dpo_pairs",
            specialists: [.preference],
            format: .dpo,
            approxSize: "~50 MB",
            license: "apache-2.0",
            notes: "12k DPO pairs. {system, question, chosen, rejected}."
        ),

        // ── General / pretraining ────────────────────────────────────
        RegistryEntry(
            id: "roneneldan/TinyStories",
            specialists: [.general],
            format: .plain,
            approxSize: "~1 GB",
            license: "CDLA-Sharing-1.0",
            notes: "Synthetic children's stories for nano-LM pretraining. 'text' field."
        ),
        RegistryEntry(
            id: "HuggingFaceFW/fineweb-edu",
            specialists: [.general],
            format: .plain,
            approxSize: "~1.3 TB (use sample)",
            license: "ODC-BY",
            notes: "Educational subset of FineWeb. ENORMOUS — pick a config like 'sample-10BT' or stream."
        ),
    ]

    public static func entries(for specialist: DatasetSpecialist?) -> [RegistryEntry] {
        guard let spec = specialist else { return all }
        return all.filter { $0.specialists.contains(spec) }
    }

    public static func entry(id: String) -> RegistryEntry? {
        all.first(where: { $0.id == id })
    }
}

/// Curated GitHub repositories that produce high-signal training data
/// for a code-specialist agent (debugger / reviewer / commit-msg). HF
/// datasets are pre-curated and license-clean; GitHub data is raw and
/// per-repo licensed — we still recommend a hand-picked shortlist so
/// users don't have to guess which repos have well-labelled bugs,
/// review-heavy PRs, etc.
///
/// API:
///   - GitHubRecipes.all                 — every recipe
///   - GitHubRecipes.entries(for: .debugger)  — filter by specialist
///   - `tinygpt list-datasets --specialist debugger` includes these.
///
/// All recipes are public repos. Fetching with `tinygpt fetch-github`
/// honours their underlying source license — record metadata carries
/// `repo` and `kind` so downstream license attribution stays attached.
public struct GitHubRecipe: Sendable {
    public let repo: String                  // "owner/name"
    public let specialists: [DatasetSpecialist]
    public let language: String
    public let approxBugIssues: String       // human-readable rough count
    public let recommendedKinds: [String]    // "issues-prs" | "reviews" | "commits"
    public let license: String
    public let notes: String
    public init(repo: String, specialists: [DatasetSpecialist], language: String,
                approxBugIssues: String, recommendedKinds: [String],
                license: String, notes: String) {
        self.repo = repo; self.specialists = specialists; self.language = language
        self.approxBugIssues = approxBugIssues; self.recommendedKinds = recommendedKinds
        self.license = license; self.notes = notes
    }
}

public enum GitHubRecipes {

    /// Hand-picked good-bug-fix repos. Bias: well-labelled bug
    /// trackers, PRs that close issues with a "Fixes #" pattern (so
    /// issue→PR linkage is high-recall), and permissive licenses.
    public static let all: [GitHubRecipe] = [
        // ── Python ML / DL ────────────────────────────────────────
        GitHubRecipe(
            repo: "pytorch/pytorch",
            specialists: [.debugger, .code],
            language: "python/c++",
            approxBugIssues: "~10k closed bug issues",
            recommendedKinds: ["issues-prs", "commits"],
            license: "BSD-3-Clause",
            notes: "Excellent bug labels. Long PR descriptions, often with reproducer + fix."
        ),
        GitHubRecipe(
            repo: "huggingface/transformers",
            specialists: [.debugger, .code],
            language: "python",
            approxBugIssues: "~6k closed bug issues",
            recommendedKinds: ["issues-prs", "reviews", "commits"],
            license: "Apache-2.0",
            notes: "Heavy review culture — strong PR-review training signal."
        ),
        GitHubRecipe(
            repo: "tensorflow/tensorflow",
            specialists: [.debugger, .code],
            language: "python/c++",
            approxBugIssues: "~15k closed bug issues",
            recommendedKinds: ["issues-prs", "commits"],
            license: "Apache-2.0",
            notes: "Enormous corpus; cap with --limit / --since."
        ),
        GitHubRecipe(
            repo: "numpy/numpy",
            specialists: [.debugger, .code],
            language: "python/c",
            approxBugIssues: "~3k closed bug issues",
            recommendedKinds: ["issues-prs", "commits"],
            license: "BSD-3-Clause",
            notes: "Stable corpus, careful PRs. Good for numeric-fix learning."
        ),
        GitHubRecipe(
            repo: "scikit-learn/scikit-learn",
            specialists: [.debugger, .code],
            language: "python",
            approxBugIssues: "~2k closed bug issues",
            recommendedKinds: ["issues-prs"],
            license: "BSD-3-Clause",
            notes: "Excellent reviewer culture. Used as one of SWE-bench's source repos."
        ),

        // ── Rust ──────────────────────────────────────────────────
        GitHubRecipe(
            repo: "rust-lang/rust",
            specialists: [.debugger, .code],
            language: "rust",
            approxBugIssues: "~20k closed bug issues",
            recommendedKinds: ["issues-prs", "reviews", "commits"],
            license: "MIT/Apache-2.0",
            notes: "Massive corpus. PR descriptions include 'Fixes #N' consistently."
        ),
        GitHubRecipe(
            repo: "tokio-rs/tokio",
            specialists: [.debugger, .code],
            language: "rust",
            approxBugIssues: "~500 closed bug issues",
            recommendedKinds: ["issues-prs", "commits"],
            license: "MIT",
            notes: "Async runtime; subtle concurrency bugs and detailed fix discussion."
        ),

        // ── Python web / general ──────────────────────────────────
        GitHubRecipe(
            repo: "django/django",
            specialists: [.debugger, .code],
            language: "python",
            approxBugIssues: "~5k closed bug issues",
            recommendedKinds: ["issues-prs", "commits"],
            license: "BSD-3-Clause",
            notes: "Used by SWE-bench. Mature project, clean PR descriptions."
        ),
        GitHubRecipe(
            repo: "pallets/flask",
            specialists: [.debugger, .code],
            language: "python",
            approxBugIssues: "~400 closed bug issues",
            recommendedKinds: ["issues-prs"],
            license: "BSD-3-Clause",
            notes: "Small enough to fetch fully without much pagination."
        ),
        GitHubRecipe(
            repo: "pandas-dev/pandas",
            specialists: [.debugger, .code],
            language: "python",
            approxBugIssues: "~8k closed bug issues",
            recommendedKinds: ["issues-prs", "commits"],
            license: "BSD-3-Clause",
            notes: "Wide range of bug types: indexing, dtype, IO. Used in SWE-bench."
        ),

        // ── Go / Node / JS ────────────────────────────────────────
        GitHubRecipe(
            repo: "golang/go",
            specialists: [.debugger, .code],
            language: "go",
            approxBugIssues: "~15k closed bug issues",
            recommendedKinds: ["issues-prs", "commits"],
            license: "BSD-3-Clause",
            notes: "Use commit log: Go's PR workflow is non-standard (Gerrit), but commits are clean."
        ),
        GitHubRecipe(
            repo: "nodejs/node",
            specialists: [.debugger, .code],
            language: "javascript/c++",
            approxBugIssues: "~5k closed bug issues",
            recommendedKinds: ["issues-prs", "reviews"],
            license: "MIT",
            notes: "Strong review culture — high-quality review-comment dataset."
        ),
        GitHubRecipe(
            repo: "vuejs/core",
            specialists: [.debugger, .code],
            language: "typescript",
            approxBugIssues: "~1k closed bug issues",
            recommendedKinds: ["issues-prs"],
            license: "MIT",
            notes: "Focused frontend bug-fix examples."
        ),
    ]

    public static func entries(for specialist: DatasetSpecialist?) -> [GitHubRecipe] {
        guard let spec = specialist else { return all }
        return all.filter { $0.specialists.contains(spec) }
    }

    public static func entry(repo: String) -> GitHubRecipe? {
        all.first(where: { $0.repo == repo })
    }
}
