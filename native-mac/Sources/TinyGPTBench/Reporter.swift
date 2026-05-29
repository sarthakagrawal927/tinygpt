import Foundation

/// Turns the WorkloadController's run results into:
///   1. A canonical JSON object (the source of truth).
///   2. A human-readable markdown table (derived from the JSON).
///
/// JSON schema documented in docs/benchmark_harness_design.md §3.4.
public enum Reporter {
    public struct Provenance {
        public let harnessVersion: String
        public let gitCommit: String?
        public let gitDirty: Bool
        public let macOSBuild: String
        public let hardwareModel: String
        public let physicalRamGB: Double
        public let thermalState: String

        public init() {
            self.harnessVersion = "0.1.0"
            self.gitCommit = Provenance.readSysctl("/usr/bin/env", ["git", "rev-parse", "--short", "HEAD"])
            self.gitDirty = !(Provenance.readSysctl("/usr/bin/env", ["git", "status", "--porcelain"]) ?? "").isEmpty
            self.macOSBuild = Provenance.readSysctl("/usr/bin/sw_vers", ["-buildVersion"]) ?? "unknown"
            self.hardwareModel = Provenance.readSysctl("/usr/sbin/sysctl", ["-n", "hw.model"]) ?? "unknown"
            if let memStr = Provenance.readSysctl("/usr/sbin/sysctl", ["-n", "hw.memsize"]),
               let mem = Double(memStr) {
                self.physicalRamGB = mem / 1_073_741_824.0
            } else {
                self.physicalRamGB = 0
            }
            self.thermalState = Provenance.readSysctl("/usr/bin/pmset", ["-g", "therm"]) ?? "unknown"
        }

        static func readSysctl(_ path: String, _ args: [String]) -> String? {
            let p = Process()
            p.launchPath = path
            p.arguments = args
            let pipe = Pipe()
            p.standardOutput = pipe
            p.standardError = Pipe()
            do {
                try p.run()
                p.waitUntilExit()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                let s = String(data: data, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if p.terminationStatus == 0, let s = s, !s.isEmpty { return s }
            } catch {
                return nil
            }
            return nil
        }
    }

    /// All inputs to a benchmark report.
    public struct Report {
        public let engineName: String
        public let engineCommit: String?
        public let modelPath: String
        public let modelParams: Int
        public let workload: WorkloadController.Config
        public let provenance: Provenance
        public let runs: [WorkloadController.RunResult]
    }

    /// Emit JSON. Returns a JSON string suitable for `--output <file>`.
    public static func toJSON(_ report: Report) throws -> String {
        let runs = report.runs.map { r -> [String: Any] in
            [
                "run_index": r.runIndex,
                "warm": r.warm,
                "ttft_ms": r.metrics.ttftMs,
                "prefill_tokens": r.metrics.prefillTokens,
                "prefill_tps": r.metrics.prefillTokensPerSec,
                "decode_tokens": r.metrics.decodeTokens,
                "decode_ms": r.metrics.decodeMs,
                "decode_tps": r.metrics.decodeTokensPerSec,
                "itl_ms_count": r.metrics.interTokenLatenciesMs.count,
                "itl_ms_median": median(r.metrics.interTokenLatenciesMs),
                "peak_rss_mb": r.metrics.peakResidentMB,
                "energy_per_token_j": jsonFloat(r.metrics.energyPerTokenJ),
                "ane_residency_pct": jsonFloat(r.metrics.aneResidencyPct),
                "gpu_mean_watts": jsonFloat(r.metrics.gpuMeanWatts),
                "cpu_mean_watts": jsonFloat(r.metrics.cpuMeanWatts),
                "warnings": r.metrics.warnings,
            ] as [String: Any]
        }
        let summary = summaryDict(report.runs)
        let warnings = aggregateWarnings(report)
        let dict: [String: Any] = [
            "harness_version": report.provenance.harnessVersion,
            "git_commit": report.provenance.gitCommit ?? NSNull(),
            "git_dirty": report.provenance.gitDirty,
            "engine": [
                "name": report.engineName,
                "commit": report.engineCommit ?? NSNull(),
            ] as [String: Any],
            "model": [
                "path": report.modelPath,
                "params": report.modelParams,
            ] as [String: Any],
            "workload": [
                "mode": report.workload.mode.rawValue,
                "batch_size": report.workload.batchSize,
                "prompt_tokens": report.workload.promptTokens,
                "gen_tokens": report.workload.genTokens,
                "n_runs": report.workload.nRuns,
                "warm_runs": report.workload.warmRuns,
                "energy_metrics_enabled": report.workload.enableEnergy,
            ] as [String: Any],
            "system": [
                "macos_build": report.provenance.macOSBuild,
                "hardware_model": report.provenance.hardwareModel,
                "physical_ram_gb": report.provenance.physicalRamGB,
                "thermal_state": report.provenance.thermalState,
            ] as [String: Any],
            "metrics": summary,
            "runs": runs,
            "warnings": warnings,
        ]
        let data = try JSONSerialization.data(withJSONObject: dict,
                                              options: [.prettyPrinted, .sortedKeys])
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    /// Emit a markdown table summary. Designed for pasting into PRs.
    public static func toMarkdown(_ report: Report) -> String {
        let s = summaryStruct(report.runs)
        var md = "## tinygpt bench — \(report.engineName)\n\n"
        md += "model: `\(report.modelPath)` (\(formatInt(report.modelParams)) params)\n"
        md += "workload: \(report.workload.mode.rawValue), prompt=\(report.workload.promptTokens) tok, gen=\(report.workload.genTokens) tok, batch=\(report.workload.batchSize), n_runs=\(report.workload.nRuns) (+\(report.workload.warmRuns) warm)\n"
        md += "system: \(report.provenance.hardwareModel), \(String(format: "%.1f", report.provenance.physicalRamGB)) GB, macOS build \(report.provenance.macOSBuild)\n"
        md += "harness: v\(report.provenance.harnessVersion) @ \(report.provenance.gitCommit ?? "??")\(report.provenance.gitDirty ? "-dirty" : "")\n"
        if let ec = report.engineCommit { md += "engine commit: \(ec)\n" }
        md += "\n"
        md += "| metric | median | p95 | p99 | n |\n"
        md += "|---|---|---|---|---|\n"
        md += row("TTFT (ms)", s.ttft, fmt: "%.2f")
        md += row("decode tok/s", s.decodeTps, fmt: "%.2f")
        md += row("prefill tok/s", s.prefillTps, fmt: "%.2f")
        md += row("ITL (ms)", s.itl, fmt: "%.2f")
        md += row("peak RSS (MB)", s.peakRss, fmt: "%.1f")
        if !s.energyPerToken.median.isNaN {
            md += row("energy/token (J)", s.energyPerToken, fmt: "%.4f")
            md += row("ANE residency (%)", s.aneResidency, fmt: "%.1f")
            md += row("GPU mean (W)", s.gpuWatts, fmt: "%.2f")
            md += row("CPU mean (W)", s.cpuWatts, fmt: "%.2f")
        } else {
            md += "| energy/token (J) | — | — | — | (skip — no sudo for powermetrics) |\n"
        }
        let warns = aggregateWarnings(report)
        if !warns.isEmpty {
            md += "\nWarnings:\n"
            for w in warns { md += "- \(w)\n" }
        }
        return md
    }

    // ===== summary math =====

    public struct Summary {
        public var median: Double = .nan
        public var p95: Double = .nan
        public var p99: Double = .nan
        public var n: Int = 0
    }

    public struct AllSummaries {
        public var ttft = Summary()
        public var decodeTps = Summary()
        public var prefillTps = Summary()
        public var itl = Summary()
        public var peakRss = Summary()
        public var energyPerToken = Summary()
        public var aneResidency = Summary()
        public var gpuWatts = Summary()
        public var cpuWatts = Summary()
    }

    static func summaryStruct(_ runs: [WorkloadController.RunResult]) -> AllSummaries {
        var s = AllSummaries()
        s.ttft         = summarize(runs.map { $0.metrics.ttftMs })
        s.decodeTps    = summarize(runs.map { $0.metrics.decodeTokensPerSec })
        s.prefillTps   = summarize(runs.map { $0.metrics.prefillTokensPerSec })
        // ITL aggregated across all decode tokens across all runs.
        let allItls = runs.flatMap { Array($0.metrics.interTokenLatenciesMs.dropFirst()) }
        s.itl          = summarize(allItls)
        s.peakRss      = summarize(runs.map { $0.metrics.peakResidentMB })
        s.energyPerToken = summarize(runs.compactMap { $0.metrics.energyPerTokenJ.isFinite ? $0.metrics.energyPerTokenJ : nil })
        s.aneResidency = summarize(runs.compactMap { $0.metrics.aneResidencyPct.isFinite ? $0.metrics.aneResidencyPct : nil })
        s.gpuWatts     = summarize(runs.compactMap { $0.metrics.gpuMeanWatts.isFinite ? $0.metrics.gpuMeanWatts : nil })
        s.cpuWatts     = summarize(runs.compactMap { $0.metrics.cpuMeanWatts.isFinite ? $0.metrics.cpuMeanWatts : nil })
        return s
    }

    static func summaryDict(_ runs: [WorkloadController.RunResult]) -> [String: Any] {
        let s = summaryStruct(runs)
        func summaryToDict(_ s: Summary) -> [String: Any] {
            return [
                "median": jsonFloat(s.median),
                "p95": jsonFloat(s.p95),
                "p99": jsonFloat(s.p99),
                "n": s.n,
            ] as [String: Any]
        }
        return [
            "ttft_ms":           summaryToDict(s.ttft),
            "decode_tps":        summaryToDict(s.decodeTps),
            "prefill_tps":       summaryToDict(s.prefillTps),
            "itl_ms":            summaryToDict(s.itl),
            "peak_rss_mb":       summaryToDict(s.peakRss),
            "energy_per_token_j": summaryToDict(s.energyPerToken),
            "ane_residency_pct": summaryToDict(s.aneResidency),
            "gpu_mean_watts":    summaryToDict(s.gpuWatts),
            "cpu_mean_watts":    summaryToDict(s.cpuWatts),
        ]
    }

    static func summarize(_ values: [Double]) -> Summary {
        var s = Summary()
        let xs = values.filter { $0.isFinite }.sorted()
        s.n = xs.count
        if xs.isEmpty { return s }
        s.median = percentile(xs, 0.50)
        s.p95 = percentile(xs, 0.95)
        s.p99 = percentile(xs, 0.99)
        return s
    }

    static func percentile(_ sorted: [Double], _ p: Double) -> Double {
        if sorted.isEmpty { return .nan }
        if sorted.count == 1 { return sorted[0] }
        // Nearest-rank (no interpolation) — matches what most ops folks
        // expect. With n<20 the difference between methods is dominated
        // by the n=5 noise floor anyway.
        let rank = Int((p * Double(sorted.count - 1)).rounded())
        return sorted[max(0, min(sorted.count - 1, rank))]
    }

    static func median(_ xs: [Double]) -> Double {
        let sorted = xs.filter { $0.isFinite }.sorted()
        if sorted.isEmpty { return .nan }
        let mid = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[mid - 1] + sorted[mid]) / 2.0
            : sorted[mid]
    }

    static func aggregateWarnings(_ report: Report) -> [String] {
        var ws: [String] = []
        for r in report.runs {
            ws.append(contentsOf: r.metrics.warnings)
        }
        // Dedup while preserving order.
        var seen = Set<String>()
        var out: [String] = []
        for w in ws where !seen.contains(w) {
            out.append(w); seen.insert(w)
        }
        if report.runs.count < 20 {
            out.append("n=\(report.runs.count) is small; p95/p99 are unstable. Use --n-runs 20+ for paper-quality numbers.")
        }
        if report.provenance.gitDirty {
            out.append("git tree is dirty — uncommitted changes; results not reproducible. Commit before publishing.")
        }
        return out
    }

    // ===== formatting helpers =====

    static func row(_ name: String, _ s: Summary, fmt: String) -> String {
        if s.n == 0 {
            return "| \(name) | — | — | — | 0 |\n"
        }
        return "| \(name) | \(String(format: fmt, s.median)) | \(String(format: fmt, s.p95)) | \(String(format: fmt, s.p99)) | \(s.n) |\n"
    }

    static func formatInt(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    /// JSONSerialization doesn't accept NaN/Inf; serialize as null.
    static func jsonFloat(_ x: Double) -> Any {
        if x.isFinite { return x }
        return NSNull()
    }
}
