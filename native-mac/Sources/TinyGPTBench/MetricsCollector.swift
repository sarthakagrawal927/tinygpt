import Foundation
import Darwin

/// Aggregates per-run timings, memory samples, and the `powermetrics`
/// time-series into a single `RunMetrics`. The WorkloadController calls
/// `start()` before the engine call and `stop()` after.
///
/// Two cooperating pieces:
///   1. In-process counters (timers, RSS sampling). Always available.
///   2. `PowerSampler` — spawns a child `powermetrics` process. Requires
///      root; we degrade gracefully if not available.
///
/// See docs/benchmark_harness_design.md §3.3.
public final class MetricsCollector {
    /// Per-run captured metrics.
    public struct RunMetrics {
        public var ttftMs: Double = 0
        public var prefillTokens: Int = 0
        public var prefillTokensPerSec: Double = 0
        public var decodeTokens: Int = 0
        public var decodeMs: Double = 0
        public var decodeTokensPerSec: Double = 0
        public var interTokenLatenciesMs: [Double] = []
        public var peakResidentMB: Double = 0
        /// Energy / token over the decode window, in joules. NaN if
        /// powermetrics was unavailable.
        public var energyPerTokenJ: Double = .nan
        /// Fraction of the decode window with ANE power > threshold.
        /// NaN if powermetrics was unavailable.
        public var aneResidencyPct: Double = .nan
        /// Mean GPU power in the decode window, watts. NaN if
        /// powermetrics was unavailable.
        public var gpuMeanWatts: Double = .nan
        /// Mean CPU power in the decode window, watts. NaN if
        /// powermetrics was unavailable.
        public var cpuMeanWatts: Double = .nan
        public var warnings: [String] = []
    }

    private var current = RunMetrics()
    private var powerSampler: PowerSampler?
    private var decodeStart: Date?

    public init() {}

    /// Begin collection. If `enableEnergy` is true, attempt to spawn
    /// powermetrics; if that fails (no sudo), append a warning and
    /// continue without energy metrics.
    public func start(enableEnergy: Bool) {
        current = RunMetrics()
        if enableEnergy {
            let sampler = PowerSampler()
            do {
                try sampler.start()
                self.powerSampler = sampler
            } catch {
                current.warnings.append("powermetrics unavailable: \(error). Energy/ANE metrics will be NaN. Try running with sudo.")
                self.powerSampler = nil
            }
        }
    }

    /// Mark the start of the decode-only window (after prefill) so the
    /// energy attribution can isolate decode from prefill.
    public func markDecodeStart() {
        decodeStart = Date()
        powerSampler?.markDecodeStart()
    }

    /// Stop and return the assembled metrics.
    public func stop(peakResidentBytes: Int) -> RunMetrics {
        current.peakResidentMB = Double(peakResidentBytes) / 1_000_000.0
        if let sampler = powerSampler {
            let snapshot = sampler.stop()
            // Energy in joules over the decode window.
            let totalEnergyJ = snapshot.decodeEnergyJ
            if current.decodeTokens > 0 && totalEnergyJ.isFinite {
                current.energyPerTokenJ = totalEnergyJ / Double(current.decodeTokens)
            }
            current.aneResidencyPct = snapshot.aneResidencyPct
            current.gpuMeanWatts = snapshot.gpuMeanWatts
            current.cpuMeanWatts = snapshot.cpuMeanWatts
        }
        return current
    }

    /// Called by the WorkloadController after `engine.prefill`.
    public func recordPrefill(tokenCount: Int, ttftMs: Double) {
        current.ttftMs = ttftMs
        current.prefillTokens = tokenCount
        if ttftMs > 0 {
            current.prefillTokensPerSec = Double(tokenCount) / (ttftMs / 1000.0)
        }
    }

    /// Called after `engine.decode`.
    public func recordDecode(tokenCount: Int, totalMs: Double, itlsMs: [Double]) {
        current.decodeTokens = tokenCount
        current.decodeMs = totalMs
        current.interTokenLatenciesMs = itlsMs
        if totalMs > 0 {
            current.decodeTokensPerSec = Double(tokenCount) / (totalMs / 1000.0)
        }
    }
}

// =============================================================================
// PowerSampler — spawns `powermetrics`, parses the plist stream.
// =============================================================================

/// Wraps a child `powermetrics` process. Streams XML plist samples and
/// extracts ANE/GPU/CPU power. We pick the plist format because it's
/// the only one stable enough to parse line-by-line without ambiguity.
///
/// powermetrics requires root. If `start()` fails because the binary
/// returns immediately or refuses to run, we throw and let the
/// caller log a warning and continue without energy metrics.
public final class PowerSampler {
    public struct Snapshot {
        public let aneResidencyPct: Double   // % of decode window with ane > threshold
        public let gpuMeanWatts: Double
        public let cpuMeanWatts: Double
        public let decodeEnergyJ: Double     // ∫(ane+gpu+cpu) dt over decode window
    }

    private var process: Process?
    private var outputPipe: Pipe?
    private var queue = DispatchQueue(label: "tinygpt.bench.powermetrics")
    private var samples: [(t: Date, anePowerMW: Double, gpuPowerMW: Double, cpuPowerMW: Double)] = []
    private var samplesLock = NSLock()
    private var decodeStartTime: Date?
    private var decodeEndTime: Date?

    public init() {}

    /// Try to spawn `powermetrics`. Throws if the binary isn't found
    /// or terminates before we can read a single sample (no-sudo
    /// path).
    public func start() throws {
        let p = Process()
        p.launchPath = "/usr/bin/sudo"
        // -n: non-interactive — fail immediately rather than prompt.
        // We intentionally avoid an interactive sudo prompt; the
        // operator runs the harness under sudo if they want energy
        // metrics.
        p.arguments = [
            "-n", "/usr/bin/powermetrics",
            "--samplers", "ane_power,gpu_power,cpu_power",
            "-i", "100",     // 100 ms interval
            "-f", "plist"
        ]
        let outPipe = Pipe()
        let errPipe = Pipe()
        p.standardOutput = outPipe
        p.standardError = errPipe
        do {
            try p.run()
        } catch {
            throw PowerSamplerError.spawnFailed("\(error)")
        }
        self.process = p
        self.outputPipe = outPipe

        // Verify it didn't immediately fail (e.g., "sudo: a password
        // is required"). Give it 200 ms to produce something or die.
        Thread.sleep(forTimeInterval: 0.2)
        if !p.isRunning {
            throw PowerSamplerError.noPermission("powermetrics requires sudo; run the harness with sudo to enable energy metrics")
        }

        // Read in a background thread. The plist stream emits one
        // <plist>…</plist> document per sample interval, separated by
        // a literal \0 byte (powermetrics's "null-terminated plist"
        // mode). For robustness we just look for the closing </plist>.
        let handle = outPipe.fileHandleForReading
        queue.async { [weak self] in
            guard let self = self else { return }
            var buffer = Data()
            while let data = try? handle.read(upToCount: 8192), !data.isEmpty {
                buffer.append(data)
                // Split on </plist> close tags.
                while let range = buffer.range(of: Data("</plist>".utf8)) {
                    let docEnd = range.upperBound
                    let doc = buffer.subdata(in: 0..<docEnd)
                    buffer.removeSubrange(0..<docEnd)
                    self.parseAndAppend(doc)
                }
            }
        }
    }

    /// Mark the start of the decode-only window so we can attribute
    /// energy to decode rather than prefill.
    public func markDecodeStart() {
        decodeStartTime = Date()
    }

    /// Stop the sampler and aggregate the captured samples into the
    /// decode-window snapshot.
    public func stop() -> Snapshot {
        decodeEndTime = Date()
        if let p = process, p.isRunning {
            p.terminate()
            // Give it a beat to flush any in-flight sample.
            Thread.sleep(forTimeInterval: 0.05)
        }
        samplesLock.lock()
        let snap = samples
        samplesLock.unlock()

        guard let start = decodeStartTime else {
            // No decode window marked — return all-NaN.
            return Snapshot(aneResidencyPct: .nan, gpuMeanWatts: .nan,
                            cpuMeanWatts: .nan, decodeEnergyJ: .nan)
        }
        let end = decodeEndTime ?? Date()
        let inWindow = snap.filter { $0.t >= start && $0.t <= end }
        if inWindow.isEmpty {
            return Snapshot(aneResidencyPct: .nan, gpuMeanWatts: .nan,
                            cpuMeanWatts: .nan, decodeEnergyJ: .nan)
        }
        // Trapezoidal integration of (ane+gpu+cpu) power → energy.
        var energyJ: Double = 0
        var aneActive: Double = 0
        var aneTotal: Double = 0
        var gpuSum: Double = 0
        var cpuSum: Double = 0
        for i in 1..<inWindow.count {
            let a = inWindow[i - 1]
            let b = inWindow[i]
            let dt = b.t.timeIntervalSince(a.t)
            // Average power × dt; powers in mW → /1000 → W → × s → J.
            let avgMW = (a.anePowerMW + a.gpuPowerMW + a.cpuPowerMW
                       + b.anePowerMW + b.gpuPowerMW + b.cpuPowerMW) / 2.0
            energyJ += (avgMW / 1000.0) * dt
            aneTotal += dt
            // ANE active if power > 50 mW (noise floor).
            if a.anePowerMW > 50.0 { aneActive += dt }
            gpuSum += a.gpuPowerMW / 1000.0
            cpuSum += a.cpuPowerMW / 1000.0
        }
        let nSamples = Double(inWindow.count - 1)
        let aneResidencyPct = aneTotal > 0 ? (aneActive / aneTotal * 100.0) : .nan
        let gpuMeanW = nSamples > 0 ? gpuSum / nSamples : .nan
        let cpuMeanW = nSamples > 0 ? cpuSum / nSamples : .nan
        return Snapshot(aneResidencyPct: aneResidencyPct,
                         gpuMeanWatts: gpuMeanW,
                         cpuMeanWatts: cpuMeanW,
                         decodeEnergyJ: energyJ)
    }

    /// Parse a single plist document and append the (ANE, GPU, CPU)
    /// power triple. Best-effort — if the keys we expect aren't in this
    /// sample we just skip it rather than abort the run.
    private func parseAndAppend(_ data: Data) {
        // PropertyListSerialization parses the binary OR XML plist.
        guard let raw = try? PropertyListSerialization.propertyList(from: data, format: nil),
              let dict = raw as? [String: Any] else { return }
        let ane = readPowerMW(dict, keys: ["ane_power", "ANE Power"]) ?? 0
        let gpu = readPowerMW(dict, keys: ["gpu_power", "GPU Power"]) ?? 0
        let cpu = readPowerMW(dict, keys: ["cpu_power", "CPU Power", "package_joules"]) ?? 0
        let now = Date()
        samplesLock.lock()
        samples.append((t: now, anePowerMW: ane, gpuPowerMW: gpu, cpuPowerMW: cpu))
        samplesLock.unlock()
    }

    /// powermetrics reports either a scalar in mW or a nested
    /// "_power_mw" structure depending on macOS version. Try both.
    private func readPowerMW(_ dict: [String: Any], keys: [String]) -> Double? {
        for key in keys {
            if let v = dict[key] {
                if let d = v as? Double { return d }
                if let i = v as? Int { return Double(i) }
                if let nested = v as? [String: Any] {
                    for nestedKey in ["mw", "power_mw", "_power_mw"] {
                        if let d = nested[nestedKey] as? Double { return d }
                        if let i = nested[nestedKey] as? Int { return Double(i) }
                    }
                }
            }
        }
        // Walk one level for plat-specific layouts (e.g.,
        // dict["processor"]["ane_power_mw"] in some macOS versions).
        if let processor = dict["processor"] as? [String: Any] {
            for key in keys {
                if let v = processor[key] as? Double { return v }
                if let v = processor[key] as? Int { return Double(v) }
                if let v = processor[key + "_mw"] as? Double { return v }
                if let v = processor[key + "_mw"] as? Int { return Double(v) }
            }
        }
        return nil
    }
}

public enum PowerSamplerError: Error, CustomStringConvertible {
    case spawnFailed(String)
    case noPermission(String)

    public var description: String {
        switch self {
        case .spawnFailed(let why): return "powermetrics spawn failed: \(why)"
        case .noPermission(let why): return "powermetrics: \(why)"
        }
    }
}

// =============================================================================
// ProcessMemory — task_info(TASK_VM_INFO) wrapper.
// =============================================================================

/// Reads `phys_footprint` from `task_info(TASK_VM_INFO)`. This is the
/// number Apple's Instruments uses for "Memory" — it includes wired,
/// resident, and compressed pages. Matches what people mean when they
/// say "this model uses X GB of RAM on my Mac".
public enum ProcessMemory {
    public static func residentBytes() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO),
                          intPtr, &count)
            }
        }
        if kr == KERN_SUCCESS {
            return Int(info.phys_footprint)
        }
        return 0
    }
}
