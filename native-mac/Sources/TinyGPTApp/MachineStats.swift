import Foundation
import Darwin
#if canImport(Metal)
@preconcurrency import Metal
#endif

/// Live system / process metrics the app pulls itself — no external
/// dependencies. Polled at 2 Hz from the UI.
@MainActor
final class MachineStats: ObservableObject {
    @Published var processRSSBytes: Int = 0    // resident memory this process holds
    @Published var totalRAMBytes: Int = 0
    @Published var freeRAMBytes: Int = 0
    @Published var cpuCores: Int = 0
    @Published var cpuModel: String = ""
    @Published var gpuName: String = ""
    @Published var gpuRegistryMB: Int = 0      // GPU recommended max working set
    private var pollTimer: Timer?

    init() {
        self.totalRAMBytes = Int(ProcessInfo.processInfo.physicalMemory)
        self.cpuCores = ProcessInfo.processInfo.activeProcessorCount
        self.cpuModel = Self.sysctlString("machdep.cpu.brand_string") ?? "Apple Silicon"
        #if canImport(Metal)
        if let device = MTLCreateSystemDefaultDevice() {
            self.gpuName = device.name
            // .recommendedMaxWorkingSetSize is the chunk Metal expects you
            // to stay under for sustained perf; on M-series it's typically
            // ~75% of unified RAM.
            self.gpuRegistryMB = Int(device.recommendedMaxWorkingSetSize / (1024 * 1024))
        }
        #endif
        refresh()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        }
    }

    // Note: poll timer naturally tears down when the @MainActor object is
    // deallocated by ARC; explicit invalidate() in deinit isn't legal under
    // Swift 6 strict concurrency (deinit is nonisolated and Timer isn't
    // Sendable). The timer holds a weak ref to self, so no retain cycle.

    func refresh() {
        self.processRSSBytes = Self.processRSS()
        self.freeRAMBytes = Self.freeMemoryBytes()
    }

    // MARK: - System probes

    private static func processRSS() -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kerr == KERN_SUCCESS ? Int(info.resident_size) : 0
    }

    private static func freeMemoryBytes() -> Int {
        // vm_statistics — pages free + speculative + inactive (roughly
        // what's available; "inactive" can be reclaimed without paging).
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        // vm_kernel_page_size is a global mutable in Darwin headers — Swift 6
        // strict-concurrency flags it. The page size is set once at boot and
        // never changes during the process lifetime; resolve via sysctl to
        // sidestep the lint.
        var pageSize: vm_size_t = 0
        var psSize = MemoryLayout<vm_size_t>.size
        _ = sysctlbyname("hw.pagesize", &pageSize, &psSize, nil, 0)
        if pageSize == 0 { pageSize = 16384 }  // Apple Silicon default
        let kerr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kerr == KERN_SUCCESS else { return 0 }
        let free = UInt64(stats.free_count + stats.speculative_count) * UInt64(pageSize)
        return Int(free)
    }

    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
        return String(cString: buf)
    }
}

/// Small helpers for human-friendly formatting.
enum FormatBytes {
    static func compact(_ n: Int) -> String {
        if n >= 1_073_741_824 { return String(format: "%.1f GB", Double(n) / 1_073_741_824) }
        if n >= 1_048_576 { return String(format: "%.0f MB", Double(n) / 1_048_576) }
        if n >= 1_024 { return String(format: "%.0f KB", Double(n) / 1_024) }
        return "\(n) B"
    }
}
