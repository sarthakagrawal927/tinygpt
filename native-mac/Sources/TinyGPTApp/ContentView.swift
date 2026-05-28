import SwiftUI

enum AppTab: Hashable { case sample, train }

struct ContentView: View {
    @StateObject private var controller = ModelController()
    @StateObject private var stats = MachineStats()
    @State private var galleryItems: [GalleryItem] = []
    @State private var selectedItem: GalleryItem? = nil
    @State private var prompt: String = "ROMEO:"
    @State private var maxTokens: Int = 200
    @State private var temperature: Double = 0.8
    @State private var tab: AppTab = .sample

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                sidebar
                    .frame(width: 220)
                    .background(Theme.panel)

                Divider().background(Theme.line)

                VStack(spacing: 0) {
                    tabBar
                    Divider().background(Theme.line)
                    Group {
                        switch tab {
                        case .sample: mainPane
                        case .train: TrainView()
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
                .background(Theme.base)
            }

            // Machine-stats strip — sticky bottom, mono+compact
            Divider().background(Theme.line)
            machineStatsBar
        }
        .onAppear {
            galleryItems = GalleryDiscovery.discover()
        }
    }

    private var machineStatsBar: some View {
        HStack(spacing: 16) {
            statsBlock("CHIP", stats.cpuModel.replacingOccurrences(of: "Apple ", with: ""))
            statsBlock("CORES", "\(stats.cpuCores)")
            statsBlock("GPU", stats.gpuName.isEmpty ? "—" : stats.gpuName)
            Divider().frame(height: 18).background(Theme.line)
            statsBlock("APP RAM", FormatBytes.compact(stats.processRSSBytes))
            statsBlock("FREE RAM", FormatBytes.compact(stats.freeRAMBytes))
            statsBlock("TOTAL", FormatBytes.compact(stats.totalRAMBytes))
            Spacer()
            statsBlock("GPU MAX SET", "\(stats.gpuRegistryMB) MB")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(Theme.panel2)
    }

    private func statsBlock(_ label: String, _ value: String) -> some View {
        HStack(spacing: 4) {
            Text(label)
                .font(.system(size: 9, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.faint)
                .tracking(1)
            Text(value)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(Theme.fg)
        }
    }

    private var tabBar: some View {
        HStack(spacing: 0) {
            tabButton(.sample, label: "Sample")
            tabButton(.train, label: "Train")
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.top, 12)
        .padding(.bottom, 4)
        .background(Theme.panel)
    }

    private func tabButton(_ which: AppTab, label: String) -> some View {
        let active = tab == which
        return Button {
            tab = which
        } label: {
            VStack(spacing: 6) {
                Text(label)
                    .font(.system(size: 13, weight: active ? .semibold : .regular))
                    .foregroundStyle(active ? Theme.accent : Theme.muted)
                Rectangle()
                    .fill(active ? Theme.accent : Color.clear)
                    .frame(height: 2)
            }
            .padding(.horizontal, 12)
        }
        .buttonStyle(.plain)
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header — app brand
            VStack(alignment: .leading, spacing: 4) {
                Text("TinyGPT")
                    .font(.tgDisplay)
                    .foregroundStyle(Theme.fg)
                Text("native macOS")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.muted)
            }
            .padding(.horizontal, 20)
            .padding(.top, 24)
            .padding(.bottom, 24)

            // Gallery list
            Text("GALLERY")
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.faint)
                .tracking(1)
                .padding(.horizontal, 20)
                .padding(.bottom, 8)

            ScrollView {
                LazyVStack(spacing: 4) {
                    if galleryItems.isEmpty {
                        Text("no models found — drop checkpoints in browser/public/gallery/")
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.faint)
                            .padding(.horizontal, 20)
                            .padding(.vertical, 8)
                    } else {
                        ForEach(galleryItems) { item in
                            galleryRow(item)
                        }
                    }
                }
                .padding(.horizontal, 12)
            }

            Spacer()

            // Status bar
            VStack(alignment: .leading, spacing: 6) {
                Divider().background(Theme.line)
                Text(controller.status)
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.muted)
                    .lineLimit(2)
                    .truncationMode(.tail)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 12)
            }
        }
    }

    private func galleryRow(_ item: GalleryItem) -> some View {
        let isSelected = controller.loadedItem?.id == item.id
        return Button {
            selectedItem = item
            prompt = item.prompt
            Task { await controller.load(item) }
        } label: {
            HStack(spacing: 10) {
                Text(item.icon)
                    .font(.system(size: 18))
                Text(item.displayName)
                    .font(.system(size: 13, weight: isSelected ? .medium : .regular))
                    .foregroundStyle(isSelected ? Theme.accent : Theme.fg)
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(isSelected ? Theme.accentGlow : Color.clear)
            )
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var mainPane: some View {
        if controller.loadedItem == nil {
            placeholderPane
        } else {
            generationPane
        }
    }

    private var placeholderPane: some View {
        VStack(spacing: 14) {
            Text("←")
                .font(.system(size: 40, weight: .ultraLight))
                .foregroundStyle(Theme.faint)
            Text("Pick a model from the gallery")
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(Theme.muted)
            Text("Each gallery model is a 9.6M-parameter byte-level transformer\ntrained on a different corpus. Same architecture, different mind.")
                .multilineTextAlignment(.center)
                .font(.system(size: 12))
                .foregroundStyle(Theme.faint)
                .lineSpacing(4)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var generationPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Model header
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(controller.loadedItem?.icon ?? "•")
                    .font(.system(size: 24))
                Text(controller.loadedItem?.displayName ?? "")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text("\(formattedInt(controller.paramCount)) params")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.muted)
                Spacer()
                if controller.isGenerating || controller.tokensPerSec > 0 {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(controller.isGenerating ? Theme.accent : Theme.muted)
                            .frame(width: 6, height: 6)
                        Text(String(format: "%.0f tok/s", controller.tokensPerSec))
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(Theme.muted)
                    }
                }
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 20)
            Divider().background(Theme.line)

            // Output panel — always reads top-down, scrolls if long
            ScrollView {
                ScrollViewReader { proxy in
                    Text(controller.generated.isEmpty
                         ? "Output will appear here as the model generates token-by-token."
                         : controller.generated)
                        .font(.tgMono)
                        .foregroundStyle(controller.generated.isEmpty ? Theme.faint : Theme.fg)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(24)
                        .id("output-end")
                        .onChange(of: controller.generated) { _, _ in
                            withAnimation(.linear(duration: 0.1)) {
                                proxy.scrollTo("output-end", anchor: .bottom)
                            }
                        }
                }
            }
            .frame(maxHeight: .infinity)

            Divider().background(Theme.line)

            // Controls
            HStack(spacing: 16) {
                TextField("Prompt", text: $prompt, axis: .horizontal)
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Theme.panel)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .font(.tgMono)

                HStack(spacing: 6) {
                    Text("temp")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                    Slider(value: $temperature, in: 0...1.5)
                        .frame(width: 100)
                    Text(String(format: "%.2f", temperature))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                        .frame(width: 36, alignment: .leading)
                }

                HStack(spacing: 6) {
                    Text("tokens")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                    TextField("", value: $maxTokens, format: .number)
                        .textFieldStyle(.plain)
                        .frame(width: 60)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)
                        .background(Theme.panel)
                        .overlay(RoundedRectangle(cornerRadius: 4).stroke(Theme.line))
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                        .font(.system(size: 11, design: .monospaced))
                }

                if controller.isGenerating {
                    Button("Stop") {
                        controller.cancelGeneration()
                    }
                    .keyboardShortcut(.cancelAction)
                    .buttonStyle(PrimaryButtonStyle(color: Theme.danger))
                } else {
                    Button("Generate") {
                        controller.generate(prompt: prompt, maxTokens: maxTokens,
                                          temperature: Float(temperature))
                    }
                    .keyboardShortcut(.return, modifiers: [.command])
                    .buttonStyle(PrimaryButtonStyle(color: Theme.accent))
                    .disabled(controller.loadedItem == nil)
                }
            }
            .padding(20)
            .background(Theme.panel)
        }
    }

    private func formattedInt(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }
}

private struct PrimaryButtonStyle: ButtonStyle {
    let color: Color
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(color.opacity(configuration.isPressed ? 0.25 : 0.15))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(color.opacity(0.5), lineWidth: 1)
            )
    }
}
