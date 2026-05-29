import SwiftUI
import UniformTypeIdentifiers
import AppKit

/// LoRA fine-tune tab. Same shape as TrainView (header + controls +
/// chart + stats), but with two pickers up top — base model and corpus
/// — and a rank/alpha control instead of a preset picker.
struct FinetuneView: View {
    @StateObject private var controller = FinetuneController()
    @State private var availableCorpora: [CorpusItem] = []
    @State private var selectedCorpus: CorpusItem? = nil

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().background(Theme.line)

            controlsRow
                .padding(.horizontal, 24)
                .padding(.vertical, 12)
                .background(Theme.panel)
            Divider().background(Theme.line)

            LossChart(points: controller.lossHistory,
                      targetSteps: controller.targetSteps)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(.horizontal, 24)
                .padding(.top, 16)

            statsRow
                .padding(.horizontal, 24)
                .padding(.vertical, 16)
        }
        .background(Theme.base)
        .onAppear { availableCorpora = CorpusDiscovery.discover() }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text("Fine-tune")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(Theme.fg)
            Text("LoRA adapter on top of a frozen base · ~100KB-1MB output")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Theme.muted)
            Spacer()
            if controller.isTraining {
                HStack(spacing: 6) {
                    Circle().fill(Theme.accent).frame(width: 6, height: 6)
                    Text(String(format: "step %d / %d · %.1f step/s",
                                controller.stepCount, controller.targetSteps,
                                controller.stepsPerSec))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                }
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 18)
    }

    private var controlsRow: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 14) {
                basePicker
                corpusPicker
                Spacer()
                if controller.isTraining {
                    Button("Stop") { controller.cancel() }
                        .keyboardShortcut(.cancelAction)
                        .buttonStyle(FinetunePrimaryButtonStyle(color: Theme.danger))
                } else {
                    Button("Start") { controller.start() }
                        .keyboardShortcut(.return, modifiers: [.command])
                        .buttonStyle(FinetunePrimaryButtonStyle(color: Theme.accent))
                        .disabled(controller.basePath == nil || controller.corpusPath == nil)
                }
            }
            HStack(spacing: 14) {
                rankPicker
                alphaPicker
                stepsPicker
                Spacer()
            }
        }
    }

    private var basePicker: some View {
        HStack(spacing: 6) {
            Text("base")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(Theme.faint)
            Button {
                pickBase()
            } label: {
                HStack(spacing: 4) {
                    Text(controller.basePath.map {
                        URL(fileURLWithPath: $0).lastPathComponent
                    } ?? "pick .tinygpt or HF dir…")
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            .frame(maxWidth: 220)
        }
    }

    private var corpusPicker: some View {
        HStack(spacing: 6) {
            Text("corpus")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(Theme.faint)
            Menu {
                if availableCorpora.isEmpty {
                    Text("no corpora found").disabled(true)
                }
                ForEach(availableCorpora) { c in
                    Button {
                        selectedCorpus = c
                        controller.corpusPath = c.url.path
                    } label: {
                        HStack {
                            Text(c.icon)
                            Text(c.displayName)
                            Spacer()
                            Text(formattedBytes(c.size))
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                Divider()
                Button("Other file…") { pickCorpusFile() }
            } label: {
                if let c = selectedCorpus {
                    HStack {
                        Text(c.icon)
                        Text(c.displayName).lineLimit(1)
                    }
                } else if let path = controller.corpusPath {
                    Text(URL(fileURLWithPath: path).lastPathComponent)
                        .lineLimit(1)
                        .truncationMode(.middle)
                } else {
                    Text("pick corpus…")
                }
            }
            .menuStyle(.borderlessButton)
            .frame(maxWidth: 200)
        }
    }

    private var rankPicker: some View {
        HStack(spacing: 6) {
            Text("rank")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(Theme.faint)
            Picker("", selection: $controller.rank) {
                ForEach([2, 4, 8, 16], id: \.self) { r in
                    Text("\(r)").tag(r)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 64)
        }
    }

    private var alphaPicker: some View {
        HStack(spacing: 6) {
            Text("alpha")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(Theme.faint)
            TextField("", value: $controller.alpha, format: .number)
                .textFieldStyle(.plain)
                .frame(width: 50)
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(Theme.panel)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(Theme.line))
                .clipShape(RoundedRectangle(cornerRadius: 4))
                .font(.system(size: 11, design: .monospaced))
        }
    }

    private var stepsPicker: some View {
        HStack(spacing: 6) {
            Text("steps")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(Theme.faint)
            TextField("", value: $controller.targetSteps, format: .number)
                .textFieldStyle(.plain)
                .frame(width: 70)
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(Theme.panel)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(Theme.line))
                .clipShape(RoundedRectangle(cornerRadius: 4))
                .font(.system(size: 11, design: .monospaced))
        }
    }

    private var statsRow: some View {
        HStack(spacing: 24) {
            statBlock(label: "STEP", value: "\(controller.stepCount)")
            statBlock(label: "LOSS",
                      value: controller.currentLoss == 0 ? "—" :
                        String(format: "%.3f", controller.currentLoss))
            statBlock(label: "STEP/S",
                      value: controller.stepsPerSec == 0 ? "—" :
                        String(format: "%.1f", controller.stepsPerSec))
            Spacer()
            VStack(alignment: .leading, spacing: 4) {
                Text(controller.status)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.muted)
                    .lineLimit(2)
                    .truncationMode(.tail)
                if let path = controller.savedAdapterPath {
                    Button {
                        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                    } label: {
                        Text("→ reveal in Finder")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(Theme.accent)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private func statBlock(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 9, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.faint)
                .tracking(1)
            Text(value)
                .font(.system(size: 18, weight: .medium, design: .monospaced))
                .foregroundStyle(Theme.fg)
        }
        .frame(minWidth: 60, alignment: .leading)
    }

    /// Open panel that accepts either a `.tinygpt` file OR a folder
    /// (HuggingFace model directory).
    private func pickBase() {
        let panel = NSOpenPanel()
        panel.message = "Pick a .tinygpt file or a HuggingFace model directory."
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            controller.basePath = url.path
        }
    }

    private func pickCorpusFile() {
        let panel = NSOpenPanel()
        panel.message = "Pick a UTF-8 text file to fine-tune on."
        panel.allowedContentTypes = [.plainText, .utf8PlainText, .text]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            controller.corpusPath = url.path
            selectedCorpus = nil
        }
    }

    private func formattedBytes(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1fMB", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0fKB", Double(n) / 1_000) }
        return "\(n)B"
    }
}

private struct FinetunePrimaryButtonStyle: ButtonStyle {
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
