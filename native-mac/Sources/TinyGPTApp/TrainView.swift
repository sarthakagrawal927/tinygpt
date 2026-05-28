import SwiftUI
import UniformTypeIdentifiers

struct TrainView: View {
    @StateObject private var controller = TrainController()
    @State private var corpusText: String = "(no corpus loaded — drop a UTF-8 text file or pick one below)"
    @State private var hasRealCorpus: Bool = false
    @State private var corpusBytes: Int = 0
    @State private var availableCorpora: [CorpusItem] = []
    @State private var selectedCorpus: CorpusItem? = nil

    var body: some View {
        VStack(spacing: 0) {
            // Header — current run summary
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Text("Train")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text("watch a model learn from scratch")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.muted)
                Spacer()
                if controller.isTraining {
                    HStack(spacing: 6) {
                        Circle().fill(Theme.accent).frame(width: 6, height: 6)
                        Text(String(format: "step %d / %d · %.1f step/s", controller.stepCount, controller.targetSteps, controller.stepsPerSec))
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(Theme.muted)
                    }
                }
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 18)
            Divider().background(Theme.line)

            // Controls row
            HStack(spacing: 14) {
                // Preset picker
                HStack(spacing: 6) {
                    Text("preset")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                    Picker("", selection: $controller.presetIdx) {
                        ForEach(0..<TrainController.presets.count, id: \.self) { i in
                            Text(TrainController.presets[i].name).tag(i)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 100)
                }

                HStack(spacing: 6) {
                    Text("steps")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                    TextField("", value: $controller.targetSteps, format: .number)
                        .textFieldStyle(.plain)
                        .frame(width: 60)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)
                        .background(Theme.panel)
                        .overlay(RoundedRectangle(cornerRadius: 4).stroke(Theme.line))
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                        .font(.system(size: 11, design: .monospaced))
                }

                // Starter corpora menu — one click loads any of the
                // fetched Project Gutenberg classics or browser gallery
                // corpora. Falls through to "Other..." for arbitrary files.
                Menu {
                    if availableCorpora.isEmpty {
                        Text("no corpora found — run scripts/fetch_corpora.sh").disabled(true)
                    }
                    ForEach(availableCorpora) { c in
                        Button {
                            loadCorpus(c)
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
                    Button("Other file…") { pickCorpus() }
                } label: {
                    HStack(spacing: 6) {
                        if let sel = selectedCorpus {
                            Text(sel.icon)
                            Text(sel.displayName)
                            Text("(\(formattedBytes(corpusBytes)))")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(Theme.faint)
                        } else if hasRealCorpus {
                            Text("Corpus (\(formattedBytes(corpusBytes)))")
                        } else {
                            Text("Pick corpus…")
                        }
                    }
                }
                .menuStyle(.borderlessButton)
                .frame(maxWidth: 200)

                Spacer()

                if controller.isTraining {
                    Button("Stop") { controller.cancel() }
                        .keyboardShortcut(.cancelAction)
                        .buttonStyle(PrimaryButtonStyle(color: Theme.danger))
                } else {
                    Button("Start") {
                        let corpus: Data
                        if hasRealCorpus {
                            corpus = Data(corpusText.utf8)
                        } else {
                            // Random bytes — perf demo even without a corpus.
                            // Loss will land at ln(256), not below.
                            corpus = Data((0..<200_000).map { _ in UInt8.random(in: 0...255) })
                        }
                        controller.start(corpus: corpus)
                    }
                    .keyboardShortcut(.return, modifiers: [.command])
                    .buttonStyle(PrimaryButtonStyle(color: Theme.accent))
                }
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(Theme.panel)
            Divider().background(Theme.line)

            // Chart
            LossChart(points: controller.lossHistory, targetSteps: controller.targetSteps)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(.horizontal, 24)
                .padding(.top, 16)

            // Stats row
            HStack(spacing: 24) {
                statBlock(label: "STEP", value: "\(controller.stepCount)")
                statBlock(label: "LOSS",
                          value: controller.currentLoss == 0
                            ? "—"
                            : String(format: "%.3f", controller.currentLoss))
                statBlock(label: "STEP/S",
                          value: controller.stepsPerSec == 0
                            ? "—"
                            : String(format: "%.1f", controller.stepsPerSec))
                Spacer()
                Text(controller.status)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.muted)
                    .lineLimit(2)
                    .truncationMode(.tail)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 16)
        }
        .background(Theme.base)
        .onAppear { availableCorpora = CorpusDiscovery.discover() }
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

    private func loadCorpus(_ c: CorpusItem) {
        do {
            let text = try String(contentsOf: c.url, encoding: .utf8)
            corpusText = text
            corpusBytes = text.utf8.count
            hasRealCorpus = true
            selectedCorpus = c
        } catch {
            corpusText = "(couldn't load \(c.url.lastPathComponent): \(error))"
            hasRealCorpus = false
            selectedCorpus = nil
        }
    }

    private func pickCorpus() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.plainText, .utf8PlainText, .text]
        panel.allowsMultipleSelection = false
        panel.message = "Pick a UTF-8 text file to train on."
        if panel.runModal() == .OK, let url = panel.url {
            do {
                let text = try String(contentsOf: url, encoding: .utf8)
                corpusText = text
                corpusBytes = text.utf8.count
                hasRealCorpus = true
            } catch {
                corpusText = "(couldn't load \(url.lastPathComponent): \(error))"
                hasRealCorpus = false
            }
        }
    }

    private func formattedBytes(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1fMB", Double(n)/1_000_000) }
        if n >= 1_000 { return String(format: "%.0fKB", Double(n)/1_000) }
        return "\(n)B"
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
