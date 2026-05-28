import SwiftUI

@main
struct TinyGPTApp: App {
    var body: some Scene {
        Window("TinyGPT", id: "main") {
            ContentView()
                .frame(minWidth: 880, minHeight: 560)
                .preferredColorScheme(.dark)
        }
        .windowStyle(.titleBar)
        .windowResizability(.contentSize)
    }
}
