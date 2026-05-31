// ScreenCapture.swift — capture the active window via ScreenCaptureKit
// (the modern, non-deprecated path; macOS 12.3+, with the per-window
// SCContentFilter init we use here available on macOS 14+ which is our
// minimum platform via Package.swift).
//
// Why ScreenCaptureKit:
//   - CGWindowListCreateImage is deprecated on macOS 15+ and emits a
//     console warning every time it's called.
//   - ScreenCaptureKit is Apple's canonical capture API and surfaces the
//     same window list with proper Screen Recording permission gating.
//
// CLI caveat:
//   ScreenCaptureKit's permission grant lives in System Settings under
//   Screen Recording, and the macOS TCC subsystem keys grants to the
//   *signed bundle identifier* of the calling binary. A bare `swift run`
//   build of `tinygpt` does NOT have a stable bundle ID — TCC will prompt
//   on first use and may not remember the grant across rebuilds. The
//   workaround is to launch tinygpt from a terminal app that already has
//   Screen Recording permission (Terminal.app, iTerm, Ghostty) — the grant
//   is then inherited by the child process. We document this in the CLI
//   help text below.
//
// API:
//   ScreenCapture.captureActiveWindow() -> CGImage
//     Finds the front-most on-screen window owned by the currently active
//     application and returns its raster as a CGImage.

import Foundation
import CoreGraphics
import AppKit
#if canImport(ScreenCaptureKit)
import ScreenCaptureKit
#endif
#if canImport(ImageIO)
import ImageIO
import UniformTypeIdentifiers
#endif

public enum ScreenCapture {
    /// Capture the currently active (front-most, focused) window as a CGImage.
    ///
    /// Synchronous wrapper around the async ScreenCaptureKit API; we use a
    /// semaphore so callers don't need to opt into Swift concurrency at
    /// the CLI layer.
    public static func captureActiveWindow() throws -> CGImage {
        #if canImport(ScreenCaptureKit)
        if #available(macOS 14.0, *) {
            return try captureViaSCKit()
        } else {
            throw ScreenError.unsupportedOS(
                "ScreenCaptureKit window capture requires macOS 14.0+"
            )
        }
        #else
        throw ScreenError.unsupportedOS("ScreenCaptureKit not available on this build target")
        #endif
    }

    /// Save a CGImage as PNG to the given file path. Returns the byte count
    /// written for caller-side logging.
    @discardableResult
    public static func writePNG(_ image: CGImage, to path: String) throws -> Int {
        #if canImport(ImageIO)
        let url = URL(fileURLWithPath: path) as CFURL
        let type = UTType.png.identifier as CFString
        guard let dest = CGImageDestinationCreateWithURL(url, type, 1, nil) else {
            throw ScreenError.encodeFailed("CGImageDestinationCreateWithURL failed")
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw ScreenError.encodeFailed("CGImageDestinationFinalize failed")
        }
        // Best-effort size readback (not load-bearing — purely for logs).
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        return (attrs?[.size] as? Int) ?? 0
        #else
        throw ScreenError.encodeFailed("ImageIO not available on this build target")
        #endif
    }

    // MARK: - SCKit implementation

    #if canImport(ScreenCaptureKit)
    @available(macOS 14.0, *)
    private static func captureViaSCKit() throws -> CGImage {
        // Pick the front-most on-screen window of the active application.
        // SCShareableContent.current is the canonical entry point; it also
        // doubles as a permission gate — if Screen Recording is denied,
        // it throws with a recognisable error and returns no windows.
        let content = try awaitContent()

        guard let target = pickTargetWindow(from: content) else {
            throw ScreenError.captureFailed("no on-screen window found for the active application")
        }

        // SCContentFilter for a single window. Excluding-windows + the
        // window's own display gives us the cleanest single-window crop.
        let filter = SCContentFilter(desktopIndependentWindow: target)

        let config = SCStreamConfiguration()
        // Match the window's pixel dimensions on the current backing
        // scale so the PNG is "what you see, at native res".
        let scale = NSScreen.main?.backingScaleFactor ?? 2.0
        config.width = Int(target.frame.width * scale)
        config.height = Int(target.frame.height * scale)
        config.showsCursor = false
        config.scalesToFit = true
        config.pixelFormat = kCVPixelFormatType_32BGRA

        return try awaitScreenshot(filter: filter, config: config)
    }

    /// Mutable result box bridged across the async/sync boundary. Marked
    /// `@unchecked Sendable` because all reads happen *after* the
    /// DispatchSemaphore signal, which is a strict happens-before
    /// barrier (no concurrent access in practice). This is the standard
    /// "run async work synchronously from a CLI" pattern under Swift 6
    /// strict concurrency.
    private final class Box<T>: @unchecked Sendable {
        var value: Result<T, Error>?
    }

    /// Synchronously fetch SCShareableContent.current. Maps the
    /// permission-denied error to our typed ScreenError so callers can
    /// print a clear remediation message.
    @available(macOS 14.0, *)
    private static func awaitContent() throws -> SCShareableContent {
        let box = Box<SCShareableContent>()
        let sem = DispatchSemaphore(value: 0)
        Task.detached {
            do {
                let c = try await SCShareableContent.current
                box.value = .success(c)
            } catch {
                box.value = .failure(error)
            }
            sem.signal()
        }
        sem.wait()
        switch box.value! {
        case .success(let c):
            return c
        case .failure(let e):
            // ScreenCaptureKit surfaces denied permission as
            // NSError code -3801 in the com.apple.ScreenCaptureKit
            // domain on most macOS versions. We pattern-match loosely
            // (code OR message contains "permission") and route to our
            // typed error; everything else surfaces with full detail.
            let ns = e as NSError
            let msg = ns.localizedDescription.lowercased()
            if ns.code == -3801 || msg.contains("permission") || msg.contains("not authorized") {
                throw ScreenError.screenRecordingPermissionDenied
            }
            throw ScreenError.captureFailed("\(e)")
        }
    }

    @available(macOS 14.0, *)
    private static func awaitScreenshot(
        filter: SCContentFilter,
        config: SCStreamConfiguration
    ) throws -> CGImage {
        let box = Box<CGImage>()
        let sem = DispatchSemaphore(value: 0)
        // SCContentFilter / SCStreamConfiguration are reference-typed
        // ObjC objects; under Swift 6 strict concurrency the compiler
        // wants us to launder them across the Task boundary. We use a
        // local @unchecked-Sendable holder for that — no shared mutation,
        // we hand them in once and never touch them again from this side.
        let captureInputs = CaptureInputs(filter: filter, config: config)
        Task.detached {
            do {
                let image = try await SCScreenshotManager.captureImage(
                    contentFilter: captureInputs.filter,
                    configuration: captureInputs.config
                )
                box.value = .success(image)
            } catch {
                box.value = .failure(error)
            }
            sem.signal()
        }
        sem.wait()
        switch box.value! {
        case .success(let img):
            return img
        case .failure(let e):
            let ns = e as NSError
            let msg = ns.localizedDescription.lowercased()
            if ns.code == -3801 || msg.contains("permission") || msg.contains("not authorized") {
                throw ScreenError.screenRecordingPermissionDenied
            }
            throw ScreenError.captureFailed("\(e)")
        }
    }

    /// Internal "carry the SCKit inputs into a detached Task without
    /// tripping Swift 6 sendability checks" wrapper. The wrapped objects
    /// are Apple framework types that are effectively immutable from our
    /// side after construction; mark unchecked-Sendable.
    @available(macOS 14.0, *)
    private final class CaptureInputs: @unchecked Sendable {
        let filter: SCContentFilter
        let config: SCStreamConfiguration
        init(filter: SCContentFilter, config: SCStreamConfiguration) {
            self.filter = filter
            self.config = config
        }
    }

    /// Pick the front-most on-screen window of the currently active app.
    /// SCShareableContent returns windows ordered front-to-back, but we
    /// filter to (a) on-screen, (b) the active application by bundle ID,
    /// and fall back to "any on-screen window" if there's no match.
    @available(macOS 14.0, *)
    private static func pickTargetWindow(from content: SCShareableContent) -> SCWindow? {
        let active = NSWorkspace.shared.frontmostApplication
        let pid = active?.processIdentifier ?? -1

        let onScreen = content.windows.filter { w in
            // ScreenCaptureKit reports many on-screen windows as zero-sized
            // overlays (status indicators, IME helpers) which would crash
            // SCStreamConfiguration. Require a sensible minimum.
            w.frame.width >= 64 && w.frame.height >= 64 && w.isOnScreen
        }

        if pid != -1, let match = onScreen.first(where: {
            $0.owningApplication?.processID == pid
        }) {
            return match
        }
        return onScreen.first
    }
    #endif
}
