// AccessibilityTree.swift — read the macOS Accessibility (AX) tree of the
// currently focused window using ApplicationServices' AXUIElement APIs.
//
// This is the LM-friendly half of the screen-reader: a structured,
// hierarchical text view of every UI element the focused app exposes —
// role, title, label, value, position, size, children. That representation
// is dramatically more useful to a small tool-calling LM than raw pixels,
// and it's the canonical source-of-truth for click/type actuators we'll
// build on top later.
//
// Requires the host process to be granted Accessibility permission in
// System Settings → Privacy & Security → Accessibility. If the permission
// is missing we raise `ScreenError.accessibilityPermissionDenied` with a
// human-readable hint rather than crashing or silently returning empty.
//
// Scope:
//   - One-shot capture of the FOCUSED window (the system-wide focused UI
//     element's containing window).
//   - Walks the entire subtree. We cap the depth and per-node child count
//     to keep pathological apps (long lists) from blowing up the output.
//   - We *do not* run any AX queries that require elevated trust beyond
//     plain Accessibility access (no screen-recording escalation, no
//     AppleEvent automation).
//
// Out of scope (future work):
//   - Continuous polling / observers.
//   - Click/type actuators (would set AXValue / post AXPress actions).
//   - OCR fallback for un-labelled elements.

import Foundation
import ApplicationServices
import AppKit

/// Codable representation of a single AX element. Children are nested so
/// the whole tree round-trips to JSON cleanly.
public struct AXNode: Codable, Sendable {
    public var role: String?
    public var subrole: String?
    public var title: String?
    public var label: String?
    public var value: String?
    public var help: String?
    public var roleDescription: String?
    public var identifier: String?
    public var enabled: Bool?
    public var focused: Bool?
    public var frame: Frame?
    public var children: [AXNode]

    public struct Frame: Codable, Sendable {
        public var x: Double
        public var y: Double
        public var width: Double
        public var height: Double
    }

    public init(
        role: String? = nil,
        subrole: String? = nil,
        title: String? = nil,
        label: String? = nil,
        value: String? = nil,
        help: String? = nil,
        roleDescription: String? = nil,
        identifier: String? = nil,
        enabled: Bool? = nil,
        focused: Bool? = nil,
        frame: Frame? = nil,
        children: [AXNode] = []
    ) {
        self.role = role
        self.subrole = subrole
        self.title = title
        self.label = label
        self.value = value
        self.help = help
        self.roleDescription = roleDescription
        self.identifier = identifier
        self.enabled = enabled
        self.focused = focused
        self.frame = frame
        self.children = children
    }
}

public enum ScreenError: Error, CustomStringConvertible {
    case accessibilityPermissionDenied
    case screenRecordingPermissionDenied
    case noFocusedApplication
    case noFocusedWindow
    case captureFailed(String)
    case encodeFailed(String)
    case writeFailed(String)
    case unsupportedOS(String)

    public var description: String {
        switch self {
        case .accessibilityPermissionDenied:
            return """
            Accessibility permission denied.

            tinygpt needs Accessibility access to read the focused window's
            UI tree. Grant it in:
              System Settings → Privacy & Security → Accessibility
            and add (or enable) the `tinygpt` binary or the terminal you
            launched it from. After granting, restart the terminal.
            """
        case .screenRecordingPermissionDenied:
            return """
            Screen Recording permission denied.

            tinygpt needs Screen Recording access to capture window pixels
            via ScreenCaptureKit. Grant it in:
              System Settings → Privacy & Security → Screen Recording
            and add (or enable) the `tinygpt` binary or the terminal you
            launched it from. After granting, restart the terminal.
            """
        case .noFocusedApplication:
            return "AX: no application currently has focus."
        case .noFocusedWindow:
            return "AX: focused application has no focused window."
        case .captureFailed(let m):
            return "Screen capture failed: \(m)"
        case .encodeFailed(let m):
            return "Image encode failed: \(m)"
        case .writeFailed(let m):
            return "Write failed: \(m)"
        case .unsupportedOS(let m):
            return "Unsupported OS: \(m)"
        }
    }
}

public enum AccessibilityTree {
    /// Maximum depth we descend into the AX subtree. Past this we drop
    /// children (with a synthetic role placeholder) to keep JSON bounded.
    public static let defaultMaxDepth = 16

    /// Maximum children we expand per node. Long lists (e.g. Mail's
    /// message list) can contain thousands; cap to keep output tractable.
    public static let defaultMaxChildrenPerNode = 64

    /// Check whether the process has Accessibility permission. Does NOT
    /// trigger the permission prompt by itself.
    public static func hasPermission() -> Bool {
        return AXIsProcessTrusted()
    }

    /// Request Accessibility permission, popping the system prompt if
    /// needed. Returns true if the process is already trusted; otherwise
    /// returns false (user must grant in System Settings + re-launch).
    @discardableResult
    public static func requestPermission(prompt: Bool = true) -> Bool {
        // `kAXTrustedCheckOptionPrompt` is imported as a mutable global
        // CFStringRef, which Swift 6 flags as non-concurrency-safe even
        // though it's effectively a constant CFString. Reconstructing the
        // key by its public name avoids the warning and is semantically
        // identical (HIServices documents the value as
        // "AXTrustedCheckOptionPrompt").
        let key = "AXTrustedCheckOptionPrompt" as CFString
        let opts: CFDictionary = [key: prompt] as CFDictionary
        return AXIsProcessTrustedWithOptions(opts)
    }

    /// Read the AX tree of the currently focused window.
    public static func readFocused(
        maxDepth: Int = defaultMaxDepth,
        maxChildrenPerNode: Int = defaultMaxChildrenPerNode
    ) throws -> AXNode {
        guard hasPermission() else {
            // Pop the prompt so the user sees the system dialog the first
            // time they run this; the call itself still throws because
            // permission won't be granted within the same process lifetime.
            _ = requestPermission(prompt: true)
            throw ScreenError.accessibilityPermissionDenied
        }

        // Find the focused application via the system-wide AXUIElement.
        let systemWide = AXUIElementCreateSystemWide()
        guard let focusedApp = copyAttribute(systemWide, kAXFocusedApplicationAttribute) else {
            throw ScreenError.noFocusedApplication
        }
        // Force-cast: AX returns AnyObject; AXUIElement is the AX type.
        let appElement = focusedApp as! AXUIElement

        // Prefer the focused window; fall back to main window.
        let windowAny =
            copyAttribute(appElement, kAXFocusedWindowAttribute)
            ?? copyAttribute(appElement, kAXMainWindowAttribute)
        guard let windowAny = windowAny else {
            throw ScreenError.noFocusedWindow
        }
        let windowElement = windowAny as! AXUIElement

        return walk(
            element: windowElement,
            depth: 0,
            maxDepth: maxDepth,
            maxChildrenPerNode: maxChildrenPerNode
        )
    }

    // MARK: - AX walker

    private static func walk(
        element: AXUIElement,
        depth: Int,
        maxDepth: Int,
        maxChildrenPerNode: Int
    ) -> AXNode {
        var node = AXNode()
        node.role = stringAttr(element, kAXRoleAttribute)
        node.subrole = stringAttr(element, kAXSubroleAttribute)
        node.title = stringAttr(element, kAXTitleAttribute)
        node.label = stringAttr(element, kAXDescriptionAttribute)
        node.value = stringAttr(element, kAXValueAttribute)
        node.help = stringAttr(element, kAXHelpAttribute)
        node.roleDescription = stringAttr(element, kAXRoleDescriptionAttribute)
        node.identifier = stringAttr(element, kAXIdentifierAttribute)
        node.enabled = boolAttr(element, kAXEnabledAttribute)
        node.focused = boolAttr(element, kAXFocusedAttribute)
        node.frame = frameAttr(element)

        if depth >= maxDepth {
            // Mark truncation so consumers can see the tree was clipped.
            node.children = [AXNode(role: "AXTruncated")]
            return node
        }

        if let kids = copyAttribute(element, kAXChildrenAttribute) as? [AXUIElement] {
            let bounded = Array(kids.prefix(maxChildrenPerNode))
            var children: [AXNode] = []
            children.reserveCapacity(bounded.count)
            for kid in bounded {
                children.append(walk(
                    element: kid,
                    depth: depth + 1,
                    maxDepth: maxDepth,
                    maxChildrenPerNode: maxChildrenPerNode
                ))
            }
            if kids.count > bounded.count {
                children.append(AXNode(
                    role: "AXTruncated",
                    label: "+\(kids.count - bounded.count) more children"
                ))
            }
            node.children = children
        }

        return node
    }

    // MARK: - AX attribute helpers
    //
    // These wrap the C-style AXUIElementCopyAttributeValue API into typed
    // optionals. We swallow errors silently — most attributes are missing
    // on most elements, and an "attribute not present" error is a normal
    // signal rather than a problem.

    private static func copyAttribute(_ el: AXUIElement, _ attr: String) -> AnyObject? {
        var value: AnyObject?
        let status = AXUIElementCopyAttributeValue(el, attr as CFString, &value)
        return status == .success ? value : nil
    }

    private static func stringAttr(_ el: AXUIElement, _ attr: String) -> String? {
        guard let v = copyAttribute(el, attr) else { return nil }
        if let s = v as? String { return s.isEmpty ? nil : s }
        // Many AX attributes can be returned as NSAttributedString or
        // even as numeric types (AXValueAttribute on sliders, for
        // instance). Coerce to string descriptively rather than crash.
        if let n = v as? NSNumber { return n.stringValue }
        if let a = v as? NSAttributedString { return a.string }
        return nil
    }

    private static func boolAttr(_ el: AXUIElement, _ attr: String) -> Bool? {
        guard let v = copyAttribute(el, attr) else { return nil }
        if let b = v as? Bool { return b }
        if let n = v as? NSNumber { return n.boolValue }
        return nil
    }

    private static func frameAttr(_ el: AXUIElement) -> AXNode.Frame? {
        // Position + size live in their own attributes and are wrapped
        // in AXValue (a CoreFoundation type). Decode each separately.
        var posPoint = CGPoint.zero
        var sizeSize = CGSize.zero
        var havePos = false
        var haveSize = false

        if let raw = copyAttribute(el, kAXPositionAttribute) {
            // Force-cast: we know AXPositionAttribute returns AXValue.
            let axval = raw as! AXValue
            if AXValueGetValue(axval, .cgPoint, &posPoint) {
                havePos = true
            }
        }
        if let raw = copyAttribute(el, kAXSizeAttribute) {
            let axval = raw as! AXValue
            if AXValueGetValue(axval, .cgSize, &sizeSize) {
                haveSize = true
            }
        }
        guard havePos || haveSize else { return nil }
        return AXNode.Frame(
            x: Double(posPoint.x),
            y: Double(posPoint.y),
            width: Double(sizeSize.width),
            height: Double(sizeSize.height)
        )
    }
}
