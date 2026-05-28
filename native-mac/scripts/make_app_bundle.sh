#!/bin/bash
# Wrap the SwiftPM-built TinyGPTApp executable in a minimal .app bundle so
# launchd treats it as a GUI app (dock icon, menubar, frontmost window).
# Run after `xcodebuild -scheme TinyGPTApp ... build`.
set -euo pipefail

BUILD_DIR="${1:-.xcode-build/Build/Products/Debug}"
APP_NAME="TinyGPT"
BUNDLE_ID="dev.sarthakagrawal.tinygpt"
EXEC_NAME="TinyGPTApp"
OUT="${BUILD_DIR}/${APP_NAME}.app"

rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Resources"

# Copy the executable
cp "${BUILD_DIR}/${EXEC_NAME}" "$OUT/Contents/MacOS/${APP_NAME}"

# Info.plist
cat > "$OUT/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF


# Ad-hoc sign the bundle so macOS lets it launch without Gatekeeper
# prompting. Real signing requires a Developer ID; for local builds
# the empty-identity signature is enough to clear the immediate launch
# block (the bundle still won't pass notarization, but it runs).
codesign --force --deep --sign - "$OUT" >/dev/null 2>&1 || true

# Strip the com.apple.quarantine extended attribute that downloads /
# xcodebuild outputs sometimes inherit.
xattr -dr com.apple.quarantine "$OUT" 2>/dev/null || true

echo "wrote $OUT"
echo ""
echo "To launch: open '$OUT'"
echo "Or double-click in Finder. First time may need right-click → Open."
