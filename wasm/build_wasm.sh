#!/usr/bin/env bash
# build_wasm.sh — compile the kernels + model to a WebAssembly ES module with
# Emscripten. Run wasm/build_native.sh first: it verifies the same C++ with the
# host compiler, which is far faster to iterate on than the browser.
#
# Output: browser/public/tinygpt.{js,wasm} — the Worker imports these.
# Needs the Emscripten SDK on PATH (source ~/emsdk/emsdk_env.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/browser/public"
mkdir -p "$OUT_DIR"

if ! command -v emcc >/dev/null 2>&1; then
  # Fall back to the default emsdk location.
  if [ -f "$HOME/emsdk/emsdk_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/emsdk/emsdk_env.sh" >/dev/null
  else
    echo "error: emcc not found. Install the Emscripten SDK first." >&2
    exit 1
  fi
fi

echo "emcc: $(emcc --version | head -1)"
emcc \
  "$ROOT"/wasm/src/tensor.cpp \
  "$ROOT"/wasm/src/matmul.cpp \
  "$ROOT"/wasm/src/layernorm.cpp \
  "$ROOT"/wasm/src/attention.cpp \
  "$ROOT"/wasm/src/adamw.cpp \
  "$ROOT"/wasm/src/model.cpp \
  -O3 -std=c++17 -msimd128 \
  -s MODULARIZE=1 -s EXPORT_ES6=1 -s EXPORT_NAME=createTinyGPT \
  -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=33554432 \
  -s ENVIRONMENT=web,worker,node \
  -s EXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8,HEAPF32 \
  -s EXPORTED_FUNCTIONS=_malloc,_free \
  -o "$OUT_DIR/tinygpt.js"

echo "built -> $OUT_DIR/tinygpt.js  ($(du -h "$OUT_DIR/tinygpt.wasm" | cut -f1) wasm)"
