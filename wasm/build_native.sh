#!/usr/bin/env bash
# build_native.sh — compile the WASM kernels + model with the host compiler and
# run the correctness tests. Needs only clang/g++ — NOT Emscripten — so all of
# the training math is verified before the browser build exists.
#
# The Emscripten (emcc) build that produces dist/tinygpt.js is a separate, later
# step; see wasm/build_wasm.sh and wasm/README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="${TMPDIR:-/tmp}"
CXX="${CXX:-clang++}"
FLAGS=(-std=c++17 -O2 -Wall -Wextra)

KERNELS=(
  "$ROOT"/wasm/src/tensor.cpp
  "$ROOT"/wasm/src/matmul.cpp
  "$ROOT"/wasm/src/layernorm.cpp
  "$ROOT"/wasm/src/attention.cpp
  "$ROOT"/wasm/src/adamw.cpp
)

echo "compiling kernel tests with $CXX ..."
"$CXX" "${FLAGS[@]}" "${KERNELS[@]}" \
  "$ROOT"/tests/test_wasm_kernels.cpp -o "$TMP"/tinygpt_test_kernels

echo "compiling model tests with $CXX ..."
"$CXX" "${FLAGS[@]}" "${KERNELS[@]}" "$ROOT"/wasm/src/model.cpp \
  "$ROOT"/tests/test_wasm_model.cpp -o "$TMP"/tinygpt_test_model

echo
echo "=== kernel tests ==="
"$TMP"/tinygpt_test_kernels
echo
echo "=== model tests ==="
"$TMP"/tinygpt_test_model
