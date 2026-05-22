# wasm/ — Phase 4

C++ tensor ops compiled to WebAssembly with Emscripten. This is the CPU backend
for browser training — build it correct before touching WebGPU.

## Files

| File               | Kernel |
| ------------------ | ------ |
| `src/tensor.cpp`      | Flat float32 tensor type + allocation helpers |
| `src/matmul.cpp`      | Matrix multiply, forward + backward |
| `src/layernorm.cpp`   | LayerNorm, forward + backward |
| `src/attention.cpp`   | Causal multi-head self-attention |
| `src/adamw.cpp`       | AdamW optimizer step |

No general autograd — each kernel implements its own forward and backward.

## Build

Baseline:

```bash
emcc src/*.cpp -O3 -s MODULARIZE=1 -s EXPORT_ES6=1 -s ALLOW_MEMORY_GROWTH=1 \
  -o dist/tinygpt.js
```

SIMD (verify it matches the scalar build first):

```bash
emcc src/*.cpp -O3 -msimd128 -s MODULARIZE=1 -s EXPORT_ES6=1 -s ALLOW_MEMORY_GROWTH=1 \
  -o dist/tinygpt.simd.js
```

## Native verification (no Emscripten needed)

The kernels are plain C++ — compile them with the host compiler and run the
finite-difference gradient checks before ever touching Emscripten:

```bash
bash wasm/build_native.sh        # clang++ build + tests/test_wasm_kernels.cpp
```

Each kernel's hand-written backward is checked against a numerical gradient;
18/18 checks currently pass.

## Status

All five kernels are **implemented and natively verified**. The Emscripten build
(`emcc` → `dist/tinygpt.js`) and the Web Worker wiring are the remaining
milestone-5 work; they require the Emscripten SDK. See `../docs/browser_notes.md`.
