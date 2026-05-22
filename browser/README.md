# browser/ — Phase 4

The browser app: main-thread UI + a training Web Worker + a tokenizer + storage.

## Architecture

```
Main thread (main.ts)      UI, file upload, charts, controls
        |  postMessage (dataset + config)  /  TrainingProgress
Worker (worker.ts)         dataset, training loop, sampling, checkpoints
        |  calls
WASM backend (../wasm/)    tensor ops, forward/backward, optimizer
```

Training must never run on the main thread — the UI has to stay responsive.

## Files

| File               | Role |
| ------------------ | ---- |
| `src/main.ts`         | UI controller; spawns the worker, renders progress |
| `src/worker.ts`       | Training loop off the main thread |
| `src/tokenizer.ts`    | Byte-level encode/decode (vocab 256) |
| `src/storage.ts`      | OPFS / IndexedDB checkpoint persistence |
| `src/charts.ts`       | Loss / throughput charts |
| `src/runtime_detect.ts` | Picks backend: webgpu → wasm-simd → wasm |

## Files

| File                  | Role |
| --------------------- | ---- |
| `src/main.ts`         | UI controller; spawns the worker, renders progress |
| `src/worker.ts`       | Training loop off the main thread |
| `src/backend.ts`      | Typed wrapper around the compiled WASM module |
| `src/tokenizer.ts`    | Byte-level encode/decode (vocab 256) |
| `src/charts.ts`       | Canvas loss chart |
| `src/storage.ts`      | OPFS persistence |
| `src/runtime_detect.ts` | Backend capability detection |
| `src/types.ts`        | main ⇆ worker message protocol |

## Run it

```bash
# 1. build the WASM module (needs the Emscripten SDK on PATH)
bash ../wasm/build_wasm.sh           # -> browser/public/tinygpt.{js,wasm}

# 2. install deps and start the dev server
npm install
npm run dev                          # open the printed localhost URL
```

`npm run build` type-checks (`tsc --noEmit`) then bundles to `dist/`.

## Status

Implemented. The app trains a byte-level TinyGPT from scratch in a Web Worker on
the WASM backend; the main thread only handles UI, so it never freezes. The
compiled module is verified from Node by `../tests/smoke_wasm_node.mjs`.

Single-threaded by design. Threaded WASM needs `SharedArrayBuffer` + cross-origin
isolation (`COOP: same-origin`, `COEP: require-corp`) — a later addition.
