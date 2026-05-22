/**
 * backend.ts — typed wrapper around the WASM TinyGPT module (Phase 4).
 *
 * The kernels + model are compiled by wasm/build_wasm.sh into the ES module
 * browser/public/tinygpt.js. This file loads that module and exposes its C-ABI
 * (model.h) as a small typed class, handling the JS<->WASM memory marshalling
 * so the Worker can stay readable.
 *
 * Guide: docs/browser_notes.md ("WASM backend")
 */

/** The subset of the Emscripten module surface we rely on. */
interface WasmModule {
  HEAPU8: Uint8Array;
  HEAPF32: Float32Array;
  _malloc(bytes: number): number;
  _free(ptr: number): void;
  cwrap(name: string, ret: string | null, args: string[]): (...a: number[]) => number;
}

type Factory = () => Promise<WasmModule>;

/** Where the compiled module is served from (see wasm/build_wasm.sh output). */
const MODULE_URL = "/tinygpt.js";

export class TinyGptBackend {
  private constructor(
    private readonly m: WasmModule,
    private readonly fns: {
      create: (...a: number[]) => number;
      free: (...a: number[]) => number;
      numParams: (...a: number[]) => number;
      step: (...a: number[]) => number;
      setData: (...a: number[]) => number;
      trainStep: (...a: number[]) => number;
      evalLoss: (...a: number[]) => number;
      generate: (...a: number[]) => number;
      matmul: (...a: number[]) => number;
      stateBytes: (...a: number[]) => number;
      exportState: (...a: number[]) => number;
      importState: (...a: number[]) => number;
    },
  ) {}

  /** Load and instantiate the compiled WASM module. */
  static async load(): Promise<TinyGptBackend> {
    const mod = (await import(/* @vite-ignore */ MODULE_URL)) as { default: Factory };
    const m = await mod.default();
    const N = "number";
    return new TinyGptBackend(m, {
      create: m.cwrap("tg_model_create", N, [N, N, N, N, N, N, N]),
      free: m.cwrap("tg_model_free", null, [N]),
      numParams: m.cwrap("tg_model_num_params", N, [N]),
      step: m.cwrap("tg_model_step", N, [N]),
      setData: m.cwrap("tg_set_data", null, [N, N, N, N]),
      trainStep: m.cwrap("tg_train_step", N, [N, N, N, N]),
      evalLoss: m.cwrap("tg_eval", N, [N, N, N, N]),
      generate: m.cwrap("tg_generate", N, [N, N, N, N, N, N, N, N]),
      matmul: m.cwrap("matmul_forward", null, [N, N, N, N, N, N]),
      stateBytes: m.cwrap("tg_state_bytes", N, [N]),
      exportState: m.cwrap("tg_export_state", null, [N, N]),
      importState: m.cwrap("tg_import_state", null, [N, N]),
    });
  }

  /** Raw C = A @ B via the WASM matmul kernel — the WebGPU parity reference. */
  matmul(a: Float32Array, b: Float32Array, M: number, K: number, N: number): Float32Array {
    const aPtr = this.m._malloc(M * K * 4);
    const bPtr = this.m._malloc(K * N * 4);
    const cPtr = this.m._malloc(M * N * 4);
    try {
      this.m.HEAPF32.set(a, aPtr >> 2);
      this.m.HEAPF32.set(b, bPtr >> 2);
      this.fns.matmul(aPtr, bPtr, cPtr, M, K, N);
      return this.m.HEAPF32.slice(cPtr >> 2, (cPtr >> 2) + M * N);
    } finally {
      this.m._free(aPtr);
      this.m._free(bPtr);
      this.m._free(cPtr);
    }
  }

  /** Copy a byte buffer into the WASM heap; returns a pointer to free later. */
  private push(bytes: Uint8Array): number {
    const ptr = this.m._malloc(Math.max(1, bytes.length));
    this.m.HEAPU8.set(bytes, ptr);
    return ptr;
  }

  createModel(cfg: {
    ctx: number;
    layers: number;
    heads: number;
    dModel: number;
    dMlp: number;
    seed: number;
  }): TinyGptModel {
    const handle = this.fns.create(
      256, cfg.ctx, cfg.layers, cfg.heads, cfg.dModel, cfg.dMlp, cfg.seed,
    );
    if (handle === 0) throw new Error("tg_model_create failed (d_model % heads != 0?)");
    return new TinyGptModel(this.m, this.fns, this.push.bind(this), handle);
  }
}

/** A live model handle. One per training run. */
export class TinyGptModel {
  constructor(
    private readonly m: WasmModule,
    private readonly fns: {
      free: (...a: number[]) => number;
      numParams: (...a: number[]) => number;
      step: (...a: number[]) => number;
      setData: (...a: number[]) => number;
      trainStep: (...a: number[]) => number;
      evalLoss: (...a: number[]) => number;
      generate: (...a: number[]) => number;
      stateBytes: (...a: number[]) => number;
      exportState: (...a: number[]) => number;
      importState: (...a: number[]) => number;
    },
    private readonly push: (b: Uint8Array) => number,
    private readonly handle: number,
  ) {}

  numParams(): number {
    return this.fns.numParams(this.handle);
  }

  step(): number {
    return this.fns.step(this.handle);
  }

  /** Attach a byte-token corpus, split train/val by `trainFrac`. */
  setData(tokens: Uint8Array, trainFrac: number): void {
    const ptr = this.push(tokens);
    try {
      this.fns.setData(this.handle, ptr, tokens.length, trainFrac);
    } finally {
      this.m._free(ptr);
    }
  }

  /** One AdamW step on a random batch; returns the batch loss. */
  trainStep(batchSize: number, lr: number, gradClip: number): number {
    return this.fns.trainStep(this.handle, batchSize, lr, gradClip);
  }

  /** Average loss over `nBatches`. split: 0 = train, 1 = val. */
  evalLoss(split: 0 | 1, batchSize: number, nBatches: number): number {
    return this.fns.evalLoss(this.handle, split, batchSize, nBatches);
  }

  /** Autoregressive generation. temperature <= 0 is greedy. */
  generate(
    prompt: Uint8Array,
    maxNew: number,
    temperature: number,
    topK: number,
    seed: number,
  ): Uint8Array {
    const promptPtr = this.push(prompt);
    const outPtr = this.m._malloc(Math.max(1, maxNew));
    try {
      const n = this.fns.generate(
        this.handle, promptPtr, prompt.length, outPtr, maxNew, temperature,
        topK, seed,
      );
      return this.m.HEAPU8.slice(outPtr, outPtr + n);
    } finally {
      this.m._free(promptPtr);
      this.m._free(outPtr);
    }
  }

  /** Serialise weights + AdamW moments + step for checkpointing. */
  exportState(): Uint8Array {
    const bytes = this.fns.stateBytes(this.handle);
    const ptr = this.m._malloc(bytes);
    try {
      this.fns.exportState(this.handle, ptr);
      return this.m.HEAPU8.slice(ptr, ptr + bytes);
    } finally {
      this.m._free(ptr);
    }
  }

  /** Load a state blob from exportState() — model config must match. */
  importState(state: Uint8Array): void {
    const ptr = this.push(state);
    try {
      this.fns.importState(this.handle, ptr);
    } finally {
      this.m._free(ptr);
    }
  }

  free(): void {
    this.fns.free(this.handle);
  }
}
