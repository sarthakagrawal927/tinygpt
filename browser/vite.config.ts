import { defineConfig } from "vite";

// The WebGPU kernels live in ../webgpu (shared Phase 5 location), one level
// above this Vite root — allow the dev server to serve files from there.
// Three HTML entry points: the playground, the WebGPU kernel test page,
// and the static performance-journey roadmap.
export default defineConfig({
  server: {
    fs: { allow: [".."] },
    // Cross-origin isolation enables SharedArrayBuffer, which the
    // multi-threaded WASM build needs. Production sets these via
    // browser/public/_headers (Cloudflare Pages); dev mirrors them here.
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  build: {
    rollupOptions: {
      input: {
        main: "index.html",
        "webgpu-test": "webgpu-test.html",
        roadmap: "roadmap.html",
      },
    },
  },
});
