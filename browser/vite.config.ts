import { defineConfig } from "vite";

// The WebGPU kernels live in ../webgpu (shared Phase 5 location), one level
// above this Vite root — allow the dev server to serve files from there.
//
// Every HTML entry point MUST be listed in rollupOptions.input below or
// Vite silently drops it from the production build (devlog and speedup were
// added later in this project's history and previously did not ship — same
// HTML file existed but the deployed site 404'd on /devlog and /speedup).
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
        devlog: "devlog.html",
        speedup: "speedup.html",
      },
    },
  },
});
