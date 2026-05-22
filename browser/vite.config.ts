import { defineConfig } from "vite";

// The WebGPU kernels live in ../webgpu (shared Phase 5 location), one level
// above this Vite root — allow the dev server to serve files from there.
// Two HTML entry points: the app, and the WebGPU kernel test page.
export default defineConfig({
  server: { fs: { allow: [".."] } },
  build: {
    rollupOptions: {
      input: {
        main: "index.html",
        "webgpu-test": "webgpu-test.html",
      },
    },
  },
});
