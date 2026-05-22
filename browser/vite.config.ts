import { defineConfig } from "vite";

// The WebGPU kernels live in ../webgpu (shared Phase 5 location), one level
// above this Vite root — allow the dev server to serve files from there.
export default defineConfig({
  server: { fs: { allow: [".."] } },
});
