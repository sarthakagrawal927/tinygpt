// astro.config.mjs — TinyGPT browser frontend.
//
// Astro wraps Vite, so the existing `?raw` WGSL imports from ../webgpu and the
// `new Worker(new URL("./worker.ts", import.meta.url))` pattern in src/main.ts
// continue to work via the standard Vite resolver. The build output directory
// is `dist/` (Astro default), which matches what Cloudflare Pages expects per
// docs/deploy.md, so the deploy contract is unchanged.
//
// Cross-origin isolation headers (COOP/COEP) MUST be set on the dev server —
// without them SharedArrayBuffer is unavailable and the multi-threaded WASM
// build fails to initialize. Production sets the same headers via
// browser/public/_headers (Cloudflare Pages copies that file verbatim).
//
// MDX integration is wired up so future devlog entries can be authored as
// `.mdx` files with embedded interactive Astro components. The existing
// devlog.html is migrated as a static page in this pass — converting its
// content to MDX is a future refactor, not gated on this turn.

import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";

export default defineConfig({
  // dist/ is the default Astro output dir; declared here for documentation.
  outDir: "./dist",

  // build.format = "file" emits dist/roadmap.html instead of
  // dist/roadmap/index.html, matching the legacy Vite output shape so the
  // existing audit scripts (which hit /roadmap.html) and any external
  // links/social cards keep resolving.
  build: { format: "file" },

  integrations: [mdx()],

  server: {
    // Dev-server COOP/COEP mirror of public/_headers for production parity.
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },

  vite: {
    server: {
      // The WGSL kernels live in ../webgpu (shared Phase 5 location), one
      // level above this Astro root — allow the dev server to serve files
      // from there via the ?raw imports in webgpu/kernels.ts / ops.ts.
      fs: { allow: [".."] },
      headers: {
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
      },
    },
    preview: {
      headers: {
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
      },
    },
  },
});
