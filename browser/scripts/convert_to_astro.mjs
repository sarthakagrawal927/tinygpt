// scripts/convert_to_astro.mjs — one-shot HTML → Astro page conversion.
//
// Reads each legacy HTML entry point, extracts the <style> + <body> content,
// and emits a corresponding .astro page that wraps it in the Default layout.
// Kept around so the conversion is reproducible from the originals; not
// part of the build.
//
// Run from the browser/ dir:  node scripts/convert_to_astro.mjs

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");

function extract(html) {
  const styleMatch = html.match(/<style>([\s\S]*?)<\/style>/);
  const bodyMatch = html.match(/<body[^>]*>([\s\S]*?)<\/body>/);
  const scriptMatch = html.match(/<script type="module" src="([^"]+)"><\/script>/);
  // Strip the trailing inline script from the body — we re-emit it as a
  // top-level Astro <script> so Vite picks it up.
  let body = bodyMatch ? bodyMatch[1] : "";
  if (scriptMatch) {
    body = body.replace(scriptMatch[0], "").trimEnd();
  }
  return {
    style: styleMatch ? styleMatch[1] : "",
    body,
    scriptSrc: scriptMatch ? scriptMatch[1] : null,
  };
}

// Each page declares the layout props it needs (title, description, OG fields,
// theme color). These came from the original <head> blocks verbatim — kept as
// inline literals so the conversion source is self-contained.
const pages = [
  {
    src: "index.html",
    out: "src/pages/index.astro",
    props: {
      title: "TinyGPT — train a real transformer in your browser tab",
      description:
        "Train a real transformer from scratch in your browser tab — 9.7× faster than naive WASM thanks to hand-written WebGPU kernels. 0.8M to 473M params (Memory64), hand-derived backward pass, parity-tested.",
      ogTitle: "TinyGPT — train a real transformer in your browser tab",
      ogDescription:
        "9.7× faster than naive WebAssembly. Hand-written WebGPU kernels, parity-tested to within 1.1% loss drift. 0.8M to 473M params with Memory64. No server, no install.",
      ogUrl: "https://tinygpt.sarthakagrawal.dev",
      twitterTitle: "TinyGPT — 9.7× faster transformer training in a browser tab",
      twitterDescription:
        "GPT-2 from scratch. Python reference + WASM + WebGPU. 9.7× speedup, 1.1% loss drift, 473M-param ceiling via Memory64. Every kernel hand-written and parity-tested.",
      themeColor: "#08090a",
    },
    scriptImport: "/src/main.ts",
  },
  {
    src: "roadmap.html",
    out: "src/pages/roadmap.astro",
    props: {
      title: "TinyGPT — the performance journey",
      description:
        "The performance journey of TinyGPT — every speed lever, what's shipped, what's blocked, and why.",
      ogType: "article",
      ogTitle: "TinyGPT — the performance journey",
      ogDescription:
        "Every speed lever, what's shipped, what's blocked, and why. Scalar → SIMD → threads → WebGPU naive → blocked-4×4. The 9.7× end-to-end speedup, anchored to numbers you can reproduce.",
      twitterTitle: "TinyGPT — the performance journey",
      twitterDescription:
        "Every speed lever in TinyGPT, what's shipped, what's blocked, and why — with reproducible numbers.",
      themeColor: "#08090a",
    },
  },
  {
    src: "devlog.html",
    out: "src/pages/devlog.astro",
    props: {
      title: "TinyGPT — devlog",
      description:
        "Notes from building TinyGPT — kernel measurements, honest negative results, decisions made while AI-pairing.",
      ogType: "article",
      ogTitle: "TinyGPT — devlog: kernel measurements + honest negative results",
      ogDescription:
        "What worked, what didn't. f16 doesn't compound on top of tiling. 8×8 register block lost to 4×4. vec4 wins standalone but a WGSL access-mode mismatch breaks integration — until you find it.",
      twitterTitle: "TinyGPT — devlog",
      twitterDescription:
        "Kernel measurements, honest negative results, decisions made while AI-pairing on a GPT-2 implementation.",
      // Devlog page never set a theme-color in the original HTML, so leave it
      // at the layout default. The page also never preconnected Geist; it
      // uses ui-sans-serif fallback. Disable fonts to preserve that.
      themeColor: "#0a0c10",
      loadFonts: false,
    },
  },
  {
    src: "speedup.html",
    out: "src/pages/speedup.astro",
    props: {
      title: "TinyGPT — the speedup, in one chart",
      description:
        "WebAssembly SIMD vs hand-written WebGPU on the same GPT-2 model — 6.8 s/step down to 0.7 s/step, 9.7× end-to-end, 1.1% loss drift.",
      ogType: "article",
      ogTitle: "TinyGPT — 9.7× faster in one chart",
      ogDescription:
        "WASM SIMD: 6.8 s/step. WebGPU with the blocked-4×4 matmul kernel: 0.7 s/step. Same model, same seed, 1.1% loss drift — pure float-reorder noise.",
      twitterTitle: "TinyGPT — 9.7× faster, in one chart",
      twitterDescription:
        "WASM SIMD vs hand-written WebGPU on the same GPT-2 model. 9.7× end-to-end, 1.1% loss drift.",
      themeColor: "#0a0c10",
      loadFonts: false,
    },
  },
  {
    src: "webgpu-test.html",
    out: "src/pages/webgpu-test.astro",
    props: {
      title: "TinyGPT — WebGPU kernel tests",
      description: "Live WebGPU kernel-parity tests for TinyGPT.",
      themeColor: "#010409",
      loadFonts: false,
    },
    scriptImport: "/src/webgpu-test.ts",
  },
];

// Escape a string for embedding inside a JS backtick template literal in the
// Astro frontmatter. CSS and HTML contain `${` sequences (rare but possible)
// and stray backticks; both must be neutralized.
function backtickEscape(s) {
  return s.replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$\{/g, "\\${");
}

function astroPage(p, html) {
  const { style, body, scriptSrc } = extract(html);
  const scriptImport = p.scriptImport ?? scriptSrc;

  const propsLines = Object.entries(p.props).map(([k, v]) => {
    if (typeof v === "boolean") return `  ${k}: ${v}`;
    return `  ${k}: ${JSON.stringify(v)}`;
  }).join(",\n");

  // Astro parses { } in component children as JSX-style expressions, which
  // breaks for CSS rules and any inline HTML containing literal braces. The
  // workaround used here: keep both the style block and the body markup as
  // strings in the frontmatter, then render them through `set:html`. This
  // preserves the original markup byte-for-byte without any parser ambiguity.
  return `---
// Auto-generated by scripts/convert_to_astro.mjs from ${p.src}.
// Edit the HTML source then re-run the script, or edit this .astro file
// directly — both are valid. Keep IDs/attributes stable for src/main.ts and
// the popover system to keep wiring up correctly.
import Default from "../layouts/Default.astro";

const layoutProps = {
${propsLines},
};

const pageStyle = \`${backtickEscape(style)}\`;

const pageBody = \`${backtickEscape(body)}\`;
---

<Default {...layoutProps}>
  <style is:global slot="head" set:html={pageStyle}></style>
  <Fragment set:html={pageBody} />
${scriptImport ? `  <script>import "${scriptImport}";</script>` : ""}
</Default>
`;
}

for (const p of pages) {
  const html = readFileSync(join(root, p.src), "utf8");
  const out = astroPage(p, html);
  const outPath = join(root, p.out);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, out);
  console.log(`wrote ${p.out} (${(out.length / 1024).toFixed(1)} KB)`);
}
