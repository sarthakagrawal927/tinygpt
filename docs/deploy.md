# Deploying the playground

The browser app in `browser/` is a Vite static build — no server, no API. It
loads a pre-compiled WASM module from `browser/public/`, so the deploy
environment does **not** need Emscripten; it only needs Node to run `vite build`.

Target: **`tinygpt.sarthakagrawal.dev`**, on Cloudflare Pages.

## ⚠ Deploy-gating prerequisite — the WASM artifacts must be in the repo at build time

`browser/public/tinygpt.js`, `tinygpt.wasm`, `tinygpt64.js`, and `tinygpt64.wasm`
are produced by Emscripten (`bash wasm/build_wasm.sh` + `bash wasm/build_wasm64.sh`)
and are listed in `.gitignore` by default. Cloudflare Pages does a fresh
`git clone` + `npm run build`; if these files aren't in the cloned tree, the
production site 404s on them and the playground fails to initialize.

Three resolutions, ranked:

1. **Commit the artifacts (recommended).** They're small (~80 KB each, ~230 KB
   total) and change rarely (only when the C++ kernels change). One-shot fix:

   ```sh
   git add -f browser/public/tinygpt.js browser/public/tinygpt.wasm \
              browser/public/tinygpt64.js browser/public/tinygpt64.wasm
   git commit -m "deploy: commit the compiled WASM artifacts"
   ```

   The `-f` overrides the `.gitignore` entries. Then on every C++ change,
   rebuild and commit the new artifacts as part of the same commit.

2. **Install Emscripten in Pages build environment.** CF Pages allows custom
   build commands; you'd `npm install && bash wasm/install_emsdk.sh && bash wasm/build_wasm.sh && bash wasm/build_wasm64.sh && npm run build`.
   Multiplies build time by ~3-5× and the emsdk install is finicky in CI.
   Not recommended unless option 1 becomes painful.

3. **Pre-built artifacts uploaded out-of-band** to R2 or similar, then a
   build-time fetch step. Most complex; only worth it if the WASM grows past
   1 MB and git becomes a real burden.

## Cloudflare Pages — one-time setup

In the Cloudflare dashboard:

1. **Workers & Pages → Pages → Create application → Connect to Git**
2. Select `sarthakagrawal927/tinygpt`.
3. **Build configuration:**
   - Production branch: `main`
   - Framework preset: *None* (Vite isn't in the preset list, but the explicit
     settings below work)
   - Root directory: `browser`
   - Build command: `npm install && npm run build`
   - Build output directory: `dist`
4. **Environment variables:** none required.
5. Save and Deploy.

Cloudflare will clone the whole repo, `cd` into `browser/`, install dependencies
and build. First build is ~1 minute; subsequent builds are cached.

## Custom domain

After the first successful deploy:

1. In the Pages project: **Custom domains → Set up a custom domain**.
2. Add `tinygpt.sarthakagrawal.dev`. CF Pages will then sit in **Verifying**
   state and show you a CNAME to add.
3. **Add the CNAME manually**, even though `sarthakagrawal.dev` is on the
   same Cloudflare account. The "added automatically" claim that used to
   live in this section was wrong — auto-CNAME only happens when you set up
   the custom domain at the *same time* as the Pages project, via the
   create-app flow. Adding a custom domain post-deploy requires the DNS
   record manually:

   - **Zone**: `sarthakagrawal.dev` → DNS → Records → Add record
   - **Type**: `CNAME`
   - **Name**: `tinygpt` (just the subdomain, not the full hostname)
   - **Target**: the value CF Pages shows (e.g. `tinygpt.pages.dev`)
   - **Proxy**: orange-cloud Proxied
   - **TTL**: Auto

4. Back in Pages → Custom domains, click **Check DNS records**. It flips to
   **Active** in 30–60 s. SSL is then automatic.

After a minute or two: <https://tinygpt.sarthakagrawal.dev>.

## What ships

The Vite build produces five HTML entry points (all listed in
`browser/vite.config.ts → build.rollupOptions.input`; any HTML file
not in that list is silently dropped from the production build):

- `/` — the playground (train, sample, swap backends).
- `/roadmap` — the performance journey, levers, and the speed-evolution
  chart. Each measured bar is reproducible from the bench button.
- `/devlog` — long-form notes from the AI-pairing session that produced
  the kernel work, including the negative results.
- `/speedup` — punchy "9.7× before/after" chart, social-shareable.
- `/webgpu-test` — the live kernel-parity diagnostic (30 GPU kernels
  checked against a reference, plus the GPU overfit gate). Useful as a
  "see it self-verify" link from the case study; keep it shipped.

The deployed assets total around 600 KB (WASM + JS chunks).

## After deploy — update the case study

The TinyGPT case study on the portfolio
(`sarthakagrawal927/portfolio: src/content/work/tinygpt.mdx`) has `repo:` set
but `demo:` deliberately omitted until the URL is live. Add:

```yaml
demo: 'https://tinygpt.sarthakagrawal.dev'
```

and the case-study header will render a `live demo ↗` link.

## Local sanity check before deploy

```sh
cd browser
npm install
npm run build      # produces browser/dist/
npm run preview    # serves dist/ on http://localhost:4173
```

If `npm run e2e` passes (which it does on `main`), the production build will
behave the same — the e2e drives the built bundle, not the dev server.
