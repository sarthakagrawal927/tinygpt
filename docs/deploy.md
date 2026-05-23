# Deploying the playground

The browser app in `browser/` is a Vite static build — no server, no API. It
loads a pre-compiled WASM module from `browser/public/`, so the deploy
environment does **not** need Emscripten; it only needs Node to run `vite build`.

Target: **`tinygpt.sarthakagrawal.dev`**, on Cloudflare Pages.

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
2. Add `tinygpt.sarthakagrawal.dev`.
3. Because `sarthakagrawal.dev` already lives on Cloudflare (the main site is
   deployed there), the CNAME is added automatically. SSL is automatic.

After a minute or two: <https://tinygpt.sarthakagrawal.dev>.

## What ships

The Vite build produces two HTML entry points:

- `/` — the playground (train, sample, swap backends).
- `/webgpu-test` — the live kernel-parity diagnostic (24 GPU kernels checked
  against a reference, plus the GPU overfit gate). Useful as a "see it
  self-verify" link from the case study; keep it shipped.

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
