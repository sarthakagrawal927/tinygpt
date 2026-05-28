# Gallery v1.5 and v2 — build plan

The browser playground ships v1 of the gallery: four bundled `.bin`
checkpoints (Shakespeare, TinyStories, Python, Q&A chat) served from
`browser/public/gallery/`. This doc lays out the next two iterations:
v1.5 (more diverse-data models, same hosting) and v2 (R2-backed, larger
catalog).

Roadmap context: levers 17 (gallery) and 18 (diverse data structures).

## v1.5 — "show what the architecture actually learns"

The four v1 models all train on English prose or English-adjacent text.
The point of the architecture is that the *same* network can learn *any*
local + long-range pattern — but the gallery doesn't make that visible.
v1.5 adds three models that demonstrate structure beyond natural language.

### Models to add

| # | Model            | Corpus                                                | Output proves                             |
|---|------------------|-------------------------------------------------------|-------------------------------------------|
| 5 | Sheet music      | ABC-notation folk tunes (~5-10 MB)                    | the model learns musical phrase structure |
| 6 | CSV / tabular    | Synthetic structured data (1.5 MB)                    | column rhythm, header repetition, value ranges |
| 7 | Chord sheets     | Lyric + chord lines (~1 MB)                           | the chord-above-lyric vertical convention |

Each entry follows the v1 template:

- Same architecture: 12L, d=256, ctx=256, ~9.6M params
- ~5000 steps, sequential under `caffeinate -i`
- Trained via `browser/train_gallery_one.mjs --corpus=<path> --out=<name>
  --prompt="<seed>"`
- Finalized through `browser/finalize_gallery.mjs` which rebuilds
  `manifest.json`

### Why these three

- **Sheet music (ABC notation):** the most surprising — bar lines, key
  signatures, and the repetition of `:|` markers are pure local structure
  with no semantic content. Visitors who *play* an instrument can read the
  output. Highest "wait, really?" payoff per training hour.
- **CSV:** boring but pedagogically valuable. Shows the model picking up
  delimiter rhythm and column conventions. Pairs with a side-by-side
  "input format vs generated output" presentation card.
- **Chord sheets:** two-line vertical structure (chords above, lyrics
  below) tests the model's handling of inter-line alignment. The most
  visually distinctive output.

### What v1.5 explicitly does NOT add

- **Code-in-other-languages** (Go, Rust, JS). Python already represents
  code. Adding three more programming-language models is gallery bloat
  without new pedagogy.
- **Multi-lingual prose.** Same reason — Shakespeare already covers
  English prose. A non-English model would need its own tokenizer
  conversation; that's a bigger project (BPE per language), not a v1.5
  addition.
- **Tiny models** (under 1M params). The gallery is for the *Huge* preset
  by design — switching architectures mid-gallery confuses the "same model,
  different corpus" framing.

### Prompt prefills per model

`manifest.json` already supports a `prompt` field used by `loadGalleryCard`
in `main.ts` to pre-fill the prompt input. Suggested defaults:

```
sheetmusic    "T:"                       # ABC tune-title header
csv           "id,name,value\n1,"        # invites a row continuation
chords        "[Verse 1]\n"              # invites verse generation
```

### Hosting decision for v1.5

Stay bundled in `browser/public/gallery/`. Adding three models at ~18MB
each = ~54MB extra. With v1's 4 models (~70MB total), v1.5 brings the
public directory to ~125MB. Cloudflare Pages's bundle limit is 25MB per
file but no project-wide cap (verified). Caching works via the
`.bin`-extension rename trick documented in `README.md`. Defer R2 until
v2.

## v2 — R2-backed, expandable catalog

v1.5 hits the bundled ceiling. v2 moves to Cloudflare R2 with the same
manifest pattern but the models live behind a CDN.

### What changes

- **Bundle stays small.** Only the manifest ships in
  `browser/public/gallery/`. Model `.bin` files move to R2.
- **Manifest entries gain a `url` field.** Resolved at load time. Same
  `loadGalleryCard` path with one extra fetch.
- **OPFS cache stays as-is.** The existing `saveCachedGalleryModel` /
  `loadCachedGalleryModel` flow in `storage.ts` already caches by id;
  R2 URLs are just the cold-fetch source.
- **Bandwidth.** R2 charges $0.015/GB-month storage; zero egress to
  Pages. For a 100MB catalog and 10K cold-pulls/month, that's $0.50
  storage + $0 egress. Free tier covers a fair amount.

### Catalog targets for v2

Open the gallery to ~10-15 models without bloating the bundle:

- v1's 4 (Shakespeare, TinyStories, Python, Q&A chat)
- v1.5's 3 (sheet music, CSV, chord sheets)
- 3-5 community submissions: a "submit a corpus, get a model" flow
  where you PR a corpus to `browser/community_corpora/`, GitHub Actions
  trains it via the same Playwright trainer, and the bot pushes the
  checkpoint to R2 + opens a PR to update the manifest.

The community flow is the structural feature that makes v2 worth doing —
v1.5 you can ship by hand-running three more `train_gallery_one.mjs`
invocations. v2 lets the gallery grow without your wall time.

### Implementation sketch

```
browser/
├── public/gallery/manifest.json        # has v1 + v1.5 entries with `url` to R2
├── src/main.ts                         # loadGalleryCard fetches `url` if present
├── src/storage.ts                      # already caches by id, unchanged
└── community_corpora/                  # PR-able corpus directory
    ├── README.md                       # corpus format + submission rules
    └── <yourname>.txt                  # submitted corpus

.github/workflows/
└── train_community_corpus.yml          # runs train_gallery_one.mjs on
                                        # PR-added corpora, uploads .bin
                                        # to R2, opens a manifest-update PR
```

### What v2 explicitly does NOT do

- **Inference in the cloud.** Models still run in the browser. R2 is
  hosting only.
- **A login.** Community submissions are GitHub PRs against a public
  repo, not a user account on the site.
- **Comment / vote / rating.** Engagement bait that erodes the "this is
  a serious technical artifact" framing. Keep the gallery curated.

## Mac app crossover

When the Mac app ships (see `mac_app_plan.md`), the same R2-backed
manifest powers its gallery dialog. Browser and Mac app both pull
identical `.tinygpt` files from R2; the local OPFS cache (browser) and
`~/Library/Caches/TinyGPT/` (Mac) keep cold-fetches rare.

No file-format change needed — `.tinygpt` is already the shared
contract. The manifest's `url` field is just as resolvable from Swift's
`URLSession` as from `fetch`.

## Sequencing

Don't ship v1.5 and v2 together.

1. **v1.5 first.** Three more training runs, manifest update, ship.
   ~2-3 days of wall time (mostly training) but maybe 4 hours of human
   work. Validates the "show the architecture, not just English" pitch.
2. **R2 setup separately.** Create the bucket, upload v1's existing
   `.bin` files, manually update the manifest's `url` fields, verify
   cold-load + cache hit. ~half a day. This is the v2 plumbing without
   the community flow.
3. **Community submission flow last.** This is a real feature with
   real engineering — GitHub Actions, corpus validation, R2 upload
   credentials, manifest PR automation. ~1 week. Worth doing only if v1
   and v1.5 get measurable usage; otherwise it's infrastructure for
   nobody.

## Open questions for you to answer

- **R2 bucket location** — pick a region close to most traffic. EU vs.
  US? Check PostHog data on the next ~week of post-launch traffic
  before deciding.
- **Community corpus license.** What does a PR-er agree to? Public
  domain only? CC-BY? Worth a one-page CONTRIBUTING.md before the
  community flow opens.
- **Mac app priority vs. gallery v2 priority.** Both are listed in
  `roadmap.astro`. The Mac app is the bigger leverage (lever 20 ~30×
  throughput unlock); v2 gallery is the bigger surface-area expansion
  (more demos, more shareable links). Picking one to focus on first
  matters — see `decision_log.md` if you want to formalise.
