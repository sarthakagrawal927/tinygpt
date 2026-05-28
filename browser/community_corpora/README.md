# Community corpora

A staging area for community-submitted training corpora. Files dropped here
are picked up by the gallery training workflow, trained into `.tinygpt`
checkpoints, and (if approved) added to the live gallery manifest.

**This is the gallery v2 anchor feature** — see `docs/gallery_v2_plan.md`.
It is currently a placeholder; the training workflow lands when v2 ships.

## Submitting a corpus

1. Fork the repo and add your corpus as a single UTF-8 plain-text file under
   `browser/community_corpora/<your-id>.txt`. Use a kebab-case id; this also
   becomes the gallery card's `id`.
2. Add a sibling YAML file `browser/community_corpora/<your-id>.yaml` with
   the metadata schema below.
3. Open a PR titled `community-corpus: <your-id>`.
4. Once merged, the GitHub Actions workflow trains a 5000-step model on the
   corpus, uploads the checkpoint to R2, and opens a follow-up PR to add the
   new entry to `browser/public/gallery/manifest.json`.

## Constraints

- **Size**: 0.5–5 MB of UTF-8 text. Smaller than 500 KB overfits; larger than
  5 MB exceeds the workflow's wall-time budget.
- **Encoding**: UTF-8, Unix line endings. No BOM.
- **License**: the corpus must be public-domain or under a permissive license
  (CC0, CC-BY, MIT, Apache 2.0, public domain dedication). Provide the SPDX
  identifier in the metadata YAML. PRs without a clear license are closed
  without merging.
- **No copyrighted material you don't own.** This includes scraped books,
  song lyrics, paywalled articles, code under restrictive licenses. The
  bar is "would the original author be fine with this corpus being trained
  on and the model distributed publicly."
- **No personal data.** No real names, emails, addresses, IDs.
- **No content that violates the Code of Conduct** in the repo root.

## Metadata schema (`<your-id>.yaml`)

```yaml
name: "My Corpus"                    # display name in the gallery card
icon: "🎵"                            # single emoji
blurb: "One-line description"        # what the model learned
corpus: "Source name"                # human-readable corpus origin
corpusUrl: "https://..."             # link to the canonical source
license: "CC0-1.0"                   # SPDX identifier
prompt: "Once upon a time"           # starting text the model expects
submitter: "github-username"         # for credit in the gallery card
```

The training workflow fills in the rest (`paramCount`, `trainLoss`, `steps`,
`sample`, `fileBytes`, `gpuBytes`, `cdnVersion`, `url`) and opens the
manifest-update PR.

## When the workflow lands

Phase 1 (manual): PR author trains locally via
`browser/train_gallery_one.mjs --corpus=community_corpora/<id>.txt --out=<id>`
and includes the resulting `.bin` in the PR. Reviewer uploads to R2 by hand
and updates the manifest.

Phase 2 (automated): GitHub Actions runs the trainer, uploads to R2 with
`CLOUDFLARE_R2_*` repo secrets, and opens a manifest-update PR.

See `docs/gallery_v2_plan.md` for the rollout sequence.
