# GitHub data integration

`tinygpt fetch-github` pulls structured training data from GitHub for
the code-specialist agent track (debugger, reviewer, commit-message
generator). It is the GitHub-side counterpart to
`tinygpt download-dataset` (HuggingFace Hub).

> **Why this matters.** HuggingFace gives you pre-curated datasets;
> GitHub gives you the in-the-wild signal that nobody has packaged yet.
> Issue→PR pairs are *the* training data for a debugger agent: "user
> reported X, here is the diff that fixed it". Review-comment pairs are
> the training data for a PR-reviewer agent. Commit-message pairs train
> a "write a commit message for this diff" model. All three live on
> GitHub, all three are huge, and all three were missing from tinygpt's
> data pipeline before this module.

---

## Quick start

```bash
# Set your token (5000 req/h vs 60 without — huge difference).
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx

# Issue→PR pairs from a smallish repo, capped at 50 records.
tinygpt fetch-github pallets/flask --kind issues-prs --limit 50

# Commits from rust-lang/rust since 2024 — commit-message training.
tinygpt fetch-github rust-lang/rust --kind commits \
    --since 2024-01-01T00:00:00Z --limit 5000 \
    --out rust.commits.jsonl

# PR review comments — reviewer-agent training signal.
tinygpt fetch-github huggingface/transformers --kind reviews --limit 500

# Browse the curated GitHub recipes for a specialist.
tinygpt list-datasets --specialist debugger
# (will list HF datasets first, then the curated GitHub recipes)
```

Output lands in the JSONL file passed to `--out` (or
`./<owner>__<repo>.<kind>.jsonl` by default).

---

## CLI flags

| Flag | Meaning | Default |
|------|---------|---------|
| `<owner/repo>` | positional repo to fetch from | required |
| `--kind` | `issues-prs` \| `reviews` \| `commits` | `issues-prs` |
| `--label` | GitHub issue label filter | `bug` (for issues-prs only) |
| `--state` | `open` \| `closed` \| `all` | `closed` |
| `--since` | ISO 8601 timestamp; updated-after | unset |
| `--max-diff-bytes` | truncate diffs above N chars | `10000` |
| `--limit` | max records to emit | `1000` |
| `--out` | output JSONL path | `./<repo>.<kind>.jsonl` |
| `--multi-repo` | newline-separated repo list to aggregate | unset |
| `--resume` | skip records already present in `--out` | off |
| `--dry-run` | print plan, no fetch | off |

---

## Output formats

All three record kinds use the same outer shape so they drop into
`tinygpt sft` without extra adapters:

```json
{ "instruction": "...", "response": "...", "metadata": { ... } }
```

`metadata` carries provenance (repo, issue/PR/commit number, labels,
`kind`) so downstream filtering can dedupe / weight / license-check.
`tinygpt sft` ignores `metadata` and consumes only
`instruction` + `response`.

### `issues-prs` — bug fix training

```json
{
  "instruction": "<issue title>\n\n<issue body>",
  "response": "<PR description>\n\n--- diff ---\n<unified diff (possibly truncated)>",
  "metadata": {
    "repo": "owner/repo",
    "issue_number": 123,
    "pr_number": 124,
    "labels": ["bug", "regression"],
    "kind": "issue-pr",
    "files_changed": 3
  }
}
```

### `reviews` — PR reviewer training

```json
{
  "instruction": "review this code:\n<diff hunk>",
  "response": "<reviewer's comment>",
  "metadata": {
    "repo": "owner/repo",
    "pr_number": 124,
    "comment_id": 8675309,
    "file": "src/foo.py",
    "kind": "review"
  }
}
```

### `commits` — commit message training

```json
{
  "instruction": "<diff>",
  "response": "<commit message>",
  "metadata": {
    "repo": "owner/repo",
    "sha": "abc123...",
    "files_changed": 5,
    "kind": "commit"
  }
}
```

---

## GitHub token setup

GitHub's REST API has two rate limits:

* **Without a token**: 60 requests per hour per IP. This is essentially
  unusable — even one repo's worth of issue→PR pairs typically needs
  hundreds of requests (one for the issue, one for the timeline, two
  for the PR).
* **With a token**: 5000 requests per hour per user.

Create a token at <https://github.com/settings/tokens> and set:

```bash
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

Scopes:
* `public_repo` is enough for the curated recipes (all public).
* `repo` if you also want to fetch from private repos you have access to.

The tool reads the env var via `ProcessInfo.processInfo.environment`. It
runs without a token but prints a clear warning and you will hit the
60/h ceiling fast.

### Rate-limit handling

The client reads two response headers:

* `X-RateLimit-Remaining` — when this drops below 5, the next request
  sleeps until reset.
* `X-RateLimit-Reset` — unix timestamp when the window rolls over.

A 403 with `"rate limit"` in the body or any 429 is surfaced as
`GHError.rateLimited` with the reset time — the CLI prints a clear
message rather than throwing a raw HTTP error.

---

## Issue → PR linkage heuristic

GitHub does not have a first-class "this PR closes that issue" pointer
in REST (it does in GraphQL, but we don't depend on that). We resolve
the linkage in two layers:

1. **Timeline events** — `GET /repos/{owner}/{repo}/issues/{n}/timeline`
   returns a stream of events. For each `cross-referenced` event whose
   `source.issue.pull_request` is present, we extract the PR number
   from `source.issue.pull_request.url`. This is the high-recall path;
   it picks up PRs whether they say "Fixes #" or not, as long as a PR
   referenced the issue.
2. **Body regex fallback** — if no timeline hit, we scan the issue
   body for `Closes #N` / `Fixes #N` / `Resolves #N` (and the
   `closed:`, `fix`, `resolve` variants), and the same with
   `owner/repo#N`. We accept only the *current* repo's references to
   avoid noise from cross-repo links.

If neither path resolves, the record is skipped and counted in
`Stats.skippedNoPR`. Closed-by-direct-commit issues (no PR involved,
common in projects with merge-commit workflows) are dropped — they're
not useful training signal for a PR-writer agent.

---

## Resume + caching

* Cache root: `~/.cache/tinygpt/github/<owner>/<repo>/`. Override with
  `TINYGPT_GITHUB_CACHE`.
* The cache directory is created on demand but currently not populated
  with per-record JSON blobs — the resume path works by reading back
  `--out` and rebuilding the set of already-written record ids. Future
  work can add a per-request HTTP cache layer (an
  `If-None-Match` ETag round-trip would let us re-fetch a repo cheaply).
* Re-running with `--resume` opens the output in append mode and skips
  ids whose records are already in the file. Ids are constructed from
  the metadata block:
    * commits: `commit:<repo>@<sha>`
    * reviews: `review:<repo>#<pr>c<comment_id>`
    * issue-prs: `issuepr:<repo>#<issue>`
* Without `--resume`, the output file is truncated.

---

## Curated recipes

`tinygpt list-datasets --specialist debugger` (or `--specialist code`)
prints both the HF datasets *and* the GitHub recipes — a hand-picked
set of repos that have well-labelled bug trackers, "Fixes #N" PR
discipline, and permissive licenses. The shortlist lives in
`native-mac/Sources/TinyGPTData/DatasetRegistry.swift` under
`GitHubRecipes.all`.

Current entries (subject to growth):

| Repo | Language | Recommended kinds | License |
|------|----------|-------------------|---------|
| `pytorch/pytorch` | python/c++ | issues-prs, commits | BSD-3 |
| `huggingface/transformers` | python | issues-prs, reviews, commits | Apache-2 |
| `tensorflow/tensorflow` | python/c++ | issues-prs, commits | Apache-2 |
| `numpy/numpy` | python/c | issues-prs, commits | BSD-3 |
| `scikit-learn/scikit-learn` | python | issues-prs | BSD-3 |
| `rust-lang/rust` | rust | issues-prs, reviews, commits | MIT/Apache-2 |
| `tokio-rs/tokio` | rust | issues-prs, commits | MIT |
| `django/django` | python | issues-prs, commits | BSD-3 |
| `pallets/flask` | python | issues-prs | BSD-3 |
| `pandas-dev/pandas` | python | issues-prs, commits | BSD-3 |
| `golang/go` | go | issues-prs, commits | BSD-3 |
| `nodejs/node` | javascript/c++ | issues-prs, reviews | MIT |
| `vuejs/core` | typescript | issues-prs | MIT |

Adding a recipe: edit `GitHubRecipes.all` in `DatasetRegistry.swift`.
Pick repos with:

* an explicit "bug" label (or equivalent) on the issue tracker;
* PR descriptions that mention "Fixes #N" so the body-regex fallback
  catches what the timeline misses;
* a permissive open-source license (BSD / MIT / Apache-2 / similar);
* enough closed bug issues to be worth fetching (~hundreds minimum).

---

## Caveats

* **Private repos.** Need a token with `repo` scope. The 404 you get
  without one looks the same as "repo doesn't exist", which is
  intentional on GitHub's side — we surface it as
  `GHError.notFound(url:)`.
* **Deleted issues / force-pushed PRs.** A PR that was force-pushed to
  rewrite history will still serve a diff via
  `Accept: application/vnd.github.diff`, but the diff reflects the
  *current* HEAD, not the diff that closed the issue. We can't detect
  this reliably; treat the dataset as "what the PR looks like *now*".
* **Mega-PRs.** PRs with thousands of files get truncated to
  `--max-diff-bytes` (default 10 KB). A truncation marker is appended
  so downstream training never silently learns from cut-off context.
* **Binary file changes.** `patch` is absent on files/* records for
  binaries; we skip those. The unified diff endpoint
  (`application/vnd.github.diff`) emits a `Binary files differ` line
  which gets included verbatim.
* **GraphQL would be cheaper.** The REST timeline + per-PR fetch is
  N+1: one issue list call → one timeline call per issue → one PR call
  per pair. A GraphQL query could fetch all of this in one round-trip.
  We're on REST for now to keep the dependency surface zero
  (`URLSession` only) — GraphQL is a follow-up.
* **Search vs list.** We use the list endpoints
  (`/repos/{o}/{r}/issues`) rather than `/search/issues` because the
  list endpoints don't count against the 30 req/min search-API
  sub-limit. The downside: list returns issues *and* PRs together
  (PRs are issues with a `pull_request` member); we filter PRs out
  for issue→PR scanning.
* **`since` semantics.** GitHub's `since` parameter filters by
  *updated_at*, not *created_at*. An issue from 2018 that got a new
  comment yesterday will appear in a `--since 2024-01-01` fetch.
* **Cross-repo "Fixes" links** are intentionally rejected by the body
  regex. Mixing PRs from different repos into one training corpus
  poisons the metadata.

---

## File layout

```
native-mac/Sources/TinyGPTData/GitHubAPI.swift      REST client (URLSession + pagination + rate-limit)
native-mac/Sources/TinyGPTData/GitHubCorpus.swift   issue→PR / reviews / commits extractor
native-mac/Sources/TinyGPTData/DatasetRegistry.swift  GitHubRecipes.all (added)
native-mac/Sources/TinyGPT/FetchGitHub.swift        CLI subcommand
native-mac/Sources/TinyGPT/TinyGPT.swift            pre-switch shim wires `fetch-github`
docs/github_data_integration.md                     (this file)
```

The CLI is wired through the same pre-switch shim pattern as
`download-dataset` and `list-datasets` (see TinyGPT.swift's
`TODO(github-data-merge)` marker) so concurrent agents working on the
main switch block don't merge-conflict.
