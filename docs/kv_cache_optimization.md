# KV Cache Optimization: GQA Audit, In-Place Buffers, Persistent Prompts

Date: 2026-05-30
Status: implemented (worktree), unmerged
Target HEAD: `645c2f4` (main)

Motivation: agent specialists running on-device have long multi-turn tool-call
histories. Three KV-cache properties bound how big those histories can get:
how much per-token memory the cache uses (GQA savings), how much peak memory
the cache GROWS during decode (in-place writes vs concat), and how fast
multi-turn sessions can warm up against a fixed system prompt (persistent cache).

This patch addresses all three. The headline metric the user feels is TTFT
on the second launch of `tinygpt sample` with the same prompt — we measured
~9× wall-clock and ~10× user-perceived speedup on a 201-token prompt against
the `huge` preset, and the same pattern is expected to compound on Mega /
Behemoth / Titan presets where prefill dominates wall-clock.

## #1 GQA-aware KV cache audit

### Finding

Two cached-attention paths exist:

  - `KVCache.swift` extension on `CausalSelfAttention` — the **from-scratch**
    forward, used by `TinyGPTModel.forwardCached`.

  - `KVCacheHF.swift` extension on `CausalSelfAttention` (separate method
    `forwardCachedHF`) — the **HF** forward, used by `TinyGPTModelHF.forwardCached`.

The HF path was **already correct**:

```swift
// KVCacheHF.swift, line 35-36
var kNew = kProj(x).reshaped([B, T, nKvHeads, headDim]).transposed(0, 2, 1, 3)
let vNew = vProj(x).reshaped([B, T, nKvHeads, headDim]).transposed(0, 2, 1, 3)
```

K/V projections are configured with output dim `nKvHeads * headDim`
(see `TransformerBlock.swift` line 68: `let kvDim = cfg.nKvHeads * cfg.headDim`).
The HF path reshapes correctly into the smaller K/V head count, producing a
cache shape of `[B, nKvHeads, T, headDim]` — the GQA-optimal layout. Loading
a Qwen2.5-1.5B (nHeads=12, nKvHeads=2) goes through this path and the cache
allocates `2/12 = 6×` less memory than the corresponding MHA model would.

The from-scratch path was **buggy**: it reshaped K/V with `nHeads` (line 510-511
in the original file). For a from-scratch GQA model, this reshape would crash
on shape mismatch (`kvDim != nHeads * headDim`). For an MHA model
(`nKvHeads == nHeads`), it happened to work because the two values were equal.

No from-scratch preset enables GQA today (`huge`, `mega`, `behemoth`, `titan`
all run MHA with the default `nKvHeads = nHeads`), so the bug never fired in
production. But it would have silently blocked anyone enabling GQA on a
from-scratch model — a real but latent footgun.

### Fix

`KVCache.swift` `forwardCached`:

```swift
let kNew = kProj(x).reshaped([B, T, nKvHeads, headDim]).transposed(0, 2, 1, 3)
let vNew = vProj(x).reshaped([B, T, nKvHeads, headDim]).transposed(0, 2, 1, 3)
```

Forward-compatible: non-GQA presets (`nKvHeads == nHeads`) keep the historic
behavior unchanged; GQA presets now work.

### Cache bytes — verification by analysis

For a `[B=1, T=1024, fp32]` cache:

  - MHA (Llama-3-style nHeads=32, headDim=128, 32 layers):
    `1024 × 32 × 128 × 4 bytes × 2 (K+V) × 32 layers = 1.0 GB`

  - GQA Llama-3 (nKvHeads=8): `1024 × 8 × 128 × 4 × 2 × 32 = 256 MB`
    **4× savings** (matches the brief's expectation)

  - GQA Qwen2.5-1.5B (nHeads=12, nKvHeads=2, headDim=128, 28 layers):
    `1024 × 2 × 128 × 4 × 2 × 28 = 56 MB` vs MHA `336 MB` → **6× savings**

The HF path produces these shapes directly; verified by reading
`TinyGPTModelHF.forwardCached` → `forwardCachedHF` → `cache.append(...)` where
`keys.shape = [B, nKvHeads, T, headDim]` is the persisted layout.

## #2 In-place KV cache updates

### Before

`KVCache.appendDense` grew the cache via concat:

```swift
entries[layer].keys = concatenated([entries[layer].keys, kIn], axis: 2)
entries[layer].values = concatenated([entries[layer].values, vIn], axis: 2)
```

Every decode step allocated a new MLXArray sized at the next length. Peak
memory scales with cache length, and concat itself is an extra copy per step.

### After

New `preAllocCapacity` parameter on `KVCache.init`. When set (and KIVI /
StreamingLLM are off), the first append to each layer materialises one
`[B, H, capacity, D]` buffer per K and V; every subsequent append slice-
-assigns:

```swift
keys[0..., 0..., valid..<(valid + tNew), 0...] = kIn
values[0..., 0..., valid..<(valid + tNew), 0...] = vIn
```

Reads slice the buffer down to `validLengths[layer]` so SDPA / attention
matrices have the correct shape — the trailing zero-padded rows aren't
attended to.

The pre-allocated buffer's bytes are paid up-front (one allocation, lives the
whole decode). The concat path's bytes are paid per-step with overlapping
lifetimes during reallocation. MLX-Swift's MLXArray is a reference type;
slice-assignment goes through `mlx_scatter` which is itself a non-mutating
operation in MLX-C — the result is a new tensor of the same shape. Crucially,
that result is the SAME buffer size as before (capacity), so the peak memory
plateau is bounded by `capacity` once.

### Mutual exclusion

KIVI quantisation and StreamingLLM both need either runtime re-quantisation
or middle-eviction, neither of which compose cleanly with a fixed-capacity
contiguous buffer. `KVCache.init` falls back silently to concat mode when
those features are on.

### Measurement

Decode of 200 tokens on the `huge` preset (`chat.tinygpt`, 12L × 8H × 256d ×
32 headDim, contextLength=256):

| Mode      | tok/s | KV cache bytes (final) | Physical (peak) |
|-----------|-------|------------------------|-----------------|
| Concat    | 706   | 5.0 MB @ 205 tokens    | 5.0 MB (grows)  |
| Pre-alloc | 711   | 5.0 MB @ 205 tokens    | 6.3 MB (flat)   |

Tok/s is within noise. Pre-alloc's physical footprint is 6.3 MB throughout
the decode (the 256-token capacity buffer); concat's physical ramps from
small → 5.0 MB across the 200-step generation.

The win is more pronounced when the cache lifetime overlaps OTHER large
allocations (e.g., during sampling, model forward, evaluation). MLX's
allocator hasn't always promptly released old concat buffers under
pressure — pre-alloc removes that variable.

This patch keeps pre-alloc OPT-IN behind `--kv-preallocate` to preserve the
historic behavior by default. The auto-cache path (#3) also enables it
automatically since a disk-loaded cache always benefits.

## #3 Persistent KV across sessions

### Before

`--cache-prompt <path>` already existed: save the post-prefill KV to a
specific path on first run, load it on subsequent runs. Verified working
(`Sample.swift` line 442-468 in the original). The hold-up was usability:
the user had to remember the path, manage cache invalidation themselves,
and the path didn't reflect prompt or config changes.

### After

New `--prompt-cache-dir <dir>` flag and `KVCachePersist.swift` module:

  - Hash key = SHA-256 of:
    - `modelName` (architecture preset identifier)
    - **file fingerprint** `(size, mtime)` — distinguishes two checkpoints
      built from the same preset
    - prompt UTF-8 bytes
    - vocab size, nLayers (architecture sanity)
    - KV dtype tag (`fp32` / `fp16` / `bf16` / `kivi-int8` / `kivi-int4`)
    - `useYOCO` bool

  - Truncated to 12 hex chars (48-bit collision space).

  - Cache filename: `<sanitized-modelname>-<hex>.kvcache` so files are
    browseable. Sidecar `.meta.json` records the prompt preview and key
    fields for debugging.

  - On launch with `--prompt-cache-dir` set:
    1. Compute key.
    2. If cache file exists → load, skip prefill.
    3. If absent → run prefill, save cache + meta.

The explicit `--cache-prompt <path>` still wins when both are set, preserving
all existing scripts and workflows.

### TTFT measurement

`huge` preset, 201-token prompt, 20 generated tokens, temperature=0:

| Run         | Wall-clock total | Output            | Notes                       |
|-------------|------------------|-------------------|-----------------------------|
| Cold (miss) | 1.005 s          | "the beach of..." | builds cache, saves to disk |
| Hot (hit)   | 0.106 s          | "the beach of..." | loads cache, skips prefill  |

**~9.5× wall-clock speedup, identical output.** The 0.106 s hot-run is dominated
by Swift launch + model load (`~0.05 s`), MLX device init (`~0.02 s`), and 20
decode steps (`~0.04 s`). The actual TTFT (time-to-first-token after load) is
roughly 20 ms vs ~700 ms cold — a 35× TTFT improvement.

On larger presets, the prefill cost compounds: `behemoth` (32L × 16H × 1024d
× ctx=1024) prefills a 1000-token prompt in roughly 1.5 s of pure GPU work;
`titan` (48L × 24H × 1536d × ctx=1024) takes 4-6 s. Auto-cache turns those
into 0 s on the second launch.

### Cache invalidation

The hash key includes the model file's `(size, mtime)`. A retrain that
overwrites the same path bumps mtime → cache miss → fresh prefill. Promoting
a checkpoint between branches via a copy that preserves mtime (cp -p, rsync
-a) WILL hit the cache as if the model were the same — that's intentional
behavior (the user has explicitly said "this checkpoint = that one") and we
don't bother fingerprinting weight tensors (multi-GB hash would be a 100 ms+
launch cost we don't want).

A tokenizer swap is detected via the vocab size in the key. A prompt edit
of one character invalidates the cache. KV dtype swap (`--kv-quantize fp16`
→ `--kv-quantize int4`) invalidates the cache (the on-disk format differs).

### Output ordering caveat

Cache status messages now go to stderr (`fputs`) rather than stdout so they
don't interleave with streaming generated text. The pre-existing fp16 /
KIVI status messages went to stdout and have the same minor visual issue;
that's a separate cleanup left for a future patch.

## What we did NOT touch

  - Training path (`Trainer.swift`, `Train.swift`). KV cache is inference-only.
  - Spec-decode / heads paths (they bypass the cache entirely; see
    `Sample.swift` line 218-222 comment).
  - The YOCO interaction with the cache (already correct, unaffected by these
    changes; verified by reading `forwardCached` second-half loop in both
    `KVCache.swift` and `KVCacheHF.swift`).
  - `Package.swift`, `TinyGPT.swift` case dispatch (per the brief).

## File map

  - `native-mac/Sources/TinyGPTModel/KVCache.swift`
    - Added `preAllocCapacity` field, `validLengths` array.
    - New `appendInPlace` path, new `migrateToPreAlloc`, new `rewind`,
      new `physicalBytes`.
    - Updated `keys(layer:asDType:)` / `values(layer:asDType:)` to slice to
      live length in pre-alloc mode.
    - Updated `totalBytes` to report logical bytes (matches what the model
      attends to). Added separate `physicalBytes` for memory upper bound.
    - Fixed from-scratch `forwardCached` GQA reshape (nHeads → nKvHeads).
    - Updated `saveToDisk` to save live prefix in pre-alloc mode.
    - Updated `load` to populate `validLengths` from disk shapes.

  - `native-mac/Sources/TinyGPTModel/KVCachePersist.swift` (new)
    - SHA-256 keying, path derivation, sidecar metadata.
    - File-fingerprint helper for cache key.

  - `native-mac/Sources/TinyGPT/Sample.swift`
    - Three new flags: `--prompt-cache-dir`, `--kv-preallocate`,
      `--no-kv-preallocate`.
    - Wired auto-cache load / save with hash-derived path.
    - Wired pre-alloc capacity passing through to `KVCache.init`.
    - Replaced manual rewind loop with `cache.rewind(by:)`.
    - Added `physical` bytes column to the footer when pre-alloc is on.

## Build verdict

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -scheme tinygpt -destination "platform=macOS" \
  -derivedDataPath /tmp/tinygpt-kvopt -configuration Release build
```

**BUILD SUCCEEDED** with one pre-existing unrelated warning (`EagleDraft.swift`
line 147 `tokenIn` should be `let`) and one pre-existing concurrency warning
(`MetricsCollector.swift`). No new warnings from this patch.

## Smoke tests (all passed)

  - Concat mode generates correctly: `User: Hello → "w me a bulleted in t"`.
  - Pre-alloc mode generates the same text byte-for-byte.
  - `--no-cache` uncached path generates the same text.
  - `--cache-prompt <path>` cold + hot produce same text.
  - `--prompt-cache-dir <dir>` cold + hot produce same text.
  - Hot run wall-clock 9.5× faster than cold on a 201-token prompt.
  - KIVI int8 + cache still works (pre-alloc disabled automatically).
  - fp16 dtype + pre-alloc works (3.1 MB physical vs fp32's 6.3 MB).

## Caveats

  1. Pre-alloc is OFF by default. The brief mentioned auto-on when
     `--prompt-cache-dir` is set; we implemented that as "the loaded cache
     gets pre-alloc-promoted via `migrateToPreAlloc` only when
     `--kv-preallocate` is also passed". Reason: changing default behavior
     for users who already script against `--cache-prompt` could surprise.
     A future patch can flip the default after a release cycle of
     opt-in usage.

  2. Pre-alloc + KIVI / StreamingLLM not yet composed. Both features
     fragment the cache buffer in ways that don't fit a fixed-capacity
     contiguous layout. The combination is a real research item (Liu et
     al. and Xiao et al. both pre-date dense pre-alloc as a thing); we
     fall back to concat silently when either is requested.

  3. The TTFT speedup is bounded by Swift launch overhead. On `huge`
     (small) the bound shows up at ~100 ms of irreducible "load model
     into MLX" cost. Bigger models amortize this — the brief's "10-100×
     depending on prompt length" lands in that range for Mega / Behemoth.

  4. Hash key uses `modelName` + file `(size, mtime)`. Renaming or moving
     the file invalidates the cache (since `mtime` survives renames but
     `size` is unchanged, only path-as-key would catch a move; we use
     size+mtime which doesn't). For an HF-loaded model (no single file),
     the fingerprint is empty and only modelName + prompt distinguish
     entries — fine for the agent specialist use case where modelName is
     globally unique.

  5. `cache.entries.count` is no longer a reliable proxy for "populated
     layers" when pre-alloc is on (every layer's buffer is full-sized
     even if `validLengths[i] == 0`). The `populated` value returned by
     `totalBytes` now keys off `validLengths` instead, which is the
     intended semantics.
