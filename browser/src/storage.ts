/**
 * storage.ts — browser-local persistence via OPFS (Phase 4).
 *
 * Saves a run to the Origin-Private File System so it survives a page refresh:
 * a small JSON snapshot (config + loss history) plus a binary blob of the model
 * state (weights + AdamW moments + step, from the WASM C ABI). OPFS is subject
 * to storage quota; clearing site data deletes it — durability is requested up
 * front.
 *
 * Guide: docs/browser_notes.md ("Checkpointing")
 */

const RUN_FILE = "last_run.json";
const STATE_FILE = "last_run.weights";

export interface RunSnapshot {
  savedAt: string;
  config: unknown;
  lossHistory: { step: number; trainLoss: number; valLoss?: number }[];
}

/** Ask the browser to keep our storage across eviction, and report the quota. */
export async function requestDurableStorage(): Promise<{
  persisted: boolean;
  quotaMB: number;
}> {
  let persisted = false;
  let quotaMB = 0;
  try {
    if (navigator.storage?.persist) persisted = await navigator.storage.persist();
    if (navigator.storage?.estimate) {
      const est = await navigator.storage.estimate();
      quotaMB = Math.round((est.quota ?? 0) / (1024 * 1024));
    }
  } catch {
    /* storage API unavailable — best-effort only */
  }
  return { persisted, quotaMB };
}

function opfsAvailable(): boolean {
  return typeof navigator !== "undefined" && !!navigator.storage?.getDirectory;
}

/** Persist a run snapshot to OPFS. Returns false if OPFS is unavailable. */
export async function saveRun(snapshot: RunSnapshot): Promise<boolean> {
  if (!opfsAvailable()) return false;
  try {
    const root = await navigator.storage.getDirectory();
    const handle = await root.getFileHandle(RUN_FILE, { create: true });
    const writable = await handle.createWritable();
    await writable.write(JSON.stringify(snapshot));
    await writable.close();
    return true;
  } catch {
    return false;
  }
}

/** Load the last persisted run snapshot, or null if there is none. */
export async function loadRun(): Promise<RunSnapshot | null> {
  if (!opfsAvailable()) return null;
  try {
    const root = await navigator.storage.getDirectory();
    const handle = await root.getFileHandle(RUN_FILE);
    const text = await (await handle.getFile()).text();
    return JSON.parse(text) as RunSnapshot;
  } catch {
    return null; // no prior run, or OPFS read failed
  }
}

/** Persist the binary model state (from TinyGptModel.exportState()). */
export async function saveState(state: Uint8Array): Promise<boolean> {
  if (!opfsAvailable()) return false;
  try {
    const root = await navigator.storage.getDirectory();
    const handle = await root.getFileHandle(STATE_FILE, { create: true });
    const writable = await handle.createWritable();
    // Cast pins the TS 5.7+ generic to ArrayBuffer (never SharedArrayBuffer).
    await writable.write(state as Uint8Array<ArrayBuffer>);
    await writable.close();
    return true;
  } catch {
    return false;
  }
}

/** Load the persisted model state, or null if there is none. */
export async function loadState(): Promise<Uint8Array | null> {
  if (!opfsAvailable()) return null;
  try {
    const root = await navigator.storage.getDirectory();
    const handle = await root.getFileHandle(STATE_FILE);
    const buf = await (await handle.getFile()).arrayBuffer();
    return new Uint8Array(buf);
  } catch {
    return null;
  }
}

// --- Gallery model OPFS cache ---------------------------------------------
// Gallery model files are ~18 MB each; HTTP cache + Cloudflare edge cache
// already make repeat fetches reasonable. OPFS is the next layer: cache
// the bytes so a returning user doesn't pay the network at all (and works
// fully offline once any model has been loaded once). Files live under a
// `gallery/` sub-directory in OPFS root so a wipe of one model doesn't
// affect the saved-run state above.

const GALLERY_DIR = "gallery";

/** Read a gallery model from OPFS cache. Returns null on miss / OPFS off. */
export async function loadCachedGalleryModel(
  filename: string,
): Promise<Uint8Array | null> {
  if (!opfsAvailable()) return null;
  try {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(GALLERY_DIR);
    const handle = await dir.getFileHandle(filename);
    const buf = await (await handle.getFile()).arrayBuffer();
    return new Uint8Array(buf);
  } catch {
    return null;
  }
}

/** Persist a gallery model to OPFS cache. Returns false on failure (quota,
 *  OPFS off, etc.); failures are non-fatal — the caller already has the
 *  bytes from network, the next visit just re-fetches. */
export async function saveCachedGalleryModel(
  filename: string,
  bytes: Uint8Array,
): Promise<boolean> {
  if (!opfsAvailable()) return false;
  try {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(GALLERY_DIR, { create: true });
    const handle = await dir.getFileHandle(filename, { create: true });
    const writable = await handle.createWritable();
    await writable.write(bytes as Uint8Array<ArrayBuffer>);
    await writable.close();
    return true;
  } catch {
    return false;
  }
}

/** Delete the persisted run (snapshot + state blob), if any. */
export async function clearRun(): Promise<void> {
  if (!opfsAvailable()) return;
  const root = await navigator.storage.getDirectory().catch(() => null);
  if (!root) return;
  for (const name of [RUN_FILE, STATE_FILE]) {
    try {
      await root.removeEntry(name);
    } catch {
      /* nothing to clear */
    }
  }
}
