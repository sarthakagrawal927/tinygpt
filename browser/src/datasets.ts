/**
 * datasets.ts — load training text from Hugging Face (Phase 4).
 *
 * The Hugging Face datasets-server exposes public datasets as JSON over HTTP
 * with `Access-Control-Allow-Origin: *` and no API key — so a static site can
 * pull training text straight from the browser:
 *
 *   GET https://datasets-server.huggingface.co/rows
 *       ?dataset=<id>&config=<c>&split=<s>&offset=<o>&length=<n>
 *   -> { rows: [ { row: { <textColumn>: "..." } }, ... ] }
 *
 * This module ships a small curated catalog (all verified to work with the
 * datasets-server and reasonable for a tiny byte-level model) plus a pager that
 * concatenates rows up to a character budget.
 */

export interface HfDataset {
  id: string; // short key for the <select>
  label: string;
  dataset: string; // Hugging Face dataset path
  config: string;
  split: string;
  textColumn: string; // which row field holds the text
  license: string;
  blurb: string;
}

/** Curated, datasets-server-verified datasets — all plain English text. */
export const HF_CATALOG: HfDataset[] = [
  {
    id: "tinystories",
    label: "TinyStories",
    dataset: "roneneldan/TinyStories",
    config: "default",
    split: "train",
    textColumn: "text",
    license: "CDLA-Sharing-1.0",
    blurb: "simple short stories — the best fit for a tiny model",
  },
  {
    id: "simplewiki",
    label: "Simple English Wikipedia",
    dataset: "wikimedia/wikipedia",
    config: "20231101.simple",
    split: "train",
    textColumn: "text",
    license: "CC-BY-SA",
    blurb: "encyclopedia articles in simplified English",
  },
  {
    id: "wikitext",
    label: "WikiText-2",
    dataset: "Salesforce/wikitext",
    config: "wikitext-2-raw-v1",
    split: "train",
    textColumn: "text",
    license: "CC-BY-SA-3.0",
    blurb: "the classic language-modeling benchmark corpus",
  },
  {
    id: "quotes",
    label: "English quotes",
    dataset: "Abirate/english_quotes",
    config: "default",
    split: "train",
    textColumn: "quote",
    license: "CC-BY-4.0",
    blurb: "short literary quotations",
  },
];

const SERVER = "https://datasets-server.huggingface.co";

/**
 * Fetch up to ~maxChars of text from a Hugging Face dataset, paging the
 * datasets-server `rows` endpoint. `onProgress` reports characters so far.
 */
export async function fetchHfText(
  d: HfDataset,
  maxChars = 120_000,
  onProgress?: (chars: number) => void,
): Promise<string> {
  const parts: string[] = [];
  let chars = 0;
  let offset = 0;
  const pageSize = 100;

  while (chars < maxChars && offset < 3000) {
    const url =
      `${SERVER}/rows?dataset=${encodeURIComponent(d.dataset)}` +
      `&config=${encodeURIComponent(d.config)}` +
      `&split=${encodeURIComponent(d.split)}` +
      `&offset=${offset}&length=${pageSize}`;

    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`dataset server returned HTTP ${resp.status}`);
    const json = (await resp.json()) as {
      rows?: { row: Record<string, unknown> }[];
    };
    const rows = json.rows ?? [];
    if (rows.length === 0) break;

    for (const r of rows) {
      const value = r.row[d.textColumn];
      if (typeof value === "string" && value.trim()) {
        parts.push(value.trim());
        chars += value.length;
      }
    }
    onProgress?.(chars);
    offset += rows.length;
    if (rows.length < pageSize) break; // reached the end of the split
  }

  if (parts.length === 0) throw new Error("no text rows were returned");
  return parts.join("\n\n").slice(0, maxChars);
}
