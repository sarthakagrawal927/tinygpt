/**
 * tokenizer.ts — byte-level tokenizer (Phase 4).
 *
 * The v0 tokenizer is intentionally trivial: every byte of the UTF-8 encoding
 * is one token, so vocab = 256. No BPE, no merge table. This is the exact
 * tokenizer the WASM model expects.
 *
 * Roundtrip guarantee: decode(encode(text)) === text  (tests/README.md).
 *
 * Guide: docs/model_guide.md ("What you are building")
 */

export const VOCAB_SIZE = 256;

const encoder = new TextEncoder();
const decoder = new TextDecoder(); // UTF-8; tolerates partial trailing bytes

/** UTF-8 text -> byte tokens. */
export function encode(text: string): Uint8Array {
  return encoder.encode(text);
}

/** Byte tokens -> UTF-8 text. */
export function decode(tokens: Uint8Array): string {
  return decoder.decode(tokens);
}
