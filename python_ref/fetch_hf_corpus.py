#!/usr/bin/env python
"""fetch_hf_corpus.py — stream a HuggingFace dataset to a UTF-8 text file
ready for `tinygpt train --corpus`.

Streams so the full dataset doesn't materialize on disk — pick how many
tokens you want via --target-tokens and the script stops once that's
roughly reached (rough because token-counting requires running the
tokenizer, which is slow; we estimate by byte count assuming ~4 bytes per
BPE token, the typical SmolLM2 / GPT-2 rate).

Examples:
  # 500M-token slice of FineWeb-edu (the 2024-vintage "good web text")
  python fetch_hf_corpus.py \\
      --dataset HuggingFaceFW/fineweb-edu --config sample-10BT \\
      --split train --target-tokens 500M \\
      --out /tmp/fineweb-500M.txt

  # Whole TinyStories (small enough)
  python fetch_hf_corpus.py \\
      --dataset roneneldan/TinyStories --target-tokens 100M \\
      --out /tmp/tinystories.txt

  # Wikipedia (single overnight option)
  python fetch_hf_corpus.py \\
      --dataset wikimedia/wikipedia --config 20231101.en \\
      --target-tokens 1B --out /tmp/wiki-1B.txt

Required:  pip install datasets
"""

import argparse
import sys
import time


def parse_count(s: str) -> int:
    """Parse '500M', '1.5B', '100K', '2_000_000' into an int."""
    s = s.strip().replace("_", "").upper()
    if not s:
        raise ValueError("empty count")
    if s[-1] in "KMBT":
        suffix = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}[s[-1]]
        return int(float(s[:-1]) * suffix)
    return int(s)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--dataset", required=True,
                   help='HF dataset name, e.g. "HuggingFaceFW/fineweb-edu"')
    p.add_argument("--config", default=None,
                   help='Dataset config / subset, e.g. "sample-10BT". Some datasets require this.')
    p.add_argument("--split", default="train", help='Split (default "train")')
    p.add_argument("--text-column", default="text",
                   help='Column with the text field (default "text"; some use "content" or "raw")')
    p.add_argument("--target-tokens", default="100M",
                   help='Stop after roughly this many tokens (assuming ~4 bytes/token). '
                        'Accepts 500M / 1.5B / etc. Default 100M.')
    p.add_argument("--out", required=True, help='Output UTF-8 text file path')
    p.add_argument("--bytes-per-token", type=float, default=4.0,
                   help='Bytes-per-token estimate for the target cap (default 4.0; '
                        'SmolLM2 BPE on English averages ~4)')
    p.add_argument("--separator", default="\n\n",
                   help='Joiner between records (default blank line)')
    p.add_argument("--no-streaming", action="store_true",
                   help='Materialize the full dataset before iterating. Only safe for small datasets.')
    args = p.parse_args()

    target_tokens = parse_count(args.target_tokens)
    target_bytes = int(target_tokens * args.bytes_per_token)
    print(f"[fetch] target ≈ {target_tokens:,} tokens ≈ {target_bytes / 1e6:.0f} MB", file=sys.stderr)

    try:
        from datasets import load_dataset
    except ImportError:
        print("[fetch] ERROR: `datasets` not installed. Run:  pip install datasets", file=sys.stderr)
        sys.exit(2)

    print(f"[fetch] loading {args.dataset}"
          + (f" ({args.config})" if args.config else "")
          + f" split={args.split} streaming={'no' if args.no_streaming else 'yes'}…",
          file=sys.stderr)
    ds = load_dataset(
        args.dataset, args.config,
        split=args.split,
        streaming=not args.no_streaming,
    )

    bytes_written = 0
    records = 0
    t0 = time.monotonic()
    last_log = t0

    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get(args.text_column)
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            f.write(text)
            f.write(args.separator)
            bytes_written += len(text.encode("utf-8")) + len(args.separator.encode("utf-8"))
            records += 1

            now = time.monotonic()
            if now - last_log > 5 or bytes_written >= target_bytes:
                elapsed = now - t0
                mb = bytes_written / 1e6
                rate = mb / elapsed if elapsed > 0 else 0
                tokens_est = bytes_written / args.bytes_per_token
                pct = 100 * bytes_written / max(target_bytes, 1)
                print(f"[fetch] {records:,} rec · {mb:.0f} MB · ~{tokens_est / 1e6:.1f}M tok "
                      f"· {rate:.1f} MB/s · {pct:.1f}%", file=sys.stderr)
                last_log = now

            if bytes_written >= target_bytes:
                break

    elapsed = time.monotonic() - t0
    final_mb = bytes_written / 1e6
    final_tokens = bytes_written / args.bytes_per_token
    print(f"[fetch] done: {records:,} records · {final_mb:.1f} MB · "
          f"~{final_tokens / 1e6:.1f}M tokens · {elapsed:.0f}s · wrote {args.out}",
          file=sys.stderr)
    print(f"[fetch] next:  tinygpt train --corpus {args.out} --tokenizer <hf-dir> ...",
          file=sys.stderr)


if __name__ == "__main__":
    main()
