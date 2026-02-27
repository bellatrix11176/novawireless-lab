"""
build_transcripts_csv.py
========================
Joins transcripts.jsonl with calls_metadata.csv on call_id and writes
one flat CSV with every metadata column + full transcript on the same row.

No API key required. No external services. Runs on any machine with Python.

Inputs (both in output/ by default):
    output/transcripts.jsonl     -- one JSON record per line
    output/calls_metadata.csv    -- 43 columns, one row per call

Output:
    output/calls_enriched.csv    -- all metadata columns + transcript_text

Join key: call_id  (e.g. "CALL-0000001")

Usage (from repo root):
    pip install pandas
    python src/build_transcripts_csv.py

    # Custom paths:
    python src/build_transcripts_csv.py ^
        --jsonl  output/transcripts.jsonl ^
        --meta   output/calls_metadata.csv ^
        --out    output/calls_enriched.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Join transcripts.jsonl with calls_metadata.csv on call_id."
    )
    ap.add_argument("--jsonl", default="output/transcripts.jsonl")
    ap.add_argument("--meta",  default="output/calls_metadata.csv")
    ap.add_argument("--out",   default="output/calls_enriched.csv")
    return ap.parse_args()


def resolve(repo: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (repo / p).resolve()


def load_transcripts(path: Path) -> pd.DataFrame:
    """
    Read transcripts.jsonl -> DataFrame with columns: call_id, transcript_text
    Uses transcript_text field if present, otherwise rebuilds from turns list.
    """
    rows = []
    skipped = 0

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                if skipped <= 3:
                    print(f"  [WARN] Bad JSON at line {lineno} — skipping")
                continue

            call_id = obj.get("call_id")
            if not call_id:
                skipped += 1
                continue

            transcript = obj.get("transcript_text") or ""
            if not transcript:
                turns = obj.get("turns", [])
                transcript = "\n".join(
                    f"[{t['speaker']}]: {t['text']}" for t in turns
                )

            rows.append({"call_id": call_id, "transcript_text": transcript})

    if skipped:
        print(f"  [WARN] {skipped} lines skipped (bad JSON or missing call_id)")

    return pd.DataFrame(rows, columns=["call_id", "transcript_text"])


def main() -> int:
    args    = parse_args()
    repo    = Path(__file__).resolve().parent.parent
    jsonl_p = resolve(repo, args.jsonl)
    meta_p  = resolve(repo, args.meta)
    out_p   = resolve(repo, args.out)

    out_p.parent.mkdir(parents=True, exist_ok=True)

    missing = [(p, n) for p, n in [(jsonl_p, "transcripts.jsonl"),
                                    (meta_p,  "calls_metadata.csv")]
               if not p.exists()]
    if missing:
        for p, n in missing:
            print(f"ERROR: {n} not found at: {p}")
        return 1

    print("=" * 62)
    print("NovaWireless — Call Enrichment (Join Only, No API Required)")
    print(f"  JSONL:    {jsonl_p}")
    print(f"  Metadata: {meta_p}")
    print(f"  Output:   {out_p}")
    print("=" * 62)

    print(f"\nLoading transcripts.jsonl ...", end=" ", flush=True)
    transcripts_df = load_transcripts(jsonl_p)
    print(f"{len(transcripts_df):,} records")

    print(f"Loading calls_metadata.csv  ...", end=" ", flush=True)
    meta_df = pd.read_csv(meta_p, dtype=str)
    print(f"{len(meta_df):,} rows  x  {len(meta_df.columns)} columns")

    print(f"\nJoining on call_id ...", end=" ", flush=True)
    enriched_df = meta_df.merge(transcripts_df, on="call_id", how="left")
    print("done")

    matched   = enriched_df["transcript_text"].notna().sum()
    unmatched = enriched_df["transcript_text"].isna().sum()

    if unmatched > 0:
        print(f"  [WARN] {unmatched} rows had no matching transcript")

    # Column order: all metadata cols first, transcript_text last
    meta_cols  = [c for c in meta_df.columns if c != "transcript_text"]
    final_df   = enriched_df[meta_cols + ["transcript_text"]]

    # utf-8-sig so Excel opens without garbled characters
    final_df.to_csv(out_p, index=False, encoding="utf-8-sig")

    print(f"\n{'='*62}")
    print(f"Done.")
    print(f"  Rows:               {len(final_df):,}")
    print(f"  Columns:            {len(final_df.columns)}  (43 metadata + transcript_text)")
    print(f"  Transcripts joined: {matched:,} / {len(final_df):,}")
    print(f"  Output:             {out_p}")
    print(f"{'='*62}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
