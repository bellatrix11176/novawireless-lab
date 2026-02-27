#!/usr/bin/env python3
"""
01b_generate_calls_append.py
=============================
Appends a new monthly batch of calls continuing the existing call ID sequence.

HOW IT WORKS
------------
Auto-detects the next month to generate by scanning output/ for existing
calls_metadata_YYYY-MM.csv files. Auto-detects the highest call ID already
in output/ and continues from there.

Hit F5 in IDLE 11 times (after the first run_all) to finish the year.

OUTPUTS (one set per month)
---------------------------
  output/calls_metadata_2025-02.csv
  output/transcripts_2025-02.jsonl
  output/calls_sanitized_2025-02.csv  <- use this for analysis

USAGE
-----
  python src/01b_generate_calls_append.py               # auto month
  python src/01b_generate_calls_append.py --month 2025-04
  python src/01b_generate_calls_append.py --n_calls 5000
  python src/01b_generate_calls_append.py --info         # preview only
"""

from __future__ import annotations

import argparse
import calendar
import importlib
import importlib.util
import inspect
import json
import re
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def find_repo_root(start=None) -> Path:
    cur = Path(start or __file__).resolve()
    if cur.is_file():
        cur = cur.parent
    labroot_paths = []
    node = cur
    while True:
        if (node / ".labroot").exists():
            labroot_paths.append(node)
        if node.parent == node:
            break
        node = node.parent
    if labroot_paths:
        return labroot_paths[-1]
    node = cur
    while True:
        if (node / "src").is_dir() and (node / "data").is_dir():
            return node
        if node.parent == node:
            break
        node = node.parent
    return Path.cwd().resolve()


REPO_ROOT  = find_repo_root()
DATA_DIR   = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
SRC_DIR    = Path(__file__).resolve().parent

sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

def get_current_max_id() -> int:
    meta_files = sorted(OUTPUT_DIR.glob("calls_metadata*.csv"))
    if not meta_files:
        return 0
    max_id = 0
    for path in meta_files:
        try:
            df = pd.read_csv(path, usecols=["call_id"], dtype=str)
            nums = df["call_id"].str.extract(r"CALL-(\d+)")[0].dropna().astype(int)
            if len(nums):
                max_id = max(max_id, int(nums.max()))
        except Exception:
            continue
    return max_id


def get_next_month() -> tuple:
    """
    Scan output/ for existing calls_metadata_YYYY-MM.csv files.
    Return the month after the latest one found.
    If no files exist, return (2025, 1).
    """
    meta_files = sorted(OUTPUT_DIR.glob("calls_metadata_????-??.csv"))
    if not meta_files:
        return 2025, 1
    latest = (2025, 1)
    for path in meta_files:
        try:
            df = pd.read_csv(path, usecols=["call_date"], dtype=str)
            months = df["call_date"].str[:7].dropna().unique()
            for m in months:
                y, mo = int(m[:4]), int(m[5:7])
                if (y, mo) > latest:
                    latest = (y, mo)
        except Exception:
            continue
    y, mo = latest
    mo += 1
    if mo > 12:
        mo, y = 1, y + 1
    return y, mo


def auto_seed() -> int:
    return int(time.time() * 1000) % (2**31)


# ---------------------------------------------------------------------------
# Load generate() from generate_calls.py
# ---------------------------------------------------------------------------

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_generate_fn():
    for name in ["generate_calls", "01_generate_calls"]:
        path = SRC_DIR / f"{name}.py"
        if path.exists():
            mod = load_module(name, path)
            print(f"  Loaded generator: {path.name}")
            return mod.generate
    print("[ERROR] generate_calls.py not found in src/")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

def run_sanitization(meta_path: Path, jsonl_path: Path, out_path: Path) -> bool:
    path = SRC_DIR / "02_sanitize_calls.py"
    if not path.exists():
        print(f"  [WARN] 02_sanitize_calls.py not found — skipping")
        return False
    try:
        mod = load_module("02_sanitize_calls", path)
        fake_args = types.SimpleNamespace(
            meta=str(meta_path),
            jsonl=str(jsonl_path),
            out=str(out_path),
            seed=42,
            no_transcripts=False,
        )
        original_parse = mod.parse_args
        mod.parse_args = lambda: fake_args
        result = mod.main()
        mod.parse_args = original_parse
        return result == 0
    except Exception as e:
        print(f"  [WARN] Sanitization failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def run_generation(n_calls: int, start_id: int, seed: int,
                   sim_start: datetime, sim_end: datetime):
    generate_fn = load_generate_fn()
    rng = np.random.default_rng(seed)

    print(f"  Generating {n_calls:,} calls (seed={seed}) ...")
    records, transcripts = generate_fn(n_calls, rng, sim_start, sim_end)

    # Renumber call IDs to continue from start_id
    print(f"  Renumbering: CALL-{start_id+1:07d} to CALL-{start_id+len(records):07d} ...")
    id_map = {}
    for i, rec in enumerate(records):
        old_id = rec["call_id"]
        new_id = f"CALL-{start_id + i + 1:07d}"
        id_map[old_id] = new_id
        rec["call_id"] = new_id

    for rec in transcripts:
        rec["call_id"] = id_map.get(rec["call_id"], rec["call_id"])

    for rec in records:
        parent = rec.get("parent_call_id")
        if parent and parent in id_map:
            rec["parent_call_id"] = id_map[parent]

    for rec in transcripts:
        parent = rec.get("parent_call_id")
        if parent and parent in id_map:
            rec["parent_call_id"] = id_map[parent]

    return records, transcripts


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Append a monthly batch of NovaWireless calls."
    )
    ap.add_argument("--n_calls", type=int, default=5_000,
                    help="Number of calls to generate (default: 5000)")
    ap.add_argument("--month",   type=str, default=None,
                    help="Override auto month detection. Format: YYYY-MM")
    ap.add_argument("--info",    action="store_true",
                    help="Preview only — show next month and max ID, then exit")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    current_max = get_current_max_id()
    start_id    = current_max
    seed        = auto_seed()

    # Determine month
    if args.month:
        year, mon = int(args.month[:4]), int(args.month[5:7])
    else:
        year, mon = get_next_month()

    sim_start = datetime(year, mon, 1)
    last_day  = calendar.monthrange(year, mon)[1]
    sim_end   = datetime(year, mon, last_day, 23, 59, 59)
    month_tag = f"{year}-{mon:02d}"

    print("=" * 64)
    print("NovaWireless — Append Call Batch")
    print(f"  Month:             {month_tag}  ({sim_start.date()} to {sim_end.date()})")
    print(f"  Current max ID:    CALL-{current_max:07d}")
    print(f"  New calls start:   CALL-{start_id + 1:07d}")
    print(f"  Calls to generate: {args.n_calls:,}")
    print(f"  Auto seed:         {seed}")
    print("=" * 64)

    if args.info:
        print("\n--info mode: no files written.")
        return 0

    # Output paths
    meta_path  = OUTPUT_DIR / f"calls_metadata_{month_tag}.csv"
    jsonl_path = OUTPUT_DIR / f"transcripts_{month_tag}.jsonl"
    san_path   = OUTPUT_DIR / f"calls_sanitized_{month_tag}.csv"

    # Generate
    records, transcripts = run_generation(args.n_calls, start_id, seed,
                                          sim_start, sim_end)
    actual_count = len(records)
    new_max_id   = start_id + actual_count

    # Write metadata
    print(f"\nWriting {meta_path.name} ...", end=" ", flush=True)
    df = pd.DataFrame(records)
    df.to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f"{len(df):,} rows")

    # Write transcripts
    print(f"Writing {jsonl_path.name} ...", end=" ", flush=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for obj in transcripts:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"{len(transcripts):,} records")

    # Sanitize
    print(f"\nSanitizing ...")
    ok = run_sanitization(meta_path, jsonl_path, san_path)

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  COMPLETE  —  Month: {month_tag}")
    print(f"  Call IDs:  CALL-{start_id+1:07d} to CALL-{new_max_id:07d}")
    print(f"  Records:   {actual_count:,}")
    print(f"  Seed:      {seed}  <- save to reproduce exactly")
    print(f"  {meta_path.name}")
    print(f"  {jsonl_path.name}")
    if ok:
        print(f"  {san_path.name}  <- use this for analysis")
    else:
        print(f"  {san_path.name}  <- sanitization failed, run manually")
    if mon < 12:
        print(f"  Next run will generate: {year}-{mon+1:02d}")
    else:
        print(f"  All 12 months of {year} complete!")
    print(f"{sep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
