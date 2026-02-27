#!/usr/bin/env python3
"""
call_gen__run_all.py — NovaWireless Call Generator
====================================================
Runs the full call generation pipeline for one calendar month.

HOW IT WORKS
------------
Auto-detects the next month to generate by scanning output/ for existing
calls_metadata_YYYY-MM.csv files. First run produces 2025-01, second run
produces 2025-02, and so on through 2025-12.

Hit F5 in IDLE 12 times to generate the full year. No arguments needed.

OUTPUTS (one set per month)
---------------------------
  output/calls_metadata_2025-01.csv      structured metadata
  output/transcripts_2025-01.jsonl       full dialogue
  output/calls_sanitized_2025-01.csv     analysis-ready (use this)

USAGE
-----
  python src/call_gen__run_all.py                    # auto month
  python src/call_gen__run_all.py --month 2025-03    # override month
  python src/call_gen__run_all.py --n_calls 5000     # override call count
"""

from __future__ import annotations

import argparse
import calendar
import importlib.util
import json
import random
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Path resolution — walks up to .labroot sentinel for lab-root data/ + output/
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


REQUIRED_DATA_FILES = [
    "customers.csv",
    "novawireless_employee_database.csv",
    "master_account_ledger.csv",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_inputs() -> None:
    missing = [f for f in REQUIRED_DATA_FILES if not (DATA_DIR / f).exists()]
    if missing:
        print("\n[ERROR] Missing required data files:")
        for f in missing:
            print(f"  data/{f}")
        sys.exit(1)


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


def run_sanitization(meta_path: Path, jsonl_path: Path, out_path: Path) -> None:
    for name in ["02_sanitize_calls"]:
        path = SRC_DIR / f"{name}.py"
        if not path.exists():
            print(f"  [WARN] {name}.py not found — skipping sanitization")
            return
        mod = load_module(name, path)
        fake_args = types.SimpleNamespace(
            meta=str(meta_path),
            jsonl=str(jsonl_path),
            out=str(out_path),
            seed=42,
            no_transcripts=False,
        )
        original_parse = mod.parse_args
        mod.parse_args = lambda: fake_args
        mod.main()
        mod.parse_args = original_parse
        return


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="NovaWireless Call Generator — monthly pipeline")
    ap.add_argument("--n_calls", type=int, default=5_000,
                    help="Number of calls to generate (default: 5000)")
    ap.add_argument("--seed",    type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--month",   type=str, default=None,
                    help="Override auto month detection. Format: YYYY-MM")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_inputs()

    # Determine which month to generate
    if args.month:
        year, mon = int(args.month[:4]), int(args.month[5:7])
    else:
        year, mon = get_next_month()

    sim_start = datetime(year, mon, 1)
    last_day  = calendar.monthrange(year, mon)[1]
    sim_end   = datetime(year, mon, last_day, 23, 59, 59)
    month_tag = f"{year}-{mon:02d}"

    print("=" * 60)
    print("NovaWireless Call Generator")
    print(f"  Repo root: {REPO_ROOT}")
    print(f"  Month:     {month_tag}  ({sim_start.date()} to {sim_end.date()})")
    print(f"  N calls:   {args.n_calls:,}")
    print(f"  Seed:      {args.seed}")
    print("=" * 60)

    # Output paths
    meta_path   = OUTPUT_DIR / f"calls_metadata_{month_tag}.csv"
    jsonl_path  = OUTPUT_DIR / f"transcripts_{month_tag}.jsonl"
    san_path    = OUTPUT_DIR / f"calls_sanitized_{month_tag}.csv"

    # Step 1 — Generate
    print(f"\n{'='*60}")
    print(f"  STEP 1: Generate calls")
    print(f"{'='*60}")
    generate_fn = load_generate_fn()
    rng = np.random.default_rng(args.seed)
    records, transcripts = generate_fn(args.n_calls, rng, sim_start, sim_end)

    dates = [r["call_date"] for r in records]
    print(f"  Call date range: {min(dates)} to {max(dates)}")

    df = pd.DataFrame(records)
    df.to_csv(meta_path, index=False)
    print(f"  Wrote: {meta_path.name}  ({len(df):,} rows)")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for obj in transcripts:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"  Wrote: {jsonl_path.name}  ({len(transcripts):,} records)")

    # Step 2 — Sanitize
    print(f"\n{'='*60}")
    print(f"  STEP 2: Sanitize")
    print(f"{'='*60}")
    run_sanitization(meta_path, jsonl_path, san_path)

    # Done
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  COMPLETE  —  Month: {month_tag}")
    print(f"  {meta_path.name}")
    print(f"  {jsonl_path.name}")
    print(f"  {san_path.name}  <- use this for analysis")
    if mon < 12:
        print(f"  Next run will generate: {year}-{mon+1:02d}")
    else:
        print(f"  All 12 months of {year} complete!")
    print(f"{sep}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
