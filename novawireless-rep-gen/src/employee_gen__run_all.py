#!/usr/bin/env python3
"""
run_all.py — NovaWireless Employee Generator
=============================================
Runs the full representative generation pipeline in one command.
Produces novawireless_employee_database.csv ready for the Call Generator.

Usage (from repo root):
    python src/run_all.py
    python src/run_all.py --n 500 --seed 999

Outputs written to output/:
    novawireless_employee_database.csv   ← canonical file for Call Generator
    rep_persona_profiles__v1.csv
    employees__csr_one_queue__<run_id>.csv   ← versioned archive copy
    employees__csr_one_queue__<run_id>__metadata.json
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_repo_root(start=None) -> Path:
    """
    Locate the lab root by searching for .labroot sentinel files.

    Search order (walking UP from this file):
      1. First checks every ancestor for .labroot
      2. Returns the HIGHEST .labroot found — that is the
         NovaWireless Call Center Lab root, which owns the
         shared data/ and output/ directories.
      3. Falls back to src/+data/ detection if no .labroot exists.

    This means all scripts resolve data/ and output/ to:
        NovaWireless Call Center Lab/data/
        NovaWireless Call Center Lab/output/
    regardless of which sub-project they live in.
    """
    cur = Path(start or __file__).resolve()
    if cur.is_file():
        cur = cur.parent

    # Collect ALL .labroot files walking up the tree
    labroot_paths = []
    node = cur
    while True:
        candidate = node / ".labroot"
        if candidate.exists():
            labroot_paths.append(node)
        if node.parent == node:
            break
        node = node.parent

    # Highest .labroot = lab root (last in list since we walked up)
    if labroot_paths:
        return labroot_paths[-1]

    # Fallback: classic src/ + data/ detection
    node = cur
    while True:
        if (node / "src").is_dir() and (node / "data").is_dir():
            return node
        if node.parent == node:
            break
        node = node.parent

    return Path.cwd().resolve()

def run_step(label: str, script: Path, extra_args: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"{'='*60}")
    cmd = [sys.executable, str(script)] + extra_args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed: {label}")
        print(f"  Script: {script}")
        sys.exit(result.returncode)
    print(f"[OK] {label} complete.")


def find_latest_employee_csv(out_dir: Path) -> Path:
    """Find the most recently written versioned employee CSV."""
    candidates = sorted(
        out_dir.glob("employees__csr_one_queue__*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No employee CSV found in {out_dir}. "
            "generate_employees_call_center_one_queue.py may have failed."
        )
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="NovaWireless Employee Generator — full pipeline")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--site", type=str, default="NovaWireless")
    ap.add_argument("--queue_name", type=str, default="General Support")
    args = ap.parse_args()

    repo = find_repo_root()
    src  = Path(__file__).resolve().parent  # this script's own src/ folder
    out  = repo / "output"
    out.mkdir(parents=True, exist_ok=True)

    canonical = out / "novawireless_employee_database.csv"

    print(f"NovaWireless Employee Generator")
    print(f"  Repo root: {repo}")
    print(f"  N reps:    {args.n}")
    print(f"  Seed:      {args.seed}")

    run_step(
        "Generate employee roster",
        src / "generate_employees_call_center_one_queue.py",
        [
            f"--n={args.n}",
            f"--seed={args.seed}",
            f"--site={args.site}",
            f"--queue_name={args.queue_name}",
        ],
    )

    # Rename versioned output to canonical name for downstream use
    latest = find_latest_employee_csv(out)
    shutil.copy(latest, canonical)
    print(f"[OK] Copied {latest.name} → novawireless_employee_database.csv")

    run_step(
        "Enrich with persona traits (04_rep_persona_compiler)",
        src / "04_rep_persona_compiler.py",
        [],
    )

    print(f"\n{'='*60}")
    print(f"  ALL STEPS COMPLETE")
    print(f"  Canonical output: {canonical}")
    print(f"  Copy this file to the Call Generator's data/ folder.")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
