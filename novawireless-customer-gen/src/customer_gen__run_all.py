#!/usr/bin/env python3
"""
run_all.py — NovaWireless Customer Generator
=============================================
Runs the full customer + account graph pipeline in one command.

Usage (from repo root):
    python src/run_all.py
    python src/run_all.py --n_customers 5000 --seed 99

Outputs written to output/:
    customers.csv
    lines.csv
    eip_agreements.csv
    line_device_usage.csv
    devices.csv
    master_account_ledger.csv   ← ledger built and anomalies injected
    customer_generation_receipt.json
    master_account_ledger_receipt.json
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser(description="NovaWireless Customer Generator — full pipeline")
    ap.add_argument("--n_customers", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p_inject_eip_usage_mismatch", type=float, default=0.06)
    ap.add_argument("--p_voice_eip_attach_if_plan", type=float, default=0.85)
    ap.add_argument("--usage_snapshot_date", type=str, default="2026-02-22")
    args = ap.parse_args()

    repo = find_repo_root()
    src  = Path(__file__).resolve().parent  # this script's own src/ folder

    print(f"NovaWireless Customer Generator")
    print(f"  Repo root:   {repo}")
    print(f"  N customers: {args.n_customers:,}")
    print(f"  Seed:        {args.seed}")

    run_step(
        "Generate customers + account graph",
        src / "generate_customers.py",
        [
            f"--n_customers={args.n_customers}",
            f"--seed={args.seed}",
            f"--p_inject_eip_usage_mismatch={args.p_inject_eip_usage_mismatch}",
            f"--p_voice_eip_attach_if_plan={args.p_voice_eip_attach_if_plan}",
            f"--usage_snapshot_date={args.usage_snapshot_date}",
        ],
    )

    run_step(
        "Build master account ledger",
        src / "02_build_master_account_ledger.py",
        [],
    )

    run_step(
        "Inject IMEI anomalies",
        src / "03_inject_imei_anomalies.py",
        [],
    )

    print(f"\n{'='*60}")
    print(f"  ALL STEPS COMPLETE")
    print(f"  Outputs in: {repo / 'output'}")
    print(f"  Copy output files to: {repo / 'data'} for use by other generators")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
