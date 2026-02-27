#!/usr/bin/env python3
"""
03_inject_imei_anomalies.py

Injects Goodhart-style account metadata defects into the master ledger:

A) IMEI mismatches:
   EIP IMEI != usage IMEI (swap usage_imei within-customer for realism)

B) Missing usage IMEI:
   usage_imei becomes missing (capture failure / ghost line)

Reads (prefers params_sources/, falls back to output/):
- master_account_ledger.csv

Writes (dual-write):
- output/master_account_ledger__anomalies.csv
- data/external/params_sources/master_account_ledger__anomalies.csv
- output/imei_anomaly_examples.csv
- output/imei_anomaly_injection_receipt.json

Optional:
- --overwrite_base : also overwrite master_account_ledger.csv in BOTH places
  (so downstream call generator attaches anomalies automatically)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


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

def safe_write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def read_first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("Missing input ledger. Looked for:\n" + "\n".join(str(p) for p in paths))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--p_mismatch", type=float, default=0.07)
    ap.add_argument("--p_missing", type=float, default=0.04)
    ap.add_argument("--within_customer_only", action="store_true", default=True)
    ap.add_argument("--overwrite_base", action="store_true", default=False)
    args = ap.parse_args()

    repo = find_repo_root()
    out_dir = repo / "output"
    params_dir = repo / "data" / "external" / "params_sources"
    out_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)

    src_path = read_first_existing(
        [
            params_dir / "master_account_ledger.csv",
            out_dir / "master_account_ledger.csv",
        ]
    )

    df = pd.read_csv(src_path, low_memory=False)

    required = ["customer_id", "line_id", "product_type", "agreement_number", "eip_imei", "usage_imei"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "master_account_ledger.csv missing required columns:\n"
            + ", ".join(missing)
            + "\nRun 02_build_master_account_ledger.py first."
        )

    # Derive EIP exists if not already present
    if "eip_exists_flag" in df.columns:
        eip_exists = safe_int_series(df["eip_exists_flag"])
    else:
        eip_exists = df["agreement_number"].astype(str).str.len().gt(0).astype(int)

    rng = np.random.default_rng(int(args.seed))
    out = df.copy()

    # Track what we changed
    anomaly_type = np.array([""] * len(out), dtype=object)

    # ----------------------------
    # A) IMEI mismatches (swap usage_imei)
    # Eligible: voice lines with EIP + both IMEIs present
    # ----------------------------
    eligible_mismatch = (
        (out["product_type"].astype(str).str.lower() == "voice")
        & (eip_exists == 1)
        & (out["usage_imei"].notna())
        & (out["eip_imei"].notna())
    )

    eligible_idx = out.index[eligible_mismatch].to_numpy()
    n_mismatch = int(len(eligible_idx) * float(args.p_mismatch))

    mismatch_pairs: List[Tuple[int, int]] = []
    if n_mismatch > 0 and len(eligible_idx) >= 2:
        chosen = rng.choice(eligible_idx, size=n_mismatch, replace=False)
        for i in chosen:
            cust = out.at[int(i), "customer_id"]

            if args.within_customer_only:
                pool = out.index[eligible_mismatch & (out["customer_id"] == cust) & (out.index != i)].to_numpy()
            else:
                pool = out.index[eligible_mismatch & (out.index != i)].to_numpy()

            if len(pool) == 0:
                continue

            j = int(rng.choice(pool))
            mismatch_pairs.append((int(i), int(j)))

        for i, j in mismatch_pairs:
            ui = out.at[i, "usage_imei"]
            uj = out.at[j, "usage_imei"]
            out.at[i, "usage_imei"] = uj
            out.at[j, "usage_imei"] = ui

            anomaly_type[i] = "imei_mismatch_swap"
            anomaly_type[j] = "imei_mismatch_swap"

    # ----------------------------
    # B) Missing usage IMEI (blank usage_imei)
    # Eligible: voice or hsi lines with usage_imei present
    # Split: 70% capture failure (EIP exists), 30% ghost/unknown (no EIP)
    # ----------------------------
    eligible_missing = out["usage_imei"].notna() & (out["product_type"].astype(str).str.lower().isin(["voice", "5g_home_internet"]))
    miss_idx = out.index[eligible_missing].to_numpy()
    n_missing = int(len(miss_idx) * float(args.p_missing))

    if n_missing > 0 and len(miss_idx) > 0:
        capture_pool = out.index[eligible_missing & (eip_exists == 1)].to_numpy()
        ghost_pool = out.index[eligible_missing & (eip_exists == 0)].to_numpy()

        n_capture = int(round(n_missing * 0.70))
        n_ghost = n_missing - n_capture

        chosen_capture = rng.choice(capture_pool, size=min(n_capture, len(capture_pool)), replace=False) if len(capture_pool) else np.array([], dtype=int)
        chosen_ghost = rng.choice(ghost_pool, size=min(n_ghost, len(ghost_pool)), replace=False) if len(ghost_pool) else np.array([], dtype=int)

        for i in chosen_capture:
            out.at[int(i), "usage_imei"] = np.nan
            anomaly_type[int(i)] = "missing_usage_imei_capture_failure"

        for i in chosen_ghost:
            out.at[int(i), "usage_imei"] = np.nan
            anomaly_type[int(i)] = "missing_usage_imei_ghost_line"

    # Recompute flags
    out["missing_usage_imei_flag"] = out["usage_imei"].isna().astype(int)

    out["imei_mismatch_flag"] = (
        (eip_exists == 1)
        & out["eip_imei"].notna()
        & out["usage_imei"].notna()
        & (out["eip_imei"].astype(str) != out["usage_imei"].astype(str))
    ).astype(int)

    # Keep eip_mismatch_flag if present; else recompute as 0 by default (agreement mismatch handled in ledger build)
    if "eip_mismatch_flag" not in out.columns:
        out["eip_mismatch_flag"] = 0

    out["upstream_friction_risk_flag"] = ((out["imei_mismatch_flag"] == 1) | (out["missing_usage_imei_flag"] == 1) | (out["eip_mismatch_flag"] == 1)).astype(int)
    out["imei_anomaly_type"] = anomaly_type

    evidence_cols = [
        "customer_id",
        "line_id",
        "product_type",
        "agreement_number",
        "eip_imei",
        "usage_imei",
        "imei_mismatch_flag",
        "missing_usage_imei_flag",
        "eip_mismatch_flag",
        "upstream_friction_risk_flag",
        "imei_anomaly_type",
    ]
    evidence = out.loc[
        (out["imei_anomaly_type"].astype(str) != "") | (out["upstream_friction_risk_flag"] == 1),
        evidence_cols,
    ].copy()

    # Write outputs (dual-write)
    out_path = out_dir / "master_account_ledger__anomalies.csv"
    params_out_path = params_dir / "master_account_ledger__anomalies.csv"
    evidence_path = out_dir / "imei_anomaly_examples.csv"
    receipt_path = out_dir / "imei_anomaly_injection_receipt.json"

    out.to_csv(out_path, index=False)
    out.to_csv(params_out_path, index=False)
    evidence.to_csv(evidence_path, index=False)

    # Optional overwrite base ledger (so call generator attaches anomalies automatically)
    if args.overwrite_base:
        out.to_csv(out_dir / "master_account_ledger.csv", index=False)
        out.to_csv(params_dir / "master_account_ledger.csv", index=False)

    receipt: Dict[str, Any] = {
        "run_ts": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "input": str(src_path),
        "outputs": {
            "output_anomalies_ledger": str(out_path),
            "params_anomalies_ledger": str(params_out_path),
            "evidence_file": str(evidence_path),
        },
        "params": {
            "p_mismatch": float(args.p_mismatch),
            "p_missing": float(args.p_missing),
            "within_customer_only": bool(args.within_customer_only),
            "overwrite_base": bool(args.overwrite_base),
            "missing_split": {"capture_failure": 0.70, "ghost_line": 0.30},
        },
        "counts": {
            "rows_total": int(len(out)),
            "rows_anomaly_tagged": int((out["imei_anomaly_type"].astype(str) != "").sum()),
            "imei_mismatch_flag_count": int(out["imei_mismatch_flag"].sum()),
            "missing_usage_imei_flag_count": int(out["missing_usage_imei_flag"].sum()),
            "upstream_friction_risk_flag_count": int(out["upstream_friction_risk_flag"].sum()),
        },
        "notes": [
            "Mismatch swaps operate on usage_imei to simulate 'device used on different line than financed'.",
            "Missing usage_imei simulates capture failure + ghost lines.",
        ],
    }

    safe_write_json(receipt_path, receipt)

    print(f"[ok] wrote: {out_path}")
    print(f"[ok] wrote: {params_out_path}")
    if args.overwrite_base:
        print(f"[ok] overwrote base ledger in output/ and params_sources/")
    print(f"[ok] wrote evidence: {evidence_path}")
    print(f"[ok] wrote receipt: {receipt_path}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
