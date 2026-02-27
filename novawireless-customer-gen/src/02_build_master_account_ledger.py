#!/usr/bin/env python3
"""
02_build_master_account_ledger.py

Builds a flattened master account ledger:
One row per line_id with full customer + line + EIP + usage context.

This is the file your call generator wants to attach:
- msisdn
- line_id
- agreement_number
- eip_imei
- usage_imei
- mismatch flags

Reads (prefers params_sources/, falls back to output/):
- customers_v1.csv OR customers.csv
- lines.csv
- eip_agreements.csv
- line_device_usage.csv

Writes (dual-write):
- output/master_account_ledger.csv
- data/external/params_sources/master_account_ledger.csv
- output/master_account_ledger_receipt.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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

def read_first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("None of the expected input files exist:\n" + "\n".join(str(p) for p in paths))


def safe_write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    repo = find_repo_root()
    out_dir = repo / "output"
    params_dir = repo / "data" / "external" / "params_sources"
    out_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)

    customers_path = read_first_existing(
        [
            params_dir / "customers_v1.csv",
            params_dir / "customers.csv",
            out_dir / "customers.csv",
        ]
    )
    lines_path = read_first_existing([params_dir / "lines.csv", out_dir / "lines.csv"])
    eip_path = read_first_existing([params_dir / "eip_agreements.csv", out_dir / "eip_agreements.csv"])
    usage_path = read_first_existing([params_dir / "line_device_usage.csv", out_dir / "line_device_usage.csv"])

    customers = pd.read_csv(customers_path, low_memory=False)
    lines = pd.read_csv(lines_path, low_memory=False)
    eip = pd.read_csv(eip_path, low_memory=False)
    usage = pd.read_csv(usage_path, low_memory=False)

    # Normalize id types
    for df in (customers, lines, eip, usage):
        for c in ["customer_id", "account_id", "line_id", "msisdn"]:
            if c in df.columns:
                df[c] = df[c].astype(str)

    # Ensure expected column names in EIP
    if "eip_imei" not in eip.columns and "imei" in eip.columns:
        eip = eip.rename(columns={"imei": "eip_imei"})

    # Ensure expected column names in usage
    if "usage_imei" not in usage.columns and "imei" in usage.columns:
        usage = usage.rename(columns={"imei": "usage_imei"})

    # Merge sequence: lines -> customers -> eip -> usage
    df = lines.merge(customers, on=["customer_id", "account_id"], how="left")

    df = df.merge(
        eip,
        on=["customer_id", "account_id", "line_id", "msisdn"],
        how="left",
        suffixes=("", "_eip"),
    )

    df = df.merge(
        usage,
        on=["customer_id", "account_id", "line_id", "msisdn"],
        how="left",
        suffixes=("", "_usage"),
    )

    # Agreement number field should be agreement_number (canonical)
    if "agreement_number" not in df.columns and "eip_agreement_number" in df.columns:
        df["agreement_number"] = df["eip_agreement_number"]
    if "agreement_number" not in df.columns:
        df["agreement_number"] = ""

    # usage-side agreement number (if present)
    if "usage_eip_agreement_number" not in df.columns:
        df["usage_eip_agreement_number"] = ""

    # Flags
    df["eip_exists_flag"] = df["agreement_number"].astype(str).str.len().gt(0).astype(int)

    # Missing usage IMEI flag
    df["missing_usage_imei_flag"] = df["usage_imei"].isna().astype(int) if "usage_imei" in df.columns else 1

    # IMEI mismatch: EIP exists, both IMEIs present, and different
    if "eip_imei" not in df.columns:
        df["eip_imei"] = ""
    if "usage_imei" not in df.columns:
        df["usage_imei"] = ""

    df["imei_mismatch_flag"] = (
        (df["eip_exists_flag"] == 1)
        & df["eip_imei"].notna()
        & df["usage_imei"].notna()
        & (df["eip_imei"].astype(str) != df["usage_imei"].astype(str))
    ).astype(int)

    # EIP mismatch flag: EIP exists, and usage's agreement number (if recorded) differs
    df["eip_mismatch_flag"] = (
        (df["eip_exists_flag"] == 1)
        & df["usage_eip_agreement_number"].astype(str).str.len().gt(0)
        & (df["usage_eip_agreement_number"].astype(str) != df["agreement_number"].astype(str))
    ).astype(int)

    # Upstream friction risk: mismatch OR missing usage IMEI OR HSI company-owned equipment
    df["upstream_friction_risk_flag"] = (
        (df["imei_mismatch_flag"] == 1)
        | (df["missing_usage_imei_flag"] == 1)
        | ((df["product_type"].astype(str) == "5g_home_internet") & (df.get("company_owned_equipment_flag", 0).astype(int) == 1))
    ).astype(int)

    # Keep a clean, explicit column order for key fields (leave the rest after)
    key_cols = [
        "customer_id",
        "account_id",
        "line_id",
        "msisdn",
        "product_type",
        "status",
        "agreement_number",
        "eip_imei",
        "usage_imei",
        "eip_exists_flag",
        "imei_mismatch_flag",
        "missing_usage_imei_flag",
        "eip_mismatch_flag",
        "upstream_friction_risk_flag",
    ]
    for c in key_cols:
        if c not in df.columns:
            df[c] = "" if c.endswith("_number") or c.endswith("_imei") else 0

    remaining = [c for c in df.columns if c not in key_cols]
    df = df[key_cols + remaining]

    out_path = out_dir / "master_account_ledger.csv"
    params_path = params_dir / "master_account_ledger.csv"
    df.to_csv(out_path, index=False)
    df.to_csv(params_path, index=False)

    receipt: Dict[str, Any] = {
        "run_ts": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "customers": str(customers_path),
            "lines": str(lines_path),
            "eip_agreements": str(eip_path),
            "line_device_usage": str(usage_path),
        },
        "outputs": {
            "output_master_ledger": str(out_path),
            "params_master_ledger": str(params_path),
        },
        "counts": {
            "rows": int(len(df)),
            "eip_exists": int(df["eip_exists_flag"].sum()),
            "imei_mismatch": int(df["imei_mismatch_flag"].sum()),
            "missing_usage_imei": int(df["missing_usage_imei_flag"].sum()),
            "eip_mismatch": int(df["eip_mismatch_flag"].sum()),
            "upstream_friction_risk": int(df["upstream_friction_risk_flag"].sum()),
        },
        "notes": [
            "Ledger is now written to params_sources so call generator can attach it.",
            "Canonical IMEI columns are eip_imei and usage_imei.",
        ],
    }

    safe_write_json(out_dir / "master_account_ledger_receipt.json", receipt)

    print(f"[ok] wrote: {out_path}")
    print(f"[ok] wrote: {params_path}")
    print(f"[ok] wrote receipt: {out_dir / 'master_account_ledger_receipt.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
