#!/usr/bin/env python3
"""
03_build_call_subreason_priors.py

Synthetic Representative Generator (supplier project)

Goal:
Create subreason priors that match NovaWireless transcript taxonomy WITHOUT needing
a call dataset from the Call Generator project.

Inputs (repo-relative):
- data/employee_generation_inputs/fcc_cgb_consumer_complaints__issue_priors.csv

Outputs (repo-relative, supplier artifacts):
- output/call_subreason_priors__v1.csv
    columns: call_type, subreason, p_within_call_type, n_seed
- output/call_type_subreason_catalog__v1.csv
    columns: call_type, subreason
- output/call_subreason_priors__v1__qa.csv
    per call_type QA: p_sum should be 1

Design:
- Repo-root aware
- No CLI required
- Mapping is explicit + auditable (edit once, affects everything)
"""

from __future__ import annotations

import argparse
from pathlib import Path
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

def resolve_fcc_issue_priors(repo: Path) -> Path:
    candidates = [
        repo / "data" / "employee_generation_inputs" / "fcc_cgb_consumer_complaints__issue_priors.csv",
        repo / "data" / "fcc_cgb_consumer_complaints__issue_priors.csv",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p.resolve()
    raise FileNotFoundError("Missing fcc_cgb_consumer_complaints__issue_priors.csv in data/employee_generation_inputs/ or data/.")


def normalize_weights(d: dict[str, float]) -> dict[str, float]:
    s = sum(d.values())
    if s <= 0:
        raise ValueError("Weight dict sums to 0.")
    return {k: v / s for k, v in d.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_priors", default="output/call_subreason_priors__v1.csv")
    ap.add_argument("--out_catalog", default="output/call_type_subreason_catalog__v1.csv")
    ap.add_argument("--out_qa", default="output/call_subreason_priors__v1__qa.csv")
    args = ap.parse_args()

    repo = find_repo_root()

    fcc_path = resolve_fcc_issue_priors(repo)
    fcc = pd.read_csv(fcc_path)
    if not {"issue", "p"}.issubset(set(fcc.columns)):
        raise ValueError(f"FCC issue priors must contain columns issue,p. Found: {list(fcc.columns)}")

    # -------------------------------------------------------
    # Call-type -> subreason priors (EDIT HERE)
    # Weights should sum to 1 per call_type (we normalize anyway).
    # This is your transcript taxonomy backbone.
    # -------------------------------------------------------
    call_type_to_subreasons = {
        "Billing Dispute": {
            "Bill too high": 0.35,
            "Unexpected charges": 0.25,
            "Proration confusion": 0.20,
            "Fees dispute": 0.15,
            "NRF fee applied": 0.05,
        },
        "Payment Arrangement": {
            "Set up arrangement": 0.55,
            "Extension request": 0.25,
            "Promise to pay": 0.20,
        },
        "Promotion Inquiry": {
            "Promo not applied": 0.50,
            "Trade-in credit missing": 0.35,
            "Eligibility question": 0.15,
        },
        "Network Coverage": {
            "No signal": 0.35,
            "Dropped calls": 0.30,
            "Slow data": 0.25,
            "Coverage abroad": 0.10,
        },
        "Device Issue": {
            "Activation problem": 0.30,
            "SIM/eSIM trouble": 0.30,
            "Device not working": 0.25,
            "IMEI mismatch": 0.15,
        },
        "Fraud/Security": {
            "SIM swap concern": 0.45,
            "Unauthorized access": 0.25,
            "Account takeover": 0.20,
            "Line added without consent": 0.10,
        },
        "Account Inquiry": {
            "General question": 0.55,
            "Account changes": 0.30,
            "Number portability": 0.15,
        },
        "Cancellation": {
            "Too expensive": 0.70,
            "Coverage issues": 0.20,
            "Switching providers": 0.10,
        },
        "International/Roaming": {
            "Travel pass": 0.55,
            "Roaming charges": 0.35,
            "Coverage abroad": 0.10,
        },
    }

    # normalize maps
    call_type_to_subreasons = {ct: normalize_weights(m) for ct, m in call_type_to_subreasons.items()}

    # Create priors table (n_seed is a synthetic "sample size" so you can see relative weight)
    rows = []
    for call_type, sub_map in call_type_to_subreasons.items():
        for subreason, w in sub_map.items():
            rows.append({
                "call_type": call_type,
                "subreason": subreason,
                "p_within_call_type": round(float(w), 6),
                "n_seed": int(round(w * 1000)),
            })

    priors = pd.DataFrame(rows).sort_values(["call_type", "p_within_call_type"], ascending=[True, False])

    out_priors = (repo / args.out_priors).resolve()
    out_catalog = (repo / args.out_catalog).resolve()
    out_qa = (repo / args.out_qa).resolve()
    out_priors.parent.mkdir(parents=True, exist_ok=True)
    out_catalog.parent.mkdir(parents=True, exist_ok=True)
    out_qa.parent.mkdir(parents=True, exist_ok=True)

    priors.to_csv(out_priors, index=False)
    priors[["call_type", "subreason"]].to_csv(out_catalog, index=False)

    qa = priors.groupby("call_type", as_index=False)["p_within_call_type"].sum()
    qa["p_sum_ok"] = (qa["p_within_call_type"].round(6) == 1.0)
    qa.to_csv(out_qa, index=False)

    print(f"[OK] Read FCC issue priors: {fcc_path}")
    print(f"[OK] Wrote subreason priors: {out_priors}")
    print(f"[OK] Wrote catalog:         {out_catalog}")
    print(f"[OK] Wrote QA:              {out_qa}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
