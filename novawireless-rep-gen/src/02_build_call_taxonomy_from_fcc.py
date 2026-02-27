#!/usr/bin/env python3
"""
02_build_call_taxonomy_from_fcc.py (hardened)

- Auto-finds FCC issue priors inside THIS repo (no CLI needed)
- Prints exactly what paths it uses
- Writes outputs atomically so you can trust the result
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import pandas as pd


def find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "data").exists() and (p / "src").exists():
            return p
    return here


def resolve_fcc_issue_priors(repo: Path) -> Path:
    inputs_dirs = [
        repo / "data" / "employee_generation_inputs",
        repo / "data" / "employee generation inputs",
        repo / "data",
    ]
    filenames = [
        "fcc_cgb_consumer_complaints__issue_priors.csv",
        "fcc_cgb_consumer_complaints_issue_priors.csv",
        "fcc_cgb_complaints__issue_priors.csv",
    ]

    tried = []
    for d in inputs_dirs:
        for fn in filenames:
            p = (d / fn).resolve()
            tried.append(p)
            if p.exists() and p.is_file():
                return p

    raise FileNotFoundError(
        "Missing FCC issue priors. Looked for:\n" + "\n".join(str(p) for p in tried)
    )


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_call_type_priors", default="data/call_type_priors__from_fcc.csv")
    ap.add_argument("--out_mapping", default="data/fcc_issue_to_call_type_map.csv")
    args = ap.parse_args()

    repo = find_repo_root()
    print(f"[INFO] repo_root = {repo}")

    fcc_path = resolve_fcc_issue_priors(repo)
    print(f"[INFO] fcc_issue_priors = {fcc_path}")

    out_priors = (repo / args.out_call_type_priors).resolve()
    out_map = (repo / args.out_mapping).resolve()
    print(f"[INFO] out_call_type_priors = {out_priors}")
    print(f"[INFO] out_mapping = {out_map}")

    fcc = pd.read_csv(fcc_path)
    print(f"[INFO] fcc_rows = {len(fcc)} cols = {list(fcc.columns)}")

    if not {"issue", "p"}.issubset(set(fcc.columns)):
        raise ValueError(f"FCC file must contain columns issue,p. Found: {list(fcc.columns)}")

    issue_to_calltype = {
        "Billing": {"Billing Dispute": 0.70, "Payment Arrangement": 0.25, "Promotion Inquiry": 0.05},
        "Cramming (unauthorized charges on your phone bill)": {"Billing Dispute": 0.70, "Fraud/Security": 0.30},
        "Slamming (change of your carrier without permission)": {"Fraud/Security": 0.70, "Account Inquiry": 0.30},
        "Number Portability (keeping your number if you change providers)": {"Account Inquiry": 0.70, "Fraud/Security": 0.30},
        "Availability": {"Network Coverage": 0.85, "Account Inquiry": 0.15},
        "Availability (including rural call completion)": {"Network Coverage": 0.90, "Account Inquiry": 0.10},
        "Rural Call Completion": {"Network Coverage": 0.95, "Account Inquiry": 0.05},
        "Tower": {"Network Coverage": 0.95, "Account Inquiry": 0.05},
        "Speed": {"Network Coverage": 0.95, "Device Issue": 0.05},
        "Interference": {"Network Coverage": 0.90, "Device Issue": 0.10},
        "Interference (including signal jammers)": {"Network Coverage": 0.85, "Fraud/Security": 0.15},
        "Equipment": {"Device Issue": 0.85, "Account Inquiry": 0.15},
        "Phone": {"Device Issue": 0.90, "Account Inquiry": 0.10},
        "Robocalls": {"Fraud/Security": 0.90, "Account Inquiry": 0.10},
        "Telemarketing (including do not call and spoofing)": {"Fraud/Security": 0.90, "Account Inquiry": 0.10},
        "Junk Faxes": {"Fraud/Security": 0.80, "Account Inquiry": 0.20},
        "Open Internet/Net Neutrality": {"Account Inquiry": 0.80, "Billing Dispute": 0.20},
        "Other": {"Account Inquiry": 0.65, "Billing Dispute": 0.20, "Device Issue": 0.10, "Network Coverage": 0.05},
    }

    rows = []
    for _, r in fcc.iterrows():
        issue = str(r["issue"]).strip()
        p_issue = float(r["p"])
        mix = issue_to_calltype.get(issue, {"Account Inquiry": 1.0})

        s = sum(mix.values()) or 1.0
        for call_type, w in mix.items():
            rows.append({
                "fcc_issue": issue,
                "fcc_issue_p": p_issue,
                "call_type": call_type,
                "within_issue_weight": w / s,
                "contribution": p_issue * (w / s),
            })

    mapping_df = pd.DataFrame(rows)
    print(f"[INFO] mapping_rows = {len(mapping_df)}")
    if len(mapping_df) == 0:
        raise ValueError("Mapping produced 0 rows. Check FCC input and mapping dictionary keys.")

    priors = (
        mapping_df.groupby("call_type", as_index=False)["contribution"]
        .sum()
        .rename(columns={"contribution": "p"})
        .sort_values("p", ascending=False)
    )
    priors["p"] = priors["p"] / priors["p"].sum()

    atomic_write_csv(mapping_df, out_map)
    atomic_write_csv(priors, out_priors)

    print(f"[OK] wrote mapping: {out_map}")
    print(f"[OK] wrote priors:  {out_priors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
