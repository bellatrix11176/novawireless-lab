"""
fix_ledger_contract_proxy.py
============================
Removes the inaccurate `contract_proxy` column from master_account_ledger.csv
and replaces it with `billing_agreement_type` — derived from eip_exists_flag
and installment_months, which are the actual ground truth for NovaWireless.

NovaWireless has no service contracts. Customers are either:
  - Month-to-month (no EIP, or EIP with 0 months)
  - On an Equipment Installment Plan (24, 30, or 36 months)

Usage (from repo root):
    python src/fix_ledger_contract_proxy.py

Overwrites:
    data/master_account_ledger.csv  (backup written alongside before overwrite)
"""

from pathlib import Path
import pandas as pd
import shutil

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

LEDGER_PATH  = DATA_DIR / "master_account_ledger.csv"
BACKUP_PATH  = DATA_DIR / "master_account_ledger__pre_contract_fix.csv"
OUTPUT_PATH  = OUTPUT_DIR / "master_account_ledger.csv"


def billing_agreement_type(row) -> str:
    """
    Derive billing agreement type from EIP data.
    eip_exists_flag == 1 AND installment_months > 0 → EIP-{n}mo
    Everything else → Month-to-month
    """
    months_val = row["installment_months"]
    if row["eip_exists_flag"] == 1 and pd.notna(months_val) and months_val > 0:
        months = int(months_val)
        return f"EIP-{months}mo"
    return "Month-to-month"


def main():
    print(f"Loading: {LEDGER_PATH}")
    df = pd.read_csv(LEDGER_PATH)

    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # Integrity check — confirm columns exist
    required = ["eip_exists_flag", "installment_months", "contract_proxy"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Expected column not found: {col}")

    # Show what we're replacing
    print(f"\n  contract_proxy value counts (being removed):")
    for val, count in df["contract_proxy"].value_counts().items():
        print(f"    {val:<20} {count:>6,}")

    # Backup original before touching anything
    shutil.copy(LEDGER_PATH, BACKUP_PATH)
    print(f"\n  Backup written: {BACKUP_PATH.name}")

    # Build replacement column
    df["billing_agreement_type"] = df.apply(billing_agreement_type, axis=1)

    # Drop contract_proxy, keep billing_agreement_type in its place
    col_pos = df.columns.get_loc("contract_proxy")
    cols_new = [c for c in df.columns if c != "contract_proxy" and c != "billing_agreement_type"]
    cols_new.insert(col_pos, "billing_agreement_type")
    df = df[cols_new]

    bat_counts = df["billing_agreement_type"].value_counts()
    print(f"\n  billing_agreement_type value counts (replacement):")
    for val, count in bat_counts.items():
        print(f"    {val:<20} {count:>6,}")

    # Cross-check: EIP rows should all have eip_exists_flag == 1
    eip_rows = df[df["billing_agreement_type"] != "Month-to-month"]
    assert (eip_rows["eip_exists_flag"] == 1).all(), \
        "Integrity failure: non-MTM rows found with eip_exists_flag != 1"

    mtm_rows = df[df["billing_agreement_type"] == "Month-to-month"]
    print(f"\n  Integrity checks passed:")
    print(f"    EIP rows:             {len(eip_rows):>6,}  (all have eip_exists_flag=1 ✓)")
    print(f"    Month-to-month rows:  {len(mtm_rows):>6,}")
    print(f"    Total:                {len(df):>6,}")

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Written: {OUTPUT_PATH}")
    print(f"  Columns: {len(df.columns)}  (was {len(df.columns)}, net change: 0 — swapped in place)")
    print("\nDone.")


if __name__ == "__main__":
    main()
