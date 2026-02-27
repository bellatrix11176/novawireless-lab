"""
generate_calls.py
=================
Orchestrator for the NovaWireless synthetic call center dataset.

Usage (from repo root):
    python src/generate_calls.py

Outputs (written to output/):
    calls_metadata_YYYY-MM.csv   — one row per call, all structured flags and KPIs
    transcripts_YYYY-MM.jsonl    — one JSON object per call, full turn-by-turn transcript

Inputs (read from data/):
    customers.csv
    novawireless_employee_database.csv
    master_account_ledger.csv

Reproducibility:
    Set RANDOM_SEED below. Same seed = identical output every run.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import calendar

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Path setup
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

from scenario_router import (
    assign_scenario,
    build_detection_flags,
    build_outcome_flags,
    build_credit,
    get_aht,
    SCENARIO_CALL_TYPE,
    SCENARIO_SUBREASON,
    AHT_MULTIPLIERS,
    SCENARIO_CHURN_MULTIPLIERS,
    SCENARIO_TRUST_DECAY,
)
from transcript_builder import build_transcript, transcript_to_text


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RANDOM_SEED    = 42
N_CALLS        = 10_000
BASE_AHT_SECS  = 900
SIM_START_DATE = datetime(2025, 1, 1)
SIM_END_DATE   = datetime(2025, 12, 31)

SCENARIO_MIX = {
    "clean":               0.44,
    "unresolvable_clean":  0.11,
    "gamed_metric":        0.10,
    "fraud_store_promo":   0.07,
    "fraud_line_add":      0.06,
    "fraud_hic_exchange":  0.03,
    "fraud_care_promo":    0.03,
    "activation_clean":    0.08,
    "activation_failed":   0.04,
    "line_add_legitimate": 0.04,
}

CALL_TYPE_PRIORS = {
    "Billing Dispute":       0.28,
    "Network Coverage":      0.22,
    "Device Issue":          0.18,
    "Promotion Inquiry":     0.14,
    "Account Inquiry":       0.10,
    "Payment Arrangement":   0.05,
    "International/Roaming": 0.03,
}

FRICTION_TIERS = {
    "low":    0.20,
    "normal": 0.55,
    "high":   0.18,
    "peak":   0.07,
}


# ---------------------------------------------------------------------------
# Rep state management
# ---------------------------------------------------------------------------

def init_rep_state(agent: dict) -> dict:
    return {
        "gaming_propensity": float(np.clip(agent.get("compliance_risk", 0.2), 0.0, 1.0)),
        "policy_skill":      float(np.clip(agent.get("policy_accuracy", 0.5), 0.0, 1.0)),
        "burnout_level":     float(np.clip(agent.get("burnout_index",   0.3), 0.0, 1.0)),
        "calls_handled":     0,
    }


def update_rep_state(state: dict, outcome_flags: dict, credit_info: dict) -> dict:
    """
    Mutate rep state after each call.

    gaming_propensity:
      +2% per gamed call (proxy resolved, not truly resolved)
      +1% per unauthorized credit (bandaid)  ← NEW
      -0.1% passive decay per call

    burnout_level:
      +1.5% per escalation
      -0.05% passive recovery per call

    policy_skill:
      slow increase per call, diminishing returns
    """
    gamed     = outcome_flags.get("resolution_flag") and not outcome_flags.get("true_resolution")
    escalated = outcome_flags.get("escalation_flag", False)
    bandaid   = credit_info.get("credit_applied") and not credit_info.get("credit_authorized")

    state["calls_handled"] += 1

    if gamed:
        state["gaming_propensity"] = min(1.0, state["gaming_propensity"] + 0.02)
    if bandaid:
        state["gaming_propensity"] = min(1.0, state["gaming_propensity"] + 0.01)
    if not gamed and not bandaid:
        state["gaming_propensity"] = max(0.0, state["gaming_propensity"] - 0.001)

    if escalated:
        state["burnout_level"] = min(1.0, state["burnout_level"] + 0.015)
    else:
        state["burnout_level"] = max(0.0, state["burnout_level"] - 0.0005)

    headroom = 1.0 - state["policy_skill"]
    state["policy_skill"] = min(1.0, state["policy_skill"] + headroom * 0.002)

    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_weighted(rng, mapping: dict):
    keys  = list(mapping.keys())
    probs = np.array(list(mapping.values()), dtype=float)
    probs /= probs.sum()
    return rng.choice(keys, p=probs)


def random_date(rng, start: datetime, end: datetime) -> datetime:
    delta = (end - start).days
    if delta <= 0:
        return start
    return start + timedelta(days=int(rng.integers(0, delta)))


def month_date_range(month_str: str):
    year, mon = int(month_str[:4]), int(month_str[5:7])
    start    = datetime(year, mon, 1)
    last_day = calendar.monthrange(year, mon)[1]
    end      = datetime(year, mon, last_day, 23, 59, 59)
    return start, end


def load_data():
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    employees = pd.read_csv(DATA_DIR / "novawireless_employee_database.csv")
    ledger    = pd.read_csv(DATA_DIR / "master_account_ledger.csv")
    return customers, employees, ledger


def save_ledger(ledger: pd.DataFrame) -> None:
    path = OUTPUT_DIR / "master_account_ledger.csv"
    ledger.to_csv(path, index=False)
    print(f"  [ledger write-back] Saved {len(ledger):,} rows → {path.name}")


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(n_calls: int, rng: np.random.Generator,
             sim_start: datetime = None, sim_end: datetime = None):
    if sim_start is None:
        sim_start = datetime(2025, 1, 1)
    if sim_end is None:
        sim_end   = datetime(2025, 12, 31)

    customers, employees, ledger = load_data()
    ledger    = ledger.copy()
    customers = customers.copy().reset_index(drop=True)
    cust_index = {row["customer_id"]: idx for idx, row in customers.iterrows()}

    rep_states = {
        row["rep_id"]: init_rep_state(row.to_dict())
        for _, row in employees.iterrows()
    }

    ledger_by_account = ledger.groupby("account_id")

    records     = []
    transcripts = []
    call_counter = 0

    def make_call(customer, agent, scenario, is_repeat=False, parent_call_id=None):
        nonlocal call_counter
        call_counter += 1
        call_id = f"CALL-{call_counter:07d}"

        rep_id    = agent.get("rep_id")
        rep_state = rep_states.get(rep_id)

        # Call type
        forced_type = SCENARIO_CALL_TYPE.get(scenario)
        call_type   = forced_type if forced_type else sample_weighted(rng, CALL_TYPE_PRIORS)
        subreason   = SCENARIO_SUBREASON.get(scenario)

        # Friction
        friction_tier = sample_weighted(rng, FRICTION_TIERS)
        friction_mult_map = {"low": 0.85, "normal": 1.00, "high": 1.15, "peak": 1.30}
        friction_mult = friction_mult_map[friction_tier]

        # Ledger lookup
        account_id  = customer.get("account_id")
        ledger_rows = ledger_by_account.get_group(account_id) if account_id in ledger_by_account.groups else None
        if ledger_rows is not None and len(ledger_rows) > 0:
            eip_voice = ledger_rows[
                (ledger_rows["product_type"].astype(str) == "voice") &
                (ledger_rows["eip_exists_flag"].astype(int) == 1)
            ] if "product_type" in ledger_rows.columns and "eip_exists_flag" in ledger_rows.columns else pd.DataFrame()
            ledger_row = eip_voice.iloc[0] if len(eip_voice) > 0 else ledger_rows.iloc[0]
        else:
            ledger_row = None

        # Flags
        detection_flags = build_detection_flags(rng, scenario, ledger_row, rep_state)
        outcome_flags   = build_outcome_flags(rng, scenario, friction_tier, rep_state)

        # Credit — must come AFTER detection flags so rep_aware_gaming is set
        rep_aware   = detection_flags.get("rep_aware_gaming", False)
        credit_info = build_credit(rng, scenario, call_type, rep_aware)

        # AHT — bandaid credits add a small amount of time (rep has to process it)
        agent_aht = float(agent.get("aht_secs", BASE_AHT_SECS))
        aht_secs  = get_aht(rng, scenario, BASE_AHT_SECS, agent_aht, friction_mult, rep_state)
        if credit_info.get("credit_applied"):
            aht_secs = int(aht_secs * 1.05)   # ~5% longer when credit is processed

        # Update rep state
        if rep_state is not None:
            update_rep_state(rep_state, outcome_flags, credit_info)

        # Call date
        call_date = random_date(rng, sim_start, sim_end)

        # Trust decay
        decay   = SCENARIO_TRUST_DECAY.get(scenario, 0.0)
        cust_id = customer.get("customer_id")
        if decay > 0.0 and cust_id in cust_index:
            idx = cust_index[cust_id]
            current_trust = float(customers.at[idx, "trust_baseline"])
            customers.at[idx, "trust_baseline"] = max(0.0, current_trust - decay)
            customer = customers.iloc[idx].to_dict()

        # Churn
        churn_mult           = SCENARIO_CHURN_MULTIPLIERS.get(scenario, 1.0)
        effective_churn_risk = min(float(customer.get("churn_risk_score", 0.27)) * churn_mult, 0.99)

        # Transcript — pass credit_info so dialogue matches metadata exactly
        scenario_meta = {**detection_flags, **outcome_flags}
        turns = build_transcript(
            scenario      = scenario,
            call_type     = call_type,
            agent         = agent,
            customer      = customer,
            scenario_meta = scenario_meta,
            credit_info   = credit_info,
            rng           = rng,
        )
        transcript_text = transcript_to_text(turns)

        # Metadata record
        record = {
            "call_id":              call_id,
            "is_repeat_call":       int(is_repeat),
            "parent_call_id":       parent_call_id,
            "call_date":            call_date.strftime("%Y-%m-%d"),
            "call_type":            call_type,
            "call_subreason":       subreason,
            "scenario":             scenario,
            "customer_id":          cust_id,
            "account_id":           account_id,
            "rep_id":               rep_id,
            "rep_name":             agent.get("rep_name"),
            "site":                 agent.get("site"),
            "queue_name":           agent.get("queue_name"),
            "department":           agent.get("department"),
            "agent_tenure_months":  agent.get("tenure_months"),
            "agent_strain_tier":    agent.get("strain_tier"),
            "agent_qa_score":       agent.get("qa_score"),
            "agent_aht_secs_base":  agent_aht,
            "friction_tier":        friction_tier,
            "aht_secs":             aht_secs,
            # rep state snapshot
            "rep_gaming_propensity": round(rep_state["gaming_propensity"], 4) if rep_state else None,
            "rep_policy_skill":      round(rep_state["policy_skill"],      4) if rep_state else None,
            "rep_burnout_level":     round(rep_state["burnout_level"],     4) if rep_state else None,
            "rep_calls_handled":     rep_state["calls_handled"]               if rep_state else None,
            **detection_flags,
            **outcome_flags,
            # credit columns
            "credit_applied":        credit_info["credit_applied"],
            "credit_amount":         credit_info["credit_amount"],
            "credit_type":           credit_info["credit_type"],
            "credit_authorized":     credit_info["credit_authorized"],
            # customer
            "customer_tenure_months":        customer.get("tenure_months"),
            "customer_monthly_charges":      customer.get("monthly_charges"),
            "customer_lines":                customer.get("lines_on_account"),
            "customer_churn_risk":           customer.get("churn_risk_score"),
            "customer_churn_risk_effective": round(effective_churn_risk, 6),
            "customer_trust_baseline":       customer.get("trust_baseline"),
            "customer_patience":             customer.get("patience"),
            "customer_is_churned":           customer.get("is_churned"),
        }

        transcript_obj = {
            "call_id":         call_id,
            "is_repeat_call":  int(is_repeat),
            "parent_call_id":  parent_call_id,
            "scenario":        scenario,
            "call_type":       call_type,
            "call_date":       call_date.strftime("%Y-%m-%d"),
            "rep_id":          rep_id,
            "customer_id":     cust_id,
            "credit_applied":  credit_info["credit_applied"],
            "credit_amount":   credit_info["credit_amount"],
            "credit_type":     credit_info["credit_type"],
            "turns":           turns,
            "transcript_text": transcript_text,
        }

        return record, transcript_obj, outcome_flags, credit_info

    for i in range(n_calls):
        cust_row  = customers.iloc[int(rng.integers(0, len(customers)))]
        customer  = cust_row.to_dict()
        agent_row = employees.iloc[int(rng.integers(0, len(employees)))]
        agent     = agent_row.to_dict()

        scenario = assign_scenario(rng, SCENARIO_MIX)

        record, transcript_obj, outcome_flags, credit_info = make_call(customer, agent, scenario)
        records.append(record)
        transcripts.append(transcript_obj)

        # Ledger write-back for legitimate line additions
        if scenario == "line_add_legitimate" and outcome_flags.get("true_resolution"):
            account_id = customer.get("account_id")
            cust_id    = customer.get("customer_id")
            new_line_number = int(ledger[ledger["account_id"] == account_id].shape[0]) + 1
            new_agreement   = f"AGR-{account_id}-L{new_line_number:02d}"
            new_imei        = f"35{rng.integers(100000000000000, 999999999999999):015d}"
            new_row = {
                "account_id":             account_id,
                "customer_id":            cust_id,
                "product_type":           "voice",
                "line_number":            new_line_number,
                "agreement_number":       new_agreement,
                "imei":                   new_imei,
                "eip_exists_flag":        0,
                "installment_months":     0,
                "billing_agreement_type": "Month-to-month",
                "imei_mismatch_flag":     0,
                "source_call_id":         record["call_id"],
                "added_date":             record["call_date"],
            }
            ledger = pd.concat([ledger, pd.DataFrame([new_row])], ignore_index=True)
            if cust_id in cust_index:
                idx = cust_index[cust_id]
                current_lines = int(customers.at[idx, "lines_on_account"])
                customers.at[idx, "lines_on_account"] = current_lines + 1

        # Repeat contacts generate actual call records
        if outcome_flags.get("repeat_contact_30d"):
            repeat_agent = employees.iloc[int(rng.integers(0, len(employees)))].to_dict()
            rr, rt, _, _ = make_call(customer, repeat_agent, scenario,
                                     is_repeat=True, parent_call_id=record["call_id"])
            records.append(rr)
            transcripts.append(rt)

        if outcome_flags.get("repeat_contact_31_60d"):
            repeat_agent = employees.iloc[int(rng.integers(0, len(employees)))].to_dict()
            rr, rt, _, _ = make_call(customer, repeat_agent, scenario,
                                     is_repeat=True, parent_call_id=record["call_id"])
            records.append(rr)
            transcripts.append(rt)

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i+1:,} / {n_calls:,} primary calls  (total records so far: {len(records):,})")

    # Write ledger back if any lines were added
    original_ledger_len = len(load_data()[2])
    new_rows = len(ledger) - original_ledger_len
    if new_rows > 0:
        save_ledger(ledger)
        print(f"  [ledger] {new_rows} new line(s) added to master_account_ledger.csv")

    return records, transcripts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="NovaWireless Call Generator")
    ap.add_argument("--n_calls", type=int, default=N_CALLS)
    ap.add_argument("--seed",    type=int, default=RANDOM_SEED)
    ap.add_argument("--month",   type=str, default=None,
                    help="Constrain call dates to YYYY-MM")
    args = ap.parse_args()

    if args.month:
        sim_start, sim_end = month_date_range(args.month)
        print(f"NovaWireless Call Generator  |  Month: {args.month}  ({sim_start.date()} → {sim_end.date()})")
    else:
        sim_start = SIM_START_DATE
        sim_end   = SIM_END_DATE
        print(f"NovaWireless Call Generator  |  {sim_start.date()} → {sim_end.date()}")

    print(f"  Seed: {args.seed}  |  N calls: {args.n_calls:,}  |  Data: {DATA_DIR}  |  Output: {OUTPUT_DIR}")

    for f in [DATA_DIR / "customers.csv",
              DATA_DIR / "novawireless_employee_database.csv",
              DATA_DIR / "master_account_ledger.csv"]:
        if not f.exists():
            print(f"ERROR: Missing required input file: {f}")
            sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    print("\nGenerating calls...")
    records, transcripts = generate(args.n_calls, rng, sim_start, sim_end)

    # Month tag for output filenames
    month_tag = args.month if args.month else "all"
    meta_path = OUTPUT_DIR / f"calls_metadata_{month_tag}.csv"
    jsonl_path = OUTPUT_DIR / f"transcripts_{month_tag}.jsonl"

    df = pd.DataFrame(records)
    df.to_csv(meta_path, index=False)
    print(f"\nWrote metadata:    {meta_path.name}  ({len(df):,} rows, {len(df.columns)} columns)")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for obj in transcripts:
            f.write(json.dumps(obj) + "\n")
    print(f"Wrote transcripts: {jsonl_path.name}  ({len(transcripts):,} records)")

    # Integrity report
    print("\nIntegrity check:")
    print(f"  Total records:  {len(df):,}  (primary: {int((df['is_repeat_call']==0).sum()):,}  repeats: {int((df['is_repeat_call']==1).sum()):,})")
    print(f"\n  Scenario distribution:")
    for sc, cnt in df["scenario"].value_counts().items():
        print(f"    {sc:<25} {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")

    print(f"\n  Credit summary:")
    credit_df = df[df["credit_applied"] == True]
    print(f"    Credit applied:        {len(credit_df):,}  ({len(credit_df)/len(df)*100:.1f}%)")
    print(f"    Mean credit amount:    ${credit_df['credit_amount'].mean():.2f}")
    print(f"    Unauthorized credits:  {int((credit_df['credit_authorized']==False).sum()):,}  (bandaids)")
    print(f"    Credit type breakdown:")
    for ct, cnt in credit_df["credit_type"].value_counts().items():
        print(f"      {ct:<20} {cnt:>6,}")

    # Bandaid effectiveness signal
    bandaid_df = df[(df["credit_type"] == "bandaid")]
    if len(bandaid_df) > 0:
        failed = bandaid_df["repeat_contact_31_60d"].mean() * 100
        print(f"\n  Bandaid effectiveness:")
        print(f"    Bandaid credits issued:     {len(bandaid_df):,}")
        print(f"    Repeat 31-60d rate:         {failed:.1f}%  (issue resurfaced after gaming window)")

    print(f"\n  Resolution rate (proxy):  {df['resolution_flag'].mean()*100:.1f}%")
    print(f"  True resolution rate:     {df['true_resolution'].mean()*100:.1f}%")
    print(f"  Escalation rate:          {df['escalation_flag'].mean()*100:.1f}%")
    print(f"  Mean AHT:                 {df['aht_secs'].mean():.0f}s")
    print("\nDone.")


if __name__ == "__main__":
    main()
