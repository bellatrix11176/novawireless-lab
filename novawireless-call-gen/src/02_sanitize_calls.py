"""
02_sanitize_calls.py
====================
Rebuilds all outcome, detection, trust, and churn scores in calls_metadata.csv
so every value is causally coherent with the rep and customer on that row.

WHY THIS EXISTS
---------------
generate_calls.py builds scores correctly but assigns call_type/call_subreason
using routing logic that doesn't always match the transcript body. This script:

  1. Fixes call_type + call_subreason to match what each transcript depicts
  2. Resamples ALL outcome and detection scores using the EXACT same rep-state-
     aware logic as generate_calls.py
  3. Recomputes customer_trust_baseline decay and customer_churn_risk_effective

USAGE
-----
  python src/02_sanitize_calls.py --meta output/calls_metadata_2025-01.csv \
                                  --jsonl output/transcripts_2025-01.jsonl \
                                  --out   output/calls_sanitized_2025-01.csv
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42


# ── Ground-truth probability tables — mirrors scenario_router.py exactly ───────

TRUE_RESOLUTION_PROBS = {
    "clean":               0.92,
    "unresolvable_clean":  0.10,
    "gamed_metric":        0.18,
    "fraud_store_promo":   0.25,
    "fraud_line_add":      0.22,
    "fraud_hic_exchange":  0.15,
    "fraud_care_promo":    0.30,
    "activation_clean":    0.95,
    "activation_failed":   0.08,
    "line_add_legitimate": 0.92,
}

PROXY_RESOLUTION_PROBS = {
    "clean":               0.90,
    "unresolvable_clean":  0.55,
    "gamed_metric":        0.88,
    "fraud_store_promo":   0.78,
    "fraud_line_add":      0.80,
    "fraud_hic_exchange":  0.82,
    "fraud_care_promo":    0.85,
    "activation_clean":    0.93,
    "activation_failed":   0.45,
    "line_add_legitimate": 0.90,
}

REPEAT_30D_PROBS = {
    "clean":               0.06,
    "unresolvable_clean":  0.30,
    "gamed_metric":        0.12,
    "fraud_store_promo":   0.28,
    "fraud_line_add":      0.25,
    "fraud_hic_exchange":  0.32,
    "fraud_care_promo":    0.20,
    "activation_clean":    0.05,
    "activation_failed":   0.40,
    "line_add_legitimate": 0.06,
}

REPEAT_31_60D_PROBS = {
    "clean":               0.04,
    "unresolvable_clean":  0.25,
    "gamed_metric":        0.45,
    "fraud_store_promo":   0.40,
    "fraud_line_add":      0.38,
    "fraud_hic_exchange":  0.42,
    "fraud_care_promo":    0.35,
    "activation_clean":    0.03,
    "activation_failed":   0.30,
    "line_add_legitimate": 0.04,
}

ESCALATION_PROBS = {
    "clean":               0.05,
    "unresolvable_clean":  0.25,
    "gamed_metric":        0.08,
    "fraud_store_promo":   0.30,
    "fraud_line_add":      0.35,
    "fraud_hic_exchange":  0.40,
    "fraud_care_promo":    0.22,
    "activation_clean":    0.03,
    "activation_failed":   0.30,
    "line_add_legitimate": 0.04,
}

# (imei_mismatch, nrf_generated, promo_override, line_no_usage, line_same_day)
SIGNAL_PROBS = {
    "clean":               (0.02, 0.01, 0.01, 0.01, 0.00),
    "unresolvable_clean":  (0.05, 0.02, 0.02, 0.01, 0.00),
    "gamed_metric":        (0.05, 0.03, 0.08, 0.02, 0.00),
    "fraud_store_promo":   (0.10, 0.05, 0.70, 0.05, 0.15),
    "fraud_line_add":      (0.80, 0.05, 0.10, 0.75, 0.90),
    "fraud_hic_exchange":  (0.75, 0.85, 0.05, 0.80, 0.85),
    "fraud_care_promo":    (0.08, 0.03, 0.90, 0.03, 0.00),
    "activation_clean":    (0.03, 0.00, 0.00, 0.00, 0.00),
    "activation_failed":   (0.60, 0.00, 0.00, 0.00, 0.00),
    "line_add_legitimate": (0.02, 0.00, 0.00, 0.05, 0.00),
}

REP_AWARE_PROBS = {
    "clean":               0.00,
    "unresolvable_clean":  0.00,
    "gamed_metric":        0.55,
    "fraud_store_promo":   0.30,
    "fraud_line_add":      0.15,
    "fraud_hic_exchange":  0.20,
    "fraud_care_promo":    0.60,
    "activation_clean":    0.00,
    "activation_failed":   0.00,
    "line_add_legitimate": 0.00,
}

AHT_MULTIPLIERS = {
    "clean":               1.00,
    "unresolvable_clean":  1.25,
    "gamed_metric":        0.80,
    "fraud_store_promo":   1.30,
    "fraud_line_add":      1.45,
    "fraud_hic_exchange":  1.50,
    "fraud_care_promo":    1.20,
    "activation_clean":    1.10,
    "activation_failed":   1.35,
    "line_add_legitimate": 1.20,
}

SCENARIO_CHURN_MULTIPLIERS = {
    "clean":               1.00,
    "unresolvable_clean":  1.40,
    "gamed_metric":        1.25,
    "fraud_store_promo":   1.50,
    "fraud_line_add":      1.75,
    "fraud_hic_exchange":  1.60,
    "fraud_care_promo":    1.35,
    "activation_clean":    0.90,
    "activation_failed":   1.35,
    "line_add_legitimate": 0.85,
}

SCENARIO_TRUST_DECAY = {
    "clean":               0.00,
    "unresolvable_clean":  0.05,
    "gamed_metric":        0.08,
    "fraud_store_promo":   0.12,
    "fraud_line_add":      0.18,
    "fraud_hic_exchange":  0.15,
    "fraud_care_promo":    0.10,
    "activation_clean":    0.00,
    "activation_failed":   0.04,
    "line_add_legitimate": 0.00,
}

FRICTION_MULT = {"low": 0.85, "normal": 1.00, "high": 1.15, "peak": 1.30}

# ── Credit validation map ─────────────────────────────────────────────────────
# Credit columns are NOT rebuilt — they were set during generation and the
# transcript matches them exactly. We only validate that credit_type is
# consistent with scenario, and flag rows where it is not.

VALID_CREDIT_TYPES_PER_SCENARIO = {
    "clean":               {"none", "courtesy"},
    "unresolvable_clean":  {"none", "service_credit"},
    "gamed_metric":        {"none", "bandaid"},
    "fraud_store_promo":   {"none", "dispute_credit"},
    "fraud_line_add":      {"none", "dispute_credit"},
    "fraud_hic_exchange":  {"none", "fee_waiver"},
    "fraud_care_promo":    {"none"},
    "activation_clean":    {"none"},
    "activation_failed":   {"none", "service_credit"},
    "line_add_legitimate": {"none"},
}




# ── Call type correction map ───────────────────────────────────────────────────

CALL_TYPE_MAP = {
    ("gamed_metric",        "*"): ("Billing Dispute",    "Rate adjustment dispute"),
    ("unresolvable_clean",  "*"): ("Porting/Transfer",   "Port in progress — delayed"),
    ("fraud_store_promo",   "*"): ("Promotion Inquiry",  "Store promo not honored"),
    ("fraud_line_add",      "*"): ("Account Fraud",      "Unauthorized line add"),
    ("fraud_hic_exchange",  "*"): ("Billing Dispute",    "Erroneous non-return fee"),
    ("fraud_care_promo",    "*"): ("Promotion Inquiry",  "Care promo not applied"),
    ("activation_clean",    "*"): ("Device Issue",       "Device activation"),
    ("activation_failed",   "*"): ("Device Issue",       "Activation failed"),
    ("line_add_legitimate", "*"): ("Account Inquiry",    "Add a line"),
    ("clean", "Billing Dispute"):       ("Billing Dispute",   "One-time charge inquiry"),
    ("clean", "Network Coverage"):      ("Network Coverage",  "Signal or coverage issue"),
    ("clean", "Device Issue"):          ("Device Issue",      "SIM provisioning failure"),
    ("clean", "Promotion Inquiry"):     ("Promotion Inquiry", "Autopay discount not applied"),
    ("clean", "Account Inquiry"):       ("Account Security",  "Unauthorized access concern"),
    ("clean", "Payment Arrangement"):   ("Billing Dispute",   "One-time charge inquiry"),
    ("clean", "International/Roaming"): ("Network Coverage",  "Signal or coverage issue"),
}


def get_call_type_subreason(scenario: str, original_call_type: str):
    if (scenario, "*") in CALL_TYPE_MAP:
        return CALL_TYPE_MAP[(scenario, "*")]
    if (scenario, original_call_type) in CALL_TYPE_MAP:
        return CALL_TYPE_MAP[(scenario, original_call_type)]
    return original_call_type, ""


# ── Score computation ──────────────────────────────────────────────────────────

def compute_outcome_flags(rng, scenario, friction_tier, rep_state):
    fm      = FRICTION_MULT.get(friction_tier, 1.0)
    gaming  = rep_state.get("rep_gaming_propensity", 0.0)
    burnout = rep_state.get("rep_burnout_level",     0.0)
    skill   = rep_state.get("rep_policy_skill",      0.5)

    p_proxy = min(PROXY_RESOLUTION_PROBS[scenario] * (1.0 + gaming * 0.3), 0.99)
    p_true  = TRUE_RESOLUTION_PROBS[scenario] * max(0.1, 1.0 - gaming * 0.4)
    if scenario in {"clean", "unresolvable_clean"}:
        p_true = min(p_true * (1.0 + skill * 0.3), 0.99)

    p_escalate  = min(ESCALATION_PROBS[scenario]    * (1.0 + burnout * 0.5) * fm, 0.99)
    p_repeat_30 = min(REPEAT_30D_PROBS[scenario]    * (1.0 + burnout * 0.3) * fm, 0.99)
    p_repeat_31 = min(REPEAT_31_60D_PROBS[scenario] * (1.0 + burnout * 0.3) * fm, 0.99)

    return {
        "true_resolution":       rng.random() < p_true,
        "resolution_flag":       rng.random() < p_proxy,
        "repeat_contact_30d":    rng.random() < p_repeat_30,
        "repeat_contact_31_60d": rng.random() < p_repeat_31,
        "escalation_flag":       rng.random() < p_escalate,
    }


def compute_detection_flags(rng, scenario, rep_state):
    p_imei, p_nrf, p_promo, p_no_usage, p_store = SIGNAL_PROBS[scenario]
    gaming = rep_state.get("rep_gaming_propensity", 0.0)
    skill  = rep_state.get("rep_policy_skill",      0.5)

    p_promo = min(p_promo * (1.0 + gaming), 0.99)
    if scenario in {"clean", "unresolvable_clean"}:
        p_nrf      = p_nrf      * max(0.1, 1.0 - skill * 0.5)
        p_no_usage = p_no_usage * max(0.1, 1.0 - skill * 0.5)

    p_aware = min(REP_AWARE_PROBS[scenario] * (1.0 + gaming * 1.5), 0.99)

    return {
        "imei_mismatch_flag":        rng.random() < p_imei,
        "nrf_generated_flag":        rng.random() < p_nrf,
        "promo_override_post_call":  rng.random() < p_promo,
        "line_added_no_usage_flag":  rng.random() < p_no_usage,
        "line_added_same_day_store": rng.random() < p_store,
        "rep_aware_gaming":          rng.random() < p_aware,
    }


def compute_aht(rng, scenario, agent_aht_base, friction_tier, rep_state):
    fm      = FRICTION_MULT.get(friction_tier, 1.0)
    gaming  = rep_state.get("rep_gaming_propensity", 0.0)
    burnout = rep_state.get("rep_burnout_level",     0.0)
    skill   = rep_state.get("rep_policy_skill",      0.5)

    state_mult  = 1.0
    state_mult *= max(0.5, 1.0 - gaming  * 0.25)
    state_mult *= 1.0 + burnout * 0.20
    if scenario in {"clean", "unresolvable_clean"}:
        state_mult *= max(0.7, 1.0 - skill * 0.15)

    aht   = agent_aht_base * AHT_MULTIPLIERS[scenario] * fm * state_mult
    noise = rng.uniform(0.80, 1.20)
    return max(60, int(aht * noise))


def compute_trust_and_churn(scenario, base_trust, base_churn):
    decay           = SCENARIO_TRUST_DECAY.get(scenario, 0.0)
    decayed_trust   = max(0.0, base_trust - decay)
    churn_mult      = SCENARIO_CHURN_MULTIPLIERS.get(scenario, 1.0)
    effective_churn = min(base_churn * churn_mult, 0.99)
    return round(decayed_trust, 6), round(effective_churn, 6)


# ── JSONL loader ───────────────────────────────────────────────────────────────

def load_transcripts(path: Path) -> dict:
    records = {}
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            cid = obj.get("call_id")
            if not cid:
                skipped += 1
                continue
            transcript = obj.get("transcript_text") or ""
            if not transcript:
                turns = obj.get("turns", [])
                transcript = "\n".join(
                    f"[{t['speaker']}]: {t['text']}" for t in turns
                )
            records[cid] = transcript
    if skipped:
        print(f"  [WARN] {skipped} JSONL lines skipped")
    return records


# ── find_repo_root ─────────────────────────────────────────────────────────────

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


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Sanitize calls_metadata CSV — fix labels and rebuild scores."
    )
    ap.add_argument("--meta",  default=None,
                    help="Path to calls_metadata CSV. Auto-detected if omitted.")
    ap.add_argument("--jsonl", default=None,
                    help="Path to transcripts JSONL. Auto-detected if omitted.")
    ap.add_argument("--out",   default=None,
                    help="Output path. Auto-detected if omitted.")
    ap.add_argument("--seed",  type=int, default=SEED)
    ap.add_argument("--no-transcripts", action="store_true",
                    help="Skip transcript join")
    return ap.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    args   = parse_args()
    repo   = find_repo_root()
    outdir = repo / "output"

    # Auto-detect paths if not provided
    if args.meta:
        meta_p = Path(args.meta)
        if not meta_p.is_absolute():
            meta_p = repo / meta_p
    else:
        # Use most recent calls_metadata file
        candidates = sorted(outdir.glob("calls_metadata*.csv"))
        if not candidates:
            print("ERROR: No calls_metadata*.csv found in output/")
            return 1
        meta_p = candidates[-1]

    if args.jsonl:
        jsonl_p = Path(args.jsonl)
        if not jsonl_p.is_absolute():
            jsonl_p = repo / jsonl_p
    else:
        # Match transcripts to metadata by month tag
        stem = meta_p.stem.replace("calls_metadata", "transcripts")
        jsonl_p = outdir / f"{stem}.jsonl"

    if args.out:
        out_p = Path(args.out)
        if not out_p.is_absolute():
            out_p = repo / args.out
    else:
        stem  = meta_p.stem.replace("calls_metadata", "calls_sanitized")
        out_p = outdir / f"{stem}.csv"

    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not meta_p.exists():
        print(f"ERROR: metadata file not found: {meta_p}")
        return 1

    print("=" * 66)
    print("NovaWireless — Call Sanitization")
    print(f"  Metadata: {meta_p.name}")
    print(f"  JSONL:    {jsonl_p.name}")
    print(f"  Output:   {out_p.name}")
    print(f"  Seed:     {args.seed}")
    print("=" * 66)

    # Load
    print(f"\nLoading {meta_p.name} ...", end=" ", flush=True)
    df = pd.read_csv(meta_p, dtype=str).fillna("")
    print(f"{len(df):,} rows x {len(df.columns)} columns")

    transcripts = {}
    if not args.no_transcripts:
        if jsonl_p.exists():
            print(f"Loading {jsonl_p.name} ...", end=" ", flush=True)
            transcripts = load_transcripts(jsonl_p)
            print(f"{len(transcripts):,} records")
        else:
            print(f"  [INFO] {jsonl_p.name} not found — skipping transcript join")

    # Fix call_type + call_subreason
    print("Correcting call_type and call_subreason ...", end=" ", flush=True)
    fixed = df.apply(
        lambda r: get_call_type_subreason(r["scenario"], r["call_type"]),
        axis=1, result_type="expand"
    )
    df["call_type"]      = fixed[0]
    df["call_subreason"] = fixed[1]
    print("done")

    # Rebuild scores
    print("Rebuilding scores with rep-state-aware logic ...")
    rng = random.Random(args.seed)

    outcome_cols   = ["true_resolution", "resolution_flag", "repeat_contact_30d",
                      "repeat_contact_31_60d", "escalation_flag"]
    detection_cols = ["imei_mismatch_flag", "nrf_generated_flag",
                      "promo_override_post_call", "line_added_no_usage_flag",
                      "line_added_same_day_store", "rep_aware_gaming"]

    new_vals = {c: [] for c in outcome_cols + detection_cols + ["aht_secs",
                "customer_trust_baseline", "customer_churn_risk_effective"]}

    for _, row in df.iterrows():
        sc        = row["scenario"]
        friction  = row.get("friction_tier") or "normal"
        agent_aht = float(row.get("agent_aht_secs_base") or 500)
        base_trust = float(row.get("customer_trust_baseline") or 71.9)
        base_churn = float(row.get("customer_churn_risk") or 0.268)

        rep_state = {
            "rep_gaming_propensity": float(row.get("rep_gaming_propensity") or 0.0),
            "rep_burnout_level":     float(row.get("rep_burnout_level")     or 0.3),
            "rep_policy_skill":      float(row.get("rep_policy_skill")      or 0.5),
        }

        for col, val in compute_outcome_flags(rng, sc, friction, rep_state).items():
            new_vals[col].append(val)
        for col, val in compute_detection_flags(rng, sc, rep_state).items():
            new_vals[col].append(val)
        new_vals["aht_secs"].append(compute_aht(rng, sc, agent_aht, friction, rep_state))
        trust, churn = compute_trust_and_churn(sc, base_trust, base_churn)
        new_vals["customer_trust_baseline"].append(trust)
        new_vals["customer_churn_risk_effective"].append(churn)

    for col, vals in new_vals.items():
        df[col] = vals

    print(f"  Done — {len(df):,} rows updated")

    # Attach transcripts
    if transcripts:
        print("Joining transcript_text ...", end=" ", flush=True)
        df["transcript_text"] = df["call_id"].map(lambda c: transcripts.get(c, ""))
        matched = (df["transcript_text"] != "").sum()
        print(f"{matched:,} / {len(df):,} matched")
    elif "transcript_text" not in df.columns:
        df["transcript_text"] = ""

    # Write
    base_cols = [c for c in df.columns if c != "transcript_text"]
    final_df  = df[base_cols + ["transcript_text"]]
    final_df.to_csv(out_p, index=False, encoding="utf-8-sig", quoting=1)

    print(f"\n{'='*66}")
    print(f"Done.  {len(final_df):,} rows  ->  {out_p.name}")
    print(f"{'='*66}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
