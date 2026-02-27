"""
scenario_router.py
Assigns scenario label, detection signal flags, and latent ground truth
(true_resolution) to each call record.

All business logic for what makes a call fraudulent/gamed lives here.
The transcript_builder uses these flags to pick dialogue templates.
"""
import numpy as np
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# Scenario-level outcome tables
# ---------------------------------------------------------------------------

TRUE_RESOLUTION_PROBS = {
    "clean":              0.92,
    "unresolvable_clean": 0.10,
    "gamed_metric":       0.18,
    "fraud_store_promo":  0.25,
    "fraud_line_add":     0.22,
    "fraud_hic_exchange": 0.15,
    "fraud_care_promo":   0.30,
    "activation_clean":   0.95,
    "activation_failed":  0.08,
    "line_add_legitimate": 0.92,
}

PROXY_RESOLUTION_PROBS = {
    "clean":              0.90,
    "unresolvable_clean": 0.55,
    "gamed_metric":       0.88,
    "fraud_store_promo":  0.78,
    "fraud_line_add":     0.80,
    "fraud_hic_exchange": 0.82,
    "fraud_care_promo":   0.85,
    "activation_clean":   0.93,
    "activation_failed":  0.45,
    "line_add_legitimate": 0.90,
}

REPEAT_30D_PROBS = {
    "clean":              0.06,
    "unresolvable_clean": 0.30,
    "gamed_metric":       0.12,
    "fraud_store_promo":  0.28,
    "fraud_line_add":     0.25,
    "fraud_hic_exchange": 0.32,
    "fraud_care_promo":   0.20,
    "activation_clean":   0.05,
    "activation_failed":  0.40,
    "line_add_legitimate": 0.06,
}

REPEAT_31_60D_PROBS = {
    "clean":              0.04,
    "unresolvable_clean": 0.25,
    "gamed_metric":       0.45,
    "fraud_store_promo":  0.40,
    "fraud_line_add":     0.38,
    "fraud_hic_exchange": 0.42,
    "fraud_care_promo":   0.35,
    "activation_clean":   0.03,
    "activation_failed":  0.30,
    "line_add_legitimate": 0.04,
}

SIGNAL_PROBS = {
    # (imei_mismatch, nrf_generated, promo_override_post_call,
    #  line_added_no_usage, line_added_same_day_store)
    "clean":              (0.02, 0.01, 0.01, 0.01, 0.00),
    "unresolvable_clean": (0.05, 0.02, 0.02, 0.01, 0.00),
    "gamed_metric":       (0.05, 0.03, 0.08, 0.02, 0.00),
    "fraud_store_promo":  (0.10, 0.05, 0.70, 0.05, 0.15),
    "fraud_line_add":     (0.80, 0.05, 0.10, 0.75, 0.90),
    "fraud_hic_exchange": (0.75, 0.85, 0.05, 0.80, 0.85),
    "fraud_care_promo":   (0.08, 0.03, 0.90, 0.03, 0.00),
    "activation_clean":   (0.03, 0.00, 0.00, 0.00, 0.00),
    "activation_failed":  (0.60, 0.00, 0.00, 0.00, 0.00),
    "line_add_legitimate": (0.02, 0.00, 0.00, 0.05, 0.00),
}

REP_AWARE_PROBS = {
    "clean":              0.00,
    "unresolvable_clean": 0.00,
    "gamed_metric":       0.55,
    "fraud_store_promo":  0.30,
    "fraud_line_add":     0.15,
    "fraud_hic_exchange": 0.20,
    "fraud_care_promo":   0.60,
    "activation_clean":   0.00,
    "activation_failed":  0.00,
    "line_add_legitimate": 0.00,
}

AHT_MULTIPLIERS = {
    "clean":              1.00,
    "unresolvable_clean": 1.25,
    "gamed_metric":       0.80,
    "fraud_store_promo":  1.30,
    "fraud_line_add":     1.45,
    "fraud_hic_exchange": 1.50,
    "fraud_care_promo":   1.20,
    "activation_clean":   1.10,
    "activation_failed":  1.35,
    "line_add_legitimate": 1.20,
}

ESCALATION_PROBS = {
    "clean":              0.05,
    "unresolvable_clean": 0.25,
    "gamed_metric":       0.08,
    "fraud_store_promo":  0.30,
    "fraud_line_add":     0.35,
    "fraud_hic_exchange": 0.40,
    "fraud_care_promo":   0.22,
    "activation_clean":   0.03,
    "activation_failed":  0.30,
    "line_add_legitimate": 0.04,
}

SCENARIO_CALL_TYPE = {
    "clean":              None,
    "unresolvable_clean": None,
    "gamed_metric":       None,
    "fraud_store_promo":  "Promotion Inquiry",
    "fraud_line_add":     "Billing Dispute",
    "fraud_hic_exchange": "Billing Dispute",
    "fraud_care_promo":   "Promotion Inquiry",
    "activation_clean":   "Device Issue",
    "activation_failed":  "Device Issue",
    "line_add_legitimate": "Account Inquiry",
}

SCENARIO_SUBREASON = {
    "clean":              None,
    "unresolvable_clean": None,
    "gamed_metric":       None,
    "fraud_store_promo":  "Promo not applied",
    "fraud_line_add":     "Line added without consent",
    "fraud_hic_exchange": "NRF fee applied",
    "fraud_care_promo":   "Eligibility question",
    "activation_clean":   "Device activation",
    "activation_failed":  "Activation failed",
    "line_add_legitimate": "Add a line",
}

SCENARIO_CHURN_MULTIPLIERS = {
    "clean":              1.00,
    "unresolvable_clean": 1.40,
    "gamed_metric":       1.25,
    "fraud_store_promo":  1.50,
    "fraud_line_add":     1.75,
    "fraud_hic_exchange": 1.60,
    "fraud_care_promo":   1.35,
    "activation_clean":   0.90,
    "activation_failed":  1.35,
    "line_add_legitimate": 0.85,
}

SCENARIO_TRUST_DECAY = {
    "clean":              0.00,
    "unresolvable_clean": 0.05,
    "gamed_metric":       0.08,
    "fraud_store_promo":  0.12,
    "fraud_line_add":     0.18,
    "fraud_hic_exchange": 0.15,
    "fraud_care_promo":   0.10,
    "activation_clean":   0.00,
    "activation_failed":  0.04,
    "line_add_legitimate": 0.00,
}


# ---------------------------------------------------------------------------
# Credit tables
# ---------------------------------------------------------------------------
# credit_type values:
#   "none"           — no credit applied
#   "courtesy"       — legitimate goodwill credit, authorized
#   "service_credit" — appeasement for unresolved issue, authorized
#   "bandaid"        — hush money to suppress repeat contact / game FCR, NOT authorized
#   "dispute_credit" — interim credit while investigation is open, authorized
#   "fee_waiver"     — erroneous fee reversed, authorized
#
# CREDIT_PROB: probability a credit is applied on this scenario at all
# CREDIT_AUTHORIZED: whether the credit type is within policy
# CREDIT_AMOUNT_RANGE: (min, max) in dollars — sampled uniformly
# CREDIT_TYPE: string label

CREDIT_CONFIG = {
    # scenario: (prob, credit_type, authorized, amount_min, amount_max)
    "clean_billing":       (0.70, "courtesy",       True,  10.0, 15.0),
    "clean_promo":         (0.90, "courtesy",       True,  10.0, 10.0),  # fixed $10 autopay credit
    "clean_other":         (0.00, "none",            True,   0.0,  0.0),  # network/device/security — no credit
    "unresolvable_clean":  (0.85, "service_credit",  True,  20.0, 25.0),
    "gamed_metric_aware":  (0.80, "bandaid",         False, 10.0, 20.0),  # rep deliberately games it
    "gamed_metric_naive":  (0.00, "none",            True,   0.0,  0.0),  # rep deflects, no credit
    "fraud_store_promo":   (0.90, "dispute_credit",  True,  25.0, 25.0),
    "fraud_line_add":      (0.85, "dispute_credit",  True,  30.0, 50.0),
    "fraud_hic_exchange":  (0.95, "fee_waiver",      True,  35.0, 75.0),  # NRF amount varies
    "fraud_care_promo":    (0.00, "none",            True,   0.0,  0.0),  # promise dispute, no credit yet
    "activation_clean":    (0.00, "none",            True,   0.0,  0.0),
    "activation_failed":   (0.90, "service_credit",  True,   5.0, 10.0),
    "line_add_legitimate": (0.00, "none",            True,   0.0,  0.0),
}


def build_credit(rng: np.random.Generator,
                 scenario: str,
                 call_type: str,
                 rep_aware: bool) -> dict:
    """
    Determine whether a credit is applied, the amount, type, and authorization.

    Returns:
        credit_applied    (bool)
        credit_amount     (float, 0.0 if none)
        credit_type       (str)
        credit_authorized (bool)
    """
    # Resolve which config key to use
    if scenario == "clean":
        if call_type in ("Billing Dispute", "Payment Arrangement"):
            key = "clean_billing"
        elif call_type == "Promotion Inquiry":
            key = "clean_promo"
        else:
            key = "clean_other"
    elif scenario == "gamed_metric":
        key = "gamed_metric_aware" if rep_aware else "gamed_metric_naive"
    else:
        key = scenario

    cfg = CREDIT_CONFIG.get(key, (0.00, "none", True, 0.0, 0.0))
    prob, credit_type, authorized, amt_min, amt_max = cfg

    if prob == 0.0 or rng.random() > prob:
        return {
            "credit_applied":    False,
            "credit_amount":     0.0,
            "credit_type":       "none",
            "credit_authorized": True,
        }

    amount = round(float(rng.uniform(amt_min, amt_max)), 2) if amt_max > amt_min else amt_min

    return {
        "credit_applied":    True,
        "credit_amount":     amount,
        "credit_type":       credit_type,
        "credit_authorized": authorized,
    }


# ---------------------------------------------------------------------------
# Detection flags
# ---------------------------------------------------------------------------

def build_detection_flags(rng: np.random.Generator,
                          scenario: str,
                          ledger_row,
                          rep_state: dict | None = None) -> dict:
    probs = SIGNAL_PROBS[scenario]
    p_imei, p_nrf, p_promo_override, p_no_usage, p_store_same_day = probs

    gaming = rep_state.get("gaming_propensity", 0.0) if rep_state else 0.0
    skill  = rep_state.get("policy_skill",      0.5) if rep_state else 0.5

    p_promo_override = min(p_promo_override * (1.0 + gaming), 0.99)
    if scenario in {"clean", "unresolvable_clean"}:
        p_nrf      = p_nrf      * max(0.1, 1.0 - skill * 0.5)
        p_no_usage = p_no_usage * max(0.1, 1.0 - skill * 0.5)

    fraud_scenarios = {"fraud_line_add", "fraud_hic_exchange"}
    if (scenario in fraud_scenarios and ledger_row is not None
            and "imei_mismatch_flag" in ledger_row.index):
        imei_mismatch = bool(ledger_row["imei_mismatch_flag"])
    else:
        imei_mismatch = bool(rng.random() < p_imei)

    p_aware = min(REP_AWARE_PROBS[scenario] * (1.0 + gaming * 1.5), 0.99)

    return {
        "imei_mismatch_flag":        imei_mismatch,
        "nrf_generated_flag":        bool(rng.random() < p_nrf),
        "promo_override_post_call":  bool(rng.random() < p_promo_override),
        "line_added_no_usage_flag":  bool(rng.random() < p_no_usage),
        "line_added_same_day_store": bool(rng.random() < p_store_same_day),
        "rep_aware_gaming":          bool(rng.random() < p_aware),
    }


# ---------------------------------------------------------------------------
# Outcome flags
# ---------------------------------------------------------------------------

def build_outcome_flags(rng: np.random.Generator,
                        scenario: str,
                        agent_friction_tier: str,
                        rep_state: dict | None = None) -> dict:
    friction_mult = {"low": 0.85, "normal": 1.00, "high": 1.15, "peak": 1.30}
    fm = friction_mult.get(agent_friction_tier, 1.0)

    gaming  = rep_state.get("gaming_propensity", 0.0) if rep_state else 0.0
    burnout = rep_state.get("burnout_level",     0.0) if rep_state else 0.0
    skill   = rep_state.get("policy_skill",      0.5) if rep_state else 0.5

    p_proxy = min(PROXY_RESOLUTION_PROBS[scenario] * (1.0 + gaming * 0.3), 0.99)
    p_true  = TRUE_RESOLUTION_PROBS[scenario] * max(0.1, 1.0 - gaming * 0.4)

    if scenario in {"clean", "unresolvable_clean"}:
        p_true = min(p_true * (1.0 + skill * 0.3), 0.99)

    p_escalate  = min(ESCALATION_PROBS[scenario]    * (1.0 + burnout * 0.5) * fm, 0.99)
    p_repeat_30 = min(REPEAT_30D_PROBS[scenario]    * (1.0 + burnout * 0.3) * fm, 0.99)
    p_repeat_31 = min(REPEAT_31_60D_PROBS[scenario] * (1.0 + burnout * 0.3) * fm, 0.99)

    return {
        "true_resolution":       bool(rng.random() < p_true),
        "resolution_flag":       bool(rng.random() < p_proxy),
        "repeat_contact_30d":    bool(rng.random() < p_repeat_30),
        "repeat_contact_31_60d": bool(rng.random() < p_repeat_31),
        "escalation_flag":       bool(rng.random() < p_escalate),
    }


# ---------------------------------------------------------------------------
# AHT
# ---------------------------------------------------------------------------

def get_aht(rng: np.random.Generator,
            scenario: str,
            base_secs: float,
            agent_aht_secs: float,
            friction_multiplier: float,
            rep_state: dict | None = None) -> int:
    gaming  = rep_state.get("gaming_propensity", 0.0) if rep_state else 0.0
    burnout = rep_state.get("burnout_level",     0.0) if rep_state else 0.0
    skill   = rep_state.get("policy_skill",      0.5) if rep_state else 0.5

    state_mult  = 1.0
    state_mult *= max(0.5, 1.0 - gaming  * 0.25)
    state_mult *= 1.0 + burnout * 0.20
    if scenario in {"clean", "unresolvable_clean"}:
        state_mult *= max(0.7, 1.0 - skill * 0.15)

    aht   = agent_aht_secs * AHT_MULTIPLIERS[scenario] * friction_multiplier * state_mult
    noise = rng.uniform(0.80, 1.20)
    return max(60, int(aht * noise))


# ---------------------------------------------------------------------------
# Scenario assignment
# ---------------------------------------------------------------------------

def assign_scenario(rng: np.random.Generator, scenario_mix: dict) -> str:
    scenarios = list(scenario_mix.keys())
    weights   = np.array(list(scenario_mix.values()), dtype=float)
    weights  /= weights.sum()
    return rng.choice(scenarios, p=weights)
