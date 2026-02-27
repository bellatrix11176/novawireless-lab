"""
pressure_experiment.py
======================
Ecosystem-level experiment: Baseline vs. High-Pressure Workforce

What it does:
  1. Re-synthesizes two representative populations from scratch using the
     same KPI model as generate_employees_call_center_one_queue.py —
     BASELINE (base_strain=0.52, pressure=0.14) vs
     HIGH_PRESSURE (base_strain=0.72, pressure=0.60)

  2. Runs a simulated call session of N_CALLS against each workforce,
     sampling scenarios with the same SCENARIO_MIX but letting each rep's
     performance profile modulate outcome probabilities.

  3. Measures four ecosystem-level outcomes:
       - Mean FCR shift
       - Compliance risk shift
       - Repeat contact inflation (30d window)
       - Gaming scenario concentration (gamed_metric + fraud scenarios)

  4. Writes:
       output/experiment_rep_rosters.csv
       output/experiment_calls.csv
       output/experiment_summary.csv
       output/experiment_figures/  (4 PNG charts)

Fully self-contained — does not depend on generate_employees_call_center_one_queue.py.
Re-implements the KPI synthesis model inline so the experiment parameters
are explicit and auditable.

Run:
  python src/pressure_experiment.py
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcript_builder import build_transcript, transcript_to_text

# ── Paths ─────────────────────────────────────────────────────────────────────
def find_repo_root(start=None) -> Path:
    """
    Locate the lab root by searching for .labroot sentinel files.
    Returns the HIGHEST .labroot found — that is the
    NovaWireless Call Center Lab root with shared data/ and output/.
    Falls back to src/+data/ detection if no .labroot exists.
    """
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
OUTPUT_DIR = REPO_ROOT / "output"
FIG_DIR    = OUTPUT_DIR / "experiment_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Experiment config ─────────────────────────────────────────────────────────
SEED       = 2026_02_26
N_REPS     = 250        # reps per condition
N_CALLS    = 5_000      # calls per condition
BASE_TRAINING = 6.5     # months (same for both conditions)

CONDITIONS = {
    "baseline": {
        "base_strain":  0.52,
        "pressure":     0.14,    # derived from Kaggle weekday pressure mean
        "label":        "Baseline",
        "color":        "#2E5FA3",
    },
    "high_pressure": {
        "base_strain":  0.72,    # +20pp strain (understaffed / high-volume scenario)
        "pressure":     0.62,    # +48pp pressure (peak/crisis staffing)
        "label":        "High Pressure",
        "color":        "#C45B1A",
    },
}

SCENARIO_MIX = {
    "clean":              0.52,
    "unresolvable_clean": 0.13,
    "gamed_metric":       0.12,
    "fraud_store_promo":  0.08,
    "fraud_line_add":     0.07,
    "fraud_hic_exchange": 0.04,
    "fraud_care_promo":   0.04,
}

GAMING_SCENARIOS = {"gamed_metric", "fraud_store_promo", "fraud_line_add",
                    "fraud_hic_exchange", "fraud_care_promo"}

# ── KPI synthesis (mirrors generate_employees_call_center_one_queue.py) ───────

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def z_noise(rng: random.Random, sigma: float = 0.15) -> float:
    u1 = max(1e-9, rng.random())
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2) * sigma

def synthesize_rep(rng: random.Random, base_strain: float,
                   pressure: float, training: float) -> dict:
    """
    Synthesize one representative's persona and KPI profile.
    Persona traits drawn from Kaggle prior means with Gaussian noise.
    """
    # Persona — sampled around Kaggle prior means
    patience           = clamp(0.505 + z_noise(rng, 0.08))
    empathy            = clamp(0.500 + z_noise(rng, 0.07))
    escalation_prone   = clamp(0.512 + z_noise(rng, 0.09))
    burnout_risk_prior = clamp(0.481 + z_noise(rng, 0.10))

    # Core state
    burnout    = clamp(0.55 * burnout_risk_prior
                       + 0.30 * base_strain
                       + 0.15 * (pressure - 0.5)
                       - 0.10 * patience
                       + z_noise(rng, 0.10))
    resilience = clamp(1.0 - burnout * 0.65
                       + (training / 12.0) * 0.20
                       + z_noise(rng, 0.08))
    volatility = clamp(0.30 + burnout * 0.60 + z_noise(rng, 0.12))

    # Quality signal
    qa = clamp(0.58
               + 0.12 * patience
               + 0.10 * (training / 12.0)
               - 0.18 * burnout
               + z_noise(rng, 0.06))

    # Outcome KPIs
    fcr = clamp(0.55
                + 0.18 * qa
                + 0.10 * patience
                - 0.18 * burnout
                - 0.08 * (pressure - 0.5)
                + z_noise(rng, 0.06), 0.10, 0.95)

    base_aht = 560.0
    aht = clamp(base_aht
                * (1.0 + 0.35 * burnout + 0.18 * (pressure - 0.5))
                * (1.0 - 0.08 * (training / 12.0))
                * (1.0 - 0.06 * qa)
                + z_noise(rng, 0.20) * 120,
                240.0, 1600.0)

    escalation = clamp(0.05
                       + 0.18 * escalation_prone
                       + 0.10 * burnout
                       - 0.12 * qa
                       + z_noise(rng, 0.04), 0.01, 0.55)

    repeat_rate = clamp(0.08
                        + 0.65 * (1.0 - fcr)
                        + 0.06 * (pressure - 0.5)
                        + z_noise(rng, 0.04), 0.02, 0.80)

    compliance = clamp(0.10
                       + 0.35 * burnout
                       + 0.25 * (1.0 - qa)
                       + 0.12 * escalation
                       + z_noise(rng, 0.06), 0.01, 0.95)

    aht_norm      = clamp((1600.0 - aht) / (1600.0 - 240.0))
    productivity  = clamp(0.42 * fcr + 0.33 * qa + 0.25 * aht_norm
                          - 0.15 * burnout + z_noise(rng, 0.04))

    # Strain tier
    strain_score = clamp(0.6 * base_strain + 0.4 * clamp(0.15 + 0.85 * burnout))
    if strain_score < 0.35:   strain_tier = "low"
    elif strain_score < 0.55: strain_tier = "medium"
    elif strain_score < 0.75: strain_tier = "high"
    else:                      strain_tier = "peak"

    # Gaming propensity: high burnout + high compliance risk = susceptible
    gaming_propensity = clamp(0.40 * burnout + 0.40 * compliance
                               + 0.20 * (1.0 - qa)
                               + z_noise(rng, 0.05))

    return {
        "fcr_30d":           round(fcr, 4),
        "qa_score":          round(qa, 4),
        "aht_secs":          round(aht, 2),
        "compliance_risk":   round(compliance, 4),
        "repeat_contact_rate": round(repeat_rate, 4),
        "burnout_index":     round(clamp(0.15 + 0.85 * burnout), 4),
        "resilience_index":  round(resilience, 4),
        "volatility_index":  round(volatility, 4),
        "productivity_index":round(productivity, 4),
        "escalation_rate":   round(escalation, 4),
        "strain_tier":       strain_tier,
        "strain_score":      round(strain_score, 4),
        "gaming_propensity": round(gaming_propensity, 4),
        "patience":          round(patience, 4),
        "empathy":           round(empathy, 4),
    }


def build_roster(condition_name: str, cfg: dict, n: int, rng: random.Random) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rep = synthesize_rep(rng, cfg["base_strain"], cfg["pressure"], BASE_TRAINING)
        rep["rep_id"]    = f"{condition_name.upper()[:3]}-REP{i+1:04d}"
        rep["condition"] = condition_name
        rows.append(rep)
    return pd.DataFrame(rows)


# ── Call simulation ───────────────────────────────────────────────────────────

def sample_scenario(rng: np.random.Generator) -> str:
    keys  = list(SCENARIO_MIX.keys())
    probs = np.array(list(SCENARIO_MIX.values()), dtype=float)
    probs /= probs.sum()
    return rng.choice(keys, p=probs)


CALL_TYPE_PRIORS = {
    "clean":              ["Billing Dispute", "Network Coverage", "Device Issue",
                           "Promotion Inquiry", "Account Inquiry", "Payment Arrangement"],
    "unresolvable_clean": ["Account Inquiry"],
    "gamed_metric":       ["Billing Dispute"],
    "fraud_store_promo":  ["Promotion Inquiry"],
    "fraud_line_add":     ["Account Inquiry"],
    "fraud_hic_exchange": ["Device Issue"],
    "fraud_care_promo":   ["Promotion Inquiry"],
}

CLEAN_CALL_TYPE_WEIGHTS = [0.30, 0.22, 0.18, 0.14, 0.10, 0.06]


def simulate_call(rng: np.random.Generator, rep: dict, scenario: str) -> dict:
    """
    Modulate outcome probabilities by rep's actual KPI profile.

    Key design decisions:
      - Rep gaming propensity biases toward gaming scenarios being
        ASSIGNED to that rep more often (via call_routed_to_gaming_rep flag)
      - FCR outcome for clean calls is sampled from rep's fcr_30d
      - Compliance event probability scales with rep's compliance_risk
      - Repeat contact probability is rep's repeat_contact_rate, scenario-adjusted
    """
    is_gaming = scenario in GAMING_SCENARIOS

    # True resolution (whether the issue is actually fixed)
    BASE_TRUE_RES = {
        "clean": 0.92, "unresolvable_clean": 0.10, "gamed_metric": 0.18,
        "fraud_store_promo": 0.25, "fraud_line_add": 0.22,
        "fraud_hic_exchange": 0.15, "fraud_care_promo": 0.30,
    }
    base_true_res  = BASE_TRUE_RES[scenario]
    # Rep FCR modulates true resolution on clean calls
    fcr_adj = (rep["fcr_30d"] - 0.70) * 0.3 if scenario == "clean" else 0.0
    true_res = bool(rng.random() < clamp(base_true_res + fcr_adj))

    # Proxy resolution (what gets marked in CRM — gaming inflates this)
    BASE_PROXY_RES = {
        "clean": 0.90, "unresolvable_clean": 0.55, "gamed_metric": 0.88,
        "fraud_store_promo": 0.60, "fraud_line_add": 0.55,
        "fraud_hic_exchange": 0.50, "fraud_care_promo": 0.65,
    }
    proxy_res = bool(rng.random() < BASE_PROXY_RES[scenario])

    # Repeat contact — rep's repeat_contact_rate modulates base prob
    BASE_REPEAT = {
        "clean": 0.10, "unresolvable_clean": 0.55, "gamed_metric": 0.52,
        "fraud_store_promo": 0.45, "fraud_line_add": 0.50,
        "fraud_hic_exchange": 0.48, "fraud_care_promo": 0.42,
    }
    repeat_adj = (rep["repeat_contact_rate"] - 0.245) * 0.5  # mean-centered
    repeat_30d = bool(rng.random() < clamp(BASE_REPEAT[scenario] + repeat_adj))

    # Compliance event — scales with rep compliance_risk
    compliance_event = bool(rng.random() < rep["compliance_risk"] * (1.4 if is_gaming else 0.5))

    # AHT — rep base AHT with scenario multiplier
    AHT_MULT = {
        "clean": 1.00, "unresolvable_clean": 1.25, "gamed_metric": 0.80,
        "fraud_store_promo": 1.30, "fraud_line_add": 1.45,
        "fraud_hic_exchange": 1.50, "fraud_care_promo": 1.20,
    }
    aht = rep["aht_secs"] * AHT_MULT[scenario] * (1.0 + rng.normal(0, 0.08))

    # ── Call type ─────────────────────────────────────────────────────────────
    ct_options = CALL_TYPE_PRIORS[scenario]
    if scenario == "clean":
        wts = np.array(CLEAN_CALL_TYPE_WEIGHTS[:len(ct_options)], dtype=float)
        wts /= wts.sum()
        call_type = str(rng.choice(ct_options, p=wts))
    else:
        call_type = ct_options[0]

    # ── Transcript ────────────────────────────────────────────────────────────
    agent_dict    = {"rep_name": rep.get("rep_id", "Agent"), "rep_id": rep.get("rep_id")}
    customer_dict = {"customer_id": f"CUST-{abs(hash(rep['rep_id']))%99999:05d}",
                     "account_id":  f"ACCT-{abs(hash(rep['rep_id']))%99999:05d}",
                     "monthly_charges": 85.0,
                     "lines_on_account": 2}
    scenario_meta = {"rep_aware_gaming": bool(rep.get("gaming_propensity", 0) > 0.65)}
    turns = build_transcript(scenario, call_type, agent_dict, customer_dict, scenario_meta, rng)
    transcript_text = transcript_to_text(turns)

    return {
        "scenario":          scenario,
        "call_type":         call_type,
        "is_gaming":         is_gaming,
        "true_resolution":   true_res,
        "proxy_resolution":  proxy_res,
        "repeat_30d":        repeat_30d,
        "compliance_event":  compliance_event,
        "aht_secs":          round(max(aht, 120.0), 1),
        "rep_fcr":           rep["fcr_30d"],
        "rep_compliance":    rep["compliance_risk"],
        "rep_burnout":       rep["burnout_index"],
        "rep_gaming_prop":   rep["gaming_propensity"],
        "rep_strain_tier":   rep["strain_tier"],
        "transcript":        transcript_text,
    }


def run_condition(condition_name: str, cfg: dict,
                  np_rng: np.random.Generator,
                  py_rng: random.Random) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate roster + simulate calls for one condition."""
    print(f"  Synthesizing {N_REPS} reps [{cfg['label']}]...")
    roster = build_roster(condition_name, cfg, N_REPS, py_rng)

    print(f"  Simulating {N_CALLS:,} calls [{cfg['label']}]...")
    rep_pool = roster.to_dict("records")
    call_rows = []

    for i in range(N_CALLS):
        # Sample a rep, with gaming-prone reps slightly more likely on gaming scenarios
        scenario = sample_scenario(np_rng)
        if scenario in GAMING_SCENARIOS:
            # Weight toward reps with higher gaming propensity
            gp = np.array([r["gaming_propensity"] for r in rep_pool])
            gp_norm = gp / gp.sum()
            idx = int(np_rng.choice(len(rep_pool), p=gp_norm))
        else:
            idx = int(np_rng.integers(0, len(rep_pool)))

        rep = rep_pool[idx]
        result = simulate_call(np_rng, rep, scenario)
        result["call_id"]    = f"{condition_name.upper()[:3]}-CALL-{i+1:06d}"
        result["condition"]  = condition_name
        result["rep_id"]     = rep["rep_id"]
        call_rows.append(result)

    calls_df = pd.DataFrame(call_rows)
    return roster, calls_df


# ── Analysis ─────────────────────────────────────────────────────────────────

def compute_summary(calls: pd.DataFrame, roster: pd.DataFrame,
                    condition_name: str, cfg: dict) -> dict:
    gaming_calls = calls[calls["is_gaming"]]
    return {
        "condition":               condition_name,
        "label":                   cfg["label"],
        "base_strain":             cfg["base_strain"],
        "pressure":                cfg["pressure"],
        # Rep-level (roster)
        "roster_mean_fcr":         round(roster["fcr_30d"].mean(), 4),
        "roster_mean_compliance":  round(roster["compliance_risk"].mean(), 4),
        "roster_mean_burnout":     round(roster["burnout_index"].mean(), 4),
        "roster_mean_aht":         round(roster["aht_secs"].mean(), 1),
        "roster_pct_high_strain":  round((roster["strain_tier"].isin(["high","peak"])).mean(), 4),
        # Call-level outcomes
        "call_mean_fcr":           round(calls["rep_fcr"].mean(), 4),
        "call_true_res_rate":      round(calls["true_resolution"].mean(), 4),
        "call_proxy_res_rate":     round(calls["proxy_resolution"].mean(), 4),
        "call_res_gap":            round(calls["proxy_resolution"].mean() - calls["true_resolution"].mean(), 4),
        "call_repeat_30d_rate":    round(calls["repeat_30d"].mean(), 4),
        "call_compliance_event_rate": round(calls["compliance_event"].mean(), 4),
        "call_mean_aht":           round(calls["aht_secs"].mean(), 1),
        # Gaming concentration
        "gaming_call_share":       round(calls["is_gaming"].mean(), 4),
        "gaming_mean_compliance":  round(gaming_calls["rep_compliance"].mean(), 4),
        "gaming_mean_burnout":     round(gaming_calls["rep_burnout"].mean(), 4),
        "gaming_repeat_rate":      round(gaming_calls["repeat_30d"].mean(), 4),
        "n_calls":                 len(calls),
        "n_reps":                  len(roster),
    }


# ── Figures ───────────────────────────────────────────────────────────────────

PALETTE = {
    "baseline":      "#2E5FA3",
    "high_pressure": "#C45B1A",
}
LABELS = {
    "baseline":      "Baseline",
    "high_pressure": "High Pressure",
}


def bar_comparison(metric_pairs: list[tuple[str, str, str]],
                   summaries: dict, title: str,
                   fname: str, higher_is_bad: bool = False) -> None:
    """
    metric_pairs: [(label, baseline_key, hp_key), ...]
    """
    n = len(metric_pairs)
    x = np.arange(n)
    w = 0.32

    fig, ax = plt.subplots(figsize=(max(8, n * 1.8), 5.2))
    fig.patch.set_facecolor("white")

    b_vals = [summaries["baseline"][k] for _, k, _ in metric_pairs]
    hp_vals = [summaries["high_pressure"][k] for _, k, _ in metric_pairs]
    labels_x = [l for l, _, _ in metric_pairs]
    # optional override key unused but kept for future
    bars_b  = ax.bar(x - w/2, b_vals,  w, color=PALETTE["baseline"],      label="Baseline",      alpha=0.88, zorder=3)
    bars_hp = ax.bar(x + w/2, hp_vals, w, color=PALETTE["high_pressure"], label="High Pressure", alpha=0.88, zorder=3)

    for bar, val in [(bars_b, b_vals), (bars_hp, hp_vals)]:
        for b, v in zip(bar, val):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#333")

    # Delta annotation
    for i, (bv, hv) in enumerate(zip(b_vals, hp_vals)):
        delta = hv - bv
        sign  = "+" if delta >= 0 else ""
        color = "#8B0000" if (delta > 0) == higher_is_bad else "#1A7A6E"
        ax.text(i, max(bv, hv) + 0.030, f"Δ{sign}{delta:.3f}",
                ha="center", va="bottom", fontsize=8.5, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    ax.set_ylim(0, max(max(b_vals), max(hp_vals)) * 1.28)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Saved: {fname}")


def rep_distribution_plot(rosters: dict[str, pd.DataFrame], metric: str,
                           xlabel: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")

    for cond, roster in rosters.items():
        vals = roster[metric].dropna()
        ax.hist(vals, bins=30, alpha=0.55, color=PALETTE[cond],
                label=LABELS[cond], density=True, edgecolor="white", linewidth=0.4)
        ax.axvline(vals.mean(), color=PALETTE[cond], linestyle="--",
                   linewidth=1.8, label=f"{LABELS[cond]} mean = {vals.mean():.3f}")

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Rep Population Distribution: {xlabel}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Saved: {fname}")


def gaming_heatmap(calls_dict: dict[str, pd.DataFrame], fname: str) -> None:
    """Scenario × condition call share table rendered as heatmap."""
    rows = []
    for cond, calls in calls_dict.items():
        sc_share = calls["scenario"].value_counts(normalize=True)
        for sc, p in sc_share.items():
            rows.append({"condition": LABELS[cond], "scenario": sc, "share": p})
    df = pd.DataFrame(rows).pivot(index="scenario", columns="condition", values="share")
    df["delta_pp"] = (df["High Pressure"] - df["Baseline"]) * 100

    scenarios_ordered = [
        "clean", "unresolvable_clean", "gamed_metric",
        "fraud_store_promo", "fraud_line_add", "fraud_hic_exchange", "fraud_care_promo"
    ]
    df = df.reindex([s for s in scenarios_ordered if s in df.index])

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("white")

    colors = []
    for sc in df.index:
        is_gam = sc in GAMING_SCENARIOS
        delta = df.loc[sc, "delta_pp"]
        if is_gam:
            colors.append("#C45B1A" if delta > 0 else "#8B0000")
        else:
            colors.append("#2E5FA3" if delta < 0 else "#555")

    y = np.arange(len(df))
    bars = ax.barh(y, df["delta_pp"], color=colors, alpha=0.80, zorder=3)

    for bar, val in zip(bars, df["delta_pp"]):
        sign = "+" if val >= 0 else ""
        offset = 0.12 if val >= 0 else -0.12
        ax.text(val + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{val:.2f}pp", va="center",
                ha="left" if val >= 0 else "right", fontsize=8.5, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=9)
    ax.axvline(0, color="#888", linewidth=1.0)
    ax.set_xlabel("Δ Call Share (High Pressure − Baseline, percentage points)", fontsize=9)
    ax.set_title("Scenario Mix Shift Under High Pressure", fontsize=12, fontweight="bold")
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    legend_items = [
        mpatches.Patch(color="#2E5FA3", alpha=0.8, label="Non-gaming (share decreases)"),
        mpatches.Patch(color="#C45B1A", alpha=0.8, label="Gaming (share increases)"),
    ]
    ax.legend(handles=legend_items, fontsize=8.5, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Saved: {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("NovaWireless Ecosystem Pressure Experiment")
    print(f"  N_REPS:  {N_REPS} per condition")
    print(f"  N_CALLS: {N_CALLS:,} per condition")
    print(f"  Seed:    {SEED}")
    print("=" * 60)

    rng_seed = SEED
    np_rng  = np.random.default_rng(rng_seed)
    py_rng  = random.Random(rng_seed)

    all_rosters = {}
    all_calls   = {}
    summaries   = {}

    for condition, cfg in CONDITIONS.items():
        print(f"\n[{cfg['label']}]  strain={cfg['base_strain']}  pressure={cfg['pressure']}")
        roster, calls = run_condition(condition, cfg, np_rng, py_rng)
        all_rosters[condition] = roster
        all_calls[condition]   = calls
        summaries[condition]   = compute_summary(calls, roster, condition, cfg)

    # ── Print report ──────────────────────────────────────────────────────────
    b = summaries["baseline"]
    h = summaries["high_pressure"]

    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    metrics = [
        ("Rep Population (Roster Level)", [
            ("Mean FCR",               "roster_mean_fcr",        False),
            ("Mean Compliance Risk",   "roster_mean_compliance", True),
            ("Mean Burnout Index",     "roster_mean_burnout",    True),
            ("Mean AHT (secs)",        "roster_mean_aht",        True),
            ("% High/Peak Strain",     "roster_pct_high_strain", True),
        ]),
        ("Call Outcomes", [
            ("True Resolution Rate",   "call_true_res_rate",          False),
            ("Proxy Resolution Rate",  "call_proxy_res_rate",         None),
            ("Res Gap (proxy−true)",   "call_res_gap",                True),
            ("Repeat Contact 30d",     "call_repeat_30d_rate",        True),
            ("Compliance Event Rate",  "call_compliance_event_rate",  True),
            ("Mean AHT (secs)",        "call_mean_aht",               True),
        ]),
        ("Gaming Concentration", [
            ("Gaming Call Share",       "gaming_call_share",      True),
            ("Gaming Mean Compliance",  "gaming_mean_compliance", True),
            ("Gaming Mean Burnout",     "gaming_mean_burnout",    True),
            ("Gaming Repeat Rate",      "gaming_repeat_rate",     True),
        ]),
    ]

    for section, items in metrics:
        print(f"\n  ── {section} ──")
        print(f"  {'Metric':<30} {'Baseline':>10} {'HighPres':>10} {'Delta':>10}")
        print(f"  {'-'*62}")
        for label, key, higher_is_bad in items:
            bv = b[key]
            hv = h[key]
            delta = hv - bv
            sign  = "+" if delta >= 0 else ""
            flag  = "▲" if (delta > 0 and higher_is_bad) else ("▼" if (delta < 0 and higher_is_bad is False) else "")
            print(f"  {label:<30} {bv:>10.4f} {hv:>10.4f} {sign}{delta:>+9.4f}  {flag}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print(f"\nGenerating figures → {FIG_DIR}")

    # Figure 1: FCR + Compliance + Repeat
    bar_comparison(
        [
            ("True FCR",          "call_true_res_rate",         "call_true_res_rate"),
            ("Proxy FCR",         "call_proxy_res_rate",        "call_proxy_res_rate"),
            ("Resolution Gap",    "call_res_gap",               "call_res_gap"),
            ("Repeat 30d",        "call_repeat_30d_rate",       "call_repeat_30d_rate"),
        ],
        summaries,
        "Figure 1 — FCR & Repeat Contact: Baseline vs. High Pressure",
        "fig1_fcr_repeat.png",
        higher_is_bad=False,
    )

    # Figure 2: Compliance + Burnout + Strain
    bar_comparison(
        [
            ("Compliance Risk",   "roster_mean_compliance", "roster_mean_compliance"),
            ("Burnout Index",     "roster_mean_burnout",    "roster_mean_burnout"),
            ("% High/Peak Strain","roster_pct_high_strain", "roster_pct_high_strain"),
            ("Compliance Events", "call_compliance_event_rate", "call_compliance_event_rate"),
        ],
        summaries,
        "Figure 2 — Risk & Strain Accumulation: Baseline vs. High Pressure",
        "fig2_compliance_strain.png",
        higher_is_bad=True,
    )

    # Figure 3: Rep population distributions
    rep_distribution_plot(all_rosters, "fcr_30d",
                          "First Contact Resolution (FCR)", "fig3_fcr_distribution.png")
    rep_distribution_plot(all_rosters, "compliance_risk",
                          "Compliance Risk", "fig4_compliance_distribution.png")

    # Figure 4: Gaming scenario shift
    gaming_heatmap(all_calls, "fig5_gaming_shift.png")

    # ── Write outputs ─────────────────────────────────────────────────────────
    print(f"\nWriting data outputs → {OUTPUT_DIR}")

    all_roster_df = pd.concat(all_rosters.values(), ignore_index=True)
    all_calls_df  = pd.concat(all_calls.values(),   ignore_index=True)
    summary_df    = pd.DataFrame(summaries.values())

    all_roster_df.to_csv(OUTPUT_DIR / "experiment_rep_rosters.csv", index=False)
    all_calls_df.to_csv( OUTPUT_DIR / "experiment_calls.csv",       index=False)
    summary_df.to_csv(   OUTPUT_DIR / "experiment_summary.csv",     index=False)

    print(f"  experiment_rep_rosters.csv  ({len(all_roster_df)} rows)")
    print(f"  experiment_calls.csv        ({len(all_calls_df):,} rows)")
    print(f"  experiment_summary.csv      (2 rows)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
