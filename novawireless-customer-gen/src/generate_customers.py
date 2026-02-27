#!/usr/bin/env python3
"""
NovaWireless — Customer + Account Graph Generator (Governance-Ready)

Generates:
- customers_v1.csv (and customers.csv compat)
- lines.csv
- devices.csv
- eip_agreements.csv
- line_device_usage.csv

Key guarantees:
- Every line has msisdn + line_id
- Every line has a "current" IMEI (device assigned)
- Usage table records usage_imei for every line (unless later anomaly injection blanks it)
- EIP agreements exist for:
  - Most voice lines when customer has device_payment_plan=Yes (configurable)
  - All 5G Home Internet lines (company-owned equipment / lease-like)
- Optional injection of EIP-vs-usage IMEI mismatch at generation time (lightweight)

Writes to BOTH:
- output/ (debug + receipts + compat)
- data/external/params_sources/ (what call generator expects)

Run:
  python src/generate_customers.py
  python src/generate_customers.py --n_customers 50000 --seed 123
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Repo helpers
# --------------------------------------------------------------------------------------
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

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Sampling utilities
# --------------------------------------------------------------------------------------
def _normalize_probs(items: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
    total = float(sum(float(p) for _, p in items))
    if total <= 0:
        raise ValueError("Probabilities sum to 0; cannot normalize.")
    return [(k, float(p) / total) for k, p in items]


def sample_categorical(rng: np.random.Generator, dist: Dict[str, float], n: int) -> np.ndarray:
    items = _normalize_probs(list(dist.items()))
    keys = [k for k, _ in items]
    probs = np.array([p for _, p in items], dtype=float)
    return rng.choice(keys, size=n, replace=True, p=probs)


def parse_interval_label(label: str) -> Tuple[float, float]:
    s = str(label).strip()
    s = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Unrecognized interval label: {label}")
    return float(parts[0]), float(parts[1])


def sample_binned_numeric_uniform(
    rng: np.random.Generator,
    bins_prob: Dict[str, float],
    n: int,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    integer: bool = False,
) -> np.ndarray:
    items = _normalize_probs(list(bins_prob.items()))
    labels = [lab for lab, _ in items]
    probs = np.array([p for _, p in items], dtype=float)

    chosen = rng.choice(labels, size=n, replace=True, p=probs)

    lows = np.empty(n, dtype=float)
    highs = np.empty(n, dtype=float)
    for i, lab in enumerate(chosen):
        lo, hi = parse_interval_label(lab)
        lows[i] = lo
        highs[i] = hi

    vals = rng.uniform(lows, highs)
    if clamp_min is not None:
        vals = np.maximum(vals, clamp_min)
    if clamp_max is not None:
        vals = np.minimum(vals, clamp_max)

    if integer:
        vals = np.floor(vals + 1e-9).astype(int)
    return vals


# --------------------------------------------------------------------------------------
# Business rules
# --------------------------------------------------------------------------------------
def multiple_lines_to_line_count(rng: np.random.Generator, ml: str) -> int:
    ml = str(ml).strip().lower()
    if ml == "no phone service":
        return 0
    if ml == "no":
        return 1
    if ml == "yes":
        k = int(rng.geometric(p=0.55))
        return int(min(2 + (k - 1), 5))
    return 1


def remap_internet_to_5g_only(raw_internet: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Force internet offering to cellular-only 5G Home Internet.
    Anything that isn't "No" becomes "5G Home Internet".
    """
    raw = pd.Series(raw_internet).astype(str).str.strip()
    has_internet = (raw.str.lower() != "no").to_numpy().astype(int)
    mapped = np.where(has_internet == 1, "5G Home Internet", "No")
    return mapped, has_internet


def has_device_plan_from_contract_proxy(
    rng: np.random.Generator,
    contract_proxy: str,
    month_to_month_plan_rate: float,
) -> bool:
    c = str(contract_proxy).strip().lower()
    if c in {"one year", "two year"}:
        return True
    if c == "month-to-month":
        return rng.random() < float(month_to_month_plan_rate)
    return rng.random() < float(month_to_month_plan_rate)


def sample_device_months_remaining(rng: np.random.Generator, tenure_months: int, term: int) -> int:
    term = int(term)
    if term <= 0:
        return 0
    progress = float(np.clip(tenure_months / 48.0, 0.0, 1.0))
    a = 1.5 + 4.0 * progress
    b = 3.5
    frac_paid = float(rng.beta(a, b))
    months_paid = int(np.clip(round(frac_paid * term), 0, term - 1))
    remaining = term - months_paid
    return int(np.clip(remaining, 1, term))


def sample_device_monthly_payment(rng: np.random.Generator, has_plan: bool, monthly_charges: float) -> float:
    if not has_plan:
        return 0.0
    share = float(np.clip(rng.normal(loc=0.22, scale=0.07), 0.08, 0.40))
    amt = float(monthly_charges) * share
    amt = float(np.clip(amt, 10.0, 55.0))
    return round(amt, 2)


# --------------------------------------------------------------------------------------
# IMEI / MSISDN / Agreement generators
# --------------------------------------------------------------------------------------
def luhn_checksum(digits: List[int]) -> int:
    s = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d2 = d * 2
            s += d2 - 9 if d2 > 9 else d2
        else:
            s += d
    return (10 - (s % 10)) % 10


def make_imei(rng: np.random.Generator) -> str:
    body = [int(x) for x in rng.integers(0, 10, size=14)]
    chk = luhn_checksum(body)
    return "".join(map(str, body)) + str(chk)


def make_msisdn(rng: np.random.Generator) -> str:
    npa = rng.integers(200, 999)
    nxx = rng.integers(200, 999)
    xxxx = rng.integers(0, 9999)
    return f"{npa:03d}{nxx:03d}{xxxx:04d}"


def make_agreement_number(rng: np.random.Generator) -> str:
    return f"EIP-{rng.integers(10**9, 10**10 - 1)}"


@dataclass
class LineGenConfig:
    usage_snapshot_date: str = "2026-02-22"
    p_inject_eip_usage_mismatch: float = 0.06
    mismatch_within_customer_only: bool = True
    p_voice_eip_attach_if_plan: float = 0.85  # if customer has plan, attach EIP to voice line at this rate


def build_account_graph(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    line_cfg: LineGenConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds:
      lines.csv
      devices.csv
      eip_agreements.csv
      line_device_usage.csv
    """
    # 1) Lines
    line_rows: List[Dict[str, Any]] = []
    line_id_counter = 1

    for _, c in customers.iterrows():
        cust_id = str(c["customer_id"])
        acct_id = str(c["account_id"])
        voice_lines = int(c["lines_on_account"])
        has_hsi = int(c.get("has_5g_home_internet", 0))

        for _k in range(max(voice_lines, 0)):
            line_rows.append(
                {
                    "line_id": f"LINE-{line_id_counter:09d}",
                    "customer_id": cust_id,
                    "account_id": acct_id,
                    "msisdn": make_msisdn(rng),
                    "product_type": "voice",
                    "status": rng.choice(["active", "suspended"], p=[0.93, 0.07]),
                    "company_owned_equipment_flag": 0,
                }
            )
            line_id_counter += 1

        if has_hsi == 1:
            line_rows.append(
                {
                    "line_id": f"LINE-{line_id_counter:09d}",
                    "customer_id": cust_id,
                    "account_id": acct_id,
                    "msisdn": make_msisdn(rng),
                    "product_type": "5g_home_internet",
                    "status": "active",
                    "company_owned_equipment_flag": 1,
                }
            )
            line_id_counter += 1

    lines = pd.DataFrame(line_rows)
    if lines.empty:
        lines = pd.DataFrame(
            columns=["line_id", "customer_id", "account_id", "msisdn", "product_type", "status", "company_owned_equipment_flag"]
        )

    # 2) Devices inventory (one per line + buffer)
    n_devices = int(max(len(lines) * 1.15, 1))
    devices = pd.DataFrame(
        {
            "device_id": [f"DEV-{i:09d}" for i in range(1, n_devices + 1)],
            "imei": [make_imei(rng) for _ in range(n_devices)],
            "device_category": rng.choice(["handset", "gateway_5g_hsi"], size=n_devices, p=[0.82, 0.18]),
        }
    )

    # Assign IMEI to each line
    hsi_mask = (lines["product_type"] == "5g_home_internet").to_numpy()
    handset_pool = devices[devices["device_category"] == "handset"].sample(frac=1.0, random_state=1).reset_index(drop=True)
    gateway_pool = devices[devices["device_category"] == "gateway_5g_hsi"].sample(frac=1.0, random_state=2).reset_index(drop=True)

    handset_i = 0
    gateway_i = 0
    current_imei: List[str] = []

    for is_hsi in hsi_mask:
        if is_hsi:
            if gateway_i >= len(gateway_pool):
                imei = devices.loc[int(rng.integers(0, len(devices))), "imei"]
            else:
                imei = gateway_pool.loc[gateway_i, "imei"]
                gateway_i += 1
        else:
            if handset_i >= len(handset_pool):
                imei = devices.loc[int(rng.integers(0, len(devices))), "imei"]
            else:
                imei = handset_pool.loc[handset_i, "imei"]
                handset_i += 1
        current_imei.append(str(imei))

    lines["current_imei"] = current_imei

    # 3) EIP agreements
    eip_rows: List[Dict[str, Any]] = []
    cust_plan = customers.set_index("customer_id")["device_payment_plan"].to_dict()

    for _, ln in lines.iterrows():
        cust_id = str(ln["customer_id"])
        prod = str(ln["product_type"])
        has_plan = str(cust_plan.get(cust_id, "No")) == "Yes"

        attach = False
        if prod == "5g_home_internet":
            attach = True
        else:
            attach = bool(has_plan and (rng.random() < float(line_cfg.p_voice_eip_attach_if_plan)))

        if not attach:
            continue

        eip_rows.append(
            {
                "agreement_number": make_agreement_number(rng),
                "customer_id": cust_id,
                "account_id": str(ln["account_id"]),
                "line_id": str(ln["line_id"]),
                "msisdn": str(ln["msisdn"]),
                "eip_imei": str(ln["current_imei"]),
                "product_type": prod,
                "eip_status": rng.choice(["active", "active", "closed"], p=[0.95, 0.03, 0.02]),
                "installment_months": int(rng.choice([24, 30, 36], p=[0.35, 0.25, 0.40])) if prod == "voice" else 0,
            }
        )

    eip = pd.DataFrame(eip_rows)

    # 4) Usage mapping
    usage = lines[["customer_id", "account_id", "line_id", "msisdn", "product_type", "current_imei"]].copy()
    usage = usage.rename(columns={"current_imei": "usage_imei"})
    usage["usage_snapshot_date"] = str(line_cfg.usage_snapshot_date)
    usage["imei_line_mismatch_flag"] = 0
    usage["usage_eip_agreement_number"] = ""

    if not eip.empty:
        eip_lookup = eip.set_index("line_id")["agreement_number"].to_dict()
        usage["usage_eip_agreement_number"] = usage["line_id"].map(eip_lookup).fillna("")

    # Optional mismatch injection (lightweight)
    if (not eip.empty) and float(line_cfg.p_inject_eip_usage_mismatch) > 0:
        eip_line_ids = set(eip["line_id"].astype(str).tolist())
        eip_idx = usage.index[usage["line_id"].astype(str).isin(eip_line_ids)].to_numpy()
        n_mismatch = int(len(eip_idx) * float(line_cfg.p_inject_eip_usage_mismatch))

        if n_mismatch > 0 and len(eip_idx) > 1:
            chosen = rng.choice(eip_idx, size=n_mismatch, replace=False)
            for li in chosen:
                cust = usage.at[li, "customer_id"]
                if line_cfg.mismatch_within_customer_only:
                    pool = usage.index[(usage["customer_id"] == cust) & (usage.index != li)].to_numpy()
                else:
                    pool = usage.index[usage.index != li].to_numpy()
                if len(pool) == 0:
                    continue
                donor = int(rng.choice(pool))
                usage.at[li, "usage_imei"] = usage.at[donor, "usage_imei"]
                usage.at[li, "imei_line_mismatch_flag"] = 1

    return lines, devices, eip, usage


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_customers", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--month_to_month_plan_rate", type=float, default=0.55)
    ap.add_argument("--p_inject_eip_usage_mismatch", type=float, default=0.06)
    ap.add_argument("--p_voice_eip_attach_if_plan", type=float, default=0.85)
    ap.add_argument("--usage_snapshot_date", type=str, default="2026-02-22")
    args = ap.parse_args()

    root = find_repo_root()

    # ---------------------------
    # Baseline payload: try config, else fallback
    # ---------------------------
    candidates = [
        root / "config" / "novawireless_public_baseline_config.json",
        root / "src" / "config" / "novawireless_public_baseline_config.json",
        Path(__file__).resolve().parent / "config" / "novawireless_public_baseline_config.json",
    ]

    baseline_path = next((p for p in candidates if p.exists()), None)

    if baseline_path is not None:
        cfg = load_json(baseline_path)
        reg = cfg.get("registry", {})
        customer_key = reg.get("customer_params")
        if not customer_key:
            raise ValueError("Baseline config missing registry.customer_params")
        payload = cfg["sources"][customer_key]["payload"]
    else:
        # Fallback: built-in telco-like baseline (portable)
        payload = {
            "churn_rate": 0.27,
            "tenure": {
                "min": 0,
                "max": 72,
                "bins_prob": {
                    "[0, 12]": 0.34,
                    "[12, 24]": 0.20,
                    "[24, 36]": 0.16,
                    "[36, 48]": 0.14,
                    "[48, 60]": 0.10,
                    "[60, 72]": 0.06,
                },
            },
            "monthly_charges": {
                "min": 20,
                "max": 120,
                "bins_prob": {
                    "[20, 40]": 0.18,
                    "[40, 60]": 0.28,
                    "[60, 80]": 0.26,
                    "[80, 100]": 0.18,
                    "[100, 120]": 0.10,
                },
            },
            "InternetService": {"No": 0.55, "DSL": 0.10, "Fiber optic": 0.35},  # remapped to 5G HINT or No
            "PaymentMethod": {
                "Electronic check": 0.30,
                "Mailed check": 0.10,
                "Bank transfer (automatic)": 0.35,
                "Credit card (automatic)": 0.25,
            },
            "MultipleLines": {"No": 0.40, "Yes": 0.55, "No phone service": 0.05},
            "Contract": {"Month-to-month": 0.62, "One year": 0.10, "Two year": 0.28},
            "PaperlessBilling": {"Yes": 0.60, "No": 0.40},
            "OnlineSecurity": {"Yes": 0.35, "No": 0.65},
            "OnlineBackup": {"Yes": 0.30, "No": 0.70},
            "DeviceProtection": {"Yes": 0.42, "No": 0.58},
            "StreamingTV": {"Yes": 0.33, "No": 0.67},
            "StreamingMovies": {"Yes": 0.28, "No": 0.72},
        }

    rng = np.random.default_rng(int(args.seed))
    n = int(args.n_customers)

    # Pull distributions from baseline payload (IBM telco-derived)
    global_churn = float(payload.get("churn_rate", 0.27))

    tenure_bins_prob = payload["tenure"]["bins_prob"]
    tenure_min = float(payload["tenure"]["min"])
    tenure_max = float(payload["tenure"]["max"])

    mc_bins_prob = payload["monthly_charges"]["bins_prob"]
    mc_min = float(payload["monthly_charges"]["min"])
    mc_max = float(payload["monthly_charges"]["max"])

    internet_dist = payload["InternetService"]
    payment_dist = payload["PaymentMethod"]
    multiple_lines_dist = payload["MultipleLines"]
    contract_proxy_dist = payload["Contract"]

    paperless_dist = payload.get("PaperlessBilling", {"Yes": 0.6, "No": 0.4})
    add_on_fields = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies"]
    add_on_dists = {k: payload[k] for k in add_on_fields if k in payload}

    tenure = sample_binned_numeric_uniform(rng, tenure_bins_prob, n=n, clamp_min=tenure_min, clamp_max=tenure_max, integer=True)
    monthly_charges = sample_binned_numeric_uniform(rng, mc_bins_prob, n=n, clamp_min=mc_min, clamp_max=mc_max, integer=False)
    monthly_charges = np.round(monthly_charges, 2)

    internet_raw = sample_categorical(rng, internet_dist, n)
    internet_service, has_5g = remap_internet_to_5g_only(internet_raw)

    payment = sample_categorical(rng, payment_dist, n)
    multiple_lines = sample_categorical(rng, multiple_lines_dist, n)
    paperless = sample_categorical(rng, paperless_dist, n)
    contract_proxy = sample_categorical(rng, contract_proxy_dist, n)

    add_ons: Dict[str, np.ndarray] = {}
    for k, dist in add_on_dists.items():
        add_ons[k] = sample_categorical(rng, dist, n)

    # Simple churn-risk score + trust baseline (kept lightweight; you can get fancier later)
    churn_risk = np.clip(rng.normal(loc=global_churn, scale=0.09, size=n), 0.01, 0.95)
    trust_baseline = np.clip(rng.normal(loc=72.0, scale=12.0, size=n), 5.0, 98.0)
    patience = np.clip(rng.beta(2.2, 2.0, size=n), 0.02, 0.98)

    # Derived fields
    lines_on_account = np.array([multiple_lines_to_line_count(rng, ml) for ml in multiple_lines], dtype=int)
    device_payment_plan = np.array(
        [
            "Yes" if has_device_plan_from_contract_proxy(rng, contract_proxy[i], float(args.month_to_month_plan_rate)) else "No"
            for i in range(n)
        ],
        dtype=object,
    )
    device_term = np.where(device_payment_plan == "Yes", rng.choice([24, 30, 36], size=n, p=[0.35, 0.25, 0.40]), 0).astype(int)
    device_months_remaining = np.array(
        [sample_device_months_remaining(rng, int(tenure[i]), int(device_term[i])) if device_term[i] > 0 else 0 for i in range(n)],
        dtype=int,
    )
    device_monthly_payment = np.array(
        [sample_device_monthly_payment(rng, device_payment_plan[i] == "Yes", float(monthly_charges[i])) for i in range(n)],
        dtype=float,
    )

    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i:07d}" for i in range(1, n + 1)],
            "account_id": [f"A{i:07d}" for i in range(1, n + 1)],
            "tenure_months": tenure.astype(int),
            "monthly_charges": monthly_charges.astype(float),
            "paperless_billing": paperless.astype(str),
            "payment_method": payment.astype(str),
            "contract_proxy": contract_proxy.astype(str),
            "internet_service": internet_service.astype(str),
            "has_5g_home_internet": has_5g.astype(int),
            "lines_on_account": lines_on_account.astype(int),
            "device_payment_plan": device_payment_plan.astype(str),
            "device_term_months": device_term.astype(int),
            "device_months_remaining": device_months_remaining.astype(int),
            "device_monthly_payment": np.round(device_monthly_payment.astype(float), 2),
            # add-ons
            **{k: v.astype(str) for k, v in add_ons.items()},
            # behavioral knobs/state
            "churn_risk_score": np.round(churn_risk, 6),
            "trust_baseline": np.round(trust_baseline, 6),
            "patience": np.round(patience, 6),
            # state placeholders used later in call simulation
            "is_churned": (rng.random(n) < churn_risk).astype(int),
            "repeat_contacts_30d": 0,
            "last_call_day_index": -1,
        }
    )

    # Build account graph tables
    line_cfg = LineGenConfig(
        usage_snapshot_date=str(args.usage_snapshot_date),
        p_inject_eip_usage_mismatch=float(args.p_inject_eip_usage_mismatch),
        mismatch_within_customer_only=True,
        p_voice_eip_attach_if_plan=float(args.p_voice_eip_attach_if_plan),
    )
    lines, devices, eip, usage = build_account_graph(rng=rng, customers=customers, line_cfg=line_cfg)

    # Output locations (output-only)
    out_dir = root / "output"
    ensure_dir(out_dir)

    # Customers
    (out_dir / "customers.csv").write_text("", encoding="utf-8")  # create file early if path watchers exist
    customers.to_csv(out_dir / "customers.csv", index=False)
    customers.to_csv(out_dir / "customers_v1.csv", index=False)  # canonical for downstream

    # Account graph
    lines.to_csv(out_dir / "lines.csv", index=False)
    devices.to_csv(out_dir / "devices.csv", index=False)
    eip.to_csv(out_dir / "eip_agreements.csv", index=False)
    usage.to_csv(out_dir / "line_device_usage.csv", index=False)
    
    receipt: Dict[str, Any] = {
        "dataset": "novawireless_customers_plus_account_graph",
        "run_ts": datetime.now().isoformat(timespec="seconds"),
        "n_customers": int(n),
        "seed": int(args.seed),
        "params": {
            "month_to_month_plan_rate": float(args.month_to_month_plan_rate),
            "p_inject_eip_usage_mismatch": float(args.p_inject_eip_usage_mismatch),
            "p_voice_eip_attach_if_plan": float(args.p_voice_eip_attach_if_plan),
            "usage_snapshot_date": str(args.usage_snapshot_date),
        },
        "counts": {
            "lines": int(len(lines)),
            "devices": int(len(devices)),
            "eip_agreements": int(len(eip)),
            "imei_mismatched_lines_generation_time": int((usage["imei_line_mismatch_flag"] == 1).sum()) if len(usage) else 0,
            "hsi_lines": int((lines["product_type"] == "5g_home_internet").sum()) if len(lines) else 0,
        },
        "outputs": {
            "output_customers_csv": "output/customers.csv",
            "params_customers_v1_csv": "data/external/params_sources/customers_v1.csv",
            "output_lines_csv": "output/lines.csv",
            "output_devices_csv": "output/devices.csv",
            "output_eip_agreements_csv": "output/eip_agreements.csv",
            "output_line_device_usage_csv": "output/line_device_usage.csv",
        },
        "notes": [
            "This generator is the upstream truth for account metadata required by Dirty Frank governance detection.",
            "Ledger build step (02) will flatten these tables into master_account_ledger.csv for call generation joins.",
        ],
    }

    save_json(receipt, out_dir / "customer_generation_receipt.json")
    save_json(receipt, out_dir / "customer_generation_receipt.json")

    print("[ok] wrote customers + account graph:")
    print(f" - {out_dir / 'customers.csv'}")
    print(f" - {out_dir / 'customers_v1.csv'}")
    print(f" - {out_dir / 'lines.csv'}")
    print(f" - {out_dir / 'eip_agreements.csv'}")
    print(f" - {out_dir / 'line_device_usage.csv'}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
