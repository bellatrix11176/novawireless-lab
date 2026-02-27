# novawireless-lab

A reproducible synthetic call center dataset framework for AI governance and fraud detection research.

Generates customers, representatives, and calls with realistic fraud scenarios, rep-state drift, service credits, and full dialogue transcripts — designed to support KPI integrity research and the detection of metric gaming in AI-optimized environments.

> *"Bad metrics don't just mislead dashboards — they contaminate training data and teach the system to believe its own scorecard."*
> — When KPIs Lie: Governance Signals for AI-Optimized Call Centers (Aulabaugh, 2026)

---

## What This Is

NovaWireless is a fictional wireless carrier. This lab simulates one year of call center operations — 60,000+ calls across 12 months — with ground-truth labels for fraud, metric gaming, and durable resolution that are never available in real operational data.

The dataset is designed to:

- Demonstrate **Goodhart's Law** and **Campbell's Law** in a measurable, reproducible environment
- Provide training and validation data for **KPI drift detection** models
- Support **governance signal** computation: DAR, DRL, DOV, POR, and SII as defined in the accompanying paper
- Power **Dirty Frank's Decision Engine** — a drift and fraud detection SaaS built on this framework

---

## Repository Structure

```
novawireless-lab/                          ← this repo (lab root)
    .labroot                               ← sentinel file for path resolution
    data/                                  ← shared inputs (not tracked by git)
        customers.csv
        master_account_ledger.csv
        novawireless_employee_database.csv
        employee_generation_inputs/        ← Kaggle/IBM-derived priors
    output/                                ← all generated files land here (not tracked)
    novawireless-customer-gen/             ← Customer Generator
    novawireless-rep-gen/                  ← Representative Generator
    novawireless-call-gen/                 ← Call Generator
```

Each generator is a self-contained repository nested inside the lab root. All scripts use the `.labroot` sentinel to resolve shared `data/` and `output/` paths automatically — no hardcoded paths, no manual configuration.

---

## The Three Generators

### 1. novawireless-customer-gen
Produces `customers.csv` and `master_account_ledger.csv`.

Generates synthetic wireless customers with tenure, churn risk, trust baseline, patience, device payment plans, and account ledger entries. Distributions are grounded in the IBM Telco Customer Churn dataset.

**Run this first.**

---

### 2. novawireless-rep-gen
Produces `novawireless_employee_database.csv`.

Generates 250 synthetic call center representatives with correlated KPI profiles: FCR, AHT, escalation rate, compliance risk, burnout index, resilience, and volatility. KPIs are synthesized as correlated proxies — not independent draws — so bad weeks cluster and high-pressure conditions degrade performance realistically.

**Run this second.**

---

### 3. novawireless-call-gen
Produces 36 files across 12 months: metadata CSVs, transcript JSONL files, and sanitized analysis-ready CSVs.

Simulates 5,000 calls per month with 10 scenario types, rep state drift, service credit tracking, and full turn-by-turn dialogue transcripts. The proxy KPI (`resolution_flag`) diverges from the ground truth (`true_resolution`) in ways that are measurable, detectable, and consistent with the paper's governance framework.

**Run this third — 12 times, once per month.**

---

## Scenario Mix

| Scenario | Share | Type |
|---|---|---|
| `clean` | 44% | Legitimate, correctly resolved |
| `unresolvable_clean` | 11% | Legitimate but genuinely unresolvable |
| `gamed_metric` | 10% | Rep games proxy KPI — bandaid credits, false closure |
| `fraud_store_promo` | 7% | Unauthorized store promotion |
| `fraud_line_add` | 6% | Line added without customer consent |
| `fraud_hic_exchange` | 3% | Erroneous NRF via incorrect HIC exchange |
| `fraud_care_promo` | 3% | Care rep offers unauthorized promotion |
| `activation_clean` | 8% | Successful device activation |
| `activation_failed` | 4% | Failed activation — SIM or IMEI error |
| `line_add_legitimate` | 4% | Customer legitimately adds a line |

---

## Key Signals in the Data

Every call record includes:

| Column | Description |
|---|---|
| `resolution_flag` | Proxy KPI — rep marks call resolved |
| `true_resolution` | Ground truth — was the issue actually fixed |
| `repeat_contact_30d` | Repeat within the gaming window |
| `repeat_contact_31_60d` | Repeat after the gaming window — the tell |
| `credit_applied` | Was a service credit issued |
| `credit_amount` | Dollar amount |
| `credit_type` | `courtesy`, `service_credit`, `bandaid`, `dispute_credit`, `fee_waiver` |
| `credit_authorized` | Was the credit within policy |
| `rep_gaming_propensity` | Rep state snapshot — drifts upward over gamed calls |
| `rep_burnout_level` | Rep state snapshot — increases with escalations |
| `customer_trust_baseline` | Decays per scenario across the simulation |

The `bandaid` credit type is the core gaming signal: a rep applies a credit not to fix the problem but to suppress a repeat contact within the 30-day FCR window. `credit_type=bandaid` + `repeat_contact_31_60d=True` = the issue resurfaced after the gaming window. That pattern is detectable.

---

## Governance Signals (Paper)

This framework operationalizes the following metrics from *When KPIs Lie* (Aulabaugh, 2026):

| Signal | Definition | Columns Used |
|---|---|---|
| **DAR** | Delayed Adverse Rate — failures after labeled success | `resolution_flag`, `repeat_contact_31_60d` |
| **DRL** | Downstream Remediation Load — drift in post-success workload | `is_repeat_call`, `call_type`, `parent_call_id` |
| **DOV** | Durable Outcome Validation — decay in proxy predictive validity | `resolution_flag`, `true_resolution`, `call_date` |
| **POR** | Proxy Overfit Ratio — acceleration gap between proxy and truth | `resolution_flag`, `true_resolution` rolling delta |
| **SII** | System Integrity Index — weighted composite governance score | All of the above |

---

## Setup

### Prerequisites
- Python 3.10+
- pip

### Install dependencies
```bash
pip install -r requirements.txt
```

Dependencies are the same across all three generators:
```
numpy>=1.24
pandas>=2.0
```

### Place source data
The `data/` folder is not tracked by git. You will need to generate or supply:
- `customers.csv` — output of novawireless-customer-gen
- `master_account_ledger.csv` — output of novawireless-customer-gen
- `novawireless_employee_database.csv` — output of novawireless-rep-gen
- `data/employee_generation_inputs/` — Kaggle/IBM-derived priors (see novawireless-rep-gen docs)

### Generate the full dataset
```bash
# Step 1 — Customers
python novawireless-customer-gen/src/run_all.py

# Step 2 — Representatives
python novawireless-rep-gen/src/run_all.py

# Step 3 — Calls (run 12 times or use the append script)
python novawireless-call-gen/src/call_gen__run_all.py
python novawireless-call-gen/src/01b_generate_calls_append.py  # x11
```

All scripts are F5-runnable in IDLE with no arguments. Auto-detection handles month sequencing and file naming.

---

## Data Sources and Citations

This project uses distributional parameters derived from publicly available datasets. No raw source data is included in this repository. All outputs are fully synthetic.

Sources include the IBM Telco Customer Churn dataset, FCC CGB Consumer Complaints data, Kaggle call center and employee datasets, and MTA contact center performance data. Full citations are in each generator's `docs/` folder.

---

## Research Context

This lab was built to support:

- **When KPIs Lie: Governance Signals for AI-Optimized Call Centers** (Aulabaugh, 2026) — formal definitions of DAR, DRL, DOV, POR, and SII
- **Dirty Frank's Decision Engine** — a forthcoming drift and fraud detection SaaS that consumes this dataset to demonstrate real-time KPI integrity monitoring

---

## License

MIT License — see LICENSE for details.

---

## Author

Gina Aulabaugh — February 2026
