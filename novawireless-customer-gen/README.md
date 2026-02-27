# NovaWireless Synthetic Customer Generator

Generates a synthetic customer base and account graph for NovaWireless — a fictional wireless carrier. Produces the data files required by the NovaWireless Call Generator.

---

## What It Generates

| File | Description |
|---|---|
| `customers.csv` | 10,000 synthetic customer accounts with behavioral attributes |
| `lines.csv` | Per-line records (voice + 5G home internet) |
| `eip_agreements.csv` | Equipment Installment Plan agreements |
| `line_device_usage.csv` | Device usage snapshots per line |
| `master_account_ledger.csv` | Flattened ledger joining all of the above, with IMEI flags |

All data is synthetic and derived from publicly available telco datasets (IBM Telco Churn, FCC CGB Consumer Complaints). No real customer data is used.

---

## Quickstart

```bash
git clone <repo>
cd novawireless-customer-generator
pip install -r requirements.txt
python src/run_all.py
```

Output files are written to `output/`. Copy `customers.csv` and `master_account_ledger.csv` to the Call Generator's `data/` folder.

### Options

```bash
python src/run_all.py --n_customers 5000 --seed 99
```

| Argument | Default | Description |
|---|---|---|
| `--n_customers` | 10000 | Number of customer accounts to generate |
| `--seed` | 42 | Random seed for reproducibility |
| `--p_inject_eip_usage_mismatch` | 0.06 | Rate of EIP/usage IMEI mismatches injected |
| `--p_voice_eip_attach_if_plan` | 0.85 | Probability a voice line has an EIP when customer has a device plan |
| `--usage_snapshot_date` | 2026-02-22 | Snapshot date for usage records |

---

## Repo Structure

```
novawireless-customer-generator/
│
├── data/
│   └── external/
│       └── params_sources/         ← intermediate files written by pipeline
│
├── src/
│   ├── run_all.py                  ← run this
│   ├── generate_customers.py       ← step 1: customers + account graph
│   ├── 02_build_master_account_ledger.py  ← step 2: flatten to ledger
│   ├── 03_inject_imei_anomalies.py        ← step 3: inject IMEI defects
│   └── fix_ledger_contract_proxy.py       ← step 4: fix billing agreement field
│
├── output/                         ← generated files (gitignored)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Pipeline Steps

Each step can also be run individually:

```bash
python src/generate_customers.py
python src/02_build_master_account_ledger.py
python src/03_inject_imei_anomalies.py
python src/fix_ledger_contract_proxy.py
```

**Step 1 — `generate_customers.py`**
Samples customer attributes from IBM Telco-derived distributions. Generates account graph tables (lines, devices, EIP agreements, usage snapshots). Derives `churn_risk_score` per customer from tenure, service type, and payment method.

**Step 2 — `02_build_master_account_ledger.py`**
Joins customers, lines, EIP agreements, and usage into a single flat ledger. Computes IMEI mismatch flags and upstream friction risk signals.

**Step 3 — `03_inject_imei_anomalies.py`**
Injects Goodhart-style data defects: IMEI swaps between lines (simulating device used on wrong account) and missing usage IMEIs (ghost lines / capture failure). Rates configurable via CLI.

**Step 4 — `fix_ledger_contract_proxy.py`**
Removes the upstream `contract_proxy` column (an artifact of IBM Telco source data). Replaces it with `billing_agreement_type` derived from actual EIP data. NovaWireless has no service contracts — only optional Equipment Installment Plans.

---

## Key Design Decisions

**No service contracts.** NovaWireless customers are month-to-month or on optional 24/30/36-month Equipment Installment Plans. The `billing_agreement_type` field reflects this.

**`churn_risk_score` is per-customer.** Derived from individual attributes, not a global mean. Ranges from ~0.01 to ~0.63.

**IMEI anomalies are intentional.** They simulate real-world fraud patterns that the Call Generator uses to assign scenarios like `fraud_line_add` and `fraud_hic_exchange`.

---

## Requirements

```
pandas
numpy
```

---

## Related Repos

- **NovaWireless Employee Generator** — generates the rep database
- **NovaWireless Call Generator** — uses customer + employee data to generate synthetic call records and transcripts
