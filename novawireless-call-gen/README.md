# NovaWireless Call Generator

Synthetic call center dataset generator for the NovaWireless Call Center Lab.
Produces 60,000 calls across 12 months of 2025 with realistic rep-state drift,
fraud scenarios, and full dialogue transcripts.

---

## Prerequisites

This generator is part of the **NovaWireless Call Center Lab** multi-repo
structure. Before running, you must have already run:

1. **Customer Generator** → produces `customers.csv` and `master_account_ledger.csv`
2. **Employee Generator** → produces `novawireless_employee_database.csv`

The lab root folder must contain a `.labroot` sentinel file so scripts can
locate the shared `data/` and `output/` directories automatically.

Expected lab structure:
```
NovaWireless Call Center Lab/       <- lab root (.labroot lives here)
    .labroot
    data/
        customers.csv
        master_account_ledger.csv
        novawireless_employee_database.csv
    output/                         <- all generated files land here
    NovaWireless Call Generator/    <- this repo
        src/
        ...
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Generating the Full Year

### Batch 1 — January 2025
Open `src/call_gen__run_all.py` in IDLE and press **F5**.

### Batches 2–12 — February through December 2025
Open `src/01b_generate_calls_append.py` in IDLE and press **F5** eleven times.

Both scripts auto-detect which month to generate next based on what already
exists in `output/`. No arguments or command-line setup required.

---

## Outputs

Each monthly run produces three files in `output/`:

| File | Description |
|------|-------------|
| `calls_metadata_2025-01.csv` | Structured metadata — one row per call, 43+ columns |
| `transcripts_2025-01.jsonl` | Full turn-by-turn dialogue per call |
| `calls_sanitized_2025-01.csv` | Analysis-ready — use this for modeling |

After all 12 months you will have 36 files and approximately 60,000 calls.

---

## Scenario Mix

| Scenario | Share | Description |
|----------|-------|-------------|
| `clean` | 44% | Legitimate call, resolved correctly |
| `unresolvable_clean` | 11% | Legitimate but genuinely unresolvable |
| `gamed_metric` | 10% | Rep games proxy KPIs (short calls, false resolution) |
| `fraud_store_promo` | 7% | Store rep applies unauthorized promotion |
| `fraud_line_add` | 6% | Unauthorized line added to account |
| `fraud_hic_exchange` | 3% | IMEI swap / handset fraud |
| `fraud_care_promo` | 3% | Care rep applies unauthorized care promotion |
| `activation_clean` | 8% | Successful device activation |
| `activation_failed` | 4% | Failed activation — SIM or IMEI error |
| `line_add_legitimate` | 4% | Customer legitimately adds a new line |

---

## Rep State Drift

250 reps accumulate state across all calls in the order generated:

- `gaming_propensity` — increases when proxy KPIs are gamed
- `burnout_level` — increases with call volume and escalations
- `policy_skill` — improves with clean call experience

Rep state modulates outcome probabilities, AHT, and detection flag rates
for every call that rep handles.

---

## Overriding Auto-Detection

Both scripts accept optional arguments if you need to regenerate a specific month:

```bash
python src/call_gen__run_all.py --month 2025-03 --n_calls 5000
python src/01b_generate_calls_append.py --month 2025-06 --n_calls 5000
```

---

## Scripts

| Script | Purpose |
|--------|---------|
| `src/generate_calls.py` | Core call generator — do not run directly |
| `src/call_gen__run_all.py` | **Run this first** — generates batch 1 (January) |
| `src/01b_generate_calls_append.py` | **Run this 11 times** — batches 2–12 |
| `src/02_sanitize_calls.py` | Rebuilds all scores with rep-state-aware logic |
