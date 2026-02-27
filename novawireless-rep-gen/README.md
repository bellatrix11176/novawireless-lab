# NovaWireless Synthetic Employee Generator

Generates a synthetic call center representative database for NovaWireless — a fictional wireless carrier. Produces the employee file required by the NovaWireless Call Generator.

---

## What It Generates

| File | Description |
|---|---|
| `novawireless_employee_database.csv` | 250 synthetic CSR profiles with KPIs, persona traits, and behavioral state |
| `rep_persona_profiles__v1.csv` | Slim persona-only file for transcript use |
| `employees__csr_one_queue__<run_id>.csv` | Versioned archive copy of the roster |

All data is synthetic and derived from publicly available workforce and call center datasets (Kaggle Employee datasets, FCC CGB Consumer Complaints, IBM Telco). No real employee data is used.

---

## Quickstart

```bash
git clone <repo>
cd novawireless-employee-generator
pip install -r requirements.txt
python src/run_all.py
```

Output files are written to `output/`. Copy `novawireless_employee_database.csv` to the Call Generator's `data/` folder.

### Options

```bash
python src/run_all.py --n 500 --seed 999
```

| Argument | Default | Description |
|---|---|---|
| `--n` | 250 | Number of representatives to generate |
| `--seed` | 1337 | Random seed for reproducibility |
| `--site` | NovaWireless | Site name embedded in rep records |
| `--queue_name` | General Support | Queue name embedded in rep records |

---

## Repo Structure

```
novawireless-employee-generator/
│
├── data/
│   └── employee_generation_inputs/     ← prior files read by generator
│       ├── kaggle_employee_persona_priors.csv
│       ├── fcc_cgb_consumer_complaints__rep_specialization_priors.csv
│       ├── kaggle_call_center_weekday_pressure.csv
│       └── ibm_telco_segment_pressure.csv
│
├── src/
│   ├── run_all.py                                      ← run this
│   ├── generate_employees_call_center_one_queue.py     ← step 1: generate roster
│   ├── 02_build_call_taxonomy_from_fcc.py              ← utility: rebuild call type priors
│   ├── 03_build_call_subreason_priors.py               ← utility: rebuild subreason priors
│   └── 04_rep_persona_compiler.py                      ← step 2: enrich with persona traits
│
├── output/                             ← generated files (gitignored)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Pipeline Steps

```bash
python src/run_all.py
```

runs both steps automatically. Individual steps:

```bash
python src/generate_employees_call_center_one_queue.py
python src/04_rep_persona_compiler.py
```

**Step 1 — `generate_employees_call_center_one_queue.py`**
Samples rep KPIs from multi-source priors (Kaggle workforce data, FCC complaint data, IBM Telco pressure signals). Assigns skill tags, strain tiers, and behavioral knobs. Every rep is a CSR in a single queue — no department routing or transfer logic.

**Step 2 — `04_rep_persona_compiler.py`**
Enriches the roster with derived persona traits: `policy_accuracy`, `discovery_skill`, `conflict_tolerance`, `technical_skill`, `credit_discipline`, `ownership_bias`, `emotional_regulation`, `aht_pressure_bias`. These feed directly into the Call Generator's rep state drift model.

### Utility Scripts

`02_build_call_taxonomy_from_fcc.py` and `03_build_call_subreason_priors.py` only need to be rerun if you update the FCC source data. Their outputs are already committed to `data/`.

---

## Prior Files

The generator reads four files from `data/employee_generation_inputs/`:

| File | Source | Used For |
|---|---|---|
| `kaggle_employee_persona_priors.csv` | Kaggle Employee Churn dataset | Patience, empathy, burnout, escalation proneness per role |
| `fcc_cgb_consumer_complaints__rep_specialization_priors.csv` | FCC CGB Consumer Complaints | Skill tag distribution (billing, network, device, fraud, etc.) |
| `kaggle_call_center_weekday_pressure.csv` | Kaggle Call Center dataset | Weekday pressure index baseline |
| `ibm_telco_segment_pressure.csv` | IBM Telco Churn dataset | Segment-level pressure adjustment |

All prior files are included in the repo. The generator runs without internet access.

---

## Key Design Decisions

**Single queue, single role.** All reps are Customer Service Representatives in General Support. No multi-department routing — that complexity belongs in the Call Generator.

**Skills tilt outcomes, not determine them.** A billing specialist is slightly better at billing calls. All reps can handle all call types.

**Persona traits are deterministic.** `04_rep_persona_compiler.py` derives traits from KPIs using fixed formulas — no additional randomness. Same input always produces same output.

**Rep state drifts during simulation.** The Call Generator evolves `gaming_propensity`, `burnout_level`, and `policy_skill` per rep across calls. The employee database provides the starting state.

---

## Requirements

```
pandas
numpy
```

---

## Related Repos

- **NovaWireless Customer Generator** — generates the customer and account graph
- **NovaWireless Call Generator** — uses employee + customer data to generate synthetic call records and transcripts
