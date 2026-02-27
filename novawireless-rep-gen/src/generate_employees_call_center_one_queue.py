#!/usr/bin/env python3
"""
NovaWireless Representative Generator — One Queue CSR (Unique Simple Names + Skill Tags)

Guarantees:
- Unique rep_id
- Unique (first_name, last_name)
- ASCII-only names (no encoding weirdness)
- Single queue / single department / single role
- primary_skill_tag + secondary_skill_tag retained as strengths
- network_service is treated as tech support via *_skill_label fields

Reads priors from:
  repo_root/data/employee_generation_inputs/

Writes outputs to:
  repo_root/output/

Run:
  python src/generate_employees_call_center_one_queue.py --n 3000 --seed 20260216 --site NovaWireless
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Repo-root + IO helpers
# -----------------------------

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

def pick_input_dir(repo_root: Path) -> Path:
    cand = repo_root / "data" / "employee_generation_inputs"
    if cand.exists():
        return cand
    sandbox = Path("/mnt/data")
    if sandbox.exists():
        return sandbox
    raise FileNotFoundError("Could not find data/employee_generation_inputs/ (and no /mnt/data fallback).")

def ensure_output_dir(repo_root: Path) -> Path:
    out = repo_root / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out

def stable_run_id(seed: int, files: List[Path]) -> str:
    h = hashlib.sha256()
    h.update(str(seed).encode("utf-8"))
    for p in sorted(files, key=lambda x: x.name.lower()):
        try:
            stat = p.stat()
            h.update(p.name.encode("utf-8"))
            h.update(str(stat.st_size).encode("utf-8"))
            h.update(str(int(stat.st_mtime)).encode("utf-8"))
        except Exception:
            continue
    return h.hexdigest()[:10]

def non_overwriting_path(out_dir: Path, base_name: str, ext: str) -> Path:
    p = out_dir / f"{base_name}.{ext}"
    if not p.exists():
        return p
    i = 2
    while True:
        p2 = out_dir / f"{base_name}__v{i}.{ext}"
        if not p2.exists():
            return p2
        i += 1


# -----------------------------
# Random utilities
# -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def z_noise(rng: random.Random, sigma: float = 0.15) -> float:
    u1 = max(1e-9, rng.random())
    u2 = rng.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return z * sigma

def weighted_choice(rng: random.Random, items: List[Tuple[str, float]]) -> str:
    total = sum(w for _, w in items)
    if total <= 0:
        return items[0][0]
    r = rng.random() * total
    acc = 0.0
    for v, w in items:
        acc += w
        if r <= acc:
            return v
    return items[-1][0]


# -----------------------------
# Simple ASCII name pools (unique)
# -----------------------------

FIRST_NAMES = [
    "Aaliyah","Aaron","Abigail","Adam","Adrian","Aiden","Alana","Alejandro","Alex","Alexa","Alexander","Alexis",
    "Amelia","Amir","Amy","Ana","Andre","Andrea","Andrew","Angela","Anita","Anna","Anthony","Ari","Ariana",
    "Ashley","Ashton","Ava","Avery","Bailey","Barbara","Benjamin","Brianna","Brittany","Caleb","Cameron","Carlos",
    "Carmen","Caroline","Carter","Casey","Catherine","Charles","Charlotte","Chloe","Chris","Christian","Christina",
    "Claire","Clara","Cole","Connor","Courtney","Daisy","Dakota","Daniel","Danielle","David","Derek","Diana","Diego",
    "Dominic","Drew","Dylan","Eleanor","Elena","Eli","Elijah","Elizabeth","Ella","Elliot","Emily","Emma","Eric",
    "Ethan","Eva","Evelyn","Faith","Finn","Gabriel","Gabriella","Gavin","Genesis","George","Grace","Hailey","Hannah",
    "Harper","Hayden","Hazel","Henry","Isabella","Isaiah","Jace","Jack","Jackson","Jacob","Jada","Jaden","Jake",
    "James","Jamie","Jasmine","Jason","Jayden","Jennifer","Jeremiah","Jessica","John","Jonathan","Jordan","Jose",
    "Joseph","Joshua","Julia","Julian","Kaitlyn","Katherine","Kayla","Kevin","Kimberly","Kyle","Landon","Laura",
    "Lauren","Leah","Leo","Liam","Lily","Logan","Lucas","Lucy","Luis","Madison","Makayla","Maria","Mason","Mateo",
    "Matthew","Maya","Megan","Mia","Michael","Michelle","Mila","Naomi","Natalie","Nathan","Nicholas","Noah","Nora",
    "Oliver","Olivia","Owen","Parker","Paul","Penelope","Quinn","Riley","Robert","Samantha","Samuel","Sara","Sarah",
    "Sebastian","Sofia","Sophia","Taylor","Thomas","Tristan","Tyler","Victoria","William","Zoe"
]

LAST_NAMES = [
    "Adams","Allen","Alvarez","Anderson","Baker","Barnes","Bell","Bennett","Brooks","Brown","Butler","Campbell",
    "Carter","Castillo","Chang","Chen","Clark","Collins","Cook","Cooper","Cox","Cruz","Davis","Diaz","Edwards",
    "Evans","Flores","Foster","Garcia","Gomez","Gonzalez","Gray","Green","Gupta","Hall","Harris","Hernandez","Hill",
    "Howard","Hughes","Jackson","James","Jenkins","Johnson","Jones","Kaur","Kelley","Kelly","Khan","Kim","King",
    "Lee","Lewis","Lopez","Martin","Martinez","Miller","Mitchell","Moore","Morales","Morgan","Murphy","Nelson",
    "Nguyen","Ortiz","Parker","Patel","Perez","Peterson","Phillips","Powell","Price","Ramirez","Reed","Richardson",
    "Rivera","Roberts","Robinson","Rodriguez","Rogers","Ross","Ruiz","Sanchez","Sanders","Scott","Shah","Singh",
    "Smith","Stewart","Taylor","Thomas","Thompson","Torres","Turner","Walker","Ward","Watson","White","Williams",
    "Wilson","Wong","Wright","Young"
]

def make_unique_simple_name(rng: random.Random, used_pairs: set[Tuple[str, str]], max_tries: int = 20000) -> Tuple[str, str]:
    for _ in range(max_tries):
        fn = rng.choice(FIRST_NAMES)
        ln = rng.choice(LAST_NAMES)
        key = (fn, ln)
        if key not in used_pairs:
            used_pairs.add(key)
            return fn, ln
    raise RuntimeError("Unable to generate a unique first/last pair. Increase pools or reduce n.")


# -----------------------------
# Priors loading (skills + persona + pressure)
# -----------------------------

@dataclass
class Priors:
    persona_priors: Optional[pd.DataFrame]
    fcc_specialization: Optional[pd.DataFrame]
    pressure_weekday: Optional[pd.DataFrame]
    telco_segment_pressure: Optional[pd.DataFrame]

def load_priors(input_dir: Path) -> Tuple[Priors, List[Path]]:
    used: List[Path] = []

    def first(patterns: List[str]) -> Optional[Path]:
        for pat in patterns:
            hits = sorted(input_dir.glob(pat))
            if hits:
                used.append(hits[0])
                return hits[0]
        return None

    persona_p = first(["kaggle_employee_persona_priors*.csv"])
    fcc_spec_p = first(["fcc_cgb_consumer_complaints__rep_specialization_priors*.csv"])
    weekday_p = first(["kaggle_call_center_weekday_pressure*.csv"])
    telco_p = first(["ibm_telco_segment_pressure*.csv"])

    return Priors(
        persona_priors=pd.read_csv(persona_p) if persona_p else None,
        fcc_specialization=pd.read_csv(fcc_spec_p) if fcc_spec_p else None,
        pressure_weekday=pd.read_csv(weekday_p) if weekday_p else None,
        telco_segment_pressure=pd.read_csv(telco_p) if telco_p else None,
    ), used


# -----------------------------
# Skill tag mapping (your semantics)
# -----------------------------

SKILL_LABELS = {
    "general_support": "general_support",
    "billing_resolution": "billing_support",
    "device_support": "device_support",
    "network_service": "tech_support",          # <-- your rule
    "porting_transfer": "porting_support",
    "fraud_unwanted_calls": "fraud_unwanted_calls",
}


def lookup_persona(priors: Priors) -> Dict[str, float]:
    """
    If persona priors exist, we sample from the overall distribution by taking a random row.
    (We avoid dept/role matching since everyone is one queue/role.)
    """
    out = {"patience": 0.55, "empathy": 0.55, "escalation_proneness": 0.45, "burnout_risk": 0.45}
    df = priors.persona_priors
    if df is None or df.empty:
        return out

    # pick a weighted row if 'n' exists; else uniform
    if "n" in df.columns:
        w = pd.to_numeric(df["n"], errors="coerce").fillna(1.0).tolist()
        total = float(sum(w) or 1.0)
        r = random.random() * total
        acc = 0.0
        idx = 0
        for i, wi in enumerate(w):
            acc += wi
            if r <= acc:
                idx = i
                break
        row = df.iloc[idx].to_dict()
    else:
        row = df.sample(1).iloc[0].to_dict()

    for k, col in [
        ("patience", "patience_mean"),
        ("empathy", "empathy_mean"),
        ("escalation_proneness", "escalation_proneness_mean"),
        ("burnout_risk", "burnout_risk_mean"),
    ]:
        if col in row and pd.notna(row[col]):
            out[k] = clamp(float(row[col]), 0.0, 1.0)
    return out


def sample_skill_pair(priors: Priors, rng: random.Random) -> Tuple[str, str]:
    default = [
        ("general_support", 0.25), ("network_service", 0.20), ("device_support", 0.18),
        ("billing_resolution", 0.18), ("porting_transfer", 0.10), ("fraud_unwanted_calls", 0.09),
    ]

    df = priors.fcc_specialization
    skills = default
    if df is not None and not df.empty and "skill_tag" in df.columns:
        tmp = df.copy()
        if "p" in tmp.columns:
            tmp["p"] = pd.to_numeric(tmp["p"], errors="coerce").fillna(0.0)
            skills = [(str(r["skill_tag"]), float(r["p"])) for _, r in tmp.iterrows() if float(r["p"]) > 0]
        else:
            tmp["count"] = pd.to_numeric(tmp.get("count", 1), errors="coerce").fillna(1.0)
            total = float(tmp["count"].sum() or 1.0)
            skills = [(str(r["skill_tag"]), float(r["count"]) / total) for _, r in tmp.iterrows()]

    primary = weighted_choice(rng, skills)
    bias = {k: w for k, w in skills}
    bias[primary] = bias.get(primary, 0.0) * 0.15
    secondary = weighted_choice(rng, list(bias.items()))
    if secondary == primary:
        secondary = "general_support" if primary != "general_support" else "device_support"
    return primary, secondary


def assign_strain_tier(x: float) -> str:
    if x < 0.35: return "low"
    if x < 0.55: return "medium"
    if x < 0.75: return "high"
    return "very_high"


def synthesize_kpis(rng: random.Random, persona: Dict[str, float], base_strain: float, base_training: float, pressure: float, primary_skill: str) -> Dict[str, float]:
    """
    CSR-only KPI model. Everyone can do everything, skills just tilt outcomes a bit.
    """
    # burnout rises with strain + pressure, eased by training/patience
    burnout = clamp(0.55 * persona["burnout_risk"] + 0.30 * base_strain + 0.15 * (pressure - 0.5) - 0.10 * persona["patience"] + z_noise(rng, 0.10), 0.0, 1.0)
    resilience = clamp(1.0 - burnout * 0.65 + (base_training / 12.0) * 0.20 + z_noise(rng, 0.08), 0.0, 1.0)
    volatility = clamp(0.30 + burnout * 0.60 + z_noise(rng, 0.12), 0.0, 1.0)

    # quality proxy
    qa = clamp(0.58 + 0.12 * persona["patience"] + 0.10 * (base_training / 12.0) - 0.18 * burnout + z_noise(rng, 0.06), 0.0, 1.0)

    # skill tilt (small)
    skill_bonus = {
        "billing_resolution": 0.02,
        "network_service": 0.02,      # tech_support tilt
        "device_support": 0.015,
        "porting_transfer": 0.01,
        "fraud_unwanted_calls": 0.01,
        "general_support": 0.00,
    }.get(primary_skill, 0.0)

    fcr = clamp(0.55 + 0.18 * qa + 0.10 * persona["patience"] + skill_bonus - 0.18 * burnout - 0.08 * (pressure - 0.5) + z_noise(rng, 0.06), 0.10, 0.95)

    base_aht = 560.0
    aht = base_aht * (1.0 + 0.35 * burnout + 0.18 * (pressure - 0.5)) * (1.0 - 0.08 * (base_training / 12.0)) * (1.0 - 0.06 * qa)
    aht = clamp(aht + (z_noise(rng, 0.20) * 120), 240.0, 1600.0)

    escalation = clamp(0.05 + 0.18 * persona["escalation_proneness"] + 0.10 * burnout - 0.12 * qa + z_noise(rng, 0.04), 0.01, 0.55)
    transfer = clamp(0.03 + 0.10 * (1.0 - qa) + 0.08 * burnout - 0.04 * persona["patience"] + z_noise(rng, 0.04), 0.01, 0.45)

    repeat = clamp(0.08 + 0.65 * (1.0 - fcr) + 0.06 * (pressure - 0.5) + z_noise(rng, 0.04), 0.02, 0.80)
    csat = clamp(0.52 + 0.24 * persona["empathy"] + 0.16 * qa - 0.18 * burnout - 0.08 * escalation + z_noise(rng, 0.05), 0.10, 0.95)

    compliance_risk = clamp(0.10 + 0.35 * burnout + 0.25 * (1.0 - qa) + 0.12 * escalation + z_noise(rng, 0.06), 0.01, 0.95)

    aht_norm = clamp((1600.0 - aht) / (1600.0 - 240.0), 0.0, 1.0)
    productivity = clamp(0.42 * fcr + 0.33 * qa + 0.25 * aht_norm - 0.15 * burnout + z_noise(rng, 0.04), 0.0, 1.0)

    return {
        "qa_score": round(qa, 4),
        "fcr_30d": round(fcr, 4),
        "repeat_contact_rate": round(repeat, 4),
        "aht_secs": round(aht, 2),
        "csat_proxy": round(csat, 4),
        "transfer_rate": round(transfer, 4),
        "escalation_rate": round(escalation, 4),
        "compliance_risk": round(compliance_risk, 4),
        "productivity_index": round(productivity, 4),
        "burnout_index": round(clamp(0.15 + 0.85 * burnout, 0.0, 1.0), 4),
        "resilience_index": round(resilience, 4),
        "volatility_index": round(volatility, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--site", type=str, default="NovaWireless")
    ap.add_argument("--queue_name", type=str, default="General Support")
    args = ap.parse_args()

    repo_root = find_repo_root()
    input_dir = pick_input_dir(repo_root)
    out_dir = ensure_output_dir(repo_root)

    priors, used_files = load_priors(input_dir)
    rng = random.Random(args.seed)

    # Pressure baseline (mild)
    pressure = 0.5
    if priors.pressure_weekday is not None and "pressure_index" in priors.pressure_weekday.columns:
        v = pd.to_numeric(priors.pressure_weekday["pressure_index"], errors="coerce").dropna()
        if not v.empty:
            pressure = float(v.mean())

    if priors.telco_segment_pressure is not None and "pressure_index" in priors.telco_segment_pressure.columns:
        v = pd.to_numeric(priors.telco_segment_pressure["pressure_index"], errors="coerce").dropna()
        if not v.empty:
            pressure = clamp(pressure + 0.05 * (float(v.mean()) - 0.5), 0.0, 1.0)

    # One queue / one role / one dept
    department = "Call Center"
    job_role = "Customer Service Representative"

    # Global baseline knobs (CSRs can do everything)
    base_strain = 0.52
    base_training = 6.5

    used_name_pairs: set[Tuple[str, str]] = set()
    reps: List[dict] = []

    for i in range(args.n):
        rep_id = f"REP{(i+1):05d}"
        first_name, last_name = make_unique_simple_name(rng, used_name_pairs)
        rep_name = f"{first_name} {last_name}"

        persona = lookup_persona(priors)
        primary_skill, secondary_skill = sample_skill_pair(priors, rng)

        kpis = synthesize_kpis(
            rng=rng,
            persona=persona,
            base_strain=base_strain,
            base_training=base_training,
            pressure=pressure,
            primary_skill=primary_skill,
        )

        strain_score = clamp(0.6 * base_strain + 0.4 * kpis["burnout_index"], 0.0, 1.0)
        strain_tier = assign_strain_tier(strain_score)

        # Skills are literally strengths
        strengths = [primary_skill, secondary_skill, "generalist"]
        weaknesses = []
        if kpis["burnout_index"] >= 0.75:
            weaknesses.append("burnout_risk")
        if kpis["escalation_rate"] >= 0.25:
            weaknesses.append("escalation_prone")
        if not weaknesses:
            weaknesses.append("none_flagged")

        reps.append({
            "rep_id": rep_id,
            "first_name": first_name,
            "last_name": last_name,
            "rep_name": rep_name,
            "site": args.site,
            "queue_name": args.queue_name,
            "department": department,
            "job_role": job_role,
            "can_transfer_departments": False,
            "tenure_months": int(clamp(24 + (z_noise(rng, 0.9) * 8) + (base_training * 1.2), 1, 180)),
            "primary_skill_tag": primary_skill,
            "secondary_skill_tag": secondary_skill,
            "primary_skill_label": SKILL_LABELS.get(primary_skill, primary_skill),
            "secondary_skill_label": SKILL_LABELS.get(secondary_skill, secondary_skill),
            "strengths": "|".join(strengths),
            "weaknesses": "|".join(weaknesses),
            "strain_tier": strain_tier,
            "pressure_index_baseline": round(pressure, 4),
            **kpis
        })

    df = pd.DataFrame(reps)

    # Hard checks
    if df["rep_id"].duplicated().any():
        raise RuntimeError("Duplicate rep_id detected (should be impossible).")
    if df[["first_name", "last_name"]].duplicated().any():
        raise RuntimeError("Duplicate first+last detected (should be impossible).")

    run_id = stable_run_id(args.seed, used_files)
    base_name = f"employees__csr_one_queue__{args.site.lower()}__n{args.n}__seed{args.seed}__{run_id}"

    out_csv = non_overwriting_path(out_dir, base_name, "csv")
    out_json = non_overwriting_path(out_dir, base_name + "__metadata", "json")

    df.to_csv(out_csv, index=False)

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n": args.n,
        "seed": args.seed,
        "site": args.site,
        "queue_name": args.queue_name,
        "used_files": [p.name for p in used_files],
        "rules": {
            "single_queue": True,
            "single_department": department,
            "single_role": job_role,
            "no_transfers": True,
            "skills_as_strengths": True,
            "network_service_means": "tech_support"
        }
    }
    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Wrote roster: {out_csv.relative_to(repo_root)}")
    print(f"[OK] Wrote metadata: {out_json.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
