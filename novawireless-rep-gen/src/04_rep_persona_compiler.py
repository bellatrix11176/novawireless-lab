#!/usr/bin/env python3
"""
04_rep_persona_compiler.py

Enrich novawireless_employee_database.csv with behavioral persona traits.

Inputs:
- output/novawireless_employee_database.csv

Outputs:
- output/novawireless_employee_database.csv (overwritten with persona fields)
- output/rep_persona_profiles__v1.csv (transcript-ready slim file)

Design:
- Deterministic trait derivation (no random drift)
- Weakness + strength tags derived from trait thresholds
- Strain tier computed from burnout/resilience/volatility
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np


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

def clamp01(series: pd.Series) -> pd.Series:
    return series.clip(0, 1)


def normalize(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def main() -> int:
    repo = find_repo_root()
    path = repo / "output" / "novawireless_employee_database.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing employee database: {path}")

    df = pd.read_csv(path)

    required = {
        "qa_score",
        "fcr_30d",
        "repeat_contact_rate",
        "escalation_rate",
        "compliance_risk",
        "burnout_index",
        "resilience_index",
        "volatility_index",
        "productivity_index",
        "csat_proxy",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Employee file missing required columns: {missing}")

    # --- Normalize key signals ---
    qa = normalize(df["qa_score"])
    fcr = normalize(df["fcr_30d"])
    repeat = normalize(df["repeat_contact_rate"])
    escalation = normalize(df["escalation_rate"])
    compliance = normalize(df["compliance_risk"])
    burnout = normalize(df["burnout_index"])
    resilience = normalize(df["resilience_index"])
    volatility = normalize(df["volatility_index"])
    productivity = normalize(df["productivity_index"])
    csat = normalize(df["csat_proxy"])

    # --- Trait Derivations ---
    df["policy_accuracy"] = clamp01(qa)
    df["discovery_skill"] = clamp01(1 - repeat)
    df["conflict_tolerance"] = clamp01(1 - escalation)
    df["technical_skill"] = clamp01((qa + fcr) / 2)
    df["credit_discipline"] = clamp01(1 - compliance)
    df["ownership_bias"] = clamp01((csat + resilience) / 2)
    df["emotional_regulation"] = clamp01(1 - volatility)
    df["aht_pressure_bias"] = clamp01((productivity + (1 - fcr)) / 2)

    # --- Strain Score ---
    df["strain_score"] = clamp01(
        (burnout * 0.4 + volatility * 0.3 + (1 - resilience) * 0.3)
    )

    def tier(x):
        if x < 0.33:
            return "low"
        elif x < 0.66:
            return "medium"
        return "high"

    df["strain_tier"] = df["strain_score"].apply(tier)

    # --- Weakness Tags ---
    def weakness_tags(row):
        tags = []
        if row["discovery_skill"] < 0.35:
            tags.append("discovery_gap")
        if row["policy_accuracy"] < 0.35:
            tags.append("policy_confusion")
        if row["conflict_tolerance"] < 0.35:
            tags.append("escalation_prone")
        if row["aht_pressure_bias"] > 0.65:
            tags.append("speed_over_accuracy")
        if row["credit_discipline"] < 0.35:
            tags.append("credit_leak_risk")
        if row["emotional_regulation"] < 0.35:
            tags.append("reactive_under_stress")
        return "|".join(tags[:3]) if tags else "none"

    df["weakness_tags"] = df.apply(weakness_tags, axis=1)

    # --- Strength Tags ---
    def strength_tags(row):
        tags = []
        if row["discovery_skill"] > 0.75:
            tags.append("thorough_probe")
        if row["policy_accuracy"] > 0.75:
            tags.append("policy_expert")
        if row["ownership_bias"] > 0.75:
            tags.append("accountability_driven")
        if row["conflict_tolerance"] > 0.75:
            tags.append("deescalation_strong")
        return "|".join(tags[:3]) if tags else "balanced"

    df["strength_tags"] = df.apply(strength_tags, axis=1)

    # --- Write Back ---
    df.to_csv(path, index=False)

    persona_cols = [
        "rep_id",
        "policy_accuracy",
        "discovery_skill",
        "conflict_tolerance",
        "technical_skill",
        "credit_discipline",
        "ownership_bias",
        "emotional_regulation",
        "aht_pressure_bias",
        "strain_tier",
        "weakness_tags",
        "strength_tags",
    ]

    slim = df[persona_cols]
    slim.to_csv(repo / "output" / "rep_persona_profiles__v1.csv", index=False)

    print("[OK] Persona enrichment complete.")
    print("[OK] Updated employee database.")
    print("[OK] Wrote rep_persona_profiles__v1.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
