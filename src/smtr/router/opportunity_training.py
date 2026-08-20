"""Opportunity-factorized training data builder (Counterfactual Opportunity v1).

Converts paired records into three binary-head datasets:

- **baseline**: learns b = P(Y_0=1 | t, o_r, r) — must NOT see candidate memory.
- **rescue**: learns g = P(Y_1=1 | Y_0=0, t, o_r, m, r) — only rows where Y_0=0.
- **damage**: learns h = P(Y_1=0 | Y_0=1, t, o_r, m, r) — only rows where Y_0=1.

The four-outcome probabilities are recovered as:

    q00 = (1-b)(1-g),  q01 = b*h,  q10 = (1-b)*g,  q11 = b*(1-h)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from smtr.core.types import CandidateExposureInput
from smtr.counterfactual.edge_keys import (
    ControlFamilyKey,
    ControlGroupKey,
    TreatmentEdgeKey,
    control_family_key,
    control_group_key,
    treatment_edge_key,
)
from smtr.marble.paired_outcomes import get_paired_outcomes, paired_record_label


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryHeadDataset:
    """One binary classification head's training data."""

    inputs: list
    targets: np.ndarray
    sample_weights: np.ndarray
    family_ids: list[str]
    edge_ids: list[str | None]


@dataclass(frozen=True)
class OpportunityTrainingData:
    """Three-head training data plus a diagnostic support report."""

    baseline: BinaryHeadDataset
    rescue: BinaryHeadDataset
    damage: BinaryHeadDataset
    support_report: dict[str, Any]


# ---------------------------------------------------------------------------
# Canonical outcome reader
# ---------------------------------------------------------------------------


def paired_outcomes(record: dict) -> tuple[int, int]:
    """Return (Y1, Y0) from the canonical paired schema.

    Y1 = share.team_success, Y0 = withhold.team_success.
    """
    y1 = int(bool(record["share"]["team_success"]))
    y0 = int(bool(record["withhold"]["team_success"]))
    return y1, y0


def _assert_label_consistency(record: dict) -> None:
    """Verify that the stored label matches the canonical (Y1, Y0)."""
    y1, y0 = paired_outcomes(record)
    expected = paired_record_label(record)
    existing = record.get("label")
    if existing is not None and existing != expected:
        raise ValueError(
            f"record label mismatch: stored={existing!r}, "
            f"canonical={expected!r} (Y1={y1}, Y0={y0})"
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_opportunity_training_data(
    inputs: list[CandidateExposureInput],
    records: list[dict[str, Any]],
) -> OpportunityTrainingData:
    """Build three-head training data from paired records.

    Parameters
    ----------
    inputs:
        CandidateExposureInput list (same length as ``records``).
    records:
        Raw paired record dicts with canonical share/withhold outcomes.

    Returns
    -------
    OpportunityTrainingData with baseline, rescue, damage datasets and
    a diagnostic support report.
    """
    if len(inputs) != len(records):
        raise ValueError("inputs and records must have identical lengths")

    # ---- Step 1: validate label consistency ----
    for rec in records:
        _assert_label_consistency(rec)

    # ---- Step 2: baseline deduplication by control_group_id ----
    # One (task_id, receiver_agent_id, generation_seed) contributes
    # exactly one baseline example regardless of candidate count.
    control_rows: dict[
        ControlGroupKey,
        tuple[CandidateExposureInput, dict[str, Any]],
    ] = {}
    for item, rec in zip(inputs, records):
        gid = control_group_key(rec)
        if gid not in control_rows:
            control_rows[gid] = (item, rec)
        else:
            # Same control group must have the same Y0.
            existing_y0 = int(bool(
                control_rows[gid][1]["withhold"]["team_success"]
            ))
            current_y0 = int(bool(rec["withhold"]["team_success"]))
            if existing_y0 != current_y0:
                raise ValueError(
                    f"control group {gid} has inconsistent Y0: "
                    f"{existing_y0} vs {current_y0}"
                )

    # ---- Step 3: build baseline dataset ----
    baseline_inputs: list = []
    baseline_targets: list[int] = []
    baseline_family_ids: list[str] = []

    # Family counts for task-receiver equal weighting.
    family_control_count: Counter[str] = Counter()
    for gid, (item, rec) in control_rows.items():
        fid = f"{rec['task_id']}::{rec['receiver_agent_id']}"
        family_control_count[fid] += 1

    for gid, (item, rec) in control_rows.items():
        y1, y0 = paired_outcomes(rec)
        target_b = y0  # baseline learns P(Y_0=1)
        baseline_inputs.append(item)
        baseline_targets.append(target_b)
        fid = f"{rec['task_id']}::{rec['receiver_agent_id']}"
        baseline_family_ids.append(fid)

    # Baseline family-equal weighting: w_i = 1 / n_f
    baseline_weights = np.array(
        [1.0 / family_control_count[fid] for fid in baseline_family_ids],
        dtype=float,
    )

    baseline_ds = BinaryHeadDataset(
        inputs=baseline_inputs,
        targets=np.array(baseline_targets, dtype=int),
        sample_weights=baseline_weights,
        family_ids=baseline_family_ids,
        edge_ids=[None] * len(baseline_inputs),
    )

    # ---- Step 4: build rescue and damage datasets ----
    rescue_inputs: list = []
    rescue_targets: list[int] = []
    rescue_edge_ids: list[str] = []
    rescue_family_ids: list[str] = []

    damage_inputs: list = []
    damage_targets: list[int] = []
    damage_edge_ids: list[str] = []
    damage_family_ids: list[str] = []

    for item, rec in zip(inputs, records):
        y1, y0 = paired_outcomes(rec)
        edge_key = treatment_edge_key(rec)
        edge_str = "::".join(edge_key)
        fid = f"{rec['task_id']}::{rec['receiver_agent_id']}"

        if y0 == 0:
            # Rescue opportunity: baseline failed.
            rescue_inputs.append(item)
            rescue_targets.append(y1)  # 1=rescue, 0=neutral failure
            rescue_edge_ids.append(edge_str)
            rescue_family_ids.append(fid)

        if y0 == 1:
            # Damage opportunity: baseline succeeded.
            damage_inputs.append(item)
            damage_targets.append(1 - y1)  # 1=damage, 0=neutral success
            damage_edge_ids.append(edge_str)
            damage_family_ids.append(fid)

    # Edge-equal weighting for rescue: w_i = 1 / n_e^+
    rescue_edge_counts: Counter[str] = Counter(rescue_edge_ids)
    rescue_weights = np.array(
        [1.0 / rescue_edge_counts[eid] for eid in rescue_edge_ids],
        dtype=float,
    ) if rescue_edge_ids else np.array([], dtype=float)

    # Edge-equal weighting for damage: w_i = 1 / n_e^-
    damage_edge_counts: Counter[str] = Counter(damage_edge_ids)
    damage_weights = np.array(
        [1.0 / damage_edge_counts[eid] for eid in damage_edge_ids],
        dtype=float,
    ) if damage_edge_ids else np.array([], dtype=float)

    rescue_ds = BinaryHeadDataset(
        inputs=rescue_inputs,
        targets=np.array(rescue_targets, dtype=int) if rescue_targets else np.array([], dtype=int),
        sample_weights=rescue_weights,
        family_ids=rescue_family_ids,
        edge_ids=rescue_edge_ids,
    )

    damage_ds = BinaryHeadDataset(
        inputs=damage_inputs,
        targets=np.array(damage_targets, dtype=int) if damage_targets else np.array([], dtype=int),
        sample_weights=damage_weights,
        family_ids=damage_family_ids,
        edge_ids=damage_edge_ids,
    )

    # ---- Step 5: support report ----
    support_report = _build_support_report(
        baseline_ds=baseline_ds,
        rescue_ds=rescue_ds,
        damage_ds=damage_ds,
        control_rows=control_rows,
        records=records,
    )

    return OpportunityTrainingData(
        baseline=baseline_ds,
        rescue=rescue_ds,
        damage=damage_ds,
        support_report=support_report,
    )


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def bootstrap_family_multiplicities(
    family_ids: list[str],
    rng: np.random.Generator,
) -> Counter[str]:
    """Bootstrap task-receiver families and return multiplicity counts.

    Draws ``len(unique_families)`` samples with replacement; returns how
    many times each family was chosen. The same multiplicity must be
    applied to all three heads so one member's (b, g, h) come from the
    same empirical population.
    """
    unique_families = sorted(set(family_ids))
    if not unique_families:
        return Counter()
    chosen = rng.choice(
        len(unique_families), size=len(unique_families), replace=True
    )
    mult: Counter[str] = Counter()
    for idx in chosen:
        mult[unique_families[idx]] += 1
    return mult


def apply_family_multiplicities(
    ds: BinaryHeadDataset,
    multiplicities: Counter[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply bootstrap family multiplicities to a BinaryHeadDataset.

    Returns (row_indices, adjusted_weights) where weights are multiplied
    by the family's bootstrap count.
    """
    indices: list[int] = []
    weights: list[float] = []
    for i, fid in enumerate(ds.family_ids):
        m = multiplicities.get(fid, 0)
        if m > 0:
            indices.append(i)
            weights.append(ds.sample_weights[i] * m)
    return np.array(indices, dtype=int), np.array(weights, dtype=float)


# ---------------------------------------------------------------------------
# Support report
# ---------------------------------------------------------------------------


def _build_support_report(
    *,
    baseline_ds: BinaryHeadDataset,
    rescue_ds: BinaryHeadDataset,
    damage_ds: BinaryHeadDataset,
    control_rows: dict,
    records: list[dict],
) -> dict[str, Any]:
    """Diagnostic report of training support for each head."""
    n_baseline = len(baseline_ds.inputs)
    n_baseline_pos = int(baseline_ds.targets.sum()) if n_baseline else 0
    n_baseline_neg = n_baseline - n_baseline_pos
    n_control_families = len(set(baseline_ds.family_ids)) if n_baseline else 0

    n_rescue = len(rescue_ds.inputs)
    n_rescue_pos = int(rescue_ds.targets.sum()) if n_rescue else 0
    n_rescue_neg = n_rescue - n_rescue_pos
    rescue_edges = set(rescue_ds.edge_ids) if rescue_ds.edge_ids else set()

    n_damage = len(damage_ds.inputs)
    n_damage_pos = int(damage_ds.targets.sum()) if n_damage else 0
    n_damage_neg = n_damage - n_damage_pos
    damage_edges = set(damage_ds.edge_ids) if damage_ds.edge_ids else set()

    # All treatment edges.
    all_edges = {treatment_edge_key(rec) for rec in records}
    zero_rescue_edges = len(all_edges) - len(rescue_edges)
    zero_damage_edges = len(all_edges) - len(damage_edges)
    both_supported = len(rescue_edges & damage_edges)

    return {
        "baseline": {
            "n_examples": n_baseline,
            "positives": n_baseline_pos,
            "negatives": n_baseline_neg,
            "n_control_families": n_control_families,
        },
        "rescue": {
            "n_opportunities": n_rescue,
            "n_rescues": n_rescue_pos,
            "n_non_rescues": n_rescue_neg,
            "n_supported_edges": len(rescue_edges),
            "positive_rate": (
                n_rescue_pos / n_rescue if n_rescue > 0 else 0.0
            ),
        },
        "damage": {
            "n_opportunities": n_damage,
            "n_damages": n_damage_pos,
            "n_non_damages": n_damage_neg,
            "n_supported_edges": len(damage_edges),
            "positive_rate": (
                n_damage_pos / n_damage if n_damage > 0 else 0.0
            ),
        },
        "edges": {
            "total_edges": len(all_edges),
            "zero_rescue_opportunity_edges": zero_rescue_edges,
            "zero_damage_opportunity_edges": zero_damage_edges,
            "both_supported_edges": both_supported,
        },
    }
