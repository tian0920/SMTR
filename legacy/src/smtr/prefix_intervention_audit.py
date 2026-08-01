"""Strict prefix intervention audit."""

from collections import defaultdict
from math import isnan

import numpy as np
from pydantic import BaseModel, ConfigDict


class PrefixInterventionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    prefix_intervention_group_id: str
    base_decision_digest: str
    receiver_agent_id: str
    target_memory_id: str
    prefix_variant_id: str
    target_action: str
    outcome_success: bool
    m0_pred_tau: float
    a1_pred_tau: float


class PrefixInterventionAuditResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_groups: int
    delta_tau_mae: float
    delta_tau_correlation: float
    direction_accuracy: float
    effect_region_flip_accuracy: float
    positive_to_negative_accuracy: float
    negative_to_positive_accuracy: float
    neutral_to_negative_accuracy: float
    neutral_to_positive_accuracy: float


def audit_prefix_interventions(
    records: list[PrefixInterventionRecord],
) -> PrefixInterventionAuditResult:
    groups: dict[str, list[PrefixInterventionRecord]] = defaultdict(list)
    for record in records:
        groups[record.prefix_intervention_group_id].append(record)

    true_delta: list[float] = []
    pred_delta: list[float] = []
    for group_id, group in groups.items():
        branches = {(r.prefix_variant_id, r.target_action): r for r in group}
        required = {
            ("S0", "share"),
            ("S0", "withhold"),
            ("S1", "share"),
            ("S1", "withhold"),
        }
        if set(branches) != required:
            raise ValueError(f"prefix intervention group {group_id} lacks four branches")
        digests = {record.base_decision_digest for record in group}
        receivers = {record.receiver_agent_id for record in group}
        targets = {record.target_memory_id for record in group}
        if len(digests) != 1 or len(receivers) != 1 or len(targets) != 1:
            raise ValueError(f"prefix intervention group {group_id} is not aligned")

        y_share_s0 = float(branches[("S0", "share")].outcome_success)
        y_withhold_s0 = float(branches[("S0", "withhold")].outcome_success)
        y_share_s1 = float(branches[("S1", "share")].outcome_success)
        y_withhold_s1 = float(branches[("S1", "withhold")].outcome_success)
        tau_s0 = y_share_s0 - y_withhold_s0
        tau_s1 = y_share_s1 - y_withhold_s1
        true_delta.append(tau_s1 - tau_s0)
        pred_delta.append(
            branches[("S1", "share")].m0_pred_tau
            - branches[("S0", "share")].m0_pred_tau
        )

    if not true_delta:
        return PrefixInterventionAuditResult(
            n_groups=0,
            delta_tau_mae=0.0,
            delta_tau_correlation=0.0,
            direction_accuracy=0.0,
            effect_region_flip_accuracy=0.0,
            positive_to_negative_accuracy=0.0,
            negative_to_positive_accuracy=0.0,
            neutral_to_negative_accuracy=0.0,
            neutral_to_positive_accuracy=0.0,
        )
    true = np.array(true_delta)
    pred = np.array(pred_delta)
    corr = float(np.corrcoef(true, pred)[0, 1]) if len(true) > 1 else 0.0
    if isnan(corr):
        corr = 0.0
    return PrefixInterventionAuditResult(
        n_groups=len(true_delta),
        delta_tau_mae=float(np.mean(np.abs(true - pred))),
        delta_tau_correlation=corr,
        direction_accuracy=float(np.mean(np.sign(true) == np.sign(pred))),
        effect_region_flip_accuracy=_transition_accuracy(true, pred, lambda x: x != 0.0),
        positive_to_negative_accuracy=_transition_accuracy(true, pred, lambda x: x < 0.0),
        negative_to_positive_accuracy=_transition_accuracy(true, pred, lambda x: x > 0.0),
        neutral_to_negative_accuracy=_transition_accuracy(true, pred, lambda x: x < 0.0),
        neutral_to_positive_accuracy=_transition_accuracy(true, pred, lambda x: x > 0.0),
    )


def _transition_accuracy(true: np.ndarray, pred: np.ndarray, predicate) -> float:
    mask = np.array([predicate(value) for value in true])
    if not mask.any():
        return 0.0
    return float(np.mean(np.sign(true[mask]) == np.sign(pred[mask])))
