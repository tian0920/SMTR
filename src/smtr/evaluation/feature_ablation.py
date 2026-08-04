"""Feature-mode ablation framework for the transfer critic.

Four precisely defined feature modes isolate the writer-receiver mechanism:

- ``full``: task + environment + memory card + writer marginals + receiver
  marginals + writer-receiver pair interactions.
- ``no_pair_interaction``: keep writer and receiver marginals, drop all
  writer-receiver interaction tokens. Answers whether gains come from the
  two sides individually or from their interaction.
- ``no_receiver``: drop receiver identity/profile and interactions, keep
  task/environment/memory/writer. Answers whether routing is possible
  without knowing the receiver identity.
- ``memory_task_only``: keep only task context, environment and memory card;
  this is the global transfer critic baseline.

The legacy ``no_writer_receiver`` block (writer+interaction removed while
receiver kept) is intentionally NOT part of this framework because its mixed
definition cannot be cleanly interpreted.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from smtr.core.types import CandidateExposureInput
from smtr.router.transfer_critic import FourOutcomeTransferCritic

FEATURE_MODES = [
    "full",
    "no_pair_interaction",
    "no_receiver",
    "memory_task_only",
]

_LABELS = ["neutral_failure", "negative_transfer", "positive_transfer", "neutral_success"]


def split_by_task_group(
    paired_data: list[tuple[CandidateExposureInput, str]],
    *,
    seed: int,
    test_fraction: float = 0.2,
) -> tuple[list[tuple[CandidateExposureInput, str]], list[tuple[CandidateExposureInput, str]]]:
    """Split paired data so all records of a task stay on one side."""
    groups: dict[str, list[tuple[CandidateExposureInput, str]]] = {}
    for item in paired_data:
        groups.setdefault(item[0].receiver_state.task_id, []).append(item)
    task_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(task_ids)
    n_test = max(1, int(round(len(task_ids) * test_fraction))) if len(task_ids) > 1 else 0
    test_ids = set(task_ids[:n_test])
    train, test = [], []
    for task_id in task_ids:
        (test if task_id in test_ids else train).extend(groups[task_id])
    return train, test


def evaluate_mode(
    critic: FourOutcomeTransferCritic,
    test_data: list[tuple[CandidateExposureInput, str]],
) -> dict[str, Any]:
    """Accuracy and macro F1 of a fitted critic on held-out paired data."""
    if not test_data:
        return {"accuracy": None, "macro_f1": None, "n_test": 0}
    inputs = [item for item, _ in test_data]
    labels = [label for _, label in test_data]
    preds = critic.predict_batch(inputs)
    pred_labels: list[str] = []
    for pred in preds:
        probs = [
            pred.q00_neutral_failure,
            pred.q01_negative_transfer,
            pred.q10_positive_transfer,
            pred.q11_neutral_success,
        ]
        pred_labels.append(_LABELS[int(np.argmax(probs))])
    accuracy = sum(1 for p, t in zip(pred_labels, labels) if p == t) / len(labels)
    f1_scores: list[float] = []
    for label in set(labels):
        tp = sum(1 for p, t in zip(pred_labels, labels) if p == label and t == label)
        fp = sum(1 for p, t in zip(pred_labels, labels) if p == label and t != label)
        fn = sum(1 for p, t in zip(pred_labels, labels) if p != label and t == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else None,
        "n_test": len(test_data),
        "test_label_distribution": dict(Counter(labels)),
    }


def audit_feature_modes(
    paired_data: list[tuple[CandidateExposureInput, str]],
    *,
    seed: int,
    n_bootstrap: int,
    n_features: int = 512,
) -> dict[str, Any]:
    """Train and evaluate one critic per feature mode on the same split.

    All modes share an identical task-group split so differences in metrics
    are attributable to the feature mode alone.
    """
    train, test = split_by_task_group(paired_data, seed=seed)
    report: dict[str, Any] = {
        "modes": {},
        "split_manifest": {
            "train_records": len(train),
            "test_records": len(test),
            "train_label_distribution": dict(Counter(label for _, label in train)),
        },
    }
    train_inputs = [item for item, _ in train]
    train_labels = [label for _, label in train]
    for mode in FEATURE_MODES:
        critic = FourOutcomeTransferCritic(
            n_features=n_features,
            n_bootstrap=n_bootstrap,
            feature_block=mode,
            seed=seed,
        )
        critic.fit(train_inputs, train_labels)
        report["modes"][mode] = evaluate_mode(critic, test)

    full_f1 = report["modes"]["full"].get("macro_f1") or 0.0
    best_ablation = max(
        (m for m in FEATURE_MODES if m != "full"),
        key=lambda m: report["modes"][m].get("macro_f1") or 0.0,
    )
    report["best_ablation_mode"] = best_ablation
    report["full_gain_over_no_pair_interaction"] = full_f1 - (
        report["modes"]["no_pair_interaction"].get("macro_f1") or 0.0
    )
    report["full_gain_over_memory_task_only"] = full_f1 - (
        report["modes"]["memory_task_only"].get("macro_f1") or 0.0
    )
    return report
