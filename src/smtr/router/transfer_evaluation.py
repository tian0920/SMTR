import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from smtr.counterfactual.schemas import PairedInterventionRecord
from smtr.router.transfer_critic import CLASS_ORDER, LABEL_TO_CLASS, FourOutcomeTransferCritic
from smtr.router.transfer_features import prediction_input_from_record


def group_split(
    records: list[PairedInterventionRecord],
    *,
    seed: int,
    test_fraction: float,
) -> tuple[list[PairedInterventionRecord], list[PairedInterventionRecord]]:
    episode_ids = sorted({record.episode_id for record in records})
    random.Random(seed).shuffle(episode_ids)
    test_count = max(1, int(round(len(episode_ids) * test_fraction))) if episode_ids else 0
    test_ids = set(episode_ids[:test_count])
    train = [record for record in records if record.episode_id not in test_ids]
    test = [record for record in records if record.episode_id in test_ids]
    if not train and test:
        train, test = test, []
    return train, test


def evaluate_records(
    critic: FourOutcomeTransferCritic,
    records: list[PairedInterventionRecord],
) -> dict:
    if not records:
        return {}
    y_true = [CLASS_ORDER.index(LABEL_TO_CLASS[record.transfer_class]) for record in records]
    estimates = [critic.predict(prediction_input_from_record(record)) for record in records]
    probs = np.array(
        [
            [e.q00_mean, e.q01_mean, e.q10_mean, e.q11_mean]
            for e in estimates
        ]
    )
    y_pred = probs.argmax(axis=1).tolist()
    tau_true = np.array([record.y_share - record.y_withhold for record in records])
    tau_pred = np.array([estimate.tau_mean for estimate in estimates])
    neg_true = np.array([1 if record.transfer_class == "negative" else 0 for record in records])
    neg_pred = np.array([estimate.negative_risk_mean for estimate in estimates])
    metrics = {
        "four_class_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "multiclass_log_loss": float(log_loss(y_true, probs, labels=[0, 1, 2, 3])),
        "multiclass_brier_score": float(
            np.mean(np.sum((probs - np.eye(4)[y_true]) ** 2, axis=1))
        ),
        "tau_mae": float(np.mean(np.abs(tau_true - tau_pred))),
        "tau_sign_accuracy": float(np.mean(np.sign(tau_true) == np.sign(tau_pred))),
        "negative_risk_brier": float(np.mean((neg_true - neg_pred) ** 2)),
        "negative_risk_auroc": _safe_auroc(neg_true, neg_pred),
        "negative_risk_ece": _ece(neg_true, neg_pred),
        "metrics_by_prefix_size": _metrics_by_prefix_size(critic, records),
        "low_support_rate": float(np.mean([estimate.low_support for estimate in estimates])),
        "metrics_on_low_support": _subset_accuracy(records, estimates, True),
        "metrics_on_in_support": _subset_accuracy(records, estimates, False),
    }
    return metrics


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def distribution(values) -> dict[str, int]:
    return dict(Counter(str(value) for value in values))


def _safe_auroc(y_true, y_pred) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_pred))


def _ece(y_true, y_pred, bins: int = 10) -> float | None:
    if len(y_true) < bins:
        return None
    total = len(y_true)
    ece = 0.0
    for start in np.linspace(0, 1, bins, endpoint=False):
        end = start + 1 / bins
        mask = (y_pred >= start) & (y_pred < end if end < 1 else y_pred <= end)
        if not np.any(mask):
            continue
        ece += np.sum(mask) / total * abs(np.mean(y_true[mask]) - np.mean(y_pred[mask]))
    return float(ece)


def _metrics_by_prefix_size(critic, records) -> dict:
    grouped: dict[int, list[PairedInterventionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.prefix_size].append(record)
    return {
        str(size): _simple_accuracy(critic, group_records)
        for size, group_records in sorted(grouped.items())
    }


def _simple_accuracy(critic, records) -> dict:
    y_true = [CLASS_ORDER.index(LABEL_TO_CLASS[record.transfer_class]) for record in records]
    estimates = [critic.predict(prediction_input_from_record(record)) for record in records]
    y_pred = [
        int(np.argmax([e.q00_mean, e.q01_mean, e.q10_mean, e.q11_mean]))
        for e in estimates
    ]
    return {
        "record_count": len(records),
        "four_class_accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _subset_accuracy(records, estimates, low_support: bool) -> dict:
    selected = [
        (record, estimate)
        for record, estimate in zip(records, estimates, strict=True)
        if estimate.low_support is low_support
    ]
    if not selected:
        return {"record_count": 0}
    y_true = [
        CLASS_ORDER.index(LABEL_TO_CLASS[record.transfer_class])
        for record, _ in selected
    ]
    y_pred = [
        int(np.argmax([e.q00_mean, e.q01_mean, e.q10_mean, e.q11_mean]))
        for _, e in selected
    ]
    return {
        "record_count": len(selected),
        "four_class_accuracy": float(accuracy_score(y_true, y_pred)),
    }
