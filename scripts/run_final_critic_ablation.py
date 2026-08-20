#!/usr/bin/env python3
"""Three-model final critic ablation (Codex Task 7).

Compares three critic training variants with identical splits:
  - Model A: observational (flat, no TCI)
  - Model B: factorized (three-head, no TCI)
  - Model C: tci_augmented (flat + TCI, alpha=1 fixed)

Metrics:
  - Intervention ranking accuracy (TCI pairwise)
  - Routing metrics (PTC, NTE, Regret, Top-1 Hit)
  - Test classification accuracy

Output: final_critic_ablation.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from smtr.marble.training import (
    _build_tci_inputs_for_critic,
    train_critic,
)
from smtr.router.tci_routing_eval import (
    compute_routing_metrics_from_paired_records,
)
from smtr.router.tci_supervision import evaluate_tci_loss_on_critic
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import load_paired_records_with_metadata


ROOT = Path("artifacts/marble")
PAIRED = ROOT / "paired"
INTERV = ROOT / "interventions" / "p2_pilot_real"
OUT_DIR = INTERV / "final_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN = PAIRED / "train" / "paired_records.jsonl"
VAL = PAIRED / "validation" / "paired_records.jsonl"
TEST = PAIRED / "test" / "paired_records.jsonl"
POOL = ROOT / "real_data" / "database_v1" / "memory_pool.jsonl"

CONTRASTS = INTERV / "intervention_contrasts.jsonl"
PERTURB = INTERV / "perturbations.json"


def _load_pool() -> dict:
    pool: dict = {}
    for line in POOL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            m = json.loads(line)
            pool[m["memory_id"]] = m
    return pool


def _load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _variant(name: str, **extra) -> dict:
    output = OUT_DIR / f"{name}.joblib"
    t0 = time.time()
    metrics = train_critic(output_path=output, **extra)
    metrics["wall_seconds"] = round(time.time() - t0, 1)
    metrics["variant"] = name
    metrics["checkpoint_path"] = str(output)
    return metrics


def _evaluate_critic(
    name: str,
    critic: FourOutcomeTransferCritic,
    train_metrics: dict,
    tci_tuples: list,
    test_records: list[dict],
    pool: dict,
) -> dict:
    """Compute all metrics for one critic."""
    # Intervention ranking.
    tci_eval = evaluate_tci_loss_on_critic(critic, tci_tuples)

    # Routing metrics.
    routing = compute_routing_metrics_from_paired_records(
        critic, test_records, pool
    )

    # Test classification accuracy.
    from smtr.marble.paired_outcomes import paired_record_label
    test_data = load_paired_records_with_metadata(TEST, POOL)
    inputs = [item for item, _, _ in test_data]
    labels = [paired_record_label(rec) for _, _, rec in test_data]
    preds = critic.predict_batch(inputs)
    pred_labels = []
    for p in preds:
        dist = [p.q00_neutral_failure, p.q01_negative_transfer,
                p.q10_positive_transfer, p.q11_neutral_success]
        names = ["neutral_failure", "negative_transfer",
                 "positive_transfer", "neutral_success"]
        pred_labels.append(names[int(np.argmax(dist))])
    correct = sum(1 for p, g in zip(pred_labels, labels) if p == g)
    test_acc = round(correct / max(1, len(labels)), 4)

    return {
        "training": {
            "training_mode": critic.training_mode,
            "n_observational_examples": critic.n_observational_examples,
            "n_tci_examples": critic.n_tci_examples,
            "tci_schema_version": critic.tci_schema_version,
            "wall_seconds": train_metrics.get("wall_seconds"),
            "critic_mode": train_metrics.get("critic_mode"),
        },
        "intervention_ranking": {
            "pairwise_accuracy": tci_eval.get("pairwise_accuracy", 0.0),
            "pairwise_margin": tci_eval.get("pairwise_margin", 0.0),
            "pairwise_loss": tci_eval.get("pairwise_loss", 0.0),
            "n_pairs": tci_eval.get("n_pairs", 0),
            "precondition_accuracy": tci_eval.get("precondition_accuracy"),
            "environment_constraint_accuracy": tci_eval.get(
                "environment_constraint_accuracy"
            ),
        },
        "routing": {
            "positive_capture": round(routing.positive_capture, 4),
            "negative_exposure": round(routing.negative_exposure, 4),
            "transfer_regret": round(routing.transfer_regret, 4),
            "top1_hit_rate": round(routing.top1_hit_rate, 4),
            "mean_selected_effect": round(
                routing.mean_selected_effect, 4
            ),
            "mean_best_effect": round(routing.mean_best_effect, 4),
            "n_selections": routing.n_selections,
        },
        "test_classification": {
            "test_accuracy": test_acc,
            "test_n": len(labels),
        },
    }


def main() -> None:
    print("Loading data...")
    pool = _load_pool()
    test_records = _load_records(TEST)
    tci_tuples = _build_tci_inputs_for_critic(
        tci_contrasts_path=CONTRASTS,
        perturbations_manifest_path=PERTURB,
        paired_records_path=TRAIN,
        memory_pool_path=POOL,
    )
    print(f"  TCI pairs: {len(tci_tuples)}")
    print(f"  Test records: {len(test_records)}")

    common = dict(
        train_records_path=TRAIN,
        validation_records_path=VAL,
        test_records_path=TEST,
        memory_pool_path=POOL,
        seed=7,
        n_bootstrap=11,
        n_features=512,
        feature_block="full",
        coverage_mode="pilot",
    )

    # Model A: Observational.
    print("\n=== Model A: observational ===")
    mA = _variant("observational", critic_mode="flat", **common)
    cA = FourOutcomeTransferCritic.load(Path(mA["checkpoint_path"]))

    # Model B: Factorized.
    print("=== Model B: factorized ===")
    mB = _variant("factorized", critic_mode="opportunity_factorized", **common)
    cB = FourOutcomeTransferCritic.load(Path(mB["checkpoint_path"]))

    # Model C: TCI augmented.
    print("=== Model C: tci_augmented ===")
    mC = _variant(
        "tci_augmented",
        critic_mode="flat",
        tci_contrasts_path=CONTRASTS,
        tci_perturbations_manifest_path=PERTURB,
        tci_paired_records_path=TRAIN,
        **common,
    )
    cC = FourOutcomeTransferCritic.load(Path(mC["checkpoint_path"]))

    # Evaluate all three.
    report: dict = {"variants": {}}
    for name, critic, metrics in [
        ("observational", cA, mA),
        ("factorized", cB, mB),
        ("tci_augmented", cC, mC),
    ]:
        print(f"\n---- Evaluating {name} ----")
        block = _evaluate_critic(
            name, critic, metrics, tci_tuples, test_records, pool
        )
        report["variants"][name] = block
        print(json.dumps(block, indent=2, sort_keys=True))

    # Gate judgement.
    obs_route = report["variants"]["observational"]["routing"]
    tci_route = report["variants"]["tci_augmented"]["routing"]
    obs_int = report["variants"]["observational"]["intervention_ranking"]
    tci_int = report["variants"]["tci_augmented"]["intervention_ranking"]

    gate = {
        "gate1_tci_ranking_pass": (
            tci_int["pairwise_accuracy"] >= 0.70
        ),
        "gate2_routing_improvement": (
            tci_route["positive_capture"] > obs_route["positive_capture"]
            or tci_route["negative_exposure"] < obs_route["negative_exposure"]
        ),
        "obs_positive_capture": obs_route["positive_capture"],
        "tci_positive_capture": tci_route["positive_capture"],
        "obs_negative_exposure": obs_route["negative_exposure"],
        "tci_negative_exposure": tci_route["negative_exposure"],
        "obs_transfer_regret": obs_route["transfer_regret"],
        "tci_transfer_regret": tci_route["transfer_regret"],
        "obs_top1_hit": obs_route["top1_hit_rate"],
        "tci_top1_hit": tci_route["top1_hit_rate"],
    }
    report["gate"] = gate

    out_path = OUT_DIR / "final_critic_ablation.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n=== Report written to {out_path} ===")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
