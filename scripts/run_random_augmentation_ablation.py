#!/usr/bin/env python3
"""Random vs TCI augmentation comparison (Codex Task 9).

Compares three critics trained on identical observational data:
  - Observational only (baseline)
  - TCI augmentation (intervention-specific pairs)
  - Random augmentation (random pairs, same count as TCI)

If TCI > Random on intervention ranking, the improvement is
attributable to intervention-specific supervision.

Output: random_vs_tci_ablation.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from smtr.marble.paired_outcomes import paired_record_label
from smtr.marble.training import (
    _build_tci_inputs_for_critic,
    train_critic,
)
from smtr.router.random_supervision import (
    build_random_supervision_pairs,
)
from smtr.router.tci_routing_eval import (
    compute_routing_metrics_from_paired_records,
)
from smtr.router.tci_supervision import evaluate_tci_loss_on_critic
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import (
    load_paired_records_with_metadata,
)


ROOT = Path("artifacts/marble")
PAIRED = ROOT / "paired"
INTERV = ROOT / "interventions" / "p2_pilot_real"
OUT_DIR = INTERV / "random_ablation"
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


def _evaluate(
    critic: FourOutcomeTransferCritic,
    tci_tuples: list,
    test_records: list[dict],
    pool: dict,
) -> dict:
    tci_eval = evaluate_tci_loss_on_critic(critic, tci_tuples)
    routing = compute_routing_metrics_from_paired_records(
        critic, test_records, pool
    )
    # Test classification accuracy.
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
        "intervention_ranking": {
            "pairwise_accuracy": round(
                tci_eval.get("pairwise_accuracy", 0.0), 4
            ),
            "pairwise_margin": round(
                tci_eval.get("pairwise_margin", 0.0), 4
            ),
            "n_pairs": tci_eval.get("n_pairs", 0),
        },
        "routing": {
            "positive_capture": round(routing.positive_capture, 4),
            "negative_exposure": round(routing.negative_exposure, 4),
            "transfer_regret": round(routing.transfer_regret, 4),
            "top1_hit_rate": round(routing.top1_hit_rate, 4),
            "n_selections": routing.n_selections,
        },
        "test_classification": {
            "test_accuracy": test_acc,
            "test_n": len(labels),
        },
        "training": {
            "training_mode": critic.training_mode,
            "n_observational_examples": critic.n_observational_examples,
            "n_tci_examples": critic.n_tci_examples,
        },
    }


def main() -> None:
    print("Loading data...")
    pool = _load_pool()
    test_records = _load_records(TEST)
    train_records = _load_records(TRAIN)

    # Full TCI tuples (for evaluation and count reference).
    tci_tuples = _build_tci_inputs_for_critic(
        tci_contrasts_path=CONTRASTS,
        perturbations_manifest_path=PERTURB,
        paired_records_path=TRAIN,
        memory_pool_path=POOL,
    )
    n_tci_pairs = len(tci_tuples)
    print(f"  TCI pairs: {n_tci_pairs}")

    # Build random pairs (same count as TCI).
    random_pairs = build_random_supervision_pairs(
        memory_pool=pool,
        paired_records=train_records,
        n_pairs=n_tci_pairs,
        seed=7,
    )
    print(f"  Random pairs: {len(random_pairs)}")

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

    # 1. Observational baseline.
    print("\n=== Observational ===")
    out_obs = OUT_DIR / "observational.joblib"
    train_critic(output_path=out_obs, critic_mode="flat", **common)
    c_obs = FourOutcomeTransferCritic.load(out_obs)
    eval_obs = _evaluate(c_obs, tci_tuples, test_records, pool)
    print(json.dumps(eval_obs, indent=2, sort_keys=True))

    # 2. TCI augmented.
    print("\n=== TCI Augmented ===")
    out_tci = OUT_DIR / "tci_augmented.joblib"
    train_critic(
        output_path=out_tci,
        critic_mode="flat",
        tci_contrasts_path=CONTRASTS,
        tci_perturbations_manifest_path=PERTURB,
        tci_paired_records_path=TRAIN,
        **common,
    )
    c_tci = FourOutcomeTransferCritic.load(out_tci)
    eval_tci = _evaluate(c_tci, tci_tuples, test_records, pool)
    print(json.dumps(eval_tci, indent=2, sort_keys=True))

    # 3. Random augmented: train directly with fit().
    print("\n=== Random Augmented ===")
    from smtr.router.transfer_features import (
        build_training_data_from_records,
    )
    train_data = load_paired_records_with_metadata(TRAIN, POOL)
    obs_inputs = [item for item, _, _ in train_data]
    obs_labels = [paired_record_label(rec) for _, _, rec in train_data]

    c_rand = FourOutcomeTransferCritic(
        n_features=512, n_bootstrap=11, seed=7, critic_mode="flat"
    )
    t0 = time.time()
    c_rand.fit(
        obs_inputs, obs_labels,
        coverage_mode="pilot",
        tci_inputs=random_pairs,  # Same interface as TCI inputs.
    )
    wall = round(time.time() - t0, 1)
    eval_rand = _evaluate(c_rand, tci_tuples, test_records, pool)
    eval_rand["training"]["wall_seconds"] = wall
    print(json.dumps(eval_rand, indent=2, sort_keys=True))

    # Comparison.
    gate3 = (
        eval_tci["intervention_ranking"]["pairwise_accuracy"]
        > eval_rand["intervention_ranking"]["pairwise_accuracy"]
    )
    report = {
        "observational": eval_obs,
        "tci_augmented": eval_tci,
        "random_augmented": eval_rand,
        "gate3_tci_superior_to_random": gate3,
        "tci_pairwise_accuracy": eval_tci["intervention_ranking"]["pairwise_accuracy"],
        "random_pairwise_accuracy": eval_rand["intervention_ranking"]["pairwise_accuracy"],
        "n_tci_pairs": n_tci_pairs,
        "n_random_pairs": len(random_pairs),
    }
    out_path = OUT_DIR / "random_vs_tci_ablation.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n=== Report: {out_path} ===")
    print(f"Gate 3 (TCI > Random): {'PASS' if gate3 else 'FAIL'}")
    print(f"  TCI pairwise_accuracy: {eval_tci['intervention_ranking']['pairwise_accuracy']}")
    print(f"  Random pairwise_accuracy: {eval_rand['intervention_ranking']['pairwise_accuracy']}")


if __name__ == "__main__":
    main()
