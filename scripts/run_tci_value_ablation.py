#!/usr/bin/env python3
"""TCI Value Ablation: Three-Model Comparison (Task 9).

Compares three critic training variants:
  - Model A: observational (L_obs only)
  - Model B: tci_augmented (L_obs + L_rank)
  - Model C: tci_value_augmented (L_obs + L_rank + L_value)

Also includes random value supervision baseline for comparison.

Fixed:
  - Same train/validation/test splits
  - Same feature encoder
  - Same candidate evaluation sets

Output: tci_value_ablation.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from smtr.intervention.intervention_contrast import InterventionContrast
from smtr.marble.paired_outcomes import paired_record_label
from smtr.marble.training import (
    _build_tci_inputs_for_critic,
    train_critic,
)
from smtr.router.random_effect_baseline import (
    build_random_effect_baseline,
    compare_tci_vs_random_value,
)
from smtr.router.tci_candidate_eval import (
    evaluate_candidate_effect_ranking,
)
from smtr.router.tci_effect_builder import (
    build_tci_effect_examples,
    compute_effect_accuracy,
)
from smtr.router.tci_routing_eval import (
    compute_routing_metrics_from_paired_records,
)
from smtr.router.tci_supervision import evaluate_tci_loss_on_critic
from smtr.router.tci_synthetic_eval import evaluate_synthetic_candidates
from smtr.router.transfer_critic import (
    FourOutcomeTransferCritic,
    TCIValueHead,
)
from smtr.router.transfer_features import (
    HashingTransferFeatureEncoder,
    load_paired_records_with_metadata,
)


ROOT = Path("artifacts/marble")
PAIRED = ROOT / "paired"
INTERV = ROOT / "interventions" / "p2_pilot_real"
OUT_DIR = INTERV / "value_ablation"
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


def _load_contrasts(path: Path) -> list[InterventionContrast]:
    contrasts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            contrasts.append(InterventionContrast.from_dict(data))
    return contrasts


def _evaluate_model(
    name: str,
    critic: FourOutcomeTransferCritic,
    tci_tuples: list,
    contrasts: list[InterventionContrast],
    effect_batch,
    test_records: list[dict],
    pool: dict,
    paired_records: list[dict],
) -> dict:
    """Evaluate one model on all metrics."""
    print(f"\n---- Evaluating {name} ----")

    # 1. Intervention ranking (pairwise).
    tci_eval = evaluate_tci_loss_on_critic(critic, tci_tuples)

    # 2. Effect prediction accuracy (if value head exists).
    if critic.tci_value_head is not None and effect_batch.n_examples > 0:
        effect_eval = compute_effect_accuracy(
            critic.tci_value_head, effect_batch
        )
    else:
        effect_eval = {
            "accuracy": 0.0,
            "per_class_accuracy": {},
            "n_examples": 0,
        }

    # 3. Routing metrics.
    routing = compute_routing_metrics_from_paired_records(
        critic, test_records, pool
    )

    # 4. Synthetic candidate evaluation.
    synthetic = evaluate_synthetic_candidates(
        critic, contrasts, tci_tuples, pool, paired_records
    )

    # 5. Test classification accuracy.
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
            "tci_rank_examples": critic.tci_rank_examples,
            "tci_value_examples": critic.tci_value_examples,
            "tci_schema_version": critic.tci_schema_version,
        },
        "intervention_ranking": {
            "pairwise_accuracy": round(
                tci_eval.get("pairwise_accuracy", 0.0), 4
            ),
            "pairwise_margin": round(
                tci_eval.get("pairwise_margin", 0.0), 4
            ),
            "n_pairs": tci_eval.get("n_pairs", 0),
        },
        "effect_prediction": {
            "accuracy": round(effect_eval["accuracy"], 4),
            "per_class": effect_eval["per_class_accuracy"],
            "n_examples": effect_eval["n_examples"],
        },
        "routing": {
            "positive_capture": round(routing.positive_capture, 4),
            "negative_exposure": round(routing.negative_exposure, 4),
            "transfer_regret": round(routing.transfer_regret, 4),
            "top1_hit_rate": round(routing.top1_hit_rate, 4),
            "n_selections": routing.n_selections,
        },
        "synthetic_candidates": {
            "top1_hit_rate": round(synthetic.top1_hit_rate, 4),
            "mean_regret": round(synthetic.mean_regret, 4),
            "n_contrasts": synthetic.n_contrasts,
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
    train_records = _load_records(TRAIN)
    contrasts = _load_contrasts(CONTRASTS)

    # Build TCI tuples.
    tci_tuples = _build_tci_inputs_for_critic(
        tci_contrasts_path=CONTRASTS,
        perturbations_manifest_path=PERTURB,
        paired_records_path=TRAIN,
        memory_pool_path=POOL,
    )
    print(f"  TCI pairs: {len(tci_tuples)}")
    print(f"  Contrasts: {len(contrasts)}")

    # Build effect batch.
    encoder = HashingTransferFeatureEncoder(
        n_features=512, feature_block="full"
    )
    effect_batch = build_tci_effect_examples(
        contrasts, encoder, tci_inputs=tci_tuples
    )
    print(f"  Effect examples: {effect_batch.n_examples}")
    dist = effect_batch.effect_distribution()
    print(f"  Effect distribution: -1={dist[-1]}, 0={dist[0]}, +1={dist[+1]}")

    # Build random effect baseline.
    random_batch = build_random_effect_baseline(effect_batch, seed=7)

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
    out_a = OUT_DIR / "observational.joblib"
    t0 = time.time()
    train_critic(output_path=out_a, critic_mode="flat", **common)
    wall_a = round(time.time() - t0, 1)
    cA = FourOutcomeTransferCritic.load(out_a)

    # Model B: TCI augmented (rank only).
    print("\n=== Model B: tci_augmented (rank only) ===")
    out_b = OUT_DIR / "tci_rank.joblib"
    t0 = time.time()
    train_critic(
        output_path=out_b,
        critic_mode="flat",
        tci_contrasts_path=CONTRASTS,
        tci_perturbations_manifest_path=PERTURB,
        tci_paired_records_path=TRAIN,
        **common,
    )
    wall_b = round(time.time() - t0, 1)
    cB = FourOutcomeTransferCritic.load(out_b)

    # Model C: TCI value augmented (rank + value).
    print("\n=== Model C: tci_value_augmented (rank + value) ===")
    out_c = OUT_DIR / "tci_value.joblib"
    t0 = time.time()
    train_critic(
        output_path=out_c,
        critic_mode="flat",
        tci_contrasts_path=CONTRASTS,
        tci_perturbations_manifest_path=PERTURB,
        tci_paired_records_path=TRAIN,
        tci_effect_batch=effect_batch,
        **common,
    )
    wall_c = round(time.time() - t0, 1)
    cC = FourOutcomeTransferCritic.load(out_c)

    # Evaluate all three.
    eval_a = _evaluate_model(
        "observational", cA, tci_tuples, contrasts, effect_batch,
        test_records, pool, train_records
    )
    eval_a["training"]["wall_seconds"] = wall_a

    eval_b = _evaluate_model(
        "tci_rank", cB, tci_tuples, contrasts, effect_batch,
        test_records, pool, train_records
    )
    eval_b["training"]["wall_seconds"] = wall_b

    eval_c = _evaluate_model(
        "tci_value", cC, tci_tuples, contrasts, effect_batch,
        test_records, pool, train_records
    )
    eval_c["training"]["wall_seconds"] = wall_c

    # Random value baseline comparison.
    print("\n=== Random Value Baseline ===")
    if cC.tci_value_head is not None:
        # Train a random value head.
        from sklearn.linear_model import LogisticRegression
        X_rand = random_batch.features
        y_rand = random_batch.effects
        clf_rand = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        )
        clf_rand.fit(X_rand, y_rand)
        random_value_head = TCIValueHead(model=clf_rand, n_examples=len(y_rand))

        random_comparison = compare_tci_vs_random_value(
            effect_batch, random_batch,
            cC.tci_value_head, random_value_head,
        )
    else:
        random_comparison = {
            "tci_accuracy": 0.0,
            "random_accuracy": 0.0,
            "improvement": 0.0,
            "gate_pass": False,
            "n_examples": 0,
        }

    # Gate judgement.
    gate = {
        "gateA_pairwise_ranking": (
            eval_c["intervention_ranking"]["pairwise_accuracy"] >= 0.70
        ),
        "gateB_effect_prediction": (
            eval_c["effect_prediction"]["accuracy"]
            > random_comparison["random_accuracy"]
        ),
        "gateC_candidate_ranking": (
            eval_c["synthetic_candidates"]["top1_hit_rate"]
            > eval_a["synthetic_candidates"]["top1_hit_rate"]
        ),
        "gateD_random_superiority": random_comparison["gate_pass"],
    }

    report = {
        "observational": eval_a,
        "tci_rank": eval_b,
        "tci_value": eval_c,
        "random_value_comparison": random_comparison,
        "effect_distribution": dist,
        "gate": gate,
    }

    out_path = OUT_DIR / "tci_value_ablation.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"\n=== Report: {out_path} ===")
    print(f"Effect distribution: {dist}")
    print(f"\nGate judgement:")
    for key, val in gate.items():
        print(f"  {key}: {'PASS' if val else 'FAIL'}")


if __name__ == "__main__":
    main()
