#!/usr/bin/env python3
"""Three-way critic ablation: observational vs factorized vs observational+TCI.

Runs three critic training variants with identical splits and evaluates:
  1. Intervention pair ranking accuracy (TCI metric)
  2. Negative transfer ranking (focus on (1,1,0) contrasts)
  3. Transfer effect ranking (Spearman / top1 hit / regret)

The three variants share:
  - train / validation / test splits (from artifacts/marble/paired/)
  - memory pool
  - evaluation metrics

Differences:
  - Baseline 1: ``observational`` — flat critic, no TCI supervision.
  - Baseline 2: ``opportunity_factorized`` — factorized three-head critic.
  - Model 3: ``observational+tci`` — flat critic + TCI distillation
    (appended soft-labeled examples, alpha=1.0).

Output: a JSON ablation report in
``artifacts/marble/interventions/p2_pilot_real/critic_ablation_report.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
)
from smtr.marble.training import (
    _build_tci_inputs_for_critic,
    train_critic,
)
from smtr.router.tci_supervision import evaluate_tci_loss_on_critic
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import (
    build_routing_card_from_pool_entry,
    load_paired_records_with_metadata,
)


ROOT = Path("artifacts/marble")
PAIRED = ROOT / "paired"
INTERV = ROOT / "interventions" / "p2_pilot_real"
OUT_DIR = INTERV / "critic_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN = PAIRED / "train" / "paired_records.jsonl"
VAL = PAIRED / "validation" / "paired_records.jsonl"
TEST = PAIRED / "test" / "paired_records.jsonl"
POOL = ROOT / "real_data" / "database_v1" / "memory_pool.jsonl"

CONTRASTS = INTERV / "intervention_contrasts.jsonl"
PERTURB = INTERV / "perturbations.json"

COMMON = dict(
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


def _variant(name: str, **extra) -> dict:
    output = OUT_DIR / f"{name}.joblib"
    t0 = time.time()
    metrics = train_critic(output_path=output, **COMMON, **extra)
    metrics["wall_seconds"] = round(time.time() - t0, 1)
    metrics["variant"] = name
    metrics["checkpoint_path"] = str(output)
    return metrics


def _build_tci_eval_tuples() -> list:
    """Build (input_orig, input_pert, direction, contrast_type) for
    evaluation of all three critics on the same TCI pairs."""
    return _build_tci_inputs_for_critic(
        tci_contrasts_path=CONTRASTS,
        perturbations_manifest_path=PERTURB,
        paired_records_path=TRAIN,
        memory_pool_path=POOL,
    )


def _score_test_set(critic: FourOutcomeTransferCritic) -> dict:
    """Compute four-outcome classification metrics on the test split."""
    from smtr.marble.paired_outcomes import paired_record_label
    test_data = load_paired_records_with_metadata(TEST, POOL)
    if not test_data:
        return {"test_n": 0}
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
    return {
        "test_n": len(labels),
        "test_accuracy": round(correct / max(1, len(labels)), 4),
        "test_label_counts": dict(
            zip(
                ("neutral_failure", "negative_transfer",
                 "positive_transfer", "neutral_success"),
                (sum(1 for l in labels if l == n) for n in
                 ("neutral_failure", "negative_transfer",
                  "positive_transfer", "neutral_success")),
            )
        ),
    }


def _transfer_ranking_metrics(
    critic: FourOutcomeTransferCritic,
    tci_tuples: list,
) -> dict:
    """Transfer effect ranking metrics over TCI pairs.

    For each (m, m~, direction) pair:
      - tau(m) = q10 - q01 (transfer effect estimand)
      - If direction > 0: require tau(m) > tau(m~)
      - If direction < 0: require tau(m) < tau(m~)
    Reports pairwise accuracy and average margin.
    """
    if not tci_tuples:
        return {"ranking_n": 0}
    correct = 0
    margins = []
    induced_damage_correct = 0
    induced_damage_n = 0
    rescue_destruction_correct = 0
    rescue_destruction_n = 0
    damage_repair_correct = 0
    damage_repair_n = 0
    for (inp_orig, inp_pert, direction, ct) in tci_tuples:
        pred_orig = critic.predict(inp_orig)
        pred_pert = critic.predict(inp_pert)
        tau_orig = (pred_orig.q10_positive_transfer
                    - pred_orig.q01_negative_transfer)
        tau_pert = (pred_pert.q10_positive_transfer
                    - pred_pert.q01_negative_transfer)
        margin = direction * (tau_orig - tau_pert)
        margins.append(margin)
        if margin > 0:
            correct += 1
        if ct == "precondition":
            induced_damage_n += 1
            if margin > 0:
                induced_damage_correct += 1
        elif ct == "environment_constraint":
            rescue_destruction_n += 1
            if margin > 0:
                rescue_destruction_correct += 1
        elif ct == "capability":
            damage_repair_n += 1
            if margin > 0:
                damage_repair_correct += 1
    n = len(tci_tuples)
    return {
        "ranking_n": n,
        "pairwise_accuracy": round(correct / n, 4) if n else 0.0,
        "average_margin": round(float(np.mean(margins)), 4) if margins else 0.0,
        "induced_damage_accuracy": (
            round(induced_damage_correct / induced_damage_n, 4)
            if induced_damage_n else None
        ),
        "induced_damage_n": induced_damage_n,
        "rescue_destruction_accuracy": (
            round(rescue_destruction_correct / rescue_destruction_n, 4)
            if rescue_destruction_n else None
        ),
        "rescue_destruction_n": rescue_destruction_n,
        "damage_repair_accuracy": (
            round(damage_repair_correct / damage_repair_n, 4)
            if damage_repair_n else None
        ),
        "damage_repair_n": damage_repair_n,
    }


def main() -> None:
    print("=== Baseline 1: observational (flat, no TCI) ===")
    m1 = _variant("baseline_observational", critic_mode="flat")
    c1 = FourOutcomeTransferCritic.load(Path(m1["checkpoint_path"]))

    print("=== Baseline 2: opportunity_factorized ===")
    m2 = _variant("factorized", critic_mode="opportunity_factorized")
    c2 = FourOutcomeTransferCritic.load(Path(m2["checkpoint_path"]))

    print("=== Model 3: observational + TCI distillation (alpha=1.0) ===")
    m3 = _variant(
        "tci_distilled",
        critic_mode="flat",
        tci_contrasts_path=CONTRASTS,
        tci_perturbations_manifest_path=PERTURB,
        tci_paired_records_path=TRAIN,
        tci_alpha=1.0,
    )
    c3 = FourOutcomeTransferCritic.load(Path(m3["checkpoint_path"]))

    # ---- Common evaluation ----
    tci_tuples = _build_tci_eval_tuples()
    print(f"Loaded {len(tci_tuples)} TCI pairs for evaluation.")

    variants = [
        ("baseline_observational", c1, m1),
        ("factorized", c2, m2),
        ("tci_distilled", c3, m3),
    ]

    report: dict = {
        "variants": {},
        "gate_criteria": {
            "intervention_ranking_improvement": "+10% over baseline",
            "observational_metrics_no_regression": True,
            "tci_over_factorized": True,
        },
    }

    for name, critic, metrics in variants:
        print(f"\n---- Evaluating {name} ----")
        transfer = _transfer_ranking_metrics(critic, tci_tuples)
        test_block = _score_test_set(critic)
        tci_eval = evaluate_tci_loss_on_critic(critic, tci_tuples)
        block = {
            "training": {
                "wall_seconds": metrics.get("wall_seconds"),
                "n_train_records": metrics.get("train_records"),
                "critic_mode": metrics.get("critic_mode"),
                "tci_training_mode": metrics.get("tci_training_mode"),
                "tci_distillation_n_examples": metrics.get(
                    "tci_distillation_n_examples"
                ),
                "tci_distillation_alpha": metrics.get(
                    "tci_distillation_alpha"
                ),
                "coverage_report": metrics.get("coverage_report"),
            },
            "intervention_ranking": transfer,
            "test_classification": test_block,
            "tci_pairwise_loss": tci_eval,
        }
        report["variants"][name] = block
        print(json.dumps(block, indent=2, sort_keys=True))

    # ---- Gate judgement ----
    base_acc = report["variants"]["baseline_observational"][
        "intervention_ranking"
    ]["pairwise_accuracy"]
    tci_acc = report["variants"]["tci_distilled"][
        "intervention_ranking"
    ]["pairwise_accuracy"]
    fact_acc = report["variants"]["factorized"][
        "intervention_ranking"
    ]["pairwise_accuracy"]
    base_test = report["variants"]["baseline_observational"][
        "test_classification"
    ]["test_accuracy"]
    tci_test = report["variants"]["tci_distilled"][
        "test_classification"
    ]["test_accuracy"]

    gate = {
        "baseline_observational_intervention_accuracy": base_acc,
        "tci_distilled_intervention_accuracy": tci_acc,
        "factorized_intervention_accuracy": fact_acc,
        "baseline_observational_test_accuracy": base_test,
        "tci_distilled_test_accuracy": tci_test,
        "intervention_improvement_over_baseline": (
            round((tci_acc - base_acc) / max(1e-9, base_acc) * 100, 2)
        ),
        "test_accuracy_no_regression": (tci_test >= base_test - 0.02),
        "tci_beats_factorized": (tci_acc >= fact_acc),
    }
    gate_passed = (
        gate["intervention_improvement_over_baseline"] >= 10.0
        and gate["test_accuracy_no_regression"]
        and gate["tci_beats_factorized"]
    )
    gate["gate_passed"] = gate_passed
    gate["gate_verdict"] = "PASS" if gate_passed else "FAIL"
    report["gate"] = gate

    out_path = OUT_DIR / "critic_ablation_report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n=== Ablation report written to {out_path} ===")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
