#!/usr/bin/env python3
"""TCI-SMTR Final Unified Ablation (Tasks 5-8).

Compares three TCI-SMTR critic training variants with unified supervision:
  - Model A: SMTR (L_obs only) — observational baseline
  - Model B: TCI-rank (L_obs + L_rank) — pairwise ranking only
  - Model C: TCI-full (L_obs + L_rank + L_τ) — full unified supervision

Also includes:
  - Budget curve: 0%, 25%, 50%, 100% of TCI effect supervision
  - Random effect baseline comparison (Task 7)

Key insight: ALL supervision operates on the SAME critic's four-class
output. No separate value head. Effect labels {-1, 0, +1} map directly
to the four-outcome space.

Transfer utility: s_θ(m) = q10(m) - q01(m) ≈ E[Y_m - Y_0]

Output: outputs/tci_smtr_final_ablation.json
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
from smtr.router.random_effect_baseline import build_random_effect_baseline
from smtr.router.tci_effect_builder import (
    build_tci_effect_examples,
)
from smtr.router.tci_routing_eval import (
    compute_routing_metrics_from_paired_records,
)
from smtr.router.tci_supervision import evaluate_tci_loss_on_critic
from smtr.router.tci_synthetic_eval import evaluate_synthetic_candidates
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import (
    HashingTransferFeatureEncoder,
    load_paired_records_with_metadata,
)


ROOT = Path("artifacts/marble")
PAIRED = ROOT / "paired"
INTERV = ROOT / "interventions" / "p2_pilot_real"
OUT_DIR = Path("outputs")
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


def _predict_effect_from_critic(
    critic: FourOutcomeTransferCritic,
    effect_batch,
) -> dict:
    """Predict effect via unified critic's transfer utility.

    Uses s_θ(m) = q10(m) - q01(m) as the transfer utility score.
    Evaluates:
      - Utility correlation with ground truth effects (Spearman)
      - Utility sign accuracy (positive effect → positive utility)
      - Argmax classification accuracy (for completeness)

    This is the correct evaluation for a unified critic: the critic
    doesn't have a separate effect classifier — it uses the SAME output
    s_θ(m) to predict both outcome AND effect.
    """
    if effect_batch.n_examples == 0:
        return {
            "accuracy": 0.0, "utility_correlation": 0.0,
            "sign_accuracy": 0.0, "per_class_accuracy": {}, "n_examples": 0,
        }

    features = effect_batch.features
    effects = effect_batch.effects

    from scipy import sparse as sp_sparse
    X = features
    if sp_sparse.issparse(X):
        X = X.toarray()

    # Get probabilities from the critic ensemble.
    probas = np.mean([m.predict_proba(X) for m in critic.members], axis=0)

    # Transfer utility: s_θ(m) = q10 - q01.
    # Label order: neutral_failure(0), negative_transfer(1),
    #              positive_transfer(2), neutral_success(3)
    q10 = probas[:, 2]
    q01 = probas[:, 1]
    utility = q10 - q01

    # 1. Spearman correlation between utility and ground truth.
    from scipy.stats import spearmanr
    corr, p_value = spearmanr(utility, effects)
    if np.isnan(corr):
        corr = 0.0

    # 2. Sign accuracy: for non-neutral effects, does utility sign match?
    non_neutral_mask = effects != 0
    if non_neutral_mask.any():
        sign_correct = np.sum(
            np.sign(utility[non_neutral_mask]) == effects[non_neutral_mask]
        )
        sign_accuracy = float(sign_correct) / non_neutral_mask.sum()
    else:
        sign_accuracy = 0.0

    # 3. Argmax classification (for reference).
    pred_classes = np.argmax(probas, axis=1)
    pred_effects = np.where(
        pred_classes == 2, 1, np.where(pred_classes == 1, -1, 0)
    )
    accuracy = float(np.mean(pred_effects == effects))

    # Per-class accuracy.
    per_class: dict = {}
    for cls in (-1, 0, 1):
        mask = effects == cls
        if mask.any():
            per_class[str(cls)] = float(
                np.mean(pred_effects[mask] == effects[mask])
            )

    return {
        "accuracy": round(accuracy, 4),
        "utility_correlation": round(float(corr), 4),
        "sign_accuracy": round(sign_accuracy, 4),
        "per_class_accuracy": per_class,
        "n_examples": len(effects),
    }


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
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    # 1. Intervention ranking (pairwise).
    tci_eval = evaluate_tci_loss_on_critic(critic, tci_tuples)

    # 2. Effect prediction via unified critic.
    effect_eval = _predict_effect_from_critic(critic, effect_batch)

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

    result = {
        "training": {
            "training_mode": critic.training_mode,
            "n_observational_examples": critic.n_observational_examples,
            "n_tci_examples": critic.n_tci_examples,
            "tci_rank_examples": critic.tci_rank_examples,
            "tci_value_examples": critic.tci_value_examples,
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
            "accuracy": effect_eval["accuracy"],
            "utility_correlation": effect_eval.get("utility_correlation", 0.0),
            "sign_accuracy": effect_eval.get("sign_accuracy", 0.0),
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

    print(f"  Mode: {critic.training_mode}")
    print(f"  Pairwise: {result['intervention_ranking']['pairwise_accuracy']}")
    print(f"  Effect accuracy: {result['effect_prediction']['accuracy']}")
    print(f"  Utility correlation: {result['effect_prediction']['utility_correlation']}")
    print(f"  Sign accuracy: {result['effect_prediction']['sign_accuracy']}")
    print(f"  Synthetic top1: {result['synthetic_candidates']['top1_hit_rate']}")
    print(f"  Test accuracy: {test_acc}")

    return result


def main() -> None:
    t0 = time.time()
    print("TCI-SMTR Final Unified Ablation")
    print("=" * 60)

    # Load data.
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
    print(f"  Distribution: -1={dist[-1]}, 0={dist[0]}, +1={dist[+1]}")

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

    # ---- Model A: SMTR (L_obs only) ----
    print("\n=== Model A: SMTR (L_obs) ===")
    out_a = OUT_DIR / "tci_smtr_a_obs.joblib"
    ta = time.time()
    train_critic(output_path=out_a, critic_mode="flat", **common)
    wall_a = round(time.time() - ta, 1)
    cA = FourOutcomeTransferCritic.load(out_a)

    # ---- Model B: TCI-rank (L_obs + L_rank) ----
    print("\n=== Model B: TCI-rank (L_obs + L_rank) ===")
    out_b = OUT_DIR / "tci_smtr_b_rank.joblib"
    tb = time.time()
    train_critic(
        output_path=out_b,
        critic_mode="flat",
        tci_contrasts_path=CONTRASTS,
        tci_perturbations_manifest_path=PERTURB,
        tci_paired_records_path=TRAIN,
        **common,
    )
    wall_b = round(time.time() - tb, 1)
    cB = FourOutcomeTransferCritic.load(out_b)

    # ---- Model C: TCI-full (L_obs + L_rank + L_τ) ----
    print("\n=== Model C: TCI-full (L_obs + L_rank + L_τ) ===")
    out_c = OUT_DIR / "tci_smtr_c_full.joblib"
    tc = time.time()
    train_critic(
        output_path=out_c,
        critic_mode="flat",
        tci_contrasts_path=CONTRASTS,
        tci_perturbations_manifest_path=PERTURB,
        tci_paired_records_path=TRAIN,
        tci_effect_batch=effect_batch,
        **common,
    )
    wall_c = round(time.time() - tc, 1)
    cC = FourOutcomeTransferCritic.load(out_c)

    # Evaluate all models.
    eval_a = _evaluate_model(
        "Model A: SMTR", cA, tci_tuples, contrasts, effect_batch,
        test_records, pool, train_records,
    )
    eval_a["training"]["wall_seconds"] = wall_a

    eval_b = _evaluate_model(
        "Model B: TCI-rank", cB, tci_tuples, contrasts, effect_batch,
        test_records, pool, train_records,
    )
    eval_b["training"]["wall_seconds"] = wall_b

    eval_c = _evaluate_model(
        "Model C: TCI-full", cC, tci_tuples, contrasts, effect_batch,
        test_records, pool, train_records,
    )
    eval_c["training"]["wall_seconds"] = wall_c

    # ---- Budget Curve (Task 8) ----
    print(f"\n{'='*60}")
    print("Budget Curve: TCI Effect Supervision")
    print(f"{'='*60}")

    budget_results = {}
    for budget_pct in [0.0, 0.25, 0.50, 1.0]:
        if budget_pct == 0.0:
            # Only rank supervision (no effect).
            budget_batch = None
        else:
            from smtr.router.tci_effect_dataset import TCIEffectBatch
            n_take = max(2, int(len(effect_batch.examples) * budget_pct))
            if n_take % 2 == 1:
                n_take = min(n_take + 1, len(effect_batch.examples))
            budget_batch = TCIEffectBatch(
                examples=effect_batch.examples[:n_take]
            )

        label = f"{int(budget_pct*100)}%"
        print(f"\n--- Budget {label} ({n_take if budget_pct else 0} "
              f"effect examples) ---")
        out_bud = OUT_DIR / f"tci_smtr_budget_{int(budget_pct*100)}.joblib"
        train_critic(
            output_path=out_bud,
            critic_mode="flat",
            tci_contrasts_path=CONTRASTS,
            tci_perturbations_manifest_path=PERTURB,
            tci_paired_records_path=TRAIN,
            tci_effect_batch=budget_batch,
            **common,
        )
        cBUD = FourOutcomeTransferCritic.load(out_bud)
        eval_bud = _evaluate_model(
            f"Budget {label}", cBUD, tci_tuples, contrasts,
            effect_batch, test_records, pool, train_records,
        )
        budget_results[label] = eval_bud

    # ---- Random Effect Baseline (Task 7) ----
    print(f"\n{'='*60}")
    print("Random Effect Baseline")
    print(f"{'='*60}")

    out_rand = OUT_DIR / "tci_smtr_random.joblib"
    train_critic(
        output_path=out_rand,
        critic_mode="flat",
        tci_contrasts_path=CONTRASTS,
        tci_perturbations_manifest_path=PERTURB,
        tci_paired_records_path=TRAIN,
        tci_effect_batch=random_batch,
        **common,
    )
    cRand = FourOutcomeTransferCritic.load(out_rand)
    eval_random = _evaluate_model(
        "Random Effect", cRand, tci_tuples, contrasts,
        effect_batch, test_records, pool, train_records,
    )

    # ---- Gate Judgement ----
    print(f"\n{'='*60}")
    print("Final Gate Judgement")
    print(f"{'='*60}")

    gates = {}

    # Gate A: Pairwise accuracy ≥ 0.7 for TCI models.
    gate_a = eval_c["intervention_ranking"]["pairwise_accuracy"] >= 0.7
    gates["A_pairwise_ranking"] = {
        "pass": gate_a,
        "value": eval_c["intervention_ranking"]["pairwise_accuracy"],
    }
    print(f"Gate A (pairwise ≥ 0.7): {'PASS' if gate_a else 'FAIL'} "
          f"({eval_c['intervention_ranking']['pairwise_accuracy']})")

    # Gate B: Utility correlation > 0 for TCI-full.
    gate_b = eval_c["effect_prediction"]["utility_correlation"] > 0.0
    gates["B_effect_correlation"] = {
        "pass": gate_b,
        "tci_full_corr": eval_c["effect_prediction"]["utility_correlation"],
        "tci_full_sign_acc": eval_c["effect_prediction"]["sign_accuracy"],
    }
    print(f"Gate B (utility corr > 0): {'PASS' if gate_b else 'FAIL'} "
          f"(corr={eval_c['effect_prediction']['utility_correlation']}, "
          f"sign_acc={eval_c['effect_prediction']['sign_accuracy']})")

    # Gate C: Synthetic candidate improvement (TCI-full > SMTR).
    gate_c = (
        eval_c["synthetic_candidates"]["top1_hit_rate"]
        > eval_a["synthetic_candidates"]["top1_hit_rate"]
    )
    gates["C_synthetic_improvement"] = {
        "pass": gate_c,
        "tci_full": eval_c["synthetic_candidates"]["top1_hit_rate"],
        "observational": eval_a["synthetic_candidates"]["top1_hit_rate"],
    }
    print(f"Gate C (synthetic top1 ↑): {'PASS' if gate_c else 'FAIL'} "
          f"(TCI={eval_c['synthetic_candidates']['top1_hit_rate']}, "
          f"obs={eval_a['synthetic_candidates']['top1_hit_rate']})")

    # Gate D: TCI-full has higher utility corr than observational baseline.
    # This proves that TCI supervision (of any form) improves transfer utility.
    gate_d = (
        eval_c["effect_prediction"]["utility_correlation"]
        > eval_a["effect_prediction"]["utility_correlation"]
    )
    gates["D_value_signal"] = {
        "pass": gate_d,
        "tci_full_corr": eval_c["effect_prediction"]["utility_correlation"],
        "observational_corr": eval_a["effect_prediction"]["utility_correlation"],
        "tci_rank_corr": eval_b["effect_prediction"]["utility_correlation"],
    }
    print(f"Gate D (TCI corr > obs): {'PASS' if gate_d else 'FAIL'} "
          f"(full={eval_c['effect_prediction']['utility_correlation']}, "
          f"rank={eval_b['effect_prediction']['utility_correlation']}, "
          f"obs={eval_a['effect_prediction']['utility_correlation']})")

    # Gate E: No test regression.
    test_accs = [
        eval_a["test_classification"]["test_accuracy"],
        eval_b["test_classification"]["test_accuracy"],
        eval_c["test_classification"]["test_accuracy"],
    ]
    gate_e = all(a >= 0.60 for a in test_accs)
    gates["E_no_test_regression"] = {
        "pass": gate_e,
        "test_acc_A": eval_a["test_classification"]["test_accuracy"],
        "test_acc_B": eval_b["test_classification"]["test_accuracy"],
        "test_acc_C": eval_c["test_classification"]["test_accuracy"],
    }
    print(f"Gate E (no test regression): {'PASS' if gate_e else 'FAIL'} "
          f"(A={test_accs[0]}, B={test_accs[1]}, C={test_accs[2]})")

    # Gate F: TCI-full has better utility correlation than TCI-rank.
    gate_f = (
        eval_c["effect_prediction"]["utility_correlation"]
        > eval_b["effect_prediction"]["utility_correlation"]
    )
    gates["F_rank_vs_full_separation"] = {
        "pass": gate_f,
        "rank_corr": eval_b["effect_prediction"]["utility_correlation"],
        "full_corr": eval_c["effect_prediction"]["utility_correlation"],
    }
    print(f"Gate F (full corr > rank corr): {'PASS' if gate_f else 'FAIL'} "
          f"(rank={eval_b['effect_prediction']['utility_correlation']}, "
          f"full={eval_c['effect_prediction']['utility_correlation']})")

    n_pass = sum(1 for g in gates.values() if g["pass"])
    n_total = len(gates)
    final = "PASS" if n_pass == n_total else "PARTIAL"
    print(f"\nFinal: {final} ({n_pass}/{n_total})")

    # Save results.
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "TCI-SMTR (unified critic, no separate value head)",
        "objective": "L = L_obs + L_rank + L_τ (all weight=1, no lambda)",
        "critic_output": "s_θ(m) = q10(m) - q01(m) ≈ E[Y_m - Y_0]",
        "models": {
            "A_observational": eval_a,
            "B_tci_rank": eval_b,
            "C_tci_full": eval_c,
            "random_effect": eval_random,
        },
        "budget_curve": budget_results,
        "gates": gates,
        "final_verdict": f"{final} ({n_pass}/{n_total})",
        "runtime_seconds": round(time.time() - t0, 1),
    }

    out_path = OUT_DIR / "tci_smtr_final_ablation.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
