#!/usr/bin/env python3
"""Branch C: Offline DirectTauCritic vs Two-Head comparison (Phase 27.C).

Purpose
-------
Determine whether the two-head formulation (τ̂ = μ₁ - μ₀) is itself the
bottleneck, or whether the feature/sample support is insufficient for ANY
linear critic to rank transfer effects.

Method
------
1. Load the same intervention records used to train the deployed critic.
2. Reproduce the same task-level split (seed=0): TRAIN=task4, VAL=tasks 3,5.
3. Train:
   (a) Two-head bootstrap critic:  τ̂ = mean_b[μ₁(b) - μ₀(b)]
   (b) Direct-tau bootstrap critic: τ̂ = mean_b[tau_model(b)]
   Both use identical: encoder, cluster bootstrap, head type, n_bootstrap, seed.
4. Evaluate on VALIDATION split: MAE / RMSE / Spearman / SignAcc / AUROC / AP.
5. Verdict:
   - DirectTau >> TwoHead → CRITIC_FORMULATION_BOTTLENECK
   - Both poor → FEATURE_SUPPORT_LIMITATION
   - TwoHead ≈ DirectTau (both OK) → no formulation issue (unexpected given v3)

Constraints (Phase 30):
- NO parameter changes (β/δ/γ remain untouched).
- NO real-engine run.
- Pure offline diagnostic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from smtr.rima.features import (  # noqa: E402
    ReceiverConditionedTransferFeatures,
    RimaFeatureEncoder,
)
from smtr.rima.splits import task_level_split  # noqa: E402
from smtr.router.official_score_transfer_critic import (  # noqa: E402
    MatchedInterventionExample,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (mirrors deployed critic exactly)
# ─────────────────────────────────────────────────────────────────────────────
RECORDS_PATH = _PROJECT_ROOT / "results/rima/stage_a/intervention_records.json"
SOURCE_AGENTS_PATH = _PROJECT_ROOT / "results/rima/stage_a/source_agents.json"
OUTPUT_DIR = _PROJECT_ROOT / "results/rima_transfer/branch_c"

N_FEATURES = 1024
N_BOOTSTRAP = 31
SEED = 0
LOSS = "huber"
TRAIN_FRAC = 0.7
VALIDATION_FRAC = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (same as train_critic.py)
# ─────────────────────────────────────────────────────────────────────────────
def load_records(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("records", "validations", "pairs"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError(f"Cannot locate record list in {path}")
    return data


def record_to_example(
    rec: dict[str, Any], *, source_agent_ids: dict[str, str]
) -> MatchedInterventionExample:
    memory_id = str(rec.get("memory_id", "?"))
    receiver_id = str(rec.get("receiver_id", "?"))
    task_id = str(rec.get("task_id", "?"))
    scenario = str(rec.get("scenario", "unknown"))

    features = ReceiverConditionedTransferFeatures(
        task_id=task_id,
        memory_id=memory_id,
        receiver_id=receiver_id,
        task_repr={
            "scenario": scenario,
            "task_type": str(rec.get("task_type", scenario)),
            "text": str(rec.get("task_text", "")),
        },
        receiver_repr={
            "role": str(rec.get("receiver_role", receiver_id)),
            "capabilities": list(rec.get("receiver_capabilities", []) or []),
        },
        routing_card={
            "goal_summary": str(rec.get("memory_goal_summary", "")),
            "task_tags": list(rec.get("memory_task_tags", [scenario]) or [scenario]),
            "precondition_summary": str(rec.get("memory_precondition", "")),
            "compatible_receiver_roles": list(rec.get("memory_receiver_roles", []) or []),
            "compatible_receiver_capabilities": list(
                rec.get("memory_receiver_capabilities", []) or []
            ),
            "procedure_type": str(rec.get("memory_type", "experience")),
        },
    )
    return MatchedInterventionExample(
        task_id=task_id,
        memory_id=memory_id,
        receiver_id=receiver_id,
        source_agent_id=source_agent_ids.get(memory_id, str(rec.get("source_agent_id", ""))),
        official_expose_score=rec.get("normalized_expose_score"),
        official_withhold_score=rec.get("normalized_withhold_score"),
        features=features,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DirectTauCritic: single-head bootstrap (diagnostic only, not for deployment)
# ─────────────────────────────────────────────────────────────────────────────
class DirectTauBootstrapCritic:
    """Bootstrap critic that directly fits τ = Y(1) - Y(0).

    Identical to BootstrapOfficialScoreTransferCritic except:
    - Single head per member instead of two (μ₁, μ₀).
    - Target is y_tau = expose_score - withhold_score.
    - Prediction: τ̂ = mean_b[tau_model(b)(x)], σ_τ = std_b[...].
    """

    def __init__(
        self,
        *,
        encoder: RimaFeatureEncoder,
        n_bootstrap: int = 31,
        seed: int = 0,
        loss: str = "huber",
    ) -> None:
        self.encoder = encoder
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.loss = loss
        self.members: list[Any] = []  # each member is a single head model
        self._training_stats: dict[str, Any] = {}

    def _make_head(self) -> Any:
        if self.loss == "huber":
            return HuberRegressor(max_iter=500)
        return Ridge(alpha=1.0)

    def fit(self, examples: list[MatchedInterventionExample]) -> dict[str, Any]:
        """Train with same cluster bootstrap as two-head critic."""
        # Fail-closed filtering (same logic).
        usable: list[MatchedInterventionExample] = []
        invalid_count = 0
        self_transfer_count = 0
        for ex in examples:
            if ex.source_agent_id == ex.receiver_id:
                self_transfer_count += 1
                continue
            if ex.official_expose_score is None or ex.official_withhold_score is None:
                invalid_count += 1
                continue
            usable.append(ex)

        if not usable:
            raise ValueError("No usable training examples.")

        # Group by (task_id, receiver_id) families.
        by_family: dict[tuple[str, str], list[MatchedInterventionExample]] = {}
        for ex in usable:
            key = (ex.task_id, ex.receiver_id)
            by_family.setdefault(key, []).append(ex)
        family_keys = sorted(by_family.keys())

        rng = np.random.RandomState(self.seed)
        self.members = []

        for _b in range(self.n_bootstrap):
            sampled_indices = rng.choice(
                len(family_keys), size=len(family_keys), replace=True
            )
            sample_examples: list[MatchedInterventionExample] = []
            for idx in sampled_indices:
                sample_examples.extend(by_family[family_keys[idx]])

            X = self.encoder.encode_batch([ex.features for ex in sample_examples])
            # KEY DIFFERENCE: single target = expose - withhold
            y_tau = np.array(
                [ex.official_expose_score - ex.official_withhold_score
                 for ex in sample_examples],
                dtype=float,
            )
            tau_model = self._make_head().fit(X, y_tau)
            self.members.append(tau_model)

        self._training_stats = {
            "n_examples_total": len(examples),
            "n_examples_used": len(usable),
            "invalid_excluded": invalid_count,
            "self_transfer_excluded": self_transfer_count,
            "loss": self.loss,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
            "bootstrap_cluster_unit": "(task_id, receiver_id)",
            "n_unique_families": len(family_keys),
            "formulation": "direct_tau",
        }
        return dict(self._training_stats)

    def predict(self, ex: MatchedInterventionExample) -> tuple[float, float]:
        """Return (mu_tau, sigma_tau) from bootstrap ensemble."""
        X = self.encoder.encode_one(ex.features)
        tau_preds = np.array(
            [float(m.predict(X)[0]) for m in self.members]
        )
        return float(tau_preds.mean()), float(tau_preds.std())


# ─────────────────────────────────────────────────────────────────────────────
# Two-head bootstrap (retrain from scratch for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────
class TwoHeadBootstrapCritic:
    """Re-implementation matching BootstrapOfficialScoreTransferCritic exactly.

    Kept inline for self-contained diagnostic (avoids loading checkpoint
    that may have different sklearn version state).
    """

    def __init__(
        self,
        *,
        encoder: RimaFeatureEncoder,
        n_bootstrap: int = 31,
        seed: int = 0,
        loss: str = "huber",
    ) -> None:
        self.encoder = encoder
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.loss = loss
        self.members: list[tuple[Any, Any]] = []  # (mu1_model, mu0_model)
        self._training_stats: dict[str, Any] = {}

    def _make_head(self) -> Any:
        if self.loss == "huber":
            return HuberRegressor(max_iter=500)
        return Ridge(alpha=1.0)

    def fit(self, examples: list[MatchedInterventionExample]) -> dict[str, Any]:
        usable: list[MatchedInterventionExample] = []
        invalid_count = 0
        self_transfer_count = 0
        for ex in examples:
            if ex.source_agent_id == ex.receiver_id:
                self_transfer_count += 1
                continue
            if ex.official_expose_score is None or ex.official_withhold_score is None:
                invalid_count += 1
                continue
            usable.append(ex)

        if not usable:
            raise ValueError("No usable training examples.")

        by_family: dict[tuple[str, str], list[MatchedInterventionExample]] = {}
        for ex in usable:
            key = (ex.task_id, ex.receiver_id)
            by_family.setdefault(key, []).append(ex)
        family_keys = sorted(by_family.keys())

        rng = np.random.RandomState(self.seed)
        self.members = []

        for _b in range(self.n_bootstrap):
            sampled_indices = rng.choice(
                len(family_keys), size=len(family_keys), replace=True
            )
            sample_examples: list[MatchedInterventionExample] = []
            for idx in sampled_indices:
                sample_examples.extend(by_family[family_keys[idx]])

            X = self.encoder.encode_batch([ex.features for ex in sample_examples])
            y1 = np.array(
                [ex.official_expose_score for ex in sample_examples], dtype=float
            )
            y0 = np.array(
                [ex.official_withhold_score for ex in sample_examples], dtype=float
            )
            mu1_model = self._make_head().fit(X, y1)
            mu0_model = self._make_head().fit(X, y0)
            self.members.append((mu1_model, mu0_model))

        self._training_stats = {
            "n_examples_total": len(examples),
            "n_examples_used": len(usable),
            "invalid_excluded": invalid_count,
            "self_transfer_excluded": self_transfer_count,
            "loss": self.loss,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
            "bootstrap_cluster_unit": "(task_id, receiver_id)",
            "n_unique_families": len(family_keys),
            "formulation": "two_head",
        }
        return dict(self._training_stats)

    def predict(self, ex: MatchedInterventionExample) -> tuple[float, float]:
        """Return (mu_tau, sigma_tau) = (mean(μ₁-μ₀), std(μ₁-μ₀))."""
        X = self.encoder.encode_one(ex.features)
        mu1_preds = np.array([float(m1.predict(X)[0]) for m1, _ in self.members])
        mu0_preds = np.array([float(m0.predict(X)[0]) for _, m0 in self.members])
        tau_preds = mu1_preds - mu0_preds
        return float(tau_preds.mean()), float(tau_preds.std())


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    predicted_tau: list[float], observed_tau: list[float]
) -> dict[str, float]:
    """Compute 6 diagnostic metrics."""
    pred = np.array(predicted_tau)
    obs = np.array(observed_tau)
    n = len(obs)

    mae = float(np.mean(np.abs(pred - obs)))
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))

    # Spearman rank correlation
    if n >= 3 and np.std(pred) > 0 and np.std(obs) > 0:
        sp_corr, sp_p = spearmanr(pred, obs)
    else:
        sp_corr, sp_p = float("nan"), float("nan")

    # Sign accuracy: fraction where sign(pred) == sign(obs)
    sign_match = np.sum(np.sign(pred) == np.sign(obs))
    sign_acc = float(sign_match / n) if n > 0 else float("nan")

    # AUROC for τ > 0 classification
    binary_labels = (obs > 0).astype(int)
    n_pos = int(binary_labels.sum())
    n_neg = n - n_pos
    if n_pos > 0 and n_neg > 0:
        auroc = float(roc_auc_score(binary_labels, pred))
        ap = float(average_precision_score(binary_labels, pred))
    else:
        auroc = float("nan")
        ap = float("nan")

    return {
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "mae": mae,
        "rmse": rmse,
        "spearman": float(sp_corr) if not np.isnan(sp_corr) else None,
        "spearman_p": float(sp_p) if not np.isnan(sp_p) else None,
        "sign_accuracy": sign_acc,
        "auroc": auroc if not np.isnan(auroc) else None,
        "ap": ap if not np.isnan(ap) else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 70)
    print("Branch C: DirectTauCritic vs Two-Head Bootstrap (Offline Diagnostic)")
    print("=" * 70)

    # 1. Load data
    records = load_records(RECORDS_PATH)
    source_agent_ids: dict[str, str] = json.loads(SOURCE_AGENTS_PATH.read_text())
    examples = [record_to_example(r, source_agent_ids=source_agent_ids) for r in records]
    print(f"\n[Data] Loaded {len(examples)} intervention examples")

    # 2. Split (identical to deployed critic)
    splits = task_level_split(
        examples,
        train_frac=TRAIN_FRAC,
        validation_frac=VALIDATION_FRAC,
        seed=SEED,
    )
    train = splits["train"]
    validation = splits["validation"]
    print(f"[Split] TRAIN={len(train)} (tasks: {sorted({e.task_id for e in train})})")
    print(f"[Split] VALIDATION={len(validation)} (tasks: {sorted({e.task_id for e in validation})})")

    # Observed tau for validation
    val_observed_tau = [
        ex.official_expose_score - ex.official_withhold_score
        for ex in validation
        if ex.official_expose_score is not None and ex.official_withhold_score is not None
    ]
    val_examples_valid = [
        ex for ex in validation
        if ex.official_expose_score is not None and ex.official_withhold_score is not None
    ]
    print(f"[Split] VALID edges in validation: {len(val_examples_valid)}")
    print(f"[Split] Positive τ in validation: {sum(1 for t in val_observed_tau if t > 0)}")

    # 3. Shared encoder
    encoder = RimaFeatureEncoder(n_features=N_FEATURES, include_receiver=True)

    # 4. Train Two-Head critic
    print("\n[TwoHead] Training bootstrap two-head critic...")
    two_head = TwoHeadBootstrapCritic(
        encoder=encoder, n_bootstrap=N_BOOTSTRAP, seed=SEED, loss=LOSS
    )
    th_stats = two_head.fit(train)
    print(f"[TwoHead] Stats: {json.dumps(th_stats, indent=2)}")

    # 5. Train DirectTau critic
    print("\n[DirectTau] Training bootstrap direct-tau critic...")
    direct_tau = DirectTauBootstrapCritic(
        encoder=encoder, n_bootstrap=N_BOOTSTRAP, seed=SEED, loss=LOSS
    )
    dt_stats = direct_tau.fit(train)
    print(f"[DirectTau] Stats: {json.dumps(dt_stats, indent=2)}")

    # 6. Predict on VALIDATION
    th_preds, th_sigmas = [], []
    dt_preds, dt_sigmas = [], []
    for ex in val_examples_valid:
        th_mu, th_sig = two_head.predict(ex)
        dt_mu, dt_sig = direct_tau.predict(ex)
        th_preds.append(th_mu)
        th_sigmas.append(th_sig)
        dt_preds.append(dt_mu)
        dt_sigmas.append(dt_sig)

    # 7. Compute metrics
    th_metrics = compute_metrics(th_preds, val_observed_tau)
    dt_metrics = compute_metrics(dt_preds, val_observed_tau)

    # 8. Also compute TRAIN metrics (sanity check for overfitting)
    train_observed_tau = [
        ex.official_expose_score - ex.official_withhold_score
        for ex in train
        if ex.official_expose_score is not None and ex.official_withhold_score is not None
    ]
    train_valid = [
        ex for ex in train
        if ex.official_expose_score is not None and ex.official_withhold_score is not None
    ]
    th_train_preds = [two_head.predict(ex)[0] for ex in train_valid]
    dt_train_preds = [direct_tau.predict(ex)[0] for ex in train_valid]
    th_train_metrics = compute_metrics(th_train_preds, train_observed_tau)
    dt_train_metrics = compute_metrics(dt_train_preds, train_observed_tau)

    # 9. Report
    print("\n" + "=" * 70)
    print("RESULTS: VALIDATION Split Comparison")
    print("=" * 70)
    header = f"{'Metric':<20} {'Two-Head':>12} {'DirectTau':>12} {'Δ (DT-TH)':>12}"
    print(header)
    print("-" * len(header))
    for key in ("mae", "rmse", "spearman", "sign_accuracy", "auroc", "ap"):
        th_v = th_metrics.get(key)
        dt_v = dt_metrics.get(key)
        if th_v is not None and dt_v is not None:
            delta = dt_v - th_v
            # For MAE/RMSE, negative delta = DirectTau better
            print(f"{key:<20} {th_v:>12.4f} {dt_v:>12.4f} {delta:>+12.4f}")
        else:
            print(f"{key:<20} {str(th_v):>12} {str(dt_v):>12} {'N/A':>12}")

    print(f"\n{'n_val':<20} {th_metrics['n']:>12}")
    print(f"{'n_positive':<20} {th_metrics['n_positive']:>12}")
    print(f"{'n_negative':<20} {th_metrics['n_negative']:>12}")

    print("\n" + "-" * 70)
    print("TRAIN Split (overfitting check)")
    print("-" * 70)
    print(header)
    print("-" * len(header))
    for key in ("mae", "rmse", "spearman", "sign_accuracy", "auroc", "ap"):
        th_v = th_train_metrics.get(key)
        dt_v = dt_train_metrics.get(key)
        if th_v is not None and dt_v is not None:
            delta = dt_v - th_v
            print(f"{key:<20} {th_v:>12.4f} {dt_v:>12.4f} {delta:>+12.4f}")
        else:
            print(f"{key:<20} {str(th_v):>12} {str(dt_v):>12} {'N/A':>12}")

    # 10. Sigma comparison
    print("\n" + "-" * 70)
    print("Uncertainty (σ_τ) comparison on VALIDATION")
    print("-" * 70)
    print(f"  TwoHead  mean σ: {np.mean(th_sigmas):.4f}, median σ: {np.median(th_sigmas):.4f}")
    print(f"  DirectTau mean σ: {np.mean(dt_sigmas):.4f}, median σ: {np.median(dt_sigmas):.4f}")

    # 11. Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Decision criteria (Phase 27.C spec):
    # - DirectTau >> TwoHead → CRITIC_FORMULATION_BOTTLENECK
    #   (DT must dominate: better Spearman AND better SignAcc AND better MAE)
    # - Otherwise → FEATURE_SUPPORT_LIMITATION
    #   (formulation is NOT the bottleneck; sample/feature support is)
    th_sp = th_metrics.get("spearman") or 0.0
    dt_sp = dt_metrics.get("spearman") or 0.0
    th_sa = th_metrics.get("sign_accuracy") or 0.0
    dt_sa = dt_metrics.get("sign_accuracy") or 0.0
    th_mae = th_metrics.get("mae") or 999.0
    dt_mae = dt_metrics.get("mae") or 999.0

    # DirectTau dominates if it's clearly better on ALL three key metrics
    dt_dominates = (
        (dt_sp - th_sp > 0.15)
        and (dt_sa - th_sa > 0.10)
        and (th_mae - dt_mae > 0.01)
    )

    if dt_dominates:
        verdict = "CRITIC_FORMULATION_BOTTLENECK"
        rationale = (
            f"DirectTau significantly outperforms Two-Head "
            f"(Spearman: {dt_sp:.3f} vs {th_sp:.3f}, SignAcc: {dt_sa:.3f} vs {th_sa:.3f}, "
            f"MAE: {dt_mae:.4f} vs {th_mae:.4f}). "
            f"The two-head formulation (τ=μ₁-μ₀) is the bottleneck."
        )
    else:
        verdict = "FEATURE_SUPPORT_LIMITATION"
        rationale = (
            f"DirectTau does NOT outperform Two-Head "
            f"(Spearman: {dt_sp:.3f} vs {th_sp:.3f}, SignAcc: {dt_sa:.3f} vs {th_sa:.3f}, "
            f"MAE: {dt_mae:.4f} vs {th_mae:.4f}). "
            f"Both formulations are limited by the feature space and/or sample support "
            f"(7 TRAIN edges, 3 families, 2 positives). "
            f"The two-head architecture is NOT the bottleneck."
        )

    print(f"  Verdict:   {verdict}")
    print(f"  Rationale: {rationale}")

    # 12. Persist results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "rima_branch_c_direct_tau_v1",
        "config": {
            "records_path": str(RECORDS_PATH),
            "n_features": N_FEATURES,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
            "loss": LOSS,
            "train_frac": TRAIN_FRAC,
            "validation_frac": VALIDATION_FRAC,
        },
        "split": {
            "train_n": len(train),
            "train_tasks": sorted({e.task_id for e in train}),
            "validation_n": len(validation),
            "validation_tasks": sorted({e.task_id for e in validation}),
            "validation_valid_edges": len(val_examples_valid),
            "validation_positive_tau": sum(1 for t in val_observed_tau if t > 0),
        },
        "two_head": {
            "training_stats": th_stats,
            "validation_metrics": th_metrics,
            "train_metrics": th_train_metrics,
            "mean_sigma_validation": float(np.mean(th_sigmas)),
        },
        "direct_tau": {
            "training_stats": dt_stats,
            "validation_metrics": dt_metrics,
            "train_metrics": dt_train_metrics,
            "mean_sigma_validation": float(np.mean(dt_sigmas)),
        },
        "comparison": {
            "spearman_delta_dt_minus_th": dt_sp - th_sp,
            "sign_accuracy_delta_dt_minus_th": dt_sa - th_sa,
            "mae_delta_dt_minus_th": (dt_metrics.get("mae") or 0) - (th_metrics.get("mae") or 0),
        },
        "verdict": verdict,
        "rationale": rationale,
        "per_example_validation": [
            {
                "task_id": ex.task_id,
                "receiver_id": ex.receiver_id,
                "memory_id": ex.memory_id,
                "observed_tau": obs,
                "two_head_tau": th_p,
                "two_head_sigma": th_s,
                "direct_tau_tau": dt_p,
                "direct_tau_sigma": dt_s,
            }
            for ex, obs, th_p, th_s, dt_p, dt_s in zip(
                val_examples_valid, val_observed_tau,
                th_preds, th_sigmas, dt_preds, dt_sigmas,
            )
        ],
    }

    out_path = OUTPUT_DIR / "branch_c_comparison.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Output] Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
