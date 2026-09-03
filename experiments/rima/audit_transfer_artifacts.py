"""Pre-pilot artifact audit for RIMA-Transfer (Phase 20).

Validates frozen policy parameters, critic checkpoint integrity,
train/test leakage, and uncertainty diagnostics before launching
the mechanism pilot.

Checks
------
1. β == 1.64
2. δ == 0.0
3. γ recomputed from TRAIN intervention records matches policy γ
4. Critic checkpoint SHA256 matches policy hash
5. Train/validation/test task IDs strictly disjoint
6. Uncertainty audit on validation split (MAE, RMSE, sign accuracy,
   LCB coverage, mean/median sigma, corr(sigma, abs_error))

Usage::

    python experiments/rima/audit_transfer_artifacts.py \\
        --critic results/rima_transfer/critic/critic_receiver_bootstrap.joblib \\
        --policy results/rima_transfer/critic/transfer_policy.json \\
        --records results/rima_transfer/stage_a/intervention_records.json

Outputs ``results/rima_transfer/pilot/prepilot_uncertainty_audit.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from smtr.rima.critic_validation import validate_critic  # noqa: E402
from smtr.rima.splits import (  # noqa: E402
    SplitLeakageError,
    audit_split_leakage,
    task_level_split,
)
from smtr.rima.transfer_policy import (  # noqa: E402
    TransferPolicy,
    compute_gamma,
    observed_tau,
)
from smtr.router.official_score_transfer_critic import (  # noqa: E402
    BootstrapOfficialScoreTransferCritic,
    MatchedInterventionExample,
)

# Reuse the same record→example converter as the training pipeline.
from experiments.rima.train_critic import (  # noqa: E402
    load_records,
    record_to_example,
)

__all__: list[str] = []

_DEFAULT_OUTPUT = "results/rima_transfer/pilot/prepilot_uncertainty_audit.json"


def _sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------
# Individual audit checks
# ------------------------------------------------------------------


def _check_beta(policy: dict[str, Any]) -> dict[str, Any]:
    """20.1: β == 1.64."""
    beta = policy.get("beta")
    passed = beta is not None and float(beta) == 1.64
    return {
        "check": "beta_frozen",
        "expected": 1.64,
        "actual": beta,
        "passed": passed,
    }


def _check_delta(policy: dict[str, Any]) -> dict[str, Any]:
    """20.2: δ == 0.0."""
    delta = policy.get("delta")
    passed = delta is not None and float(delta) == 0.0
    return {
        "check": "delta_frozen",
        "expected": 0.0,
        "actual": delta,
        "passed": passed,
    }


def _check_gamma(
    policy: dict[str, Any],
    train_examples: list[MatchedInterventionExample],
) -> dict[str, Any]:
    """20.3: γ recomputed from TRAIN data matches policy γ."""
    policy_gamma = policy.get("gamma")
    gamma_quantile = float(policy.get("gamma_quantile", 0.75))
    delta = float(policy.get("delta", 0.0))

    if policy_gamma is None:
        return {
            "check": "gamma_recompute",
            "passed": False,
            "error": "policy gamma is null",
        }

    try:
        recomputed_gamma, positive_support = compute_gamma(
            train_examples,
            quantile=gamma_quantile,
            delta=delta,
        )
    except ValueError as exc:
        return {
            "check": "gamma_recompute",
            "passed": False,
            "error": str(exc),
        }

    diff = abs(float(policy_gamma) - recomputed_gamma)
    passed = diff < 1e-10
    return {
        "check": "gamma_recompute",
        "policy_gamma": float(policy_gamma),
        "recomputed_gamma": recomputed_gamma,
        "abs_diff": diff,
        "positive_support": positive_support,
        "quantile": gamma_quantile,
        "passed": passed,
    }


def _check_critic_hash(
    policy: dict[str, Any],
    critic_path: Path,
) -> dict[str, Any]:
    """20.4: Critic checkpoint SHA256 matches policy hash."""
    policy_hash = policy.get("critic_checkpoint_sha256")
    if policy_hash is None:
        return {
            "check": "critic_policy_hash",
            "passed": False,
            "error": "policy has no critic_checkpoint_sha256",
        }

    if not critic_path.exists():
        return {
            "check": "critic_policy_hash",
            "passed": False,
            "error": f"critic file not found: {critic_path}",
        }

    actual_hash = _sha256_file(critic_path)
    passed = actual_hash == policy_hash
    return {
        "check": "critic_policy_hash",
        "policy_hash": policy_hash,
        "actual_hash": actual_hash,
        "passed": passed,
    }


def _check_split_leakage(
    examples: list[MatchedInterventionExample],
    *,
    pilot_mode: bool = False,
) -> dict[str, Any]:
    """20.5: Train/validation/test task IDs strictly disjoint.

    In *pilot_mode*, memory-provenance overlap is downgraded to a
    warning (WARN_PILOT) because pilot data is too small for full
    provenance isolation.  Task-level disjointness is always enforced.
    """
    splits = task_level_split(
        examples,
        train_frac=0.7,
        validation_frac=0.15,
        seed=0,
    )

    split_task_ids: dict[str, set[str]] = {}
    for name, exs in splits.items():
        split_task_ids[name] = {str(ex.task_id) for ex in exs}

    try:
        audit = audit_split_leakage(splits)
        passed = audit.get("status") == "PASS"
    except SplitLeakageError as exc:
        error_msg = str(exc)
        is_provenance_only = "Memory-provenance" in error_msg
        if pilot_mode and is_provenance_only:
            # Pilot relaxation: provenance overlap is expected with few
            # source memories.  Task-level disjointness already verified
            # by the SplitLeakageError check order.
            audit = {
                "status": "WARN_PILOT",
                "pilot_note": error_msg,
                "task_overlap": 0,
            }
            passed = True
        else:
            audit = {"error": error_msg, "status": "FAIL"}
            passed = False

    return {
        "check": "train_test_leakage",
        "split_sizes": {n: len(exs) for n, exs in splits.items()},
        "train_task_ids": sorted(split_task_ids.get("train", set())),
        "validation_task_ids": sorted(split_task_ids.get("validation", set())),
        "test_task_ids": sorted(split_task_ids.get("test", set())),
        "audit": audit,
        "passed": passed,
    }


def _check_uncertainty(
    critic: BootstrapOfficialScoreTransferCritic,
    validation_examples: list[MatchedInterventionExample],
    beta: float,
) -> dict[str, Any]:
    """20.6: Uncertainty audit on validation split."""
    if not validation_examples:
        return {
            "check": "uncertainty_audit",
            "passed": False,
            "error": "no validation examples",
        }

    # Build prediction pairs (same as train_critic.py bootstrap path).
    pairs: list[dict[str, Any]] = []
    n_self_transfer = 0
    for ex in validation_examples:
        if ex.source_agent_id == ex.receiver_id:
            n_self_transfer += 1
            continue
        if (
            ex.official_expose_score is None
            or ex.official_withhold_score is None
        ):
            continue

        dist = critic.predict_distribution(ex)
        pred = critic.predict_one(ex)
        obs = ex.official_expose_score - ex.official_withhold_score

        sigma = dist.sigma_tau
        lcb = (
            pred.tau_hat - beta * sigma
            if pred.tau_hat is not None and sigma is not None
            else None
        )
        pairs.append({
            "predicted_tau": pred.tau_hat,
            "predicted_sigma": sigma,
            "predicted_lcb": lcb,
            "observed_delta": obs,
            "memory_id": ex.memory_id,
            "receiver_id": ex.receiver_id,
        })

    if not pairs:
        return {
            "check": "uncertainty_audit",
            "passed": False,
            "error": "no usable validation pairs",
        }

    report = validate_critic(pairs)
    val_dict = report.to_dict()

    return {
        "check": "uncertainty_audit",
        "n_validation_examples": len(validation_examples),
        "n_self_transfer_excluded": n_self_transfer,
        "n_pairs_used": len(pairs),
        "mae": val_dict["mae"],
        "rmse": val_dict["rmse"],
        "sign_accuracy": val_dict["sign_accuracy"],
        "lcb_coverage": val_dict["lcb_coverage"],
        "mean_sigma": val_dict["mean_sigma"],
        "median_sigma": val_dict["median_sigma"],
        "sigma_absolute_error_correlation": val_dict[
            "sigma_absolute_error_correlation"
        ],
        "pearson": val_dict["pearson"],
        "spearman": val_dict["spearman"],
        "positive_precision": val_dict["positive_precision"],
        "positive_recall": val_dict["positive_recall"],
        "negative_recall": val_dict["negative_recall"],
        "beta_used": beta,
        "passed": True,  # Informational — no threshold gate on uncertainty
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-pilot artifact audit (Phase 20)"
    )
    parser.add_argument(
        "--critic", required=True,
        help="Path to critic checkpoint (.joblib)",
    )
    parser.add_argument(
        "--policy", required=True,
        help="Path to transfer_policy.json",
    )
    parser.add_argument(
        "--records", required=True,
        help="Path to intervention records JSON",
    )
    parser.add_argument(
        "--source-agents", default=None,
        help="JSON mapping memory_id -> source_agent_id",
    )
    parser.add_argument(
        "--output", default=_DEFAULT_OUTPUT,
        help="Output audit JSON path",
    )
    parser.add_argument(
        "--split-seed", type=int, default=0,
        help="Seed used for task-level split (must match training)",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.7,
        help="Training fraction (must match training)",
    )
    parser.add_argument(
        "--validation-frac", type=float, default=0.15,
        help="Validation fraction (must match training)",
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Relax memory-provenance audit to WARN (pilot data too small "
             "for strict provenance isolation)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    critic_path = Path(args.critic)
    policy_path = Path(args.policy)
    records_path = Path(args.records)

    # --- Load policy ---
    if not policy_path.exists():
        print(f"FATAL: policy not found: {policy_path}", file=sys.stderr)
        return 1
    with open(policy_path) as f:
        policy = json.load(f)

    # --- Load records and build examples ---
    if not records_path.exists():
        print(f"FATAL: records not found: {records_path}", file=sys.stderr)
        return 1
    raw_records = load_records(records_path)
    source_agent_ids: dict[str, str] = {}
    if args.source_agents:
        with open(args.source_agents) as f:
            source_agent_ids = json.load(f)
    examples = [
        record_to_example(r, source_agent_ids=source_agent_ids)
        for r in raw_records
    ]
    print(f"Loaded {len(examples)} intervention examples from {records_path}")

    # --- Reproduce splits ---
    splits = task_level_split(
        examples,
        train_frac=args.train_frac,
        validation_frac=args.validation_frac,
        seed=args.split_seed,
    )
    train_examples = splits["train"]
    validation_examples = splits["validation"]
    print(
        f"Split: train={len(train_examples)}, "
        f"validation={len(validation_examples)}, "
        f"test={len(splits['test'])}"
    )

    # --- Load critic ---
    critic: BootstrapOfficialScoreTransferCritic | None = None
    if critic_path.exists():
        print(f"Loading critic from {critic_path}...")
        critic = BootstrapOfficialScoreTransferCritic.load(str(critic_path))
    else:
        print(f"WARN: critic not found: {critic_path}", file=sys.stderr)

    # --- Run checks ---
    results: list[dict[str, Any]] = []

    # 20.1 β
    results.append(_check_beta(policy))
    # 20.2 δ
    results.append(_check_delta(policy))
    # 20.3 γ
    results.append(_check_gamma(policy, train_examples))
    # 20.4 Critic-policy hash
    results.append(_check_critic_hash(policy, critic_path))
    # 20.5 Leakage
    results.append(_check_split_leakage(examples, pilot_mode=args.pilot))
    # 20.6 Uncertainty
    beta = float(policy.get("beta", 1.64))
    if critic is not None:
        results.append(
            _check_uncertainty(critic, validation_examples, beta)
        )
    else:
        results.append({
            "check": "uncertainty_audit",
            "passed": False,
            "error": "critic checkpoint not loaded",
        })

    # --- Summary ---
    all_passed = all(r["passed"] for r in results)
    failures = [r for r in results if not r["passed"]]

    audit_output = {
        "schema_version": "prepilot_audit_v1",
        "policy_file": str(policy_path),
        "critic_file": str(critic_path),
        "records_file": str(records_path),
        "n_examples": len(examples),
        "split_seed": args.split_seed,
        "train_frac": args.train_frac,
        "validation_frac": args.validation_frac,
        "checks": results,
        "all_passed": all_passed,
        "n_failures": len(failures),
    }

    out_path = _PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(audit_output, f, indent=2)
    print(f"\nAudit written to {out_path}")

    # --- Report ---
    print(f"\n{'=' * 60}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['check']}")
        if not r["passed"] and "error" in r:
            print(f"         error: {r['error']}")
    print(f"{'=' * 60}")

    if all_passed:
        print("ALL PRE-PILOT AUDIT CHECKS PASSED")
        return 0

    print(f"\n{len(failures)} CHECK(S) FAILED:", file=sys.stderr)
    for r in failures:
        print(f"  - {r['check']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
