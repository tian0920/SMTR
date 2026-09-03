"""RIMA critic training pipeline (Phases 16-17).

Stage A: historical intervention collection (records from the matched
         expose/withhold collector; already on disk),
Stage B: train the official-score critic on the task-level TRAIN split,
Stage C: freeze the critic and persist the checkpoint + audits.

The trained checkpoint is the ONLY admission authority in Stage E
(continual evaluation); Stage E never re-fits.

Outputs (in ``--output-dir``):

* ``critic_receiver.joblib``   (receiver-conditioned checkpoint)
* ``critic_uniform.joblib``    (receiver-agnostic ablation checkpoint)
* ``split_leakage_audit.json`` (task-level isolation proof)
* ``critic_validation.json``   (held-out metrics, Phase 18)
* ``training_report.json``     (config + training stats + sha256)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from smtr.rima.critic_validation import validate_critic  # noqa: E402
from smtr.rima.features import (  # noqa: E402
    ReceiverConditionedTransferFeatures,
    RimaFeatureEncoder,
)
from smtr.rima.splits import task_level_split, write_split_audit, audit_split_leakage, SplitLeakageError  # noqa: E402
from smtr.rima.transfer_policy import (  # noqa: E402
    TransferPolicy,
    compute_gamma,
)
from smtr.router.official_score_transfer_critic import (  # noqa: E402
    BootstrapOfficialScoreTransferCritic,
    MatchedInterventionExample,
    OfficialScoreTransferCritic,
)


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
    """Convert one stored intervention record into a training example.

    Routing-card-only features are reconstructed from record metadata —
    never from payloads or outcomes.
    """
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RIMA critic training (Stage B-C)")
    parser.add_argument("--records", required=True, help="intervention records JSON")
    parser.add_argument(
        "--source-agents", default=None,
        help="JSON mapping memory_id -> source_agent_id (self-transfer guard)",
    )
    parser.add_argument("--output-dir", default="results/rima/critic")
    parser.add_argument("--n-features", type=int, default=1024)
    parser.add_argument("--loss", default="huber", choices=["huber", "mse"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--validation-frac", type=float, default=0.15)
    parser.add_argument(
        "--skip-uniform", action="store_true",
        help="skip training the receiver-agnostic RIMA-Uniform critic",
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="relax memory-provenance audit to WARN (pilot data too small "
             "for strict task-level + provenance isolation)",
    )
    # RIMA-v2 bootstrap mode (§12)
    parser.add_argument(
        "--critic-mode", default="point", choices=["point", "bootstrap"],
        help="point (default, legacy) or bootstrap (RIMA-v2)",
    )
    parser.add_argument("--n-bootstrap", type=int, default=31)
    parser.add_argument("--beta", type=float, default=1.64)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--gamma-quantile", type=float, default=0.75)
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(Path(args.records))
    source_agent_ids: dict[str, str] = {}
    if args.source_agents:
        source_agent_ids = json.load(open(args.source_agents))

    examples = [record_to_example(r, source_agent_ids=source_agent_ids) for r in records]
    print(f"Loaded {len(examples)} intervention examples")

    # Phase 16: task-level split (record-level random split forbidden).
    splits = task_level_split(
        examples,
        train_frac=args.train_frac,
        validation_frac=args.validation_frac,
        seed=args.seed,
    )
    try:
        audit = write_split_audit(splits, str(out_dir / "split_leakage_audit.json"))
    except SplitLeakageError as exc:
        if not args.pilot:
            raise
        # Pilot mode: provenance overlap is expected with few source
        # memories × few tasks.  Downgrade to warning and continue.
        import warnings
        warnings.warn(f"Pilot split-leakage relaxed: {exc}")
        # Build a non-raising audit for the record.
        names = sorted(splits)
        pairs = []
        for i, a in enumerate(names):
            for b_name in names[i + 1:]:
                task_ov = {str(getattr(ex, "task_id", "?")) for ex in splits[a]} & \
                          {str(getattr(ex, "task_id", "?")) for ex in splits[b_name]}
                mem_ov = {str(getattr(ex, "memory_id", "?")) for ex in splits[a]} & \
                         {str(getattr(ex, "memory_id", "?")) for ex in splits[b_name]}
                pairs.append({"splits": [a, b_name],
                              "task_overlap": sorted(task_ov),
                              "memory_provenance_overlap": sorted(mem_ov)})
        audit = {
            "split_sizes": {n: len(splits[n]) for n in names},
            "pairs": pairs,
            "status": "WARN_PILOT",
            "pilot_note": str(exc),
        }
        with open(out_dir / "split_leakage_audit.json", "w") as _f:
            json.dump(audit, _f, indent=2)
    print(f"Split: {audit['split_sizes']} — leakage audit {audit['status']}")

    train, validation = splits["train"], splits["validation"]
    if not train:
        print("FATAL: empty training split", file=sys.stderr)
        return 1

    report: dict[str, Any] = {"loss": args.loss, "seed": args.seed}

    # ---------------------------------------------------------------
    # Bootstrap mode (RIMA-v2 §12)
    # ---------------------------------------------------------------
    if args.critic_mode == "bootstrap":
        encoder = RimaFeatureEncoder(n_features=args.n_features, include_receiver=True)
        critic = BootstrapOfficialScoreTransferCritic(
            encoder=encoder,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            loss=args.loss,
        )
        stats = critic.fit(train)
        critic.freeze()

        ckpt_name = "critic_receiver_bootstrap.joblib"
        ckpt_path = out_dir / ckpt_name
        sha = critic.save(str(ckpt_path))

        # Compute gamma from TRAIN split only (§10)
        gamma, positive_support = compute_gamma(
            train,
            quantile=args.gamma_quantile,
            delta=args.delta,
        )

        # Write transfer_policy.json (§11)
        policy = TransferPolicy(
            beta=args.beta,
            delta=args.delta,
            gamma=gamma,
            gamma_quantile=args.gamma_quantile,
            gamma_positive_support=positive_support,
            gamma_source_split="train",
            critic_checkpoint_sha256=sha,
        )
        policy_dict = {
            "schema_version": "rima_transfer_policy_v1",
            "beta": policy.beta,
            "delta": policy.delta,
            "gamma": policy.gamma,
            "gamma_quantile": policy.gamma_quantile,
            "gamma_definition": "q75_of_positive_observed_train_tau",
            "gamma_source_split": policy.gamma_source_split,
            "gamma_positive_support": policy.gamma_positive_support,
            "critic_checkpoint_sha256": policy.critic_checkpoint_sha256,
            "bootstrap_members": args.n_bootstrap,
            "bootstrap_cluster_unit": "task_id",
        }

        # ---- Phase 12/14: low-support warnings (artifact flags only) ----
        # Warnings never modify beta/delta/gamma automatically.
        warnings_list: list[str] = []

        # Phase 12: gamma Q75 is statistically meaningless with very few
        # positive edges. Flag it; do NOT change gamma.
        GAMMA_LOW_SUPPORT_THRESHOLD = 5
        gamma_low_support = positive_support < GAMMA_LOW_SUPPORT_THRESHOLD
        policy_dict["gamma_low_support_warning"] = gamma_low_support
        if gamma_low_support:
            warnings_list.append("GAMMA_LOW_SUPPORT")

        # Phase 14: critic training support audit.
        valid_train = [
            ex for ex in train
            if ex.official_expose_score is not None
            and ex.official_withhold_score is not None
        ]
        n_valid_edges = len(valid_train)
        task_families = {ex.task_id for ex in valid_train}
        n_positive_edges = sum(
            1 for ex in valid_train
            if (ex.official_expose_score - ex.official_withhold_score) > 0
        )
        # Minimum support targets (Phase 23-25): >=60 valid edges,
        # >=15 task families, >=15 positive edges.
        CRITIC_MIN_EDGES, CRITIC_MIN_FAMILIES, CRITIC_MIN_POSITIVES = 60, 15, 15
        critic_underpowered = (
            n_valid_edges < CRITIC_MIN_EDGES
            or len(task_families) < CRITIC_MIN_FAMILIES
            or n_positive_edges < CRITIC_MIN_POSITIVES
        )
        policy_dict["critic_low_support_warning"] = critic_underpowered
        policy_dict["critic_status"] = (
            "UNDERPOWERED" if critic_underpowered else "OK"
        )
        policy_dict["dataset_support"] = {
            "n_valid_edges": n_valid_edges,
            "n_task_families": len(task_families),
            "n_positive_edges": n_positive_edges,
            "n_negative_edges": n_valid_edges - n_positive_edges,
        }
        policy_dict["warnings"] = warnings_list
        for w in warnings_list:
            print(f"WARNING: {w}", file=sys.stderr)

        with open(out_dir / "transfer_policy.json", "w") as f:
            json.dump(policy_dict, f, indent=2)

        # Validation on held-out split
        pairs = []
        for ex in (validation or splits["test"]):
            dist = critic.predict_distribution(ex)
            pred = critic.predict_one(ex)
            observed = (
                None
                if ex.official_expose_score is None or ex.official_withhold_score is None
                else ex.official_expose_score - ex.official_withhold_score
            )
            pairs.append(
                {
                    "predicted_tau": pred.tau_hat,
                    "predicted_sigma": dist.sigma_tau,
                    "predicted_lcb": (
                        pred.tau_hat - args.beta * dist.sigma_tau
                        if pred.tau_hat is not None and dist.sigma_tau is not None
                        else None
                    ),
                    "observed_delta": observed,
                    "memory_id": ex.memory_id,
                    "receiver_id": ex.receiver_id,
                }
            )
        val_report = validate_critic(pairs)

        report["critic_receiver_bootstrap"] = {
            "checkpoint": str(ckpt_path),
            "critic_checkpoint_sha256": sha,
            "training_stats": stats,
            "frozen": True,
            "validation": val_report.to_dict(),
            "transfer_policy": policy_dict,
        }
        print(f"[bootstrap] sha256={sha[:16]} gamma={gamma:.4f} "
              f"positive_support={positive_support}")
        print(f"[bootstrap] validation={val_report.to_dict()}")

        with open(out_dir / "training_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {out_dir}/training_report.json")
        return 0

    # ---------------------------------------------------------------
    # Point mode (legacy, unchanged)
    # ---------------------------------------------------------------
    for name, include_receiver in (
        ("receiver", True),
        *(() if args.skip_uniform else (("uniform", False),)),
    ):
        encoder = RimaFeatureEncoder(n_features=args.n_features, include_receiver=include_receiver)
        critic = OfficialScoreTransferCritic(
            encoder=encoder, loss=args.loss, receiver_conditioned=include_receiver
        )
        stats = critic.fit(train)

        # Stage C: freeze BEFORE any held-out evaluation / deployment.
        critic.freeze()
        ckpt_path = out_dir / f"critic_{name}.joblib"
        sha = critic.save(str(ckpt_path))

        # Phase 18 validation on held-out split.
        pairs = []
        for ex in (validation or splits["test"]):
            pred = critic.predict_one(ex)
            observed = (
                None
                if ex.official_expose_score is None or ex.official_withhold_score is None
                else ex.official_expose_score - ex.official_withhold_score
            )
            pairs.append(
                {
                    "predicted_tau": pred.tau_hat,
                    "observed_delta": observed,
                    "memory_id": ex.memory_id,
                    "receiver_id": ex.receiver_id,
                }
            )
        val_report = validate_critic(pairs)

        report[f"critic_{name}"] = {
            "checkpoint": str(ckpt_path),
            "critic_checkpoint_sha256": sha,
            "training_stats": stats,
            "frozen": True,
            "validation": val_report.to_dict(),
        }
        print(f"[{name}] sha256={sha[:16]} validation={val_report.to_dict()}")

    with open(out_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_dir}/training_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
