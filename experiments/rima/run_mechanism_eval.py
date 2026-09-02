"""RIMA held-out mechanism evaluation (Phase 19).

Keeps REAL expose/withhold interventions but repurposes them:

* validate the FROZEN critic (predicted tau vs observed delta);
* NEVER use observed deltas for admission (purpose = mechanism eval).

Focus of the report:

* positive observed deltas exist;
* negative observed deltas exist;
* receiver disagreement (same memory, different receivers, different
  decisions);
* same memory opposite-sign deltas across receivers — the direct
  evidence for "same memory yields different transfer effects across
  receivers".

Outputs ``mechanism_eval.json`` with per-pair rows
(observed_delta, predicted_tau, receiver_id, memory_id, task_id).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from smtr.baselines.base_memory_controller import CandidateMemory  # noqa: E402
from smtr.marble.task_loader import MarbleTaskLoader  # noqa: E402
from smtr.memory.online_receiver_intervention import (  # noqa: E402
    OnlineReceiverInterventionEvaluator,
)
from smtr.rima.critic_validation import validate_critic  # noqa: E402
from smtr.rima.features import ReceiverConditionedTransferFeatures  # noqa: E402
from smtr.rima.intervention_collection import (  # noqa: E402
    InterventionPurpose,
    MatchedInterventionCollector,
)
from smtr.rima.receiver_topology import select_receivers  # noqa: E402
from smtr.router.official_score_transfer_critic import (  # noqa: E402
    MatchedInterventionExample,
    OfficialScoreTransferCritic,
)


def _candidate_from_dict(cand: dict[str, Any]) -> CandidateMemory:
    """Convert one candidates-json entry into an injectable CandidateMemory."""
    return CandidateMemory(
        memory_id=str(cand.get("memory_id", "?")),
        type=str(cand.get("type", "experience")),
        content=str(cand.get("content") or cand.get("procedure_payload") or ""),
        source_episode=int(cand.get("source_episode", 0) or 0),
        metadata=cand.get("metadata") or {},
    )


def _features_from(record: Any, task: Any, receiver_repr: dict[str, Any]):
    return ReceiverConditionedTransferFeatures(
        task_id=str(getattr(record, "task_id", "?")),
        memory_id=str(getattr(record, "memory_id", "?")),
        receiver_id=str(getattr(record, "receiver_id", "?")),
        task_repr={"scenario": task.scenario, "task_type": task.scenario, "text": ""},
        receiver_repr=receiver_repr,
        routing_card={"goal_summary": "", "task_tags": [task.scenario]},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RIMA held-out mechanism evaluation")
    parser.add_argument("--critic", required=True, help="frozen critic checkpoint")
    parser.add_argument("--scenarios", nargs="+", default=["bargaining"])
    parser.add_argument("--limit-per-scenario", type=int, default=10)
    parser.add_argument("--task-offset", type=int, default=0,
                        help="skip the first N tasks (used to reach held-out "
                             "tasks disjoint from the critic training set)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--candidates-json", required=True,
                        help="JSON mapping task_id -> candidate memory dicts")
    parser.add_argument("--source-agents", default=None,
                        help="JSON mapping memory_id -> source_agent_id")
    parser.add_argument("--output", default="results/rima/mechanism_eval/mechanism_eval.json")
    args = parser.parse_args(argv)

    if not (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("FATAL: Missing LLM API credential.", file=sys.stderr)
        return 1

    critic = OfficialScoreTransferCritic.load(args.critic)
    if not critic.is_frozen:
        critic.freeze()

    collector = MatchedInterventionCollector(
        purpose=InterventionPurpose.MECHANISM_EVAL,
        evaluator=OnlineReceiverInterventionEvaluator(),
    )
    loader = MarbleTaskLoader()
    with open(args.candidates_json) as f:
        candidates_by_task = json.load(f)
    source_agents = json.load(open(args.source_agents)) if args.source_agents else {}

    pairs: list[dict[str, Any]] = []
    for scenario in args.scenarios:
        all_tasks = loader.load_scenario(
            scenario, limit=args.task_offset + args.limit_per_scenario
        )
        tasks = all_tasks[args.task_offset:]
        for task in tasks:
            receivers = select_receivers(
                task={**task.raw_task, "agent_ids": task.get_agent_ids()},
                task_id=task.task_id, receiver_count=3
            )
            candidates = candidates_by_task.get(
                task.task_id, candidates_by_task.get("*", [])
            )
            for cand in candidates:
                memory_id = str(cand.get("memory_id"))
                candidate_mem = _candidate_from_dict(cand)
                for recv in receivers:
                    if source_agents.get(memory_id) == recv.receiver_id:
                        continue  # self-transfer excluded entirely
                    for seed in args.seeds:
                        record = collector.collect(candidate_mem, recv.receiver_id, task, seed=seed)
                        observed = record.delta if record.delta == record.delta else None
                        example = MatchedInterventionExample(
                            task_id=task.task_id,
                            memory_id=memory_id,
                            receiver_id=recv.receiver_id,
                            source_agent_id=source_agents.get(memory_id, ""),
                            official_expose_score=None,
                            official_withhold_score=None,
                            features=_features_from(record, task, {"role": recv.receiver_role}),
                        )
                        pred = critic.predict_one(example)
                        pairs.append(
                            {
                                "task_id": task.task_id,
                                "memory_id": memory_id,
                                "receiver_id": recv.receiver_id,
                                "observed_delta": observed,
                                "predicted_tau": pred.tau_hat,
                            }
                        )

    report = validate_critic(pairs)
    valid = [p for p in pairs if p["observed_delta"] is not None]
    positives = [p for p in valid if p["observed_delta"] > 0]
    negatives = [p for p in valid if p["observed_delta"] < 0]

    # Same-memory opposite-sign evidence.
    by_memory: dict[str, set[int]] = {}
    for p in valid:
        by_memory.setdefault(p["memory_id"], set()).add(
            1 if p["observed_delta"] > 0 else (-1 if p["observed_delta"] < 0 else 0)
        )
    opposite_sign = [m for m, signs in by_memory.items() if {1, -1} <= signs]
    disagreement = [m for m, signs in by_memory.items() if len(signs - {0}) > 1]

    out = {
        "purpose": InterventionPurpose.MECHANISM_EVAL,
        "decision_source_used_for_admission": None,  # admission never happens here
        "critic_checkpoint_sha256": critic.checkpoint_sha256(),
        "n_pairs": len(pairs),
        "validation": report.to_dict(),
        "mechanism_evidence": {
            "positive_observed_delta_exists": bool(positives),
            "negative_observed_delta_exists": bool(negatives),
            "receiver_disagreement_memories": disagreement,
            "opposite_sign_memories": opposite_sign,
        },
        "pairs": pairs,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path} ({len(pairs)} pairs)")
    print(json.dumps(out["mechanism_evidence"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
