"""RIMA Stage A: historical training-intervention collection (Phase 32).

Canonical pipeline for producing critic SUPERVISION data::

    1. Run no_memory episodes on the first ``--source-tasks`` tasks.
    2. Extract + sanitize procedural memories into the shared pool
       (M_{t+1} semantics: task-t memories never visible to task t).
    3. For every later task, collect matched expose/withhold
       interventions for (historical candidate, receiver) pairs via
       ``MatchedInterventionCollector`` with
       purpose=TRAINING_COLLECTION.

Legal use only: the observed deltas produced here are critic ground
truth. They NEVER drive formal admission (Phase 4 invariant).

Outputs (in ``--output-dir``):

* ``intervention_records.json``  (train_critic.py --records)
* ``candidates.json``            (run_mechanism_eval.py --candidates-json)
* ``source_agents.json``         (memory_id -> source_agent_id)
* ``collection_summary.json``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.rima.run_continual_main import (  # noqa: E402
    _receiver_repr,
    _task_repr,
    extract_to_shared_memories,
)
from smtr.baselines.base_memory_controller import CandidateMemory  # noqa: E402
from smtr.marble.experience_extractor import ExperienceExtractor  # noqa: E402
from smtr.marble.task_loader import MarbleTaskLoader  # noqa: E402
from smtr.marble.trajectory_collector import TrajectoryCollector  # noqa: E402
from smtr.memory.online_receiver_intervention import (  # noqa: E402
    OnlineReceiverInterventionEvaluator,
)
from smtr.memory.receiver_knowledge import ReceiverKnowledgeContainer  # noqa: E402
from smtr.memory.shared_memory_pool import SharedMemoryPool  # noqa: E402
from smtr.memory.shared_memory_pool import memory_task_relevance_score  # noqa: E402
from smtr.rima.intervention_collection import (  # noqa: E402
    InterventionPurpose,
    MatchedInterventionCollector,
)
from smtr.rima.receiver_topology import ReceiverExclusionPolicy, select_receivers  # noqa: E402

logger = logging.getLogger("rima.stage_a")


def _shared_to_candidate(mem) -> CandidateMemory:
    """Pool entry -> injectable CandidateMemory (payload preserved)."""
    return CandidateMemory(
        memory_id=mem.memory_id,
        type=str(mem.routing_card.get("procedure_type", "experience")),
        content=mem.procedure_payload,
        source_episode=int(mem.origin_task_position),
        metadata={
            "source_agent": mem.source_agent_id,
            "origin_task_id": mem.origin_task_id,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RIMA Stage A training-intervention collection"
    )
    parser.add_argument("--scenario", default="bargaining")
    parser.add_argument("--source-tasks", type=int, default=2,
                        help="first N tasks run no_memory to seed the pool")
    parser.add_argument("--intervention-tasks", type=int, default=3,
                        help="number of downstream tasks to intervene on")
    parser.add_argument("--max-candidates-per-task", type=int, default=2)
    parser.add_argument("--receiver-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidate-mode",
        choices=["sequential", "retrieval"],
        default="sequential",
        help=(
            "How to select intervention candidates from the pool. "
            "'sequential' = first N historical memories (original). "
            "'retrieval' = rank by task-relevance score (§53)."
        ),
    )
    parser.add_argument("--engine-timeout", type=int, default=1800)
    parser.add_argument("--output-dir", default="results/rima/stage_a")
    args = parser.parse_args(argv)

    if not (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("FATAL: Missing LLM API credential.", file=sys.stderr)
        return 1

    out_dir = _PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = MarbleTaskLoader()
    collector = TrajectoryCollector(engine_timeout=args.engine_timeout)
    extractor = ExperienceExtractor()
    knowledge = ReceiverKnowledgeContainer()
    pool = SharedMemoryPool()

    tasks = loader.load_scenario(
        args.scenario, limit=args.source_tasks + args.intervention_tasks
    )
    source_tasks, intervention_tasks = (
        tasks[: args.source_tasks],
        tasks[args.source_tasks:],
    )
    logger.info(
        "Stage A: %d source tasks -> %d intervention tasks",
        len(source_tasks), len(intervention_tasks),
    )

    # -- Step 1-2: seed the pool from no_memory episodes -----------------
    for position, task in enumerate(source_tasks):
        traj = collector.collect(task, seed=args.seed, method="no_memory")
        memories = extract_to_shared_memories(
            traj, extractor=extractor, task=task, task_position=position
        )
        added = 0
        for m in memories:
            if m.memory_id not in pool:
                pool.add(m)
                added += 1
        logger.info(
            "source task %s: valid=%s score=%s extracted=%d added=%d pool=%d",
            task.task_id, traj.official_metric_valid,
            traj.official_metric_normalized, len(memories), added, len(pool),
        )

    if len(pool) == 0:
        print("FATAL: no memories extracted from source tasks.", file=sys.stderr)
        return 1

    source_agents = {m.memory_id: m.source_agent_id for m in pool.memories_before(10**9)}

    # -- Step 3: matched expose/withhold interventions --------------------
    intervention_collector = MatchedInterventionCollector(
        purpose=InterventionPurpose.TRAINING_COLLECTION,
        evaluator=OnlineReceiverInterventionEvaluator(collector=collector),
    )

    records: list[dict[str, Any]] = []
    candidates_by_task: dict[str, list[dict[str, Any]]] = {}
    n_self_excluded = 0

    for offset, task in enumerate(intervention_tasks):
        position = args.source_tasks + offset
        hist = pool.memories_before(position)  # historical-only candidates

        # Candidate selection mode (§53 / §8 retrieval alignment)
        if args.candidate_mode == "retrieval":
            # Rank all historical memories by lexical relevance to this task
            scored = [
                (m, memory_task_relevance_score(m, task.raw_task))
                for m in hist
            ]
            scored.sort(key=lambda pair: (-pair[1], pair[0].memory_id))
            cands = [m for m, _ in scored[: args.max_candidates_per_task]]
        else:
            cands = hist[: args.max_candidates_per_task]
        assignments = select_receivers(
            task={**task.raw_task, "agent_ids": task.get_agent_ids()},
            task_id=task.task_id,
            receiver_count=args.receiver_count,
            exclusion_policy=ReceiverExclusionPolicy(),
        )
        task_candidates: list[dict[str, Any]] = []
        task_repr = _task_repr(task)
        for mem in cands:
            for asg in assignments:
                if mem.source_agent_id == asg.receiver_id:
                    n_self_excluded += 1
                    continue  # self-transfer excluded entirely
                start = time.time()
                rec = intervention_collector.collect(
                    _shared_to_candidate(mem), asg.receiver_id, task, seed=args.seed
                )
                row = rec.to_dict()
                row.update(
                    {
                        "task_type": task_repr["task_type"],
                        "task_text": task_repr["text"],
                        "receiver_role": asg.receiver_role,
                        "receiver_capabilities": _receiver_repr(task, asg.receiver_id).get(
                            "capabilities", []
                        ),
                        "memory_goal_summary": str(mem.routing_card.get("goal_summary", "")),
                        "memory_task_tags": list(mem.routing_card.get("task_tags", [])),
                        "memory_precondition": str(
                            mem.routing_card.get("precondition_summary", "")
                        ),
                        "memory_receiver_roles": list(
                            mem.routing_card.get("compatible_receiver_roles", [])
                        ),
                        "memory_receiver_capabilities": list(
                            mem.routing_card.get("compatible_receiver_capabilities", [])
                        ),
                        "memory_type": str(mem.routing_card.get("procedure_type", "experience")),
                        "source_agent_id": mem.source_agent_id,
                        "wall_seconds": round(time.time() - start, 1),
                        "purpose": InterventionPurpose.TRAINING_COLLECTION,
                    }
                )
                records.append(row)
                task_candidates.append(
                    {
                        "memory_id": mem.memory_id,
                        "type": row["memory_type"],
                        "content": mem.procedure_payload,
                        "source_episode": int(mem.origin_task_position),
                    }
                )
                logger.info(
                    "pair %s/%s/%s: decision=%s delta=%s (%.0fs)",
                    task.task_id, mem.memory_id, asg.receiver_id,
                    rec.decision, rec.delta, row["wall_seconds"],
                )
        if task_candidates:
            # Deduplicate candidates per task (same memory may serve several
            # receivers); mechanism eval iterates receivers itself.
            seen: set[str] = set()
            candidates_by_task[str(task.task_id)] = [
                c for c in task_candidates if not (c["memory_id"] in seen or seen.add(c["memory_id"]))
            ]

        # Incremental persistence so long runs survive interruptions.
        with open(out_dir / "intervention_records.json", "w") as f:
            json.dump({"purpose": InterventionPurpose.TRAINING_COLLECTION,
                       "records": records}, f, indent=2)

    valid = [r for r in records if r["expose_metric_valid"] and r["withhold_metric_valid"]]
    summary = {
        "scenario": args.scenario,
        "seed": args.seed,
        "n_source_tasks": len(source_tasks),
        "n_intervention_tasks": len(intervention_tasks),
        "pool_size": len(pool),
        "n_pairs": len(records),
        "n_valid_pairs": len(valid),
        "valid_pair_rate": (len(valid) / len(records)) if records else 0.0,
        "self_transfer_excluded_count": n_self_excluded,
        "decision_source_used_for_admission": None,
        "candidate_mode": args.candidate_mode,
    }
    with open(out_dir / "candidates.json", "w") as f:
        json.dump(candidates_by_task, f, indent=2)
    with open(out_dir / "source_agents.json", "w") as f:
        json.dump(source_agents, f, indent=2)
    with open(out_dir / "collection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    raise SystemExit(main())
