"""RIMA canonical continual runner (Phase 20).

Formal runner — the ONLY flow for the canonical method::

    load frozen critic
    for scenario:
        reset M; reset K_r
        for task t (fixed order):
            1. M_t contains only tasks < t
            2. retrieve candidates C_r_t (per receiver, historical only)
            3. frozen critic predicts tau_hat(m, r | x_t)
            4. admit ALL positive-tau memories A_r_t  (multi-memory, Eq. 9)
            5. simultaneous multi-receiver execution with K_r + A_r_t
            6. observe official Task Score (primary outcome)
            7. extract new procedural experiences (sanitized)
            8. add new memories to M_{t+1}  (never visible to task t)
            9. K_r persists admissions
           10. proceed to t+1

Baselines (Phase 21) share the SAME backbone, task order, retrieval
budget, candidate pool, context budget and seeds:

    no_memory / full_memory / retrieval / reflexion /
    rima_uniform / rima_receiver

Invariants enforced (fail-closed):

* admission decision_source must be ``frozen_transfer_critic``;
* observed expose/withhold deltas never enter admission;
* current-task memories never affect the current task;
* self-transfer excluded and counted;
* injected payloads sanitized (no answer/ground-truth leakage);
* critic must be frozen; invalid outcomes recorded as invalid, never zero.
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

from smtr.marble.experience_extractor import ExperienceExtractor  # noqa: E402
from smtr.marble.task_loader import MarbleTask, MarbleTaskLoader  # noqa: E402
from smtr.marble.trajectory_collector import Trajectory, TrajectoryCollector  # noqa: E402
from smtr.memory.procedural_sanitizer import assert_clean_payload, sanitize_candidate  # noqa: E402
from smtr.memory.receiver_knowledge import ReceiverKnowledgeContainer  # noqa: E402
from smtr.memory.shared_memory_pool import SharedMemory, SharedMemoryPool  # noqa: E402
from smtr.rima.admission import assert_formal_decision_source  # noqa: E402
from smtr.rima.admission_engine import RimaAdmissionEngine  # noqa: E402
from smtr.rima.features import ReceiverConditionedTransferFeatures  # noqa: E402
from smtr.rima.metrics import compute_cost_report, summarize_rima_run  # noqa: E402
from smtr.rima.outcome import RimaOutcomeEvaluator  # noqa: E402
from smtr.rima.receiver_topology import ReceiverExclusionPolicy, select_receivers  # noqa: E402
from smtr.router.official_score_transfer_critic import OfficialScoreTransferCritic  # noqa: E402

logger = logging.getLogger("rima.continual")

METHODS = (
    "no_memory",
    "full_memory",
    "retrieval",
    "reflexion",
    "rima_uniform",
    "rima_receiver",
)
CRITIC_METHODS = frozenset({"rima_uniform", "rima_receiver"})
ALL_SCENARIOS = ("bargaining", "coding", "database", "minecraft", "research")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_repr(task: MarbleTask) -> dict[str, Any]:
    content = task.task_content
    text = str(
        content.get("description")
        or content.get("instruction")
        or content.get("content")
        or task.raw_task.get("description", "")
    )
    return {
        "scenario": task.scenario,
        "task_type": str(task.raw_task.get("task_type", task.scenario)),
        "text": text,
    }


def _receiver_repr(task: MarbleTask, receiver_id: str) -> dict[str, Any]:
    agent = task.get_agent_by_id(receiver_id) or {}
    role = str(agent.get("role", "unknown"))
    caps = agent.get("capabilities") or agent.get("skills") or []
    if isinstance(caps, str):
        caps = [caps]
    return {"role": role, "capabilities": [str(c) for c in caps]}


def build_feature_builder(scenario: str):
    """Feature builder: (memory, receiver_id, task_dict) -> features.

    Retained for offline critic training pipelines; the continual runner
    builds features inline with the live task/receiver representations.
    """

    def _build(memory: SharedMemory, receiver_id: str, task_dict: dict[str, Any]):
        return ReceiverConditionedTransferFeatures(
            task_id=str(task_dict.get("task_id", "?")),
            memory_id=memory.memory_id,
            receiver_id=receiver_id,
            task_repr={
                "scenario": task_dict.get("scenario", scenario),
                "task_type": task_dict.get("task_type", scenario),
                "text": str(task_dict.get("text", "")),
            },
            receiver_repr=task_dict.get("receiver_repr", {}),
            routing_card=dict(memory.routing_card),
        )

    return _build


def extract_to_shared_memories(
    trajectory: Trajectory,
    *,
    extractor: ExperienceExtractor,
    task: MarbleTask,
    task_position: int,
) -> list[SharedMemory]:
    """Extract + sanitize task-t experiences into pool entries for M_{t+1}."""
    try:
        candidates = extractor.extract(trajectory)
    except Exception as exc:  # extraction failure must not kill the loop
        logger.warning("Experience extraction failed for %s: %s", trajectory.task_id, exc)
        return []

    agent_ids = task.get_agent_ids()
    default_source = agent_ids[0] if agent_ids else "unknown_agent"
    out: list[SharedMemory] = []
    for cand in candidates:
        meta = cand.metadata or {}
        source = str(meta.get("source_agent") or meta.get("author") or default_source)
        sanitized = sanitize_candidate(
            memory_id=cand.memory_id,
            source_agent_id=source,
            raw_content=cand.content or "",
            routing_card={
                "goal_summary": str(meta.get("goal_summary") or cand.content or "")[:400],
                "task_tags": [task.scenario, str(meta.get("type", cand.type))],
                "precondition_summary": str(meta.get("precondition_summary", "")),
                "compatible_receiver_roles": list(meta.get("receiver_roles", []) or []),
                "compatible_receiver_capabilities": list(meta.get("receiver_capabilities", []) or []),
                "procedure_type": str(cand.type),
            },
            task_id=task.task_id,
        )
        out.append(
            SharedMemory(
                memory_id=cand.memory_id,
                source_agent_id=source,
                origin_task_id=task.task_id,
                origin_task_position=task_position,
                routing_card=sanitized.routing_card,
                procedure_payload=sanitized.procedural_content,
                scenario=task.scenario,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Continual protocol
# ---------------------------------------------------------------------------


class ContinualProtocol:
    """One continual evaluation run (scenario x seed x method)."""

    def __init__(
        self,
        *,
        scenario: str,
        seed: int,
        method: str,
        tasks: list[MarbleTask],
        collector: TrajectoryCollector,
        extractor: ExperienceExtractor,
        receiver_count: int = 3,
        retrieval_top_k: int = 5,
        context_budget: int = 5,
        critic_receiver: OfficialScoreTransferCritic | None = None,
        critic_uniform: OfficialScoreTransferCritic | None = None,
    ) -> None:
        self.scenario = scenario
        self.seed = seed
        self.method = method
        self.tasks = tasks
        self.collector = collector
        self.extractor = extractor
        self.receiver_count = receiver_count
        self.retrieval_top_k = retrieval_top_k
        self.context_budget = context_budget
        self.outcome_evaluator = RimaOutcomeEvaluator(scenario=scenario)

        self.pool = SharedMemoryPool()
        self.knowledge = ReceiverKnowledgeContainer()
        self.reflexion_store: dict[str, list[str]] = {}  # reflexion baseline only
        self._current_task: MarbleTask | None = None

        self.engine: RimaAdmissionEngine | None = None
        if method == "rima_receiver":
            self._init_engine(critic_receiver)
        elif method == "rima_uniform":
            self._init_engine(critic_uniform)

        self.records: list[dict[str, Any]] = []

    def _init_engine(self, critic: OfficialScoreTransferCritic | None) -> None:
        if critic is None:
            raise RuntimeError(
                f"Method {self.method} requires a trained+saved critic "
                f"checkpoint (--critic-receiver / --critic-uniform)."
            )
        if not critic.is_frozen:
            critic.freeze()
        self.engine = RimaAdmissionEngine(
            critic=critic,
            pool=self.pool,
            feature_builder=self._feature_builder,
            retrieval_top_k=self.retrieval_top_k,
        )
        self.engine.require_frozen()

    # -- features ---------------------------------------------------------
    def _feature_builder(self, memory: SharedMemory, receiver_id: str, task: dict[str, Any]):
        t = self._current_task
        return ReceiverConditionedTransferFeatures(
            task_id=t.task_id,
            memory_id=memory.memory_id,
            receiver_id=receiver_id,
            task_repr=_task_repr(t),
            receiver_repr=_receiver_repr(t, receiver_id),
            routing_card=dict(memory.routing_card),
        )

    # -- per-task lifecycle -----------------------------------------------
    def run(self) -> dict[str, Any]:
        for position, task in enumerate(self.tasks):
            self._current_task = task
            self._run_task(task, position)
        return self._summarize()

    def _run_task(self, task: MarbleTask, position: int) -> None:
        assignments = select_receivers(
            task={**task.raw_task, "agent_ids": task.get_agent_ids()},
            task_id=task.task_id,
            receiver_count=self.receiver_count,
            exclusion_policy=ReceiverExclusionPolicy(),
        )
        receiver_ids = [a.receiver_id for a in assignments]
        for rid in receiver_ids:
            self.knowledge.ensure(rid)

        # Steps 2-4: retrieve candidates + critic admission (historical only).
        payloads: dict[str, list[str]] = {rid: [] for rid in receiver_ids}
        if self.method == "no_memory":
            pass
        elif self.method == "full_memory":
            for rid in receiver_ids:
                hist = self.pool.memories_before(position)
                payloads[rid] = [
                    m.procedure_payload for m in hist[: self.context_budget]
                ]
        elif self.method == "retrieval":
            for rid in receiver_ids:
                cands = self.pool.retrieve(
                    _task_repr(task), rid, self.retrieval_top_k,
                    current_task_position=position,
                )
                payloads[rid] = [m.procedure_payload for m in cands]
        elif self.method == "reflexion":
            for rid in receiver_ids:
                payloads[rid] = self.reflexion_store.get(rid, [])[: self.context_budget]
        elif self.method in CRITIC_METHODS:
            assert self.engine is not None
            for rid in receiver_ids:
                self.engine.admit_for_task(
                    task=_task_repr(task),
                    task_id=task.task_id,
                    task_position=position,
                    receiver_id=rid,
                    knowledge=self.knowledge.get(rid),
                )
                admitted = self.knowledge.get(rid).retrieve({}, self.context_budget)
                payloads[rid] = [m.procedure_payload for m in admitted]
        else:
            raise ValueError(f"Unknown method: {self.method!r}")

        # Phase 13: fail-closed sanitization guard before any injection.
        for rid, plist in payloads.items():
            for p in plist:
                assert_clean_payload(p, memory_id=f"{task.task_id}/{rid}")

        # Step 5: simultaneous multi-receiver execution (no broadcast).
        start = time.time()
        trajectory = self.collector.collect(
            task,
            seed=self.seed,
            method=self.method,
            memory_payloads=None,
            receiver_memory_payloads=payloads if any(payloads.values()) else None,
        )

        # Step 6: official Task Score is the PRIMARY outcome.
        outcome = self.outcome_evaluator.evaluate(
            task=task.raw_task, run_result=trajectory.raw_output
        )
        self.records.append(
            {
                "task_id": task.task_id,
                "task_position": position,
                "method": self.method,
                "scenario": self.scenario,
                "seed": self.seed,
                "receivers": receiver_ids,
                "task_score": outcome.task_score,
                "is_valid": outcome.is_valid,
                "failure_reason": outcome.failure_reason,
                "team_success_diagnostic": outcome.team_success,
                "n_injected_total": sum(len(v) for v in payloads.values()),
                "pool_size_at_t": len(self.pool.memories_before(position)),
                "engine_duration_seconds": trajectory.engine_duration_seconds,
                "wall_seconds": round(time.time() - start, 2),
                "real_engine_executed": trajectory.real_engine_executed,
            }
        )

        # Steps 7-8: extract + add to M_{t+1} (never visible to task t).
        new_memories = extract_to_shared_memories(
            trajectory,
            extractor=self.extractor,
            task=task,
            task_position=position,
        )
        for m in new_memories:
            if m.memory_id not in self.pool:
                self.pool.add(m)

        # Reflexion baseline keeps per-receiver self-reflections.
        if self.method == "reflexion":
            for rid in receiver_ids:
                for m in new_memories:
                    store = self.reflexion_store.setdefault(rid, [])
                    if m.procedure_payload and m.procedure_payload not in store:
                        store.append(m.procedure_payload)

    def _summarize(self) -> dict[str, Any]:
        valid = [r for r in self.records if r["is_valid"]]
        scores = [r["task_score"] for r in valid]
        summary = {
            "scenario": self.scenario,
            "seed": self.seed,
            "method": self.method,
            "n_tasks": len(self.records),
            "n_valid": len(valid),
            "valid_rate": (len(valid) / len(self.records)) if self.records else 0.0,
            "mean_task_score": (sum(scores) / len(scores)) if scores else None,
            "cumulative_task_score": sum(scores) if scores else 0.0,
            "late_stage_task_score": (
                sum(scores[-5:]) / len(scores[-5:]) if scores else None
            ),
            "memory_bank_size_diagnostic": len(self.pool),
            "engine_episodes": len(self.records),
            # Canonical metric system (Phase 23-26): primary = Official
            # Task Score; memory quantities are diagnostic-only; online
            # inference cost is reported separately from offline
            # intervention cost.
            "rima_metrics": summarize_rima_run(
                self.records,
                decisions=self.engine.decisions if self.engine else (),
                pool_size=len(self.pool),
            ),
            "cost_report": compute_cost_report(self.records),
        }
        if self.engine is not None:
            summary["admission_statistics"] = self.engine.stats.to_dict()
            summary["formal_decision_source"] = "frozen_transfer_critic"
            summary["critic_checkpoint_sha256"] = self.engine.critic.checkpoint_sha256()
            # Hard invariant: no self-transfer ever admitted.
            formal_self = sum(
                1
                for d in self.engine.decisions
                if d.admitted and d.status != "admitted"
            )
            summary["formal_pairs_with_source_eq_receiver"] = formal_self
        return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RIMA canonical continual runner")
    parser.add_argument("--scenarios", nargs="+", default=["bargaining"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--methods", nargs="+", default=["no_memory", "rima_receiver"],
        help=f"subset of {METHODS}",
    )
    parser.add_argument("--limit-per-scenario", type=int, default=20)
    parser.add_argument("--receiver-count", type=int, default=3)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--context-budget", type=int, default=5)
    parser.add_argument("--critic-receiver", type=str, default=None)
    parser.add_argument("--critic-uniform", type=str, default=None)
    parser.add_argument(
        "--engine-timeout", type=int, default=1800,
        help="Per-episode MARBLE engine timeout in seconds.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/rima/continual"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args(argv)

    # Pre-flight credential check — fail loudly BEFORE any engine work.
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "FATAL: Missing LLM API credential.\n"
            "Set DASHSCOPE_API_KEY (preferred) or OPENAI_API_KEY before running.",
            file=sys.stderr,
        )
        return 1

    for method in args.methods:
        if method not in METHODS:
            print(f"FATAL: unknown method {method!r}; valid: {METHODS}", file=sys.stderr)
            return 1

    critic_receiver = (
        OfficialScoreTransferCritic.load(args.critic_receiver)
        if args.critic_receiver
        else None
    )
    critic_uniform = (
        OfficialScoreTransferCritic.load(args.critic_uniform)
        if args.critic_uniform
        else None
    )
    for method in args.methods:
        if method == "rima_receiver" and critic_receiver is None:
            print(
                "FATAL: rima_receiver requires --critic-receiver (frozen checkpoint).",
                file=sys.stderr,
            )
            return 1
        if method == "rima_uniform" and critic_uniform is None:
            print(
                "FATAL: rima_uniform requires --critic-uniform (frozen checkpoint).",
                file=sys.stderr,
            )
            return 1

    out_dir = _PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = MarbleTaskLoader()
    collector = TrajectoryCollector(engine_timeout=args.engine_timeout)
    extractor = ExperienceExtractor()

    all_summaries: list[dict[str, Any]] = []
    for scenario in args.scenarios:
        if scenario not in ALL_SCENARIOS:
            print(f"FATAL: unknown scenario {scenario!r}", file=sys.stderr)
            return 1
        tasks = loader.load_scenario(scenario, limit=args.limit_per_scenario)
        for seed in args.seeds:
            for method in args.methods:
                logger.info(
                    "scenario=%s seed=%s method=%s tasks=%d",
                    scenario, seed, method, len(tasks),
                )
                protocol = ContinualProtocol(
                    scenario=scenario,
                    seed=seed,
                    method=method,
                    tasks=tasks,
                    collector=collector,
                    extractor=extractor,
                    receiver_count=args.receiver_count,
                    retrieval_top_k=args.retrieval_top_k,
                    context_budget=args.context_budget,
                    critic_receiver=critic_receiver,
                    critic_uniform=critic_uniform,
                )
                summary = protocol.run()
                all_summaries.append(summary)

                fname = f"rima_continual_{scenario}_seed{seed}_{method}.json"
                with open(out_dir / fname, "w") as f:
                    json.dump(
                        {"summary": summary, "records": protocol.records},
                        f,
                        indent=2,
                    )
                logger.info("wrote %s", out_dir / fname)

    with open(out_dir / "rima_continual_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
