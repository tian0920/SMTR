"""Transfer continual runner — RIMA-v2 formal method (§27-37).

Independent from the existing ``run_continual_main.py``.

Methods::

    no_memory       — no memory injection
    retrieval       — relevance-based retrieval from global pool
    rima_receiver   — old static causal admission (baseline)
    rima_transfer   — new TransferAwareMemoryController

Invariants (fail-closed):

* critic must be frozen (RuntimeError if not)
* context_budget = 1 for rima_transfer
* no online critic refit
* no online tau update from task outcomes
* transfer metadata never enters LLM prompt
* self-transfer excluded
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
from smtr.memory.procedural_sanitizer import assert_clean_payload  # noqa: E402
from smtr.memory.receiver_knowledge import ReceiverKnowledgeContainer  # noqa: E402
from smtr.memory.shared_memory_pool import SharedMemory, SharedMemoryPool  # noqa: E402
from smtr.rima.admission_engine import RimaAdmissionEngine  # noqa: E402
from smtr.rima.features import ReceiverConditionedTransferFeatures  # noqa: E402
from smtr.rima.metrics import compute_cost_report, summarize_rima_run  # noqa: E402
from smtr.rima.outcome import RimaOutcomeEvaluator  # noqa: E402
from smtr.rima.receiver_topology import ReceiverExclusionPolicy, select_receivers  # noqa: E402
from smtr.rima.transfer_controller import TransferAwareMemoryController  # noqa: E402
from smtr.rima.transfer_metrics import (  # noqa: E402
    compute_transfer_cost,
    compute_transfer_routing_metrics,
)
from smtr.rima.transfer_policy import TransferPolicy  # noqa: E402
from smtr.rima.transfer_state import ReceiverTransferStateContainer  # noqa: E402
from smtr.router.official_score_transfer_critic import (  # noqa: E402
    BootstrapOfficialScoreTransferCritic,
    OfficialScoreTransferCritic,
)

# Reuse helpers from existing runner.
from experiments.rima.run_continual_main import (  # noqa: E402
    _task_repr,
    _receiver_repr,
    extract_to_shared_memories,
)

logger = logging.getLogger("rima.continual_transfer")

METHODS = (
    "no_memory",
    "retrieval",
    "rima_receiver",
    "rima_transfer",
)
ALL_SCENARIOS = ("bargaining", "coding", "database", "minecraft", "research")


# ---------------------------------------------------------------------------
# Transfer policy loader
# ---------------------------------------------------------------------------


def load_transfer_policy(path: str) -> TransferPolicy:
    """Load a frozen TransferPolicy from a JSON file."""
    with open(path) as f:
        d = json.load(f)
    return TransferPolicy(
        beta=float(d["beta"]),
        delta=float(d["delta"]),
        gamma=float(d["gamma"]),
        gamma_quantile=float(d.get("gamma_quantile", 0.75)),
        gamma_positive_support=int(d.get("gamma_positive_support", 0)),
        gamma_source_split=str(d.get("gamma_source_split", "train")),
        critic_checkpoint_sha256=d.get("critic_checkpoint_sha256"),
    )


# ---------------------------------------------------------------------------
# Transfer continual protocol
# ---------------------------------------------------------------------------


class TransferContinualProtocol:
    """Continual evaluation run for rima_transfer and baselines (§27-37).

    Supports four methods:

    * ``no_memory`` — no memory injection
    * ``retrieval`` — relevance-based retrieval from global pool
    * ``rima_receiver`` — old static causal admission (baseline)
    * ``rima_transfer`` — new TransferAwareMemoryController
    """

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
        context_budget: int = 1,
        critic_receiver: BootstrapOfficialScoreTransferCritic | None = None,
        transfer_policy: TransferPolicy | None = None,
        known_probe_top_k: int = 20,
        global_explore_top_k: int = 5,
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

        # §25: rima_transfer enforces context_budget=1
        if method == "rima_transfer" and context_budget != 1:
            raise ValueError(
                "rima_transfer v1 uses single-memory exposure only"
            )

        self.pool = SharedMemoryPool()
        self.knowledge = ReceiverKnowledgeContainer()
        self.transfer_states = ReceiverTransferStateContainer()
        self._current_task: MarbleTask | None = None

        # Controller (rima_transfer) or engine (rima_receiver baseline).
        self.controller: TransferAwareMemoryController | None = None
        self.engine: RimaAdmissionEngine | None = None

        if method == "rima_transfer":
            if critic_receiver is None:
                raise RuntimeError(
                    "rima_transfer requires a trained+frozen bootstrap "
                    "critic checkpoint (--critic-receiver)."
                )
            if transfer_policy is None:
                raise RuntimeError(
                    "rima_transfer requires a frozen transfer policy "
                    "(--transfer-policy)."
                )
            # §29: fail-closed — critic MUST already be frozen
            if not critic_receiver.is_frozen:
                raise RuntimeError(
                    "Critic is not frozen. The formal continual runner "
                    "requires a frozen checkpoint. Do NOT call freeze() "
                    "here — retrain with freeze enabled."
                )
            self.controller = TransferAwareMemoryController(
                critic=critic_receiver,
                pool=self.pool,
                transfer_states=self.transfer_states,
                policy=transfer_policy,
                feature_builder=self._transfer_feature_builder,
                known_probe_top_k=known_probe_top_k,
                global_explore_top_k=global_explore_top_k,
                context_budget=1,
            )
        elif method == "rima_receiver":
            self._init_engine(critic_receiver)

        self.records: list[dict[str, Any]] = []
        self.routing_diagnostics: list[dict[str, Any]] = []

    # -- baseline engine init ------------------------------------------

    def _init_engine(
        self,
        critic: OfficialScoreTransferCritic | None,
    ) -> None:
        if critic is None:
            raise RuntimeError(
                f"Method {self.method} requires a trained+saved critic "
                f"checkpoint (--critic-receiver)."
            )
        if not critic.is_frozen:
            critic.freeze()
        self.engine = RimaAdmissionEngine(
            critic=critic,
            pool=self.pool,
            feature_builder=self._legacy_feature_builder,
            retrieval_top_k=self.retrieval_top_k,
        )
        self.engine.require_frozen()

    # -- feature builders ----------------------------------------------

    def _transfer_feature_builder(
        self,
        memory: SharedMemory,
        receiver_id: str,
        task: dict[str, Any],
        task_id: str,
    ) -> ReceiverConditionedTransferFeatures:
        t = self._current_task
        assert t is not None
        return ReceiverConditionedTransferFeatures(
            task_id=task_id,
            memory_id=memory.memory_id,
            receiver_id=receiver_id,
            task_repr=_task_repr(t),
            receiver_repr=_receiver_repr(t, receiver_id),
            routing_card=dict(memory.routing_card),
        )

    def _legacy_feature_builder(
        self,
        memory: SharedMemory,
        receiver_id: str,
        task: dict[str, Any],
    ) -> ReceiverConditionedTransferFeatures:
        t = self._current_task
        assert t is not None
        return ReceiverConditionedTransferFeatures(
            task_id=t.task_id,
            memory_id=memory.memory_id,
            receiver_id=receiver_id,
            task_repr=_task_repr(t),
            receiver_repr=_receiver_repr(t, receiver_id),
            routing_card=dict(memory.routing_card),
        )

    # -- main loop -----------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run all tasks and return summary."""
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

        # ---- Build per-receiver payloads ----
        payloads: dict[str, list[str]] = {rid: [] for rid in receiver_ids}
        task_routing_diags: list[dict[str, Any]] = []

        if self.method == "no_memory":
            pass

        elif self.method == "retrieval":
            for rid in receiver_ids:
                cands = self.pool.retrieve(
                    _task_repr(task),
                    rid,
                    self.retrieval_top_k,
                    current_task_position=position,
                )
                payloads[rid] = [m.procedure_payload for m in cands]

        elif self.method == "rima_receiver":
            assert self.engine is not None
            for rid in receiver_ids:
                self.engine.admit_for_task(
                    task=_task_repr(task),
                    task_id=task.task_id,
                    task_position=position,
                    receiver_id=rid,
                    knowledge=self.knowledge.get(rid),
                )
                admitted = self.knowledge.get(rid).retrieve(
                    {}, self.context_budget
                )
                payloads[rid] = [m.procedure_payload for m in admitted]

        elif self.method == "rima_transfer":
            assert self.controller is not None
            for rid in receiver_ids:
                plan = self.controller.plan_for_task(
                    task=_task_repr(task),
                    task_id=task.task_id,
                    task_position=position,
                    receiver_id=rid,
                )
                for mid in plan.selected_memory_ids:
                    mem = self.pool.get(mid)
                    if mem is not None:
                        payloads[rid].append(mem.procedure_payload)
                task_routing_diags.append(
                    self._build_routing_diagnostic(plan, rid, position)
                )
        else:
            raise ValueError(f"Unknown method: {self.method!r}")

        # Sanitization guard
        for rid, plist in payloads.items():
            for p in plist:
                assert_clean_payload(p, memory_id=f"{task.task_id}/{rid}")

        # ---- Execute ----
        start = time.time()
        trajectory = self.collector.collect(
            task,
            seed=self.seed,
            method=self.method,
            memory_payloads=None,
            receiver_agent_ids=(
                receiver_ids if any(payloads.values()) else None
            ),
            receiver_memory_payloads=(
                payloads if any(payloads.values()) else None
            ),
        )

        # ---- Evaluate ----
        outcome = self.outcome_evaluator.evaluate(
            task=task.raw_task, run_result=trajectory.raw_output
        )

        record: dict[str, Any] = {
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
        if task_routing_diags:
            record["routing_diagnostics"] = task_routing_diags
        self.records.append(record)
        self.routing_diagnostics.extend(task_routing_diags)

        # ---- Extract memories → M_{t+1} (§37: no tau update) ----
        new_memories = extract_to_shared_memories(
            trajectory,
            extractor=self.extractor,
            task=task,
            task_position=position,
        )
        for m in new_memories:
            if m.memory_id not in self.pool:
                self.pool.add(m)

    # -- routing diagnostic builder (§33) --------------------------------

    def _build_routing_diagnostic(
        self, plan: Any, receiver_id: str, position: int
    ) -> dict[str, Any]:
        state = self.transfer_states.get(receiver_id)

        best_known_mu: float | None = None
        best_known_sigma: float | None = None
        for c in plan.known_candidates:
            if c.mu_tau is not None:
                if best_known_mu is None or (
                    c.lcb is not None
                    and c.lcb > (plan.best_known_lcb or float("-inf"))
                ):
                    best_known_mu = c.mu_tau
                    best_known_sigma = c.sigma_tau

        # Find best known LCB candidate details
        best_known_cand = None
        for c in plan.known_candidates:
            if c.lcb is not None:
                if best_known_cand is None or c.lcb > (
                    best_known_cand.lcb or float("-inf")
                ):
                    best_known_cand = c
        if best_known_cand is not None:
            best_known_mu = best_known_cand.mu_tau
            best_known_sigma = best_known_cand.sigma_tau

        selected_mid: str | None = None
        selected_source: str = "none"
        selected_mu: float | None = None
        selected_sigma: float | None = None
        selected_lcb: float | None = None

        if plan.selected_memory_ids:
            selected_mid = plan.selected_memory_ids[0]
            for c in plan.known_candidates + plan.global_candidates:
                if c.memory_id == selected_mid and c.selected_for_context:
                    selected_source = c.candidate_source
                    selected_mu = c.mu_tau
                    selected_sigma = c.sigma_tau
                    selected_lcb = c.lcb
                    break

        policy = self.controller.policy if self.controller else None

        return {
            "receiver_id": receiver_id,
            "task_position": position,
            "routing_mode": plan.routing_mode,
            "transfer_state_size_before": len(state) if state else 0,
            "transfer_state_size_after": len(state) if state else 0,
            "n_known_candidates_considered": len(plan.known_candidates),
            "n_global_candidates_considered": len(plan.global_candidates),
            "global_retrieval_triggered": plan.global_retrieval_triggered,
            "best_known_mu": best_known_mu,
            "best_known_sigma": best_known_sigma,
            "best_known_lcb": plan.best_known_lcb,
            "beta": policy.beta if policy else None,
            "delta": policy.delta if policy else None,
            "gamma": policy.gamma if policy else None,
            "selected_memory_id": selected_mid,
            "selected_source": selected_source,
            "selected_mu": selected_mu,
            "selected_sigma": selected_sigma,
            "selected_lcb": selected_lcb,
            "global_candidate_ids": [
                c.memory_id for c in plan.global_candidates
            ],
        }

    # -- summary -------------------------------------------------------

    def _summarize(self) -> dict[str, Any]:
        valid = [r for r in self.records if r["is_valid"]]
        scores = [r["task_score"] for r in valid]
        summary: dict[str, Any] = {
            "scenario": self.scenario,
            "seed": self.seed,
            "method": self.method,
            "n_tasks": len(self.records),
            "n_valid": len(valid),
            "valid_rate": (
                (len(valid) / len(self.records)) if self.records else 0.0
            ),
            "mean_task_score": (
                (sum(scores) / len(scores)) if scores else None
            ),
            "cumulative_task_score": sum(scores) if scores else 0.0,
            "late_stage_task_score": (
                (sum(scores[-5:]) / len(scores[-5:])) if scores else None
            ),
            "memory_bank_size_diagnostic": len(self.pool),
            "engine_episodes": len(self.records),
            "rima_metrics": summarize_rima_run(
                self.records,
                decisions=(
                    self.engine.decisions if self.engine else ()
                ),
                pool_size=len(self.pool),
            ),
            "cost_report": compute_cost_report(self.records),
        }

        # Transfer-specific metrics (rima_transfer only)
        if self.routing_diagnostics:
            summary["transfer_routing_metrics"] = (
                compute_transfer_routing_metrics(self.routing_diagnostics)
            )
            summary["transfer_cost"] = compute_transfer_cost(
                self.routing_diagnostics
            )

        if self.engine is not None:
            summary["admission_statistics"] = self.engine.stats.to_dict()
            summary["formal_decision_source"] = "frozen_transfer_critic"
            summary["critic_checkpoint_sha256"] = (
                self.engine.critic.checkpoint_sha256()
            )

        return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RIMA-Transfer continual runner (§27-37)"
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=["bargaining"]
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0]
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["no_memory", "rima_transfer"],
        help=f"subset of {METHODS}",
    )
    parser.add_argument("--limit-per-scenario", type=int, default=20)
    parser.add_argument("--receiver-count", type=int, default=3)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument(
        "--context-budget",
        type=int,
        default=1,
        help="Forced to 1 for rima_transfer.",
    )
    parser.add_argument("--critic-receiver", type=str, default=None)
    parser.add_argument("--transfer-policy", type=str, default=None)
    parser.add_argument("--known-probe-top-k", type=int, default=20)
    parser.add_argument("--global-explore-top-k", type=int, default=5)
    parser.add_argument("--engine-timeout", type=int, default=1800)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/rima_transfer/continual",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s"
    )
    args = parse_args(argv)

    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    if not api_key:
        print(
            "FATAL: Missing LLM API credential.\n"
            "Set DASHSCOPE_API_KEY (preferred) or OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return 1

    for method in args.methods:
        if method not in METHODS:
            print(
                f"FATAL: unknown method {method!r}; valid: {METHODS}",
                file=sys.stderr,
            )
            return 1

    # Load critic
    critic_receiver = None
    if args.critic_receiver:
        critic_receiver = BootstrapOfficialScoreTransferCritic.load(
            args.critic_receiver
        )

    # Load transfer policy (§31: path only, no --beta/--gamma overrides)
    transfer_policy = None
    if args.transfer_policy:
        transfer_policy = load_transfer_policy(args.transfer_policy)

    # Pre-flight checks
    for method in args.methods:
        if method == "rima_transfer":
            if critic_receiver is None:
                print(
                    "FATAL: rima_transfer requires --critic-receiver "
                    "(frozen bootstrap checkpoint).",
                    file=sys.stderr,
                )
                return 1
            if transfer_policy is None:
                print(
                    "FATAL: rima_transfer requires --transfer-policy.",
                    file=sys.stderr,
                )
                return 1
            if not critic_receiver.is_frozen:
                print(
                    "FATAL: critic must be frozen (trained checkpoint).",
                    file=sys.stderr,
                )
                return 1
        if method == "rima_receiver" and critic_receiver is None:
            print(
                "FATAL: rima_receiver requires --critic-receiver.",
                file=sys.stderr,
            )
            return 1

    if "rima_transfer" in args.methods and args.context_budget != 1:
        print(
            "FATAL: rima_transfer v1 uses single-memory exposure only "
            "(context_budget must be 1).",
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
            print(
                f"FATAL: unknown scenario {scenario!r}", file=sys.stderr
            )
            return 1
        tasks = loader.load_scenario(
            scenario, limit=args.limit_per_scenario
        )
        for seed in args.seeds:
            for method in args.methods:
                logger.info(
                    "scenario=%s seed=%s method=%s tasks=%d",
                    scenario,
                    seed,
                    method,
                    len(tasks),
                )
                protocol = TransferContinualProtocol(
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
                    transfer_policy=transfer_policy,
                    known_probe_top_k=args.known_probe_top_k,
                    global_explore_top_k=args.global_explore_top_k,
                )
                summary = protocol.run()
                all_summaries.append(summary)

                fname = (
                    f"transfer_continual_{scenario}"
                    f"_seed{seed}_{method}.json"
                )
                with open(out_dir / fname, "w") as f:
                    json.dump(
                        {
                            "summary": summary,
                            "records": protocol.records,
                            "routing_diagnostics": (
                                protocol.routing_diagnostics
                            ),
                        },
                        f,
                        indent=2,
                    )
                logger.info("wrote %s", out_dir / fname)

    with open(out_dir / "transfer_continual_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
