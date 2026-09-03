"""Transfer continual runner — RIMA-v2 formal method + pilot variants.

Legacy methods (``--methods``)::

    no_memory       — no memory injection
    retrieval       — relevance-based retrieval from global pool
    rima_receiver   — old static causal admission (baseline)
    rima_transfer   — new TransferAwareMemoryController

Method variants (``--method-variant``)::

    rima_receiver               — static RIMA baseline
    rima_transfer_frozen        — frozen transfer cache
    rima_transfer_adaptive      — full adaptive continual transfer
    rima_transfer_positive_stop — ablation: stop when bestLCB > delta
    rima_transfer_no_uncertainty— ablation: score = mu (no LCB)
    rima_static_same_probe_budget — cost-matched baseline

Phase 25 raw data output (MethodVariant mode): per-run directories with
run_manifest.json, tasks.jsonl, routing.jsonl, probe_events.jsonl,
critic_versions.jsonl, costs.json, summary.json, DONE/FAILED marker.

Invariants (fail-closed):

* critic must be frozen (RuntimeError if not)
* context_budget = 1 for transfer methods
* forward-only: probe evidence never affects current task
* self-transfer excluded
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import os
import subprocess
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
from smtr.rima.continual_transfer_learner import ContinualTransferLearner  # noqa: E402
from smtr.rima.experiment_config import (  # noqa: E402
    ALL_METHOD_VARIANTS,
    MethodVariant,
    get_method_variant,
)
from smtr.rima.features import (  # noqa: E402
    ReceiverConditionedTransferFeatures,
    RimaFeatureEncoder,
)
from smtr.rima.metrics import compute_cost_report, summarize_rima_run  # noqa: E402
from smtr.rima.online_transfer_evidence import OnlineTransferEvidence  # noqa: E402
from smtr.rima.outcome import RimaOutcomeEvaluator  # noqa: E402
from smtr.rima.post_task_probe import (  # noqa: E402
    PostTaskTransferProbe,
    select_probe_candidate,
)
from smtr.rima.receiver_topology import (  # noqa: E402
    ReceiverExclusionPolicy,
    select_receivers,
)
from smtr.rima.transfer_controller import (  # noqa: E402
    EpisodeTransferDecision,
    TransferAwareMemoryController,
    select_episode_edge,
)
from smtr.rima.transfer_metrics import (  # noqa: E402
    build_curve_records,
    compute_transfer_cost,
    compute_transfer_routing_metrics,
)
from smtr.rima.transfer_policy import TransferPolicy  # noqa: E402
from smtr.rima.transfer_state import ReceiverTransferStateContainer  # noqa: E402
from smtr.router.official_score_transfer_critic import (  # noqa: E402
    BootstrapOfficialScoreTransferCritic,
    MatchedInterventionExample,
    OfficialScoreTransferCritic,
)
from experiments.rima.run_continual_main import (  # noqa: E402
    _task_repr,
    _receiver_repr,
    extract_to_shared_memories,
)
from experiments.rima.train_critic import (  # noqa: E402
    load_records,
    record_to_example,
)

logger = logging.getLogger("rima.continual_transfer")

METHODS = (
    "no_memory", "retrieval", "rima_receiver", "rima_transfer",
)
ALL_SCENARIOS = ("bargaining", "coding", "database", "minecraft", "research")


# ---------------------------------------------------------------------------
# MarbleEpisodeRunner — EpisodeRunner protocol adapter
# ---------------------------------------------------------------------------


class MarbleEpisodeRunner:
    """Adapts :class:`TrajectoryCollector` to the ``EpisodeRunner`` protocol.

    Used by :class:`PostTaskTransferProbe` to run matched expose/withhold
    episodes for causal probing.
    """

    def __init__(
        self,
        collector: TrajectoryCollector,
        evaluator: RimaOutcomeEvaluator,
        method: str = "no_memory",
    ) -> None:
        self._collector = collector
        self._evaluator = evaluator
        self._method = method

    def run_episode(
        self,
        *,
        task: Any,
        receiver_id: str,
        memory_id: str | None,
        generation_seed: int,
    ) -> float:
        """Run one MARBLE episode and return the official task score."""
        payloads: dict[str, list[str]] | None = None
        if memory_id is not None:
            mem = self._pool_lookup(memory_id)
            if mem is not None:
                payloads = {receiver_id: [mem.procedure_payload]}
        mt = self._current_marble_task
        if mt is None:
            raise RuntimeError("MarbleTask not set on episode runner")
        traj = self._collector.collect(
            mt,
            seed=generation_seed,
            method=self._method,
            receiver_memory_payloads=payloads,
        )
        # ``task`` arrives as a raw task dict (see PostTaskTransferProbe
        # protocol) — pass it straight to the evaluator.
        outcome = self._evaluator.evaluate(
            task=task, run_result=traj.raw_output,
        )
        return outcome.task_score

    # Pool and task references set by the protocol before each probe session.
    pool: SharedMemoryPool | None = None
    _current_marble_task: MarbleTask | None = None

    def _pool_lookup(self, memory_id: str) -> SharedMemory | None:
        if self.pool is None:
            return None
        return self.pool.get(memory_id)


# ---------------------------------------------------------------------------
# Helpers
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


def _build_run_id(
    scenario: str, stream_seed: int, exec_seed: int, method: str,
) -> str:
    return f"{scenario}__stream{stream_seed}__exec{exec_seed}__method{method}"


def _git_info() -> dict[str, str]:
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        rev = "unknown"
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        dirty = ""
    return {"git_sha": rev, "git_dirty": dirty}


def _evidence_to_dict(ev: OnlineTransferEvidence) -> dict[str, Any]:
    return {
        "task_id": ev.task_id, "task_position": ev.task_position,
        "receiver_id": ev.receiver_id, "memory_id": ev.memory_id,
        "expose_scores": ev.expose_scores,
        "withhold_scores": ev.withhold_scores,
        "observed_tau": ev.observed_tau, "tau_std": ev.tau_std,
        "generation_seeds": ev.generation_seeds,
    }


def _decision_to_dict(d: EpisodeTransferDecision) -> dict[str, Any]:
    return {
        "task_id": d.task_id,
        "selected_receiver_id": d.selected_receiver_id,
        "selected_memory_id": d.selected_memory_id,
        "mu_tau": d.mu_tau, "sigma_tau": d.sigma_tau,
        "lcb": d.lcb, "source": d.source,
    }


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Transfer continual protocol
# ---------------------------------------------------------------------------


class TransferContinualProtocol:
    """Continual evaluation run for transfer methods and baselines.

    Supports legacy methods (``no_memory``, ``retrieval``, ``rima_receiver``,
    ``rima_transfer``) and :class:`MethodVariant`-controlled runs with
    causal probing, online critic refit, and per-run JSONL output.
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
        # MethodVariant extension
        method_variant: MethodVariant | None = None,
        stream_seed: int = 0,
        execution_seed: int | None = None,
        probe_seeds: list[int] | None = None,
        base_examples: list[MatchedInterventionExample] | None = None,
        source_agent_ids: dict[str, str] | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self.scenario = scenario
        self.method = method
        self.variant = method_variant
        self.stream_seed = stream_seed
        self.execution_seed = execution_seed if execution_seed is not None else seed
        self.seed = self.execution_seed
        self.tasks = tasks
        self.collector = collector
        self.extractor = extractor
        self.receiver_count = receiver_count
        self.retrieval_top_k = retrieval_top_k
        self.context_budget = context_budget
        self.outcome_evaluator = RimaOutcomeEvaluator(scenario=scenario)

        # Effective policy (may be overridden by gamma_mode).
        self._policy = transfer_policy
        if self.variant and self.variant.gamma_mode == "positive_stop" and transfer_policy:
            self._policy = dataclasses.replace(
                transfer_policy, gamma=transfer_policy.delta,
            )

        self.pool = SharedMemoryPool()
        self.knowledge = ReceiverKnowledgeContainer()
        self.transfer_states = ReceiverTransferStateContainer()
        self._current_task: MarbleTask | None = None

        # --- Controller / engine setup ---
        self.controller: TransferAwareMemoryController | None = None
        self.engine: RimaAdmissionEngine | None = None
        self._use_transfer_state = False
        self._use_controller = False

        if self.variant:
            self._use_transfer_state = self.variant.use_transfer_state
            need_ctrl = self.variant.use_transfer_state or self.variant.conditional_global_retrieval
            if need_ctrl:
                self._init_controller(critic_receiver)
                self._use_controller = True
            elif critic_receiver and self.method in ("rima_receiver",):
                self._init_engine(critic_receiver)
        elif method == "rima_transfer":
            self._init_controller(critic_receiver)
            self._use_controller = True
            self._use_transfer_state = True
        elif method == "rima_receiver":
            self._init_engine(critic_receiver)

        # --- Continual learner (§15) ---
        self.learner: ContinualTransferLearner | None = None
        if self.variant and self.variant.use_critic_update:
            encoder = RimaFeatureEncoder()
            self.learner = ContinualTransferLearner(
                base_examples=list(base_examples or []),
                encoder=encoder,
                source_agent_ids=dict(source_agent_ids or {}),
            )

        # --- Post-task probe (§14) ---
        self.probe: PostTaskTransferProbe | None = None
        self._episode_runner: MarbleEpisodeRunner | None = None
        if self.variant and self.variant.use_causal_probe:
            self._episode_runner = MarbleEpisodeRunner(
                collector, self.outcome_evaluator, method="no_memory",
            )
            self.probe = PostTaskTransferProbe(
                self._episode_runner,
                generation_seeds=probe_seeds or [0],
            )

        # --- Data stores ---
        self.records: list[dict[str, Any]] = []
        self.routing_diagnostics: list[dict[str, Any]] = []
        self.routing_events: list[dict[str, Any]] = []
        self.probe_events: list[dict[str, Any]] = []
        self.critic_version_log: list[dict[str, Any]] = []
        self._last_episode_decision: EpisodeTransferDecision | None = None
        self._last_receiver_plans: dict[str, Any] = {}

        # Run output directory (set by _init_run_dir or externally).
        self._run_dir = run_dir

        # Single-edge invariant for transfer variants.
        if self._use_transfer_state and context_budget != 1:
            raise ValueError("single-memory transfer requires context_budget=1")

    # -- init helpers --------------------------------------------------

    def _init_controller(
        self,
        critic: BootstrapOfficialScoreTransferCritic | None,
    ) -> None:
        if critic is None:
            raise RuntimeError("Transfer method requires --critic-receiver.")
        if self._policy is None:
            raise RuntimeError("Transfer method requires --transfer-policy.")
        if not critic.is_frozen:
            raise RuntimeError("Critic must be frozen before continual run.")
        self.controller = TransferAwareMemoryController(
            critic=critic, pool=self.pool,
            transfer_states=self.transfer_states,
            policy=self._policy,
            feature_builder=self._transfer_feature_builder,
            known_probe_top_k=20, global_explore_top_k=5,
            context_budget=1,
        )

    def _init_engine(
        self, critic: OfficialScoreTransferCritic | None,
    ) -> None:
        if critic is None:
            raise RuntimeError(f"{self.method} requires --critic-receiver.")
        if not critic.is_frozen:
            critic.freeze()
        self.engine = RimaAdmissionEngine(
            critic=critic, pool=self.pool,
            feature_builder=self._legacy_feature_builder,
            retrieval_top_k=self.retrieval_top_k,
        )
        self.engine.require_frozen()

    # -- feature builders ----------------------------------------------

    def _transfer_feature_builder(
        self, memory: SharedMemory, receiver_id: str,
        task: dict[str, Any], task_id: str,
    ) -> ReceiverConditionedTransferFeatures:
        t = self._current_task
        assert t is not None
        return ReceiverConditionedTransferFeatures(
            task_id=task_id, memory_id=memory.memory_id,
            receiver_id=receiver_id,
            task_repr=_task_repr(t),
            receiver_repr=_receiver_repr(t, receiver_id),
            routing_card=dict(memory.routing_card),
        )

    def _legacy_feature_builder(
        self, memory: SharedMemory, receiver_id: str,
        task: dict[str, Any],
    ) -> ReceiverConditionedTransferFeatures:
        t = self._current_task
        assert t is not None
        return ReceiverConditionedTransferFeatures(
            task_id=t.task_id, memory_id=memory.memory_id,
            receiver_id=receiver_id,
            task_repr=_task_repr(t),
            receiver_repr=_receiver_repr(t, receiver_id),
            routing_card=dict(memory.routing_card),
        )

    def _build_probe_features(
        self, task: MarbleTask, receiver_id: str, memory_id: str,
    ) -> ReceiverConditionedTransferFeatures:
        mem = self.pool.get(memory_id)
        assert mem is not None, f"probe target {memory_id} not in pool"
        return ReceiverConditionedTransferFeatures(
            task_id=task.task_id, memory_id=memory_id,
            receiver_id=receiver_id,
            task_repr=_task_repr(task),
            receiver_repr=_receiver_repr(task, receiver_id),
            routing_card=dict(mem.routing_card),
        )

    # -- run directory setup -------------------------------------------

    def _init_run_dir(self, base_dir: Path) -> Path:
        run_id = _build_run_id(
            self.scenario, self.stream_seed,
            self.execution_seed, self.method,
        )
        run_dir = base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir
        manifest = {
            "run_id": run_id, "scenario": self.scenario,
            "stream_seed": self.stream_seed,
            "execution_seed": self.execution_seed,
            "method": self.method,
            "method_variant": self.variant.method_id if self.variant else None,
            "n_tasks": len(self.tasks),
            **_git_info(),
        }
        with open(run_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return run_dir

    # -- main loop -----------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run all tasks and return summary."""
        for position, task in enumerate(self.tasks):
            self._current_task = task
            self._run_task(task, position)
        return self._summarize()

    def _run_task(self, task: MarbleTask, position: int) -> None:
        ev: dict[str, float] = {}  # event timestamps

        # ---- 1. Routing ----
        ev["routing_started"] = time.time()
        assignments = select_receivers(
            task={**task.raw_task, "agent_ids": task.get_agent_ids()},
            task_id=task.task_id, receiver_count=self.receiver_count,
            exclusion_policy=ReceiverExclusionPolicy(),
        )
        receiver_ids = [a.receiver_id for a in assignments]
        for rid in receiver_ids:
            self.knowledge.ensure(rid)

        payloads: dict[str, list[str]] = {rid: [] for rid in receiver_ids}
        task_routing_diags: list[dict[str, Any]] = []

        if self.variant and self._use_controller:
            self._route_variant(
                task, position, receiver_ids, payloads, task_routing_diags,
            )
        elif self.method == "no_memory":
            pass
        elif self.method == "retrieval":
            for rid in receiver_ids:
                cands = self.pool.retrieve(
                    _task_repr(task), rid, self.retrieval_top_k,
                    current_task_position=position,
                )
                payloads[rid] = [m.procedure_payload for m in cands]
        elif self.method == "rima_receiver":
            assert self.engine is not None
            for rid in receiver_ids:
                self.engine.admit_for_task(
                    task=_task_repr(task), task_id=task.task_id,
                    task_position=position, receiver_id=rid,
                    knowledge=self.knowledge.get(rid),
                )
                admitted = self.knowledge.get(rid).retrieve(
                    {}, self.context_budget,
                )
                payloads[rid] = [m.procedure_payload for m in admitted]
        elif self.method == "rima_transfer":
            self._route_legacy_transfer(
                task, position, receiver_ids, payloads, task_routing_diags,
            )
        else:
            raise ValueError(f"Unknown method: {self.method!r}")
        ev["routing_finished"] = time.time()

        # ---- 2. Sanitize ----
        for rid, plist in payloads.items():
            for p in plist:
                assert_clean_payload(p, memory_id=f"{task.task_id}/{rid}")

        # ---- 3. Scored execution ----
        ev["scored_execution_started"] = time.time()
        trajectory = self.collector.collect(
            task, seed=self.seed, method=self.method,
            memory_payloads=None,
            receiver_memory_payloads=(
                payloads if any(payloads.values()) else None
            ),
        )
        outcome = self.outcome_evaluator.evaluate(
            task=task.raw_task, run_result=trajectory.raw_output,
        )
        ev["scored_execution_finished"] = time.time()
        ev["task_score_frozen"] = time.time()

        # ---- 4. Record ----
        record = self._build_record(
            task, position, receiver_ids, payloads,
            trajectory, outcome, ev, task_routing_diags,
        )
        self.records.append(record)
        self.routing_diagnostics.extend(task_routing_diags)
        if self._run_dir:
            _write_jsonl(self._run_dir / "tasks.jsonl", record)
            for diag in task_routing_diags:
                _write_jsonl(self._run_dir / "routing.jsonl", diag)

        # ---- 5. Post-task probe (§14) ----
        if self.probe and self._episode_runner:
            ev["post_task_probe_started"] = time.time()
            self._run_probe(task, position, receiver_ids)
            ev["post_task_probe_finished"] = time.time()

        # ---- 6. Critic refit (§15) ----
        if self.learner and self.learner.should_refit():
            ev["critic_refit_started"] = time.time()
            self.learner.maybe_refit()
            ev["critic_refit_finished"] = time.time()
            vlog = {
                "task_position": position,
                "critic_version": self.learner.critic_version,
                "trained_through": (
                    self.learner.critic_trained_through_task_position
                ),
                "n_online": self.learner.n_online_examples,
            }
            self.critic_version_log.append(vlog)
            if self._run_dir:
                _write_jsonl(self._run_dir / "critic_versions.jsonl", vlog)

        # ---- 7. Extract memories → M_{t+1} ----
        ev["memory_extraction_started"] = time.time()
        new_memories = extract_to_shared_memories(
            trajectory, extractor=self.extractor,
            task=task, task_position=position,
        )
        for m in new_memories:
            if m.memory_id not in self.pool:
                self.pool.add(m)
        ev["memory_extraction_finished"] = time.time()

    # -- routing: MethodVariant ----------------------------------------

    def _route_variant(
        self, task: MarbleTask, position: int,
        receiver_ids: list[str],
        payloads: dict[str, list[str]],
        task_routing_diags: list[dict[str, Any]],
    ) -> None:
        assert self.controller is not None
        assert self.variant is not None
        receiver_plans: dict[str, Any] = {}
        for rid in receiver_ids:
            plan = self.controller.plan_for_task(
                task=_task_repr(task), task_id=task.task_id,
                task_position=position, receiver_id=rid,
            )
            # No-uncertainty: override LCB/UCB with mu (beta*sigma := 0).
            if not self.variant.use_uncertainty:
                for c in plan.known_candidates + plan.global_candidates:
                    if c.mu_tau is not None:
                        c.lcb = c.mu_tau
                        c.ucb = c.mu_tau
                if plan.best_known_lcb is not None:
                    best_mu = max(
                        (c.mu_tau for c in plan.known_candidates
                         if c.mu_tau is not None),
                        default=plan.best_known_lcb,
                    )
                    plan.best_known_lcb = best_mu
            receiver_plans[rid] = plan
            task_routing_diags.append(
                self._build_routing_diagnostic(plan, rid, position),
            )
        self._last_receiver_plans = receiver_plans
        episode_decision = select_episode_edge(
            receiver_plans, delta=self.controller.policy.delta,
        )
        if not self.variant.use_uncertainty and episode_decision.mu_tau is not None:
            episode_decision.lcb = episode_decision.mu_tau
            episode_decision.ucb = episode_decision.mu_tau
        payloads.update({rid: [] for rid in receiver_ids})
        if episode_decision.selected_receiver_id is not None:
            sr = episode_decision.selected_receiver_id
            sm = episode_decision.selected_memory_id
            mem = self.pool.get(sm) if sm else None
            if mem is not None:
                payloads[sr] = [mem.procedure_payload]
        self._last_episode_decision = episode_decision
        # Single-edge invariant
        n_rx = sum(bool(payloads[r]) for r in payloads)
        n_mem = sum(len(payloads[r]) for r in payloads)
        assert n_rx <= 1, f"single-edge violation: {n_rx} receivers"
        assert n_mem <= 1, f"single-edge violation: {n_mem} memories"

    # -- routing: legacy rima_transfer ---------------------------------

    def _route_legacy_transfer(
        self, task: MarbleTask, position: int,
        receiver_ids: list[str],
        payloads: dict[str, list[str]],
        task_routing_diags: list[dict[str, Any]],
    ) -> None:
        assert self.controller is not None
        receiver_plans: dict[str, Any] = {}
        for rid in receiver_ids:
            plan = self.controller.plan_for_task(
                task=_task_repr(task), task_id=task.task_id,
                task_position=position, receiver_id=rid,
            )
            receiver_plans[rid] = plan
            task_routing_diags.append(
                self._build_routing_diagnostic(plan, rid, position),
            )
        self._last_receiver_plans = receiver_plans
        episode_decision = select_episode_edge(
            receiver_plans, delta=self.controller.policy.delta,
        )
        payloads.update({rid: [] for rid in receiver_ids})
        if episode_decision.selected_receiver_id is not None:
            sr = episode_decision.selected_receiver_id
            sm = episode_decision.selected_memory_id
            mem = self.pool.get(sm) if sm else None
            if mem is not None:
                payloads[sr] = [mem.procedure_payload]
        self._last_episode_decision = episode_decision

    # -- post-task probe -----------------------------------------------

    def _run_probe(
        self, task: MarbleTask, position: int,
        receiver_ids: list[str],
    ) -> None:
        assert self.probe is not None
        assert self._episode_runner is not None
        self._episode_runner.pool = self.pool
        self._episode_runner._current_marble_task = task
        probe_rid, probe_cand = select_probe_candidate(
            self._last_receiver_plans,
        )
        if probe_rid is None or probe_cand is None:
            return
        evidence = self.probe.collect(
            task=task.raw_task, task_id=task.task_id,
            task_position=position, receiver_id=probe_rid,
            memory_id=probe_cand.memory_id,
        )
        pe = _evidence_to_dict(evidence)
        self.probe_events.append(pe)
        if self._run_dir:
            _write_jsonl(self._run_dir / "probe_events.jsonl", pe)
        # Upgrade K_r evidence type: a matched probe yields CAUSAL_OBSERVED
        # (§13). The candidate is normally registered during planning, but
        # register defensively before recording.
        state = self.transfer_states.ensure(probe_rid)
        if not state.contains(probe_cand.memory_id):
            probe_mem = self.pool.get(probe_cand.memory_id)
            if probe_mem is not None:
                state.register_memory(
                    memory_id=probe_cand.memory_id,
                    source_agent_id=probe_mem.source_agent_id,
                    task_id=task.task_id,
                    task_position=position,
                )
        if state.contains(probe_cand.memory_id):
            state.record_causal_observation(
                memory_id=probe_cand.memory_id,
                task_id=task.task_id,
                observed_tau=evidence.observed_tau,
            )
        # Feed to learner.
        if self.learner:
            features = self._build_probe_features(
                task, probe_rid, probe_cand.memory_id,
            )
            src = None
            mem = self.pool.get(probe_cand.memory_id)
            if mem:
                src = mem.source_agent_id
            self.learner.add_online_evidence(
                evidence, features=features, source_agent_id=src,
            )



    # -- record builder ------------------------------------------------

    def _build_record(
        self, task: MarbleTask, position: int,
        receiver_ids: list[str],
        payloads: dict[str, list[str]],
        trajectory: Trajectory, outcome: Any,
        ev: dict[str, float],
        task_routing_diags: list[dict[str, Any]],
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "task_id": task.task_id,
            "task_position": position,
            "method": self.method,
            "scenario": self.scenario,
            "stream_seed": self.stream_seed,
            "execution_seed": self.execution_seed,
            "seed": self.seed,
            "receivers": receiver_ids,
            "task_score": outcome.task_score,
            "is_valid": outcome.is_valid,
            "failure_reason": outcome.failure_reason,
            "team_success_diagnostic": outcome.team_success,
            "n_injected_total": sum(len(v) for v in payloads.values()),
            "pool_size_at_t": len(self.pool.memories_before(position)),
            "engine_duration_seconds": trajectory.engine_duration_seconds,
            "wall_seconds": round(time.time() - ev.get("routing_started", 0), 2),
            "real_engine_executed": trajectory.real_engine_executed,
            "event_timestamps": {k: round(v, 4) for k, v in ev.items()},
        }
        ep_dec = getattr(self, "_last_episode_decision", None)
        if ep_dec is not None:
            record["episode_decision"] = _decision_to_dict(ep_dec)
        # Transfer single-edge invariant (legacy)
        if self.method == "rima_transfer" and not self.variant:
            n_rx = sum(bool(payloads[r]) for r in payloads)
            n_mem = sum(len(payloads[r]) for r in payloads)
            assert n_rx <= 1, f"single-edge violation: {n_rx} receivers"
            assert n_mem <= 1, f"single-edge violation: {n_mem} memories"
        if task_routing_diags:
            record["routing_diagnostics"] = task_routing_diags
        # Critic version at time of this task's routing.
        if self.learner:
            record["selection_critic_version"] = self.learner.critic_version
            record["critic_trained_through"] = (
                self.learner.critic_trained_through_task_position
            )
        return record

    # -- routing diagnostic builder (§33) ------------------------------

    def _build_routing_diagnostic(
        self, plan: Any, receiver_id: str, position: int,
    ) -> dict[str, Any]:
        state = self.transfer_states.get(receiver_id)
        best_known_mu: float | None = None
        best_known_sigma: float | None = None
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
        sel_mid: str | None = None
        sel_source = "none"
        sel_mu: float | None = None
        sel_sigma: float | None = None
        sel_lcb: float | None = None
        if plan.selected_memory_ids:
            sel_mid = plan.selected_memory_ids[0]
            for c in plan.known_candidates + plan.global_candidates:
                if c.memory_id == sel_mid and c.selected_for_context:
                    sel_source = c.candidate_source
                    sel_mu = c.mu_tau
                    sel_sigma = c.sigma_tau
                    sel_lcb = c.lcb
                    break
        policy = self._policy
        # Strict state semantics (§ Phase 6): len(state) includes
        # PREDICTED_ONLY registrations; causal state counts only edges
        # with CAUSAL_OBSERVED evidence.
        n_predicted = len(state.predicted_only_entries()) if state else 0
        n_causal = len(state.causal_observed_entries()) if state else 0
        return {
            "receiver_id": receiver_id,
            "task_position": position,
            "routing_mode": plan.routing_mode,
            "transfer_state_size_before": len(state) if state else 0,
            "transfer_state_size_after": len(state) if state else 0,
            "transfer_state_predicted_only_after": n_predicted,
            "transfer_state_causal_observed_after": n_causal,
            "n_known_candidates_considered": len(plan.known_candidates),
            "n_global_candidates_considered": len(plan.global_candidates),
            "global_retrieval_triggered": plan.global_retrieval_triggered,
            "best_known_mu": best_known_mu,
            "best_known_sigma": best_known_sigma,
            "best_known_lcb": plan.best_known_lcb,
            "beta": policy.beta if policy else None,
            "delta": policy.delta if policy else None,
            "gamma": policy.gamma if policy else None,
            "selected_memory_id": sel_mid,
            "selected_source": sel_source,
            "selected_mu": sel_mu,
            "selected_sigma": sel_sigma,
            "selected_lcb": sel_lcb,
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
            "stream_seed": self.stream_seed,
            "execution_seed": self.execution_seed,
            "method": self.method,
            "method_variant": (
                self.variant.method_id if self.variant else None
            ),
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
                decisions=(self.engine.decisions if self.engine else ()),
                pool_size=len(self.pool),
            ),
            "cost_report": compute_cost_report(self.records),
            "n_probe_events": len(self.probe_events),
            "n_critic_refits": len(self.critic_version_log),
        }
        if self.routing_diagnostics:
            summary["transfer_routing_metrics"] = (
                compute_transfer_routing_metrics(self.routing_diagnostics)
            )
            summary["transfer_cost"] = compute_transfer_cost(
                self.routing_diagnostics,
            )
            summary["curve_records"] = build_curve_records(
                self.routing_diagnostics, self.records,
            )
        if self.engine is not None:
            summary["admission_statistics"] = self.engine.stats.to_dict()
            summary["formal_decision_source"] = "frozen_transfer_critic"
            summary["critic_checkpoint_sha256"] = (
                self.engine.critic.checkpoint_sha256()
            )
        # Write output files if run_dir is set.
        if self._run_dir:
            with open(self._run_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            with open(self._run_dir / "costs.json", "w") as f:
                json.dump(summary.get("cost_report", {}), f, indent=2)
            (self._run_dir / "DONE").touch()
        return summary


# ---------------------------------------------------------------------------
# Base-examples builder for ContinualTransferLearner
# ---------------------------------------------------------------------------


def build_base_examples(
    records_path: str,
    source_agents_path: str | None = None,
) -> tuple[list[MatchedInterventionExample], dict[str, str]]:
    """Build base training examples from intervention records."""
    recs = load_records(Path(records_path))
    sa: dict[str, str] = {}
    if source_agents_path:
        with open(source_agents_path) as f:
            sa = json.load(f)
    examples = [record_to_example(r, source_agent_ids=sa) for r in recs]
    return examples, sa


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RIMA-Transfer continual runner (§27-37 + pilot)",
    )
    p.add_argument("--scenarios", nargs="+", default=["bargaining"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0],
                   help="Legacy seeds (execution seeds in variant mode).")
    p.add_argument("--methods", nargs="+",
                   default=["no_memory", "rima_transfer"],
                   help=f"Legacy methods: subset of {METHODS}")
    p.add_argument("--method-variant", type=str, default=None,
                   help=f"MethodVariant ID: {sorted(ALL_METHOD_VARIANTS)}")
    p.add_argument("--stream-seeds", nargs="+", type=int, default=None,
                   help="Task-order seeds (pilot mode).")
    p.add_argument("--execution-seeds", nargs="+", type=int, default=None,
                   help="MARBLE execution seeds (pilot mode).")
    p.add_argument("--probe-seeds", nargs="+", type=int, default=None,
                   help="Post-task probe generation seeds.")
    p.add_argument("--limit-per-scenario", type=int, default=20)
    p.add_argument("--n-tasks-per-stream", type=int, default=None,
                   help="Override task count per stream.")
    p.add_argument("--receiver-count", type=int, default=3)
    p.add_argument("--retrieval-top-k", type=int, default=5)
    p.add_argument("--context-budget", type=int, default=1)
    p.add_argument("--critic-receiver", type=str, default=None)
    p.add_argument("--transfer-policy", type=str, default=None)
    p.add_argument("--intervention-records", type=str, default=None,
                   help="Intervention records for learner base examples.")
    p.add_argument("--source-agents", type=str, default=None)
    p.add_argument("--known-probe-top-k", type=int, default=20)
    p.add_argument("--global-explore-top-k", type=int, default=5)
    p.add_argument("--engine-timeout", type=int, default=1800)
    p.add_argument("--output-dir", type=str,
                   default="results/rima_transfer/continual")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get(
        "OPENAI_API_KEY",
    )
    if not api_key:
        print("FATAL: set DASHSCOPE_API_KEY or OPENAI_API_KEY",
              file=sys.stderr)
        return 1

    # Load critic.
    critic_receiver = None
    if args.critic_receiver:
        critic_receiver = BootstrapOfficialScoreTransferCritic.load(
            args.critic_receiver,
        )

    # Load transfer policy.
    transfer_policy = None
    if args.transfer_policy:
        transfer_policy = load_transfer_policy(args.transfer_policy)

    # Load base examples for learner.
    base_examples: list[MatchedInterventionExample] = []
    source_agent_ids: dict[str, str] = {}
    if args.intervention_records:
        base_examples, source_agent_ids = build_base_examples(
            args.intervention_records, args.source_agents,
        )

    loader = MarbleTaskLoader()
    collector = TrajectoryCollector(engine_timeout=args.engine_timeout)
    extractor = ExperienceExtractor()
    out_dir = _PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []

    if args.method_variant:
        # ---- MethodVariant mode (pilot) ----
        variant = get_method_variant(args.method_variant)
        stream_seeds = args.stream_seeds or args.seeds
        exec_seeds = args.execution_seeds or args.seeds
        probe_seeds = args.probe_seeds or [0]

        for scenario in args.scenarios:
            tasks = loader.load_scenario(
                scenario,
                limit=args.n_tasks_per_stream or args.limit_per_scenario,
            )
            for ss in stream_seeds:
                for es in exec_seeds:
                    logger.info(
                        "variant=%s scenario=%s stream=%d exec=%d tasks=%d",
                        variant.method_id, scenario, ss, es, len(tasks),
                    )
                    proto = TransferContinualProtocol(
                        scenario=scenario, seed=es,
                        method=variant.method_id, tasks=tasks,
                        collector=collector, extractor=extractor,
                        receiver_count=args.receiver_count,
                        retrieval_top_k=args.retrieval_top_k,
                        context_budget=args.context_budget,
                        critic_receiver=critic_receiver,
                        transfer_policy=transfer_policy,
                        known_probe_top_k=args.known_probe_top_k,
                        global_explore_top_k=args.global_explore_top_k,
                        method_variant=variant,
                        stream_seed=ss, execution_seed=es,
                        probe_seeds=probe_seeds,
                        base_examples=base_examples,
                        source_agent_ids=source_agent_ids,
                    )
                    proto._init_run_dir(out_dir)
                    summary = proto.run()
                    all_summaries.append(summary)
    else:
        # ---- Legacy mode ----
        for method in args.methods:
            if method not in METHODS:
                print(f"FATAL: unknown method {method!r}", file=sys.stderr)
                return 1
        for scenario in args.scenarios:
            tasks = loader.load_scenario(
                scenario, limit=args.limit_per_scenario,
            )
            for seed in args.seeds:
                for method in args.methods:
                    logger.info(
                        "scenario=%s seed=%s method=%s tasks=%d",
                        scenario, seed, method, len(tasks),
                    )
                    proto = TransferContinualProtocol(
                        scenario=scenario, seed=seed, method=method,
                        tasks=tasks, collector=collector,
                        extractor=extractor,
                        receiver_count=args.receiver_count,
                        retrieval_top_k=args.retrieval_top_k,
                        context_budget=args.context_budget,
                        critic_receiver=critic_receiver,
                        transfer_policy=transfer_policy,
                        known_probe_top_k=args.known_probe_top_k,
                        global_explore_top_k=args.global_explore_top_k,
                    )
                    summary = proto.run()
                    all_summaries.append(summary)
                    fname = (
                        f"transfer_continual_{scenario}"
                        f"_seed{seed}_{method}.json"
                    )
                    with open(out_dir / fname, "w") as f:
                        json.dump({
                            "summary": summary,
                            "records": proto.records,
                            "routing_diagnostics": proto.routing_diagnostics,
                        }, f, indent=2)
                    logger.info("wrote %s", out_dir / fname)

    with open(out_dir / "transfer_continual_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
