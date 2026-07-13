"""Ablation comparison experiment runner.

The runner separates base episodes from traversal repetitions, runs B0 before
all transfer labels are computed, and persists only complete run records.
"""

import copy
import json
import time
import traceback
from typing import Any
from uuid import uuid4

from smtr.config import RuntimeConfig
from smtr.counterfactual.decision_points import canonical_digest
from smtr.counterfactual.snapshot import ReadOnlyPinnedMemoryView
from smtr.counterfactual.task_provider import CounterfactualToyTaskProvider
from smtr.evaluation.ablation_gates import EffectOnlyGate
from smtr.experiment.schemas import (
    VALID_METHOD_IDS,
    BaseEpisodeManifestRecord,
    ComparisonRunRecord,
    DecisionRecord,
    ExperimentConfig,
    ExperimentSummary,
    RoutingInvocationRecord,
)
from smtr.experiment.summary import compute_summary, compute_transfer_label
from smtr.experiment.writer import ExperimentWriter
from smtr.memory.execution_evidence import selected_set_signature
from smtr.memory.seed_memories import seed_repository
from smtr.memory.store import SQLiteSharedMemoryRepository
from smtr.router.baseline_router import NoMemoryRouter
from smtr.router.baselines import (
    BudgetManifestConfig,
    BudgetMatchedTopKRouter,
    RelevanceTopKRouter,
    RelevanceTopKRouterConfig,
)
from smtr.router.candidate_proposer import DeterministicHybridCandidateProposer
from smtr.router.factory import build_router, build_smtr_router
from smtr.router.sequential_router import SequentialRouterConfig
from smtr.runtime.environment import ToyEnvironment
from smtr.runtime.graph import build_graph
from smtr.runtime.state import initial_state

FULL_ABLATION_METHODS = [
    "B0",
    "B1-Top1",
    "B1-Matched",
    "SMTR",
]
FORMAL_LEARNED_METHODS = frozenset({"SMTR", "EffectOnly-SMTR"})
TRAVERSAL_METHODS = FORMAL_LEARNED_METHODS
CRITIC_METHODS = FORMAL_LEARNED_METHODS


class ExperimentRunError(RuntimeError):
    """Raised when a runtime failure invalidates an experiment run."""


class ComparisonRunner:
    """Runs the SMTR ablation experiment with strict persistence semantics."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.experiment_id = str(uuid4())[:8]

    def _resolve_methods(self) -> list[str]:
        methods = list(self.config.methods or FULL_ABLATION_METHODS)
        unknown = [method for method in methods if method not in VALID_METHOD_IDS]
        if unknown:
            raise ValueError(f"unknown method IDs for new runner: {unknown}")
        learned_methods = [method for method in methods if method in CRITIC_METHODS]
        if learned_methods and not self.config.critic_checkpoint:
            raise ValueError("SMTR methods require critic_checkpoint")
        if "B1-Matched" in methods and not self.config.budget_manifest_path:
            raise ValueError("B1-Matched requires budget_manifest_path")
        return methods

    def _build_router_for_method(
        self,
        method: str,
        *,
        traversal_seed: int | None,
        base_episode_id: str,
    ) -> tuple[Any, str]:
        if method == "B0":
            return NoMemoryRouter(), "NoMemoryRouter"
        if method == "B1-Top1":
            return (
                RelevanceTopKRouter(
                    config=RelevanceTopKRouterConfig(max_shares_per_invocation=1)
                ),
                "RelevanceTopKRouter",
            )
        if method == "B1-Top3":
            return (
                RelevanceTopKRouter(
                    config=RelevanceTopKRouterConfig(max_shares_per_invocation=3)
                ),
                "RelevanceTopKRouter",
            )
        if method == "B1-Matched":
            manifest = self._load_budget_manifest()
            router = BudgetMatchedTopKRouter(
                manifest_config=manifest,
                invocation_seed=canonical_int_seed(
                    self.experiment_id, base_episode_id, traversal_seed
                ),
            )
            return router, "BudgetMatchedTopKRouter"
        if method == "SMTR":
            router = build_smtr_router(
                critic_checkpoint=self.config.critic_checkpoint,
                negative_risk_budget=self.config.negative_risk_budget,
                max_shares_per_invocation=self.config.max_shares_per_invocation,
                seed=traversal_seed or 0,
            )
            return router, "ProductionSequentialRouter"
        if method == "EffectOnly-SMTR":
            router = build_router(
                mode="learned",
                critic_checkpoint=self.config.critic_checkpoint,
                expected_feature_block="full",
                max_shares_per_invocation=self.config.max_shares_per_invocation,
                seed=traversal_seed or 0,
                critic_config=SequentialRouterConfig(),
                negative_risk_budget=self.config.negative_risk_budget,
            )
            router.gate = EffectOnlyGate()
            return router, "ProductionSequentialRouter"
        raise ValueError(f"unknown method: {method!r}")

    def _load_budget_manifest(self) -> BudgetManifestConfig:
        assert self.config.budget_manifest_path is not None
        with open(self.config.budget_manifest_path, encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("source_split") != "validation":
            raise ValueError("B1-Matched budget manifest must come from validation split")
        dist = data.get("count_distribution")
        if not dist:
            raise ValueError("B1-Matched budget manifest count_distribution is required")
        total = sum(float(value) for value in dist.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("B1-Matched budget distribution must sum to 1")
        max_shares = int(data.get("max_shares_per_invocation", -1))
        if max_shares < 0:
            raise ValueError("B1-Matched max_shares_per_invocation must be nonnegative")
        for raw_count in dist:
            count = int(raw_count)
            if count < 0 or count > max_shares:
                raise ValueError("B1-Matched budget count outside valid range")
        return BudgetManifestConfig(
            count_distribution={str(k): float(v) for k, v in dist.items()},
            max_shares=max_shares,
            seed=int(data.get("seed", 0)),
        )

    def _needs_traversal_seed(self, method: str) -> bool:
        return method in TRAVERSAL_METHODS

    def run(self) -> ExperimentSummary:
        writer = ExperimentWriter(self.config.output_dir, self.config.overwrite)
        writer.initialize()
        writer.write_config(self.config)

        repository = SQLiteSharedMemoryRepository(self.config.db_path)
        seed_repository(repository)
        task_provider: CounterfactualToyTaskProvider | None = None
        if self.config.scenario:
            task_provider = CounterfactualToyTaskProvider()
            task_provider.ensure_memories(repository)

        memory_snapshot = repository.create_read_snapshot()
        memory_snapshot_id = str(memory_snapshot.store_revision)
        memory_snapshot_digest = canonical_digest(
            [card.model_dump(mode="json") for card in repository.get_routing_cards()]
        )

        methods = self._resolve_methods()
        self._validate_checkpoints(methods)
        base_episodes = self._build_base_episode_manifest(
            task_provider=task_provider,
            memory_snapshot_id=memory_snapshot_id,
            memory_snapshot_digest=memory_snapshot_digest,
        )
        writer.write_base_episode_manifest(base_episodes)

        all_runs: list[ComparisonRunRecord] = []
        b0_outcomes: dict[str, bool] = {}

        for base_episode in base_episodes:
            try:
                b0_run = self._execute_and_record(
                    writer=writer,
                    method="B0",
                    base_episode=base_episode,
                    memory_snapshot=memory_snapshot,
                    repository=repository,
                    policy_level_transfer_label=None,
                )
            except ExperimentRunError as exc:
                _append_error_from_exception(writer, exc)
                raise
            all_runs.append(b0_run)
            b0_outcomes[base_episode.base_episode_id] = b0_run.team_success

        for base_episode in base_episodes:
            for method in methods:
                if method == "B0":
                    continue
                if base_episode.base_episode_id not in b0_outcomes:
                    raise ExperimentRunError(
                        f"missing B0 outcome for {base_episode.base_episode_id}"
                    )
                traversal_iter = (
                    self.config.traversal_seeds
                    if self._needs_traversal_seed(method)
                    else [None]
                )
                for traversal_seed in traversal_iter:
                    try:
                        raw_run = self._run_method(
                            method=method,
                            base_episode=base_episode,
                            memory_snapshot=memory_snapshot,
                            repository=repository,
                            traversal_seed=traversal_seed,
                        )
                    except ExperimentRunError as exc:
                        _append_error_from_exception(writer, exc)
                        raise
                    label = compute_transfer_label(
                        raw_run.team_success,
                        b0_outcomes[base_episode.base_episode_id],
                    )
                    completed = raw_run.model_copy(
                        update={"policy_level_transfer_label": label}
                    )
                    writer.append_run(completed)
                    all_runs.append(completed)

        self._assert_roundtrip(writer, all_runs)
        writer.write_invocations(all_runs)
        summary = compute_summary(all_runs, self.config)
        writer.write_summary(summary)
        return summary

    def _validate_checkpoints(self, methods: list[str]) -> None:
        if any(method in CRITIC_METHODS for method in methods):
            build_smtr_router(
                critic_checkpoint=self.config.critic_checkpoint,
                negative_risk_budget=self.config.negative_risk_budget,
                max_shares_per_invocation=self.config.max_shares_per_invocation,
                seed=0,
            )

    def _build_base_episode_manifest(
        self,
        *,
        task_provider: CounterfactualToyTaskProvider | None,
        memory_snapshot_id: str,
        memory_snapshot_digest: str,
    ) -> list[BaseEpisodeManifestRecord]:
        records: list[BaseEpisodeManifestRecord] = []
        for task_seed in self.config.task_seeds:
            for generation_seed in self.config.generation_seeds:
                for replicate_index in range(self.config.scenario_replicates):
                    task_description, env_snapshot = self._task_for_seed(
                        task_seed=task_seed,
                        task_provider=task_provider,
                    )
                    task_spec_digest = canonical_digest(
                        {
                            "task": task_description,
                            "environment_observation": env_snapshot,
                        }
                    )
                    initial_graph_state = initial_state(
                        task=task_description,
                        environment_observation=copy.deepcopy(env_snapshot),
                        run_seed=generation_seed,
                        episode_id="manifest",
                        task_id="manifest",
                        top_k=self.config.top_k,
                    )
                    base_episode_id = canonical_digest(
                        {
                            "scenario": self.config.scenario,
                            "task_spec_digest": task_spec_digest,
                            "task_seed": task_seed,
                            "generation_seed": generation_seed,
                            "replicate_index": replicate_index,
                        }
                    )[:16]
                    records.append(
                        BaseEpisodeManifestRecord(
                            base_episode_id=base_episode_id,
                            scenario=self.config.scenario,
                            task_seed=task_seed,
                            task_spec_digest=task_spec_digest,
                            generation_seed=generation_seed,
                            replicate_index=replicate_index,
                            initial_graph_state_digest=canonical_digest(
                                dict(initial_graph_state)
                            ),
                            initial_environment_digest=canonical_digest(env_snapshot),
                            memory_snapshot_id=memory_snapshot_id,
                            memory_snapshot_digest=memory_snapshot_digest,
                        )
                    )
        return records

    def _task_for_seed(
        self,
        *,
        task_seed: int,
        task_provider: CounterfactualToyTaskProvider | None,
    ) -> tuple[str, dict[str, Any]]:
        if task_provider is not None:
            if self.config.scenario is None:
                raise ValueError("scenario must be set when task_provider is used")
            task_spec = task_provider.generate(scenario=self.config.scenario, seed=task_seed)
            return task_spec.task, task_spec.environment_observation
        env = ToyEnvironment(seed=task_seed)
        return "Obtain a target artifact using the valid action sequence.", env.snapshot()

    def _execute_and_record(
        self,
        *,
        writer: ExperimentWriter,
        method: str,
        base_episode: BaseEpisodeManifestRecord,
        memory_snapshot: Any,
        repository: Any,
        policy_level_transfer_label: str | None,
    ) -> ComparisonRunRecord:
        run = self._run_method(
            method=method,
            base_episode=base_episode,
            memory_snapshot=memory_snapshot,
            repository=repository,
            traversal_seed=None,
        )
        run = run.model_copy(
            update={"policy_level_transfer_label": policy_level_transfer_label}
        )
        writer.append_run(run)
        return run

    def _run_method(
        self,
        *,
        method: str,
        base_episode: BaseEpisodeManifestRecord,
        memory_snapshot: Any,
        repository: Any,
        traversal_seed: int | None,
    ) -> ComparisonRunRecord:
        start_time = time.monotonic()
        router, router_name = self._build_router_for_method(
            method,
            traversal_seed=traversal_seed,
            base_episode_id=base_episode.base_episode_id,
        )
        task_description, env_snapshot = self._task_for_seed(
            task_seed=base_episode.task_seed,
            task_provider=CounterfactualToyTaskProvider() if self.config.scenario else None,
        )
        episode_id = base_episode.base_episode_id
        task_instance_id = base_episode.base_episode_id
        try:
            state = initial_state(
                task=task_description,
                environment_observation=copy.deepcopy(env_snapshot),
                run_seed=base_episode.generation_seed,
                episode_id=episode_id,
                task_id=task_instance_id,
                top_k=self.config.top_k,
            )
            memory_view = ReadOnlyPinnedMemoryView(
                repository=repository, snapshot=memory_snapshot
            )
            app = build_graph(
                memory_pool=memory_view,
                proposer=DeterministicHybridCandidateProposer(),
                router=router,
                config=RuntimeConfig(
                    seed=(
                        traversal_seed
                        if traversal_seed is not None
                        else base_episode.generation_seed
                    ),
                    top_k=self.config.top_k,
                ),
            )
            result = app.invoke(dict(state))
        except Exception as exc:
            error_record = {
                "experiment_id": self.experiment_id,
                "base_episode_id": base_episode.base_episode_id,
                "method": method,
                "seed": base_episode.generation_seed,
                "traversal_seed": traversal_seed,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            if self.config.fail_fast:
                raise ExperimentRunError(json.dumps(error_record, default=str)) from exc
            raise

        invocations = _build_invocations(result.get("router_trace", []))
        exposure_ids = [
            memory_id
            for invocation in invocations
            for memory_id in invocation.visible_payload_memory_ids
        ]
        unique_selected = sorted(set(exposure_ids))
        number_of_invocations = len(invocations)
        elapsed = time.monotonic() - start_time
        first_candidates = invocations[0].candidate_memory_ids if invocations else []
        legacy_trace = result.get("router_trace", [])
        return ComparisonRunRecord(
            experiment_id=self.experiment_id,
            base_episode_id=base_episode.base_episode_id,
            episode_id=episode_id,
            task_instance_id=task_instance_id,
            method=method,
            router_name=router_name,
            scenario=self.config.scenario,
            task_description=task_description,
            task_seed=base_episode.task_seed,
            environment_seed=base_episode.generation_seed,
            generation_seed=base_episode.generation_seed,
            replicate_index=base_episode.replicate_index,
            traversal_seed=traversal_seed,
            memory_snapshot_id=base_episode.memory_snapshot_id,
            memory_snapshot_digest=base_episode.memory_snapshot_digest,
            environment_snapshot_digest=base_episode.initial_environment_digest,
            team_success=bool(result.get("team_success", False)),
            runtime_seconds=elapsed,
            invocations=invocations,
            unique_selected_memory_ids=unique_selected,
            total_memory_exposures=len(exposure_ids),
            mean_selected_per_invocation=(
                len(exposure_ids) / number_of_invocations if number_of_invocations else 0.0
            ),
            number_of_invocations=number_of_invocations,
            all_withhold=len(exposure_ids) == 0,
            candidate_memory_ids=first_candidates,
            selected_memory_ids=unique_selected,
            selected_count=len(unique_selected),
            router_trace=legacy_trace,
        )

    def _assert_roundtrip(
        self,
        writer: ExperimentWriter,
        all_runs: list[ComparisonRunRecord],
    ) -> None:
        persisted = writer.load_runs()
        if len(persisted) != len(all_runs):
            raise ExperimentRunError("persisted run count differs from in-memory runs")
        for expected, actual in zip(all_runs, persisted, strict=True):
            keys = [
                "base_episode_id",
                "method",
                "team_success",
                "policy_level_transfer_label",
                "invocations",
                "total_memory_exposures",
            ]
            for key in keys:
                if getattr(expected, key) != getattr(actual, key):
                    raise ExperimentRunError(f"persisted run field mismatch: {key}")


def _build_invocations(router_trace: list[dict[str, Any]]) -> list[RoutingInvocationRecord]:
    invocations: list[RoutingInvocationRecord] = []
    for invocation_index, trace in enumerate(router_trace):
        candidates = trace.get("candidates", [])
        decisions = trace.get("decisions", [])
        selected_before: list[str] = []
        decision_records: list[DecisionRecord] = []
        for decision_index, decision in enumerate(decisions):
            traversal_position = decision.get("traversal_position")
            if traversal_position is None:
                traversal_position = decision.get("candidate_position", decision_index)
            if traversal_position is None:
                traversal_position = decision_index
            decision_records.append(
                DecisionRecord(
                    decision_index=decision_index,
                    memory_id=decision.get("memory_id", ""),
                    action=decision.get("action", "withhold"),
                    reason=decision.get("reason", ""),
                    proposal_rank=decision.get("proposal_rank"),
                    proposal_score=decision.get("proposal_score", decision.get("score")),
                    traversal_position=int(traversal_position),
                    selected_before_memory_ids=list(selected_before),
                    selected_before_digest=selected_set_signature(selected_before),
                    tau_mean=decision.get("tau_mean"),
                    tau_lcb=decision.get("tau_lcb"),
                    tau_ucb=decision.get("tau_ucb"),
                    negative_risk_mean=decision.get("negative_risk_mean"),
                    negative_risk_lcb=decision.get("negative_risk_lcb"),
                    negative_risk_ucb=decision.get("negative_risk_ucb"),
                    robust_diagnostics=decision.get("robust_diagnostics"),
                    support_distance=decision.get("support_distance"),
                    gate_name=decision.get("gate_name"),
                    effect_condition_passed=decision.get("effect_condition_passed"),
                    risk_condition_passed=decision.get("risk_condition_passed"),
                )
            )
            if decision.get("action") == "share":
                selected_before.append(decision.get("memory_id", ""))
        candidate_ids = [candidate.get("memory_id", "") for candidate in candidates]
        candidate_scores = [
            float(candidate.get("total_score", candidate.get("score", 0.0)))
            for candidate in candidates
        ]
        traversal_order = trace.get("traversal_order") or [
            record.memory_id for record in sorted(
                decision_records, key=lambda item: item.traversal_position
            )
        ]
        invocation_id = canonical_digest(
            {
                "index": invocation_index,
                "receiver_agent_id": trace.get("receiver_agent_id") or trace.get("agent", ""),
                "candidate_request_digest": trace.get("candidate_request_digest", ""),
            }
        )[:16]
        invocations.append(
            RoutingInvocationRecord(
                invocation_id=invocation_id,
                graph_node=trace.get("graph_node") or f"pre_route_{trace.get('agent', '')}",
                receiver_agent_id=trace.get("receiver_agent_id") or trace.get("agent", ""),
                receiver_role=trace.get("receiver_role") or trace.get("agent", ""),
                context_fingerprint_digest=trace.get("context_fingerprint_digest", ""),
                candidate_request_digest=trace.get("candidate_request_digest", ""),
                candidate_memory_ids=candidate_ids,
                candidate_scores=candidate_scores,
                proposal_order=candidate_ids,
                traversal_order=traversal_order,
                decisions=decision_records,
                selected_memory_ids=trace.get("selected_memory_ids", []),
                visible_payload_memory_ids=trace.get(
                    "visible_payload_memory_ids",
                    trace.get("selected_memory_ids", []),
                ),
            )
        )
    return invocations


def canonical_int_seed(*parts: Any) -> int:
    return int(canonical_digest(parts)[:8], 16)


def _append_error_from_exception(writer: ExperimentWriter, exc: ExperimentRunError) -> None:
    try:
        writer.append_error(json.loads(str(exc)))
    except json.JSONDecodeError:
        writer.append_error({"error": str(exc)})
