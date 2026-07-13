"""Ablation comparison experiment runner.

Supports 6 methods: B0, B1-Top1, B1-Top3, B1-Matched, A1-NoSet, M0-Full.
Backward compatible with the legacy B0/B1/M0 triple.
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
from smtr.experiment.schemas import (
    ComparisonRunRecord,
    ExperimentConfig,
    ExperimentSummary,
)
from smtr.experiment.summary import compute_summary, compute_transfer_label
from smtr.experiment.writer import ExperimentWriter
from smtr.memory.seed_memories import seed_repository
from smtr.memory.store import SQLiteSharedMemoryRepository
from smtr.router.baseline_router import NoMemoryRouter
from smtr.router.baselines import (
    BudgetMatchedTopKRouter,
    BudgetManifestConfig,
    RelevanceTopKRouter,
    RelevanceTopKRouterConfig,
)
from smtr.router.candidate_proposer import DeterministicHybridCandidateProposer
from smtr.router.factory import build_router
from smtr.runtime.environment import ToyEnvironment
from smtr.runtime.graph import build_graph
from smtr.runtime.state import initial_state

# Default method sets
LEGACY_METHODS = ["B0", "B1", "M0"]
FULL_ABLATION_METHODS = ["B0", "B1-Top1", "B1-Top3", "B1-Matched", "A1-NoSet", "M0-Full"]

# Methods that use a critic (for rejection metrics)
CRITIC_METHODS = frozenset({"M0", "M0-Full", "A1-NoSet"})


class ComparisonRunner:
    """Runs ablation comparison experiment with strict fairness guarantees."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.experiment_id = str(uuid4())[:8]

    def _resolve_methods(self) -> list[str]:
        """Determine which methods to run."""
        if self.config.methods:
            return list(self.config.methods)
        return LEGACY_METHODS

    def _build_router_for_method(
        self, method: str, *, traversal_seed: int | None = None
    ) -> tuple[Any, str]:
        """Build the appropriate router for a given method.

        Returns (router, router_name).
        """
        if method == "B0":
            return NoMemoryRouter(), "NoMemoryRouter"

        if method == "B1":
            router = RelevanceTopKRouter(
                config=RelevanceTopKRouterConfig(
                    max_shares_per_invocation=self.config.max_shares_per_invocation,
                )
            )
            return router, "RelevanceTopKRouter"

        if method == "B1-Top1":
            router = RelevanceTopKRouter(
                config=RelevanceTopKRouterConfig(max_shares_per_invocation=1)
            )
            return router, "RelevanceTopKRouter"

        if method == "B1-Top3":
            router = RelevanceTopKRouter(
                config=RelevanceTopKRouterConfig(max_shares_per_invocation=3)
            )
            return router, "RelevanceTopKRouter"

        if method == "B1-Matched":
            manifest_data: dict[str, Any] = {}
            if self.config.budget_manifest_path:
                with open(self.config.budget_manifest_path) as f:
                    manifest_data = json.load(f)
            manifest_config = BudgetManifestConfig(
                count_distribution=manifest_data.get("count_distribution", {}),
                max_shares=manifest_data.get("max_shares_per_invocation", 3),
                seed=manifest_data.get("seed", 0),
            )
            router = BudgetMatchedTopKRouter(
                manifest_config=manifest_config,
                invocation_seed=traversal_seed or 0,
            )
            return router, "BudgetMatchedTopKRouter"

        if method == "M0":
            router = build_router(
                mode="learned",
                critic_checkpoint=self.config.critic_checkpoint,
                max_shares_per_invocation=self.config.max_shares_per_invocation,
                seed=traversal_seed or 0,
            )
            return router, "ProductionSequentialRouter"

        if method == "M0-Full":
            router = build_router(
                mode="learned",
                critic_checkpoint=self.config.critic_checkpoint,
                max_shares_per_invocation=self.config.max_shares_per_invocation,
                seed=traversal_seed or 0,
                feature_block="full",
            )
            return router, "ProductionSequentialRouter"

        if method == "A1-NoSet":
            a1_ckpt = self.config.a1_critic_checkpoint or self.config.critic_checkpoint
            router = build_router(
                mode="learned",
                critic_checkpoint=a1_ckpt,
                max_shares_per_invocation=self.config.max_shares_per_invocation,
                seed=traversal_seed or 0,
                feature_block="context_plus_candidate",
            )
            return router, "ProductionSequentialRouter"

        raise ValueError(f"unknown method: {method!r}")

    def _needs_traversal_seed(self, method: str) -> bool:
        """Whether a method needs per-traversal-seed runs (M0-like methods)."""
        return method in CRITIC_METHODS or method in ("M0",)

    def run(self) -> ExperimentSummary:
        """Execute the full comparison experiment and return summary."""
        writer = ExperimentWriter(self.config.output_dir, self.config.overwrite)
        writer.initialize()
        writer.write_config(self.config)

        # Set up memory repository once
        repository = SQLiteSharedMemoryRepository(self.config.db_path)
        seed_repository(repository)

        # If scenario mode, ensure counterfactual memories exist
        task_provider: CounterfactualToyTaskProvider | None = None
        if self.config.scenario:
            task_provider = CounterfactualToyTaskProvider()
            task_provider.ensure_memories(repository)

        memory_snapshot = repository.create_read_snapshot()
        memory_snapshot_id = str(memory_snapshot.store_revision)

        # Validate critic checkpoint is loadable
        if self.config.critic_checkpoint:
            from pathlib import Path

            from smtr.router.transfer_critic import FourOutcomeTransferCritic

            FourOutcomeTransferCritic.load(Path(self.config.critic_checkpoint))

        # Validate A1 checkpoint if different
        a1_ckpt = self.config.a1_critic_checkpoint
        if a1_ckpt and a1_ckpt != self.config.critic_checkpoint:
            from pathlib import Path

            from smtr.router.transfer_critic import FourOutcomeTransferCritic

            FourOutcomeTransferCritic.load(Path(a1_ckpt))

        methods = self._resolve_methods()
        all_runs: list[ComparisonRunRecord] = []

        for episode_idx in range(self.config.episodes):
            task_seed = self.config.task_seeds[
                episode_idx % len(self.config.task_seeds)
            ]
            episode_id = f"ep{episode_idx}_ts{task_seed}"
            task_instance_id = f"task_{episode_idx}"

            # Determine task description and environment snapshot
            task_description: str
            env_snapshot: dict[str, Any]
            if task_provider is not None:
                task_spec = task_provider.generate(
                    scenario=self.config.scenario, seed=task_seed
                )
                task_description = task_spec.task
                env_snapshot = task_spec.environment_observation
            else:
                task_description = (
                    "Obtain a target artifact using the valid action sequence."
                )
                env = ToyEnvironment(seed=task_seed)
                env_snapshot = env.snapshot()

            env_snapshot_digest = canonical_digest(env_snapshot)

            # Shared proposer
            proposer = DeterministicHybridCandidateProposer()

            # Collect B0 success for transfer label computation
            b0_success: bool | None = None

            for gen_seed in self.config.generation_seeds:
                for method in methods:
                    needs_trav = self._needs_traversal_seed(method)

                    if needs_trav:
                        traversal_iter = self.config.traversal_seeds
                    else:
                        traversal_iter = [None]

                    for trav_seed in traversal_iter:
                        router, router_name = self._build_router_for_method(
                            method, traversal_seed=trav_seed
                        )
                        run = self._run_method(
                            method=method,
                            router=router,
                            router_name=router_name,
                            env_snapshot=env_snapshot,
                            memory_snapshot=memory_snapshot,
                            repository=repository,
                            proposer=proposer,
                            generation_seed=gen_seed,
                            task_seed=task_seed,
                            episode_id=episode_id,
                            task_instance_id=task_instance_id,
                            memory_snapshot_id=memory_snapshot_id,
                            env_snapshot_digest=env_snapshot_digest,
                            traversal_seed=trav_seed,
                            task_description=task_description,
                            scenario=self.config.scenario,
                        )
                        writer.append_run(run)
                        all_runs.append(run)
                        self._flush_error(writer)

                        # Track B0 success for transfer labels
                        if method == "B0":
                            b0_success = run.team_success

                        # Compute transfer label for non-B0 methods
                        if method != "B0" and b0_success is not None:
                            label = compute_transfer_label(
                                run.team_success, b0_success
                            )
                            run = run.model_copy(
                                update={"policy_level_transfer_label": label}
                            )
                            all_runs[-1] = run

        # Compute and write summary
        summary = compute_summary(all_runs, self.config)
        writer.write_summary(summary)
        return summary

    def _run_method(
        self,
        *,
        method: str,
        router: Any,
        router_name: str,
        env_snapshot: dict[str, Any],
        memory_snapshot: Any,
        repository: Any,
        proposer: Any,
        generation_seed: int,
        task_seed: int,
        episode_id: str,
        task_instance_id: str,
        memory_snapshot_id: str,
        env_snapshot_digest: str,
        traversal_seed: int | None,
        task_description: str = "Obtain a target artifact using the valid action sequence.",
        scenario: str | None = None,
    ) -> ComparisonRunRecord:
        """Run a single method with full isolation."""
        start_time = time.monotonic()
        failure_reason = None

        try:
            # Deep copy state for isolation
            state = initial_state(
                task=task_description,
                environment_observation=copy.deepcopy(env_snapshot),
                run_seed=generation_seed,
                episode_id=episode_id,
                task_id=task_instance_id,
                top_k=self.config.top_k,
            )

            # Clone environment from snapshot (for determinism)
            ToyEnvironment.clone_from_snapshot(
                env_snapshot, seed=generation_seed
            )

            # Read-only memory view for isolation
            memory_view = ReadOnlyPinnedMemoryView(
                repository=repository, snapshot=memory_snapshot
            )

            # Build and run graph
            app = build_graph(
                memory_pool=memory_view,
                proposer=proposer,
                router=router,
                config=RuntimeConfig(
                    seed=generation_seed, top_k=self.config.top_k
                ),
            )
            result = app.invoke(dict(state))

            elapsed = time.monotonic() - start_time

            # Extract results
            candidate_ids: list[str] = []
            selected_ids_set: set[str] = set()
            router_trace: list[dict[str, Any]] = []

            for trace_entry in result.get("router_trace", []):
                candidates = trace_entry.get("candidates", [])
                decisions = trace_entry.get("decisions", [])
                selected = trace_entry.get("selected_memory_ids", [])

                # Collect candidate IDs (from first trace entry, they're the same)
                if not candidate_ids:
                    candidate_ids = [c["memory_id"] for c in candidates]

                selected_ids_set.update(selected)

                # Build method-specific trace
                trace_record: dict[str, Any] = {
                    "agent": trace_entry.get("agent", ""),
                    "router_name": trace_entry.get("router_name", router_name),
                    "selected_memory_ids": selected,
                    "decisions": [],
                }

                for dec in decisions:
                    dec_record: dict[str, Any] = {
                        "memory_id": dec.get("memory_id", ""),
                        "action": dec.get("action", ""),
                        "reason": dec.get("reason", ""),
                        "candidate_position": dec.get("candidate_position"),
                        "score": dec.get("score"),
                    }
                    # Critic-specific fields for M0-like methods
                    if method in CRITIC_METHODS:
                        dec_record.update({
                            "tau_mean": dec.get("tau_mean"),
                            "tau_lcb": dec.get("tau_lcb"),
                            "tau_ucb": dec.get("tau_ucb"),
                            "negative_risk_mean": dec.get("negative_risk_mean"),
                            "negative_risk_ucb": dec.get("negative_risk_ucb"),
                            "support_distance": dec.get("support_distance"),
                        })
                    # B1-specific fields
                    if method in ("B1-Top1", "B1-Top3", "B1-Matched"):
                        dec_record.update({
                            "proposal_rank": dec.get("proposal_rank"),
                            "proposal_score": dec.get("proposal_score"),
                        })
                    trace_record["decisions"].append(dec_record)

                router_trace.append(trace_record)

            return ComparisonRunRecord(
                experiment_id=self.experiment_id,
                episode_id=episode_id,
                task_instance_id=task_instance_id,
                method=method,
                router_name=router_name,
                scenario=scenario,
                task_description=task_description,
                task_seed=task_seed,
                environment_seed=generation_seed,
                generation_seed=generation_seed,
                traversal_seed=traversal_seed,
                memory_snapshot_id=memory_snapshot_id,
                environment_snapshot_digest=env_snapshot_digest,
                candidate_memory_ids=candidate_ids,
                selected_memory_ids=sorted(selected_ids_set),
                selected_count=len(selected_ids_set),
                team_success=bool(result.get("team_success", False)),
                failure_reason=failure_reason,
                runtime_seconds=elapsed,
                router_trace=router_trace,
            )

        except Exception as exc:
            elapsed = time.monotonic() - start_time
            failure_reason = f"{type(exc).__name__}: {exc}"
            error_record = {
                "experiment_id": self.experiment_id,
                "episode_id": episode_id,
                "task_instance_id": task_instance_id,
                "method": method,
                "traversal_seed": traversal_seed,
                "error": failure_reason,
                "traceback": traceback.format_exc(),
            }
            # Write error — we don't have the writer here, so we store it
            # in the record and let the caller write it
            self._last_error = error_record

            return ComparisonRunRecord(
                experiment_id=self.experiment_id,
                episode_id=episode_id,
                task_instance_id=task_instance_id,
                method=method,
                router_name=router_name,
                scenario=scenario,
                task_description=task_description,
                task_seed=task_seed,
                environment_seed=generation_seed,
                generation_seed=generation_seed,
                traversal_seed=traversal_seed,
                memory_snapshot_id=memory_snapshot_id,
                environment_snapshot_digest=env_snapshot_digest,
                candidate_memory_ids=[],
                selected_memory_ids=[],
                selected_count=0,
                team_success=False,
                failure_reason=failure_reason,
                runtime_seconds=elapsed,
                router_trace=[],
            )

    _last_error: dict | None = None

    def _flush_error(self, writer: ExperimentWriter) -> None:
        """Write any pending error record and clear it."""
        if self._last_error is not None:
            writer.append_error(self._last_error)
            self._last_error = None
