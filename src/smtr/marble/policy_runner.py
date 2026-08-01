"""MARBLE policy runner: execute real episodes with router-selected memory injection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from smtr.marble.end_to_end_evaluation import MarblePolicyRunResult
from smtr.memory.render import render_procedure_payload


class MarblePolicyRunner:
    """Execute real MARBLE episodes with memory injection for end-to-end evaluation."""

    def __init__(self, *, marble_root: Path) -> None:
        self.marble_root = marble_root

    def run_episode(
        self,
        *,
        method: str,
        task_entry: dict[str, Any],
        receiver_agent_id: str,
        receiver_role: str,
        candidate_memory_ids: list[str],
        selected_memory_ids: list[str],
        memory_pool: dict[str, dict],
        generation_seed: int,
        workspace: Path,
    ) -> MarblePolicyRunResult:
        """Run a single MARBLE episode with selected memory payloads injected.

        Only the target receiver sees the injected payloads.
        Uses MARBLE native evaluator for team success.
        """
        from smtr.marble.branch_runner import MarblePairedBranchRunner
        from smtr.marble.paired_context import build_pair_execution_context

        workspace.mkdir(parents=True, exist_ok=True)
        task_id = str(task_entry["task_id"])

        # Render selected memory payloads
        rendered_payloads: list[str] = []
        for mem_id in selected_memory_ids:
            mem_entry = memory_pool.get(mem_id)
            if mem_entry is not None:
                try:
                    rendered_payloads.append(render_procedure_payload(mem_entry))
                except (ValueError, KeyError):
                    pass

        # Build execution context
        try:
            context = build_pair_execution_context(
                marble_root=self.marble_root,
                task_entry=task_entry,
                receiver_agent_id=receiver_agent_id,
                workspace=workspace,
            )
        except Exception as exc:
            return MarblePolicyRunResult(
                method=method,
                task_id=task_id,
                generation_seed=generation_seed,
                receiver_agent_id=receiver_agent_id,
                receiver_role=receiver_role,
                candidate_memory_ids=tuple(candidate_memory_ids),
                selected_memory_ids=tuple(selected_memory_ids),
                team_success=False,
                score=None,
                real_engine_executed=False,
                native_evaluator_executed=False,
                environment_valid=False,
                runtime_visibility_verified=False,
                cleanup_succeeded=False,
                invalid_reason=f"context_build_failed: {exc}",
            )

        # Execute via branch runner (share branch only for policy run)
        try:
            from smtr.marble.environment.isolation import InitialStateBundle
            from smtr.marble.environment.database_rebuild import SequentialDatabaseRebuilder
            from smtr.marble.environment.scenarios.database import MarbleDatabaseEnvironment
            from smtr.marble.memory_injection import MarbleMemoryInjector
            from smtr.marble.outcome.factory import evaluator_for_scenario

            bundle = context.initial_state_bundle
            evaluator = evaluator_for_scenario(bundle.scenario)
            injector = MarbleMemoryInjector()
            rebuilder = SequentialDatabaseRebuilder()

            fingerprint = rebuilder.materialize(
                initial_state_bundle=bundle,
                branch_workspace=workspace / "run",
            )

            env = MarbleDatabaseEnvironment(
                task=context.task,
                workspace=workspace / "run",
                initial_state_bundle=bundle,
                agent_config=context.agent_config,
            )

            base_input = env.build_agent_input(memory_payloads=())
            agent_input, input_audit = injector.build_agent_input(
                base_agent_input=base_input,
                memory_payloads=tuple(rendered_payloads),
                memory_ids=tuple(selected_memory_ids),
            )

            injection = None
            if rendered_payloads:
                injection = {
                    "receiver_agent_ids": [receiver_agent_id],
                    "memory_payloads": rendered_payloads,
                    "memory_ids": selected_memory_ids,
                    "intervention_id": f"policy_{method}_{task_id}_{generation_seed}",
                }

            run_result = env.run(
                agent_input=agent_input,
                generation_seed=generation_seed,
                memory_injection=injection,
                engine_timeout_seconds=1800,
                run_metadata={
                    "run_id": f"policy_{method}_{task_id}_{generation_seed}",
                    "task_id": task_id,
                    "method": method,
                },
            )

            outcome = evaluator.evaluate(task=context.task, run_result=run_result)
            cleanup_result = rebuilder.destroy(remove_workspace=False)
            env.close()

            return MarblePolicyRunResult(
                method=method,
                task_id=task_id,
                generation_seed=generation_seed,
                receiver_agent_id=receiver_agent_id,
                receiver_role=receiver_role,
                candidate_memory_ids=tuple(candidate_memory_ids),
                selected_memory_ids=tuple(selected_memory_ids),
                team_success=outcome.success,
                score=outcome.score,
                real_engine_executed=True,
                native_evaluator_executed=outcome.native_evaluator_executed,
                environment_valid=outcome.environment_valid,
                runtime_visibility_verified=input_audit.contains_memory_section if rendered_payloads else True,
                cleanup_succeeded=cleanup_result.succeeded,
                invalid_reason=None,
            )
        except Exception as exc:
            return MarblePolicyRunResult(
                method=method,
                task_id=task_id,
                generation_seed=generation_seed,
                receiver_agent_id=receiver_agent_id,
                receiver_role=receiver_role,
                candidate_memory_ids=tuple(candidate_memory_ids),
                selected_memory_ids=tuple(selected_memory_ids),
                team_success=False,
                score=None,
                real_engine_executed=False,
                native_evaluator_executed=False,
                environment_valid=False,
                runtime_visibility_verified=False,
                cleanup_succeeded=False,
                invalid_reason=f"engine_error: {exc}",
            )
