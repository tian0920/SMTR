"""Paired share/withhold branch runner for MARBLE database pilot.

Formal paired generation uses shared-control execution (清单
Shared-Control 第3章): one no-memory control per (task, receiver, seed)
group is paired with one candidate-specific share per treatment edge.
The legacy ``run_pair`` convenience API has been removed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.artifacts import assert_marble_artifact_path
from smtr.marble.engine_process import DEFAULT_ENGINE_TIMEOUT_SECONDS
from smtr.marble.environment.database_fingerprint import DatabaseLogicalFingerprint
from smtr.marble.environment.database_rebuild import (
    DatabaseCleanupResult,
    ParallelDatabaseRebuilder,
    SequentialDatabaseRebuilder,
)
from smtr.marble.environment.docker_slot_pool import DockerSlot, DockerSlotPool
from smtr.marble.environment.isolation import InitialStateBundle
from smtr.marble.environment.scenarios.database import MarbleDatabaseEnvironment
from smtr.marble.memory_injection import MarbleAgentInputAudit, MarbleMemoryInjector
from smtr.memory.render import render_procedure_payload
from smtr.marble.outcome.factory import evaluator_for_scenario
from smtr.marble.outcome.protocol import MarbleOutcome, outcome_from_failure
from smtr.marble.runtime_visibility_validator import (
    validate_runtime_visibility_from_path,
)


class MarbleBranchAudit(BaseModel):
    model_config = ConfigDict(frozen=True)

    branch_id: str
    workspace: str
    initial_digest: str
    initial_logical_fingerprint: dict[str, str] | None = None
    final_digest: str
    raw_result_digest: str
    input_audit: MarbleAgentInputAudit
    agent_config_digest: str
    generation_seed: int
    task_digest: str
    tool_config_digest: str
    outcome: MarbleOutcome
    real_engine_executed: bool = False
    cleanup_succeeded: bool = False
    cleanup_exit_code: int | None = None
    cleanup_failure_reason: str | None = None
    runtime_visibility_verified: bool = False
    runtime_visibility_invalid_reason: str | None = None


class PairedBranchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario: str
    task_id: str
    candidate_memory_id: str
    engine_name: str
    engine_version: str
    real_engine_executed: bool
    share: MarbleBranchAudit
    withhold: MarbleBranchAudit
    paired_record_valid: bool
    invalid_reason: str | None
    paired_label: str | None
    share_runtime_visibility_verified: bool = False
    withhold_runtime_visibility_verified: bool = False


class SharedControlResult(BaseModel):
    """One shared no-memory control execution for a (task, receiver, seed)
    group (清单 Shared-Control 第3章).

    The control carries no candidate-specific identity: no memory ID,
    writer, rank, score or candidate source appears in its metadata.
    """

    model_config = ConfigDict(frozen=True)

    control_group_id: str
    scenario: str
    task_id: str
    receiver_agent_id: str
    generation_seed: int
    engine_name: str
    engine_version: str
    audit: MarbleBranchAudit
    valid: bool
    invalid_reason: str | None = None
    forbidden_memory_ids: tuple[str, ...] = ()
    control_definition_version: str = "shared_no_memory_control_v1"


class MarblePairedBranchRunner:
    """Run paired memory interventions via shared-control execution.

    Formal paired generation uses ``run_no_memory_control`` +
    ``run_candidate_share`` + ``assemble_shared_control_pair``.
    The legacy ``run_pair`` API has been removed (清单 P0-2 第三章).

    Parameters
    ----------
    slot_pool:
        Optional ``DockerSlotPool`` for parallel execution.  When
        provided, each branch acquires a Docker compose slot with
        isolated host ports.  When ``None`` (default), the original
        sequential rebuilder is used.
    """

    def __init__(self, *, slot_pool: DockerSlotPool | None = None) -> None:
        self._slot_pool = slot_pool

    # ------------------------------------------------------------------
    # Shared-control execution (清单 Shared-Control 第3章)
    # ------------------------------------------------------------------

    def run_no_memory_control(
        self,
        *,
        control_group_id: str,
        task: dict[str, Any],
        initial_state_bundle: InitialStateBundle,
        agent_config: dict[str, Any],
        generation_seed: int,
        workspace: Path,
        forbidden_memory_ids: Sequence[str] = (),
        engine_timeout_seconds: int = DEFAULT_ENGINE_TIMEOUT_SECONDS,
    ) -> SharedControlResult:
        """Execute the one shared no-memory control of a control group.

        ``memory_injection`` is always None and the agent input carries
        no memory payloads. ``forbidden_memory_ids`` is the fixed set of
        every candidate memory in the group, computed before execution
        and never changed afterwards (清单 Shared-Control 第4章).
        """
        assert_marble_artifact_path(workspace)
        injector = MarbleMemoryInjector()
        receiver_agent_id = str(
            agent_config.get("target_receiver_agent_id", "agent1")
        )
        base_env = MarbleDatabaseEnvironment(
            task=task,
            workspace=workspace / "_base_input",
            initial_state_bundle=initial_state_bundle,
            agent_config=agent_config,
        )
        base_input = base_env.build_agent_input(memory_payloads=())
        base_env.close()
        control_input, control_input_audit = injector.build_agent_input(
            base_agent_input=base_input,
            memory_payloads=(),
            memory_ids=(),
        )
        # Control metadata must never carry candidate-specific identity:
        # no candidate_memory_id / writer_agent_id / rank / score / source.
        run_metadata = {
            "run_id": f"control_{control_group_id}",
            "task_id": initial_state_bundle.task_id,
            "scenario": initial_state_bundle.scenario,
            "method": "shared_no_memory_control",
            "branch": "withhold",
            "control_group_id": control_group_id,
            "receiver_agent_id": receiver_agent_id,
            "generation_seed": generation_seed,
        }
        audit, engine_name, engine_version = self._run_branch(
            branch_id="control",
            task=task,
            initial_state_bundle=initial_state_bundle,
            agent_config=agent_config,
            generation_seed=generation_seed,
            workspace=workspace,
            agent_input=control_input,
            input_audit=control_input_audit,
            memory_injection=None,
            run_metadata=run_metadata,
            receiver_agent_id=receiver_agent_id,
            visibility_method="shared_control",
            expected_memory_ids=(),
            forbidden_memory_ids=tuple(forbidden_memory_ids),
            engine_timeout_seconds=engine_timeout_seconds,
        )
        valid, invalid_reason = _validate_control(audit)
        return SharedControlResult(
            control_group_id=control_group_id,
            scenario=initial_state_bundle.scenario,
            task_id=initial_state_bundle.task_id,
            receiver_agent_id=receiver_agent_id,
            generation_seed=generation_seed,
            engine_name=engine_name,
            engine_version=engine_version,
            audit=audit,
            valid=valid,
            invalid_reason=invalid_reason,
            forbidden_memory_ids=tuple(forbidden_memory_ids),
        )

    def run_candidate_share(
        self,
        *,
        edge_id: str,
        task: dict[str, Any],
        candidate_memory: dict[str, Any],
        initial_state_bundle: InitialStateBundle,
        agent_config: dict[str, Any],
        generation_seed: int,
        workspace: Path,
        engine_timeout_seconds: int = DEFAULT_ENGINE_TIMEOUT_SECONDS,
    ) -> MarbleBranchAudit:
        """Execute one candidate-specific share branch.

        The payload is injected into the target receiver only. The share
        is paired with its group's shared control, never with its own
        private withhold branch (清单 Shared-Control 第1章).
        """
        assert_marble_artifact_path(workspace)
        injector = MarbleMemoryInjector()
        receiver_agent_id = str(
            agent_config.get("target_receiver_agent_id", "agent1")
        )
        base_env = MarbleDatabaseEnvironment(
            task=task,
            workspace=workspace / "_base_input",
            initial_state_bundle=initial_state_bundle,
            agent_config=agent_config,
        )
        base_input = base_env.build_agent_input(memory_payloads=())
        base_env.close()
        memory_payload = render_procedure_payload(candidate_memory)
        memory_id = str(candidate_memory.get("memory_id", "unknown"))
        share_injection: dict[str, Any] | None = None
        if memory_payload:
            share_injection = {
                "receiver_agent_ids": [receiver_agent_id],
                "memory_payloads": [memory_payload],
                "memory_ids": [memory_id],
                "intervention_id": f"share_{memory_id}_{generation_seed}",
            }
        share_input, share_input_audit = injector.build_agent_input(
            base_agent_input=base_input,
            memory_payloads=(memory_payload,),
            memory_ids=(memory_id,),
        )
        run_metadata = {
            "run_id": f"share_{edge_id}_{generation_seed}",
            "task_id": initial_state_bundle.task_id,
            "scenario": initial_state_bundle.scenario,
            "method": "shared_control_share",
            "branch": "share",
            "edge_id": edge_id,
            "receiver_agent_id": receiver_agent_id,
            "generation_seed": generation_seed,
        }
        audit, _engine_name, _engine_version = self._run_branch(
            branch_id="share",
            task=task,
            initial_state_bundle=initial_state_bundle,
            agent_config=agent_config,
            generation_seed=generation_seed,
            workspace=workspace,
            agent_input=share_input,
            input_audit=share_input_audit,
            memory_injection=share_injection,
            run_metadata=run_metadata,
            receiver_agent_id=receiver_agent_id,
            visibility_method="pair_share",
            expected_memory_ids=(memory_id,),
            forbidden_memory_ids=(),
            engine_timeout_seconds=engine_timeout_seconds,
        )
        return audit

    def assemble_shared_control_pair(
        self,
        *,
        control: SharedControlResult,
        share: MarbleBranchAudit,
        candidate_memory_id: str,
    ) -> PairedBranchResult:
        """Pair one share audit with its group's shared control.

        Reuses the standard pair validation and the unchanged
        four-outcome label mapping (清单 Shared-Control 第1/3章). An
        invalid control invalidates every record of its group.
        """
        real_engine_executed = (
            control.audit.real_engine_executed and share.real_engine_executed
        )
        valid, reason = _validate_pair(
            share=share,
            withhold=control.audit,
            real_engine_executed=real_engine_executed,
        )
        if not control.valid:
            valid = False
            reason = (
                f"shared_control_invalid:{control.invalid_reason}"
                if control.invalid_reason
                else "shared_control_invalid:unknown"
            )
        return PairedBranchResult(
            scenario=control.scenario,
            task_id=control.task_id,
            candidate_memory_id=candidate_memory_id,
            engine_name=control.engine_name,
            engine_version=control.engine_version,
            real_engine_executed=real_engine_executed,
            share=share,
            withhold=control.audit,
            paired_record_valid=valid,
            invalid_reason=reason,
            paired_label=(
                _paired_label(share.outcome.success, control.audit.outcome.success)
                if valid
                else None
            ),
            share_runtime_visibility_verified=share.runtime_visibility_verified,
            withhold_runtime_visibility_verified=(
                control.audit.runtime_visibility_verified
            ),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_branch(
        self,
        *,
        branch_id: str,
        task: dict[str, Any],
        initial_state_bundle: InitialStateBundle,
        agent_config: dict[str, Any],
        generation_seed: int,
        workspace: Path,
        agent_input: Any,
        input_audit: MarbleAgentInputAudit,
        memory_injection: dict[str, Any] | None,
        run_metadata: dict[str, Any],
        receiver_agent_id: str,
        visibility_method: str,
        expected_memory_ids: Sequence[str] = (),
        forbidden_memory_ids: Sequence[str] = (),
        engine_timeout_seconds: int = DEFAULT_ENGINE_TIMEOUT_SECONDS,
    ) -> tuple[MarbleBranchAudit, str, str]:
        """Materialize, execute, evaluate, clean up and audit one branch.

        ``expected_memory_ids`` (must be visible) and
        ``forbidden_memory_ids`` (must be completely invisible) are
        distinct semantics and are never encoded in one field
        (清单 Shared-Control 第3/4章).
        """
        evaluator = evaluator_for_scenario(initial_state_bundle.scenario)
        if self._slot_pool is not None:
            rebuilder = ParallelDatabaseRebuilder(self._slot_pool)
        else:
            rebuilder = SequentialDatabaseRebuilder()
        env = MarbleDatabaseEnvironment(
            task=task,
            workspace=workspace / branch_id,
            initial_state_bundle=initial_state_bundle,
            agent_config=agent_config,
        )
        engine_name = env.engine_name
        engine_version = env.engine_version
        try:
            fingerprint = rebuilder.materialize(
                initial_state_bundle=initial_state_bundle,
                branch_workspace=workspace / branch_id,
            )
            # Determine slot and API key for the engine subprocess
            docker_slot: DockerSlot | None = None
            api_key: str | None = None
            if isinstance(rebuilder, ParallelDatabaseRebuilder):
                docker_slot = rebuilder.slot
                if self._slot_pool is not None and docker_slot is not None:
                    api_key = self._slot_pool.get_api_key(docker_slot)
            try:
                run = env.run(
                    agent_input=agent_input,
                    generation_seed=generation_seed,
                    memory_injection=memory_injection,
                    engine_timeout_seconds=engine_timeout_seconds,
                    run_metadata=run_metadata,
                    docker_slot=docker_slot,
                    api_key=api_key,
                )
                outcome = evaluator.evaluate(task=task, run_result=run)
                branch_engine_executed = True
            except Exception as exc:
                run = {"branch": branch_id, "error": str(exc)}
                outcome = outcome_from_failure(
                    evaluator_name="marble_database_engine",
                    reason=str(exc),
                    raw_result=run,
                )
                branch_engine_executed = False
            cleanup_result = rebuilder.destroy(remove_workspace=False)
            audit = self._audit(
                branch_id=branch_id,
                env=env,
                raw_result=run,
                input_audit=input_audit,
                bundle=initial_state_bundle,
                generation_seed=generation_seed,
                outcome=outcome,
                initial_logical_fingerprint=fingerprint,
                real_engine_executed=branch_engine_executed,
                cleanup_result=cleanup_result,
                runtime_visibility_verified=False,
                runtime_visibility_invalid_reason="pending",
            )
            # Runtime visibility validation
            branch_audit_path = (
                workspace / branch_id / "memory_visibility_audit.jsonl"
            )
            rt_val = validate_runtime_visibility_from_path(
                method=visibility_method,
                branch=branch_id,
                receiver_agent_ids=[receiver_agent_id],
                expected_memory_ids=list(expected_memory_ids),
                audit_path=branch_audit_path,
                forbidden_memory_ids=list(forbidden_memory_ids),
            )
            audit = audit.model_copy(update={
                "runtime_visibility_verified": rt_val.visibility_verified,
                "runtime_visibility_invalid_reason": rt_val.invalid_reason,
            })
            return audit, engine_name, engine_version
        finally:
            env.close()
            rebuilder.destroy(remove_workspace=False)

    def _audit(
        self,
        *,
        branch_id: str,
        env: MarbleDatabaseEnvironment,
        raw_result: object,
        input_audit: MarbleAgentInputAudit,
        bundle: InitialStateBundle,
        generation_seed: int,
        outcome: MarbleOutcome,
        initial_logical_fingerprint: DatabaseLogicalFingerprint | None = None,
        real_engine_executed: bool = False,
        cleanup_result: DatabaseCleanupResult | None = None,
        runtime_visibility_verified: bool = False,
        runtime_visibility_invalid_reason: str | None = None,
    ) -> MarbleBranchAudit:
        cleanup = cleanup_result or DatabaseCleanupResult(
            exit_code=None,
            succeeded=False,
            failure_reason="cleanup_not_executed",
        )
        return MarbleBranchAudit(
            branch_id=branch_id,
            workspace=str(env.workspace),
            initial_digest=env.initial_state_digest(),
            initial_logical_fingerprint=(
                initial_logical_fingerprint.to_json()
                if initial_logical_fingerprint is not None
                else None
            ),
            final_digest=env.final_state_digest(),
            raw_result_digest=canonical_digest(raw_result),
            input_audit=input_audit,
            agent_config_digest=bundle.agent_config_digest,
            generation_seed=generation_seed,
            task_digest=bundle.task_digest,
            tool_config_digest=bundle.tool_config_digest,
            outcome=outcome,
            real_engine_executed=real_engine_executed,
            cleanup_succeeded=cleanup.succeeded,
            cleanup_exit_code=cleanup.exit_code,
            cleanup_failure_reason=cleanup.failure_reason,
            runtime_visibility_verified=runtime_visibility_verified,
            runtime_visibility_invalid_reason=runtime_visibility_invalid_reason,
        )


def _validate_control(audit: MarbleBranchAudit) -> tuple[bool, str | None]:
    """Fail-closed validity of one shared control execution (清单
    Shared-Control 第6章). Any single failure invalidates every paired
    record of the group; automatic withhold re-runs are forbidden."""
    if not audit.real_engine_executed:
        return False, "real_marble_engine_not_executed"
    if not audit.outcome.native_evaluator_executed:
        return False, "native_evaluator_not_executed"
    if not audit.outcome.environment_valid:
        return False, "environment_invalid"
    if audit.initial_logical_fingerprint is None:
        return False, "initial_state_audit_failed"
    if not audit.runtime_visibility_verified:
        reason = audit.runtime_visibility_invalid_reason or ""
        if "leaked" in reason:
            return False, "candidate_memory_leaked"
        return False, "runtime_visibility_not_verified"
    if not audit.cleanup_succeeded:
        return False, "cleanup_failed"
    return True, None


def _validate_pair(
    *,
    share: MarbleBranchAudit,
    withhold: MarbleBranchAudit,
    real_engine_executed: bool,
) -> tuple[bool, str | None]:
    checks = {
        "real_engine_executed": real_engine_executed,
        "share_real_engine_executed": share.real_engine_executed,
        "withhold_real_engine_executed": withhold.real_engine_executed,
        "share_native_evaluator_executed": share.outcome.native_evaluator_executed,
        "withhold_native_evaluator_executed": withhold.outcome.native_evaluator_executed,
        "share_cleanup_succeeded": share.cleanup_succeeded,
        "withhold_cleanup_succeeded": withhold.cleanup_succeeded,
        "initial_logical_digest": (
            share.initial_logical_fingerprint is not None
            and withhold.initial_logical_fingerprint is not None
            and share.initial_logical_fingerprint.get("combined_digest")
            == withhold.initial_logical_fingerprint.get("combined_digest")
        ),
        "initial_digest": share.initial_digest == withhold.initial_digest,
        "agent_config_digest": share.agent_config_digest == withhold.agent_config_digest,
        "generation_seed": share.generation_seed == withhold.generation_seed,
        "task_digest": share.task_digest == withhold.task_digest,
        "tool_config_digest": share.tool_config_digest == withhold.tool_config_digest,
        "workspace_paths_distinct": share.workspace != withhold.workspace,
        "environment_valid": share.outcome.environment_valid and withhold.outcome.environment_valid,
        "non_memory_input_sections_match": (
            share.input_audit.system_section_digest
            == withhold.input_audit.system_section_digest
            and share.input_audit.task_section_digest
            == withhold.input_audit.task_section_digest
            and share.input_audit.tool_section_digest
            == withhold.input_audit.tool_section_digest
        ),
        "share_memory_present": share.input_audit.contains_memory_section,
        "withhold_memory_absent": not withhold.input_audit.contains_memory_section,
        "share_runtime_visibility_verified": share.runtime_visibility_verified,
        "withhold_runtime_visibility_verified": withhold.runtime_visibility_verified,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        if "real_engine_executed" in failed:
            return False, "real_marble_engine_not_executed"
        return False, ",".join(failed)
    return True, None


def _paired_label(share_success: bool, withhold_success: bool) -> str:
    if share_success and not withhold_success:
        return "positive_transfer"
    if not share_success and withhold_success:
        return "negative_transfer"
    if share_success and withhold_success:
        return "neutral_success"
    return "neutral_failure"
