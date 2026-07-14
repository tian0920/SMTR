"""Paired share/withhold branch runner for MARBLE database pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.artifacts import assert_marble_artifact_path
from smtr.marble.environment.database_fingerprint import DatabaseLogicalFingerprint
from smtr.marble.environment.database_rebuild import SequentialDatabaseRebuilder
from smtr.marble.environment.isolation import InitialStateBundle
from smtr.marble.environment.scenarios.database import MarbleDatabaseEnvironment
from smtr.marble.memory_injection import MarbleAgentInputAudit, MarbleMemoryInjector
from smtr.marble.outcome.factory import evaluator_for_scenario
from smtr.marble.outcome.protocol import MarbleOutcome, outcome_from_failure


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
    branch_execution_order: str = "share_then_withhold"


class MarblePairedBranchRunner:
    """Run a paired memory intervention, invalidating pairs without real engine execution."""

    def run_pair(
        self,
        *,
        task: dict[str, Any],
        candidate_memory: dict[str, Any],
        initial_state_bundle: InitialStateBundle,
        agent_config: dict[str, Any],
        generation_seed: int,
        workspace: Path,
        branch_execution_order: Literal[
            "share_then_withhold",
            "withhold_then_share",
        ] = "share_then_withhold",
    ) -> PairedBranchResult:
        assert_marble_artifact_path(workspace)
        evaluator = evaluator_for_scenario(initial_state_bundle.scenario)
        injector = MarbleMemoryInjector()
        rebuilder = SequentialDatabaseRebuilder()
        share_env: MarbleDatabaseEnvironment | None = None
        withhold_env: MarbleDatabaseEnvironment | None = None
        try:
            base_env = MarbleDatabaseEnvironment(
                task=task,
                workspace=workspace / "_base_input",
                initial_state_bundle=initial_state_bundle,
                agent_config=agent_config,
            )
            base_input = base_env.build_agent_input(memory_payloads=())
            base_env.close()
            memory_payload = str(candidate_memory.get("payload", ""))
            memory_id = str(candidate_memory.get("memory_id", "unknown"))
            share_input, share_input_audit = injector.build_agent_input(
                base_agent_input=base_input,
                memory_payloads=(memory_payload,),
                memory_ids=(memory_id,),
            )
            withhold_input, withhold_input_audit = injector.build_agent_input(
                base_agent_input=base_input,
                memory_payloads=(),
                memory_ids=(),
            )
            branch_inputs = {
                "share": (share_input, share_input_audit),
                "withhold": (withhold_input, withhold_input_audit),
            }
            audits: dict[str, MarbleBranchAudit] = {}
            order = (
                ("share", "withhold")
                if branch_execution_order == "share_then_withhold"
                else ("withhold", "share")
            )
            for branch in order:
                fingerprint = rebuilder.materialize(
                    initial_state_bundle=initial_state_bundle,
                    branch_workspace=workspace / branch,
                )
                env = MarbleDatabaseEnvironment(
                    task=task,
                    workspace=workspace / branch,
                    initial_state_bundle=initial_state_bundle,
                    agent_config=agent_config,
                )
                if branch == "share":
                    share_env = env
                else:
                    withhold_env = env
                branch_input, branch_input_audit = branch_inputs[branch]
                try:
                    run = env.run(agent_input=branch_input, generation_seed=generation_seed)
                    outcome = evaluator.evaluate(task=task, run_result=run)
                    branch_engine_executed = True
                except Exception as exc:
                    run = {"branch": branch, "error": str(exc)}
                    outcome = outcome_from_failure(
                        evaluator_name="marble_database_engine",
                        reason=str(exc),
                        raw_result=run,
                    )
                    branch_engine_executed = False
                audits[branch] = self._audit(
                    branch_id=branch,
                    env=env,
                    raw_result=run,
                    input_audit=branch_input_audit,
                    bundle=initial_state_bundle,
                    generation_seed=generation_seed,
                    outcome=outcome,
                    initial_logical_fingerprint=fingerprint,
                    real_engine_executed=branch_engine_executed,
                )
                env.close()
                rebuilder.destroy()

            share = audits["share"]
            withhold = audits["withhold"]
            real_engine_executed = share.real_engine_executed and withhold.real_engine_executed
            valid, reason = _validate_pair(
                share=share,
                withhold=withhold,
                real_engine_executed=real_engine_executed,
            )
            result = PairedBranchResult(
                scenario=initial_state_bundle.scenario,
                task_id=initial_state_bundle.task_id,
                candidate_memory_id=memory_id,
                engine_name=share_env.engine_name,
                engine_version=share_env.engine_version,
                real_engine_executed=real_engine_executed,
                share=share,
                withhold=withhold,
                paired_record_valid=valid,
                invalid_reason=reason,
                paired_label=(
                    _paired_label(share.outcome.success, withhold.outcome.success)
                    if valid
                    else None
                ),
                branch_execution_order=branch_execution_order,
            )
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "branch_audit.json").write_text(
                json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            return result
        finally:
            if share_env is not None:
                share_env.close()
            if withhold_env is not None:
                withhold_env.close()
            rebuilder.destroy()

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
    ) -> MarbleBranchAudit:
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
        )


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
