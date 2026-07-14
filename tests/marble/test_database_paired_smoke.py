from pathlib import Path

from smtr.marble.branch_runner import MarbleBranchAudit, _validate_pair
from smtr.marble.database_smoke import run_database_paired_smoke
from smtr.marble.memory_injection import MarbleAgentInputAudit
from smtr.marble.outcome.protocol import MarbleOutcome


def test_paired_smoke_invalid_when_engine_fails(tmp_path: Path) -> None:
    summary = run_database_paired_smoke(
        marble_root=Path("/home/ecs-user/MARBLE"),
        task_id="1",
        memory_id="database_1_helpful",
        generation_seed=0,
        branch_order="share-then-withhold",
        output_dir=Path("artifacts/marble/outputs/database_paired_smoke_test"),
    )

    assert summary["memory_intervention_verified"] is True
    assert summary["initial_logical_digest_match"] is True
    assert summary["paired_record_valid"] is False
    assert summary["invalid_reason"] == "real_marble_engine_not_executed"


def test_paired_record_invalid_when_cleanup_fails() -> None:
    share = _audit(branch_id="share", cleanup_succeeded=False, contains_memory=True)
    withhold = _audit(branch_id="withhold", cleanup_succeeded=True, contains_memory=False)

    valid, reason = _validate_pair(
        share=share,
        withhold=withhold,
        real_engine_executed=True,
    )

    assert valid is False
    assert "share_cleanup_succeeded" in str(reason)


def _audit(
    *,
    branch_id: str,
    cleanup_succeeded: bool,
    contains_memory: bool,
) -> MarbleBranchAudit:
    input_audit = MarbleAgentInputAudit(
        system_section_digest="system",
        task_section_digest="task",
        tool_section_digest="tool",
        memory_section_digest="memory" if contains_memory else None,
        full_input_digest=branch_id,
        memory_ids=("m1",) if contains_memory else (),
        contains_memory_section=contains_memory,
    )
    outcome = MarbleOutcome(
        success=True,
        score=1.0,
        failure_reason=None,
        environment_valid=True,
        evaluator_name="native",
        raw_result_digest="raw",
        native_evaluator_executed=True,
        native_evaluator_name="native",
        native_evaluator_result_digest="eval",
    )
    return MarbleBranchAudit(
        branch_id=branch_id,
        workspace=branch_id,
        initial_digest="initial",
        initial_logical_fingerprint={"combined_digest": "logical"},
        final_digest="final",
        raw_result_digest="raw",
        input_audit=input_audit,
        agent_config_digest="agent",
        generation_seed=0,
        task_digest="task",
        tool_config_digest="tool",
        outcome=outcome,
        real_engine_executed=True,
        cleanup_succeeded=cleanup_succeeded,
        cleanup_exit_code=0 if cleanup_succeeded else 1,
        cleanup_failure_reason=None if cleanup_succeeded else "cleanup_exit_code=1",
    )
