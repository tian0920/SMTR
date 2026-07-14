from pathlib import Path

from smtr.marble.database_smoke import run_database_paired_smoke


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
