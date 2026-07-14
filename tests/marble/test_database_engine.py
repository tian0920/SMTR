from pathlib import Path

import pytest

from smtr.marble.engine_audit import audit_database_engine
from smtr.marble.environment.isolation import bundle_from_manifest_task
from smtr.marble.environment.scenarios.database import MarbleDatabaseEnvironment
from smtr.marble.task_provider import _read_jsonl_line

MARBLE_ROOT = Path("/home/ecs-user/MARBLE")


@pytest.mark.marble
def test_database_engine_audit_records_fixed_workspace_limitation(tmp_path: Path) -> None:
    output = Path("artifacts/marble/manifests/database_engine_audit_test.json")
    summary = audit_database_engine(marble_root=MARBLE_ROOT, output_path=output)

    assert summary["engine_entrypoint"].endswith("marble/main.py")
    assert summary["environment_constructor"] == "marble.environments.DBEnvironment"
    assert summary["supports_custom_workspace"] is False
    assert summary["real_engine_execution_safe_for_paired_isolation"] is False


@pytest.mark.marble
def test_database_environment_fails_fast_without_surrogate_execution(tmp_path: Path) -> None:
    path = MARBLE_ROOT / "multiagentbench/database/database_main.jsonl"
    task = _read_jsonl_line(path, 1)
    bundle = bundle_from_manifest_task(
        {"raw_task": task, "task_id": task["task_id"], "scenario": "database"}
    )
    env = MarbleDatabaseEnvironment(
        task=task,
        workspace=tmp_path / "db",
        initial_state_bundle=bundle,
        agent_config={"target_receiver_agent_id": "agent1"},
        marble_root=MARBLE_ROOT,
    )

    with pytest.raises(
        RuntimeError,
        match="real_marble_database_engine_(not_executed|import_failed)",
    ):
        env.run(agent_input=env.build_agent_input(memory_payloads=()), generation_seed=0)
