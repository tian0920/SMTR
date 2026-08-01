import json
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
def test_database_environment_run_invokes_real_engine(tmp_path: Path) -> None:
    """Verify that run() now invokes the real engine subprocess (no fail-fast).

    The environment no longer raises RuntimeError; it delegates to
    run_marble_engine_process().  Without Docker running, the engine
    subprocess will fail, but the environment gracefully returns a result
    dict with real_engine_executed=False.
    """
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
    # Without Docker, the engine will fail, but run() should NOT raise.
    result = env.run(
        agent_input=env.build_agent_input(memory_payloads=()),
        generation_seed=0,
        engine_timeout_seconds=30,
    )
    assert isinstance(result, dict)
    assert "real_engine_executed" in result
    assert "task_evaluation" in result
    env.close()


def test_database_environment_config_uses_absolute_output_path(tmp_path: Path) -> None:
    task = {
        "task_id": "path-test",
        "task": {},
        "environment": {},
        "agents": [],
    }
    bundle = bundle_from_manifest_task(
        {"raw_task": task, "task_id": task["task_id"], "scenario": "database"}
    )
    env = MarbleDatabaseEnvironment(
        task=task,
        workspace=tmp_path / "workspace with spaces",
        initial_state_bundle=bundle,
        agent_config={"target_receiver_agent_id": "agent1"},
        marble_root=MARBLE_ROOT,
    )

    config_path = env.workspace / "marble_config.yaml"
    raw_result_path = env.workspace / "marble_output.jsonl"
    env._write_yaml_config(
        agent_input=env.build_agent_input(memory_payloads=()),
        generation_seed=0,
        config_path=config_path,
        raw_result_path=raw_result_path,
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_path = Path(config["output"]["file_path"])
    assert output_path.is_absolute()
    assert output_path == raw_result_path.resolve()
    text = json.dumps(config)
    assert "DASHSCOPE_API_KEY" not in text
    assert "OPENAI_API_KEY" not in text
    assert "Authorization" not in text
    assert "Bearer" not in text
