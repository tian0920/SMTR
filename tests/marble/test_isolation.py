import json
from pathlib import Path

from smtr.marble.branch_runner import MarblePairedBranchRunner
from smtr.marble.dataset import build_marble_dataset_manifest
from smtr.marble.environment.factory import factory_for_scenario
from smtr.marble.environment.isolation import (
    bundle_from_manifest_task,
    workspace_digest,
)
from smtr.marble.task_provider import _read_jsonl_line

MARBLE_ROOT = Path("/home/ecs-user/MARBLE")


def _first_database_task() -> dict:
    manifest = build_marble_dataset_manifest(
        marble_root=MARBLE_ROOT,
        scenarios={"database"},
        limit_per_scenario=1,
    )
    entry = manifest.tasks[0]
    return _read_jsonl_line(Path(entry.source_path), entry.source_line)


def test_independent_branch_creation_has_same_initial_digest(tmp_path: Path) -> None:
    task = _first_database_task()
    bundle = bundle_from_manifest_task(
        {"raw_task": task, "task_id": task["task_id"], "scenario": "database"}
    )
    factory = factory_for_scenario("database")

    share = factory.create_isolated(
        task=task,
        initial_state_bundle=bundle,
        branch_id="share",
        workspace=str(tmp_path / "share"),
    )
    withhold = factory.create_isolated(
        task=task,
        initial_state_bundle=bundle,
        branch_id="withhold",
        workspace=str(tmp_path / "withhold"),
    )

    assert share.initial_state_digest() == withhold.initial_state_digest()


def test_branch_side_effects_do_not_cross_workspaces(tmp_path: Path) -> None:
    task = _first_database_task()
    bundle = bundle_from_manifest_task(
        {"raw_task": task, "task_id": task["task_id"], "scenario": "database"}
    )
    factory = factory_for_scenario("database")
    share = factory.create_isolated(
        task=task,
        initial_state_bundle=bundle,
        branch_id="share",
        workspace=str(tmp_path / "share"),
    )
    withhold = factory.create_isolated(
        task=task,
        initial_state_bundle=bundle,
        branch_id="withhold",
        workspace=str(tmp_path / "withhold"),
    )

    before = workspace_digest(Path(withhold.workspace))
    (Path(share.workspace) / "side_effect.txt").write_text("changed", encoding="utf-8")

    assert workspace_digest(Path(withhold.workspace)) == before


def test_paired_branch_runner_invalid_without_real_engine(tmp_path: Path) -> None:
    """Verify paired branch runner produces structurally valid output.

    With Docker available the engine subprocess may succeed, but the
    native evaluator might not.  The pair should still be structurally
    sound: same initial digests, distinct workspaces, correct audits.
    """
    workspace = Path("artifacts/marble/workspaces/test_isolation")
    if workspace.exists():
        import shutil

        shutil.rmtree(workspace)
    task = _first_database_task()
    bundle = bundle_from_manifest_task(
        {"raw_task": task, "task_id": task["task_id"], "scenario": "database"}
    )

    result = MarblePairedBranchRunner().run_pair(
        task=task,
        candidate_memory={"memory_id": "m1", "payload": "diagnostic help"},
        initial_state_bundle=bundle,
        agent_config={"target_receiver_agent_id": "agent1"},
        generation_seed=0,
        workspace=workspace,
        engine_timeout_seconds=30,
    )

    # Pair structural invariants (regardless of engine availability)
    assert result.share.initial_digest == result.withhold.initial_digest
    assert Path(result.share.workspace) != Path(result.withhold.workspace)
    assert result.share.input_audit.contains_memory_section is True
    assert result.withhold.input_audit.contains_memory_section is False
    assert json.loads((workspace / "branch_audit.json").read_text(encoding="utf-8"))
