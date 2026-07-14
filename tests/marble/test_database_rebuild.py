from pathlib import Path

from smtr.marble.environment.database_rebuild import SequentialDatabaseRebuilder
from smtr.marble.environment.isolation import bundle_from_manifest_task
from smtr.marble.task_provider import _read_jsonl_line

MARBLE_ROOT = Path("/home/ecs-user/MARBLE")


def test_database_rebuild_initial_fingerprint_is_repeatable(tmp_path: Path) -> None:
    task = _read_jsonl_line(MARBLE_ROOT / "multiagentbench/database/database_main.jsonl", 1)
    bundle = bundle_from_manifest_task(
        {"raw_task": task, "task_id": task["task_id"], "scenario": "database"}
    )
    rebuilder = SequentialDatabaseRebuilder(marble_root=MARBLE_ROOT)

    fingerprint_a = rebuilder.materialize(
        initial_state_bundle=bundle,
        branch_workspace=tmp_path / "a",
    )
    rebuilder.destroy()
    fingerprint_b = rebuilder.materialize(
        initial_state_bundle=bundle,
        branch_workspace=tmp_path / "b",
    )
    rebuilder.destroy()

    assert fingerprint_a.combined_digest == fingerprint_b.combined_digest


def test_database_rebuild_marker_does_not_leak(tmp_path: Path) -> None:
    task = _read_jsonl_line(MARBLE_ROOT / "multiagentbench/database/database_main.jsonl", 1)
    bundle = bundle_from_manifest_task(
        {"raw_task": task, "task_id": task["task_id"], "scenario": "database"}
    )
    rebuilder = SequentialDatabaseRebuilder(marble_root=MARBLE_ROOT)

    rebuilder.materialize(initial_state_bundle=bundle, branch_workspace=tmp_path / "a")
    (tmp_path / "a" / "marker.txt").write_text("marker", encoding="utf-8")
    assert (tmp_path / "a" / "marker.txt").exists()
    rebuilder.destroy()
    rebuilder.materialize(initial_state_bundle=bundle, branch_workspace=tmp_path / "b")
    assert not (tmp_path / "b" / "marker.txt").exists()
    rebuilder.destroy()
