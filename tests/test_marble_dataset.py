import json
from pathlib import Path

from smtr.marble.dataset import (
    build_marble_dataset_manifest,
    discover_marble_benchmark_tasks,
    write_marble_dataset_manifest,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_discover_marble_benchmark_tasks_reads_real_jsonl_shape(tmp_path: Path) -> None:
    root = tmp_path / "MARBLE"
    _write_jsonl(
        root / "multiagentbench/database/database_main.jsonl",
        [
            {
                "scenario": "database",
                "task_id": 7,
                "task": {
                    "content": "Diagnose a database anomaly.",
                    "labels": ["LOCK_CONTENTION", "VACUUM"],
                    "root_causes": ["LOCK_CONTENTION"],
                    "number_of_labels_pred": 2,
                },
                "environment": {
                    "type": "db",
                    "name": "postgres",
                    "init_sql": "CREATE TABLE t(id INT);",
                },
                "agents": [{"agent_id": "agent1"}, {"agent_id": "agent2"}],
                "relationships": [["agent1", "agent2", "collaborate with"]],
            }
        ],
    )

    records = discover_marble_benchmark_tasks(
        marble_root=root,
        scenarios={"database"},
    )

    assert len(records) == 1
    record = records[0]
    assert record.dataset == "database"
    assert record.scenario == "database"
    assert record.task_id == "7"
    assert record.agent_count == 2
    assert record.relationship_count == 1
    assert record.root_causes == ["LOCK_CONTENTION"]
    assert record.init_sql_digest is not None


def test_build_marble_dataset_manifest_counts_scenarios_and_limits(
    tmp_path: Path,
) -> None:
    root = tmp_path / "MARBLE"
    _write_jsonl(
        root / "multiagentbench/database/database_main.jsonl",
        [
            {"scenario": "database", "task_id": 1, "task": {"content": "a"}},
            {"scenario": "database", "task_id": 2, "task": {"content": "b"}},
        ],
    )
    _write_jsonl(
        root / "multiagentbench/coding/coding_main.jsonl",
        [
            {"scenario": "coding", "task_id": 1, "task": {"content": "c"}},
            {"scenario": "coding", "task_id": 2, "task": {"content": "d"}},
        ],
    )

    manifest = build_marble_dataset_manifest(
        marble_root=root,
        scenarios={"database", "coding"},
        limit_per_scenario=1,
    )

    assert manifest.total_tasks == 2
    assert manifest.scenario_counts == {"coding": 1, "database": 1}
    assert len(manifest.source_file_digests) == 2


def test_write_marble_dataset_manifest(tmp_path: Path) -> None:
    root = tmp_path / "MARBLE"
    _write_jsonl(
        root / "multiagentbench/research/research_main.jsonl",
        [{"scenario": "research", "task_id": 3, "task": {"content": "idea"}}],
    )
    output = tmp_path / "manifest.json"

    manifest = write_marble_dataset_manifest(
        marble_root=root,
        scenarios={"research"},
        output_path=output,
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert manifest.total_tasks == 1
    assert loaded["total_tasks"] == 1
    assert loaded["tasks"][0]["scenario"] == "research"
