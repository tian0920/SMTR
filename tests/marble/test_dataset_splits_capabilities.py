import json
from pathlib import Path

from smtr.marble.capabilities import inspect_capabilities
from smtr.marble.dataset import build_marble_dataset_manifest
from smtr.marble.splits import create_split_manifest

MARBLE_ROOT = Path("/home/ecs-user/MARBLE")


def test_real_marble_dataset_manifest_counts() -> None:
    manifest = build_marble_dataset_manifest(marble_root=MARBLE_ROOT)

    assert manifest.total_tasks == 500
    assert manifest.scenario_counts == {
        "bargaining": 100,
        "coding": 100,
        "database": 100,
        "minecraft": 100,
        "research": 100,
    }
    assert manifest.tasks[0].source_digest
    assert manifest.tasks[0].metadata is not None


def test_split_manifest_is_disjoint_and_reproducible(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset = build_marble_dataset_manifest(marble_root=MARBLE_ROOT)
    dataset_path.write_text(
        json.dumps(dataset.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )

    left = create_split_manifest(dataset_manifest_path=dataset_path, seed=0)
    right = create_split_manifest(dataset_manifest_path=dataset_path, seed=0)

    assert left == right
    assignments = {}
    for record in left.records:
        assert record.task_digest not in assignments
        assignments[record.task_digest] = record.split
    assert set(left.split_counts) == {"train", "validation", "test"}


def test_capability_matrix_distinguishes_harness_from_real_pilot() -> None:
    manifest = inspect_capabilities(marble_root=MARBLE_ROOT)

    assert manifest.pilot_scenario is None
    assert manifest.scenarios["database"].isolation_harness_supported is True
    assert manifest.scenarios["database"].pilot_supported is False
    assert manifest.scenarios["database"].real_engine_execution_verified is False
    for scenario, capability in manifest.scenarios.items():
        if scenario != "database":
            assert capability.pilot_supported is False
