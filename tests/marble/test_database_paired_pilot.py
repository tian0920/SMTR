import json
from pathlib import Path

import pytest

from smtr.marble.database_pilot import create_database_pilot_manifest, run_database_paired_pilot
from smtr.marble.dataset import build_marble_dataset_manifest
from smtr.marble.integrity import audit_marble_pilot_run
from smtr.marble.splits import create_split_manifest

MARBLE_ROOT = Path("/home/ecs-user/MARBLE")


def test_database_pilot_manifest_expected_role_is_not_label(tmp_path: Path) -> None:
    dataset = build_marble_dataset_manifest(marble_root=MARBLE_ROOT)
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset.model_dump(mode="json")), encoding="utf-8")
    splits = create_split_manifest(dataset_manifest_path=dataset_path, seed=0)
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits.model_dump(mode="json")), encoding="utf-8")
    output = Path("artifacts/marble/manifests/database_paired_pilot_test.json")

    manifest = create_database_pilot_manifest(
        dataset_manifest_path=dataset_path,
        split_manifest_path=splits_path,
        output_path=output,
        task_count=1,
    )

    memory = manifest["tasks"][0]["candidate_memories"][0]
    assert memory["expected_role"] in {"helpful", "harmful", "irrelevant"}
    assert "paired_label" not in memory


def test_only_neutral_or_invalid_pairs_do_not_enable_paired_data(tmp_path: Path) -> None:
    dataset = build_marble_dataset_manifest(marble_root=MARBLE_ROOT)
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset.model_dump(mode="json")), encoding="utf-8")
    splits = create_split_manifest(dataset_manifest_path=dataset_path, seed=0)
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits.model_dump(mode="json")), encoding="utf-8")
    pilot_manifest = Path("artifacts/marble/manifests/database_paired_pilot_test_small.json")
    create_database_pilot_manifest(
        dataset_manifest_path=dataset_path,
        split_manifest_path=splits_path,
        output_path=pilot_manifest,
        task_count=1,
    )
    run_dir = Path("artifacts/marble/records/pilot/database_test")
    with pytest.raises(RuntimeError, match="preflight failed"):
        run_database_paired_pilot(
            pilot_manifest_path=pilot_manifest,
            output_dir=run_dir,
        )
    assert audit_marble_pilot_run(run_dir=run_dir)["invalid_pair_count"] >= 0
