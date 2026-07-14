"""Frozen train/validation/test split manifests for MARBLE."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.artifacts import assert_marble_artifact_path

SplitName = Literal["train", "validation", "test"]


class MarbleSplitRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_digest: str
    task_id: str
    scenario: str
    split: SplitName
    group_id: str


class MarbleSplitManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "database_split_v1"
    dataset_manifest_path: str | None = None
    dataset_manifest_sha256: str | None = None
    dataset_manifest_digest: str
    split_strategy: str
    split_strategy_reason: str
    seed: int
    ratios: dict[str, float]
    records: list[MarbleSplitRecord] = Field(default_factory=list)
    group_ids: list[str] = Field(default_factory=list)
    train_task_ids: list[str] = Field(default_factory=list)
    validation_task_ids: list[str] = Field(default_factory=list)
    test_task_ids: list[str] = Field(default_factory=list)
    created_at: str | None = None
    smtr_commit: str = "unknown"

    @property
    def split_counts(self) -> dict[str, int]:
        counts = {"train": 0, "validation": 0, "test": 0}
        for record in self.records:
            counts[record.split] += 1
        return counts


def create_split_manifest(
    *,
    dataset_manifest_path: Path,
    seed: int,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> MarbleSplitManifest:
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    database_manifest = dataset.get("dataset_name") == "MARBLE MultiAgentBench database"
    if round(train_ratio + validation_ratio + test_ratio, 10) != 1.0:
        raise ValueError("split ratios must sum to 1")
    groups: dict[str, list[dict]] = defaultdict(list)
    for task in dataset.get("tasks", []):
        group_id = _group_id(task)
        groups[group_id].append(task)

    group_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    n_groups = len(group_ids)
    train_cut = int(n_groups * train_ratio)
    validation_cut = train_cut + int(n_groups * validation_ratio)
    split_by_group: dict[str, SplitName] = {}
    for index, group_id in enumerate(group_ids):
        if index < train_cut:
            split: SplitName = "train"
        elif index < validation_cut:
            split = "validation"
        else:
            split = "test"
        split_by_group[group_id] = split

    records = []
    seen_tasks: set[str] = set()
    for group_id in sorted(groups):
        for task in sorted(groups[group_id], key=lambda item: item["task_digest"]):
            task_digest = task["task_digest"]
            if task_digest in seen_tasks:
                raise ValueError(f"duplicate task digest in dataset manifest: {task_digest}")
            seen_tasks.add(task_digest)
            records.append(
                MarbleSplitRecord(
                    task_digest=task_digest,
                    task_id=str(task["task_id"]),
                    scenario=str(task["scenario"]),
                    split=split_by_group[group_id],
                    group_id=group_id,
                )
            )
    manifest = MarbleSplitManifest(
        dataset_manifest_path=str(dataset_manifest_path),
        dataset_manifest_sha256=_file_sha256(dataset_manifest_path),
        dataset_manifest_digest=canonical_digest(dataset),
        split_strategy=(
            "grouped_by_database_schema_family"
            if database_manifest
            else "grouped_by_scenario_task_id_prefix"
        ),
        split_strategy_reason=(
            "Database tasks are grouped by normalized initialization schema digest; tasks "
            "sharing the same CREATE/ALTER/INDEX family cannot cross splits."
            if database_manifest
            else "Generic MARBLE tasks are grouped by scenario and task-id decile."
        ),
        seed=seed,
        ratios={"train": train_ratio, "validation": validation_ratio, "test": test_ratio},
        records=records,
        group_ids=sorted(groups),
        train_task_ids=[record.task_id for record in records if record.split == "train"],
        validation_task_ids=[record.task_id for record in records if record.split == "validation"],
        test_task_ids=[record.task_id for record in records if record.split == "test"],
        created_at=dataset.get("created_at"),
        smtr_commit=_git_commit(Path(__file__).resolve().parents[3]),
    )
    if database_manifest:
        validate_split_manifest(
            manifest,
            expected_task_ids={str(t["task_id"]) for t in dataset["tasks"]},
        )
    return manifest


def write_split_manifest(
    *,
    dataset_manifest_path: Path,
    output_path: Path,
    seed: int,
) -> MarbleSplitManifest:
    assert_marble_artifact_path(output_path)
    manifest = create_split_manifest(dataset_manifest_path=dataset_manifest_path, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _group_id(task: dict) -> str:
    schema_digest = task.get("metadata", {}).get("schema_family_digest")
    if not schema_digest:
        # Existing manifests expose the full init SQL digest. It is still safer
        # than per-row random splitting and is upgraded when schema_family_digest exists.
        schema_digest = task.get("init_sql_digest")
    if schema_digest:
        return f"database:schema:{schema_digest}"
    scenario = str(task.get("scenario") or task.get("dataset") or "unknown")
    task_id = str(task.get("task_id") or task.get("source_line") or "unknown")
    try:
        bucket: int | str = (int(task_id) - 1) // 10
    except ValueError:
        bucket = task_id
    return f"{scenario}:task_bucket:{bucket}"


def validate_split_manifest(
    manifest: MarbleSplitManifest,
    *,
    expected_task_ids: set[str],
) -> None:
    ids_by_split = {
        split: {record.task_id for record in manifest.records if record.split == split}
        for split in ("train", "validation", "test")
    }
    if ids_by_split["train"] & ids_by_split["validation"]:
        raise ValueError("train/validation task overlap")
    if ids_by_split["train"] & ids_by_split["test"]:
        raise ValueError("train/test task overlap")
    if ids_by_split["validation"] & ids_by_split["test"]:
        raise ValueError("validation/test task overlap")
    all_ids = set().union(*ids_by_split.values())
    if all_ids != expected_task_ids or len(manifest.records) != len(expected_task_ids):
        raise ValueError("every dataset task must occur exactly once")
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for record in manifest.records:
        splits_by_group[record.group_id].add(record.split)
    if any(len(splits) != 1 for splits in splits_by_group.values()):
        raise ValueError("database group leakage across splits")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"
