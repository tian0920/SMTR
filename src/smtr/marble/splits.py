"""Frozen train/validation/test split manifests for MARBLE."""

from __future__ import annotations

import json
import random
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

    dataset_manifest_digest: str
    split_strategy: str
    split_strategy_reason: str
    seed: int
    ratios: dict[str, float]
    records: list[MarbleSplitRecord] = Field(default_factory=list)

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
    return MarbleSplitManifest(
        dataset_manifest_digest=canonical_digest(dataset),
        split_strategy="grouped_by_scenario_task_id_prefix",
        split_strategy_reason=(
            "MARBLE JSONL tasks expose scenario and task_id. No richer template/project "
            "field is consistently available, so splitting groups by scenario plus task-id "
            "decile prevents the same task from crossing splits while keeping scenario balance."
        ),
        seed=seed,
        ratios={"train": train_ratio, "validation": validation_ratio, "test": test_ratio},
        records=records,
    )


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
    scenario = str(task.get("scenario") or task.get("dataset") or "unknown")
    task_id = str(task.get("task_id") or task.get("source_line") or "unknown")
    try:
        bucket = (int(task_id) - 1) // 10
    except ValueError:
        bucket = task_id
    return f"{scenario}:task_bucket:{bucket}"
