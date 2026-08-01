"""Unified IO for MARBLE split/dataset manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_split_task_ids(
    split_manifest_path: Path,
    split: str,
) -> set[str]:
    manifest = load_json(split_manifest_path)

    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError(
            "split manifest must contain a 'records' list; "
            "legacy split->task-list format is unsupported"
        )

    return {
        str(record["task_id"])
        for record in records
        if record.get("split") == split
    }


def load_dataset_tasks(
    dataset_manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    manifest = load_json(dataset_manifest_path)

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("dataset manifest must contain a 'tasks' list")

    return {
        str(task["task_id"]): task
        for task in tasks
    }
