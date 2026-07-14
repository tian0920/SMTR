"""Task provider for frozen MARBLE dataset manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class MarbleTaskSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    scenario: str
    task_digest: str
    source_path: str
    source_line: int
    raw_task: dict[str, Any]


class MarbleTaskProvider:
    """Load real MARBLE tasks referenced by a frozen dataset manifest."""

    def __init__(self, *, dataset_manifest_path: Path) -> None:
        self.dataset_manifest_path = dataset_manifest_path
        self.dataset_manifest = json.loads(
            dataset_manifest_path.read_text(encoding="utf-8")
        )
        self._by_digest = {
            task["task_digest"]: task for task in self.dataset_manifest.get("tasks", [])
        }

    def get_by_digest(self, task_digest: str) -> MarbleTaskSpec:
        if task_digest not in self._by_digest:
            raise KeyError(f"task digest not in dataset manifest: {task_digest}")
        entry = self._by_digest[task_digest]
        raw = _read_jsonl_line(Path(entry["source_path"]), int(entry["source_line"]))
        return MarbleTaskSpec(
            task_id=str(entry["task_id"]),
            scenario=str(entry["scenario"]),
            task_digest=str(entry["task_digest"]),
            source_path=str(entry["source_path"]),
            source_line=int(entry["source_line"]),
            raw_task=raw,
        )

    def iter_split(
        self,
        *,
        split_manifest_path: Path,
        split: str,
        scenario: str | None = None,
        limit: int | None = None,
    ) -> list[MarbleTaskSpec]:
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        specs: list[MarbleTaskSpec] = []
        for record in split_manifest.get("records", []):
            if record["split"] != split:
                continue
            if scenario is not None and record["scenario"] != scenario:
                continue
            specs.append(self.get_by_digest(record["task_digest"]))
            if limit is not None and len(specs) >= limit:
                break
        return specs


def _read_jsonl_line(path: Path, line_number: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle, start=1):
            if current == line_number:
                return json.loads(line)
    raise ValueError(f"line {line_number} not found in {path}")
