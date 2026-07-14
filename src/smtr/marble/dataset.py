"""Discovery and manifest utilities for real MARBLE benchmark tasks."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from smtr.counterfactual.decision_points import canonical_digest

DEFAULT_MARBLE_ROOT = Path("/home/ecs-user/MARBLE")

MARBLE_BENCHMARK_FILES: dict[str, Path] = {
    "bargaining": Path("multiagentbench/bargaining/bargaining_main.jsonl"),
    "coding": Path("multiagentbench/coding/coding_main.jsonl"),
    "database": Path("multiagentbench/database/database_main.jsonl"),
    "minecraft": Path("multiagentbench/minecraft/minecraft_main.jsonl"),
    "research": Path("multiagentbench/research/research_main.jsonl"),
}


class MarbleTaskManifestRecord(BaseModel):
    """Stable identity and lightweight metadata for one MARBLE benchmark task."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    scenario: str
    task_id: str
    source_path: str
    source_line: int
    source_digest: str
    task_digest: str
    task_content_digest: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    agent_count: int
    relationship_count: int
    environment_type: str | None = None
    environment_name: str | None = None
    init_sql_digest: str | None = None
    labels: list[str] = Field(default_factory=list)
    root_causes: list[str] = Field(default_factory=list)
    number_of_labels_pred: int | None = None


class MarbleDatasetManifest(BaseModel):
    """Frozen manifest for the MARBLE tasks visible to SMTR."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "database_dataset_v1"
    dataset_name: str = "MARBLE MultiAgentBench"
    marble_root: str
    marble_commit: str = "unknown"
    smtr_commit: str = "unknown"
    dataset_relative_path: str | None = None
    dataset_absolute_path: str | None = None
    dataset_sha256: str | None = None
    ordered_task_ids: list[str] = Field(default_factory=list)
    task_schema_summary: dict[str, Any] = Field(default_factory=dict)
    associated_files: dict[str, str] = Field(default_factory=dict)
    created_at: str | None = None
    total_tasks: int
    scenario_counts: dict[str, int]
    source_file_digests: dict[str, str]
    tasks: list[MarbleTaskManifestRecord]


def discover_marble_benchmark_tasks(
    *,
    marble_root: Path = DEFAULT_MARBLE_ROOT,
    scenarios: set[str] | None = None,
    limit_per_scenario: int | None = None,
) -> list[MarbleTaskManifestRecord]:
    """Discover real MARBLE MultiAgentBench tasks from JSONL files.

    This reads the benchmark data shipped by MARBLE directly. It does not run
    MARBLE, mutate environments, call an LLM, or use any SMTR smoke fixtures.
    """

    if limit_per_scenario is not None and limit_per_scenario < 1:
        raise ValueError("limit_per_scenario must be >= 1 when provided")

    selected = set(MARBLE_BENCHMARK_FILES) if scenarios is None else set(scenarios)
    unknown = selected - set(MARBLE_BENCHMARK_FILES)
    if unknown:
        raise ValueError(f"unknown MARBLE scenario(s): {sorted(unknown)}")

    records: list[MarbleTaskManifestRecord] = []
    for scenario in sorted(selected):
        path = marble_root / MARBLE_BENCHMARK_FILES[scenario]
        if not path.exists():
            raise FileNotFoundError(f"MARBLE benchmark file not found: {path}")
        count_for_scenario = 0
        with path.open("r", encoding="utf-8") as handle:
            for source_line, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                records.append(
                    _task_record_from_raw(
                        raw=raw,
                        dataset=scenario,
                        source_path=path,
                        source_line=source_line,
                    )
                )
                count_for_scenario += 1
                if limit_per_scenario is not None and count_for_scenario >= limit_per_scenario:
                    break
    return records


def build_marble_dataset_manifest(
    *,
    marble_root: Path = DEFAULT_MARBLE_ROOT,
    scenarios: set[str] | None = None,
    limit_per_scenario: int | None = None,
) -> MarbleDatasetManifest:
    """Build a frozen manifest over discovered MARBLE benchmark tasks."""

    tasks = discover_marble_benchmark_tasks(
        marble_root=marble_root,
        scenarios=scenarios,
        limit_per_scenario=limit_per_scenario,
    )
    source_paths = sorted({Path(task.source_path) for task in tasks})
    database_path = marble_root / MARBLE_BENCHMARK_FILES["database"]
    database_only = scenarios == {"database"}
    task_ids = [task.task_id for task in tasks]
    if database_only:
        duplicates = sorted(task_id for task_id, count in Counter(task_ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate MARBLE task IDs: {duplicates}")
        for task in tasks:
            if not task.task_id or not task.task_content_digest or not task.init_sql_digest:
                raise ValueError(
                    f"MARBLE task {task.task_id!r} lacks task content or environment.init_sql"
                )
    associated = _database_associated_files(marble_root) if database_only else {}
    return MarbleDatasetManifest(
        dataset_name=(
            "MARBLE MultiAgentBench database" if database_only else "MARBLE MultiAgentBench"
        ),
        marble_root=str(marble_root),
        marble_commit=_git_commit(marble_root),
        smtr_commit=_git_commit(Path(__file__).resolve().parents[3]),
        dataset_relative_path=(str(MARBLE_BENCHMARK_FILES["database"]) if database_only else None),
        dataset_absolute_path=str(database_path) if database_only else None,
        dataset_sha256=_file_sha256(database_path) if database_only else None,
        ordered_task_ids=task_ids,
        task_schema_summary=(
            {
                "task_id": "top-level integer/string, normalized to string",
                "instruction": "task.content",
                "database_initialization": "environment.init_sql",
                "reference_fields": ["task.labels", "task.root_causes"],
                "evaluator_configuration": ["task.number_of_labels_pred", "task.output_format"],
            }
            if database_only
            else {}
        ),
        associated_files=associated,
        created_at=_git_commit_timestamp(marble_root),
        total_tasks=len(tasks),
        scenario_counts=dict(sorted(Counter(task.scenario for task in tasks).items())),
        source_file_digests={str(path): _file_sha256(path) for path in source_paths},
        tasks=tasks,
    )


def write_marble_dataset_manifest(
    *,
    output_path: Path,
    marble_root: Path = DEFAULT_MARBLE_ROOT,
    scenarios: set[str] | None = None,
    limit_per_scenario: int | None = None,
) -> MarbleDatasetManifest:
    """Write a MARBLE dataset manifest and return the in-memory object."""

    manifest = build_marble_dataset_manifest(
        marble_root=marble_root,
        scenarios=scenarios,
        limit_per_scenario=limit_per_scenario,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _task_record_from_raw(
    *,
    raw: dict[str, Any],
    dataset: str,
    source_path: Path,
    source_line: int,
) -> MarbleTaskManifestRecord:
    raw_task = raw.get("task")
    raw_environment = raw.get("environment")
    raw_agents = raw.get("agents")
    raw_relationships = raw.get("relationships")
    task: dict[str, Any] = raw_task if isinstance(raw_task, dict) else {}
    environment: dict[str, Any] = raw_environment if isinstance(raw_environment, dict) else {}
    agents: list[Any] = raw_agents if isinstance(raw_agents, list) else []
    relationships: list[Any] = raw_relationships if isinstance(raw_relationships, list) else []
    scenario = str(raw.get("scenario") or dataset)
    task_id = _stable_task_id(raw=raw, environment=environment, source_line=source_line)
    task_content = task.get("content")
    init_sql = environment.get("init_sql")
    if not isinstance(task_content, str) or not task_content.strip():
        raise ValueError(f"task at {source_path}:{source_line} lacks task.content")
    return MarbleTaskManifestRecord(
        dataset=dataset,
        scenario=scenario,
        task_id=task_id,
        source_path=str(source_path),
        source_line=source_line,
        source_digest=_cached_file_sha256(str(source_path)),
        task_digest=canonical_digest(
            {"raw": raw, "source_path": str(source_path), "source_line": source_line}
        ),
        task_content_digest=canonical_digest(task_content or ""),
        metadata={
            "coordinate_mode": raw.get("coordinate_mode", ""),
            "communication": raw.get("communication"),
            "metrics": raw.get("metrics", {}),
            "output": raw.get("output", {}),
            "schema_family_digest": (
                canonical_digest(_schema_statements(init_sql)) if init_sql else None
            ),
        },
        agent_count=len(agents),
        relationship_count=len(relationships),
        environment_type=_optional_str(environment.get("type")),
        environment_name=_optional_str(environment.get("name")),
        init_sql_digest=canonical_digest(init_sql) if init_sql else None,
        labels=[str(label) for label in task.get("labels", [])],
        root_causes=[str(label) for label in task.get("root_causes", [])],
        number_of_labels_pred=(
            int(task["number_of_labels_pred"])
            if task.get("number_of_labels_pred") is not None
            else None
        ),
    )


def _stable_task_id(
    *,
    raw: dict[str, Any],
    environment: dict[str, Any],
    source_line: int,
) -> str:
    task_id = raw.get("task_id")
    if task_id is None:
        task_id = environment.get("task_id")
    if task_id is None:
        task_id = f"line-{source_line}"
    return str(task_id)


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _schema_statements(init_sql: str) -> list[str]:
    prefixes = ("create ", "alter ", "create index", "create unique index")
    without_comments = re.sub(r"--[^\n]*", "", init_sql)
    return sorted(
        " ".join(statement.lower().split())
        for statement in without_comments.split(";")
        if statement.strip().lower().startswith(prefixes)
    )


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@lru_cache(maxsize=32)
def _cached_file_sha256(path: str) -> str:
    return _file_sha256(Path(path))


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_commit_timestamp(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "show", "-s", "--format=%cI", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def _database_associated_files(marble_root: Path) -> dict[str, str]:
    relative_paths = [
        Path("marble/environments/db_env.py"),
        Path("marble/environments/db_env_docker/anomaly_trigger/formalize_anomalies.py"),
    ]
    return {
        str(path): _file_sha256(marble_root / path)
        for path in relative_paths
        if (marble_root / path).is_file()
    }
