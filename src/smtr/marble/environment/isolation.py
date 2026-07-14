"""Initial-state bundles and filesystem isolation for MARBLE pilot tasks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from smtr.counterfactual.decision_points import canonical_digest


class InitialStateBundle(BaseModel):
    """Immutable material needed to recreate an isolated MARBLE branch."""

    model_config = ConfigDict(frozen=True)

    scenario: str
    task_id: str
    task_source_snapshot: dict[str, Any]
    environment_configuration: dict[str, Any] = Field(default_factory=dict)
    database_snapshot_or_seed: str | None = None
    database_schema_digest: str | None = None
    database_content_digest: str | None = None
    workspace_template_digest: str | None = None
    workspace_template: dict[str, str] = Field(default_factory=dict)
    world_snapshot: dict[str, Any] | None = None
    tool_configuration: dict[str, Any] = Field(default_factory=dict)
    agent_configuration: list[dict[str, Any]] = Field(default_factory=list)
    environment_seed: int = 0
    generation_seed: int = 0

    @property
    def bundle_digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json", exclude={"generation_seed"}))

    @property
    def agent_config_digest(self) -> str:
        return canonical_digest(self.agent_configuration)

    @property
    def task_digest(self) -> str:
        return canonical_digest(self.task_source_snapshot)

    @property
    def tool_config_digest(self) -> str:
        return canonical_digest(self.tool_configuration)


def bundle_from_manifest_task(
    task: dict[str, Any],
    *,
    environment_seed: int = 0,
    generation_seed: int = 0,
) -> InitialStateBundle:
    raw = task.get("raw_task") if isinstance(task.get("raw_task"), dict) else task
    environment = raw.get("environment") if isinstance(raw.get("environment"), dict) else {}
    init_sql = environment.get("init_sql") or ""
    workspace_template = {
        "task.json": json.dumps(raw, indent=2, sort_keys=True),
        "init.sql": init_sql,
    }
    return InitialStateBundle(
        scenario=str(task.get("scenario") or raw.get("scenario") or "unknown"),
        task_id=str(task.get("task_id") or raw.get("task_id") or task.get("source_line")),
        task_source_snapshot=raw,
        environment_configuration=environment,
        database_snapshot_or_seed=canonical_digest(init_sql) if init_sql else None,
        database_schema_digest=canonical_digest(_schema_lines(init_sql)) if init_sql else None,
        database_content_digest=canonical_digest(_content_lines(init_sql)) if init_sql else None,
        workspace_template_digest=canonical_digest(workspace_template),
        workspace_template=workspace_template,
        tool_configuration={
            "metrics": raw.get("metrics", {}),
            "output": raw.get("output", {}),
        },
        agent_configuration=(
            list(raw.get("agents", [])) if isinstance(raw.get("agents"), list) else []
        ),
        environment_seed=environment_seed,
        generation_seed=generation_seed,
    )


def materialize_bundle_workspace(
    *,
    bundle: InitialStateBundle,
    workspace: Path,
) -> str:
    """Materialize a fresh branch workspace and return its initial digest."""

    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    for relative_name, content in bundle.workspace_template.items():
        target = workspace / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (workspace / "bundle.json").write_text(
        json.dumps(bundle.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return workspace_digest(workspace)


def workspace_digest(workspace: Path) -> str:
    files: list[dict[str, str]] = []
    for path in sorted(item for item in workspace.rglob("*") if item.is_file()):
        files.append(
            {
                "path": str(path.relative_to(workspace)),
                "content_digest": canonical_digest(path.read_text(encoding="utf-8")),
            }
        )
    return canonical_digest(files)


def _schema_lines(sql: str) -> list[str]:
    return [
        line.strip()
        for line in sql.splitlines()
        if line.strip().upper().startswith(("CREATE TABLE", "CREATE INDEX", "ALTER TABLE"))
    ]


def _content_lines(sql: str) -> list[str]:
    return [
        line.strip()
        for line in sql.splitlines()
        if line.strip().upper().startswith(("INSERT", "UPDATE", "DELETE", "COPY"))
    ]
