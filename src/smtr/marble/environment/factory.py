"""Pilot MARBLE environment factory."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from smtr.marble.environment.isolation import (
    InitialStateBundle,
    materialize_bundle_workspace,
    workspace_digest,
)


class FilesystemMarbleEnvironmentInstance(BaseModel):
    """Isolated filesystem materialization of a MARBLE task initial state."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario: str
    task_id: str
    workspace: Path
    initial_digest: str
    closed: bool = False

    def initial_state_digest(self) -> str:
        return self.initial_digest

    def run(self, *, agent_input: object, generation_seed: int) -> object:
        run_record = {
            "agent_input": agent_input,
            "generation_seed": generation_seed,
            "workspace_digest_before_run": workspace_digest(self.workspace),
        }
        (self.workspace / "run_input.json").write_text(
            json.dumps(run_record, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return run_record

    def close(self) -> None:
        self.closed = True


class DatabaseFilesystemEnvironmentFactory:
    """Database pilot factory using independent branch workspaces."""

    scenario = "database"

    def create_isolated(
        self,
        *,
        task: object,
        initial_state_bundle: object,
        branch_id: str,
        workspace: str,
    ) -> FilesystemMarbleEnvironmentInstance:
        if not isinstance(initial_state_bundle, InitialStateBundle):
            raise TypeError("initial_state_bundle must be an InitialStateBundle")
        if initial_state_bundle.scenario != self.scenario:
            raise ValueError("DatabaseFilesystemEnvironmentFactory only supports database")
        branch_workspace = Path(workspace)
        initial_digest = materialize_bundle_workspace(
            bundle=initial_state_bundle,
            workspace=branch_workspace,
        )
        (branch_workspace / "branch_id.txt").write_text(branch_id + "\n", encoding="utf-8")
        return FilesystemMarbleEnvironmentInstance(
            scenario=initial_state_bundle.scenario,
            task_id=initial_state_bundle.task_id,
            workspace=branch_workspace,
            initial_digest=initial_digest,
        )


def factory_for_scenario(scenario: str) -> DatabaseFilesystemEnvironmentFactory:
    if scenario != "database":
        raise ValueError(f"unsupported MARBLE pilot scenario: {scenario}")
    return DatabaseFilesystemEnvironmentFactory()
