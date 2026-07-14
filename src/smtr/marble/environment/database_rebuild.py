"""Sequential database rebuild harness for MARBLE database branches."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from smtr.marble.environment.database_fingerprint import (
    DatabaseLogicalFingerprint,
    fingerprint_initial_bundle,
)
from smtr.marble.environment.isolation import InitialStateBundle, materialize_bundle_workspace


class SequentialDatabaseRebuilder:
    def __init__(self, *, marble_root: Path = Path("/home/ecs-user/MARBLE")) -> None:
        self.marble_root = marble_root
        self.current_workspace: Path | None = None

    def materialize(
        self,
        *,
        initial_state_bundle: InitialStateBundle,
        branch_workspace: Path,
    ) -> DatabaseLogicalFingerprint:
        materialize_bundle_workspace(bundle=initial_state_bundle, workspace=branch_workspace)
        self.current_workspace = branch_workspace
        return fingerprint_initial_bundle(
            initial_state_bundle=initial_state_bundle,
            branch_workspace=branch_workspace,
        )

    def destroy(self) -> None:
        compose_dir = self.marble_root / "marble/environments/db_env_docker"
        if compose_dir.exists():
            subprocess.run(
                ("sudo", "docker", "compose", "down", "-v"),
                cwd=compose_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        if self.current_workspace and self.current_workspace.exists():
            shutil.rmtree(self.current_workspace)
        self.current_workspace = None
