"""Sequential database rebuild harness for MARBLE database branches."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from smtr.marble.environment.database_fingerprint import (
    DatabaseLogicalFingerprint,
    fingerprint_initial_bundle,
)
from smtr.marble.environment.isolation import InitialStateBundle, materialize_bundle_workspace


@dataclass(frozen=True)
class DatabaseCleanupResult:
    exit_code: int | None
    succeeded: bool
    failure_reason: str | None

    def to_json(self) -> dict[str, object]:
        return asdict(self)


class SequentialDatabaseRebuilder:
    def __init__(self, *, marble_root: Path = Path("/home/ecs-user/MARBLE")) -> None:
        self.marble_root = marble_root
        self.current_workspace: Path | None = None
        self.last_cleanup_result: DatabaseCleanupResult | None = None

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

    def destroy(self, *, remove_workspace: bool = True) -> DatabaseCleanupResult:
        compose_dir = self.marble_root / "marble/environments/db_env_docker"
        exit_code: int | None = None
        failure_reason: str | None = None
        if compose_dir.exists():
            try:
                completed = subprocess.run(
                    ("sudo", "docker", "compose", "down", "-v"),
                    cwd=compose_dir,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                exit_code = completed.returncode
                if completed.returncode != 0:
                    failure_reason = f"cleanup_exit_code={completed.returncode}"
            except Exception as exc:
                exit_code = -1
                failure_reason = f"cleanup_failed: {type(exc).__name__}: {exc}"
        else:
            failure_reason = f"compose_dir_not_found: {compose_dir}"
        result = DatabaseCleanupResult(
            exit_code=exit_code,
            succeeded=failure_reason is None,
            failure_reason=failure_reason,
        )
        self.last_cleanup_result = result
        if remove_workspace and self.current_workspace and self.current_workspace.exists():
            shutil.rmtree(self.current_workspace)
        if remove_workspace:
            self.current_workspace = None
        return result
