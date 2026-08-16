"""Database rebuild harness for MARBLE database branches.

Provides two implementations:

- ``SequentialDatabaseRebuilder`` — original single-slot rebuilder that
  uses the default Docker compose project.  Kept for backward
  compatibility when ``--parallel 1``.
- ``ParallelDatabaseRebuilder`` — slot-aware rebuilder that acquires a
  Docker compose slot from a ``DockerSlotPool``, starts an isolated
  compose project with unique host ports, and tears it down after the
  run.  Used when ``--parallel > 1``.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from smtr.marble.environment.database_fingerprint import (
    DatabaseLogicalFingerprint,
    fingerprint_initial_bundle,
)
from smtr.marble.environment.docker_slot_pool import DockerSlot, DockerSlotPool
from smtr.marble.environment.isolation import InitialStateBundle, materialize_bundle_workspace


@dataclass(frozen=True)
class DatabaseCleanupResult:
    exit_code: int | None
    succeeded: bool
    failure_reason: str | None

    def to_json(self) -> dict[str, object]:
        return asdict(self)


class SequentialDatabaseRebuilder:
    """Original single-slot rebuilder (backward-compatible).

    Uses the default Docker compose project with no port overrides.
    Only one instance may be active at a time.
    """

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


class ParallelDatabaseRebuilder:
    """Slot-aware rebuilder for parallel MARBLE engine runs.

    Each instance acquires a ``DockerSlot`` from the pool on
    ``materialize()`` and releases it on ``destroy()``.  The slot's
    isolated Docker compose project runs on unique host ports so
    multiple instances can operate concurrently.
    """

    def __init__(
        self,
        slot_pool: DockerSlotPool,
        *,
        marble_root: Path | None = None,
    ) -> None:
        self.slot_pool = slot_pool
        self.marble_root = marble_root or slot_pool._marble_root
        self.slot: DockerSlot | None = None
        self.current_workspace: Path | None = None
        self.last_cleanup_result: DatabaseCleanupResult | None = None

    def materialize(
        self,
        *,
        initial_state_bundle: InitialStateBundle,
        branch_workspace: Path,
    ) -> DatabaseLogicalFingerprint:
        """Acquire a slot, start its Docker compose, materialize workspace."""
        self.slot = self.slot_pool.acquire()
        try:
            self.slot_pool.compose_up(self.slot)
        except Exception:
            self.slot_pool.release(self.slot)
            self.slot = None
            raise

        materialize_bundle_workspace(bundle=initial_state_bundle, workspace=branch_workspace)
        self.current_workspace = branch_workspace
        return fingerprint_initial_bundle(
            initial_state_bundle=initial_state_bundle,
            branch_workspace=branch_workspace,
        )

    def destroy(self, *, remove_workspace: bool = True) -> DatabaseCleanupResult:
        """Tear down the slot's compose project and release the slot."""
        exit_code: int | None = None
        failure_reason: str | None = None

        if self.slot is not None:
            ok = self.slot_pool.compose_down(self.slot)
            if not ok:
                exit_code = -1
                failure_reason = f"compose_down_failed: slot_{self.slot.slot_id}"
            else:
                exit_code = 0

            self.slot_pool.release(self.slot)
            self.slot = None
        else:
            failure_reason = "no_slot_acquired"

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

    @property
    def engine_env(self) -> dict[str, str]:
        """Environment variables the engine subprocess needs for this slot."""
        if self.slot is None:
            return {}
        return self.slot.engine_env
