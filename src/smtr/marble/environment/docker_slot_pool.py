"""Parallel Docker compose slot pool for MARBLE database tasks.

Each slot owns an independent Docker compose project with its own set
of host ports, allowing multiple MARBLE engine runs to execute
concurrently without port conflicts.

Port allocation per slot (slot_id = 0..N-1):

| Service       | Host port formula           | Slot 0  | Slot 1  |
|---------------|-----------------------------|---------|---------|
| Postgres      | 15432 + slot_id * 1000      | 15432   | 16432   |
| Prometheus    | 19090 + slot_id * 1000      | 19090   | 20090   |
| Node Exporter | 19100 + slot_id * 1000      | 19100   | 20100   |
| PG Exporter   | 19187 + slot_id * 1000      | 19187   | 20187   |
"""

from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# Base port offsets for each service
_PG_BASE = 15432
_PROM_BASE = 19090
_NODE_EXPORTER_BASE = 19100
_PG_EXPORTER_BASE = 19187
_PORT_STRIDE = 1000


@dataclass(frozen=True)
class DockerSlot:
    """One isolated Docker compose slot with unique host ports."""

    slot_id: int
    postgres_port: int
    prometheus_port: int
    node_exporter_port: int
    pg_exporter_port: int
    compose_project: str

    @property
    def compose_env(self) -> dict[str, str]:
        """Environment variables to pass to ``docker compose up``."""
        return {
            "POSTGRES_HOST_PORT": str(self.postgres_port),
            "PROMETHEUS_HOST_PORT": str(self.prometheus_port),
            "NODE_EXPORTER_HOST_PORT": str(self.node_exporter_port),
            "PG_EXPORTER_HOST_PORT": str(self.pg_exporter_port),
        }

    @property
    def engine_env(self) -> dict[str, str]:
        """Environment variables the MARBLE engine subprocess needs."""
        return {
            "MARBLE_DB_PORT": str(self.postgres_port),
            "MARBLE_PROM_PORT": str(self.prometheus_port),
        }


def _make_slot(slot_id: int) -> DockerSlot:
    return DockerSlot(
        slot_id=slot_id,
        postgres_port=_PG_BASE + slot_id * _PORT_STRIDE,
        prometheus_port=_PROM_BASE + slot_id * _PORT_STRIDE,
        node_exporter_port=_NODE_EXPORTER_BASE + slot_id * _PORT_STRIDE,
        pg_exporter_port=_PG_EXPORTER_BASE + slot_id * _PORT_STRIDE,
        compose_project=f"smtr_db_slot_{slot_id}",
    )


class DockerSlotPool:
    """Thread-safe pool of Docker compose slots.

    Parameters
    ----------
    n_slots:
        Number of parallel slots to create.
    marble_root:
        Path to the MARBLE repository root.
    api_keys:
        Optional list of LLM API keys.  Keys are assigned round-robin
        across slots.  When the list is empty the caller's default key
        is used.
    """

    def __init__(
        self,
        n_slots: int,
        marble_root: Path,
        api_keys: Sequence[str] = (),
    ) -> None:
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {n_slots}")
        self._n_slots = n_slots
        self._marble_root = marble_root
        self._api_keys = list(api_keys)
        self._compose_dir = marble_root / "marble/environments/db_env_docker"

        # Semaphore + condition for slot availability
        self._lock = threading.Lock()
        self._available: list[DockerSlot] = [_make_slot(i) for i in range(n_slots)]
        self._cond = threading.Condition(self._lock)
        self._in_use: set[int] = set()

    @property
    def n_slots(self) -> int:
        return self._n_slots

    def acquire(self, timeout: float | None = None) -> DockerSlot:
        """Block until a slot becomes available and return it.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait.  ``None`` means wait forever.

        Raises
        ------
        TimeoutError
            If no slot becomes available within *timeout* seconds.
        """
        with self._cond:
            if not self._cond.wait_for(lambda: bool(self._available), timeout=timeout):
                raise TimeoutError(
                    f"No Docker slot available within {timeout}s "
                    f"({len(self._in_use)}/{self._n_slots} in use)"
                )
            slot = self._available.pop()
            self._in_use.add(slot.slot_id)
        return slot

    def release(self, slot: DockerSlot) -> None:
        """Return a slot to the pool after its compose project is torn down."""
        with self._cond:
            if slot.slot_id in self._in_use:
                self._in_use.discard(slot.slot_id)
                self._available.append(slot)
                self._cond.notify()

    def get_api_key(self, slot: DockerSlot) -> str | None:
        """Return the API key assigned to *slot* (round-robin), or None."""
        if not self._api_keys:
            return None
        return self._api_keys[slot.slot_id % len(self._api_keys)]

    # ------------------------------------------------------------------
    # Docker compose lifecycle helpers
    # ------------------------------------------------------------------

    def compose_up(self, slot: DockerSlot, *, timeout: int = 120) -> None:
        """Start the Docker compose project for *slot*."""
        if not self._compose_dir.exists():
            raise FileNotFoundError(f"compose dir not found: {self._compose_dir}")
        env = {**_current_env(), **slot.compose_env}
        cmd = [
            "sudo", "-E", "docker", "compose",
            "-p", slot.compose_project,
            "-f", str(self._compose_dir / "docker-compose.yml"),
            "up", "-d", "--remove-orphans",
        ]
        result = subprocess.run(
            cmd,
            cwd=str(self._compose_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"compose up failed (slot {slot.slot_id}): "
                f"exit={result.returncode}\nstderr={result.stderr}"
            )
        logger.info("compose up: slot %d (pg=%d)", slot.slot_id, slot.postgres_port)

    def compose_down(
        self, slot: DockerSlot, *, remove_workspace: bool = False, timeout: int = 60
    ) -> bool:
        """Tear down the Docker compose project for *slot*.

        Returns True on success.
        """
        if not self._compose_dir.exists():
            return True  # nothing to tear down
        cmd = [
            "sudo", "-E", "docker", "compose",
            "-p", slot.compose_project,
            "-f", str(self._compose_dir / "docker-compose.yml"),
            "down", "-v",
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self._compose_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            ok = result.returncode == 0
            if not ok:
                logger.warning(
                    "compose down failed (slot %d): exit=%d stderr=%s",
                    slot.slot_id, result.returncode, result.stderr,
                )
            return ok
        except subprocess.TimeoutExpired:
            logger.warning("compose down timed out (slot %d)", slot.slot_id)
            return False

    def shutdown_all(self) -> None:
        """Tear down every slot's compose project (best-effort)."""
        for slot_id in range(self._n_slots):
            slot = _make_slot(slot_id)
            self.compose_down(slot)


def _current_env() -> dict[str, str]:
    """Snapshot of os.environ (avoids importing os at module level)."""
    import os
    return dict(os.environ)
