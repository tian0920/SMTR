"""Unit tests for DockerSlotPool — acquire/release/并发安全/API key 轮转.

These tests mock subprocess calls (docker compose) so they run without
Docker installed.  They validate the thread-safe slot management logic
of ``DockerSlotPool`` in isolation.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from smtr.marble.environment.docker_slot_pool import DockerSlot, DockerSlotPool, _make_slot


# ---------------------------------------------------------------------------
# DockerSlot port allocation
# ---------------------------------------------------------------------------


class TestDockerSlot:
    """Verify deterministic port allocation per slot."""

    def test_slot_0_ports(self) -> None:
        slot = _make_slot(0)
        assert slot.postgres_port == 15432
        assert slot.prometheus_port == 19090
        assert slot.node_exporter_port == 19100
        assert slot.pg_exporter_port == 19187

    def test_slot_1_ports(self) -> None:
        slot = _make_slot(1)
        assert slot.postgres_port == 16432
        assert slot.prometheus_port == 20090
        assert slot.node_exporter_port == 20100
        assert slot.pg_exporter_port == 20187

    def test_slot_7_ports(self) -> None:
        slot = _make_slot(7)
        assert slot.postgres_port == 15432 + 7 * 1000
        assert slot.prometheus_port == 19090 + 7 * 1000

    def test_compose_project_name(self) -> None:
        slot = _make_slot(3)
        assert slot.compose_project == "smtr_db_slot_3"

    def test_compose_env_contains_required_keys(self) -> None:
        slot = _make_slot(0)
        env = slot.compose_env
        assert env["POSTGRES_HOST_PORT"] == "15432"
        assert env["PROMETHEUS_HOST_PORT"] == "19090"
        assert env["NODE_EXPORTER_HOST_PORT"] == "19100"
        assert env["PG_EXPORTER_HOST_PORT"] == "19187"

    def test_engine_env_contains_port_vars(self) -> None:
        slot = _make_slot(2)
        env = slot.engine_env
        assert env["MARBLE_DB_PORT"] == str(15432 + 2000)
        assert env["MARBLE_PROM_PORT"] == str(19090 + 2000)

    def test_frozen_dataclass(self) -> None:
        slot = _make_slot(0)
        with pytest.raises(AttributeError):
            slot.slot_id = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DockerSlotPool basic acquire / release
# ---------------------------------------------------------------------------


class TestDockerSlotPoolBasic:
    """Acquire and release without actual Docker (subprocess mocked)."""

    @pytest.fixture()
    def pool(self, tmp_path: Path) -> DockerSlotPool:
        return DockerSlotPool(
            n_slots=3,
            marble_root=tmp_path,
            api_keys=["key-a", "key-b", "key-c"],
        )

    @patch.object(DockerSlotPool, "compose_up")
    def test_acquire_returns_unique_slots(
        self, mock_up: MagicMock, pool: DockerSlotPool
    ) -> None:
        slots = [pool.acquire(timeout=5) for _ in range(3)]
        ids = {s.slot_id for s in slots}
        assert len(ids) == 3, "each acquire should return a distinct slot"

    @patch.object(DockerSlotPool, "compose_up")
    def test_acquire_blocks_when_exhausted(
        self, mock_up: MagicMock, pool: DockerSlotPool
    ) -> None:
        # Acquire all 3 slots
        for _ in range(3):
            pool.acquire(timeout=5)

        # 4th acquire with short timeout should raise TimeoutError
        with pytest.raises(TimeoutError):
            pool.acquire(timeout=0.2)

    @patch.object(DockerSlotPool, "compose_up")
    def test_release_makes_slot_available_again(
        self, mock_up: MagicMock, pool: DockerSlotPool
    ) -> None:
        slots = [pool.acquire(timeout=5) for _ in range(3)]
        # Release one
        pool.release(slots[1])
        # Should now be able to acquire again
        new_slot = pool.acquire(timeout=5)
        assert new_slot.slot_id == slots[1].slot_id


# ---------------------------------------------------------------------------
# API key rotation
# ---------------------------------------------------------------------------


class TestApiKeyRotation:
    """Round-robin API key assignment across slots."""

    def test_no_keys_returns_none(self, tmp_path: Path) -> None:
        pool = DockerSlotPool(n_slots=2, marble_root=tmp_path, api_keys=[])
        slot = _make_slot(0)
        assert pool.get_api_key(slot) is None

    def test_single_key_always_same(self, tmp_path: Path) -> None:
        pool = DockerSlotPool(n_slots=2, marble_root=tmp_path, api_keys=["only-key"])
        assert pool.get_api_key(_make_slot(0)) == "only-key"
        assert pool.get_api_key(_make_slot(1)) == "only-key"

    def test_round_robin_rotation(self, tmp_path: Path) -> None:
        keys = ["k0", "k1", "k2"]
        pool = DockerSlotPool(n_slots=3, marble_root=tmp_path, api_keys=keys)
        # Each slot should get a key based on round-robin index
        results = {pool.get_api_key(_make_slot(i)) for i in range(3)}
        # All keys should be used
        assert results == set(keys)


# ---------------------------------------------------------------------------
# Concurrent acquire / release (threading stress test)
# ---------------------------------------------------------------------------


class TestConcurrentSafety:
    """Stress test: N threads compete for M slots."""

    @patch.object(DockerSlotPool, "compose_up")
    def test_concurrent_acquire_release(
        self, mock_up: MagicMock, tmp_path: Path
    ) -> None:
        n_slots = 2
        n_workers = 6
        pool = DockerSlotPool(n_slots=n_slots, marble_root=tmp_path)

        acquired_ids: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            slot = pool.acquire(timeout=30)
            with lock:
                acquired_ids.append(slot.slot_id)
            # Simulate work
            time.sleep(0.05)
            pool.release(slot)

        threads = [threading.Thread(target=worker) for _ in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        # All workers should have completed successfully
        assert len(acquired_ids) == n_workers
        # Only valid slot IDs should appear
        assert all(0 <= sid < n_slots for sid in acquired_ids)

    @patch.object(DockerSlotPool, "compose_up")
    def test_no_double_allocation(
        self, mock_up: MagicMock, tmp_path: Path
    ) -> None:
        """At any point in time, no two threads hold the same slot."""
        n_slots = 3
        n_workers = 9
        pool = DockerSlotPool(n_slots=n_slots, marble_root=tmp_path)

        active_slots: set[int] = set()
        violation = False
        lock = threading.Lock()

        def worker() -> None:
            nonlocal violation
            slot = pool.acquire(timeout=30)
            with lock:
                if slot.slot_id in active_slots:
                    violation = True
                active_slots.add(slot.slot_id)
            # Simulate work
            time.sleep(0.05)
            with lock:
                active_slots.discard(slot.slot_id)
            pool.release(slot)

        threads = [threading.Thread(target=worker) for _ in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not violation, "same slot was allocated to two threads simultaneously"


# ---------------------------------------------------------------------------
# shutdown_all
# ---------------------------------------------------------------------------


class TestShutdown:
    """Verify shutdown_all cleans up active slots."""

    @patch.object(DockerSlotPool, "compose_down", return_value=True)
    @patch.object(DockerSlotPool, "compose_up")
    def test_shutdown_downs_all_slots(
        self, mock_up: MagicMock, mock_down: MagicMock, tmp_path: Path
    ) -> None:
        pool = DockerSlotPool(n_slots=2, marble_root=tmp_path)
        pool.acquire(timeout=5)
        pool.acquire(timeout=5)
        pool.shutdown_all()
        # compose_down should have been called for each slot
        assert mock_down.call_count == 2
