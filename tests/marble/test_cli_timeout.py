from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from smtr.marble import cli
from smtr.marble.engine_process import DEFAULT_ENGINE_TIMEOUT_SECONDS


def test_collect_database_trajectories_cli_passes_engine_timeout(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    captured: dict[str, Any] = {}

    def fake_collect_database_trajectories(**kwargs):
        captured.update(kwargs)
        return {"attempted": 1, "valid": 0, "invalid": 1, "task_ids": ["19"], "index": "i"}

    monkeypatch.setattr(cli, "collect_database_trajectories", fake_collect_database_trajectories)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m smtr.marble.cli",
            "collect-database-trajectories",
            "--marble-root",
            "/home/ecs-user/MARBLE",
            "--dataset-manifest",
            str(tmp_path / "dataset.json"),
            "--split-manifest",
            str(tmp_path / "split.json"),
            "--split",
            "train",
            "--task-ids",
            "19",
            "--generation-seeds",
            "0",
            "--engine-timeout-seconds",
            "1800",
            "--output",
            str(tmp_path / "out"),
        ],
    )

    cli.main()

    assert captured["engine_timeout_seconds"] == 1800
    assert "engine_timeout_source" not in captured
    assert captured["task_ids"] == ["19"]
    assert '"invalid": 1' in capsys.readouterr().out


def test_collect_database_trajectories_cli_uses_default_timeout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_collect_database_trajectories(**kwargs):
        captured.update(kwargs)
        return {"attempted": 0, "valid": 0, "invalid": 0, "task_ids": [], "index": "i"}

    monkeypatch.setattr(cli, "collect_database_trajectories", fake_collect_database_trajectories)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m smtr.marble.cli",
            "collect-database-trajectories",
            "--dataset-manifest",
            str(tmp_path / "dataset.json"),
            "--split-manifest",
            str(tmp_path / "split.json"),
            "--split",
            "train",
            "--output",
            str(tmp_path / "out"),
        ],
    )

    cli.main()

    assert captured["engine_timeout_seconds"] == DEFAULT_ENGINE_TIMEOUT_SECONDS
    assert "engine_timeout_source" not in captured


@pytest.mark.parametrize(
    "removed_command",
    ["audit-real-database-mvp", "gate-database-trajectory"],
)
def test_removed_diagnostic_cli_commands_fail_fast(
    monkeypatch,
    removed_command: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["python -m smtr.marble.cli", removed_command])

    with pytest.raises(SystemExit):
        cli.main()
