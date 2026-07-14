import sys
from pathlib import Path

from smtr.marble.engine_process import run_marble_engine_process


def test_engine_process_records_exit_and_digests(tmp_path: Path) -> None:
    marble_root = tmp_path / "MARBLE"
    (marble_root / "marble").mkdir(parents=True)
    main = marble_root / "marble/main.py"
    main.write_text("import sys\nprint('hello')\nsys.exit(3)\n", encoding="utf-8")
    (marble_root / "pyproject.toml").write_text('version = "0.0.test"\n', encoding="utf-8")

    result = run_marble_engine_process(
        marble_root=marble_root,
        config_path=tmp_path / "config.yaml",
        raw_result_path=tmp_path / "missing.jsonl",
        timeout_seconds=5,
    )

    assert result.command[0] == sys.executable
    assert result.exit_code == 3
    assert result.stdout_digest
    assert result.stderr_digest
    assert result.real_engine_executed is False


def test_engine_timeout_marks_not_executed(tmp_path: Path) -> None:
    marble_root = tmp_path / "MARBLE"
    (marble_root / "marble").mkdir(parents=True)
    (marble_root / "marble/main.py").write_text(
        "import time\ntime.sleep(5)\n",
        encoding="utf-8",
    )

    result = run_marble_engine_process(
        marble_root=marble_root,
        config_path=tmp_path / "config.yaml",
        raw_result_path=tmp_path / "missing.jsonl",
        timeout_seconds=1,
    )

    assert result.timed_out is True
    assert result.real_engine_executed is False
