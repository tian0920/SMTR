"""Tests for CLI pipeline commands."""

from __future__ import annotations

import subprocess
import sys


def test_extract_command_not_placeholder():
    """extract-database-memories must not be a print-only placeholder."""
    result = subprocess.run(
        [sys.executable, "-m", "smtr.marble.cli", "extract-database-memories", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--trajectory-index" in result.stdout
    assert "--split-manifest" in result.stdout
    assert "--min-actions" in result.stdout


def test_build_candidates_command_not_placeholder():
    """build-database-candidates must not be a print-only placeholder."""
    result = subprocess.run(
        [sys.executable, "-m", "smtr.marble.cli", "build-database-candidates", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--split" in result.stdout
    assert "--memory-pool" in result.stdout


def test_generate_paired_records_has_marble_root():
    """generate-database-paired-records must accept --marble-root and --split."""
    result = subprocess.run(
        [sys.executable, "-m", "smtr.marble.cli", "generate-database-paired-records", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--marble-root" in result.stdout
    assert "--split" in result.stdout


def test_no_placeholder_commands():
    """No formal command should be a print-only placeholder."""
    result = subprocess.run(
        [sys.executable, "-m", "smtr.marble.cli", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    # All formal commands should be listed
    for cmd in [
        "inspect-dataset", "create-splits", "collect-database-trajectories",
        "extract-database-memories", "build-database-candidates",
        "generate-database-paired-records", "train-critic",
        "run-paired-decision-evaluation", "run-marble-evaluation",
        "integrity-audit",
    ]:
        assert cmd in result.stdout
