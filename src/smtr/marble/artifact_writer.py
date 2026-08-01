"""Standardized artifact writer for MARBLE experiment runs.

Provides atomic writes, digest computation, and structured output
directories for runs, pairs, and evaluation results.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from smtr.counterfactual.decision_points import canonical_digest


def write_artifact(path: Path, data: Any, *, indent: int = 2) -> str:
    """Atomically write a JSON artifact and return its digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=indent, sort_keys=True) + "\n"
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=path.stem,
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return canonical_digest(data)


def write_run_artifacts(
    *,
    run_dir: Path,
    run_identity: dict[str, Any],
    frozen_config: dict[str, Any] | None = None,
    visibility_audit: list[dict[str, Any]] | None = None,
    engine_result: dict[str, Any] | None = None,
    native_evaluator_result: dict[str, Any] | None = None,
    result_summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write all artifacts for a single engine run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    digests["run_identity"] = write_artifact(
        run_dir / "run_identity.json", run_identity,
    )
    if frozen_config is not None:
        digests["frozen_config"] = write_artifact(
            run_dir / "frozen_config.json", frozen_config,
        )
    if visibility_audit is not None:
        write_artifact(run_dir / "memory_visibility_audit.json", visibility_audit)
    if engine_result is not None:
        write_artifact(run_dir / "raw_marble_result.json", engine_result)
    if native_evaluator_result is not None:
        write_artifact(
            run_dir / "native_evaluator_result.json", native_evaluator_result,
        )
    if result_summary is not None:
        digests["result_summary"] = write_artifact(
            run_dir / "result_summary.json", result_summary,
        )
    return digests


def write_pair_artifacts(
    *,
    pair_dir: Path,
    pair_identity: dict[str, Any],
    controlled_variables: dict[str, Any],
    share_reference: dict[str, Any] | None = None,
    withhold_reference: dict[str, Any] | None = None,
    initial_state_comparison: dict[str, Any] | None = None,
    visibility_comparison: dict[str, Any] | None = None,
    paired_outcome: dict[str, Any] | None = None,
    validity_report: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write all artifacts for a paired share/withhold intervention."""
    pair_dir.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    digests["pair_identity"] = write_artifact(
        pair_dir / "pair_identity.json", pair_identity,
    )
    digests["controlled_variables"] = write_artifact(
        pair_dir / "controlled_variables.json", controlled_variables,
    )
    if share_reference is not None:
        write_artifact(pair_dir / "share_run_reference.json", share_reference)
    if withhold_reference is not None:
        write_artifact(pair_dir / "withhold_run_reference.json", withhold_reference)
    if initial_state_comparison is not None:
        write_artifact(
            pair_dir / "initial_state_comparison.json", initial_state_comparison,
        )
    if visibility_comparison is not None:
        write_artifact(
            pair_dir / "visibility_comparison.json", visibility_comparison,
        )
    if paired_outcome is not None:
        digests["paired_outcome"] = write_artifact(
            pair_dir / "paired_outcome.json", paired_outcome,
        )
    if validity_report is not None:
        digests["validity_report"] = write_artifact(
            pair_dir / "validity_report.json", validity_report,
        )
    return digests


def sanitize_artifact_data(data: Any) -> Any:
    """Redact sensitive fields from artifact data."""
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            upper = key.upper()
            if any(tok in upper for tok in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                sanitized[key] = "<redacted>" if value else "<empty>"
            else:
                sanitized[key] = sanitize_artifact_data(value)
        return sanitized
    if isinstance(data, list):
        return [sanitize_artifact_data(item) for item in data]
    return data
