"""Validation of persisted split-audit artifacts (R6 清单 P1-7).

A formal end-to-end evaluation must not trust an arbitrary JSON file: the
audit artifact is re-verified against the artifacts the evaluation is about
to consume. Every bound file must still match the digest recorded at audit
time, the audit must have passed, and calibration / epsilon selection must
have happened on the validation split — otherwise the evaluation aborts
before any MARBLE episode runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smtr.evaluation.split_audit import SPLIT_AUDIT_SCHEMA_VERSION
from smtr.marble.runtime_visibility_audit import file_digest


def validate_split_audit_artifact(
    *,
    split_audit_path: Path,
    dataset_manifest_path: Path,
    split_manifest_path: Path,
    memory_pool_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Verify a split-audit artifact against the current evaluation inputs.

    Returns the parsed audit dict on success.

    Raises:
        ValueError: when the artifact has an unsupported schema version,
            the recorded audit failed, calibration/epsilon selection used a
            non-validation split, or any recorded digest no longer matches
            the file the evaluation is about to consume.
    """
    audit = json.loads(Path(split_audit_path).read_text(encoding="utf-8"))

    if audit.get("schema_version") != SPLIT_AUDIT_SCHEMA_VERSION:
        raise ValueError(
            "split audit artifact has unsupported schema_version: "
            f"{audit.get('schema_version')!r} "
            f"(expected {SPLIT_AUDIT_SCHEMA_VERSION!r})"
        )

    if not audit.get("split_integrity_passed"):
        raise ValueError("formal evaluation aborted: split audit failed")

    if audit.get("calibration_split") != "validation":
        raise ValueError(
            "split audit calibration used a non-validation split: "
            f"calibration_split={audit.get('calibration_split')!r}"
        )

    if audit.get("epsilon_selection_split") != "validation":
        raise ValueError(
            "split audit epsilon selection used a non-validation split: "
            f"epsilon_selection_split={audit.get('epsilon_selection_split')!r}"
        )

    # Re-compute the digests of the artifacts this evaluation will consume;
    # any mismatch means the audit does not describe the current inputs.
    bindings = (
        ("dataset_manifest_digest", dataset_manifest_path, "dataset manifest"),
        ("split_manifest_digest", split_manifest_path, "split manifest"),
        ("memory_pool_digest", memory_pool_path, "memory pool"),
        ("checkpoint_digest", checkpoint_path, "checkpoint"),
    )
    for digest_key, path, label in bindings:
        current_digest = file_digest(Path(path))
        if audit.get(digest_key) != current_digest:
            raise ValueError(f"split audit does not match current {label}")

    return audit
