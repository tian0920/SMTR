"""R6 Test 10: split audit digest binding (清单 P1-5/P1-7).

An audit artifact must only validate against the exact files it audited:
swapping the memory pool, dataset manifest, split manifest or checkpoint
after the audit must abort the evaluation.
"""

from __future__ import annotations

import json

import pytest

from smtr.evaluation.split_audit import SPLIT_AUDIT_SCHEMA_VERSION
from smtr.evaluation.split_audit_validation import validate_split_audit_artifact
from smtr.marble.runtime_visibility_audit import file_digest


def _write(path, content: str):
    path.write_text(content, encoding="utf-8")
    return path


def _build_artifacts(tmp_path):
    """Two disjoint artifact sets: A (the audited one) and B (the swap-in)."""
    files = {}
    for name, content_a, content_b in (
        ("dataset_manifest", '{"dataset": "a"}', '{"dataset": "b"}'),
        ("split_manifest", '{"splits": "a"}', '{"splits": "b"}'),
        ("memory_pool", '{"memory_id": "m_a"}', '{"memory_id": "m_b"}'),
        ("checkpoint", "checkpoint-bytes-a", "checkpoint-bytes-b"),
    ):
        a = _write(tmp_path / f"{name}_a.json", content_a)
        b = _write(tmp_path / f"{name}_b.json", content_b)
        files[name] = (a, b)
    return files


def _write_audit(tmp_path, files) -> "object":
    audit = {
        "schema_version": SPLIT_AUDIT_SCHEMA_VERSION,
        "split_integrity_passed": True,
        "calibration_split": "validation",
        "epsilon_selection_split": "validation",
        "dataset_manifest_digest": file_digest(files["dataset_manifest"][0]),
        "split_manifest_digest": file_digest(files["split_manifest"][0]),
        "memory_pool_digest": file_digest(files["memory_pool"][0]),
        "train_paired_records_digest": None,
        "validation_paired_records_digest": None,
        "test_paired_records_digest": None,
        "checkpoint_digest": file_digest(files["checkpoint"][0]),
    }
    audit_path = tmp_path / "split_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit_path


def _validate(audit_path, files, *, swapped: str | None = None):
    def pick(name: str):
        return files[name][1] if swapped == name else files[name][0]

    return validate_split_audit_artifact(
        split_audit_path=audit_path,
        dataset_manifest_path=pick("dataset_manifest"),
        split_manifest_path=pick("split_manifest"),
        memory_pool_path=pick("memory_pool"),
        checkpoint_path=pick("checkpoint"),
    )


class TestSplitAuditDigestBinding:
    def test_matching_artifacts_validate(self, tmp_path):
        files = _build_artifacts(tmp_path)
        audit_path = _write_audit(tmp_path, files)
        audit = _validate(audit_path, files)
        assert audit["split_integrity_passed"] is True

    @pytest.mark.parametrize(
        ("swapped", "expected_label"),
        [
            ("memory_pool", "memory pool"),
            ("dataset_manifest", "dataset manifest"),
            ("split_manifest", "split manifest"),
            ("checkpoint", "checkpoint"),
        ],
    )
    def test_digest_mismatch_aborts(self, tmp_path, swapped, expected_label):
        # The audit was computed over set A; evaluating against a swapped-in
        # file from set B must be rejected.
        files = _build_artifacts(tmp_path)
        audit_path = _write_audit(tmp_path, files)
        with pytest.raises(
            ValueError,
            match=f"split audit does not match current {expected_label}",
        ):
            _validate(audit_path, files, swapped=swapped)
