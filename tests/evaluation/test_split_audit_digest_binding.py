"""Split-audit artifact digest binding against the v3 artifact schema.

An audit artifact must only validate against the exact files it audited:
swapping the memory pool, dataset manifest, split manifest, candidate
manifest or any bound checkpoint after the audit must abort the
evaluation (清单 P0-1/P0-2, 3.8/3.9).
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
        (
            "candidate_manifest",
            '{"candidates": "a"}',
            '{"candidates": "b"}',
        ),
        ("checkpoint_full", "checkpoint-bytes-a", "checkpoint-bytes-b"),
    ):
        a = _write(tmp_path / f"{name}_a.json", content_a)
        b = _write(tmp_path / f"{name}_b.json", content_b)
        files[name] = (a, b)
    return files


def _write_audit(tmp_path, files) -> "object":
    audit = {
        "schema_version": SPLIT_AUDIT_SCHEMA_VERSION,
        "split_integrity_passed": True,
        "legacy_schema_used": False,
        "calibration_split": "validation",
        "epsilon_selection_split": "validation",
        "dataset_manifest_digest": file_digest(files["dataset_manifest"][0]),
        "split_manifest_digest": file_digest(files["split_manifest"][0]),
        "memory_pool_digest": file_digest(files["memory_pool"][0]),
        "test_candidate_manifest_digest": file_digest(
            files["candidate_manifest"][0]
        ),
        "train_paired_records_digest": None,
        "validation_paired_records_digest": None,
        "test_paired_records_digest": None,
        "checkpoint_digests": {
            "full": file_digest(files["checkpoint_full"][0]),
        },
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
        candidate_manifest_path=pick("candidate_manifest"),
        checkpoint_paths={"full": pick("checkpoint_full")},
        enabled_methods=["smtr"],
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
            ("candidate_manifest", "candidate manifest"),
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

    def test_checkpoint_digest_mismatch_aborts(self, tmp_path):
        files = _build_artifacts(tmp_path)
        audit_path = _write_audit(tmp_path, files)
        with pytest.raises(
            ValueError, match="checkpoint digest mismatch: role='full'"
        ):
            _validate(audit_path, files, swapped="checkpoint_full")

    def test_missing_checkpoint_digest_map_aborts(self, tmp_path):
        files = _build_artifacts(tmp_path)
        audit_path = _write_audit(tmp_path, files)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit.pop("checkpoint_digests")
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        with pytest.raises(
            ValueError, match="split audit has no checkpoint digest map"
        ):
            _validate(audit_path, files)

    def test_unbound_required_role_aborts(self, tmp_path):
        # The audit only bound the full checkpoint; enabling a method that
        # needs the global_transfer role must abort.
        files = _build_artifacts(tmp_path)
        audit_path = _write_audit(tmp_path, files)
        with pytest.raises(
            ValueError,
            match="split audit is not bound to checkpoint role "
            "'global_transfer'",
        ):
            validate_split_audit_artifact(
                split_audit_path=audit_path,
                dataset_manifest_path=files["dataset_manifest"][0],
                split_manifest_path=files["split_manifest"][0],
                memory_pool_path=files["memory_pool"][0],
                candidate_manifest_path=files["candidate_manifest"][0],
                checkpoint_paths={
                    "full": files["checkpoint_full"][0],
                    "global_transfer": files["checkpoint_full"][0],
                },
                enabled_methods=["smtr", "global_transfer_critic"],
            )
