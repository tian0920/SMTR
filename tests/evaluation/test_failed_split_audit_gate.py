"""R6 Test 11: a failed split audit blocks formal evaluation (清单 P1-7/P1-8).

When the bound audit records ``split_integrity_passed=false`` the formal
end-to-end evaluation must abort immediately — before any critic is loaded
and before any MARBLE episode runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smtr.evaluation.split_audit import SPLIT_AUDIT_SCHEMA_VERSION
from smtr.marble import end_to_end_evaluation


class _ExplodingCritic:
    @staticmethod
    def load(path):
        raise AssertionError(
            "critic load reached despite failed split audit — "
            "the gate must abort before any MARBLE work"
        )


def _write_failed_audit(tmp_path) -> Path:
    audit = {
        "schema_version": SPLIT_AUDIT_SCHEMA_VERSION,
        "split_integrity_passed": False,
        "error": "target_task_id leakage across splits: ['task_v1']",
        "calibration_split": "validation",
        "epsilon_selection_split": "validation",
    }
    audit_path = tmp_path / "split_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return audit_path


class TestFailedSplitAuditGate:
    def test_formal_evaluation_aborts_before_any_episode(
        self, tmp_path, monkeypatch
    ):
        # If the gate ever passes, the stub critic load would raise
        # AssertionError instead of the expected split-audit ValueError.
        monkeypatch.setattr(
            end_to_end_evaluation, "FourOutcomeTransferCritic", _ExplodingCritic)

        dummy = Path("/nonexistent/smtr_failed_audit_probe")
        with pytest.raises(
            ValueError, match="formal evaluation aborted: split audit failed"
        ):
            end_to_end_evaluation.run_end_to_end_evaluation(
                marble_root=dummy,
                dataset_manifest_path=dummy,
                split_manifest_path=dummy,
                split="test",
                candidate_manifest_path=dummy,
                memory_pool_path=dummy,
                checkpoint_full=dummy,
                methods=["smtr"],
                generation_seeds=[0, 1, 2, 3, 4],
                experiment_mode="formal",
                split_audit_path=_write_failed_audit(tmp_path),
                output=tmp_path / "output",
            )
        # No run workspace may have been created: zero MARBLE episodes ran.
        assert not (tmp_path / "output" / "runs").exists()
