"""R6 Test 9: formal end-to-end evaluation requires a split audit (清单 P1-8).

``split_audit_path=None`` must abort a formal evaluation immediately, while
pilots may legitimately omit the artifact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smtr.marble.end_to_end_evaluation import run_end_to_end_evaluation


def _dummy_kwargs(*, experiment_mode: str):
    dummy = Path("/nonexistent/smtr_split_audit_probe")
    return dict(
        marble_root=dummy,
        dataset_manifest_path=dummy,
        split_manifest_path=dummy,
        split="test",
        candidate_manifest_path=dummy,
        memory_pool_path=dummy,
        checkpoint_full=dummy,
        methods=["smtr"],
        generation_seeds=[0, 1, 2, 3, 4],
        experiment_mode=experiment_mode,
        split_audit_path=None,
        output=dummy,
    )


class TestFormalSplitAuditRequired:
    def test_formal_without_split_audit_fails(self):
        with pytest.raises(
            ValueError, match="requires a split audit artifact"
        ):
            run_end_to_end_evaluation(**_dummy_kwargs(experiment_mode="formal"))

    def test_pilot_without_split_audit_is_allowed(self):
        # Pilots may omit the artifact: the call must pass the split-audit
        # gate and fail (if at all) later — never with the formal-only error.
        with pytest.raises(Exception) as exc_info:
            run_end_to_end_evaluation(**_dummy_kwargs(experiment_mode="pilot"))
        assert "split audit artifact" not in str(exc_info.value)
