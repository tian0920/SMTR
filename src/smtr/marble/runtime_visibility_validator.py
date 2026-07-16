"""Runtime visibility validator for MARBLE engine runs.

Validates that the actual memory exposure at the LLM call boundary
matches the expected behaviour for each method (B0, Share, Withhold,
AllShare, SMTR) and for paired branch comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from smtr.marble.runtime_visibility_audit import (
    RuntimeVisibilityRecord,
    read_runtime_visibility_records,
)


@dataclass(frozen=True)
class RuntimeVisibilityValidation:
    """Result of validating runtime visibility records."""

    visibility_verified: bool
    invalid_reason: str | None
    record_count: int
    agents_observed: tuple[str, ...]
    violations: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "visibility_verified": self.visibility_verified,
            "invalid_reason": self.invalid_reason,
            "record_count": self.record_count,
            "agents_observed": list(self.agents_observed),
            "violations": list(self.violations),
        }


class RuntimeVisibilityValidator:
    """Validate runtime visibility records against method-specific rules."""

    def validate(
        self,
        *,
        method: str,
        branch: str,
        receiver_agent_ids: Sequence[str],
        expected_memory_ids: Sequence[str],
        records: Sequence[RuntimeVisibilityRecord],
        candidate_memory_ids: Sequence[str] | None = None,
        selected_memory_ids: Sequence[str] | None = None,
        rejected_memory_ids: Sequence[str] | None = None,
    ) -> RuntimeVisibilityValidation:
        """Validate records for a single branch/method.

        Parameters
        ----------
        method:
            One of "b0", "share", "withhold", "all_share", "smtr", "pair_share",
            "pair_withhold".
        branch:
            Branch name (e.g. "b0", "share", "withhold").
        receiver_agent_ids:
            Agent IDs that should receive memory in share/all_share/smtr.
        expected_memory_ids:
            Memory IDs that should be visible to receiver.
        records:
            The runtime visibility records to validate.
        candidate_memory_ids:
            For withhold: the candidate that must NOT be visible.
        selected_memory_ids:
            For SMTR: the router-selected set.
        rejected_memory_ids:
            For SMTR: the router-rejected set.
        """
        if not records:
            return RuntimeVisibilityValidation(
                visibility_verified=False,
                invalid_reason="no_runtime_visibility_records",
                record_count=0,
                agents_observed=(),
                violations=("no_records",),
            )

        agents_observed = tuple(sorted({r.agent_id for r in records}))
        violations: list[str] = []

        if method in ("b0", "b0_no_memory"):
            violations.extend(self._validate_b0(records))
        elif method == "withhold":
            cands = list(candidate_memory_ids or expected_memory_ids)
            violations.extend(self._validate_withhold(records, cands))
        elif method == "share":
            violations.extend(
                self._validate_share(
                    records, receiver_agent_ids, expected_memory_ids
                )
            )
        elif method in ("all_share", "allshare"):
            violations.extend(
                self._validate_all_share(
                    records, receiver_agent_ids, expected_memory_ids
                )
            )
        elif method == "smtr":
            violations.extend(
                self._validate_smtr(
                    records,
                    receiver_agent_ids,
                    selected_memory_ids or expected_memory_ids,
                    rejected_memory_ids or [],
                )
            )
        elif method == "pair_share":
            violations.extend(
                self._validate_share(
                    records, receiver_agent_ids, expected_memory_ids
                )
            )
        elif method == "pair_withhold":
            cands = list(candidate_memory_ids or expected_memory_ids)
            violations.extend(self._validate_withhold(records, cands))
        else:
            violations.append(f"unknown_method:{method}")

        verified = len(violations) == 0
        reason = ",".join(violations) if violations else None
        return RuntimeVisibilityValidation(
            visibility_verified=verified,
            invalid_reason=reason,
            record_count=len(records),
            agents_observed=agents_observed,
            violations=tuple(violations),
        )

    # ------------------------------------------------------------------
    # Method-specific validators
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_b0(records: Sequence[RuntimeVisibilityRecord]) -> list[str]:
        """B0: no external SMTR memory should be visible to any agent."""
        violations: list[str] = []
        for rec in records:
            if rec.visible_memory_ids:
                violations.append("b0_external_memory_visible")
                break
        return violations

    @staticmethod
    def _validate_withhold(
        records: Sequence[RuntimeVisibilityRecord],
        candidate_memory_ids: Sequence[str],
    ) -> list[str]:
        """Withhold: candidate memories must not appear for any agent."""
        violations: list[str] = []
        candidate_set = set(candidate_memory_ids)
        if not candidate_set:
            return violations
        for rec in records:
            leaked = set(rec.visible_memory_ids) & candidate_set
            if leaked:
                violations.append("withhold_candidate_memory_leaked")
                break
        return violations

    @staticmethod
    def _validate_share(
        records: Sequence[RuntimeVisibilityRecord],
        receiver_agent_ids: Sequence[str],
        expected_memory_ids: Sequence[str],
    ) -> list[str]:
        """Share: receiver must see candidate; non-receivers must not."""
        violations: list[str] = []
        receiver_set = set(receiver_agent_ids)
        expected_set = set(expected_memory_ids)
        if not expected_set:
            return violations

        # Check receiver saw the memory at least once
        receiver_observed = False
        receiver_saw_memory = False
        for rec in records:
            if rec.agent_id in receiver_set:
                receiver_observed = True
                if set(rec.visible_memory_ids) & expected_set:
                    receiver_saw_memory = True
            else:
                # Non-receiver must not see the memory
                leaked = set(rec.visible_memory_ids) & expected_set
                if leaked:
                    violations.append("share_non_receiver_exposure")
                    break

        if not receiver_observed:
            violations.append("receiver_not_observed_at_model_boundary")
        elif not receiver_saw_memory:
            violations.append("share_receiver_did_not_see_candidate")

        return violations

    @staticmethod
    def _validate_all_share(
        records: Sequence[RuntimeVisibilityRecord],
        receiver_agent_ids: Sequence[str],
        expected_memory_ids: Sequence[str],
    ) -> list[str]:
        """AllShare: receiver sees all expected; non-receivers see none."""
        violations: list[str] = []
        receiver_set = set(receiver_agent_ids)
        expected_set = set(expected_memory_ids)
        if not expected_set:
            return violations

        receiver_saw_all = False
        for rec in records:
            if rec.agent_id in receiver_set:
                if expected_set.issubset(set(rec.visible_memory_ids)):
                    receiver_saw_all = True
            else:
                leaked = set(rec.visible_memory_ids) & expected_set
                if leaked:
                    violations.append("allshare_non_receiver_exposure")
                    break

        if not receiver_saw_all:
            violations.append("allshare_receiver_did_not_see_complete_set")

        return violations

    @staticmethod
    def _validate_smtr(
        records: Sequence[RuntimeVisibilityRecord],
        receiver_agent_ids: Sequence[str],
        selected_memory_ids: Sequence[str],
        rejected_memory_ids: Sequence[str],
    ) -> list[str]:
        """SMTR: receiver sees only selected set; non-receivers see none."""
        violations: list[str] = []
        receiver_set = set(receiver_agent_ids)
        selected_set = set(selected_memory_ids)
        rejected_set = set(rejected_memory_ids)

        receiver_observed = False
        for rec in records:
            if rec.agent_id in receiver_set:
                receiver_observed = True
                visible = set(rec.visible_memory_ids)
                # Check no rejected memory leaked
                if visible & rejected_set:
                    violations.append("smtr_rejected_memory_visible")
                    break
                # Check selected set is present (union across all turns)
            else:
                if set(rec.visible_memory_ids) & (selected_set | rejected_set):
                    violations.append("smtr_non_receiver_exposure")
                    break

        if not receiver_observed and selected_set:
            violations.append("smtr_receiver_not_observed")

        # Check selected set completeness (union across all receiver records)
        if not violations:
            union_visible: set[str] = set()
            for rec in records:
                if rec.agent_id in receiver_set:
                    union_visible.update(rec.visible_memory_ids)
            if selected_set and not selected_set.issubset(union_visible):
                violations.append("smtr_selected_set_incomplete")

        return violations


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def validate_runtime_visibility_from_path(
    *,
    method: str,
    branch: str,
    receiver_agent_ids: Sequence[str],
    expected_memory_ids: Sequence[str],
    audit_path: Path,
    candidate_memory_ids: Sequence[str] | None = None,
    selected_memory_ids: Sequence[str] | None = None,
    rejected_memory_ids: Sequence[str] | None = None,
) -> RuntimeVisibilityValidation:
    """Load records from a JSONL path and validate."""
    if not audit_path.exists():
        return RuntimeVisibilityValidation(
            visibility_verified=False,
            invalid_reason="runtime_visibility_jsonl_missing",
            record_count=0,
            agents_observed=(),
            violations=("jsonl_missing",),
        )
    records = read_runtime_visibility_records(audit_path)
    if not records:
        return RuntimeVisibilityValidation(
            visibility_verified=False,
            invalid_reason="runtime_visibility_jsonl_empty",
            record_count=0,
            agents_observed=(),
            violations=("jsonl_empty",),
        )
    return RuntimeVisibilityValidator().validate(
        method=method,
        branch=branch,
        receiver_agent_ids=receiver_agent_ids,
        expected_memory_ids=expected_memory_ids,
        records=records,
        candidate_memory_ids=candidate_memory_ids,
        selected_memory_ids=selected_memory_ids,
        rejected_memory_ids=rejected_memory_ids,
    )


def validate_pair_runtime_visibility(
    *,
    share_audit_path: Path,
    withhold_audit_path: Path,
    receiver_agent_ids: Sequence[str],
    candidate_memory_ids: Sequence[str],
) -> dict:
    """Validate both branches of a paired run.

    Returns a dict with:
    - share_runtime_visibility_verified
    - withhold_runtime_visibility_verified
    - pair_runtime_visibility_verified
    - share_validation, withhold_validation
    """
    share_val = validate_runtime_visibility_from_path(
        method="pair_share",
        branch="share",
        receiver_agent_ids=receiver_agent_ids,
        expected_memory_ids=candidate_memory_ids,
        audit_path=share_audit_path,
        candidate_memory_ids=candidate_memory_ids,
    )
    withhold_val = validate_runtime_visibility_from_path(
        method="pair_withhold",
        branch="withhold",
        receiver_agent_ids=receiver_agent_ids,
        expected_memory_ids=candidate_memory_ids,
        audit_path=withhold_audit_path,
        candidate_memory_ids=candidate_memory_ids,
    )
    pair_verified = (
        share_val.visibility_verified and withhold_val.visibility_verified
    )
    return {
        "share_runtime_visibility_verified": share_val.visibility_verified,
        "withhold_runtime_visibility_verified": withhold_val.visibility_verified,
        "pair_runtime_visibility_verified": pair_verified,
        "share_validation": share_val.to_dict(),
        "withhold_validation": withhold_val.to_dict(),
    }
