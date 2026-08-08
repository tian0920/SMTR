"""Budget selection never reads outcomes (清单 Shared-Control 第12/13章).

Outcome fields, critic predictions and adaptive signals are invisible
to budget sampling; tampered provenance flags must be caught by audit.
"""

from __future__ import annotations

from smtr.marble.budget_sampling import (
    audit_budget_manifests,
    build_budgeted_candidate_manifest,
)
from smtr.marble.real_data import (
    CandidateEntry,
    CandidateRecord,
    DatabaseCandidateManifest,
)


def _record(memory_id: str, rank: int, **extra) -> CandidateRecord:
    return CandidateRecord(
        memory_id=memory_id,
        writer_agent_id=f"w_{memory_id}",
        writer_role="planner",
        receiver_role="executor",
        match_type="compatible",
        rank=rank,
        score=0.9 - 0.1 * (rank - 1),
        **extra,
    )


def _parent() -> DatabaseCandidateManifest:
    return DatabaseCandidateManifest(
        target_split="train",
        candidates=[
            CandidateEntry(
                task_id="t1",
                receiver_agent_id="r1",
                receiver_role="executor",
                candidate_records=[
                    _record(f"m{i}", rank=i + 1) for i in range(8)
                ],
            )
        ],
    )


def test_injected_outcome_fields_are_ignored():
    """CandidateRecord is frozen and drops unknown outcome fields."""
    tainted = _record(
        "mX",
        rank=1,
        team_success=True,
        share_outcome=1.0,
        critic_prediction=0.99,
        adaptive_signal=True,
    )
    clean = _record("mX", rank=1)
    assert tainted == clean
    assert not hasattr(tainted, "team_success")


def test_budget_selection_is_invariant_under_outcome_injection():
    parent = _parent()
    tainted_parent = parent.model_copy(
        update={
            "candidates": [
                parent.candidates[0].model_copy(
                    update={
                        "candidate_records": [
                            _record(
                                rec.memory_id,
                                rank=rec.rank,
                                team_success=True,
                                tau_estimate=0.7,
                            )
                            for rec in parent.candidates[0].candidate_records
                        ]
                    }
                )
            ]
        }
    )
    clean = build_budgeted_candidate_manifest(
        parent_manifest=parent, budget_fraction=0.5
    )
    tainted = build_budgeted_candidate_manifest(
        parent_manifest=tainted_parent, budget_fraction=0.5
    )
    assert clean.model_dump(mode="json") == tainted.model_dump(mode="json")


def test_provenance_flags_default_to_clean():
    manifest = build_budgeted_candidate_manifest(
        parent_manifest=_parent(), budget_fraction=0.5
    )
    meta = manifest.budget_metadata
    assert meta is not None
    assert meta.outcome_fields_used is False
    assert meta.critic_predictions_used is False
    assert meta.adaptive_sampling_used is False


def test_audit_flags_tampered_provenance():
    parent = _parent()
    manifest = build_budgeted_candidate_manifest(
        parent_manifest=parent, budget_fraction=0.5
    )
    assert manifest.budget_metadata is not None

    for field, message in (
        ("outcome_fields_used", "budget selection used outcome fields"),
        (
            "critic_predictions_used",
            "budget selection used critic predictions",
        ),
        ("adaptive_sampling_used", "budget selection was adaptive"),
    ):
        tampered = manifest.model_copy(
            update={
                "budget_metadata": manifest.budget_metadata.model_copy(
                    update={field: True}
                )
            }
        )
        violations = audit_budget_manifests(
            parent_manifest=parent, budget_manifests={0.5: tampered}
        )
        assert any(message in v for v in violations), (field, violations)


def test_audit_passes_on_clean_manifests():
    parent = _parent()
    manifests = {
        fraction: build_budgeted_candidate_manifest(
            parent_manifest=parent, budget_fraction=fraction
        )
        for fraction in (0.25, 0.5, 0.75, 1.0)
    }
    violations = audit_budget_manifests(
        parent_manifest=parent, budget_manifests=manifests
    )
    assert violations == []
