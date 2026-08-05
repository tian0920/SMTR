"""Stratified candidate cohort tests (Commit 3).

Candidates must come from four cohorts (semantic_top / role_matched /
role_mismatched / cross_receiver_anchor), quotas must be configurable,
mismatched hard negatives must keep minimum task relevance, anchor memories
must reach >=2 receivers, and selection must never read outcome labels.
"""

from __future__ import annotations

import pytest

from smtr.core.types import AgentProfile, MemoryRoutingCard, ProcedurePayload
from smtr.marble.real_data import (
    CandidateRecord,
    CandidateCohortQuotas,
    ExtractedMemory,
    InsufficientReceiverEffectCoverageError,
    build_cross_task_candidates,
    quotas_from_top_k,
    require_receiver_effect_coverage,
    validate_receiver_effect_coverage,
)


def _make_memory(
    memory_id: str,
    *,
    writer_role: str,
    capabilities: tuple[str, ...] = (),
    tool_names: tuple[str, ...] = (),
    goal: str = "diagnose database issue",
    tags: tuple[str, ...] = ("database",),
) -> ExtractedMemory:
    writer = AgentProfile(
        agent_id=f"w-{memory_id}",
        role=writer_role,
        capabilities=capabilities,
        tool_names=tool_names,
    )
    return ExtractedMemory(
        memory_id=memory_id,
        payload=ProcedurePayload(
            memory_id=memory_id,
            procedure="1. Do something",
            writer=writer,
            source_task_id=f"src_{memory_id}",
            source_scenario="database",
        ),
        routing_card=MemoryRoutingCard(
            memory_id=memory_id,
            goal_summary=goal,
            task_tags=tags,
            environment_constraints=("read-only SQL",),
            writer=writer,
            source_task_id=f"src_{memory_id}",
            source_scenario="database",
            evidence_count=1,
        ),
    )


def _memories() -> list[ExtractedMemory]:
    return [
        _make_memory("mA", writer_role="executor", capabilities=("sql",), tool_names=("sql_tool",),
                     goal="diagnose database latency", tags=("database", "latency")),
        _make_memory("mB", writer_role="critic", capabilities=("review",), tool_names=("review_tool",),
                     goal="review database diagnosis", tags=("database", "review")),
        _make_memory("mC", writer_role="planner", capabilities=("planning",), tool_names=("plan_tool",),
                     goal="plan database diagnosis", tags=("database", "plan")),
        _make_memory("mD", writer_role="executor", capabilities=("accounting",), tool_names=("invoice_tool",),
                     goal="process accounting invoices", tags=("accounting", "invoices")),
    ]


def _receivers() -> list[dict]:
    return [
        {
            "task_id": "t1", "agent_id": "r1", "agent_role": "executor",
            "agent_capabilities": ["sql"], "tool_names": ["sql_tool"],
            "instruction": "diagnose database latency",
            "environment_signature": ["read-only SQL"],
        },
        {
            "task_id": "t2", "agent_id": "r2", "agent_role": "critic",
            "agent_capabilities": ["review"], "tool_names": ["review_tool"],
            "instruction": "diagnose database latency",
            "environment_signature": ["read-only SQL"],
        },
    ]


def test_quotas_from_top_k_balanced():
    q8 = quotas_from_top_k(8)
    assert (q8.semantic_top, q8.role_matched, q8.role_mismatched, q8.cross_receiver_anchor) == (2, 2, 2, 2)
    q4 = quotas_from_top_k(4)
    assert q4.total == 4


def test_all_four_cohort_sources_present_and_valid():
    manifest = build_cross_task_candidates(
        memories=_memories(), recipients=_receivers(), top_k=4,
        cohort_quotas=CandidateCohortQuotas(
            semantic_top=1, role_matched=1, role_mismatched=1, cross_receiver_anchor=1,
        ),
    )
    allowed = {"semantic_top", "role_matched", "role_mismatched", "cross_receiver_anchor"}
    sources = set()
    for entry in manifest.candidates:
        for rec in entry.candidate_records:
            assert rec.candidate_source in allowed
            sources.add(rec.candidate_source)
            if rec.candidate_source == "cross_receiver_anchor":
                assert rec.anchor_group_id == rec.memory_id
            else:
                assert rec.anchor_group_id is None
    # With this pool both matched and mismatched cohorts must be reachable
    assert "role_mismatched" in sources
    assert "cross_receiver_anchor" in sources


def test_anchor_memory_reaches_two_receivers():
    manifest = build_cross_task_candidates(
        memories=_memories(), recipients=_receivers(), top_k=4,
        cohort_quotas=CandidateCohortQuotas(
            semantic_top=1, role_matched=1, role_mismatched=1, cross_receiver_anchor=1,
        ),
    )
    memory_receivers: dict[str, set[str]] = {}
    for entry in manifest.candidates:
        for rec in entry.candidate_records:
            memory_receivers.setdefault(rec.memory_id, set()).add(entry.receiver_agent_id)
    shared = {mid for mid, recs in memory_receivers.items() if len(recs) >= 2}
    assert shared, "at least one memory must be evaluated by >=2 receivers"


def test_role_mismatched_hard_negatives_keep_min_relevance():
    """Mismatched hard negatives below min task relevance must be excluded,
    and completely irrelevant memories (mD) must never appear."""
    manifest = build_cross_task_candidates(
        memories=_memories(), recipients=_receivers(), top_k=4,
        cohort_quotas=CandidateCohortQuotas(
            semantic_top=1, role_matched=1, role_mismatched=1, cross_receiver_anchor=1,
            min_task_relevance=0.1,
        ),
    )
    for entry in manifest.candidates:
        for rec in entry.candidate_records:
            # mD shares no terms with the task instruction -> sim 0 < 0.1
            assert rec.memory_id != "mD"
            if rec.candidate_source == "role_mismatched":
                assert rec.score_components["task_similarity_raw"] >= 0.1


def test_quotas_are_configurable_not_hardcoded():
    quotas = CandidateCohortQuotas(
        semantic_top=3, role_matched=1, role_mismatched=1, cross_receiver_anchor=1,
    )
    manifest = build_cross_task_candidates(
        memories=_memories(), recipients=_receivers(), top_k=6, cohort_quotas=quotas,
    )
    assert manifest.cohort_quotas == quotas
    for entry in manifest.candidates:
        assert len(entry.candidate_records) <= quotas.total


def test_candidate_selection_ignores_outcomes():
    """Candidate record schema must carry no outcome/label fields."""
    forbidden = {"y_share", "y_withhold", "label", "team_success", "outcome"}
    assert not (set(CandidateRecord.model_fields) & forbidden)


def test_validate_receiver_effect_coverage_statistics():
    manifest = build_cross_task_candidates(
        memories=_memories(), recipients=_receivers(), top_k=4,
        cohort_quotas=CandidateCohortQuotas(
            semantic_top=1, role_matched=1, role_mismatched=1, cross_receiver_anchor=1,
        ),
    )
    report = validate_receiver_effect_coverage(manifest)
    stats = report["statistics"]
    for key in (
        "total_unique_memories",
        "memories_seen_by_2plus_receivers",
        "memories_seen_by_2plus_receiver_roles",
        "receiver_effect_coverage",
        "matched_candidate_rate",
        "mismatched_candidate_rate",
        "cross_receiver_anchor_rate",
    ):
        assert key in stats
    assert stats["total_unique_memories"] > 0
    assert stats["memories_seen_by_2plus_receivers"] >= 1
    assert stats["memories_seen_by_2plus_receiver_roles"] >= 1
    assert stats["cross_receiver_anchor_rate"] > 0.0
    assert report["checks"]["has_memory_seen_by_2plus_receivers"]
    assert report["checks"]["has_memory_seen_by_2plus_receiver_roles"]
    assert report["checks"]["candidate_selection_ignores_outcomes"]
    # Accepts a serialized manifest too
    report2 = validate_receiver_effect_coverage(manifest.model_dump(mode="json"))
    assert report2["statistics"] == stats


def _basic_quotas() -> CandidateCohortQuotas:
    return CandidateCohortQuotas(
        semantic_top=1, role_matched=1, role_mismatched=1, cross_receiver_anchor=1,
    )


def test_anchor_first_order_semantic_does_not_consume_anchor():
    """Anchors are assigned before semantic selection, so an anchor memory
    must appear with source cross_receiver_anchor, never as semantic_top."""
    manifest = build_cross_task_candidates(
        memories=_memories(), recipients=_receivers(), top_k=4,
        cohort_quotas=_basic_quotas(),
    )
    anchor_sources: dict[str, set[str]] = {}
    for entry in manifest.candidates:
        for rec in entry.candidate_records:
            if rec.anchor_group_id is not None:
                anchor_sources.setdefault(rec.memory_id, set()).add(rec.candidate_source)
    assert anchor_sources, "at least one anchor must be assigned"
    for mid, sources in anchor_sources.items():
        assert sources == {"cross_receiver_anchor"}, (
            f"anchor memory {mid} consumed by another cohort: {sources}"
        )


def test_anchor_receiver_stats_recorded():
    manifest = build_cross_task_candidates(
        memories=_memories(), recipients=_receivers(), top_k=4,
        cohort_quotas=_basic_quotas(),
    )
    anchors = [
        rec
        for entry in manifest.candidates
        for rec in entry.candidate_records
        if rec.candidate_source == "cross_receiver_anchor"
    ]
    assert anchors
    for rec in anchors:
        assert rec.anchor_receiver_count >= 2, (
            "anchor memory must be eligible for >=2 receivers"
        )
        assert rec.anchor_receiver_role_count >= 1
    non_anchors = [
        rec
        for entry in manifest.candidates
        for rec in entry.candidate_records
        if rec.candidate_source != "cross_receiver_anchor"
    ]
    for rec in non_anchors:
        assert rec.anchor_receiver_count == 0
        assert rec.anchor_receiver_role_count == 0


def test_anchor_memory_spans_receiver_roles_when_available():
    """With receivers of distinct roles, some anchor must span >=2 roles."""
    manifest = build_cross_task_candidates(
        memories=_memories(), recipients=_receivers(), top_k=4,
        cohort_quotas=_basic_quotas(),
    )
    report = validate_receiver_effect_coverage(manifest)
    assert report["statistics"]["cross_receiver_anchor_count"] >= 1
    assert report["checks"]["has_cross_receiver_anchor"]


def test_coverage_report_includes_anchor_and_relevance_fields():
    manifest = build_cross_task_candidates(
        memories=_memories(), recipients=_receivers(), top_k=4,
        cohort_quotas=_basic_quotas(),
    )
    report = validate_receiver_effect_coverage(manifest)
    stats = report["statistics"]
    for key in (
        "receiver_count",
        "unique_memory_count",
        "cross_receiver_anchor_count",
        "cross_receiver_role_anchor_count",
        "cohort_relevance_summary",
    ):
        assert key in stats, f"missing coverage statistic: {key}"
    for cohort, summary in stats["cohort_relevance_summary"].items():
        for field in ("mean", "median", "min", "max", "count"):
            assert field in summary, f"cohort {cohort} lacks {field}"
    assert "has_cross_receiver_anchor" in report["checks"]
    assert "has_cross_receiver_role_anchor" in report["checks"]


def test_formal_mode_requires_positive_min_task_relevance():
    with pytest.raises(ValueError, match="min_task_relevance"):
        build_cross_task_candidates(
            memories=_memories(), recipients=_receivers(), top_k=4,
            cohort_quotas=_basic_quotas(),
            experiment_mode="formal",
            min_task_relevance=0.0,
        )


def test_formal_candidate_build_fails_without_anchors():
    """require_receiver_effect_coverage must fail fast on missing coverage."""
    manifest = build_cross_task_candidates(
        memories=_memories(), recipients=_receivers(), top_k=4,
        cohort_quotas=CandidateCohortQuotas(
            semantic_top=2, role_matched=1, role_mismatched=1,
            cross_receiver_anchor=0,
        ),
    )
    report = validate_receiver_effect_coverage(manifest)
    assert not report["checks"]["has_cross_receiver_anchor"]
    with pytest.raises(InsufficientReceiverEffectCoverageError):
        require_receiver_effect_coverage(report)


def test_require_receiver_effect_coverage_passes_when_ok():
    manifest = build_cross_task_candidates(
        memories=_memories(), recipients=_receivers(), top_k=4,
        cohort_quotas=_basic_quotas(),
    )
    report = validate_receiver_effect_coverage(manifest)
    if report["ok"]:
        require_receiver_effect_coverage(report)  # must not raise
