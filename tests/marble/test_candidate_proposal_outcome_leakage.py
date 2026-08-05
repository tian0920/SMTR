"""清单 Test 7: candidate proposal must never read transfer outcomes (P0-9).

The proposer may only use pre-exposure information (task instruction,
routing card, writer/receiver roles and capabilities, environment
compatibility, source task metadata). Paired labels, share/withhold
outcomes, q values and test transfer statistics are forbidden inputs.
"""

from __future__ import annotations

import inspect

from smtr.core.types import AgentProfile, MemoryRoutingCard, ProcedurePayload
from smtr.marble.real_data import (
    CandidateEntry,
    CandidateRecord,
    DatabaseCandidateManifest,
    ExtractedMemory,
    build_cross_task_candidates,
    validate_receiver_effect_coverage,
)
from smtr.marble import real_data

FORBIDDEN_INPUT_TOKENS = (
    "team_success",
    "y_share",
    "y_withhold",
    "q01",
    "q10",
    "paired_label",
    "transfer_label",
)


def _make_memory(memory_id: str, *, writer_role: str, caps: tuple[str, ...]) -> ExtractedMemory:
    writer = AgentProfile(
        agent_id=f"w-{memory_id}", role=writer_role, capabilities=caps
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
            goal_summary="diagnose database latency",
            task_tags=("database", "latency"),
            environment_constraints=("read-only SQL",),
            writer=writer,
            source_task_id=f"src_{memory_id}",
            source_scenario="database",
            evidence_count=1,
        ),
    )


def _receivers(with_outcomes: bool = False) -> list[dict]:
    extras = (
        {"team_success": True, "y_share": True, "y_withhold": False,
         "paired_label": "positive_transfer"}
        if with_outcomes
        else {}
    )
    return [
        {
            "task_id": "t1", "agent_id": "r1", "agent_role": "executor",
            "agent_capabilities": ["sql"], "tool_names": ["sql_tool"],
            "instruction": "diagnose database latency",
            "environment_signature": ["read-only SQL"],
            **extras,
        },
        {
            "task_id": "t2", "agent_id": "r2", "agent_role": "critic",
            "agent_capabilities": ["review"], "tool_names": ["review_tool"],
            "instruction": "diagnose database latency",
            "environment_signature": ["read-only SQL"],
            **extras,
        },
    ]


class TestProposalSchemaCarriesNoOutcomes:
    def test_candidate_schema_has_no_outcome_fields(self):
        forbidden = {
            "y_share", "y_withhold", "label", "team_success", "outcome",
            "q01", "q10", "transfer_label", "paired_label",
        }
        for model in (CandidateRecord, CandidateEntry, DatabaseCandidateManifest):
            leaked = set(model.model_fields) & forbidden
            assert not leaked, f"{model.__name__} carries outcome fields: {leaked}"

    def test_coverage_audit_flags_outcome_fields(self):
        coverage = validate_receiver_effect_coverage(
            DatabaseCandidateManifest(candidates=[]).model_dump(mode="json")
        )
        assert coverage["checks"]["candidate_selection_ignores_outcomes"]


class TestProposerOnlyUsesPreExposureInformation:
    def test_proposer_functions_never_accept_outcome_parameters(self):
        for func_name in (
            "build_cross_task_candidates",
            "_score_memory_for_recipient",
            "_select_anchor_assignments",
        ):
            params = set(
                inspect.signature(getattr(real_data, func_name)).parameters
            )
            for token in FORBIDDEN_INPUT_TOKENS:
                assert not any(
                    token in p for p in params
                ), f"{func_name} accepts outcome-related parameter"

    def test_proposer_source_never_reads_outcome_keys(self):
        for func_name in (
            "build_cross_task_candidates",
            "_score_memory_for_recipient",
            "_select_anchor_assignments",
        ):
            source = inspect.getsource(getattr(real_data, func_name)).lower()
            for token in FORBIDDEN_INPUT_TOKENS:
                assert token not in source, (
                    f"{func_name} references forbidden outcome token {token!r}"
                )

    def test_injected_outcome_fields_do_not_change_selection(self):
        memories = [
            _make_memory("mA", writer_role="executor", caps=("sql",)),
            _make_memory("mB", writer_role="critic", caps=("review",)),
        ]
        clean = build_cross_task_candidates(
            memories=memories, recipients=_receivers(with_outcomes=False), top_k=4
        )
        leaked = build_cross_task_candidates(
            memories=memories, recipients=_receivers(with_outcomes=True), top_k=4
        )
        assert clean.model_dump(mode="json") == leaked.model_dump(mode="json")
