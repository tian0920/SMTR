"""清单 Test 8: cross-receiver anchor construction (P0-10).

An anchor group ``(target_task, candidate_memory)`` must link at least two
different receivers, anchor groups never mix tasks, and every receiver's
candidate set keeps the required cohort coverage.
"""

from __future__ import annotations

import json

from smtr.core.types import AgentProfile, MemoryRoutingCard, ProcedurePayload
from smtr.marble.real_data import (
    ExtractedMemory,
    build_cross_task_candidates,
    compute_proposal_support_metrics,
    write_candidate_manifest,
)


def _make_memory(
    memory_id: str,
    *,
    writer_role: str,
    caps: tuple[str, ...],
    goal: str = "diagnose database latency",
) -> ExtractedMemory:
    writer = AgentProfile(agent_id=f"w-{memory_id}", role=writer_role, capabilities=caps)
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
            task_tags=("database", "latency"),
            environment_constraints=("read-only SQL",),
            writer=writer,
            source_task_id=f"src_{memory_id}",
            source_scenario="database",
            evidence_count=1,
        ),
    )


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


def _manifest():
    memories = [
        # Task-relevant for both receivers -> anchor-eligible.
        _make_memory("mA", writer_role="executor", caps=("sql",)),
        _make_memory("mB", writer_role="critic", caps=("review",)),
        _make_memory("mC", writer_role="planner", caps=("planning",)),
        _make_memory("mD", writer_role="verifier", caps=("stats",)),
    ]
    return build_cross_task_candidates(
        memories=memories, recipients=_receivers(), top_k=8
    )


class TestCrossReceiverAnchorConstruction:
    def test_anchor_memory_reaches_at_least_two_receivers(self):
        manifest = _manifest()
        anchor_receivers: dict[str, set[str]] = {}
        for entry in manifest.candidates:
            for rec in entry.candidate_records:
                if "cross_receiver_anchor" in rec.candidate_sources:
                    anchor_receivers.setdefault(rec.memory_id, set()).add(
                        entry.receiver_agent_id
                    )
        assert anchor_receivers, "no cross-receiver anchor was constructed"
        assert all(
            len(rs) >= 2 for rs in anchor_receivers.values()
        ), "anchor group must link at least two different receivers"

    def test_anchor_groups_never_mix_tasks(self):
        """Each candidate record belongs to exactly one target task."""
        manifest = _manifest()
        per_memory_tasks: dict[tuple[str, str], set[str]] = {}
        for entry in manifest.candidates:
            for rec in entry.candidate_records:
                key = (entry.receiver_agent_id, rec.memory_id)
                per_memory_tasks.setdefault(key, set()).add(entry.task_id)
        for key, tasks in per_memory_tasks.items():
            assert len(tasks) == 1, f"receiver/memory pair {key} mixes tasks {tasks}"

    def test_every_receiver_keeps_cohort_coverage(self):
        """清单验收: 每个 receiver 有足够候选 (seed 支撑的基础).

        The candidate count must saturate the cohort budget up to the size
        of the receiver's eligible pool (one candidate per pool memory).
        """
        manifest = _manifest()
        quotas = manifest.cohort_quotas
        n_memories = 4  # one memory per writer in the fixture pool
        for entry in manifest.candidates:
            assert len(entry.candidate_records) == min(quotas.total, n_memories), (
                f"receiver {entry.receiver_agent_id} got fewer candidates "
                "than its cohort budget allows"
            )

    def test_candidate_sources_mark_all_cohort_memberships(self):
        manifest = _manifest()
        tags_seen: set[str] = set()
        for entry in manifest.candidates:
            for rec in entry.candidate_records:
                assert rec.candidate_sources, "candidate lacks source tags"
                tags_seen.update(rec.candidate_sources)
        assert "cross_receiver_anchor" in tags_seen
        assert tags_seen <= {
            "semantic_topk",
            "role_matched",
            "role_mismatched_hard_negative",
            "cross_receiver_anchor",
        }


class TestProposalSupportMetrics:
    def test_all_required_support_fields_present(self):
        metrics = compute_proposal_support_metrics(_manifest())
        for key in (
            "candidate_count_per_receiver",
            "role_matched_candidate_rate",
            "role_mismatched_candidate_rate",
            "cross_receiver_anchor_count",
            "memories_with_multiple_receivers",
            "receivers_per_anchor_memory",
            "candidate_source_distribution",
        ):
            assert key in metrics, f"missing support metric {key}"
        assert metrics["cross_receiver_anchor_count"] >= 1
        assert metrics["memories_with_multiple_receivers"] >= 1
        assert all(v >= 2 for v in metrics["receivers_per_anchor_memory"].values())

    def test_support_metrics_written_next_to_manifest(self, tmp_path):
        manifest = _manifest()
        out = tmp_path / "candidates.json"
        result = write_candidate_manifest(manifest=manifest, output_path=out)
        support = json.loads(
            (tmp_path / "proposal_support_metrics.json").read_text(encoding="utf-8")
        )
        assert support["total_candidate_count"] == sum(
            support["candidate_count_per_receiver"].values()
        )
        assert sum(support["candidate_source_distribution"].values()) >= support[
            "total_candidate_count"
        ]
