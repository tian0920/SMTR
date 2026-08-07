"""Pilot candidate memory definitions for causal pilot experiments.

Defines 5 categories of synthetic candidate memories for testing
the SMTR routing pipeline's ability to discriminate beneficial
vs. harmful vs. irrelevant memories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CandidateMemoryCategory = Literal[
    "beneficial",
    "irrelevant",
    "outdated",
    "conflicting",
    "receiver_incompatible",
]


@dataclass(frozen=True)
class PilotCandidateMemory:
    """A synthetic candidate memory for causal pilot experiments."""

    memory_id: str
    category: CandidateMemoryCategory
    task_id: str
    scenario: str
    payload: str
    routing_card: dict[str, object]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "category": self.category,
            "task_id": self.task_id,
            "scenario": self.scenario,
            "payload": self.payload,
            "routing_card": self.routing_card,
            "metadata": self.metadata,
        }


def build_beneficial_memory(
    task_id: str,
    scenario: str = "database",
    index: int = 0,
) -> PilotCandidateMemory:
    """A memory that provides correct, task-relevant diagnostic guidance."""
    return PilotCandidateMemory(
        memory_id=f"beneficial_{task_id}_{index}",
        category="beneficial",
        task_id=task_id,
        scenario=scenario,
        payload=(
            "When diagnosing database performance issues, always check "
            "pg_stat_statements for the top queries by total_time. "
            "Look for sequential scans on large tables and missing indexes "
            "on foreign key columns. Verify that VACUUM ANALYZE has run "
            "recently to keep planner statistics up to date."
        ),
        routing_card={
            "goal_summary": "Guide systematic database performance diagnosis",
            "task_tags": ["database", "performance", "diagnosis"],
            "precondition_summary": "Agent has access to pg_stat_statements",
            "expected_effect": "Agent follows structured diagnostic checklist",
            "known_risks": [],
        },
        metadata={"source": "synthetic_pilot", "quality": "high"},
    )


def build_irrelevant_memory(
    task_id: str,
    scenario: str = "database",
    index: int = 0,
) -> PilotCandidateMemory:
    """A memory about an unrelated topic (frontend debugging)."""
    return PilotCandidateMemory(
        memory_id=f"irrelevant_{task_id}_{index}",
        category="irrelevant",
        task_id=task_id,
        scenario=scenario,
        payload=(
            "When debugging React component re-renders, use React DevTools "
            "Profiler to identify unnecessary renders. Wrap expensive components "
            "in React.memo and use useMemo for computed values. Avoid inline "
            "object literals in JSX props as they create new references."
        ),
        routing_card={
            "goal_summary": "React rendering optimization guidance",
            "task_tags": ["frontend", "react", "rendering"],
            "precondition_summary": "React application with performance issues",
            "expected_effect": "Agent applies React optimization techniques",
            "known_risks": ["not applicable to database tasks"],
        },
        metadata={"source": "synthetic_pilot", "quality": "low_relevance"},
    )


def build_outdated_memory(
    task_id: str,
    scenario: str = "database",
    index: int = 0,
) -> PilotCandidateMemory:
    """A memory referencing deprecated tools or removed features."""
    return PilotCandidateMemory(
        memory_id=f"outdated_{task_id}_{index}",
        category="outdated",
        task_id=task_id,
        scenario=scenario,
        payload=(
            "Use the pg_stat_deprecated view to check query performance. "
            "The old_perf_monitor() function provides real-time index usage "
            "statistics. Note: autovacuum_vacuum_threshold has been changed "
            "to 10000 in the latest configuration update."
        ),
        routing_card={
            "goal_summary": "Database performance monitoring via deprecated views",
            "task_tags": ["database", "monitoring", "deprecated"],
            "precondition_summary": "Legacy monitoring tools available",
            "expected_effect": "Agent uses deprecated monitoring approach",
            "known_risks": ["references removed views/functions"],
        },
        metadata={"source": "synthetic_pilot", "quality": "outdated"},
    )


def build_conflicting_memory(
    task_id: str,
    scenario: str = "database",
    index: int = 0,
) -> PilotCandidateMemory:
    """A memory that provides contradictory guidance."""
    return PilotCandidateMemory(
        memory_id=f"conflicting_{task_id}_{index}",
        category="conflicting",
        task_id=task_id,
        scenario=scenario,
        payload=(
            "Never use indexes for query optimization — they always cause "
            "performance degradation. Instead, drop all secondary indexes "
            "and rely solely on sequential scans. Disable autovacuum to "
            "prevent background I/O contention."
        ),
        routing_card={
            "goal_summary": "Anti-pattern database optimization advice",
            "task_tags": ["database", "anti-pattern", "indexes"],
            "precondition_summary": "Database with index-related questions",
            "expected_effect": "Agent may follow harmful optimization path",
            "known_risks": ["contradicts best practices", "may cause harm"],
        },
        metadata={"source": "synthetic_pilot", "quality": "harmful"},
    )


def build_receiver_incompatible_memory(
    task_id: str,
    scenario: str = "database",
    index: int = 0,
) -> PilotCandidateMemory:
    """A memory intended for a different agent role."""
    return PilotCandidateMemory(
        memory_id=f"receiver_incompatible_{task_id}_{index}",
        category="receiver_incompatible",
        task_id=task_id,
        scenario=scenario,
        payload=(
            "As the security auditor agent, verify that all database users "
            "follow the principle of least privilege. Check pg_roles for "
            "SUPERUSER grants and ensure connection strings don't embed "
            "credentials. Review pg_hba.conf for overly permissive rules."
        ),
        routing_card={
            "goal_summary": "Security audit checklist for database roles",
            "task_tags": ["security", "audit", "roles"],
            "precondition_summary": "Agent has security auditor role",
            "expected_effect": "Agent performs security-focused audit",
            "known_risks": ["wrong role for performance diagnosis task"],
        },
        metadata={"source": "synthetic_pilot", "quality": "role_mismatch"},
    )


def build_candidate_set(
    task_id: str,
    scenario: str = "database",
) -> list[PilotCandidateMemory]:
    """Build a full set of 5 candidate memories for a task."""
    return [
        build_beneficial_memory(task_id, scenario, index=0),
        build_irrelevant_memory(task_id, scenario, index=0),
        build_outdated_memory(task_id, scenario, index=0),
        build_conflicting_memory(task_id, scenario, index=0),
        build_receiver_incompatible_memory(task_id, scenario, index=0),
    ]
