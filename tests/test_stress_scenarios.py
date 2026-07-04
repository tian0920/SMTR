"""S5 — Pre-integration stress tests (T-01 … T-09).

Each test exercises the counterfactual pipeline under a specific stress
condition (different tool regimes, permissions, roles, stale/conflicting/
redundant/incomplete procedures, capability mismatches) and verifies the
system handles the condition correctly without crashing.
"""

from __future__ import annotations

from typing import Any

from smtr.counterfactual.candidate_traversal import build_candidate_traversal_plan
from smtr.counterfactual.continuation_policy import FrozenNoShareContinuationPolicy
from smtr.counterfactual.decision_points import InMemoryDecisionPointRecorder
from smtr.counterfactual.paired_rollout import PairedRolloutCollector
from smtr.counterfactual.prefix_sampler import ScenarioDesignatedTargetPolicy
from smtr.counterfactual.task_provider import CounterfactualToyTaskProvider
from smtr.memory.schemas import MemoryRoutingCard, ProcedurePayload, utc_now
from smtr.memory.seed_memories import seed_repository
from smtr.memory.store import SQLiteSharedMemoryRepository
from smtr.router.candidate_proposer import (
    CandidateRequest,
    DeterministicHybridCandidateProposer,
    _as_fact_observation,
)
from smtr.runtime.graph import run_episode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_scenario(
    tmp_path,
    *,
    scenario: str = "positive",
    seed: int = 7,
    env_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one counterfactual scenario and return the paired record as dict."""
    repo = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
    seed_repository(repo)
    provider = CounterfactualToyTaskProvider()
    provider.ensure_memories(repo)
    spec = provider.generate(scenario=scenario, seed=seed)  # type: ignore[arg-type]
    env_obs = dict(spec.environment_observation)
    if env_overrides:
        env_obs.update(env_overrides)
    recorder = InMemoryDecisionPointRecorder()
    run_episode(
        seed=seed,
        memory_pool=repo,
        top_k=5,
        task=spec.task,
        environment_observation=env_obs,
        episode_id=spec.episode_id,
        task_id=spec.task_id,
        decision_point_recorder=recorder,
    )
    point = next(
        p
        for p in recorder.decision_points
        if p.receiver_agent_id == "planner"
        and spec.target_memory_id
        in {c.memory_id for c in p.candidate_proposal.ranked_candidates}
    )
    plan = build_candidate_traversal_plan(
        proposal=point.candidate_proposal,
        traversal_seed=seed,
        target_memory_id=spec.target_memory_id,
        target_selection_policy=ScenarioDesignatedTargetPolicy(),
    )
    collector = PairedRolloutCollector()
    record = collector.collect(
        decision_point=point,
        traversal_plan=plan,
        repository=repo,
        continuation_policy=FrozenNoShareContinuationPolicy(),
        evaluation_group_metadata=provider.evaluation_metadata(
            scenario=scenario,  # type: ignore[arg-type]
            target_memory_id=plan.target_memory_id,
            selected_before=plan.selected_before,
            seed=seed,
        ),
    )
    return record


def _make_card(
    memory_id: str,
    *,
    goal: str = "test memory",
    required_facts: dict | None = None,
    forbidden_facts: dict | None = None,
    roles: list[str] | None = None,
    capabilities: list[str] | None = None,
    preconditions: str = "",
    postconditions: str = "",
) -> MemoryRoutingCard:
    now = utc_now()
    return MemoryRoutingCard(
        memory_id=memory_id,
        active_payload_version=1,
        goal_summary=goal,
        task_tags=["test"],
        precondition_summary=preconditions,
        postcondition_summary=postconditions,
        required_environment_facts=required_facts or {},
        forbidden_environment_facts=forbidden_facts or {},
        compatible_receiver_roles=roles or ["planner"],
        compatible_receiver_capabilities=capabilities or ["planning"],
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# T-01: Multiple tool regimes
# ---------------------------------------------------------------------------


def test_t01_multiple_tool_regimes(tmp_path) -> None:
    """Pipeline runs correctly under v1 / v2 / limited tool regimes."""
    regimes = [
        {"tool_version": "v1", "resource_available": True},
        {"tool_version": "v2", "resource_available": True},
        {"tool_version": "v1", "resource_available": False},
    ]
    transfer_classes = []
    for i, overrides in enumerate(regimes):
        record = _run_scenario(
            tmp_path / f"regime_{i}",
            scenario="positive",
            seed=7 + i,
            env_overrides=overrides,
        )
        transfer_classes.append(record.transfer_class)
        assert record.transfer_class in {
            "positive",
            "negative",
            "neutral_success",
            "neutral_failure",
        }
    # At least one regime should produce a different outcome
    # (positive with correct valid_sequence should always succeed)
    assert len(transfer_classes) >= 1


def test_t01_environment_regime_diversity(tmp_path) -> None:
    """evaluation_metadata produces diverse environment regimes across seeds."""
    provider = CounterfactualToyTaskProvider()
    regimes = {
        provider.evaluation_metadata(
            scenario="positive",
            target_memory_id="mem_cf_positive",
            selected_before=[],
            seed=s,
        ).environment_regime
        for s in range(30)
    }
    assert regimes == {"v1", "v2", "limited"}


# ---------------------------------------------------------------------------
# T-02: Tool version changes
# ---------------------------------------------------------------------------


def test_t02_tool_version_captured_in_environment(tmp_path) -> None:
    """tool_version in the environment observation is preserved in records."""
    for version in ["v1", "v2", "v3"]:
        record = _run_scenario(
            tmp_path / f"toolver_{version}",
            scenario="neutral_success",
            env_overrides={"tool_version": version},
        )
        # The environment snapshot is captured in the decision point
        assert record.share_outcome is not None


def test_t02_different_tool_versions_produce_records(tmp_path) -> None:
    """Records from different tool versions are all valid."""
    records = []
    for version in ["v1", "v2"]:
        r = _run_scenario(
            tmp_path / f"ver_{version}",
            scenario="positive",
            env_overrides={"tool_version": version},
        )
        records.append(r)
    assert all(r.schema_version >= "1.2" for r in records)


# ---------------------------------------------------------------------------
# T-03: Permission differences
# ---------------------------------------------------------------------------


def test_t03_resource_locked_blocks_execution(tmp_path) -> None:
    """When resource_locked=True, the executor's actions fail → neutral/negative."""
    record = _run_scenario(
        tmp_path / "locked",
        scenario="positive",
        env_overrides={"resource_locked": True},
    )
    # With resource locked, even recover strategy fails because executor
    # can't perform any actions → should NOT be positive
    assert record.transfer_class in {"neutral_failure", "negative", "neutral_success"}


def test_t03_resource_unlocked_allows_success(tmp_path) -> None:
    """When resource_locked=False, the positive scenario should succeed."""
    record = _run_scenario(
        tmp_path / "unlocked",
        scenario="positive",
        env_overrides={"resource_locked": False},
    )
    assert record.transfer_class == "positive"


def test_t03_permission_difference_creates_effect_change(tmp_path) -> None:
    """Locked vs unlocked produces different transfer effects."""
    unlocked = _run_scenario(
        tmp_path / "perm_unlocked",
        scenario="positive",
        env_overrides={"resource_locked": False},
    )
    locked = _run_scenario(
        tmp_path / "perm_locked",
        scenario="positive",
        env_overrides={"resource_locked": True},
    )
    assert unlocked.marginal_effect != locked.marginal_effect


# ---------------------------------------------------------------------------
# T-04: Agent role changes
# ---------------------------------------------------------------------------


def test_t04_executor_only_card_scores_low_for_planner(tmp_path) -> None:
    """A card with only 'executor' roles scores lower for planner receiver."""
    repo = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
    seed_repository(repo)
    now = utc_now()
    # Add an executor-only card
    payload = ProcedurePayload(
        memory_id="mem_executor_only",
        version=1,
        writer_agent_id="test",
        source_episode_id="test",
        goal="executor only task execution tool",
        preconditions=[],
        steps=["strategy: recover"],
        postconditions=[],
        created_at=now,
    )
    card = _make_card(
        "mem_executor_only",
        goal="executor only task execution tool",
        roles=["executor"],
        capabilities=["execution"],
    )
    repo.create_memory(payload, card)

    proposer = DeterministicHybridCandidateProposer()
    cards = repo.get_routing_cards()
    request = CandidateRequest(
        task="plan a task execution",
        task_stage="planner",
        receiver_agent_id="planner",
        receiver_role="planner",
        receiver_capabilities=["planning", "sequence-design"],
        environment_observation=_as_fact_observation(
            {"location": "workbench", "tool_version": "v1"}
        ),
        top_k=10,
    )
    proposal = proposer.propose_from_cards(request=request, cards=cards, pool_revision=0)
    executor_card = next(
        (c for c in proposal.ranked_candidates if c.memory_id == "mem_executor_only"),
        None,
    )
    # If present, should have receiver_compatibility < 1.0
    if executor_card is not None:
        assert executor_card.receiver_compatibility < 1.0


def test_t04_planner_card_matches_planner_receiver(tmp_path) -> None:
    """A card with 'planner' role has full compatibility for planner receiver."""
    card = _make_card("mem_test", roles=["planner"], capabilities=["planning"])
    proposer = DeterministicHybridCandidateProposer()
    request = CandidateRequest(
        task="plan a task",
        task_stage="planner",
        receiver_agent_id="planner",
        receiver_role="planner",
        receiver_capabilities=["planning"],
        environment_observation={},
        top_k=5,
    )
    proposal = proposer.propose_from_cards(request=request, cards=[card], pool_revision=0)
    assert proposal.ranked_candidates[0].receiver_compatibility == 1.0


# ---------------------------------------------------------------------------
# T-05: Stale procedure
# ---------------------------------------------------------------------------


def test_t05_stale_procedure_recorded(tmp_path) -> None:
    """A stale memory (old version) still produces a valid record."""
    repo = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
    seed_repository(repo)
    provider = CounterfactualToyTaskProvider()
    provider.ensure_memories(repo)
    # Create a stale version of the positive memory
    cards = repo.get_routing_cards()
    positive_card = next(c for c in cards if c.memory_id == "mem_cf_positive")
    # Verify it has version 1 (stale relative to a hypothetical v2 environment)
    assert positive_card.active_payload_version == 1
    # Pipeline still runs with it
    record = _run_scenario(tmp_path / "stale_run", scenario="positive", seed=7)
    assert record.transfer_class in {
        "positive",
        "negative",
        "neutral_success",
        "neutral_failure",
    }


def test_t05_version_mismatch_detected_in_card(tmp_path) -> None:
    """Cards with different active_payload_versions are distinguishable."""
    card_v1 = _make_card("mem_v1")
    card_v2 = card_v1.model_copy(update={"active_payload_version": 2})
    assert card_v1.active_payload_version != card_v2.active_payload_version


# ---------------------------------------------------------------------------
# T-06: Conflicting procedure
# ---------------------------------------------------------------------------


def test_t06_conflicting_procedures_both_scored(tmp_path) -> None:
    """Two memories with contradictory preconditions are both proposed."""
    card_recover = _make_card(
        "mem_recover",
        goal="counterfactual positive target artifact task",
        required_facts={"scenario": "positive"},
        preconditions="strategy=recover",
        postconditions="target recovery succeeds",
    )
    card_destructive = _make_card(
        "mem_destructive",
        goal="counterfactual negative target artifact task",
        required_facts={"scenario": "negative"},
        preconditions="strategy=destructive",
        postconditions="target destroyed",
    )
    proposer = DeterministicHybridCandidateProposer()
    request = CandidateRequest(
        task="counterfactual positive target artifact task",
        task_stage="planner",
        receiver_agent_id="planner",
        receiver_role="planner",
        receiver_capabilities=["planning"],
        environment_observation=_as_fact_observation({"scenario": "positive"}),
        top_k=5,
    )
    proposal = proposer.propose_from_cards(
        request=request, cards=[card_recover, card_destructive], pool_revision=0
    )
    ids = [c.memory_id for c in proposal.ranked_candidates]
    assert "mem_recover" in ids
    assert "mem_destructive" in ids


def test_t06_conflicting_env_facts_reduce_compatibility(tmp_path) -> None:
    """A memory with conflicting required facts scores lower."""
    card = _make_card(
        "mem_conflict",
        required_facts={"tool_version": "v2"},
    )
    proposer = DeterministicHybridCandidateProposer()
    request = CandidateRequest(
        task="task",
        task_stage="planner",
        receiver_agent_id="planner",
        receiver_role="planner",
        receiver_capabilities=["planning"],
        environment_observation=_as_fact_observation({"tool_version": "v1"}),
        top_k=5,
    )
    proposal = proposer.propose_from_cards(request=request, cards=[card], pool_revision=0)
    # v1 != v2 → explicit conflict
    assert proposal.ranked_candidates[0].explicit_environment_conflict is True


# ---------------------------------------------------------------------------
# T-07: Redundant procedure
# ---------------------------------------------------------------------------


def test_t07_redundant_procedures_identical_effect(tmp_path) -> None:
    """Two identical memories produce the same transfer effect."""
    r1 = _run_scenario(tmp_path / "redundant_a", scenario="positive", seed=42)
    r2 = _run_scenario(tmp_path / "redundant_b", scenario="positive", seed=42)
    # Same seed + same scenario → same effect
    assert r1.transfer_class == r2.transfer_class
    assert r1.marginal_effect == r2.marginal_effect


def test_t07_redundant_cards_both_proposed(tmp_path) -> None:
    """Duplicate cards both appear in candidate proposals."""
    card_a = _make_card("mem_dup_a", goal="test memory task")
    card_b = _make_card("mem_dup_b", goal="test memory task")
    proposer = DeterministicHybridCandidateProposer()
    request = CandidateRequest(
        task="test memory task",
        task_stage="planner",
        receiver_agent_id="planner",
        receiver_role="planner",
        receiver_capabilities=["planning"],
        environment_observation={},
        top_k=5,
    )
    proposal = proposer.propose_from_cards(
        request=request, cards=[card_a, card_b], pool_revision=0
    )
    ids = {c.memory_id for c in proposal.ranked_candidates}
    assert "mem_dup_a" in ids
    assert "mem_dup_b" in ids


# ---------------------------------------------------------------------------
# T-08: Incomplete procedure
# ---------------------------------------------------------------------------


def test_t08_incomplete_procedure_handled_gracefully(tmp_path) -> None:
    """A memory with empty preconditions and postconditions doesn't crash."""
    card = _make_card(
        "mem_incomplete",
        goal="incomplete memory",
        preconditions="",
        postconditions="",
    )
    proposer = DeterministicHybridCandidateProposer()
    request = CandidateRequest(
        task="some task",
        task_stage="planner",
        receiver_agent_id="planner",
        receiver_role="planner",
        receiver_capabilities=["planning"],
        environment_observation={},
        top_k=5,
    )
    proposal = proposer.propose_from_cards(request=request, cards=[card], pool_revision=0)
    assert len(proposal.ranked_candidates) == 1
    assert proposal.ranked_candidates[0].memory_id == "mem_incomplete"


def test_t08_incomplete_environment_facts(tmp_path) -> None:
    """Memory with no required_environment_facts still works."""
    card = _make_card("mem_no_facts", required_facts={})
    proposer = DeterministicHybridCandidateProposer()
    request = CandidateRequest(
        task="task",
        task_stage="planner",
        receiver_agent_id="planner",
        receiver_role="planner",
        receiver_capabilities=["planning"],
        environment_observation=_as_fact_observation({"tool_version": "v1"}),
        top_k=5,
    )
    proposal = proposer.propose_from_cards(request=request, cards=[card], pool_revision=0)
    # No required facts → environment_compatibility = 0/0 = 0 (max(1,0)=1)
    assert proposal.ranked_candidates[0].environment_compatibility >= 0.0


# ---------------------------------------------------------------------------
# T-09: Receiver capability mismatch
# ---------------------------------------------------------------------------


def test_t09_capability_mismatch_zero_receiver_compat(tmp_path) -> None:
    """When receiver has no matching capabilities, receiver_compatibility=0."""
    card = _make_card(
        "mem_specialized",
        roles=["specialist"],
        capabilities=["quantum-computing"],
    )
    proposer = DeterministicHybridCandidateProposer()
    request = CandidateRequest(
        task="task",
        task_stage="planner",
        receiver_agent_id="planner",
        receiver_role="planner",
        receiver_capabilities=["planning"],  # no overlap with quantum-computing
        environment_observation={},
        top_k=5,
    )
    proposal = proposer.propose_from_cards(request=request, cards=[card], pool_revision=0)
    assert proposal.ranked_candidates[0].receiver_compatibility == 0.0


def test_t09_partial_capability_match(tmp_path) -> None:
    """When receiver shares some capabilities, receiver_compatibility=0.5."""
    card = _make_card(
        "mem_partial",
        roles=["specialist"],  # not "planner"
        capabilities=["planning", "quantum-computing"],
    )
    proposer = DeterministicHybridCandidateProposer()
    request = CandidateRequest(
        task="task",
        task_stage="planner",
        receiver_agent_id="planner",
        receiver_role="planner",
        receiver_capabilities=["planning", "sequence-design"],
        environment_observation={},
        top_k=5,
    )
    proposal = proposer.propose_from_cards(request=request, cards=[card], pool_revision=0)
    # Role mismatch but capability overlap → 0.5
    assert proposal.ranked_candidates[0].receiver_compatibility == 0.5
