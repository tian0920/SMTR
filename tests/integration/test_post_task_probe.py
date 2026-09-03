"""Tests for post-task causal probing (§14).

Covers:
    test_online_transfer_evidence_dataclass
    test_probe_selection_exploit_only_returns_none
    test_probe_selection_no_global_returns_none
    test_probe_selection_returns_highest_lcb_global
    test_post_task_probe_collect_matched_evidence
    test_post_task_probe_single_seed_no_std
    test_post_task_probe_shared_control_reuse
    test_forward_only_invariant_documented
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from smtr.rima.online_transfer_evidence import OnlineTransferEvidence
from smtr.rima.post_task_probe import (
    PostTaskTransferProbe,
    ProbeSelectionPolicy,
    select_probe_candidate,
)
from smtr.rima.transfer_controller import (
    RoutingMode,
    TransferCandidateDecision,
    TransferRoutingPlan,
)


# ---------------------------------------------------------------------------
# OnlineTransferEvidence tests
# ---------------------------------------------------------------------------


def test_online_transfer_evidence_dataclass():
    """OnlineTransferEvidence must be frozen and carry all required fields."""
    evidence = OnlineTransferEvidence(
        task_id="t1",
        task_position=5,
        receiver_id="agent_a",
        memory_id="m1",
        expose_scores=[0.8, 0.7, 0.9],
        withhold_scores=[0.5, 0.4, 0.6],
        observed_tau=0.3,
        tau_std=0.1,
        generation_seeds=[0, 1, 2],
    )
    assert evidence.task_id == "t1"
    assert evidence.task_position == 5
    assert evidence.receiver_id == "agent_a"
    assert evidence.memory_id == "m1"
    assert len(evidence.expose_scores) == 3
    assert evidence.observed_tau == pytest.approx(0.3)
    assert evidence.tau_std == pytest.approx(0.1)
    assert evidence.generation_seeds == [0, 1, 2]

    # Frozen dataclass
    with pytest.raises(AttributeError):
        evidence.task_id = "t2"  # type: ignore


# ---------------------------------------------------------------------------
# ProbeSelectionPolicy tests
# ---------------------------------------------------------------------------


def _make_candidate(
    memory_id: str,
    receiver_id: str,
    lcb: float,
    eligible: bool = True,
) -> TransferCandidateDecision:
    mu = lcb + 0.1
    sigma = 0.05
    return TransferCandidateDecision(
        memory_id=memory_id,
        receiver_id=receiver_id,
        task_id="t1",
        candidate_source="global",
        mu_tau=mu,
        sigma_tau=sigma,
        lcb=lcb,
        ucb=mu + 1.64 * sigma,
        eligible_for_context=eligible,
        selected_for_context=False,
        status="positive" if lcb > 0 else "negative",
    )


def test_probe_selection_exploit_only_returns_none():
    """EXPLOIT_ONLY mode must not trigger probing."""
    candidates = [_make_candidate("m1", "r1", lcb=0.5)]
    result = ProbeSelectionPolicy.select(RoutingMode.EXPLOIT_ONLY, candidates)
    assert result is None


def test_probe_selection_no_global_returns_none():
    """Empty global_candidates must not trigger probing."""
    result = ProbeSelectionPolicy.select(RoutingMode.EXPLORE_ONLY, [])
    assert result is None


def test_probe_selection_returns_highest_ucb_global():
    """Must select the global candidate with highest UCB."""
    c1 = _make_candidate("m1", "r1", lcb=0.3)
    c2 = _make_candidate("m2", "r1", lcb=0.5)
    c3 = _make_candidate("m3", "r1", lcb=0.4)

    result = ProbeSelectionPolicy.select(
        RoutingMode.EXPLORE_ONLY,
        [c1, c2, c3],
    )
    assert result is not None
    assert result.memory_id == "m2"
    assert result.ucb == pytest.approx(0.5 + 0.1 + 1.64 * 0.05)


def test_probe_selection_execution_ineligible_still_probeable():
    """Execution-ineligible (LCB <= delta) candidates remain probeable.

    This is the cold-start fix: probe eligibility is decoupled from the
    execution gate ``LCB > delta``.
    """
    c1 = _make_candidate("m1", "r1", lcb=0.5, eligible=True)
    # c2 has negative LCB (execution-blocked) but the highest UCB.
    c2 = _make_candidate("m2", "r1", lcb=-0.4, eligible=False)
    c2.ucb = 0.9

    result = ProbeSelectionPolicy.select(
        RoutingMode.EXPLOIT_EXPLORE,
        [c1, c2],
    )
    assert result is not None
    assert result.memory_id == "m2"


def test_probe_selection_skips_self_transfer():
    """Self-transfer-excluded candidates must never be probed."""
    c1 = TransferCandidateDecision(
        memory_id="m1", receiver_id="r1", task_id="t1",
        candidate_source="global",
        mu_tau=None, sigma_tau=None, lcb=None, ucb=None,
        eligible_for_context=False, selected_for_context=False,
        status="self_transfer_excluded",
    )
    c2 = _make_candidate("m2", "r1", lcb=-0.2, eligible=False)

    result = ProbeSelectionPolicy.select(
        RoutingMode.EXPLORE_ONLY,
        [c1, c2],
    )
    assert result is not None
    assert result.memory_id == "m2"


def test_select_probe_candidate_across_receivers():
    """select_probe_candidate must pick best global across all receiver plans."""
    plan_r1 = TransferRoutingPlan(
        receiver_id="r1",
        task_id="t1",
        routing_mode=RoutingMode.EXPLORE_ONLY,
        best_known_lcb=0.1,
        global_candidates=[_make_candidate("m1", "r1", lcb=0.4)],
    )
    plan_r2 = TransferRoutingPlan(
        receiver_id="r2",
        task_id="t1",
        routing_mode=RoutingMode.EXPLOIT_EXPLORE,
        best_known_lcb=0.2,
        global_candidates=[_make_candidate("m2", "r2", lcb=0.6)],
    )
    plan_r3 = TransferRoutingPlan(
        receiver_id="r3",
        task_id="t1",
        routing_mode=RoutingMode.EXPLOIT_ONLY,
        best_known_lcb=0.8,
        global_candidates=[_make_candidate("m3", "r3", lcb=0.9)],
    )

    rid, candidate = select_probe_candidate({
        "r1": plan_r1,
        "r2": plan_r2,
        "r3": plan_r3,
    })

    # r3 is EXPLOIT_ONLY, so m3 should not be selected
    # r2 has higher UCB than r1
    assert rid == "r2"
    assert candidate is not None
    assert candidate.memory_id == "m2"


def test_select_probe_candidate_all_exploit_only():
    """If all receivers are EXPLOIT_ONLY, no probe candidate."""
    plan = TransferRoutingPlan(
        receiver_id="r1",
        task_id="t1",
        routing_mode=RoutingMode.EXPLOIT_ONLY,
        best_known_lcb=0.8,
        global_candidates=[_make_candidate("m1", "r1", lcb=0.9)],
    )
    rid, candidate = select_probe_candidate({"r1": plan})
    assert rid is None
    assert candidate is None


# ---------------------------------------------------------------------------
# PostTaskTransferProbe tests
# ---------------------------------------------------------------------------


class MockEpisodeRunner:
    """Mock episode runner for testing post-task probes."""

    def __init__(self, scores: dict[tuple[str, str | None, int], float]):
        # Key: (receiver_id, memory_id, seed) -> score
        self._scores = scores
        self.call_log: list[tuple[str, str | None, int]] = []

    def run_episode(
        self,
        *,
        task: dict,
        receiver_id: str,
        memory_id: str | None,
        generation_seed: int,
    ) -> float:
        self.call_log.append((receiver_id, memory_id, generation_seed))
        key = (receiver_id, memory_id, generation_seed)
        return self._scores.get(key, 0.5)


def test_post_task_probe_collect_matched_evidence():
    """PostTaskTransferProbe.collect must produce matched tau."""
    runner = MockEpisodeRunner({
        ("r1", "m1", 0): 0.8,  # expose
        ("r1", None, 0): 0.5,  # withhold
        ("r1", "m1", 1): 0.7,  # expose
        ("r1", None, 1): 0.4,  # withhold
    })

    probe = PostTaskTransferProbe(runner, generation_seeds=[0, 1])
    evidence = probe.collect(
        task={"text": "test"},
        task_id="t1",
        task_position=5,
        receiver_id="r1",
        memory_id="m1",
    )

    assert isinstance(evidence, OnlineTransferEvidence)
    assert evidence.task_id == "t1"
    assert evidence.task_position == 5
    assert evidence.receiver_id == "r1"
    assert evidence.memory_id == "m1"

    # Expose: [0.8, 0.7], Withhold: [0.5, 0.4]
    # Differences: [0.3, 0.3], mean = 0.3, std = 0.0
    assert evidence.expose_scores == [0.8, 0.7]
    assert evidence.withhold_scores == [0.5, 0.4]
    assert evidence.observed_tau == pytest.approx(0.3)
    assert evidence.tau_std == pytest.approx(0.0)
    assert evidence.generation_seeds == [0, 1]


def test_post_task_probe_single_seed_no_std():
    """Single seed probe must have tau_std = None."""
    runner = MockEpisodeRunner({
        ("r1", "m1", 0): 0.7,
        ("r1", None, 0): 0.4,
    })

    probe = PostTaskTransferProbe(runner, generation_seeds=[0])
    evidence = probe.collect(
        task={"text": "test"},
        task_id="t1",
        task_position=3,
        receiver_id="r1",
        memory_id="m1",
    )

    assert evidence.observed_tau == pytest.approx(0.3)
    assert evidence.tau_std is None
    assert evidence.generation_seeds == [0]


def test_post_task_probe_shared_control_reuse():
    """Shared control must reuse cached withhold scores."""
    runner = MockEpisodeRunner({
        ("r1", "m1", 0): 0.8,  # expose for m1
        ("r1", None, 0): 0.5,  # withhold (will be cached)
        ("r1", "m2", 0): 0.9,  # expose for m2
        # Note: no ("r1", None, 0) call expected for m2 — should reuse cache
    })

    probe = PostTaskTransferProbe(runner, generation_seeds=[0])

    # First probe: collect for m1
    evidence1, control_cache = probe.collect_with_shared_control(
        task={"text": "test"},
        task_id="t1",
        task_position=5,
        receiver_id="r1",
        memory_id="m1",
    )
    assert evidence1.observed_tau == pytest.approx(0.3)
    assert control_cache == {0: 0.5}

    # Second probe: collect for m2, reusing control cache
    evidence2, control_cache = probe.collect_with_shared_control(
        task={"text": "test"},
        task_id="t1",
        task_position=5,
        receiver_id="r1",
        memory_id="m2",
        cached_withhold_scores=control_cache,
    )
    assert evidence2.observed_tau == pytest.approx(0.4)  # 0.9 - 0.5

    # Verify: only 3 episodes run (2 expose + 1 withhold)
    assert len(runner.call_log) == 3
    # Withhold should only be called once
    withhold_calls = [c for c in runner.call_log if c[1] is None]
    assert len(withhold_calls) == 1


def test_forward_only_invariant_documented():
    """PostTaskTransferProbe docstring must state forward-only invariant."""
    docstring = PostTaskTransferProbe.__doc__ or ""
    assert "forward-only" in docstring.lower() or "scored decision" in docstring.lower()
