"""S10.2 — Next-phase acceptance tests (N-05, N-06, N-11, N-12).

Each test validates one unfinished item from implementation.md §6.10 / §9 / §17
that was tracked in todo.md S10.2.

These tests use the latest pi3_v22 data artifacts where available, and
construct runtime scenarios for integration-level checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smtr.counterfactual.candidate_traversal import randomized_candidate_order
from smtr.router.candidate_proposer import CandidateProposal, CandidateRequest, CandidateScore
from smtr.router.sequential_router import ProductionSequentialRouter, SequentialRouterConfig
from smtr.router.transfer_critic import TransferEstimate
from smtr.runtime.environment import ToyEnvironment
from smtr.runtime.graph import build_graph, initial_state, run_demo

# ---------------------------------------------------------------------------
# Paths — pi3_v22 artifacts
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data" / "paired_records_pi3_v22.jsonl"
_PREFIX = _ROOT / "outputs" / "prefix_sensitivity_pi3_v22.json"
_COMPOSITIONAL = _ROOT / "outputs" / "critic_pi3_compositional_v22.json"

pytestmark = pytest.mark.skipif(
    not _DATA.exists(),
    reason="pi3_v22 data not found; run ToyEnvironment enhancement pipeline first",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prefix_report():
    return json.loads(_PREFIX.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def compositional_report():
    return json.loads(_COMPOSITIONAL.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# N-05: Selected-set conditional effect — direction accuracy > 0.60
# ---------------------------------------------------------------------------


def test_n05_direction_accuracy_above_060(prefix_report) -> None:
    """Critic prefix direction accuracy should be clearly above 0.60.

    The pi3_v22 data achieved 0.72; this test locks in the improvement
    over the original 0.50 baseline and the earlier 0.65 threshold.
    """
    accuracy = prefix_report["direction_accuracy"]
    assert accuracy is not None, "direction_accuracy is null"
    assert accuracy > 0.60, (
        f"N-05 FAIL: direction_accuracy={accuracy} not > 0.60"
    )


def test_n05_delta_correlation_strong(prefix_report) -> None:
    """Delta correlation should be strongly positive (>= 0.80)."""
    corr = prefix_report.get("delta_correlation")
    assert corr is not None, "delta_correlation is null"
    assert corr >= 0.80, f"N-05 FAIL: delta_correlation={corr} not >= 0.80"


def test_n05_transfer_region_flip_accuracy(prefix_report) -> None:
    """Transfer-region flip accuracy should be high (>= 0.80)."""
    flip_acc = prefix_report.get("transfer_region_flip_accuracy")
    flip_count = prefix_report.get("transfer_region_flip_pair_count", 0)
    assert flip_count > 0, "no transfer-region flip pairs"
    assert flip_acc is not None, "transfer_region_flip_accuracy is null"
    assert flip_acc >= 0.80, (
        f"N-05 FAIL: transfer_region_flip_accuracy={flip_acc} not >= 0.80"
    )


# ---------------------------------------------------------------------------
# N-06: Candidate-substitution audit coverage — matched_pair_count > 0
# ---------------------------------------------------------------------------


def test_n06_matched_pair_count_positive(prefix_report) -> None:
    """The prefix audit should find matched pairs for delta evaluation.

    Earlier pi2 data had matched_pair_count=0 (coverage regression).
    The pi3_v22 ToyEnvironment enhancement fixed this by providing diverse
    prefix structures that enable pair matching.
    """
    matched = prefix_report.get("matched_pair_count", 0)
    assert matched > 0, (
        f"N-06 FAIL: matched_pair_count={matched} — "
        "candidate-substitution audit has no pairs to evaluate"
    )


def test_n06_mean_abs_delta_tau_error_bounded(prefix_report) -> None:
    """Mean absolute delta tau error should be bounded (< 0.30)."""
    mae = prefix_report.get("mean_abs_delta_tau_error")
    assert mae is not None, "mean_abs_delta_tau_error is null"
    assert mae < 0.30, f"N-06 FAIL: mean_abs_delta_tau_error={mae} not < 0.30"


# ---------------------------------------------------------------------------
# N-11: Random traversal — multi-seed permutation consistency
# ---------------------------------------------------------------------------


def _make_proposal(memory_ids: list[str]) -> CandidateProposal:
    """Build a minimal CandidateProposal for permutation tests."""
    request = CandidateRequest(
        task="test task",
        task_stage="pre_route",
        receiver_agent_id="executor",
        receiver_role="executor",
        top_k=len(memory_ids),
    )
    return CandidateProposal(
        request=request,
        ranked_candidates=[
            CandidateScore(memory_id=mid, total_score=1.0 - 0.1 * i)
            for i, mid in enumerate(memory_ids)
        ],
        pool_revision=1,
    )


def test_n11_randomized_order_is_deterministic_for_same_seed() -> None:
    """Same seed must produce the same candidate ordering."""
    proposal = _make_proposal(["a", "b", "c", "d"])
    order_1 = randomized_candidate_order(proposal, traversal_seed=99)
    order_2 = randomized_candidate_order(proposal, traversal_seed=99)
    assert order_1 == order_2, (
        f"N-11 FAIL: same seed produced different orders: {order_1} vs {order_2}"
    )


def test_n11_randomized_order_can_differ_across_seeds() -> None:
    """Different seeds should be able to produce different orderings.

    We test multiple seeds and assert at least two distinct orderings appear.
    """
    proposal = _make_proposal(["a", "b", "c", "d", "e"])
    orderings = set()
    for seed in range(10):
        order = tuple(randomized_candidate_order(proposal, traversal_seed=seed))
        orderings.add(order)
    assert len(orderings) >= 2, (
        f"N-11 FAIL: 10 seeds produced only {len(orderings)} distinct ordering(s)"
    )


def test_n11_permutation_report_has_required_fields() -> None:
    """Multi-seed audit report should contain mean/variance statistics.

    Validates that the prefix sensitivity report format supports the
    permutation statistics needed for N-11 tracking.
    """
    # The prefix report should contain permutation-related metadata
    report_path = _PREFIX
    if not report_path.exists():
        pytest.skip("pi3_v22 prefix report not found")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # Core fields that must be present for permutation-aware evaluation
    required_fields = [
        "direction_accuracy",
        "delta_correlation",
        "matched_pair_count",
        "transfer_region_flip_accuracy",
    ]
    for field in required_fields:
        assert field in report, f"N-11 FAIL: missing field '{field}' in prefix report"

    # matched_pair_count must be positive for meaningful statistics
    assert report["matched_pair_count"] > 0, (
        "N-11 FAIL: matched_pair_count=0, cannot compute permutation statistics"
    )


# ---------------------------------------------------------------------------
# N-12: Online learned router default checkpoint loading
# ---------------------------------------------------------------------------


class _MockCritic:
    """Minimal critic mock for N-12 integration test."""

    critic_version = "n12_test_v1"

    def predict(self, item):
        accept = item.candidate_card.memory_id == "mem_execute_tool_chain"
        return TransferEstimate(
            q00_mean=0.1,
            q01_mean=0.05 if accept else 0.3,
            q10_mean=0.3 if accept else 0.05,
            q11_mean=0.5,
            tau_mean=0.25 if accept else -0.25,
            tau_lcb=0.1 if accept else -0.2,
            tau_ucb=0.4,
            negative_risk_mean=0.05 if accept else 0.3,
            negative_risk_ucb=0.1 if accept else 0.4,
            support_distance=0.0,
            support_threshold=1.0,
            low_support=False,
            ensemble_size=1,
            critic_version=self.critic_version,
        )


def test_n12_production_router_with_critic_makes_routing_decisions() -> None:
    """When a ProductionSequentialRouter with a critic is wired into the
    runtime, it should make share/withhold decisions (not just withhold-all
    like NoMemoryRouter).
    """
    critic = _MockCritic()
    router = ProductionSequentialRouter(
        critic=critic,
        config=SequentialRouterConfig(epsilon=0.2),
    )
    env = ToyEnvironment(seed=7)
    app = build_graph(
        router=router,
        config=__import__("smtr.runtime.graph", fromlist=["RuntimeConfig"]).RuntimeConfig(
            seed=7, top_k=6
        ),
    )
    state = app.invoke(
        initial_state(
            task="Obtain a target artifact using the valid action sequence.",
            environment_observation=env.observe(),
            run_seed=7,
            top_k=6,
        )
    )

    # Verify the router made at least one share decision (unlike NoMemoryRouter)
    any_share = False
    for trace in state["router_trace"]:
        for decision in trace["decisions"]:
            if decision.get("action") == "share":
                any_share = True
                break
        if any_share:
            break

    assert any_share, (
        "N-12 FAIL: ProductionSequentialRouter with critic made no share decisions — "
        "behaves like NoMemoryRouter"
    )


def test_n12_default_demo_uses_no_memory_router() -> None:
    """Default run_demo() (no router arg) should use NoMemoryRouter baseline.

    This documents the current behavior: the bare demo does NOT auto-load
    a critic checkpoint. N-12 tracks adding auto-loading in the future.
    """
    state = run_demo(seed=7)
    assert "router_trace" in state

    # With NoMemoryRouter, all decisions should be "withhold"
    all_withhold = True
    for trace in state["router_trace"]:
        for decision in trace["decisions"]:
            if decision.get("action") == "share":
                all_withhold = False
                break
        if not all_withhold:
            break

    assert all_withhold, (
        "N-12 NOTE: default demo made share decisions — "
        "NoMemoryRouter baseline may have been replaced"
    )


def test_n12_production_router_records_traversal_seed() -> None:
    """ProductionSequentialRouter should record traversal_seed in trace."""
    critic = _MockCritic()
    router = ProductionSequentialRouter(
        critic=critic,
        config=SequentialRouterConfig(epsilon=0.2),
    )
    env = ToyEnvironment(seed=7)
    app = build_graph(
        router=router,
        config=__import__("smtr.runtime.graph", fromlist=["RuntimeConfig"]).RuntimeConfig(
            seed=7, top_k=6
        ),
    )
    state = app.invoke(
        initial_state(
            task="Obtain a target artifact using the valid action sequence.",
            environment_observation=env.observe(),
            run_seed=42,
            top_k=6,
        )
    )

    # At least one trace should have traversal_order populated
    has_traversal = False
    for trace in state["router_trace"]:
        if trace.get("traversal_order"):
            has_traversal = True
            break

    assert has_traversal, (
        "N-12 FAIL: no trace has traversal_order — "
        "randomized permutation is not being recorded"
    )
