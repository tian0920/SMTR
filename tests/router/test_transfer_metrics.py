"""Tests for transfer_metrics module (RIMA-v2 §34, §36)."""

from __future__ import annotations

import pytest

from smtr.rima.transfer_metrics import (
    build_curve_records,
    compute_transfer_cost,
    compute_transfer_routing_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_diag(
    *,
    routing_mode: str = "exploit_only",
    global_triggered: bool = False,
    selected_source: str = "known",
    selected_mid: str | None = "m1",
    n_known: int = 5,
    n_global: int = 0,
    state_after: int = 10,
    receiver_id: str = "r1",
    global_ids: list[str] | None = None,
) -> dict:
    return {
        "routing_mode": routing_mode,
        "global_retrieval_triggered": global_triggered,
        "selected_source": selected_source,
        "selected_memory_id": selected_mid,
        "n_known_candidates_considered": n_known,
        "n_global_candidates_considered": n_global,
        "transfer_state_size_after": state_after,
        "receiver_id": receiver_id,
        "global_candidate_ids": global_ids or [],
    }


# ---------------------------------------------------------------------------
# Routing metrics
# ---------------------------------------------------------------------------


class TestTransferRoutingMetrics:
    def test_empty_diagnostics(self):
        result = compute_transfer_routing_metrics([])
        assert result["n_diagnostics"] == 0
        assert result["exploit_only_rate"] == 0.0

    def test_all_exploit_only(self):
        diags = [_make_diag(routing_mode="exploit_only") for _ in range(5)]
        result = compute_transfer_routing_metrics(diags)
        assert result["exploit_only_rate"] == 1.0
        assert result["exploit_explore_rate"] == 0.0
        assert result["explore_only_rate"] == 0.0

    def test_mixed_routing_modes(self):
        diags = [
            _make_diag(routing_mode="exploit_only"),
            _make_diag(routing_mode="exploit_explore"),
            _make_diag(routing_mode="explore_only"),
            _make_diag(routing_mode="exploit_only"),
        ]
        result = compute_transfer_routing_metrics(diags)
        assert result["exploit_only_rate"] == pytest.approx(0.5)
        assert result["exploit_explore_rate"] == pytest.approx(0.25)
        assert result["explore_only_rate"] == pytest.approx(0.25)

    def test_global_retrieval_rates(self):
        diags = [
            _make_diag(global_triggered=True),
            _make_diag(global_triggered=False),
            _make_diag(global_triggered=False),
            _make_diag(global_triggered=True),
        ]
        result = compute_transfer_routing_metrics(diags)
        assert result["global_retrieval_trigger_rate"] == pytest.approx(0.5)
        assert result["avoided_global_retrieval_rate"] == pytest.approx(0.5)

    def test_selection_source_rates(self):
        diags = [
            _make_diag(selected_source="known", selected_mid="m1"),
            _make_diag(selected_source="global", selected_mid="m2"),
            _make_diag(selected_source="none", selected_mid=None),
            _make_diag(selected_source="known", selected_mid="m3"),
        ]
        result = compute_transfer_routing_metrics(diags)
        assert result["known_memory_selection_rate"] == pytest.approx(0.5)
        assert result["global_memory_selection_rate"] == pytest.approx(0.25)
        assert result["no_memory_fallback_rate"] == pytest.approx(0.25)

    def test_known_memory_reuse_rate(self):
        diags = [
            _make_diag(selected_source="known", selected_mid="m1"),
            _make_diag(selected_source="known", selected_mid="m2"),
            _make_diag(selected_source="global", selected_mid="m3"),
            _make_diag(selected_source="none", selected_mid=None),
        ]
        # 2 known selections out of 3 tasks with selection = 2/3
        result = compute_transfer_routing_metrics(diags)
        assert result["known_memory_reuse_rate"] == pytest.approx(2 / 3)

    def test_candidate_counts(self):
        diags = [
            _make_diag(n_known=10, n_global=3),
            _make_diag(n_known=6, n_global=0),
        ]
        result = compute_transfer_routing_metrics(diags)
        assert result["mean_known_candidates_scored_per_task"] == pytest.approx(8.0)
        assert result["mean_global_candidates_scored_per_task"] == pytest.approx(1.5)

    def test_total_transfer_model_calls(self):
        diags = [
            _make_diag(n_known=5, n_global=2),
            _make_diag(n_known=3, n_global=0),
        ]
        result = compute_transfer_routing_metrics(diags)
        # total = 5 + 2 + 3 + 0 = 10
        assert result["total_transfer_model_calls"] == 10

    def test_transfer_state_sizes(self):
        diags = [
            _make_diag(state_after=5, receiver_id="r1"),
            _make_diag(state_after=8, receiver_id="r2"),
            _make_diag(state_after=10, receiver_id="r1"),
            _make_diag(state_after=12, receiver_id="r2"),
        ]
        result = compute_transfer_routing_metrics(diags)
        assert result["mean_transfer_state_size"] == pytest.approx(8.75)
        # Final sizes: last value per receiver
        assert result["final_transfer_state_size"]["r1"] == 10
        assert result["final_transfer_state_size"]["r2"] == 12

    def test_distinct_global_explored(self):
        diags = [
            _make_diag(global_ids=["m1", "m2", "m3"]),
            _make_diag(global_ids=["m2", "m4"]),
            _make_diag(global_ids=[]),
        ]
        result = compute_transfer_routing_metrics(diags)
        assert result["distinct_global_memories_explored"] == 4


# ---------------------------------------------------------------------------
# Transfer cost
# ---------------------------------------------------------------------------


class TestTransferCost:
    def test_empty_diagnostics(self):
        result = compute_transfer_cost([])
        cost = result["online_transfer_cost"]
        assert cost["known_candidate_critic_calls"] == 0
        assert cost["global_candidate_critic_calls"] == 0
        assert cost["online_intervention_episodes"] == 0

    def test_cost_aggregation(self):
        diags = [
            _make_diag(n_known=8, n_global=3, global_triggered=True),
            _make_diag(n_known=5, n_global=0, global_triggered=False),
            _make_diag(n_known=10, n_global=2, global_triggered=True),
        ]
        result = compute_transfer_cost(diags)
        cost = result["online_transfer_cost"]
        assert cost["known_candidate_critic_calls"] == 23
        assert cost["global_candidate_critic_calls"] == 5
        assert cost["global_retrieval_calls"] == 2
        assert cost["global_retrieval_avoided"] == 1
        assert cost["online_intervention_episodes"] == 0

    def test_intervention_always_zero(self):
        """Formal run must never have online interventions."""
        diags = [_make_diag() for _ in range(100)]
        result = compute_transfer_cost(diags)
        assert result["online_transfer_cost"]["online_intervention_episodes"] == 0


# ---------------------------------------------------------------------------
# Curve records (§34, §35)
# ---------------------------------------------------------------------------


class TestBuildCurveRecords:
    def test_empty_input(self):
        assert build_curve_records([], []) == []

    def test_basic_curve_record(self):
        diags = [
            _make_diag(
                global_triggered=True,
                selected_source="known",
                state_after=5,
                receiver_id="r1",
            ),
        ]
        # Inject task_position into the diag
        diags[0]["task_position"] = 0
        records = [{"task_position": 0, "task_score": 0.75}]
        curve = build_curve_records(diags, records)
        assert len(curve) == 1
        c = curve[0]
        assert c["task_position"] == 0
        assert c["global_retrieval_triggered"] is True
        assert c["selected_from_known"] is True
        assert c["transfer_state_size"] == 5
        assert c["task_score"] == 0.75

    def test_multi_receiver_aggregation(self):
        """Multiple receivers per position are merged."""
        diags = [
            {
                **_make_diag(
                    global_triggered=False,
                    selected_source="known",
                    state_after=3,
                    receiver_id="r1",
                ),
                "task_position": 2,
            },
            {
                **_make_diag(
                    global_triggered=True,
                    selected_source="global",
                    state_after=7,
                    receiver_id="r2",
                ),
                "task_position": 2,
            },
        ]
        records = [{"task_position": 2, "task_score": 0.6}]
        curve = build_curve_records(diags, records)
        assert len(curve) == 1
        c = curve[0]
        # any receiver triggered -> True
        assert c["global_retrieval_triggered"] is True
        # any receiver selected known -> True
        assert c["selected_from_known"] is True
        # sum of state sizes
        assert c["transfer_state_size"] == 10

    def test_multiple_positions_sorted(self):
        diags = []
        for pos in [3, 0, 1]:
            d = _make_diag(
                global_triggered=(pos == 0),
                selected_source="known" if pos > 0 else "none",
                state_after=pos + 1,
            )
            d["task_position"] = pos
            diags.append(d)
        records = [
            {"task_position": 0, "task_score": 0.5},
            {"task_position": 1, "task_score": 0.6},
            {"task_position": 3, "task_score": 0.8},
        ]
        curve = build_curve_records(diags, records)
        positions = [c["task_position"] for c in curve]
        assert positions == [0, 1, 3]
        assert curve[0]["global_retrieval_triggered"] is True
        assert curve[1]["selected_from_known"] is True
        assert curve[2]["task_score"] == 0.8

    def test_missing_task_score(self):
        diags = [{**_make_diag(), "task_position": 0}]
        curve = build_curve_records(diags, [])
        assert curve[0]["task_score"] is None
