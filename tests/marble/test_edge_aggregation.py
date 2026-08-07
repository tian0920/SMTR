"""Commit 4: multi-seed treatment edges
and edge-level empirical aggregation (q00/q01/q10/q11, tau, eta)."""

import pytest

from smtr.marble.real_pairs import (
    aggregate_edge_records,
    compute_edge_id,
    stable_hash,
)


def _make_record(
    edge_id: str,
    y_share: bool,
    y_withhold: bool,
    *,
    valid: bool = True,
    replicate_index: int = 0,
    task_id: str = "t1",
    receiver: str = "r1",
    memory: str = "m1",
) -> dict:
    return {
        "edge_id": edge_id,
        "replicate_id": f"{edge_id}:r{replicate_index}",
        "treatment_definition_version": "v1",
        "task_id": task_id,
        "receiver_agent_id": receiver,
        "candidate_memory_id": memory,
        "valid": valid,
        "share": {"team_success": y_share},
        "withhold": {"team_success": y_withhold},
    }


class TestEdgeIdentity:
    def test_edge_id_is_stable_and_order_sensitive(self):
        a = compute_edge_id("t1", "r1", "m1")
        assert a == compute_edge_id("t1", "r1", "m1")
        assert a != compute_edge_id("t1", "r1", "m2")
        assert a != compute_edge_id("r1", "t1", "m1")

    def test_stable_hash_is_deterministic(self):
        assert stable_hash("x", 3) == stable_hash("x", 3)
        assert stable_hash("x", 3) != stable_hash(3, "x")


class TestEdgeAggregation:
    def test_edge_level_empirical_probabilities_sum_to_one(self):
        edge_id = compute_edge_id("t1", "r1", "m1")
        records = [
            _make_record(edge_id, True, False, replicate_index=0),   # q10
            _make_record(edge_id, False, True, replicate_index=1),   # q01
            _make_record(edge_id, True, True, replicate_index=2),    # q11
            _make_record(edge_id, False, False, replicate_index=3),  # q00
        ]
        agg = aggregate_edge_records(records)
        assert len(agg) == 1
        a = agg[0]
        total = (
            a["q00_empirical"]
            + a["q01_empirical"]
            + a["q10_empirical"]
            + a["q11_empirical"]
        )
        assert total == pytest.approx(1.0)
        assert a["n_replicates"] == 4

    def test_empirical_tau_and_eta_are_correct(self):
        edge_id = compute_edge_id("t2", "r2", "m2")
        records = [
            _make_record(edge_id, True, False),   # q10
            _make_record(edge_id, True, False, replicate_index=1),  # q10
            _make_record(edge_id, False, True, replicate_index=2),  # q01
            _make_record(edge_id, True, True, replicate_index=3),   # q11
        ]
        agg = aggregate_edge_records(records)
        assert len(agg) == 1
        a = agg[0]
        assert a["q10_empirical"] == pytest.approx(0.5)
        assert a["q01_empirical"] == pytest.approx(0.25)
        assert a["tau_empirical"] == pytest.approx(a["q10_empirical"] - a["q01_empirical"])
        assert a["eta_empirical"] == pytest.approx(a["q01_empirical"])

    def test_invalid_records_are_excluded_from_aggregation(self):
        edge_id = compute_edge_id("t3", "r3", "m3")
        records = [
            _make_record(edge_id, True, True),
            _make_record(edge_id, False, True, valid=False, replicate_index=1),
        ]
        agg = aggregate_edge_records(records)
        assert len(agg) == 1
        assert agg[0]["n_replicates"] == 1
        assert agg[0]["n_attempted"] == 2
        assert agg[0]["q11_empirical"] == pytest.approx(1.0)

    def test_edges_are_aggregated_separately(self):
        e1 = compute_edge_id("t1", "r1", "m1")
        e2 = compute_edge_id("t1", "r1", "m2")
        records = [
            _make_record(e1, True, False),
            _make_record(e2, False, True, memory="m2"),
        ]
        agg = aggregate_edge_records(records)
        assert len(agg) == 2
        by_edge = {a["edge_id"]: a for a in agg}
        assert by_edge[e1]["tau_empirical"] == pytest.approx(1.0)
        assert by_edge[e2]["tau_empirical"] == pytest.approx(-1.0)


