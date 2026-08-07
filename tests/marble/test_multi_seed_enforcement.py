"""Multi-seed enforcement, replicate identity
and edge-level empirical probability aggregation (checklist chapter 5)."""

import pytest

from smtr.marble.real_pairs import (
    MIN_SEEDS,
    EdgeTransferEstimate,
    compute_edge_id,
    compute_edge_transfer_estimates,
    compute_replicate_id,
    generate_candidate_level_pairs,
    stable_hash,
)


def _make_record(
    edge_id: str,
    y_share: bool,
    y_withhold: bool,
    *,
    seed: int = 0,
    valid: bool = True,
) -> dict:
    return {
        "edge_id": edge_id,
        "replicate_id": compute_replicate_id(edge_id, seed),
        "treatment_definition_version": "v1",
        "task_id": "t1",
        "receiver_agent_id": "r1",
        "candidate_memory_id": "m1",
        "generation_seed": seed,
        "valid": valid,
        "share": {"team_success": y_share},
        "withhold": {"team_success": y_withhold},
    }


class TestMultiSeedEnforcement:
    def test_formal_mode_requires_at_least_five_seeds(self):
        assert MIN_SEEDS["formal"] == 5
        with pytest.raises(ValueError, match="at least 5"):
            generate_candidate_level_pairs(
                marble_root=None,  # never touched: check happens first
                dataset_manifest_path=None,
                split_manifest_path=None,
                split="test",
                candidate_manifest_path=None,
                memory_pool_path=None,
                generation_seeds=[0, 1, 2, 3],
                output_dir=None,
                experiment_mode="formal",
            )

    def test_pilot_mode_requires_at_least_three_seeds(self):
        assert MIN_SEEDS["pilot"] == 3
        with pytest.raises(ValueError, match="at least 3"):
            generate_candidate_level_pairs(
                marble_root=None,
                dataset_manifest_path=None,
                split_manifest_path=None,
                split="test",
                candidate_manifest_path=None,
                memory_pool_path=None,
                generation_seeds=[0, 1],
                output_dir=None,
                experiment_mode="pilot",
            )
        # Duplicate seeds do not count as distinct seeds.
        with pytest.raises(ValueError, match="at least 3"):
            generate_candidate_level_pairs(
                marble_root=None,
                dataset_manifest_path=None,
                split_manifest_path=None,
                split="test",
                candidate_manifest_path=None,
                memory_pool_path=None,
                generation_seeds=[0, 0, 1],
                output_dir=None,
                experiment_mode="pilot",
            )


class TestReplicateIdentity:
    def test_replicate_ids_are_unique_within_edge(self):
        edge_id = compute_edge_id("t1", "r1", "m1")
        seeds = [0, 1, 2, 3, 4]
        replicate_ids = [compute_replicate_id(edge_id, s) for s in seeds]
        assert len(replicate_ids) == len(set(replicate_ids))
        # replicate_id = stable_hash(edge_id, generation_seed)
        expected = f"rep_{stable_hash(edge_id, 0):016x}"
        assert replicate_ids[0] == expected
        # Deterministic and seed-sensitive.
        assert compute_replicate_id(edge_id, 0) == compute_replicate_id(edge_id, 0)
        assert compute_replicate_id(edge_id, 0) != compute_replicate_id(edge_id, 1)




class TestEdgeLevelAggregation:
    def test_edge_probabilities_sum_to_one(self):
        edge_id = compute_edge_id("t1", "r1", "m1")
        records = [
            _make_record(edge_id, True, False, seed=0),    # q10
            _make_record(edge_id, False, True, seed=1),    # q01
            _make_record(edge_id, True, True, seed=2),     # q11
            _make_record(edge_id, False, False, seed=3),   # q00
        ]
        estimates = compute_edge_transfer_estimates(records)
        assert len(estimates) == 1
        est = estimates[0]
        assert isinstance(est, EdgeTransferEstimate)
        total = (
            est.q00_empirical
            + est.q01_empirical
            + est.q10_empirical
            + est.q11_empirical
        )
        assert total == pytest.approx(1.0)
        assert est.n_replicates == 4
        assert est.tau_empirical == pytest.approx(
            est.q10_empirical - est.q01_empirical
        )
        assert est.eta_empirical == pytest.approx(est.q01_empirical)
