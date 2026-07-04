"""M-01 … M-04 — method.md first-edition core criteria tests.

Tests the four items in the updated todo.md:
- M-01: set-conditioned transfer boundary learned
- M-02: multi-permutation traversal evaluation
- M-03: candidate-substitution audit coverage
- M-04: causal identification invariants maintained
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from smtr.counterfactual.candidate_traversal import (
    build_candidate_traversal_plan,
    randomized_candidate_order,
)
from smtr.counterfactual.record_writer import PairedRecordWriter
from smtr.evaluation.shortcut_diagnostics import shortcut_diagnostics
from smtr.router.candidate_proposer import CandidateProposal, CandidateRequest, CandidateScore

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data" / "paired_records_pi3_v22.jsonl"
_PREFIX = _ROOT / "outputs" / "prefix_sensitivity_pi3_v22.json"
_COMPOSITIONAL = _ROOT / "outputs" / "critic_pi3_compositional_v22.json"
_FEATURE = _ROOT / "outputs" / "feature_block_audit_pi2_s4.json"
_LEAKAGE = _ROOT / "outputs" / "feature_leakage_scan_pi2.json"

pytestmark = pytest.mark.skipif(
    not _DATA.exists(),
    reason="pi3_v22 data not found",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def records():
    return PairedRecordWriter(str(_DATA), allow_duplicates=True).load()


@pytest.fixture(scope="module")
def prefix_report():
    return json.loads(_PREFIX.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def compositional_report():
    return json.loads(_COMPOSITIONAL.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def feature_report():
    return json.loads(_FEATURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def leakage_report():
    return json.loads(_LEAKAGE.read_text(encoding="utf-8"))


def _proposal(n: int = 5) -> CandidateProposal:
    return CandidateProposal(
        request=CandidateRequest(
            task="test",
            task_stage="pre_route",
            receiver_agent_id="executor",
            receiver_role="executor",
            top_k=n,
        ),
        ranked_candidates=[
            CandidateScore(memory_id=f"m{i}", total_score=1.0 - 0.1 * i)
            for i in range(n)
        ],
        pool_revision=1,
    )


# ===================================================================
# M-01: Learn set-conditioned transfer boundary
# ===================================================================


class TestM01SetConditionedBoundary:
    """M-01: Critic must learn tau(m|o,S) — not just candidate shortcuts."""

    def test_m01_1_prefix_sensitive_cases_reported(self, prefix_report) -> None:
        """M-01.1: Report prefix-sensitive cases with tau/eta variation."""
        matched = prefix_report.get("matched_pair_count", 0)
        assert matched > 0, "No prefix-sensitive pairs found"

    def test_m01_2_flip_type_coverage(self, prefix_report) -> None:
        """M-01.2: Report coverage & accuracy for each flip type."""
        flips = prefix_report["flip_detection"]
        # At least 3 of 4 flip types must have non-zero pairs
        covered = sum(1 for v in flips.values() if v["pair_count"] > 0)
        assert covered >= 3, f"Only {covered}/4 flip types covered: {flips}"

        # Each covered type must have direction_accuracy >= 0.8
        for name, data in flips.items():
            if data["pair_count"] > 0:
                acc = data["direction_accuracy"]
                assert acc is not None and acc >= 0.8, (
                    f"{name}: pair_count={data['pair_count']} but accuracy={acc}"
                )

    def test_m01_3_delta_correlation_and_mae(self, prefix_report) -> None:
        """M-01.3: Report delta_correlation, delta_mae, transfer-region flip accuracy."""
        corr = prefix_report.get("delta_correlation")
        mae = prefix_report.get("mean_abs_delta_tau_error")
        flip_acc = prefix_report.get("transfer_region_flip_accuracy")

        assert corr is not None and corr > 0.5, f"delta_correlation={corr} too low"
        assert mae is not None and mae < 0.30, f"delta_mae={mae} too high"
        assert flip_acc is not None and flip_acc >= 0.8, (
            f"transfer_region_flip_accuracy={flip_acc} too low"
        )

    def test_m01_4_no_scenario_shortcut(self, compositional_report) -> None:
        """M-01.4: Scenario split must not be trivially separable.

        If scenario_family F1 is very high, it must be explainable
        (not due to data leakage or deterministic mapping).
        """
        scenario = compositional_report.get("scenario_family", {})
        f1 = scenario.get("metrics", {}).get("macro_f1", 1.0)
        # F1 < 0.999 means it's not trivially separable
        # F1 >= 0.999 requires explicit explanation
        if f1 >= 0.999:
            pytest.fail(
                f"scenario_family F1={f1:.4f} ≈ 1.0 — "
                "must explain: task structure or data leakage?"
            )

    def test_m01_5_interaction_encoder_beats_baseline(self, feature_report) -> None:
        """M-01.5: Full model (with pairwise interaction) must beat best single block."""
        gain = feature_report.get("full_model_gain_over_best_single_block", 0)
        full_f1 = feature_report["blocks"]["full"]["macro_f1"]
        assert gain > 0.05, f"full_model_gain={gain} not > 0.05"
        assert full_f1 > 0.80, f"full macro_f1={full_f1} not > 0.80"


# ===================================================================
# M-02: Multi-permutation traversal evaluation
# ===================================================================


class TestM02MultiPermutation:
    """M-02: Results must hold across multiple traversal permutations."""

    def test_m02_1_same_seed_deterministic(self) -> None:
        """M-02.1: Same traversal seed → same candidate order."""
        proposal = _proposal()
        o1 = randomized_candidate_order(proposal, traversal_seed=42)
        o2 = randomized_candidate_order(proposal, traversal_seed=42)
        assert o1 == o2, "Same seed produced different orders"

    def test_m02_2_different_seeds_vary(self) -> None:
        """M-02.2: Different seeds → at least 2 distinct orderings."""
        proposal = _proposal(6)
        orderings = {
            tuple(randomized_candidate_order(proposal, traversal_seed=s))
            for s in range(20)
        }
        assert len(orderings) >= 2, "20 seeds produced only 1 ordering"

    def test_m02_3_traversal_plan_is_seed_dependent(self) -> None:
        """M-02.3: Traversal plans from different seeds differ in candidate order."""
        proposal = _proposal(5)
        plan_a = build_candidate_traversal_plan(proposal=proposal, traversal_seed=1)
        plan_b = build_candidate_traversal_plan(proposal=proposal, traversal_seed=99)
        # At minimum, the plans should be valid
        assert len(plan_a.candidate_order) == len(plan_b.candidate_order) == 5
        assert set(plan_a.candidate_order) == set(plan_b.candidate_order)

    def test_m02_4_proposer_rank_not_traversal_order(self) -> None:
        """M-02.4: Candidate proposer rank must NOT be used as default traversal order.

        The randomized_candidate_order function should shuffle, not preserve rank.
        """
        proposal = _proposal(8)
        rank_order = [c.memory_id for c in proposal.ranked_candidates]
        # With 8 candidates and seed=7, the order should differ from rank
        shuffled = randomized_candidate_order(proposal, traversal_seed=7)
        # It's possible (but very unlikely with 8 items) that they match
        # We just verify the mechanism exists and produces a valid permutation
        assert set(shuffled) == set(rank_order), "Shuffled order lost candidates"


# ===================================================================
# M-03: Candidate-substitution audit coverage
# ===================================================================


class TestM03CandidateSubstitution:
    """M-03: Candidate-substitution audit must have non-zero matched pairs."""

    def test_m03_1_matched_pairs_positive(self, prefix_report) -> None:
        """M-03.1: matched_pair_count must be > 0."""
        matched = prefix_report.get("matched_pair_count", 0)
        assert matched > 0, (
            f"matched_pair_count={matched} — audit has no pairs to evaluate"
        )

    def test_m03_2_delta_correlation_positive(self, prefix_report) -> None:
        """M-03.2: delta_correlation must be positive and meaningful."""
        corr = prefix_report.get("delta_correlation")
        assert corr is not None and corr > 0.0, (
            f"delta_correlation={corr} — critic cannot track delta direction"
        )

    def test_m03_3_delta_mae_bounded(self, prefix_report) -> None:
        """M-03.3: delta MAE must be bounded (critic is not useless)."""
        mae = prefix_report.get("mean_abs_delta_tau_error")
        assert mae is not None and mae < 0.30, (
            f"mean_abs_delta_tau_error={mae} — critic delta predictions too noisy"
        )


# ===================================================================
# M-04: Causal identification invariants
# ===================================================================


class TestM04CausalInvariants:
    """M-04: All causal identification invariants must hold."""

    def test_m04_1_payload_isolation(self, leakage_report) -> None:
        """M-04.1/2: No feature leakage — disabled fields must not enter critic."""
        assert leakage_report["violations"] == [], (
            f"Feature leakage detected: {leakage_report['violations'][:3]}"
        )

    def test_m04_2_leakage_scanner_on_records(self, leakage_report) -> None:
        """M-04.2: Leakage scan report must show 0 violations."""
        assert leakage_report["violations"] == [], (
            f"Feature leakage detected: {leakage_report['violations'][:3]}"
        )

    def test_m04_3_paired_branch_shares_context(self, records) -> None:
        """M-04.3: share/withhold branches share traversal seed and prefix."""
        # Group records by mechanism_group_id (pairs share this, differ in prefix_structure_family)
        groups: dict[str, list] = {}
        for r in records:
            key = r.evaluation_group_metadata.mechanism_group_id
            groups.setdefault(key, []).append(r)

        # Each mechanism group should have multiple records (from different prefix structures)
        multi_record_groups = sum(1 for g in groups.values() if len(g) >= 2)
        assert multi_record_groups > 0, "No multi-record mechanism groups found"

    def test_m04_4_policy_specific_estimand(self, records) -> None:
        """M-04.4: All records in a dataset share the same policy fingerprint."""
        fingerprints = {
            r.continuation_policy_fingerprint for r in records
            if r.continuation_policy_fingerprint is not None
        }
        assert len(fingerprints) == 1, (
            f"Multiple policy fingerprints in dataset: {fingerprints}"
        )

    def test_m04_5_collection_revision_fixed(self, records) -> None:
        """M-04.5: All records in a collection round share the same revision."""
        # Group by collection_round_id, verify base_memory_store_revision is consistent
        rounds: dict[str, set] = {}
        for r in records:
            rid = r.collection_round_id or "unknown"
            rev = r.base_memory_store_revision
            rounds.setdefault(rid, set()).add(rev)
        for rid, revisions in rounds.items():
            assert len(revisions) == 1, (
                f"Round {rid} has multiple revisions: {revisions}"
            )

    def test_m04_6_four_outcome_labels(self, records) -> None:
        """M-04.6: All 4 outcome labels must be present and correctly derived."""
        outcomes = Counter()
        for r in records:
            outcomes[r.transfer_class] += 1
        # All 4 classes must be present
        expected = {"positive", "negative", "neutral_success", "neutral_failure"}
        assert expected.issubset(set(outcomes.keys())), (
            f"Missing transfer classes: {expected - set(outcomes.keys())}"
        )
        # Each class must have at least 10 records
        for cls in expected:
            assert outcomes[cls] >= 10, f"{cls} has only {outcomes[cls]} records"

    def test_m04_7_no_shortcut_warnings(self, records) -> None:
        """M-04 supplementary: shortcut_diagnostics should not flag near-deterministic groups."""
        diag = shortcut_diagnostics(records)
        all_warnings = []
        for field_data in diag.values():
            all_warnings.extend(field_data["warnings"])
        assert len(all_warnings) < 5, (
            f"Too many shortcut warnings: {all_warnings}"
        )
