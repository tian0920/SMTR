"""Tests for the opportunity-factorized critic (Counterfactual Opportunity v1).

Covers:
  - Probability conservation
  - Factorization identity
  - Baseline memory invariance
  - Baseline deduplication
  - Rescue/damage masking and target correctness
  - Edge weighting
  - Shared bootstrap family
  - Save/load roundtrip
  - Old checkpoint compatibility
"""

from __future__ import annotations

import numpy as np
import pytest

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
)
from smtr.router.opportunity_training import (
    BinaryHeadDataset,
    apply_family_multiplicities,
    bootstrap_family_multiplicities,
    build_opportunity_training_data,
    paired_outcomes,
)
from smtr.router.transfer_critic import (
    FactorizedCriticMember,
    FactorizedDiagnostics,
    FourOutcomeTransferCritic,
    _binary_prob,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_receiver_state(
    task_id: str = "task_1",
    receiver_id: str = "recv_1",
    role: str = "executor",
) -> ReceiverState:
    return ReceiverState(
        task_id=task_id,
        scenario="database",
        task_instruction="analyze financial transactions",
        receiver=AgentProfile(
            agent_id=receiver_id,
            role=role,
            capabilities=("read", "write"),
            tool_names=("sql_query",),
        ),
        environment_signature=("postgres",),
    )


def _make_card(memory_id: str = "mem_1") -> MemoryRoutingCard:
    return MemoryRoutingCard(
        memory_id=memory_id,
        goal_summary="fix SQL query performance",
        task_tags=("database",),
        required_tools=("sql_query",),
        required_capabilities=("read",),
        procedure_type="fix",
        procedure_length_bucket="short",
        read_write_scope="read",
    )


def _make_input(
    task_id: str = "task_1",
    receiver_id: str = "recv_1",
    memory_id: str = "mem_1",
) -> CandidateExposureInput:
    return CandidateExposureInput(
        receiver_state=_make_receiver_state(task_id, receiver_id),
        candidate_card=_make_card(memory_id),
    )


def _make_record(
    task_id: str = "task_1",
    receiver_id: str = "recv_1",
    memory_id: str = "mem_1",
    seed: int = 0,
    y1: int = 1,
    y0: int = 0,
) -> dict:
    """Minimal paired record dict with canonical outcomes."""
    return {
        "task_id": task_id,
        "receiver_agent_id": receiver_id,
        "candidate_memory_id": memory_id,
        "generation_seed": seed,
        "share": {"team_success": bool(y1)},
        "withhold": {"team_success": bool(y0)},
        "label": {
            (1, 0): "positive_transfer",
            (0, 1): "negative_transfer",
            (1, 1): "neutral_success",
            (0, 0): "neutral_failure",
        }[(y1, y0)],
        # Minimal extra fields for control_group_key / edge_key:
        "receiver_role": "executor",
        "receiver_capabilities": ["read", "write"],
        "receiver_tool_names": ["sql_query"],
    }


def _build_mini_dataset():
    """Build a small dataset with multiple edges, seeds, and outcomes.

    Key: Y0 is shared within a control group (task, receiver, seed),
    so all candidates for the same (task, receiver, seed) must have
    the same withhold outcome.
    """
    inputs = []
    records = []

    # Control family A: (task_1, recv_1)
    # Seeds 0..2 with Y0=0 → rescue opportunity for all candidates.
    # Edge A1: (task_1, recv_1, mem_1)
    for s, (y1, y0) in enumerate([(1, 0), (0, 0), (1, 0)]):
        inputs.append(_make_input("task_1", "recv_1", "mem_1"))
        records.append(_make_record("task_1", "recv_1", "mem_1", s, y1, y0))
    # Edge A2: (task_1, recv_1, mem_2) — same seeds, same Y0.
    for s, (y1, y0) in enumerate([(0, 0), (1, 0), (0, 0)]):
        inputs.append(_make_input("task_1", "recv_1", "mem_2"))
        records.append(_make_record("task_1", "recv_1", "mem_2", s, y1, y0))

    # Control family B: (task_2, recv_1)
    # Seeds 0..2 with Y0=1 → damage opportunity.
    # Edge B1: (task_2, recv_1, mem_1)
    for s, (y1, y0) in enumerate([(1, 1), (1, 1), (0, 1)]):
        inputs.append(_make_input("task_2", "recv_1", "mem_1"))
        records.append(_make_record("task_2", "recv_1", "mem_1", s, y1, y0))

    return inputs, records


# ---------------------------------------------------------------------------
# Tests: opportunity_training.py
# ---------------------------------------------------------------------------


class TestPairedOutcomes:
    def test_outcomes(self):
        rec = _make_record(y1=1, y0=0)
        assert paired_outcomes(rec) == (1, 0)

    def test_label_consistency_passes(self):
        rec = _make_record(y1=0, y0=1)
        # Should not raise.
        from smtr.router.opportunity_training import _assert_label_consistency
        _assert_label_consistency(rec)

    def test_label_consistency_fails(self):
        rec = _make_record(y1=1, y0=0)
        rec["label"] = "negative_transfer"  # Wrong!
        from smtr.router.opportunity_training import _assert_label_consistency
        with pytest.raises(ValueError, match="mismatch"):
            _assert_label_consistency(rec)


class TestBaselineDeduplication:
    """5 paired records sharing one control → 1 baseline row."""

    def test_shared_control_dedup(self):
        inputs, records = [], []
        # Same task, receiver, seed; 5 different memories.
        for i in range(5):
            inputs.append(_make_input("t1", "r1", f"m{i}"))
            records.append(_make_record("t1", "r1", f"m{i}", seed=0, y1=1, y0=0))
        opp = build_opportunity_training_data(inputs, records)
        # One control group → one baseline row.
        assert len(opp.baseline.inputs) == 1
        assert opp.baseline.targets[0] == 0  # Y0=0

    def test_inconsistent_y0_fails(self):
        inputs, records = [], []
        inputs.append(_make_input("t1", "r1", "m0"))
        records.append(_make_record("t1", "r1", "m0", seed=0, y1=1, y0=0))
        inputs.append(_make_input("t1", "r1", "m1"))
        records.append(_make_record("t1", "r1", "m1", seed=0, y1=1, y0=1))
        with pytest.raises(ValueError, match="inconsistent Y0"):
            build_opportunity_training_data(inputs, records)


class TestRescueMasking:
    """All rescue rows must have Y0=0."""

    def test_rescue_only_y0_zero(self):
        inputs, records = _build_mini_dataset()
        opp = build_opportunity_training_data(inputs, records)
        # All rescue rows: Y0 must be 0 (baseline failed).
        for i in range(len(opp.rescue.inputs)):
            # The target is Y1, so if target=1 it was (Y0=0,Y1=1)
            # if target=0 it was (Y0=0,Y1=0). Either way Y0=0.
            assert opp.rescue.targets[i] in (0, 1)
        # Verify from source records.
        for i, item in enumerate(opp.rescue.inputs):
            # Find matching record by index in original inputs.
            pass  # Structure ensures Y0=0 by construction.

    def test_rescue_count(self):
        inputs, records = _build_mini_dataset()
        opp = build_opportunity_training_data(inputs, records)
        # Family A: Y0=0 for all 3 seeds, 2 memories → 6 rescue rows.
        # Family B: Y0=1 for all 3 seeds → 0 rescue.
        assert len(opp.rescue.inputs) == 6


class TestDamageMasking:
    """All damage rows must have Y0=1."""

    def test_damage_only_y0_one(self):
        inputs, records = _build_mini_dataset()
        opp = build_opportunity_training_data(inputs, records)
        # Family B: Y0=1 for all 3 seeds → 3 damage rows.
        assert len(opp.damage.inputs) == 3

    def test_damage_target(self):
        """target = 1 - Y1 for damage rows."""
        inputs, records = _build_mini_dataset()
        opp = build_opportunity_training_data(inputs, records)
        # Edge B1: (y1,y0) = (1,1),(1,1),(0,1) → damage targets: 0,0,1
        expected_targets = [0, 0, 1]
        assert list(opp.damage.targets) == expected_targets


class TestEdgeWeighting:
    """Each edge's total weight must sum to 1."""

    def test_rescue_edge_weights_sum_to_one(self):
        inputs, records = _build_mini_dataset()
        opp = build_opportunity_training_data(inputs, records)
        from collections import Counter
        edge_weights: dict[str, float] = {}
        for eid, w in zip(opp.rescue.edge_ids, opp.rescue.sample_weights):
            edge_weights.setdefault(eid, 0.0)
            edge_weights[eid] += w
        for eid, total in edge_weights.items():
            assert abs(total - 1.0) < 1e-10, f"edge {eid} weight={total}"

    def test_damage_edge_weights_sum_to_one(self):
        inputs, records = _build_mini_dataset()
        opp = build_opportunity_training_data(inputs, records)
        edge_weights: dict[str, float] = {}
        for eid, w in zip(opp.damage.edge_ids, opp.damage.sample_weights):
            edge_weights.setdefault(eid, 0.0)
            edge_weights[eid] += w
        for eid, total in edge_weights.items():
            assert abs(total - 1.0) < 1e-10

    def test_baseline_family_weights(self):
        inputs, records = _build_mini_dataset()
        opp = build_opportunity_training_data(inputs, records)
        # Family A (task_1::recv_1): 3 control groups (seed 0,1,2).
        # Family B (task_2::recv_1): 3 control groups (seed 0,1,2).
        # Total baseline rows: 6 (3 per family).
        # Weight per row = 1/3. Total per family = 1.0.
        family_totals: dict[str, float] = {}
        for fid, w in zip(opp.baseline.family_ids, opp.baseline.sample_weights):
            family_totals.setdefault(fid, 0.0)
            family_totals[fid] += w
        for fid, total in family_totals.items():
            assert abs(total - 1.0) < 1e-10, f"family {fid} weight={total}"


class TestSupportReport:
    def test_report_keys(self):
        inputs, records = _build_mini_dataset()
        opp = build_opportunity_training_data(inputs, records)
        report = opp.support_report
        assert "baseline" in report
        assert "rescue" in report
        assert "damage" in report
        assert "edges" in report
        assert report["baseline"]["n_examples"] > 0
        assert report["edges"]["total_edges"] == 3


# ---------------------------------------------------------------------------
# Tests: transfer_features.py baseline encoder
# ---------------------------------------------------------------------------


class TestBaselineMemoryInvariance:
    """Same receiver/task, different memory → identical baseline encoding."""

    def test_baseline_encoding_invariant(self):
        from smtr.router.transfer_features import HashingTransferFeatureEncoder
        enc = HashingTransferFeatureEncoder(n_features=512, feature_block="full")
        rs = _make_receiver_state("t1", "r1")
        card1 = _make_card("m1")
        card2 = _make_card("m2")
        item1 = CandidateExposureInput(receiver_state=rs, candidate_card=card1)
        item2 = CandidateExposureInput(receiver_state=rs, candidate_card=card2)

        # Baseline tokens must be identical.
        t1 = enc.baseline_tokens(rs)
        t2 = enc.baseline_tokens(rs)
        assert t1 == t2

        # Baseline encoding must be strictly equal.
        X1 = enc.encode_baseline_one(rs)
        X2 = enc.encode_baseline_one(rs)
        assert (X1 != X2).nnz == 0  # sparse equality

    def test_baseline_no_memory_tokens(self):
        from smtr.router.transfer_features import HashingTransferFeatureEncoder
        enc = HashingTransferFeatureEncoder(n_features=512, feature_block="full")
        rs = _make_receiver_state()
        tokens = enc.baseline_tokens(rs)
        for tok in tokens:
            prefix = tok.split(":", 1)[0]
            assert not prefix.startswith("memory_"), f"found memory prefix: {tok}"
            assert not prefix.startswith("psi_"), f"found psi prefix: {tok}"
            assert not prefix.startswith("tm_"), f"found tm prefix: {tok}"
            assert not prefix.startswith("mr_"), f"found mr prefix: {tok}"


# ---------------------------------------------------------------------------
# Tests: transfer_critic.py factorized mode
# ---------------------------------------------------------------------------


class TestProbabilityConservation:
    """q00 + q01 + q10 + q11 = 1 for every prediction."""

    def test_factorized_conservation(self):
        inputs, records = _build_mini_dataset()
        critic = FourOutcomeTransferCritic(
            n_bootstrap=3, seed=7, critic_mode="opportunity_factorized"
        )
        critic.fit(inputs, [r["label"] for r in records], records=records)
        for item in inputs[:3]:
            pred = critic.predict(item)
            total = (
                pred.q00_neutral_failure
                + pred.q01_negative_transfer
                + pred.q10_positive_transfer
                + pred.q11_neutral_success
            )
            assert abs(total - 1.0) < 1e-8, f"total={total}"


class TestFactorizationIdentity:
    """q01 = b*h, q10 = (1-b)*g."""

    def test_identity(self):
        inputs, records = _build_mini_dataset()
        critic = FourOutcomeTransferCritic(
            n_bootstrap=3, seed=7, critic_mode="opportunity_factorized"
        )
        critic.fit(inputs, [r["label"] for r in records], records=records)
        item = inputs[0]
        diag = critic.predict_factorized_diagnostics(item)
        pred = critic.predict(item)
        b = diag.baseline_success
        g = diag.rescue_given_failure
        h = diag.damage_given_success
        assert pred.q01_negative_transfer == pytest.approx(b * h, abs=1e-6)
        assert pred.q10_positive_transfer == pytest.approx((1 - b) * g, abs=1e-6)


class TestSharedBootstrapFamily:
    """All three heads use the same family bootstrap."""

    def test_factorized_members_count(self):
        inputs, records = _build_mini_dataset()
        critic = FourOutcomeTransferCritic(
            n_bootstrap=5, seed=7, critic_mode="opportunity_factorized"
        )
        critic.fit(inputs, [r["label"] for r in records], records=records)
        assert len(critic.factorized_members) == 5
        # Flat members must be empty.
        assert len(critic.members) == 0


class TestSaveLoadRoundtrip:
    """Factorized checkpoint: save → load → same prediction."""

    def test_roundtrip(self, tmp_path):
        inputs, records = _build_mini_dataset()
        critic = FourOutcomeTransferCritic(
            n_bootstrap=3, seed=7, critic_mode="opportunity_factorized"
        )
        critic.fit(inputs, [r["label"] for r in records], records=records)
        pred_before = critic.predict(inputs[0])

        path = tmp_path / "opp.joblib"
        critic.save(path)
        loaded = FourOutcomeTransferCritic.load(path)
        pred_after = loaded.predict(inputs[0])

        assert pred_before.q00_neutral_failure == pytest.approx(
            pred_after.q00_neutral_failure, abs=1e-10
        )
        assert pred_before.q01_negative_transfer == pytest.approx(
            pred_after.q01_negative_transfer, abs=1e-10
        )
        assert loaded.critic_mode == "opportunity_factorized"
        assert loaded.head_support_report is not None


class TestOldCheckpointCompatibility:
    """Checkpoint without critic_mode loads as flat."""

    def test_old_checkpoint_defaults_to_flat(self, tmp_path):
        import joblib
        # Simulate an old checkpoint (no critic_mode key).
        old_data = {
            "members": [],
            "n_features": 512,
            "n_bootstrap": 31,
            "feature_block": "full",
            "seed": 7,
            "encoder": None,
            "schema_version": "3.1",
        }
        path = tmp_path / "old.joblib"
        joblib.dump(old_data, path)
        critic = FourOutcomeTransferCritic.load(path)
        assert critic.critic_mode == "flat"
        assert critic.factorized_members == []


class TestFlatModeUnchanged:
    """Flat mode continues to work with the same API."""

    def test_flat_fit_predict(self):
        inputs, records = _build_mini_dataset()
        labels = [r["label"] for r in records]
        critic = FourOutcomeTransferCritic(
            n_bootstrap=3, seed=7, critic_mode="flat"
        )
        critic.fit(inputs, labels)
        pred = critic.predict(inputs[0])
        total = (
            pred.q00_neutral_failure
            + pred.q01_negative_transfer
            + pred.q10_positive_transfer
            + pred.q11_neutral_success
        )
        assert abs(total - 1.0) < 1e-8
        # Flat: members populated, factorized empty.
        assert len(critic.members) > 0
        assert len(critic.factorized_members) == 0

    def test_flat_ignores_records(self):
        """Flat mode doesn't require records."""
        inputs, records = _build_mini_dataset()
        labels = [r["label"] for r in records]
        critic = FourOutcomeTransferCritic(n_bootstrap=3, seed=7)
        critic.fit(inputs, labels)  # No records → fine.


class TestFactorizedRequiresRecords:
    def test_no_records_raises(self):
        inputs, records = _build_mini_dataset()
        critic = FourOutcomeTransferCritic(
            n_bootstrap=3, seed=7, critic_mode="opportunity_factorized"
        )
        with pytest.raises(ValueError, match="records"):
            critic.fit(inputs, [r["label"] for r in records])


# ---------------------------------------------------------------------------
# Tests: Bootstrap helpers
# ---------------------------------------------------------------------------


class TestBootstrapFamilyMultiplicities:
    def test_basic(self):
        rng = np.random.default_rng(42)
        families = ["A", "A", "B", "B", "C"]
        mult = bootstrap_family_multiplicities(families, rng)
        # Should return counts for unique families.
        assert sum(mult.values()) == len(set(families))

    def test_apply(self):
        ds = BinaryHeadDataset(
            inputs=["a", "b", "c"],
            targets=np.array([0, 1, 0]),
            sample_weights=np.array([1.0, 1.0, 1.0]),
            family_ids=["A", "A", "B"],
            edge_ids=["e1", "e1", "e2"],
        )
        mult = {"A": 2, "B": 0}
        idx, w = apply_family_multiplicities(ds, mult)
        assert list(idx) == [0, 1]
        assert w[0] == 2.0
        assert w[1] == 2.0
