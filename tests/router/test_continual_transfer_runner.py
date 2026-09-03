"""Tests for the transfer continual runner (RIMA-v2 §27-37).

Covers:
* Transfer policy loading from JSON
* context_budget=1 enforcement for rima_transfer
* Fail-closed: unfrozen critic raises RuntimeError
* Runner construction invariants
* Routing diagnostic builder
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from smtr.rima.transfer_policy import TransferPolicy

# We import the runner module components.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.rima.run_continual_transfer import (
    TransferContinualProtocol,
    load_transfer_policy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_policy_dict(
    *, beta: float = 1.64, delta: float = 0.0, gamma: float = 0.28
) -> dict:
    return {
        "schema_version": "rima_transfer_policy_v1",
        "beta": beta,
        "delta": delta,
        "gamma": gamma,
        "gamma_quantile": 0.75,
        "gamma_positive_support": 42,
        "gamma_source_split": "train",
        "critic_checkpoint_sha256": "abc123",
    }


def _write_policy_file(d: dict, tmpdir: str) -> str:
    path = Path(tmpdir) / "transfer_policy.json"
    with open(path, "w") as f:
        json.dump(d, f)
    return str(path)


def _make_frozen_critic():
    critic = MagicMock()
    critic.is_frozen = True
    critic.checkpoint_sha256.return_value = "abc123"
    return critic


def _make_unfrozen_critic():
    critic = MagicMock()
    critic.is_frozen = False
    return critic


def _make_policy():
    return TransferPolicy(
        beta=1.64,
        delta=0.0,
        gamma=0.28,
        gamma_quantile=0.75,
        gamma_positive_support=42,
        gamma_source_split="train",
        critic_checkpoint_sha256="abc123",
    )


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


class TestLoadTransferPolicy:
    def test_load_basic_policy(self, tmp_path):
        d = _make_policy_dict()
        path = _write_policy_file(d, str(tmp_path))
        policy = load_transfer_policy(path)
        assert isinstance(policy, TransferPolicy)
        assert policy.beta == pytest.approx(1.64)
        assert policy.delta == pytest.approx(0.0)
        assert policy.gamma == pytest.approx(0.28)
        assert policy.gamma_quantile == pytest.approx(0.75)
        assert policy.gamma_positive_support == 42
        assert policy.gamma_source_split == "train"
        assert policy.critic_checkpoint_sha256 == "abc123"

    def test_load_policy_with_custom_values(self, tmp_path):
        d = _make_policy_dict(beta=2.0, delta=0.1, gamma=0.5)
        path = _write_policy_file(d, str(tmp_path))
        policy = load_transfer_policy(path)
        assert policy.beta == pytest.approx(2.0)
        assert policy.delta == pytest.approx(0.1)
        assert policy.gamma == pytest.approx(0.5)

    def test_policy_is_frozen_dataclass(self, tmp_path):
        d = _make_policy_dict()
        path = _write_policy_file(d, str(tmp_path))
        policy = load_transfer_policy(path)
        with pytest.raises(AttributeError):
            policy.beta = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protocol construction invariants
# ---------------------------------------------------------------------------


class TestProtocolInvariants:
    def test_rima_transfer_context_budget_must_be_one(self):
        """rima_transfer with context_budget != 1 must raise ValueError."""
        critic = _make_frozen_critic()
        policy = _make_policy()
        with pytest.raises(ValueError, match="single-memory transfer requires context_budget=1"):
            TransferContinualProtocol(
                scenario="bargaining",
                seed=0,
                method="rima_transfer",
                tasks=[],
                collector=MagicMock(),
                extractor=MagicMock(),
                context_budget=5,
                critic_receiver=critic,
                transfer_policy=policy,
            )

    def test_rima_transfer_requires_critic(self):
        """rima_transfer without critic must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="critic"):
            TransferContinualProtocol(
                scenario="bargaining",
                seed=0,
                method="rima_transfer",
                tasks=[],
                collector=MagicMock(),
                extractor=MagicMock(),
                critic_receiver=None,
                transfer_policy=_make_policy(),
            )

    def test_rima_transfer_requires_policy(self):
        """rima_transfer without transfer policy must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="transfer.policy"):
            TransferContinualProtocol(
                scenario="bargaining",
                seed=0,
                method="rima_transfer",
                tasks=[],
                collector=MagicMock(),
                extractor=MagicMock(),
                critic_receiver=_make_frozen_critic(),
                transfer_policy=None,
            )

    def test_unfrozen_critic_raises_runtime_error(self):
        """§29: fail-closed if critic is not frozen."""
        critic = _make_unfrozen_critic()
        with pytest.raises(RuntimeError, match="must be frozen"):
            TransferContinualProtocol(
                scenario="bargaining",
                seed=0,
                method="rima_transfer",
                tasks=[],
                collector=MagicMock(),
                extractor=MagicMock(),
                critic_receiver=critic,
                transfer_policy=_make_policy(),
            )

    def test_no_memory_constructs_without_critic(self):
        """no_memory method should not require a critic."""
        protocol = TransferContinualProtocol(
            scenario="bargaining",
            seed=0,
            method="no_memory",
            tasks=[],
            collector=MagicMock(),
            extractor=MagicMock(),
        )
        assert protocol.controller is None
        assert protocol.engine is None

    def test_retrieval_constructs_without_critic(self):
        """retrieval method should not require a critic."""
        protocol = TransferContinualProtocol(
            scenario="bargaining",
            seed=0,
            method="retrieval",
            tasks=[],
            collector=MagicMock(),
            extractor=MagicMock(),
        )
        assert protocol.controller is None
        assert protocol.engine is None

    def test_rima_transfer_constructs_with_valid_inputs(self):
        """Valid rima_transfer construction should succeed."""
        critic = _make_frozen_critic()
        policy = _make_policy()
        protocol = TransferContinualProtocol(
            scenario="bargaining",
            seed=0,
            method="rima_transfer",
            tasks=[],
            collector=MagicMock(),
            extractor=MagicMock(),
            critic_receiver=critic,
            transfer_policy=policy,
        )
        assert protocol.controller is not None
        assert protocol.engine is None

    def test_rima_transfer_controller_has_budget_one(self):
        """Controller context_budget must be 1 regardless of input."""
        critic = _make_frozen_critic()
        policy = _make_policy()
        protocol = TransferContinualProtocol(
            scenario="bargaining",
            seed=0,
            method="rima_transfer",
            tasks=[],
            collector=MagicMock(),
            extractor=MagicMock(),
            critic_receiver=critic,
            transfer_policy=policy,
            context_budget=1,
        )
        assert protocol.controller.context_budget == 1


# ---------------------------------------------------------------------------
# Routing diagnostic builder
# ---------------------------------------------------------------------------


class TestRoutingDiagnosticBuilder:
    def test_diagnostic_from_plan(self):
        """Verify diagnostic dict structure from a mock plan."""
        critic = _make_frozen_critic()
        policy = _make_policy()
        protocol = TransferContinualProtocol(
            scenario="bargaining",
            seed=0,
            method="rima_transfer",
            tasks=[],
            collector=MagicMock(),
            extractor=MagicMock(),
            critic_receiver=critic,
            transfer_policy=policy,
        )

        # Build a mock plan
        mock_known = MagicMock()
        mock_known.memory_id = "m1"
        mock_known.mu_tau = 0.42
        mock_known.sigma_tau = 0.06
        mock_known.lcb = 0.32
        mock_known.selected_for_context = True
        mock_known.candidate_source = "known"

        mock_plan = MagicMock()
        mock_plan.routing_mode = "exploit_only"
        mock_plan.best_known_lcb = 0.32
        mock_plan.known_candidates = [mock_known]
        mock_plan.global_candidates = []
        mock_plan.selected_memory_ids = ["m1"]
        mock_plan.global_retrieval_triggered = False

        diag = protocol._build_routing_diagnostic(mock_plan, "r1", 5)
        assert diag["routing_mode"] == "exploit_only"
        assert diag["receiver_id"] == "r1"
        assert diag["task_position"] == 5
        assert diag["best_known_lcb"] == 0.32
        assert diag["best_known_mu"] == 0.42
        assert diag["best_known_sigma"] == 0.06
        assert diag["selected_memory_id"] == "m1"
        assert diag["selected_source"] == "known"
        assert diag["selected_mu"] == 0.42
        assert diag["selected_lcb"] == 0.32
        assert diag["global_retrieval_triggered"] is False
        assert diag["beta"] == 1.64
        assert diag["delta"] == 0.0
        assert diag["gamma"] == 0.28

    def test_diagnostic_no_selection(self):
        """When no memory is selected, diagnostic reflects that."""
        critic = _make_frozen_critic()
        policy = _make_policy()
        protocol = TransferContinualProtocol(
            scenario="bargaining",
            seed=0,
            method="rima_transfer",
            tasks=[],
            collector=MagicMock(),
            extractor=MagicMock(),
            critic_receiver=critic,
            transfer_policy=policy,
        )

        mock_plan = MagicMock()
        mock_plan.routing_mode = "explore_only"
        mock_plan.best_known_lcb = None
        mock_plan.known_candidates = []
        mock_plan.global_candidates = []
        mock_plan.selected_memory_ids = []
        mock_plan.global_retrieval_triggered = True

        diag = protocol._build_routing_diagnostic(mock_plan, "r2", 0)
        assert diag["selected_memory_id"] is None
        assert diag["selected_source"] == "none"
        assert diag["selected_mu"] is None
        assert diag["global_retrieval_triggered"] is True
        assert diag["n_known_candidates_considered"] == 0
