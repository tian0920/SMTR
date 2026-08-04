"""Tests for Commit 8: main-table baselines, receiver-effect analysis and
risk-utility curve reporting (清单第十一、十二章)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from smtr.core.types import AgentProfile, MemoryRoutingCard, ReceiverState
from smtr.evaluation.receiver_effect_analysis import analyze_receiver_effect
from smtr.marble.paired_evaluation import MAIN_TABLE_METHODS, run_paired_decision_evaluation
from smtr.router.baselines import (
    AllShareRouter,
    GlobalTransferCriticRouter,
    RoleAwareTop1Router,
    SMTRNoPairInteractionRouter,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _card(
    memory_id: str,
    *,
    goal_summary: str = "goal",
    task_tags: tuple[str, ...] = (),
    writer_role: str = "executor",
    writer_caps: tuple[str, ...] = (),
    writer_tools: tuple[str, ...] = (),
    compatible: tuple[str, ...] = (),
    incompatible: tuple[str, ...] = (),
) -> MemoryRoutingCard:
    return MemoryRoutingCard(
        memory_id=memory_id,
        goal_summary=goal_summary,
        task_tags=task_tags,
        writer=AgentProfile(
            agent_id=f"w_{memory_id}",
            role=writer_role,
            capabilities=writer_caps,
            tool_names=writer_tools,
        ),
        source_task_id="t_src",
        source_scenario="database",
        compatible_receiver_roles=compatible,
        incompatible_receiver_roles=incompatible,
    )


def _receiver_state(
    *,
    task_id: str = "task_review",
    role: str = "executor",
    caps: tuple[str, ...] = (),
    tools: tuple[str, ...] = (),
    instruction: str = "review code changes",
) -> ReceiverState:
    return ReceiverState(
        task_id=task_id,
        scenario="database",
        task_instruction=instruction,
        receiver=AgentProfile(
            agent_id="r1", role=role, capabilities=caps, tool_names=tools
        ),
    )


# ---------------------------------------------------------------------------
# 11.x label-free baselines
# ---------------------------------------------------------------------------


class TestRoleAwareTop1:
    def test_prefers_compatible_receiver_features(self):
        card_db = _card(
            "mem_db",
            goal_summary="optimize database query",
            task_tags=("database",),
            writer_caps=("sql",),
            writer_tools=("psql",),
            incompatible=("critic",),
        )
        card_review = _card(
            "mem_review",
            goal_summary="review code changes",
            task_tags=("review",),
            writer_caps=("review",),
            writer_tools=("git",),
            compatible=("executor",),
        )
        rs = _receiver_state(
            role="executor", caps=("review",), tools=("git",),
            instruction="review code changes",
        )

        router = RoleAwareTop1Router()
        decisions = router.decide(rs, [card_db, card_review])
        shared = [d.memory_id for d in decisions if d.action == "share"]
        assert shared == ["mem_review"], "role/cap/tool compatible card must win"

        # Deterministic across repeated calls
        again = router.decide(rs, [card_review, card_db])
        assert [d.memory_id for d in again if d.action == "share"] == ["mem_review"]
        # Exactly one share (v1 single-memory constraint)
        assert sum(1 for d in decisions if d.action == "share") == 1


class TestAllShareV1:
    def test_selects_single_top1_memory(self):
        card_a = _card("mem_a", goal_summary="unrelated topic", task_tags=("other",))
        card_b = _card(
            "mem_b",
            goal_summary="review code changes",
            task_tags=("review",),
            compatible=("executor",),
        )
        rs = _receiver_state(role="executor", instruction="review code changes")

        decisions = AllShareRouter().decide(rs, [card_a, card_b])
        shared = [d for d in decisions if d.action == "share"]
        assert len(shared) == 1, "AllShare in v1 must share exactly one memory"
        assert shared[0].memory_id == "mem_b"
        assert shared[0].reason == "all_share_top1_relevance"
        withheld = [d for d in decisions if d.action == "withhold"]
        assert all(d.reason == "all_share_single_memory_limit" for d in withheld)

        # Same label-free choice as RoleAwareTop1
        top1 = RoleAwareTop1Router().decide(rs, [card_a, card_b])
        assert [d.memory_id for d in top1 if d.action == "share"] == ["mem_b"]


# ---------------------------------------------------------------------------
# 11.x critic-ablation routers verify feature blocks
# ---------------------------------------------------------------------------


class TestAblationRoutersFeatureBlock:
    def test_reject_mismatched_feature_block(self):
        critic = MagicMock()
        critic.feature_block = "full"
        with pytest.raises(AssertionError):
            GlobalTransferCriticRouter(critic=critic)
        with pytest.raises(AssertionError):
            SMTRNoPairInteractionRouter(critic=critic)

    def test_accept_matching_feature_block_and_decide(self):
        critic = MagicMock()
        critic.feature_block = "memory_task_only"
        critic.predict.return_value = SimpleNamespace(tau_hat=0.5, eta_hat=0.05)
        router = GlobalTransferCriticRouter(critic=critic, negative_risk_budget=0.2)
        decisions = router.decide(_receiver_state(), [_card("mem1")])
        assert [d.action for d in decisions] == ["share"]

        critic_np = MagicMock()
        critic_np.feature_block = "no_pair_interaction"
        critic_np.predict.return_value = SimpleNamespace(tau_hat=-0.1, eta_hat=0.05)
        router_np = SMTRNoPairInteractionRouter(critic=critic_np)
        decisions_np = router_np.decide(_receiver_state(), [_card("mem1")])
        assert [d.action for d in decisions_np] == ["withhold"]


# ---------------------------------------------------------------------------
# 12.x receiver-effect analysis
# ---------------------------------------------------------------------------


def _decision(memory_id, receiver, action, tau_hat, task_id="t1", seed=0):
    return {
        "candidate_memory_id": memory_id,
        "receiver_agent_id": receiver,
        "action": action,
        "tau_hat": tau_hat,
        "task_id": task_id,
        "generation_seed": seed,
    }


def _paired(memory_id, receiver, y_share, y_withhold, *, receiver_role="executor",
            ctx_task_tags=(), ctx_caps=()):
    return {
        "candidate_memory_id": memory_id,
        "receiver_agent_id": receiver,
        "receiver_role": receiver_role,
        "y_share": y_share,
        "y_withhold": y_withhold,
        "task_id": "t1",
        "common_seed": 0,
        "decision_context": {
            "task_tags": list(ctx_task_tags),
            "receiver_capabilities": list(ctx_caps),
        },
    }


class TestReceiverEffectFlipMetrics:
    def test_flip_and_identification_rates(self):
        decisions = [
            _decision("m1", "r1", "share", 0.6),
            _decision("m1", "r2", "withhold", -0.3),
            _decision("m2", "r1", "share", 0.4),  # m2 has a single receiver
        ]
        paired = [
            _paired("m1", "r1", 1, 0),   # positive transfer
            _paired("m1", "r2", 0, 1),   # negative transfer
            _paired("m2", "r1", 1, 0),
        ]
        report = analyze_receiver_effect(decisions=decisions, paired_records=paired)
        assert report["eligible_memory_count"] == 1  # only m1 has >= 2 receivers
        assert report["predicted_decision_flip_rate"] == 1.0
        assert report["empirical_effect_sign_flip_rate"] == 1.0
        assert report["correct_flip_identification_rate"] == 1.0


class TestReceiverRanking:
    def test_ranking_quality_metrics(self):
        decisions = [
            _decision("m1", "r1", "share", 0.9),
            _decision("m1", "r2", "share", 0.5),
            _decision("m1", "r3", "withhold", 0.1),
        ]
        paired = [
            _paired("m1", "r1", 1, 0),  # empirical tau = +1
            _paired("m1", "r2", 1, 1),  # empirical tau = 0
            _paired("m1", "r3", 0, 1),  # empirical tau = -1
        ]
        ranking = analyze_receiver_effect(
            decisions=decisions, paired_records=paired)["receiver_ranking"]
        assert ranking["memories_ranked"] == 1
        assert ranking["pairwise_receiver_ranking_accuracy"] == 1.0
        assert ranking["mean_spearman_correlation"] == 1.0
        assert ranking["top_receiver_accuracy"] == 1.0


class TestRiskHeterogeneity:
    def test_stratified_negative_transfer_rates(self):
        card = _card(
            "m1", writer_role="executor", writer_caps=("sql",),
            task_tags=("database",),
        )
        paired = [
            _paired("m1", "r1", 0, 1, receiver_role="critic",
                    ctx_task_tags=("database",)),                # negative, relevant
            _paired("m1", "r2", 1, 0, receiver_role="executor",
                    ctx_task_tags=("other",), ctx_caps=("sql",)),  # positive, irrelevant
        ]
        risk = analyze_receiver_effect(
            decisions=[], paired_records=paired, cards_by_id={"m1": card},
        )["risk_heterogeneity"]

        assert risk["by_receiver_role"]["critic"]["negative_transfer_rate"] == 1.0
        assert risk["by_receiver_role"]["executor"]["negative_transfer_rate"] == 0.0
        assert risk["by_writer_receiver_role_pair"]["executor->critic"]["n"] == 1
        # writer caps {sql}: r1 has no caps -> "none" bucket, r2 matches -> "high"
        assert risk["by_capability_overlap_bucket"]["none"]["negative_transfer_rate"] == 1.0
        assert risk["by_capability_overlap_bucket"]["high"]["negative_transfer_rate"] == 0.0
        by_rel = risk["negative_transfer_rate_by_task_relevance"]
        assert by_rel["relevant"]["negative_transfer_rate"] == 1.0
        assert by_rel["irrelevant"]["negative_transfer_rate"] == 0.0


# ---------------------------------------------------------------------------
# Main table method list
# ---------------------------------------------------------------------------


class TestMainTableMethods:
    def test_main_table_contains_required_methods(self):
        assert MAIN_TABLE_METHODS == [
            "b0_no_memory",
            "role_aware_top1",
            "all_share",
            "global_transfer_critic",
            "smtr_no_pair_interaction",
            "smtr_no_risk",
            "smtr",
        ]
        assert "factual_success" not in MAIN_TABLE_METHODS


# ---------------------------------------------------------------------------
# End-to-end: risk-utility curve + receiver-effect artifacts
# ---------------------------------------------------------------------------


def _pool_line(memory_id: str) -> str:
    return json.dumps({
        "memory_id": memory_id,
        "payload": {"procedure": "step"},
        "routing_card": {
            "writer": {"agent_id": "w1", "role": "executor", "capabilities": []},
            "goal_summary": "goal",
            "task_tags": [],
            "environment_constraints": [],
            "positive_transfer_hints": [],
            "negative_transfer_hints": [],
            "source_task_id": "t_src",
            "source_scenario": "database",
            "compatible_receiver_roles": [],
            "incompatible_receiver_roles": [],
            "evidence_count": 1,
        },
    })


class TestPairedEvaluationArtifacts:
    def test_writes_risk_utility_curve_and_receiver_effect(self, tmp_path: Path):
        memory_pool = tmp_path / "pool.jsonl"
        memory_pool.write_text(
            _pool_line("mem1") + "\n" + _pool_line("mem2") + "\n", encoding="utf-8")

        candidates = tmp_path / "candidates.json"
        candidates.write_text(json.dumps({
            "candidates": [{
                "task_id": "t1",
                "receiver_agent_id": "r1",
                "receiver_role": "executor",
                "receiver_capabilities": [],
                "task_instruction": "do stuff",
                "environment_signature": [],
                "candidate_records": [
                    {"memory_id": "mem1", "rank": 1, "score": 0.9},
                    {"memory_id": "mem2", "rank": 2, "score": 0.4},
                ],
            }],
        }), encoding="utf-8")

        # mem1: positive transfer; mem2: neutral failure. Consistent Y_0=False.
        paired_records = tmp_path / "paired.jsonl"
        lines = [
            json.dumps({
                "task_id": "t1", "generation_seed": 0,
                "receiver_agent_id": "r1", "candidate_memory_id": "mem1",
                "valid": True, "label": "positive_transfer",
                "y_share": 1, "y_withhold": 0,
                "share": {"team_success": True},
                "withhold": {"team_success": False},
            }),
            json.dumps({
                "task_id": "t1", "generation_seed": 0,
                "receiver_agent_id": "r1", "candidate_memory_id": "mem2",
                "valid": True, "label": "neutral_failure",
                "y_share": 0, "y_withhold": 0,
                "share": {"team_success": False},
                "withhold": {"team_success": False},
            }),
        ]
        paired_records.write_text("\n".join(lines) + "\n", encoding="utf-8")

        mock_critic = MagicMock()
        mock_critic.feature_block = "full"
        mock_critic.epsilon_star = 0.1
        mock_critic.q01_calibrator = None

        def _predict(exposure_input):
            if exposure_input.candidate_card.memory_id == "mem1":
                return SimpleNamespace(tau_hat=0.5, eta_hat=0.05)
            return SimpleNamespace(tau_hat=-0.2, eta_hat=0.4)

        mock_critic.predict.side_effect = _predict

        output = tmp_path / "eval_out"
        with patch("smtr.marble.paired_evaluation.FourOutcomeTransferCritic") as MockCritic:
            MockCritic.load.return_value = mock_critic
            result = run_paired_decision_evaluation(
                candidate_manifest_path=candidates,
                paired_records_path=paired_records,
                memory_pool_path=memory_pool,
                checkpoint_full=tmp_path / "full.joblib",
                methods=["b0_no_memory", "role_aware_top1", "all_share", "smtr"],
                output=output,
            )

        # SMTR shares mem1 (safe) and withholds mem2
        traces = json.loads((output / "traces.json").read_text(encoding="utf-8"))
        smtr_actions = {
            t["candidate_memory_id"]: t["action"] for t in traces["smtr"]
        }
        assert smtr_actions == {"mem1": "share", "mem2": "withhold"}

        curve = json.loads((output / "risk_utility_curve.json").read_text(encoding="utf-8"))
        assert curve["n_matched_candidates"] == 2
        assert curve["epsilon_star"] == 0.1
        assert curve["epsilon_selected_on"] == "validation"
        assert "0.05" in curve["curve"]

        receiver_effect = json.loads(
            (output / "receiver_effect_analysis.json").read_text(encoding="utf-8"))
        assert receiver_effect["eligible_memory_count"] == 0  # single receiver only
        assert "risk_heterogeneity" in receiver_effect
        assert result["receiver_effect_analysis"] == receiver_effect

        # Ablation critic methods fail fast without their checkpoint
        with patch("smtr.marble.paired_evaluation.FourOutcomeTransferCritic") as MockCritic2:
            MockCritic2.load.return_value = mock_critic
            with pytest.raises(ValueError, match="global_transfer_critic"):
                run_paired_decision_evaluation(
                    candidate_manifest_path=candidates,
                    paired_records_path=paired_records,
                    memory_pool_path=memory_pool,
                    checkpoint_full=tmp_path / "full.joblib",
                    methods=["global_transfer_critic"],
                    output=tmp_path / "eval_out2",
                )
