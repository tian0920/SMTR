"""Same-memory receiver-effect tests (清单 Writer-Agnostic 18.14).

The same task + memory pair can transfer positively to one receiver and
negatively to another. The anchor-group machinery must build the group,
detect the empirical effect-sign flip, and compute the router decision
flip for identical task + memory across receivers.
"""

from __future__ import annotations

from smtr.evaluation.receiver_effect_analysis import (
    analyze_receiver_effect_anchor_groups,
    build_receiver_effect_anchor_groups,
    empirical_receiver_effects,
)


def _paired_record(receiver: str, *, y_share: int, y_withhold: int, seed: int = 0) -> dict:
    return {
        "task_id": "t1",
        "receiver_agent_id": receiver,
        "candidate_memory_id": "mem1",
        "generation_seed": seed,
        "share": {"team_success": bool(y_share)},
        "withhold": {"team_success": bool(y_withhold)},
    }


def _records() -> list[dict]:
    """Receiver A: positive transfer; receiver B: negative transfer."""
    return [
        _paired_record("recvA", y_share=1, y_withhold=0),
        _paired_record("recvB", y_share=0, y_withhold=1),
    ]


def _decisions() -> list[dict]:
    return [
        {
            "task_id": "t1",
            "candidate_memory_id": "mem1",
            "receiver_agent_id": "recvA",
            "action": "share",
            "tau_hat": 0.6,
            "eta_hat": 0.05,
            "eta_calibrated": 0.05,
        },
        {
            "task_id": "t1",
            "candidate_memory_id": "mem1",
            "receiver_agent_id": "recvB",
            "action": "withhold",
            "tau_hat": -0.4,
            "eta_hat": 0.5,
            "eta_calibrated": 0.5,
        },
    ]


class TestSameMemoryReceiverEffects:
    def test_anchor_group_constructed(self):
        anchors = build_receiver_effect_anchor_groups(_records())
        assert ("t1", "mem1") in anchors
        group = anchors[("t1", "mem1")]
        assert sorted(group) == ["recvA", "recvB"]
        assert all(len(recs) >= 1 for recs in group.values())

    def test_single_receiver_group_excluded(self):
        anchors = build_receiver_effect_anchor_groups(
            [_paired_record("recvA", y_share=1, y_withhold=0)])
        assert anchors == {}

    def test_empirical_effect_sign_flip(self):
        anchors = build_receiver_effect_anchor_groups(_records())
        effects = empirical_receiver_effects(_records())
        assert effects[("t1", "mem1", "recvA")]["tau_hat"] == 1.0
        assert effects[("t1", "mem1", "recvB")]["tau_hat"] == -1.0
        report = analyze_receiver_effect_anchor_groups(
            anchors, effects, epsilon_star=0.2, decisions=_decisions())
        assert report["anchor_group_count"] == 1
        assert report["transfer_sign_flip_count"] == 1
        assert report["transfer_sign_flip_rate"] == 1.0
        group = report["groups"]["t1|mem1"]
        assert group["transfer_sign_flip"] is True

    def test_router_decision_flip_computable(self):
        anchors = build_receiver_effect_anchor_groups(_records())
        effects = empirical_receiver_effects(_records())
        report = analyze_receiver_effect_anchor_groups(
            anchors, effects, epsilon_star=0.2, decisions=_decisions())
        assert report["decision_flip_count"] == 1
        group = report["groups"]["t1|mem1"]
        assert group["decision_flip"] is True
        assert group["receivers"] == ["recvA", "recvB"]
        # Harm-risk also flips across receivers (eta 0.05 <= 0.2 < 0.5).
        assert group["harm_risk_flip"] is True

    def test_no_decisions_keeps_flip_metrics_defined(self):
        anchors = build_receiver_effect_anchor_groups(_records())
        effects = empirical_receiver_effects(_records())
        report = analyze_receiver_effect_anchor_groups(
            anchors, effects, epsilon_star=0.2)
        assert report["decision_flip_count"] == 0
        assert report["groups"]["t1|mem1"]["transfer_sign_flip"] is True
