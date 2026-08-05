"""清单 Test 9: receiver-effect metrics (P0-12~14).

Construction: the same memory is positive transfer for receiver A and
negative transfer for receiver B on the same target task. The analysis
must identify the transfer-sign flip, the harm-risk flip and the
receiver-specific SMTR decision flip.
"""

from __future__ import annotations

from smtr.evaluation.receiver_effect_analysis import (
    analyze_receiver_effect_anchor_groups,
    build_receiver_effect_anchor_groups,
    compare_receiver_effect_methods,
    empirical_receiver_effects,
)

EPSILON_STAR = 0.10


def _record(
    receiver: str,
    *,
    label: str,
    seed: int,
    memory: str = "m1",
    task: str = "t1",
) -> dict:
    share = label in ("positive_transfer", "neutral_success")
    withhold = label in ("negative_transfer", "neutral_success")
    return {
        "task_id": task,
        "receiver_agent_id": receiver,
        "candidate_memory_id": memory,
        "generation_seed": seed,
        "valid": True,
        "label": label,
        "share": {"team_success": share},
        "withhold": {"team_success": withhold},
    }


def _paired_records() -> list[dict]:
    """m1: positive for rA, negative for rB; m2 only seen by rA."""
    records = [_record("rA", label="positive_transfer", seed=s) for s in range(2)]
    records += [_record("rB", label="negative_transfer", seed=s) for s in range(2)]
    records += [_record("rA", label="neutral_success", seed=s, memory="m2") for s in range(2)]
    return records


def _decisions(receiver_tau: dict[str, tuple[float, float]]) -> list[dict]:
    """One decision trace per (receiver, memory) cell, repeated per seed."""
    traces = []
    for receiver, (tau, eta) in receiver_tau.items():
        action = "share" if tau > 0 and eta <= EPSILON_STAR else "withhold"
        for seed in range(2):
            traces.append({
                "task_id": "t1",
                "generation_seed": seed,
                "candidate_memory_id": "m1",
                "receiver_agent_id": receiver,
                "action": action,
                "tau_hat": tau,
                "eta_hat": eta,
            })
    return traces


class TestAnchorGroupConstruction:
    def test_anchor_group_requires_two_receivers(self):
        groups = build_receiver_effect_anchor_groups(_paired_records())
        assert set(groups) == {("t1", "m1")}, "m2 has one receiver, not an anchor"
        assert set(groups[("t1", "m1")]) == {"rA", "rB"}

    def test_min_seed_requirement_filters_receivers(self):
        groups = build_receiver_effect_anchor_groups(
            _paired_records(), min_seeds_per_receiver=3
        )
        assert groups == {}, "no receiver has 3 seeds, so no anchor survives"


class TestReceiverEffectFlips:
    def test_transfer_sign_and_harm_risk_flips_identified(self):
        groups = build_receiver_effect_anchor_groups(_paired_records())
        effects = empirical_receiver_effects(_paired_records())
        # rA: tau=+1, eta=0 ; rB: tau=-1, eta=1
        assert effects[("t1", "m1", "rA")]["tau_hat"] == 1.0
        assert effects[("t1", "m1", "rB")]["tau_hat"] == -1.0
        report = analyze_receiver_effect_anchor_groups(
            groups, effects, epsilon_star=EPSILON_STAR
        )
        assert report["transfer_sign_flip_count"] == 1
        assert report["harm_risk_flip_count"] == 1
        group = report["groups"]["t1|m1"]
        assert group["transfer_sign_flip"] is True
        assert group["harm_risk_flip"] is True
        assert group["delta_tau_range"] == 2.0

    def test_smtr_decision_flip_across_receivers(self):
        """Same memory/task, different receivers -> different SMTR action."""
        decisions = _decisions({"rA": (0.8, 0.02), "rB": (-0.6, 0.9)})
        groups = build_receiver_effect_anchor_groups(_paired_records())
        effects = empirical_receiver_effects(_paired_records())
        report = analyze_receiver_effect_anchor_groups(
            groups, effects, epsilon_star=EPSILON_STAR, decisions=decisions
        )
        assert report["decision_flip_count"] == 1
        assert report["groups"]["t1|m1"]["decision_flip"] is True


class TestReceiverEffectComparisonTable:
    def test_comparison_reports_required_metrics(self):
        # SMTR is receiver-specific; the global critic gives one answer for
        # both receivers (shares for both), missing the negative receiver.
        smtr = _decisions({"rA": (0.8, 0.02), "rB": (-0.6, 0.9)})
        global_critic = _decisions({"rA": (0.5, 0.05), "rB": (0.5, 0.05)})
        table = compare_receiver_effect_methods(
            decisions_by_method={
                "smtr": smtr,
                "global_transfer_critic": global_critic,
            },
            paired_records=_paired_records(),
            epsilon_star=EPSILON_STAR,
        )["methods"]
        for method in ("smtr", "global_transfer_critic"):
            for key in (
                "receiver_effect_sign_accuracy",
                "receiver_specific_decision_accuracy",
                "harmful_exposure_rejection_rate",
                "harmful_exposure_rejection_by_receiver",
                "same_memory_decision_flip_precision",
                "same_memory_decision_flip_recall",
            ):
                assert key in table[method], f"{method} missing {key}"
        # SMTR identifies the flip and rejects the harmful exposure.
        assert table["smtr"]["same_memory_decision_flip_recall"] == 1.0
        assert table["smtr"]["same_memory_decision_flip_precision"] == 1.0
        assert table["smtr"]["harmful_exposure_rejection_rate"] == 1.0
        assert table["smtr"]["receiver_effect_sign_accuracy"] == 1.0
        # The receiver-agnostic method never flips and exposes rB.
        assert table["global_transfer_critic"]["same_memory_decision_flip_recall"] == 0.0
        assert table["global_transfer_critic"]["harmful_exposure_rejection_rate"] == 0.0
