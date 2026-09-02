"""RIMA Canonical Integrity Test Suite (Phase 31).

Target gate: ``RIMA_CANONICAL_INTEGRITY = PASS``.

Covers the 20 formal invariants (see docs/experiment_lineage/
rima_canonical_migration.md).
"""

from __future__ import annotations

import pytest

from smtr.memory.procedural_sanitizer import (
    PayloadLeakageError,
    assert_clean_payload,
    audit_payload_leakage,
    sanitize_candidate,
)
from smtr.memory.receiver_knowledge import ReceiverKnowledgeContainer
from smtr.memory.shared_memory_pool import SharedMemory, SharedMemoryPool
from smtr.rima.admission import (
    AdmissionStatus,
    ObservedDeltaAdmissionError,
    assert_formal_decision_source,
)
from smtr.rima.admission_engine import RimaAdmissionEngine
from smtr.rima.critic_validation import validate_critic
from smtr.rima.features import (
    ReceiverConditionedTransferFeatures,
    RimaFeatureEncoder,
)
from smtr.rima.outcome import RimaOutcomeEvaluator
from smtr.rima.receiver_topology import select_receivers
from smtr.rima.splits import SplitLeakageError, audit_split_leakage, task_level_split
from smtr.router.official_score_transfer_critic import (
    MatchedInterventionExample,
    OfficialScoreTransferCritic,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_memory(
    memory_id: str,
    source_agent_id: str,
    origin_task_position: int,
    *,
    tags: list[str] | None = None,
    goal: str = "negotiate price effectively",
) -> SharedMemory:
    return SharedMemory(
        memory_id=memory_id,
        source_agent_id=source_agent_id,
        origin_task_id=f"task{origin_task_position}",
        origin_task_position=origin_task_position,
        routing_card={
            "goal_summary": goal,
            "task_tags": tags or ["bargain"],
            "compatible_receiver_roles": ["buyer", "seller"],
            "compatible_receiver_capabilities": ["negotiate"],
        },
        procedure_payload=f"procedure from {memory_id}",
        scenario="bargaining",
    )


def make_features(memory, receiver_id: str, task: dict) -> ReceiverConditionedTransferFeatures:
    return ReceiverConditionedTransferFeatures(
        task_id=str(task.get("task_id", "?")),
        memory_id=memory.memory_id,
        receiver_id=receiver_id,
        task_repr={"scenario": "bargaining", "task_type": "negotiation", "text": task.get("text", "")},
        receiver_repr={"role": "buyer", "capabilities": ["negotiate"]},
        routing_card=dict(memory.routing_card),
    )


def train_critic(*, include_receiver: bool = True, receiver_bias: dict[str, float] | None = None):
    """Train a critic whose tau depends on receiver identity (when biased)."""
    encoder = RimaFeatureEncoder(n_features=256, include_receiver=include_receiver)
    critic = OfficialScoreTransferCritic(
        encoder=encoder, loss="huber", receiver_conditioned=include_receiver
    )
    bias = receiver_bias or {}
    examples = []
    for i in range(60):
        recv = ["agent1", "agent2", "agent3"][i % 3]
        src = ["agent1", "agent2", "agent3"][(i + 1) % 3]
        memory = make_memory(f"m{i % 6}", src, i % 4)
        base = 0.5
        tau = bias.get(recv, 0.05)
        examples.append(
            MatchedInterventionExample(
                task_id=f"t{i}",
                memory_id=memory.memory_id,
                receiver_id=recv,
                source_agent_id=src,
                official_expose_score=min(1.0, base + tau),
                official_withhold_score=base,
                features=make_features(
                    memory, recv, {"task_id": f"t{i}", "text": "negotiate price"}
                ),
            )
        )
    critic.fit(examples)
    return critic


def make_engine(*, receiver_bias: dict[str, float] | None = None, pool=None):
    critic = train_critic(receiver_bias=receiver_bias)
    critic.freeze()
    pool = pool or SharedMemoryPool()
    engine = RimaAdmissionEngine(
        critic=critic,
        pool=pool,
        feature_builder=make_features,
        retrieval_top_k=10,
    )
    return engine, critic, pool


TASK = {"task_id": "current", "text": "negotiate price bargain", "scenario": "bargaining"}


def _bargaining_eval(v: int) -> dict:
    side = {
        "effectiveness_of_strategies": v, "progress_and_outcome": v,
        "interaction_dynamics": v,
    }
    return {"buyer": dict(side), "seller": dict(side)}


# ---------------------------------------------------------------------------
# 1-2: decision source invariants
# ---------------------------------------------------------------------------


def test_invariant_01_admission_never_reads_observed_outcome():
    """Admission engine API has no observed-outcome parameter."""
    engine, _, pool = make_engine()
    pool.add(make_memory("m0", "agent2", 0))
    import inspect

    sig = inspect.signature(engine.admit_for_task)
    forbidden = {"observed_delta", "expose_score", "withhold_score", "outcome"}
    assert not forbidden & set(sig.parameters)


def test_invariant_02_formal_decision_source_only_frozen_critic():
    assert_formal_decision_source("frozen_transfer_critic")  # passes
    for bad in ("observed_delta", "team_success", "oracle", "ground_truth", "magic"):
        with pytest.raises(ObservedDeltaAdmissionError):
            assert_formal_decision_source(bad)


def test_invariant_02b_engine_decisions_carry_frozen_critic_source():
    engine, _, pool = make_engine()
    pool.add(make_memory("m0", "agent2", 0))
    knowledge = ReceiverKnowledgeContainer().ensure("agent1")
    engine.admit_for_task(
        task=TASK, task_id="current", task_position=5,
        receiver_id="agent1", knowledge=knowledge,
    )
    assert engine.decisions
    assert all(d.decision_source == "frozen_transfer_critic" for d in engine.decisions)


# ---------------------------------------------------------------------------
# 3-4: temporal lifecycle invariants
# ---------------------------------------------------------------------------


def test_invariant_03_current_task_memory_never_in_current_candidates():
    _, _, pool = make_engine()
    pool.add(make_memory("past", "agent2", 0))
    pool.add(make_memory("current_task_mem", "agent2", 5))  # created AT task 5
    cands = pool.retrieve(TASK, "agent1", 10, current_task_position=5)
    ids = [m.memory_id for m in cands]
    assert "past" in ids
    assert "current_task_mem" not in ids


def test_invariant_04_candidates_always_from_tasks_before_t():
    _, _, pool = make_engine()
    for pos in range(8):
        pool.add(make_memory(f"m{pos}", "agent2", pos))
    for t in range(9):
        cands = pool.retrieve(TASK, "agent1", 100, current_task_position=t)
        assert all(m.origin_task_position < t for m in cands)


# ---------------------------------------------------------------------------
# 5: self-transfer excluded
# ---------------------------------------------------------------------------


def test_invariant_05_self_transfer_excluded_and_counted():
    engine, _, pool = make_engine()
    pool.add(make_memory("own", "agent1", 0))   # source == receiver agent1
    pool.add(make_memory("other", "agent2", 0))
    knowledge = ReceiverKnowledgeContainer().ensure("agent1")
    admitted = engine.admit_for_task(
        task=TASK, task_id="current", task_position=5,
        receiver_id="agent1", knowledge=knowledge,
    )
    assert engine.stats.self_transfer_excluded_count == 1
    assert all(m.memory_id != "own" for m in admitted)
    statuses = {d.memory_id: d.status for d in engine.decisions}
    assert statuses["own"] == AdmissionStatus.SELF_TRANSFER_EXCLUDED


def test_invariant_05b_knowledge_refuses_self_transfer():
    k = ReceiverKnowledgeContainer().ensure("agent1")
    with pytest.raises(ValueError):
        k.admit(make_memory("own", "agent1", 0), 0.1, task_id="t", task_position=1)


# ---------------------------------------------------------------------------
# 6-7: receiver-specific behavior
# ---------------------------------------------------------------------------


def test_invariant_06_receiver_payloads_no_cross_leakage():
    container = ReceiverKnowledgeContainer(["agent1", "agent2"])
    container.get("agent1").admit(
        make_memory("m1", "agent3", 0), 0.1, task_id="t", task_position=1
    )
    payloads = container.payloads(context_budget=5)
    assert payloads["agent1"] == ["procedure from m1"]
    assert payloads["agent2"] == []


def test_invariant_07_same_memory_different_receiver_decisions():
    engine, _, pool = make_engine(
        receiver_bias={"agent1": 0.0, "agent2": 0.3, "agent3": -0.3}
    )
    pool.add(make_memory("shared", "agentX", 0))
    decisions: dict[str, str] = {}
    for recv in ("agent1", "agent2", "agent3"):
        k = ReceiverKnowledgeContainer().ensure(recv)
        engine.admit_for_task(
            task=TASK, task_id="current", task_position=5,
            receiver_id=recv, knowledge=k,
        )
        decisions[recv] = [d for d in engine.decisions if d.receiver_id == recv][-1].status
    assert decisions["agent2"] == AdmissionStatus.ADMITTED
    assert decisions["agent3"] == AdmissionStatus.REJECTED
    assert decisions["agent2"] != decisions["agent3"]


# ---------------------------------------------------------------------------
# 8-9: multi-memory admission rule
# ---------------------------------------------------------------------------


def test_invariant_08_all_positive_tau_admitted():
    engine, _, pool = make_engine()
    for i in range(4):
        pool.add(make_memory(f"pos{i}", "agent2", 0))
    k = ReceiverKnowledgeContainer().ensure("agent1")
    admitted = engine.admit_for_task(
        task=TASK, task_id="current", task_position=5,
        receiver_id="agent1", knowledge=k,
    )
    positive = [d for d in engine.decisions if d.tau_hat is not None and d.tau_hat > 0]
    assert len(admitted) == len(positive)
    assert len(admitted) >= 2  # multi-memory admission, not top-1


def test_invariant_09_tau_leq_zero_rejected():
    engine, _, pool = make_engine(
        receiver_bias={"agent1": -0.4, "agent2": -0.4, "agent3": -0.4}
    )
    pool.add(make_memory("neg", "agent2", 0))
    k = ReceiverKnowledgeContainer().ensure("agent1")
    admitted = engine.admit_for_task(
        task=TASK, task_id="current", task_position=5,
        receiver_id="agent1", knowledge=k,
    )
    assert admitted == []
    assert len(k) == 0


# ---------------------------------------------------------------------------
# 10: fail-closed invalid handling
# ---------------------------------------------------------------------------


def test_invariant_10_invalid_outcome_not_silently_zero():
    evaluator = RimaOutcomeEvaluator(scenario="bargaining")
    delta = evaluator.compute_delta(
        expose_result={}, withhold_result={"task_evaluation": _bargaining_eval(3)},
        receiver_id="agent1",
    )
    assert delta.oriented_delta is None  # None, not 0
    assert not delta.is_valid


def test_invariant_10b_invalid_training_examples_excluded_counted():
    encoder = RimaFeatureEncoder(n_features=128)
    critic = OfficialScoreTransferCritic(encoder=encoder, receiver_conditioned=True)
    memory = make_memory("m0", "agent2", 0)
    good = MatchedInterventionExample(
        "t0", "m0", "agent1", "agent2", 0.6, 0.5,
        make_features(memory, "agent1", {"task_id": "t0", "text": "negotiate"}),
    )
    bad = MatchedInterventionExample(
        "t1", "m0", "agent1", "agent2", None, 0.5,
        make_features(memory, "agent1", {"task_id": "t1", "text": "negotiate"}),
    )
    stats = critic.fit([good, bad])
    assert stats["invalid_excluded"] == 1
    assert stats["n_examples_used"] == 1


def test_invariant_10c_validation_report_counts_invalid_not_zero():
    report = validate_critic(
        [
            {"predicted_tau": 0.1, "observed_delta": None},
            {"predicted_tau": None, "observed_delta": 0.2},
        ]
    )
    assert report.n_pairs_valid == 0
    assert report.n_pairs_invalid == 2


# ---------------------------------------------------------------------------
# 11-12: M_t vs K_r_t distinctness and reset
# ---------------------------------------------------------------------------


def test_invariant_11_pool_and_knowledge_distinct():
    pool = SharedMemoryPool()
    container = ReceiverKnowledgeContainer(["agent1"])
    m = make_memory("m0", "agent2", 0)
    pool.add(m)
    assert len(pool) == 1 and len(container.get("agent1")) == 0
    container.get("agent1").admit(m, 0.1, task_id="t", task_position=1)
    assert len(pool) == 1 and len(container.get("agent1")) == 1
    assert pool.memories_before(5)[0].memory_id == "m0"


def test_invariant_12_scenario_reset():
    engine_a, _, pool_a = make_engine()
    pool_a.add(make_memory("m0", "agent2", 0))
    engine_b, _, pool_b = make_engine()  # fresh scenario state
    assert len(pool_a) == 1 and len(pool_b) == 0


# ---------------------------------------------------------------------------
# 13-14: frozen critic + shared candidate pool fairness
# ---------------------------------------------------------------------------


def test_invariant_13_critic_must_be_frozen_for_continual_eval():
    encoder = RimaFeatureEncoder(n_features=128)
    critic = OfficialScoreTransferCritic(encoder=encoder, receiver_conditioned=True)
    memory = make_memory("m0", "agent2", 0)
    critic.fit(
        [
            MatchedInterventionExample(
                f"t{i}", "m0", "agent1", "agent2", 0.6, 0.5,
                make_features(memory, "agent1", {"task_id": f"t{i}", "text": "negotiate"}),
            )
            for i in range(3)
        ]
    )
    engine = RimaAdmissionEngine(
        critic=critic, pool=SharedMemoryPool(), feature_builder=make_features
    )
    with pytest.raises(RuntimeError, match="FROZEN"):
        engine.require_frozen()
    critic.freeze()
    engine.require_frozen()  # ok
    with pytest.raises(RuntimeError):
        critic.fit([])


def test_invariant_14_same_candidate_pool_across_methods():
    _, _, pool = make_engine()
    for i in range(5):
        pool.add(make_memory(f"m{i}", "agent2", 0))
    hist = pool.memories_before(3)
    retr = pool.retrieve(TASK, "agent1", 10, current_task_position=3)
    assert {m.memory_id for m in hist} == {m.memory_id for m in retr}


# ---------------------------------------------------------------------------
# 15-17: outcome and leakage guards
# ---------------------------------------------------------------------------


def test_invariant_15_official_task_score_primary():
    evaluator = RimaOutcomeEvaluator(scenario="bargaining")
    outcome = evaluator.evaluate(task={}, run_result={"task_evaluation": _bargaining_eval(5)})
    assert outcome.is_valid and outcome.task_score == pytest.approx(1.0)
    assert evaluator.official_metric_name == "avg_negotiation_quality"


def test_invariant_16_team_success_cannot_affect_admission():
    evaluator = RimaOutcomeEvaluator(scenario="bargaining")
    run = {"task_evaluation": _bargaining_eval(3), "team_success": True}
    outcome = evaluator.evaluate(task={}, run_result=run)
    assert outcome.team_success is True  # diagnostic metadata only
    import inspect

    params = inspect.signature(RimaAdmissionEngine.admit_for_task).parameters
    assert "team_success" not in params


def test_invariant_17_ground_truth_features_absent():
    from smtr.rima.features import RimaFeatureEncoder as Enc

    with pytest.raises(ValueError, match="Forbidden"):
        Enc._reject_forbidden_tokens(["answer=42"])
    with pytest.raises(ValueError, match="Forbidden"):
        Enc._reject_forbidden_tokens(["ground_truth=abc"])
    with pytest.raises(ValueError, match="Forbidden"):
        Enc._reject_forbidden_tokens(["payload=secret procedure"])
    # normal routing-card tokens pass
    Enc._reject_forbidden_tokens(["scenario=bargaining", "card_tag=x", "receiver=a"])


# ---------------------------------------------------------------------------
# 18: sanitizer
# ---------------------------------------------------------------------------


def test_invariant_18_sanitizer_blocks_answer_leakage():
    raw = (
        "Strategy: anchor low. task_id: 42 "
        "Final answer: accept $90. score: 5 team_success: true"
    )
    san = sanitize_candidate(
        memory_id="m0", source_agent_id="agent2", raw_content=raw, task_id="42"
    )
    assert "final answer" not in san.procedural_content.lower()
    assert "task_id" not in san.procedural_content.lower()
    assert san.removed_fragments

    hits = audit_payload_leakage("the final answer: 42 is correct")
    assert hits
    with pytest.raises(PayloadLeakageError):
        assert_clean_payload("final answer: 42")
    assert_clean_payload("anchor low then concede slowly")  # clean passes


# ---------------------------------------------------------------------------
# 19-20: multi-receiver injection + reproducible order
# ---------------------------------------------------------------------------


def test_invariant_19_multi_receiver_simultaneous_payloads():
    container = ReceiverKnowledgeContainer(["agent1", "agent2", "agent3"])
    for i, rid in enumerate(["agent1", "agent2", "agent3"]):
        src = ["agent1", "agent2", "agent3"][(i + 1) % 3]
        container.get(rid).admit(
            make_memory(f"m{rid}", src, 0), 0.1, task_id="t", task_position=1
        )
    payloads = container.payloads(context_budget=5)
    assert set(payloads) == {"agent1", "agent2", "agent3"}
    for rid in payloads:
        assert all(f"m{rid}" in p for p in payloads[rid])


def test_invariant_20_reproducible_receiver_topology():
    task = {
        "agent_ids": ["agent1", "agent2", "agent3", "agent4"],
        "agents": [
            {"agent_id": f"agent{i}", "role": r}
            for i, r in zip(range(1, 5), ["buyer", "seller", "mediator", "analyst"])
        ],
    }
    first = select_receivers(task=task, task_id="t1", receiver_count=3)
    second = select_receivers(task=task, task_id="t1", receiver_count=3)
    assert [r.receiver_id for r in first] == [r.receiver_id for r in second]
    assert len(first) == 3
    with pytest.raises(ValueError):
        select_receivers(task={"agent_ids": []}, task_id="t")


# ---------------------------------------------------------------------------
# Bonus: task-level split audit
# ---------------------------------------------------------------------------


def test_task_level_split_no_leakage():
    memory = make_memory("m0", "agent2", 0)
    examples = [
        MatchedInterventionExample(
            f"t{i}", f"mem{i}", "agent1", "agent2", 0.6, 0.5,
            make_features(memory, "agent1", {"task_id": f"t{i}", "text": "x"}),
        )
        for i in range(30)
    ]
    splits = task_level_split(examples, seed=3)
    audit = audit_split_leakage(splits)
    assert audit["status"] == "PASS"
    bad = {"train": examples[:15], "test": examples[10:20]}  # overlap
    with pytest.raises(SplitLeakageError):
        audit_split_leakage(bad)
