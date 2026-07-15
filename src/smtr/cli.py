import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from smtr.counterfactual.candidate_traversal import build_candidate_traversal_plan
from smtr.counterfactual.decision_points import InMemoryDecisionPointRecorder, canonical_digest
from smtr.counterfactual.interaction_boundary_sampler import (
    InteractionBoundaryConfig,
    InteractionBoundaryPrefixSampler,
)
from smtr.counterfactual.paired_rollout import PairedRolloutCollector
from smtr.counterfactual.policy_round import PolicyRoundLedger
from smtr.counterfactual.prefix_sampler import (
    PrefixSamplingConfig,
    ScenarioDesignatedTargetPolicy,
    StratifiedEligiblePrefixSampler,
    UniformCandidateTargetPolicy,
)
from smtr.counterfactual.record_writer import PairedRecordWriter
from smtr.counterfactual.schemas import DecisionPoint, routing_feature_snapshot_from_card
from smtr.counterfactual.task_provider import CounterfactualToyTaskProvider
from smtr.evaluation.compositional_splits import evaluate_compositional_splits
from smtr.evaluation.experiment_integrity import audit_experiment_integrity
from smtr.evaluation.feature_ablation import audit_feature_blocks
from smtr.evaluation.gate_diagnostics import compute_gate_diagnostics
from smtr.evaluation.interaction_audit import audit_interaction
from smtr.evaluation.logging import summarize_run
from smtr.evaluation.order_sensitivity import run_order_sensitivity
from smtr.evaluation.paired_statistics import compute_group_bootstrap_ci
from smtr.evaluation.shortcut_diagnostics import shortcut_diagnostics
from smtr.evaluation.static_set_diagnostics import compute_static_set_diagnostics
from smtr.experiment.methods import build_default_specs
from smtr.experiment.runner import ComparisonRunner
from smtr.experiment.schemas import ExperimentConfig
from smtr.experiment.summary import compute_summary
from smtr.experiment.writer import ExperimentWriter
from smtr.marble.dataset import (
    DEFAULT_MARBLE_ROOT,
    MARBLE_BENCHMARK_FILES,
    build_marble_dataset_manifest,
    write_marble_dataset_manifest,
)
from smtr.memory.execution_evidence import build_context_fingerprint
from smtr.memory.paired_transfer_evidence import PairedTransferEvidenceIngestor
from smtr.memory.seed_memories import seed_repository
from smtr.memory.store import SQLiteSharedMemoryRepository
from smtr.policy.critic_sequential_policy import FrozenCriticSequentialContinuationPolicy
from smtr.policy.exploratory_policy import (
    ExplorationPolicyConfig,
    FrozenRiskConstrainedExplorationPolicy,
)
from smtr.policy.fingerprints import file_sha256
from smtr.policy.manifests import (
    ContinuationPolicyManifest,
    load_policy_manifest,
    save_policy_manifest,
    with_fingerprint,
)
from smtr.policy.no_share_policy import (
    FrozenNoShareContinuationPolicy,
    create_no_share_manifest,
)
from smtr.router.factory import build_router
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_evaluation import (
    distribution,
    evaluate_records,
    group_split,
    write_json,
)
from smtr.router.transfer_features import (
    TransferPredictionInput,
    load_paired_records_for_training,
)
from smtr.runtime.environment import ToyEnvironment
from smtr.runtime.graph import run_demo, run_demo_with_repository, run_episode


def _seed_memories(db: str) -> None:
    repository = SQLiteSharedMemoryRepository(db)
    inserted = seed_repository(repository)
    cards = repository.get_routing_cards()
    print(f"memory_count={len(cards)}")
    print(f"store_revision={repository.current_revision()}")
    print("memory_ids=" + ",".join(card.memory_id for card in cards))
    if inserted:
        print("inserted=" + ",".join(inserted))
    else:
        print("inserted=")


def _list_memories(db: str) -> None:
    repository = SQLiteSharedMemoryRepository(db)
    for card in repository.get_routing_cards():
        constraints = {
            "required": card.required_environment_facts,
            "forbidden": card.forbidden_environment_facts,
        }
        print(
            " | ".join(
                [
                    f"memory_id={card.memory_id}",
                    f"active_payload_version={card.active_payload_version}",
                    f"goal_summary={card.goal_summary}",
                    f"receiver_roles={card.compatible_receiver_roles}",
                    f"environment_constraints={constraints}",
                    f"execution_alpha={card.execution_success_alpha}",
                    f"execution_beta={card.execution_success_beta}",
                    f"execution_success_count={card.execution_success_count}",
                    f"execution_failure_count={card.execution_failure_count}",
                ]
            )
        )


def _inspect_marble_dataset(args: argparse.Namespace) -> None:
    scenarios = set(args.scenarios) if args.scenarios else None
    if args.output:
        manifest = write_marble_dataset_manifest(
            output_path=Path(args.output),
            marble_root=Path(args.marble_root),
            scenarios=scenarios,
            limit_per_scenario=args.limit_per_scenario,
        )
    else:
        manifest = build_marble_dataset_manifest(
            marble_root=Path(args.marble_root),
            scenarios=scenarios,
            limit_per_scenario=args.limit_per_scenario,
        )
    print(f"marble_root={manifest.marble_root}")
    print(f"total_tasks={manifest.total_tasks}")
    for scenario, count in manifest.scenario_counts.items():
        print(f"scenario.{scenario}={count}")
    if args.output:
        print(f"manifest={args.output}")


def _demo(
    seed: int,
    db: str | None,
    top_k: int,
    router_mode: str = "no-memory",
    critic_checkpoint: str | None = None,
    max_shares_per_invocation: int | None = None,
) -> None:
    router = build_router(
        mode=router_mode,
        critic_checkpoint=critic_checkpoint,
        max_shares_per_invocation=max_shares_per_invocation,
    )
    initial_observation = ToyEnvironment(seed=seed).observe()
    if db:
        repository = SQLiteSharedMemoryRepository(db)
        seed_repository(repository)
        state = run_demo_with_repository(
            repository=repository, seed=seed, top_k=top_k, router=router
        )
    else:
        state = run_demo(seed=seed, router=router)

    print(f"Task: {state['task']}")
    print(f"Environment observation: {initial_observation}")
    for trace in state["router_trace"]:
        print(f"\nFor {trace['agent']}:")
        print("  candidate memories and scores")
        for candidate in trace["candidates"]:
            print(f"    {candidate['memory_id']}: {candidate['total_score']}")
        print("  router decisions")
        for decision in trace["decisions"]:
            print(f"    {decision['memory_id']}: {decision['action']} ({decision['reason']})")
        print(f"  selected IDs: {trace['selected_memory_ids']}")
    print()
    print(summarize_run(state))


def _make_boundary_critic_scorer(*, critic, decision_point, cards_by_id):
    """Build an A-07.4/5/6 prefix scorer that probes the current critic.

    Scores each (target, prefix) by prediction disagreement between the empty
    prefix and the candidate prefix (A-07.4), ensemble uncertainty (A-07.5), and
    closeness of tau_hat to zero (A-07.6). Mechanism-agnostic: it never needs the
    hidden payload ``strategy``.
    """
    context = build_context_fingerprint(
        task_id=decision_point.task_id,
        task_tags=decision_point.candidate_proposal.request.task.split(),
        receiver_agent_id=decision_point.receiver_agent_id,
        receiver_role=decision_point.receiver_role,
        receiver_capabilities=decision_point.candidate_proposal.request.receiver_capabilities,
        environment_observation=decision_point.environment_snapshot,
        task_stage=decision_point.task_stage,
        selected_memory_ids=[],
        episode_id=decision_point.episode_id,
    )
    snapshots = {
        memory_id: routing_feature_snapshot_from_card(card)
        for memory_id, card in cards_by_id.items()
    }

    def score(target_id: str, prefix_id: str) -> float:
        if target_id not in snapshots or prefix_id not in snapshots:
            return 0.0
        target_card = snapshots[target_id]
        prefix_card = snapshots[prefix_id]
        empty = critic.predict(
            TransferPredictionInput(
                context=context, candidate_card=target_card, selected_cards=[]
            )
        )
        with_prefix = critic.predict(
            TransferPredictionInput(
                context=context, candidate_card=target_card, selected_cards=[prefix_card]
            )
        )
        disagreement = abs(with_prefix.tau_mean - empty.tau_mean)
        uncertainty = max(0.0, with_prefix.tau_ucb - with_prefix.tau_lcb)
        near_zero = max(0.0, 1.0 - abs(with_prefix.tau_mean) / 0.2)
        return disagreement + 0.5 * uncertainty + 0.25 * near_zero

    return score


def _collect_counterfactual(
    *,
    db: str,
    episodes: int,
    seed: int,
    top_k: int,
    scenario_mix: str,
    output: str,
    allow_duplicates: bool,
    target_policy_name: str,
    prefix_mode: str,
    max_prefix_size: int,
    continuation_policy_manifest_path: str | None = None,
    boundary_critic_checkpoint: str | None = None,
    round_id: str = "adhoc",
    round_index: int = 0,
    require_min_continuation_share_rate: float | None = None,
    require_max_continuation_share_rate: float | None = None,
    require_max_hard_risk_share_rate: float | None = None,
) -> None:
    if scenario_mix not in {"balanced", "interaction"}:
        raise ValueError("only --scenario-mix balanced|interaction is supported")
    repository = SQLiteSharedMemoryRepository(db)
    seed_repository(repository)
    provider = CounterfactualToyTaskProvider()
    provider.ensure_memories(repository)
    base_snapshot = repository.create_read_snapshot()
    base_revision = repository.current_revision()
    base_snapshot_digest = canonical_digest(base_snapshot)
    policy_manifest = (
        load_policy_manifest(continuation_policy_manifest_path)
        if continuation_policy_manifest_path
        else create_no_share_manifest()
    )
    writer = PairedRecordWriter(output, allow_duplicates=allow_duplicates)
    collector = PairedRolloutCollector()
    continuation_policy = FrozenNoShareContinuationPolicy()
    if policy_manifest.policy_kind == "frozen_critic_sequential":
        continuation_policy = FrozenCriticSequentialContinuationPolicy(
            manifest=policy_manifest,
            critic=FourOutcomeTransferCritic.load(
                Path(str(policy_manifest.source_critic_checkpoint_path))
            ),
        )
    elif policy_manifest.policy_kind == "frozen_risk_constrained_exploration":
        continuation_policy = FrozenRiskConstrainedExplorationPolicy(
            manifest=policy_manifest,
            critic=FourOutcomeTransferCritic.load(
                Path(str(policy_manifest.source_critic_checkpoint_path))
            ),
        )
    scenarios = ["positive", "negative", "neutral_success", "neutral_failure"]
    if scenario_mix == "interaction":
        scenarios = [
            *scenarios,
            "prefix_sensitive",
            "flip_pos_to_neg",
            "flip_neg_to_pos",
            "flip_neu_to_neg",
            "flip_neu_to_pos",
        ]
    counts: Counter[str] = Counter()
    prefix_counts: Counter[int] = Counter()
    cross_counts: Counter[tuple[int, str]] = Counter()
    target_policy = (
        ScenarioDesignatedTargetPolicy()
        if target_policy_name == "scenario-designated"
        else UniformCandidateTargetPolicy()
    )
    cards_by_id = {card.memory_id: card for card in repository.get_routing_cards()}
    boundary_critic = (
        FourOutcomeTransferCritic.load(Path(boundary_critic_checkpoint))
        if prefix_mode == "interaction-boundary" and boundary_critic_checkpoint
        else None
    )
    if prefix_mode == "interaction-boundary":
        prefix_sampler = InteractionBoundaryPrefixSampler(
            InteractionBoundaryConfig(max_prefix_size=max_prefix_size),
            cards_by_id=cards_by_id,
            prefix_size_counts=prefix_counts,
        )
    else:
        prefix_sampler = StratifiedEligiblePrefixSampler(
            PrefixSamplingConfig(mode=prefix_mode, max_prefix_size=max_prefix_size),
            prefix_size_counts=prefix_counts,
        )
    policy_round = PolicyRoundLedger().begin_round(
        round_id=round_id,
        round_index=round_index,
        continuation_policy=policy_manifest,
        base_memory_store_revision=base_revision,
        base_memory_snapshot_digest=base_snapshot_digest,
        top_k=top_k,
        prefix_sampling_config={
            "mode": prefix_mode,
            "max_prefix_size": max_prefix_size,
        },
        target_selection_policy_name=target_policy.policy_name,
        target_selection_policy_version=target_policy.policy_version,
        record_output_path=output,
    )

    for index in range(episodes):
        if repository.current_revision() != base_revision:
            raise RuntimeError(
                "memory store changed during collection round; abort to prevent temporal leakage"
            )
        scenario = scenarios[index % len(scenarios)]
        task_spec = provider.generate(scenario=scenario, seed=seed + index)
        recorder = InMemoryDecisionPointRecorder()
        run_episode(
            seed=seed + index,
            memory_pool=repository,
            top_k=top_k,
            task=task_spec.task,
            environment_observation=task_spec.environment_observation,
            episode_id=task_spec.episode_id,
            task_id=task_spec.task_id,
            decision_point_recorder=recorder,
        )
        decision_point = _select_decision_point(recorder, task_spec.target_memory_id)
        # Ensure target and forced_prefix memories are in the candidate proposal
        missing_ids = [task_spec.target_memory_id]
        if task_spec.forced_prefix:
            missing_ids.extend(task_spec.forced_prefix)
        decision_point = _ensure_memories_in_proposal(
            decision_point, missing_ids, cards_by_id
        )
        if boundary_critic is not None and isinstance(
            prefix_sampler, InteractionBoundaryPrefixSampler
        ):
            prefix_sampler.critic_scorer = _make_boundary_critic_scorer(
                critic=boundary_critic,
                decision_point=decision_point,
                cards_by_id=cards_by_id,
            )
        plan = build_candidate_traversal_plan(
            proposal=decision_point.candidate_proposal,
            traversal_seed=seed + index,
            target_memory_id=task_spec.target_memory_id,
            target_selection_policy=target_policy,
            prefix_sampler=prefix_sampler,
            target_selection_seed=seed + 10_000 + index,
            prefix_sampling_seed=seed + 20_000 + index,
            selected_before=list(task_spec.forced_prefix) if task_spec.forced_prefix else None,
        )
        record = collector.collect(
            decision_point=decision_point,
            traversal_plan=plan,
            repository=repository,
            continuation_policy=continuation_policy,
            policy_round=policy_round,
            evaluation_group_metadata=provider.evaluation_metadata(
                scenario=scenario,
                target_memory_id=plan.target_memory_id,
                selected_before=plan.selected_before,
                seed=seed + index,
            ),
        )
        if repository.current_revision() != base_revision:
            raise RuntimeError(
                "memory store changed during collection round; abort to prevent temporal leakage"
            )
        writer.append(record)
        policy_round = PolicyRoundLedger().append_record(policy_round)
        counts[record.transfer_class] += 1
        cross_counts[(record.prefix_size, record.transfer_class)] += 1

    print(f"record_count={sum(counts.values())}")
    for label in ["positive", "negative", "neutral_success", "neutral_failure"]:
        print(f"{label}={counts[label]}")
    print(f"prefix_size_counts={dict(prefix_counts)}")
    print(f"target_policy={target_policy.policy_name}")
    print(f"prefix_policy={prefix_sampler.policy_name}")
    print(f"prefix_sampling_fallbacks={prefix_sampler.fallback_count}")
    print(f"continuation_policy_fingerprint={policy_manifest.fingerprint}")
    print(f"collection_round_id={policy_round.round_id}")
    print(
        "prefix_class_counts="
        + str({f"{size}:{label}": count for (size, label), count in cross_counts.items()})
    )
    unique_candidate_count = len({record.candidate_memory_id for record in writer.load()})
    print(f"unique_candidate_memories={unique_candidate_count}")
    print(f"output={output}")
    if require_min_continuation_share_rate is not None:
        _validate_collection_quality(
            input_path=output,
            output=str(Path(output).with_suffix(".quality.json")),
            min_share_rate=require_min_continuation_share_rate,
            max_share_rate=require_max_continuation_share_rate
            if require_max_continuation_share_rate is not None
            else 1.0,
            max_hard_risk_share_rate=require_max_hard_risk_share_rate
            if require_max_hard_risk_share_rate is not None
            else 1.0,
        )


def _select_decision_point(
    recorder: InMemoryDecisionPointRecorder,
    target_memory_id: str,
):
    """Select a planner decision point that contains the target memory.

    If no decision point contains the target memory, return the first planner
    decision point and let the caller inject the target.
    """
    for point in recorder.decision_points:
        ids = [candidate.memory_id for candidate in point.candidate_proposal.ranked_candidates]
        if point.receiver_agent_id == "planner" and target_memory_id in ids:
            return point
    # Fallback: return first planner decision point
    for point in recorder.decision_points:
        if point.receiver_agent_id == "planner":
            return point
    raise ValueError(f"no planner decision point found for target memory {target_memory_id}")


def _ensure_memories_in_proposal(
    decision_point: DecisionPoint,
    memory_ids: list[str],
    cards_by_id: dict[str, Any],
) -> DecisionPoint:
    """Ensure specified memories are in the candidate proposal.

    If a memory is not in the ranked candidates, add it with a minimal score.
    """
    from smtr.router.candidate_proposer import CandidateScore

    existing_ids = {c.memory_id for c in decision_point.candidate_proposal.ranked_candidates}
    missing = [m for m in memory_ids if m not in existing_ids]
    if not missing:
        return decision_point

    new_candidates = list(decision_point.candidate_proposal.ranked_candidates)
    for memory_id in missing:
        new_candidates.append(
            CandidateScore(
                memory_id=memory_id,
                total_score=0.01,
                goal_similarity=0.0,
                task_tag_overlap=0.0,
                environment_compatibility=1.0,
                receiver_compatibility=1.0,
                explicit_environment_conflict=False,
                score_explanation=["injected for flip scenario"],
            )
        )
    proposal = decision_point.candidate_proposal.model_copy(
        update={"ranked_candidates": new_candidates}
    )
    return decision_point.model_copy(update={"candidate_proposal": proposal})


def _inspect_paired_records(
    input_path: str,
    *,
    show_prefix_distribution: bool = False,
    show_factor_coverage: bool = False,
    show_continuation_behavior: bool = False,
) -> None:
    records = PairedRecordWriter(input_path, allow_duplicates=True).load()
    counts: Counter[str] = Counter(record.transfer_class for record in records)
    total = len(records)
    mean_effect = (
        sum(record.marginal_effect for record in records) / total if total else 0.0
    )
    print(f"record count: {total}")
    print(f"schema versions: {dict(Counter(record.schema_version for record in records))}")
    print(f"class counts: {dict(counts)}")
    for label in ["positive", "negative", "neutral_success", "neutral_failure"]:
        rate = counts[label] / total if total else 0.0
        print(f"{label} rate: {rate:.3f}")
    print(f"mean marginal effect: {mean_effect:.3f}")
    print(
        "unique candidate memories: "
        + ",".join(sorted({record.candidate_memory_id for record in records}))
    )
    print(
        "unique receiver agents: "
        + ",".join(sorted({record.receiver_agent_id for record in records}))
    )
    print(
        "memory store revisions observed: "
        + ",".join(
            str(value)
            for value in sorted({record.memory_store_revision for record in records})
        )
    )
    if show_prefix_distribution:
        print(f"prefix-size counts: {dict(Counter(record.prefix_size for record in records))}")
        cross = Counter((record.prefix_size, record.transfer_class) for record in records)
        print(
            "prefix-size x transfer-class: "
            + str({f"{size}:{label}": count for (size, label), count in sorted(cross.items())})
        )
        print(
            "sampling-policy counts: "
            + str(
                dict(
                    Counter(
                        (
                            record.target_selection_policy_name,
                            record.prefix_sampling_policy_name,
                        )
                        for record in records
                    )
                )
            )
        )
        by_prefix: dict[int, list[int]] = {}
        for record in records:
            by_prefix.setdefault(record.prefix_size, []).append(record.marginal_effect)
        print(
            "mean marginal effect by prefix size: "
            + str(
                {
                    size: sum(values) / len(values)
                    for size, values in sorted(by_prefix.items())
                }
            )
        )
    if show_factor_coverage:
        factor_count = len(
            {record.evaluation_group_metadata.factor_combination_id for record in records}
        )
        regimes = sorted(
            {record.evaluation_group_metadata.environment_regime for record in records}
        )
        target_families = sorted(
            {record.evaluation_group_metadata.target_memory_family for record in records}
        )
        print(
            "factor combination count: "
            + str(factor_count)
        )
        print("environment regimes: " + ",".join(regimes))
        print("target families: " + ",".join(target_families))
    if show_continuation_behavior:
        print("continuation quality: " + str(_continuation_quality(records)))


def _ingest_paired_evidence(db: str, input_path: str) -> None:
    repo = SQLiteSharedMemoryRepository(db)
    records = PairedRecordWriter(input_path, allow_duplicates=True).load()
    before = repo.current_revision()
    ingestor = PairedTransferEvidenceIngestor()
    inserted = duplicates = 0
    counts: Counter[str] = Counter()
    for record in records:
        result = ingestor.ingest_record(repository=repo, record=record)
        if result == "inserted":
            inserted += 1
            counts[record.transfer_class] += 1
        else:
            duplicates += 1
    after = repo.current_revision()
    print(f"records_read={len(records)}")
    print(f"inserted={inserted}")
    print(f"duplicates={duplicates}")
    print(f"positive_ingested={counts['positive']}")
    print(f"negative_ingested={counts['negative']}")
    print(f"neutral_ingested={counts['neutral_success'] + counts['neutral_failure']}")
    print(f"store_revision_before={before}")
    print(f"store_revision_after={after}")


def _train_transfer_critic(
    input_path: str,
    output: str,
    seed: int,
    n_bootstrap: int,
    test_fraction: float,
    require_policy_fingerprint: str | None = None,
) -> None:
    records = load_paired_records_for_training(Path(input_path))
    if any(record.schema_version < "1.2" for record in records):
        raise ValueError("v1.1 records must be migrated to schema 1.2 before training")
    fingerprints = {record.continuation_policy_fingerprint for record in records}
    if len(fingerprints) != 1:
        raise ValueError(
            "mixed continuation-policy estimands detected; train one critic per "
            "frozen continuation policy"
        )
    policy_fingerprint = next(iter(fingerprints))
    if require_policy_fingerprint and policy_fingerprint != require_policy_fingerprint:
        raise ValueError("required policy fingerprint does not match input records")
    train, test = group_split(records, seed=seed, test_fraction=test_fraction)
    critic = FourOutcomeTransferCritic().fit(train, seed=seed, n_bootstrap=n_bootstrap)
    critic.critic_version = f"{critic.critic_version}_{str(policy_fingerprint)[:8]}"
    output_path = Path(output)
    critic.save(output_path)
    checkpoint_sha = file_sha256(output_path)
    metrics = evaluate_records(critic, test)
    metadata = {
        "critic_version": critic.critic_version,
        "encoder_schema_version": critic.encoder.schema_version,
        "n_features": critic.encoder.n_features,
        "n_bootstrap": n_bootstrap,
        "bootstrap_seeds": critic.bootstrap_seeds,
        "train_record_count": len(train),
        "test_record_count": len(test),
        "class_distribution_train": distribution(record.transfer_class for record in train),
        "class_distribution_test": distribution(record.transfer_class for record in test),
        "prefix_size_distribution_train": distribution(record.prefix_size for record in train),
        "prefix_size_distribution_test": distribution(record.prefix_size for record in test),
        "feature_policy_no_payload_steps": True,
        "support_threshold": critic.support_threshold,
        "estimand_policy_fingerprint": policy_fingerprint,
        "estimand_policy_name": None,
        "estimand_policy_version": None,
        "collection_round_ids": sorted({record.collection_round_id for record in records}),
        "base_memory_snapshot_digests": sorted(
            {record.base_memory_snapshot_digest for record in records}
        ),
        "temporal_integrity_verified": True,
        "checkpoint_sha256": checkpoint_sha,
    }
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    write_json(output_path.with_suffix(".metrics.json"), metrics)
    print(f"checkpoint={output}")
    print(f"train_record_count={len(train)}")
    print(f"test_record_count={len(test)}")


def _evaluate_transfer_critic(
    input_path: str,
    checkpoint: str,
    output: str,
    split_suite: str | None = None,
) -> None:
    records = load_paired_records_for_training(Path(input_path))
    critic = FourOutcomeTransferCritic.load(Path(checkpoint))
    if split_suite in {"strict", "compositional"}:
        metrics = evaluate_compositional_splits(records, critic, split_suite=split_suite)
    else:
        metrics = evaluate_records(critic, records)
    write_json(Path(output), metrics)
    print(f"record_count={len(records)}")
    print(f"output={output}")


def _create_no_share_policy(output: str) -> None:
    manifest = create_no_share_manifest()
    save_policy_manifest(manifest, output)
    print(f"policy_fingerprint={manifest.fingerprint}")


def _build_critic_continuation_policy(
    *,
    critic_path: str,
    tau_lcb_threshold: float,
    negative_risk_ucb_threshold: float,
    reject_low_support: bool,
    output: str,
) -> None:
    metadata_path = Path(critic_path).with_suffix(".metadata.json")
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = with_fingerprint(
        ContinuationPolicyManifest(
            policy_name="FrozenCriticSequentialContinuationPolicy",
            policy_version="1",
            policy_kind="frozen_critic_sequential",
            source_critic_checkpoint_path=critic_path,
            source_critic_checkpoint_sha256=file_sha256(critic_path),
            source_critic_estimand_policy_fingerprint=metadata[
                "estimand_policy_fingerprint"
            ],
            tau_lcb_threshold=tau_lcb_threshold,
            negative_risk_ucb_threshold=negative_risk_ucb_threshold,
            reject_low_support=reject_low_support,
            feature_encoder_schema_version=metadata["encoder_schema_version"],
        )
    )
    save_policy_manifest(manifest, output)
    print(f"policy_fingerprint={manifest.fingerprint}")


def _build_exploratory_continuation_policy(
    *,
    critic_path: str,
    config: ExplorationPolicyConfig,
    output: str,
) -> None:
    import json

    metadata = json.loads(
        Path(critic_path).with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    manifest = with_fingerprint(
        ContinuationPolicyManifest(
            policy_name="FrozenRiskConstrainedExplorationPolicy",
            policy_version=FrozenRiskConstrainedExplorationPolicy.policy_version,
            policy_kind="frozen_risk_constrained_exploration",
            source_critic_checkpoint_path=critic_path,
            source_critic_checkpoint_sha256=file_sha256(critic_path),
            source_critic_estimand_policy_fingerprint=metadata[
                "estimand_policy_fingerprint"
            ],
            feature_encoder_schema_version=metadata["encoder_schema_version"],
            exploration_config=config.model_dump(),
        )
    )
    save_policy_manifest(manifest, output)
    print(f"policy_fingerprint={manifest.fingerprint}")


def _migrate_paired_policy_metadata(
    *,
    input_path: str,
    output: str,
    round_id: str,
    round_index: int,
    policy_manifest_path: str,
    base_store_revision: int,
    base_memory_snapshot_digest: str,
) -> None:
    records = PairedRecordWriter(input_path, allow_duplicates=True).load()
    if any(record.schema_version != "1.1" for record in records):
        raise ValueError("migrate-paired-policy-metadata only accepts schema v1.1 records")
    manifest = load_policy_manifest(policy_manifest_path)
    writer = PairedRecordWriter(output, allow_duplicates=True)
    for record in records:
        migrated = record.model_copy(
            update={
                "schema_version": "1.2",
                "collection_round_id": round_id,
                "continuation_policy_fingerprint": manifest.fingerprint,
                "base_memory_store_revision": base_store_revision,
                "base_memory_snapshot_digest": base_memory_snapshot_digest,
                "migrated_from_schema": "1.1",
            }
        )
        writer.append(migrated)
    print(f"records_migrated={len(records)}")
    print(f"policy_fingerprint={manifest.fingerprint}")


def _diagnose_shortcuts(input_path: str, output: str) -> None:
    records = PairedRecordWriter(input_path, allow_duplicates=True).load()
    report = shortcut_diagnostics(records)
    write_json(Path(output), report)
    print(f"record_count={len(records)}")
    print(f"output={output}")


def _continuation_quality(records) -> dict:
    counts: Counter[str] = Counter()
    continuation = 0
    shares = 0
    hard_risk_shares = 0
    for record in records:
        for outcome in [record.share_outcome, record.withhold_outcome]:
            for trace in outcome.router_trace:
                for decision in trace["decisions"]:
                    if decision.get("decision_source") != "frozen_continuation":
                        continue
                    continuation += 1
                    mode = decision.get("decision_mode") or decision.get("reason")
                    counts[str(mode)] += 1
                    if decision.get("action") == "share":
                        shares += 1
                        if (decision.get("negative_risk_ucb") or 0.0) > 0.35:
                            hard_risk_shares += 1
    return {
        "continuation_decision_count": continuation,
        "continuation_share_count": shares,
        "continuation_share_rate": shares / continuation if continuation else 0.0,
        "safe_exploit_share_count": counts["safe_exploit"],
        "boundary_explore_share_count": counts["boundary_explore"],
        "risk_veto_count": counts["risk_veto"],
        "hard_ood_veto_count": counts["hard_ood_veto"],
        "budget_exhausted_count": counts["budget_exhausted"],
        "hard_risk_share_rate": hard_risk_shares / max(1, shares),
    }


def _validate_collection_quality(
    *,
    input_path: str,
    output: str,
    min_share_rate: float,
    max_share_rate: float,
    max_hard_risk_share_rate: float,
) -> None:
    records = PairedRecordWriter(input_path, allow_duplicates=True).load()
    report = _continuation_quality(records)
    report["record_count"] = len(records)
    report["prefix_size_distribution"] = dict(Counter(record.prefix_size for record in records))
    report["transfer_class_distribution"] = dict(
        Counter(record.transfer_class for record in records)
    )
    failures = []
    if report["continuation_share_rate"] < min_share_rate:
        failures.append("continuation_share_rate below minimum")
    if report["continuation_share_rate"] > max_share_rate:
        failures.append("continuation_share_rate above maximum")
    if report["hard_risk_share_rate"] > max_hard_risk_share_rate:
        failures.append("hard_risk_share_rate above maximum")
    report["failures"] = failures
    write_json(Path(output), report)
    print(f"continuation_share_rate={report['continuation_share_rate']:.3f}")
    print(f"hard_risk_share_rate={report['hard_risk_share_rate']:.3f}")
    print(f"output={output}")
    if failures:
        raise SystemExit("; ".join(failures))


def _scan_transfer_feature_leakage(input_path: str, output: str) -> None:
    from smtr.evaluation.leakage_scanner import TransferFeatureLeakageScanner

    records = PairedRecordWriter(input_path, allow_duplicates=True).load()
    report = TransferFeatureLeakageScanner().scan(records)
    write_json(Path(output), report)
    print(f"violations={len(report['violations'])}")
    print(f"output={output}")
    if report["violations"]:
        raise SystemExit("forbidden feature leakage detected")


def _audit_feature_blocks(input_path: str, seed: int, n_bootstrap: int, output: str) -> None:
    records = load_paired_records_for_training(Path(input_path))
    report = audit_feature_blocks(records, seed=seed, n_bootstrap=n_bootstrap)
    write_json(Path(output), report)
    print(f"output={output}")


def _audit_interaction(input_path: str, checkpoint: str, output: str, mode: str) -> None:
    records = load_paired_records_for_training(Path(input_path))
    critic = FourOutcomeTransferCritic.load(Path(checkpoint))
    report = audit_interaction(records, critic, mode=mode)
    write_json(Path(output), report)
    print(f"matched_pair_count={report['matched_pair_count']}")
    print(f"output={output}")


def _serve_api(host: str, port: int, model: str) -> None:
    """Start the SMTR API server."""
    from smtr.runtime.api_server import run_server, set_llm
    from smtr.runtime.real_llm import RealLLM

    llm = RealLLM(model_name=model)
    set_llm(llm)
    print(f"Starting SMTR API server on {host}:{port}")
    print(f"Model: {model}")
    run_server(host=host, port=port)


def _demo_real(
    seed: int,
    model: str,
    api_base: str | None,
    use_tool_env: bool,
    router_mode: str = "no-memory",
    critic_checkpoint: str | None = None,
    max_shares_per_invocation: int | None = None,
) -> None:
    """Run demo with real LLM."""
    from smtr.runtime.real_llm import RealLLM
    from smtr.runtime.tool_environment import ToolEnvironment

    llm = RealLLM(model_name=model, api_base=api_base)
    env_factory = (lambda s: ToolEnvironment(seed=s)) if use_tool_env else None
    router = build_router(
        mode=router_mode,
        critic_checkpoint=critic_checkpoint,
        max_shares_per_invocation=max_shares_per_invocation,
    )

    state = run_demo(seed=seed, llm=llm, env_factory=env_factory, router=router)

    print(f"Task: {state['task']}")
    print(f"Team success: {state.get('team_success')}")
    print(f"Team reward: {state.get('team_reward')}")
    print(f"Team summary: {state.get('team_summary')}")
    if state.get("agent_outputs", {}).get("planner"):
        planner_output = state["agent_outputs"]["planner"]
        print("\nPlanner output:")
        print(f"  Plan: {planner_output.get('plan')}")
        print(f"  Explanation: {planner_output.get('explanation')}")


def _compare_routers(args: argparse.Namespace) -> None:
    """Run ablation comparison experiment."""
    from pathlib import Path

    # Validate arguments
    if args.scenario_replicates <= 0:
        raise ValueError("--scenario-replicates must be > 0")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.max_shares_per_invocation is not None and args.max_shares_per_invocation < 0:
        raise ValueError("--max-shares-per-invocation must be >= 0")
    if not args.traversal_seeds:
        raise ValueError("--traversal-seeds must not be empty")
    if args.critic_checkpoint and not Path(args.critic_checkpoint).exists():
        raise ValueError(
            f"critic checkpoint not found: {args.critic_checkpoint}"
        )
    if (
        getattr(args, "factual_success_checkpoint", None)
        and not Path(args.factual_success_checkpoint).exists()
    ):
        raise ValueError(
            f"factual success checkpoint not found: {args.factual_success_checkpoint}"
        )
    if args.budget_manifest_path and not Path(args.budget_manifest_path).exists():
        raise ValueError(
            f"budget manifest not found: {args.budget_manifest_path}"
        )
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"output directory already exists: {output_dir}; "
            "use --overwrite to replace"
        )

    config = ExperimentConfig(
        db_path=args.db,
        critic_checkpoint=args.critic_checkpoint,
        task_seeds=args.task_seeds,
        generation_seeds=args.generation_seeds,
        traversal_seeds=args.traversal_seeds,
        scenario_replicates=args.scenario_replicates,
        top_k=args.top_k,
        max_shares_per_invocation=args.max_shares_per_invocation,
        negative_risk_budget=args.negative_risk_budget,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        fail_fast=not args.continue_on_error,
        scenario=args.scenario,
        methods=args.methods,
        enable_ablation_methods=args.enable_ablation_methods,
        budget_manifest_path=args.budget_manifest_path,
        factual_success_checkpoint=getattr(args, "factual_success_checkpoint", None),
        factual_success_threshold=getattr(args, "factual_success_threshold", None),
    )

    runner = ComparisonRunner(config)
    summary = runner.run()

    # Print summary for all methods
    print(f"experiment_id={runner.experiment_id}")
    print(f"n_base_episodes={summary.n_base_episodes}")
    print(f"n_traversal_runs={summary.n_traversal_runs}")
    for method_id, method_summary in summary.methods.items():
        net_transfer_rate = None
        if (
            method_summary.positive_transfer_rate is not None
            and method_summary.negative_transfer_rate is not None
        ):
            net_transfer_rate = (
                method_summary.positive_transfer_rate
                - method_summary.negative_transfer_rate
            )
        print(f"\n{method_id}:")
        print(f"  success_rate={method_summary.success_rate:.3f}")
        if method_summary.positive_transfer_rate is not None:
            print(
                f"  positive_transfer_rate={method_summary.positive_transfer_rate:.3f}"
            )
        if method_summary.negative_transfer_rate is not None:
            print(
                f"  negative_transfer_rate={method_summary.negative_transfer_rate:.3f}"
            )
        if net_transfer_rate is not None:
            print(f"  net_transfer_rate={net_transfer_rate:.3f}")
        if method_summary.mean_exposure_per_invocation is not None:
            print(
                "  mean_exposure_per_invocation="
                f"{method_summary.mean_exposure_per_invocation:.3f}"
            )
        if method_summary.total_exposure_per_episode is not None:
            print(
                "  total_exposure_per_episode="
                f"{method_summary.total_exposure_per_episode:.3f}"
            )
        print(f"  all_withhold_rate={method_summary.all_withhold_rate:.3f}")
        if method_summary.opportunity_capture is not None:
            print(f"  opportunity_capture={method_summary.opportunity_capture:.3f}")
        if method_summary.safety_preservation is not None:
            print(f"  safety_preservation={method_summary.safety_preservation:.3f}")
    if not summary.experiment_valid:
        print(f"\nWARNING: experiment invalid — {summary.invalid_reason}")
    print(f"\noutput_dir={args.output_dir}")


def _run_gate_ablation(args: argparse.Namespace) -> None:
    """Run the formal SMTR risk-condition ablation method set."""
    methods = [
        "B0",
        "B1-Top1",
        "B1-AllCandidates",
        "B1-Matched",
        "SMTR",
        "EffectOnly-SMTR",
        "Static-SMTR",
        "FactualSuccess-SMTR",
    ]
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_output_dir = output_dir / "_scenarios"
    scenario_output_dir.mkdir(exist_ok=True)
    runs = []
    errors = []
    base_manifest_lines: list[str] = []
    for scenario in args.scenarios:
        config = ExperimentConfig(
            db_path=args.db,
            critic_checkpoint=args.critic_checkpoint,
            factual_success_checkpoint=args.factual_success_checkpoint,
            factual_success_threshold=args.factual_success_threshold,
            task_seeds=args.task_seeds,
            generation_seeds=args.generation_seeds,
            traversal_seeds=args.traversal_seeds,
            scenario_replicates=args.scenario_replicates,
            top_k=args.top_k,
            max_shares_per_invocation=args.max_shares_per_invocation,
            negative_risk_budget=args.negative_risk_budget,
            output_dir=str(scenario_output_dir / scenario),
            overwrite=True,
            fail_fast=args.fail_fast,
            scenario=scenario,
            methods=methods,
            enable_ablation_methods=True,
            budget_manifest_path=args.budget_manifest,
            bootstrap_n=args.bootstrap_n,
        )
        runner = ComparisonRunner(config)
        runner.run()
        writer = ExperimentWriter(config.output_dir, overwrite=True)
        runs.extend(writer.load_runs())
        errors.extend(writer.load_errors())
        manifest_path = Path(config.output_dir) / "base_episode_manifest.jsonl"
        if manifest_path.exists():
            base_manifest_lines.extend(manifest_path.read_text(encoding="utf-8").splitlines())

    aggregate_config = {
        "db_path": args.db,
        "critic_checkpoint": args.critic_checkpoint,
        "training_manifest": args.training_manifest,
        "budget_manifest": args.budget_manifest,
        "scenarios": args.scenarios,
        "task_seeds": args.task_seeds,
        "generation_seeds": args.generation_seeds,
        "traversal_seeds": args.traversal_seeds,
        "top_k": args.top_k,
        "max_shares_per_invocation": args.max_shares_per_invocation,
        "negative_risk_budget": args.negative_risk_budget,
        "methods": methods,
    }
    (output_dir / "config.json").write_text(
        json.dumps(aggregate_config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "base_episode_manifest.jsonl").write_text(
        "\n".join(base_manifest_lines) + ("\n" if base_manifest_lines else ""),
        encoding="utf-8",
    )
    (output_dir / "runs.jsonl").write_text(
        "".join(run.model_dump_json() + "\n" for run in runs),
        encoding="utf-8",
    )
    decision_lines = []
    for run in runs:
        for invocation in run.invocations:
            for decision in invocation.decisions:
                decision_lines.append(
                    json.dumps(
                        {
                            "base_episode_id": run.base_episode_id,
                            "method": run.method,
                            "scenario": run.scenario,
                            "traversal_seed": run.traversal_seed,
                            "graph_node": invocation.graph_node,
                            "receiver_agent_id": invocation.receiver_agent_id,
                            **decision.model_dump(),
                        },
                        default=str,
                    )
                )
    (output_dir / "decisions.jsonl").write_text(
        "\n".join(decision_lines) + ("\n" if decision_lines else ""),
        encoding="utf-8",
    )
    (output_dir / "errors.jsonl").write_text(
        "".join(json.dumps(error, default=str) + "\n" for error in errors),
        encoding="utf-8",
    )
    invocation_lines = []
    for run in runs:
        for invocation in run.invocations:
            invocation_lines.append(
                json.dumps(
                    {
                        "base_episode_id": run.base_episode_id,
                        "method": run.method,
                        "traversal_seed": run.traversal_seed,
                        **invocation.model_dump(),
                    },
                    default=str,
                )
            )
    (output_dir / "invocations.jsonl").write_text(
        "\n".join(invocation_lines) + ("\n" if invocation_lines else ""),
        encoding="utf-8",
    )
    summary_config = ExperimentConfig(
        db_path=args.db,
        critic_checkpoint=args.critic_checkpoint,
        output_dir=str(output_dir),
        methods=methods,
        negative_risk_budget=args.negative_risk_budget,
        enable_ablation_methods=True,
        budget_manifest_path=args.budget_manifest,
        bootstrap_n=args.bootstrap_n,
    )
    summary = compute_summary(runs, summary_config)
    (output_dir / "summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    method_specs = {
        key: spec.__dict__
        for key, spec in build_default_specs(
            critic_checkpoint=args.critic_checkpoint,
            factual_success_checkpoint=args.factual_success_checkpoint,
            budget_manifest_path=args.budget_manifest,
            max_shares_per_invocation=args.max_shares_per_invocation,
            negative_risk_budget=args.negative_risk_budget,
            include_ablations=True,
        ).items()
        if spec.display_label in methods
    }
    (output_dir / "method_specs.json").write_text(
        json.dumps(method_specs, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    gate_diagnostics = {
        method: diagnostic.model_dump()
        for method, diagnostic in compute_gate_diagnostics(
            runs,
            epsilon=args.negative_risk_budget,
        ).items()
    }
    paired = compute_group_bootstrap_ci(
        runs,
        method_pairs=[
            ("SMTR", "B0"),
            ("SMTR", "B1-AllCandidates"),
            ("SMTR", "B1-Top1"),
            ("SMTR", "B1-Matched"),
            ("SMTR", "EffectOnly-SMTR"),
            ("SMTR", "Static-SMTR"),
            ("SMTR", "FactualSuccess-SMTR"),
        ],
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_n=config.bootstrap_n,
    )
    (output_dir / "gate_diagnostics.json").write_text(
        json.dumps(gate_diagnostics, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "paired_statistics.json").write_text(
        json.dumps(paired, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "paired_comparisons.json").write_text(
        json.dumps(paired, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "static_set_diagnostics.json").write_text(
        json.dumps(compute_static_set_diagnostics(runs), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    scenario_slices = {}
    for scenario in args.scenarios:
        scenario_runs = [run for run in runs if run.scenario == scenario]
        if scenario_runs:
            scenario_slices[scenario] = compute_summary(
                scenario_runs,
                summary_config,
            ).model_dump()
    (output_dir / "scenario_slices.json").write_text(
        json.dumps(scenario_slices, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "# Formal SMTR Core Ablation Smoke\n\n"
        "This smoke run validates implementation and metrics only.\n",
        encoding="utf-8",
    )
    print(f"experiment_id={runner.experiment_id}")
    print(f"n_base_episodes={summary.n_base_episodes}")
    print(f"n_traversal_runs={summary.n_traversal_runs}")
    print(f"output_dir={output_dir}")


def _run_order_sensitivity(args: argparse.Namespace) -> None:
    """Run the minimal SMTR-only order-sensitivity diagnostic."""
    if args.method != "SMTR":
        raise ValueError("run-order-sensitivity only supports --method SMTR")
    if not args.enumerate_permutations:
        raise ValueError("run-order-sensitivity requires --enumerate-permutations")
    metrics = run_order_sensitivity(
        db_path=args.db,
        critic_checkpoint=args.critic_checkpoint,
        output_dir=args.output_dir,
        task_seeds=args.task_seeds,
        generation_seeds=args.generation_seeds,
        scenario_replicates=args.scenario_replicates,
        scenario=args.scenario_filter.replace("-", "_"),
        candidate_count=4,
        negative_risk_budget=args.negative_risk_budget,
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
    )
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")
    print(f"output_dir={args.output_dir}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"toy", "marble", "robust"}:
        target = sys.argv[1]
        sys.argv = [f"python -m smtr.{target}.cli", *sys.argv[2:]]
        if target == "toy":
            from smtr.toy.cli import main as toy_main

            toy_main()
        elif target == "marble":
            from smtr.marble.cli import main as marble_main

            marble_main()
        else:
            from smtr.robust.cli import main as robust_main

            robust_main()
        return
    parser = argparse.ArgumentParser(prog="smtr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_parser = subparsers.add_parser("seed-memories")
    seed_parser.add_argument("--db", required=True)

    list_parser = subparsers.add_parser("list-memories")
    list_parser.add_argument("--db", required=True)

    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("--seed", type=int, default=7)
    demo_parser.add_argument("--db")
    demo_parser.add_argument("--top-k", type=int, default=4)
    demo_parser.add_argument(
        "--router-mode",
        choices=["no-memory", "relevance-topk", "learned"],
        default="no-memory",
    )
    demo_parser.add_argument("--critic-checkpoint")
    demo_parser.add_argument("--max-shares-per-invocation", type=int)

    collect_parser = subparsers.add_parser("collect-counterfactual")
    collect_parser.add_argument("--db", required=True)
    collect_parser.add_argument("--episodes", type=int, default=40)
    collect_parser.add_argument("--seed", type=int, default=7)
    collect_parser.add_argument("--top-k", type=int, default=4)
    collect_parser.add_argument("--scenario-mix", default="balanced")
    collect_parser.add_argument("--scenario-design", default="legacy")
    collect_parser.add_argument("--factorial-balance", default="stratified")
    collect_parser.add_argument(
        "--target-policy",
        choices=["uniform", "scenario-designated"],
        default="scenario-designated",
    )
    collect_parser.add_argument(
        "--prefix-mode",
        choices=["empty", "uniform", "stratified", "interaction-boundary"],
        default="empty",
    )
    collect_parser.add_argument("--max-prefix-size", type=int, default=2)
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--allow-duplicates", action="store_true")
    collect_parser.add_argument("--continuation-policy-manifest")
    collect_parser.add_argument("--boundary-critic-checkpoint")
    collect_parser.add_argument("--round-id", default="adhoc")
    collect_parser.add_argument("--round-index", type=int, default=0)
    collect_parser.add_argument("--require-min-continuation-share-rate", type=float)
    collect_parser.add_argument("--require-max-continuation-share-rate", type=float)
    collect_parser.add_argument("--require-max-hard-risk-share-rate", type=float)

    inspect_parser = subparsers.add_parser("inspect-paired-records")
    inspect_parser.add_argument("--input", required=True)
    inspect_parser.add_argument("--show-prefix-distribution", action="store_true")
    inspect_parser.add_argument("--show-factor-coverage", action="store_true")
    inspect_parser.add_argument("--show-continuation-behavior", action="store_true")

    ingest_parser = subparsers.add_parser("ingest-paired-evidence")
    ingest_parser.add_argument("--db", required=True)
    ingest_parser.add_argument("--input", required=True)

    train_parser = subparsers.add_parser("train-transfer-critic")
    train_parser.add_argument("--input", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--n-bootstrap", type=int, default=31)
    train_parser.add_argument("--test-fraction", type=float, default=0.2)
    train_parser.add_argument("--require-policy-fingerprint")

    eval_parser = subparsers.add_parser("evaluate-transfer-critic")
    eval_parser.add_argument("--input", required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument("--split-suite")

    no_share_parser = subparsers.add_parser("create-no-share-policy")
    no_share_parser.add_argument("--output", required=True)

    migrate_parser = subparsers.add_parser("migrate-paired-policy-metadata")
    migrate_parser.add_argument("--input", required=True)
    migrate_parser.add_argument("--output", required=True)
    migrate_parser.add_argument("--round-id", required=True)
    migrate_parser.add_argument("--round-index", type=int, required=True)
    migrate_parser.add_argument("--policy-manifest", required=True)
    migrate_parser.add_argument("--base-store-revision", type=int, required=True)
    migrate_parser.add_argument("--base-memory-snapshot-digest", required=True)

    build_policy_parser = subparsers.add_parser("build-critic-continuation-policy")
    build_policy_parser.add_argument("--critic", required=True)
    build_policy_parser.add_argument("--tau-lcb-threshold", type=float, default=0.0)
    build_policy_parser.add_argument("--negative-risk-ucb-threshold", type=float, default=0.2)
    build_policy_parser.add_argument("--reject-low-support", action="store_true")
    build_policy_parser.add_argument("--output", required=True)

    explore_parser = subparsers.add_parser("build-exploratory-continuation-policy")
    explore_parser.add_argument("--critic", required=True)
    explore_parser.add_argument("--safe-tau-lcb-threshold", type=float, default=0.0)
    explore_parser.add_argument("--safe-negative-risk-ucb-threshold", type=float, default=0.2)
    explore_parser.add_argument("--hard-negative-risk-veto-ucb", type=float, default=0.35)
    explore_parser.add_argument("--boundary-tau-band", type=float, default=0.15)
    explore_parser.add_argument("--soft-ood-multiplier", type=float, default=1.25)
    explore_parser.add_argument("--exploration-round-probability", type=float, default=0.30)
    explore_parser.add_argument("--max-total-shares-per-invocation", type=int, default=3)
    explore_parser.add_argument("--max-exploratory-shares-per-invocation", type=int, default=1)
    explore_parser.add_argument("--output", required=True)

    diagnose_parser = subparsers.add_parser("diagnose-shortcuts")
    diagnose_parser.add_argument("--input", required=True)
    diagnose_parser.add_argument("--output", required=True)

    quality_parser = subparsers.add_parser("validate-collection-quality")
    quality_parser.add_argument("--input", required=True)
    quality_parser.add_argument("--min-continuation-share-rate", type=float, required=True)
    quality_parser.add_argument("--max-continuation-share-rate", type=float, required=True)
    quality_parser.add_argument("--max-hard-risk-share-rate", type=float, required=True)
    quality_parser.add_argument("--require-factorial-diversity", action="store_true")
    quality_parser.add_argument("--output", required=True)

    leak_parser = subparsers.add_parser("scan-transfer-feature-leakage")
    leak_parser.add_argument("--input", required=True)
    leak_parser.add_argument("--output", required=True)

    ablation_parser = subparsers.add_parser("audit-feature-blocks")
    ablation_parser.add_argument("--input", required=True)
    ablation_parser.add_argument("--seed", type=int, default=7)
    ablation_parser.add_argument("--n-bootstrap", type=int, default=5)
    ablation_parser.add_argument("--split-suite", default="compositional")
    ablation_parser.add_argument("--output", required=True)

    prefix_audit_parser = subparsers.add_parser("audit-prefix-sensitivity")
    prefix_audit_parser.add_argument("--input", required=True)
    prefix_audit_parser.add_argument("--checkpoint", required=True)
    prefix_audit_parser.add_argument("--output", required=True)

    candidate_audit_parser = subparsers.add_parser("audit-candidate-substitution")
    candidate_audit_parser.add_argument("--input", required=True)
    candidate_audit_parser.add_argument("--checkpoint", required=True)
    candidate_audit_parser.add_argument("--output", required=True)

    marble_parser = subparsers.add_parser("inspect-marble-dataset")
    marble_parser.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    marble_parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(MARBLE_BENCHMARK_FILES),
        default=None,
        help="MARBLE MultiAgentBench scenarios to include.",
    )
    marble_parser.add_argument("--limit-per-scenario", type=int)
    marble_parser.add_argument("--output")

    serve_api_parser = subparsers.add_parser("serve-api")
    serve_api_parser.add_argument("--host", default="0.0.0.0")
    serve_api_parser.add_argument("--port", type=int, default=8000)
    serve_api_parser.add_argument("--model", default="Qwen/Qwen3.5-2B")

    demo_real_parser = subparsers.add_parser("demo-real")
    demo_real_parser.add_argument("--seed", type=int, default=7)
    demo_real_parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    demo_real_parser.add_argument("--api-base", help="Remote API base URL")
    demo_real_parser.add_argument("--use-tool-env", action="store_true")
    demo_real_parser.add_argument(
        "--router-mode",
        choices=["no-memory", "relevance-topk", "learned"],
        default="no-memory",
    )
    demo_real_parser.add_argument("--critic-checkpoint")
    demo_real_parser.add_argument("--max-shares-per-invocation", type=int)

    compare_parser = subparsers.add_parser("run-experiment")
    compare_parser.add_argument("--db", required=True)
    compare_parser.add_argument("--critic-checkpoint")
    compare_parser.add_argument("--factual-success-checkpoint")
    compare_parser.add_argument("--factual-success-threshold", type=float)
    compare_parser.add_argument("--task-seeds", type=int, nargs="+", default=[0])
    compare_parser.add_argument(
        "--generation-seeds", type=int, nargs="+", default=[0]
    )
    compare_parser.add_argument(
        "--traversal-seeds", type=int, nargs="+", default=[0, 1, 2]
    )
    compare_parser.add_argument("--scenario-replicates", type=int, default=1)
    compare_parser.add_argument("--top-k", type=int, default=4)
    compare_parser.add_argument("--max-shares-per-invocation", type=int, default=None)
    compare_parser.add_argument("--negative-risk-budget", type=float, default=0.2)
    compare_parser.add_argument("--enable-ablation-methods", action="store_true")
    compare_parser.add_argument("--output-dir", required=True)
    compare_parser.add_argument("--overwrite", action="store_true")
    compare_parser.add_argument(
        "--scenario",
        choices=[
            "positive", "negative", "neutral_success", "neutral_failure",
            "prefix_sensitive", "flip_pos_to_neg", "flip_neg_to_pos",
            "flip_neu_to_neg", "flip_neu_to_pos",
        ],
        help="Use counterfactual scenario task instead of default toy task",
    )
    compare_parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "B0",
            "B1-Top1",
            "B1-AllCandidates",
            "B1-Matched",
            "SMTR",
            "EffectOnly-SMTR",
            "Static-SMTR",
            "FactualSuccess-SMTR",
        ],
        default=None,
        help="Methods to compare (default: formal five-method registry)",
    )
    compare_parser.add_argument(
        "--budget-manifest-path",
        default=None,
        help="Path to B1-Matched budget manifest JSON",
    )
    compare_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Development only: write infrastructure failures to errors.jsonl.",
    )

    integrity_parser = subparsers.add_parser("audit-experiment-integrity")
    integrity_parser.add_argument("--experiment-dir", required=True)
    integrity_parser.add_argument("--m0-checkpoint", required=True)
    integrity_parser.add_argument("--a1-checkpoint")
    integrity_parser.add_argument("--output-dir", required=True)

    gate_parser = subparsers.add_parser("run-gate-ablation")
    gate_parser.add_argument("--db", default="data/smtr_memory.sqlite")
    gate_parser.add_argument("--critic-checkpoint", required=True)
    gate_parser.add_argument("--factual-success-checkpoint", required=True)
    gate_parser.add_argument("--factual-success-threshold", type=float)
    gate_parser.add_argument("--training-manifest")
    gate_parser.add_argument("--budget-manifest")
    gate_parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=[
            "positive",
            "negative",
            "neutral_success",
            "neutral_failure",
            "prefix_sensitive",
            "flip_pos_to_neg",
            "flip_neg_to_pos",
            "flip_neu_to_neg",
            "flip_neu_to_pos",
        ],
        default=["positive"],
    )
    gate_parser.add_argument("--task-seeds", type=int, nargs="+", default=[0, 1])
    gate_parser.add_argument("--generation-seeds", type=int, nargs="+", default=[0])
    gate_parser.add_argument("--traversal-seeds", type=int, nargs="+", default=[0, 1])
    gate_parser.add_argument("--scenario-replicates", type=int, default=1)
    gate_parser.add_argument("--top-k", type=int, default=4)
    gate_parser.add_argument("--max-shares-per-invocation", type=int, default=None)
    gate_parser.add_argument("--negative-risk-budget", type=float, default=0.2)
    gate_parser.add_argument("--output-dir", required=True)
    gate_parser.add_argument("--overwrite", action="store_true")
    gate_parser.add_argument("--fail-fast", action="store_true", default=True)
    gate_parser.add_argument("--bootstrap-n", type=int, default=1000)

    order_parser = subparsers.add_parser("run-order-sensitivity")
    order_parser.add_argument("--db", default="data/smtr_memory.sqlite")
    order_parser.add_argument("--critic-checkpoint", required=True)
    order_parser.add_argument("--method", choices=["SMTR"], default="SMTR")
    order_parser.add_argument(
        "--scenario-filter",
        choices=["prefix-sensitive"],
        default="prefix-sensitive",
    )
    order_parser.add_argument("--enumerate-permutations", action="store_true")
    order_parser.add_argument("--task-seeds", type=int, nargs="+", default=[0, 1])
    order_parser.add_argument("--generation-seeds", type=int, nargs="+", default=[0])
    order_parser.add_argument("--scenario-replicates", type=int, default=1)
    order_parser.add_argument("--negative-risk-budget", type=float, default=0.2)
    order_parser.add_argument("--output-dir", required=True)
    order_parser.add_argument("--overwrite", action="store_true")
    order_parser.add_argument("--fail-fast", action="store_true", default=True)

    args = parser.parse_args()
    if args.command == "seed-memories":
        _seed_memories(args.db)
    elif args.command == "list-memories":
        _list_memories(args.db)
    elif args.command == "demo":
        _demo(
            seed=args.seed,
            db=args.db,
            top_k=args.top_k,
            router_mode=args.router_mode,
            critic_checkpoint=args.critic_checkpoint,
            max_shares_per_invocation=args.max_shares_per_invocation,
        )
    elif args.command == "collect-counterfactual":
        _collect_counterfactual(
            db=args.db,
            episodes=args.episodes,
            seed=args.seed,
            top_k=args.top_k,
            scenario_mix=args.scenario_mix,
            output=args.output,
            allow_duplicates=args.allow_duplicates,
            target_policy_name=args.target_policy,
            prefix_mode=args.prefix_mode,
            max_prefix_size=args.max_prefix_size,
            continuation_policy_manifest_path=args.continuation_policy_manifest,
            boundary_critic_checkpoint=args.boundary_critic_checkpoint,
            round_id=args.round_id,
            round_index=args.round_index,
            require_min_continuation_share_rate=args.require_min_continuation_share_rate,
            require_max_continuation_share_rate=args.require_max_continuation_share_rate,
            require_max_hard_risk_share_rate=args.require_max_hard_risk_share_rate,
        )
    elif args.command == "inspect-paired-records":
        _inspect_paired_records(
            args.input,
            show_prefix_distribution=args.show_prefix_distribution,
            show_factor_coverage=args.show_factor_coverage,
            show_continuation_behavior=args.show_continuation_behavior,
        )
    elif args.command == "ingest-paired-evidence":
        _ingest_paired_evidence(args.db, args.input)
    elif args.command == "train-transfer-critic":
        _train_transfer_critic(
            input_path=args.input,
            output=args.output,
            seed=args.seed,
            n_bootstrap=args.n_bootstrap,
            test_fraction=args.test_fraction,
            require_policy_fingerprint=args.require_policy_fingerprint,
        )
    elif args.command == "evaluate-transfer-critic":
        _evaluate_transfer_critic(args.input, args.checkpoint, args.output, args.split_suite)
    elif args.command == "create-no-share-policy":
        _create_no_share_policy(args.output)
    elif args.command == "migrate-paired-policy-metadata":
        _migrate_paired_policy_metadata(
            input_path=args.input,
            output=args.output,
            round_id=args.round_id,
            round_index=args.round_index,
            policy_manifest_path=args.policy_manifest,
            base_store_revision=args.base_store_revision,
            base_memory_snapshot_digest=args.base_memory_snapshot_digest,
        )
    elif args.command == "build-critic-continuation-policy":
        _build_critic_continuation_policy(
            critic_path=args.critic,
            tau_lcb_threshold=args.tau_lcb_threshold,
            negative_risk_ucb_threshold=args.negative_risk_ucb_threshold,
            reject_low_support=args.reject_low_support,
            output=args.output,
        )
    elif args.command == "build-exploratory-continuation-policy":
        _build_exploratory_continuation_policy(
            critic_path=args.critic,
            config=ExplorationPolicyConfig(
                safe_tau_lcb_threshold=args.safe_tau_lcb_threshold,
                safe_negative_risk_ucb_threshold=args.safe_negative_risk_ucb_threshold,
                hard_negative_risk_veto_ucb=args.hard_negative_risk_veto_ucb,
                boundary_tau_band=args.boundary_tau_band,
                soft_ood_multiplier=args.soft_ood_multiplier,
                exploration_round_probability=args.exploration_round_probability,
                max_total_shares_per_invocation=args.max_total_shares_per_invocation,
                max_exploratory_shares_per_invocation=(
                    args.max_exploratory_shares_per_invocation
                ),
            ),
            output=args.output,
        )
    elif args.command == "diagnose-shortcuts":
        _diagnose_shortcuts(args.input, args.output)
    elif args.command == "validate-collection-quality":
        _validate_collection_quality(
            input_path=args.input,
            output=args.output,
            min_share_rate=args.min_continuation_share_rate,
            max_share_rate=args.max_continuation_share_rate,
            max_hard_risk_share_rate=args.max_hard_risk_share_rate,
        )
    elif args.command == "scan-transfer-feature-leakage":
        _scan_transfer_feature_leakage(args.input, args.output)
    elif args.command == "audit-feature-blocks":
        _audit_feature_blocks(args.input, args.seed, args.n_bootstrap, args.output)
    elif args.command == "audit-prefix-sensitivity":
        _audit_interaction(args.input, args.checkpoint, args.output, mode="prefix")
    elif args.command == "audit-candidate-substitution":
        _audit_interaction(args.input, args.checkpoint, args.output, mode="candidate")
    elif args.command == "inspect-marble-dataset":
        _inspect_marble_dataset(args)
    elif args.command == "serve-api":
        _serve_api(host=args.host, port=args.port, model=args.model)
    elif args.command == "demo-real":
        _demo_real(
            seed=args.seed,
            model=args.model,
            api_base=args.api_base,
            use_tool_env=args.use_tool_env,
            router_mode=args.router_mode,
            critic_checkpoint=args.critic_checkpoint,
            max_shares_per_invocation=args.max_shares_per_invocation,
        )
    elif args.command == "run-experiment":
        _compare_routers(args)
    elif args.command == "audit-experiment-integrity":
        summary = audit_experiment_integrity(
            experiment_dir=args.experiment_dir,
            m0_checkpoint=args.m0_checkpoint,
            a1_checkpoint=args.a1_checkpoint,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        names = [
            "integrity_summary",
            "candidate_invariance",
            "snapshot_invariance",
            "no_share_consistency",
            "checkpoint_compatibility",
            "persistence_roundtrip",
            "prefix_intervention_validation",
            "statistics_consistency",
        ]
        for name in names:
            (output_dir / f"{name}.json").write_text(
                json.dumps(summary, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        (output_dir / "report.md").write_text(
            "# Experiment Integrity Audit\n\n"
            f"READY_FOR_FORMAL_EXPERIMENT: {summary['READY_FOR_FORMAL_EXPERIMENT']}\n",
            encoding="utf-8",
        )
        print(f"READY_FOR_FORMAL_EXPERIMENT={summary['READY_FOR_FORMAL_EXPERIMENT']}")
    elif args.command == "run-gate-ablation":
        _run_gate_ablation(args)
    elif args.command == "run-order-sensitivity":
        _run_order_sensitivity(args)


if __name__ == "__main__":
    main()
