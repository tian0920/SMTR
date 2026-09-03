"""Experiment configuration for formal pilot (§18).

Defines the six method variants for the formal pilot experiment:

1. ``rima_receiver`` — Static RIMA (no transfer state)
2. ``rima_transfer_frozen`` — Frozen transfer cache (no causal probe, no critic update)
3. ``rima_transfer_adaptive`` — Full adaptive continual transfer
4. ``rima_transfer_positive_stop`` — Ablation: stop exploration when bestLCB > delta
5. ``rima_transfer_no_uncertainty`` — Ablation: use mu instead of LCB
6. ``rima_static_same_probe_budget`` — Cost-matched baseline

Also defines the pilot protocol (Phase A smoke, Phase B mechanism pilot)
and the six main report metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "MethodVariant",
    "PilotProtocol",
    "PilotReportMetrics",
    "RIMA_RECEIVER",
    "RIMA_TRANSFER_FROZEN",
    "RIMA_TRANSFER_ADAPTIVE",
    "RIMA_TRANSFER_POSITIVE_STOP",
    "RIMA_TRANSFER_NO_UNCERTAINTY",
    "RIMA_STATIC_SAME_PROBE_BUDGET",
    "ALL_METHOD_VARIANTS",
    "get_method_variant",
    "build_smoke_protocol",
    "build_pilot_protocol",
    "compute_pilot_report_metrics",
    "compute_early_late_scores",
]


# ---------------------------------------------------------------------------
# Method variant definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodVariant:
    """Configuration for one experiment condition.

    Each variant is a frozen specification that controls:
    - Whether transfer state is maintained
    - Whether global retrieval is conditional or unconditional
    - Whether causal probing is active
    - Whether the critic is updated online
    - Routing thresholds (beta, delta, gamma)

    Attributes:
        method_id: unique identifier string.
        display_label: human-readable label for tables/figures.
        use_transfer_state: whether to maintain persistent transfer state.
        conditional_global_retrieval: if True, skip global retrieval when
            best known LCB >= gamma (EXPLOIT_ONLY). If False, always retrieve.
        use_causal_probe: whether to run post-task causal probing.
        use_critic_update: whether to refit critic from online evidence.
        use_uncertainty: if True, score = LCB = mu - beta*sigma.
            If False, score = mu (no uncertainty penalty).
        beta: uncertainty coefficient (fixed at 1.64 per §16.1).
        delta: minimum LCB threshold for exploitation (fixed at 0.0).
        gamma_mode: how gamma is determined. "train_q75" = Q75 of positive
            train tau (standard); "positive_stop" = EXPLOIT_ONLY when
            bestLCB > delta (ablation).
        is_baseline: whether this is a baseline condition (not the main method).
        description: one-line description of what this variant tests.
    """

    method_id: str
    display_label: str

    use_transfer_state: bool
    conditional_global_retrieval: bool
    use_causal_probe: bool
    use_critic_update: bool
    use_uncertainty: bool

    beta: float = 1.64
    delta: float = 0.0
    gamma_mode: str = "train_q75"

    is_baseline: bool = True
    description: str = ""


# ---------------------------------------------------------------------------
# Six method variants (§18)
# ---------------------------------------------------------------------------

#: Baseline 1: Static RIMA — no transfer state, always global retrieval.
RIMA_RECEIVER = MethodVariant(
    method_id="rima_receiver",
    display_label="Static RIMA",
    use_transfer_state=False,
    conditional_global_retrieval=False,
    use_causal_probe=False,
    use_critic_update=False,
    use_uncertainty=False,
    is_baseline=True,
    description="No transfer state; every task does global retrieve + critic.",
)

#: Baseline 2: Frozen transfer cache — persistent state, no learning.
RIMA_TRANSFER_FROZEN = MethodVariant(
    method_id="rima_transfer_frozen",
    display_label="Frozen Transfer",
    use_transfer_state=True,
    conditional_global_retrieval=True,
    use_causal_probe=False,
    use_critic_update=False,
    use_uncertainty=True,
    is_baseline=True,
    description="Persistent candidate state + conditional global; no causal probe or critic update.",
)

#: Main: Adaptive continual transfer — full system.
RIMA_TRANSFER_ADAPTIVE = MethodVariant(
    method_id="rima_transfer_adaptive",
    display_label="Adaptive Continual",
    use_transfer_state=True,
    conditional_global_retrieval=True,
    use_causal_probe=True,
    use_critic_update=True,
    use_uncertainty=True,
    is_baseline=False,
    description="Persistent state + post-task causal probe + forward-only critic update.",
)

#: Ablation A: Positive stop — EXPLOIT_ONLY when bestLCB > delta.
RIMA_TRANSFER_POSITIVE_STOP = MethodVariant(
    method_id="rima_transfer_positive_stop",
    display_label="Positive Stop",
    use_transfer_state=True,
    conditional_global_retrieval=True,
    use_causal_probe=True,
    use_critic_update=True,
    use_uncertainty=True,
    gamma_mode="positive_stop",
    is_baseline=True,
    description="Ablation: stop exploration when bestLCB > delta (vs Q75 gamma).",
)

#: Ablation B: No uncertainty — use mu instead of LCB.
RIMA_TRANSFER_NO_UNCERTAINTY = MethodVariant(
    method_id="rima_transfer_no_uncertainty",
    display_label="No Uncertainty",
    use_transfer_state=True,
    conditional_global_retrieval=True,
    use_causal_probe=True,
    use_critic_update=True,
    use_uncertainty=False,
    is_baseline=True,
    description="Ablation: score = mu instead of LCB = mu - beta*sigma.",
)

#: Cost-matched baseline: same probe budget, no transfer state.
RIMA_STATIC_SAME_PROBE_BUDGET = MethodVariant(
    method_id="rima_static_same_probe_budget",
    display_label="Static + Probe Budget",
    use_transfer_state=False,
    conditional_global_retrieval=False,
    use_causal_probe=True,
    use_critic_update=False,
    use_uncertainty=False,
    is_baseline=True,
    description=(
        "Cost-matched: receives same probe episodes as adaptive, "
        "but no transfer state and no critic update."
    ),
)

#: All six method variants in canonical order.
ALL_METHOD_VARIANTS: dict[str, MethodVariant] = {
    v.method_id: v
    for v in [
        RIMA_RECEIVER,
        RIMA_TRANSFER_FROZEN,
        RIMA_TRANSFER_ADAPTIVE,
        RIMA_TRANSFER_POSITIVE_STOP,
        RIMA_TRANSFER_NO_UNCERTAINTY,
        RIMA_STATIC_SAME_PROBE_BUDGET,
    ]
}


def get_method_variant(method_id: str) -> MethodVariant:
    """Look up a method variant by ID.

    Raises:
        ValueError: if method_id is not in the registry.
    """
    if method_id not in ALL_METHOD_VARIANTS:
        raise ValueError(
            f"Unknown method_id={method_id!r}. "
            f"Available: {sorted(ALL_METHOD_VARIANTS)}"
        )
    return ALL_METHOD_VARIANTS[method_id]


# ---------------------------------------------------------------------------
# Pilot protocol (§18 Phase A / Phase B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PilotProtocol:
    """Formal pilot protocol configuration.

    Phase A (smoke):
        1 scenario, 10-15 tasks, 1 stream seed, 1 probe seed.
        Checks: no crash, no leakage, no joint treatment,
        forward-only update, K state growth.

    Phase B (mechanism pilot):
        2+ scenarios, 30+ tasks/stream, 3 task-order seeds, 3 execution seeds.
        Observes: GlobalExplorationRate(t), KnownReuseRate(t),
        CausalEvidenceSize(t), TaskScore(t).

    Attributes:
        phase: "A" (smoke) or "B" (mechanism pilot).
        n_scenarios: number of scenarios.
        n_tasks_per_stream: tasks per stream.
        stream_seeds: frozen task-order seeds.
        execution_seeds: MARBLE execution seeds.
        probe_generation_seeds: seeds for post-task probe runs.
    """

    phase: str
    n_scenarios: int
    n_tasks_per_stream: int
    stream_seeds: tuple[int, ...]
    execution_seeds: tuple[int, ...]
    probe_generation_seeds: tuple[int, ...]


def build_smoke_protocol() -> PilotProtocol:
    """Build Phase A smoke protocol."""
    return PilotProtocol(
        phase="A",
        n_scenarios=1,
        n_tasks_per_stream=15,
        stream_seeds=(0,),
        execution_seeds=(0,),
        probe_generation_seeds=(0,),
    )


def build_pilot_protocol(
    *,
    n_scenarios: int = 2,
    n_tasks_per_stream: int = 30,
    n_stream_seeds: int = 3,
    n_execution_seeds: int = 3,
) -> PilotProtocol:
    """Build Phase B mechanism pilot protocol."""
    return PilotProtocol(
        phase="B",
        n_scenarios=n_scenarios,
        n_tasks_per_stream=n_tasks_per_stream,
        stream_seeds=tuple(range(n_stream_seeds)),
        execution_seeds=tuple(range(n_execution_seeds)),
        probe_generation_seeds=(0,),
    )


# ---------------------------------------------------------------------------
# Pilot report metrics (§18 main metrics)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PilotReportMetrics:
    """Six main report metrics for the formal pilot.

    These are the metric *definitions*; actual computation uses
    routing diagnostics and task score curves.

    Attributes:
        mean_team_task_score: average official task score across stream.
        global_retrieval_calls_per_task: amortized global retrieval cost.
        known_transfer_selection_rate: fraction of tasks selecting from known state.
        extra_causal_probe_episodes_per_task: amortized learning cost.
        critic_calls_per_task: amortized model cost.
        continual_gain_delta_score_late: Score_late(adaptive) - Score_late(baseline).
    """

    mean_team_task_score: float = 0.0
    global_retrieval_calls_per_task: float = 0.0
    known_transfer_selection_rate: float = 0.0
    extra_causal_probe_episodes_per_task: float = 0.0
    critic_calls_per_task: float = 0.0
    continual_gain_delta_score_late: float = 0.0


def compute_pilot_report_metrics(
    *,
    task_scores: list[float],
    n_tasks: int,
    n_global_retrieval_calls: int = 0,
    n_known_selections: int = 0,
    n_causal_probe_episodes: int = 0,
    n_critic_calls: int = 0,
    baseline_late_score: float | None = None,
) -> PilotReportMetrics:
    """Compute the six main report metrics.

    Args:
        task_scores: official task scores in stream order.
        n_tasks: total number of tasks.
        n_global_retrieval_calls: total global retrieval calls.
        n_known_selections: total tasks that selected from known state.
        n_causal_probe_episodes: total extra probe episodes.
        n_critic_calls: total critic prediction calls.
        baseline_late_score: baseline's Score_late for continual gain.

    Returns:
        PilotReportMetrics with all six values.
    """
    if n_tasks == 0:
        return PilotReportMetrics()

    mean_score = sum(task_scores) / len(task_scores) if task_scores else 0.0

    # Continual gain: Score_late = mean(t > 2T/3)
    t = len(task_scores)
    late_start = max(0, int(2 * t / 3))
    late_scores = task_scores[late_start:] if late_start < t else []
    adaptive_late = sum(late_scores) / len(late_scores) if late_scores else 0.0

    delta_late = 0.0
    if baseline_late_score is not None:
        delta_late = adaptive_late - baseline_late_score

    return PilotReportMetrics(
        mean_team_task_score=mean_score,
        global_retrieval_calls_per_task=n_global_retrieval_calls / n_tasks,
        known_transfer_selection_rate=n_known_selections / n_tasks,
        extra_causal_probe_episodes_per_task=n_causal_probe_episodes / n_tasks,
        critic_calls_per_task=n_critic_calls / n_tasks,
        continual_gain_delta_score_late=delta_late,
    )


def compute_early_late_scores(
    task_scores: list[float],
) -> dict[str, float]:
    """Compute Score_early = mean(t <= T/3) and Score_late = mean(t > 2T/3).

    Returns:
        Dict with keys: score_early, score_late, delta_score.
    """
    t = len(task_scores)
    if t == 0:
        return {"score_early": 0.0, "score_late": 0.0, "delta_score": 0.0}

    early_end = max(1, int(t / 3))
    late_start = max(early_end, int(2 * t / 3))

    early = task_scores[:early_end]
    late = task_scores[late_start:] if late_start < t else []

    score_early = sum(early) / len(early) if early else 0.0
    score_late = sum(late) / len(late) if late else 0.0

    return {
        "score_early": score_early,
        "score_late": score_late,
        "delta_score": score_late - score_early,
    }
