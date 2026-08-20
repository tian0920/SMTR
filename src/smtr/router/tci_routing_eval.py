"""TCI routing evaluation adapter (Codex Task 5/6).

Converts critic scores into routing-level evaluation metrics:
  - Positive Transfer Capture (PTC): P(selected_effect > 0)
  - Negative Transfer Exposure (NTE): P(selected_effect < 0)
  - Transfer Regret: R = effect* - effect(selected)
  - Top-1 Transfer Hit Rate: argmax(score) == argmax(effect)

This module bridges the critic's per-candidate prediction to
routing-level evaluation: given a candidate set {m1, ..., mk},
the router selects m* = argmax score(m). The metrics measure
how well this selection aligns with the true transfer effect
Y_m - Y_0.

Forbidden:
  - Modifying candidate generation.
  - Modifying router policy.
  - Score fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from smtr.core.types import (
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic


@dataclass(frozen=True)
class RoutingSelection:
    """Result of routing one candidate set.

    Attributes
    ----------
    selected_memory_id : str
        Memory ID selected by argmax(score).
    selected_score : float
        Critic tau_hat = q10 - q01 for selected memory.
    selected_effect : float
        True transfer effect Y_m - Y_0 for selected memory.
    best_effect : float
        Maximum true transfer effect in the candidate set.
    regret : float
        best_effect - selected_effect (≥ 0).
    is_positive_transfer : bool
        selected_effect > 0.
    is_negative_transfer : bool
        selected_effect < 0.
    is_top1_hit : bool
        Selected memory is the argmax of true effect.
    """

    selected_memory_id: str
    selected_score: float
    selected_effect: float
    best_effect: float
    regret: float
    is_positive_transfer: bool
    is_negative_transfer: bool
    is_top1_hit: bool


@dataclass(frozen=True)
class RoutingMetrics:
    """Aggregate routing metrics over a set of selections.

    Attributes
    ----------
    n_selections : int
        Number of routing decisions evaluated.
    positive_capture : float
        P(selected_effect > 0).
    negative_exposure : float
        P(selected_effect < 0).
    transfer_regret : float
        Mean regret per selection.
    top1_hit_rate : float
        P(argmax(score) == argmax(effect)).
    mean_selected_effect : float
        Mean true transfer effect for selected memories.
    mean_best_effect : float
        Mean best true effect per candidate set.
    """

    n_selections: int
    positive_capture: float
    negative_exposure: float
    transfer_regret: float
    top1_hit_rate: float
    mean_selected_effect: float
    mean_best_effect: float


def score_candidate(critic: FourOutcomeTransferCritic,
                    inp: CandidateExposureInput) -> float:
    """Compute critic score (tau_hat = q10 - q01) for one candidate."""
    pred = critic.predict(inp)
    return pred.q10_positive_transfer - pred.q01_negative_transfer


def select_best_candidate(
    critic: FourOutcomeTransferCritic,
    candidates: list[CandidateExposureInput],
    effects: list[float],
) -> RoutingSelection:
    """Route one candidate set: select argmax(score) and compute metrics.

    Parameters
    ----------
    critic : fitted critic.
    candidates : list of CandidateExposureInput (one per candidate memory).
    effects : list of true transfer effects Y_m - Y_0, same order as
        candidates.

    Returns
    -------
    RoutingSelection with selected/best effect, regret, hit indicator.

    Raises
    ------
    ValueError if candidates is empty.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    if len(candidates) != len(effects):
        raise ValueError("candidates and effects must have same length")

    scores = np.array([score_candidate(critic, c) for c in candidates])
    effects_arr = np.array(effects, dtype=float)
    selected_idx = int(np.argmax(scores))
    best_idx = int(np.argmax(effects_arr))

    selected_effect = float(effects_arr[selected_idx])
    best_effect = float(effects_arr[best_idx])
    regret = best_effect - selected_effect

    return RoutingSelection(
        selected_memory_id=(
            candidates[selected_idx].candidate_card.memory_id
        ),
        selected_score=float(scores[selected_idx]),
        selected_effect=selected_effect,
        best_effect=best_effect,
        regret=max(0.0, regret),
        is_positive_transfer=selected_effect > 0,
        is_negative_transfer=selected_effect < 0,
        is_top1_hit=(selected_idx == best_idx),
    )


def evaluate_memory_selection(
    candidates: list[list[CandidateExposureInput]],
    effects: list[list[float]],
    critic: FourOutcomeTransferCritic,
) -> RoutingMetrics:
    """Evaluate routing decisions over multiple candidate sets.

    Parameters
    ----------
    candidates : list of candidate sets (each a list of
        CandidateExposureInput).
    effects : list of effect vectors (same length as candidates).
    critic : fitted critic.

    Returns
    -------
    RoutingMetrics with aggregate PTC, NTE, regret, top-1 hit rate.
    """
    if len(candidates) != len(effects):
        raise ValueError("candidates and effects must have same length")

    selections: list[RoutingSelection] = []
    for cands, effs in zip(candidates, effects):
        if not cands:
            continue
        selections.append(select_best_candidate(critic, cands, effs))

    if not selections:
        return RoutingMetrics(
            n_selections=0,
            positive_capture=0.0,
            negative_exposure=0.0,
            transfer_regret=0.0,
            top1_hit_rate=0.0,
            mean_selected_effect=0.0,
            mean_best_effect=0.0,
        )

    n = len(selections)
    pos = sum(1 for s in selections if s.is_positive_transfer)
    neg = sum(1 for s in selections if s.is_negative_transfer)
    top1 = sum(1 for s in selections if s.is_top1_hit)
    regrets = [s.regret for s in selections]
    sel_effects = [s.selected_effect for s in selections]
    best_effects = [s.best_effect for s in selections]

    return RoutingMetrics(
        n_selections=n,
        positive_capture=pos / n,
        negative_exposure=neg / n,
        transfer_regret=float(np.mean(regrets)),
        top1_hit_rate=top1 / n,
        mean_selected_effect=float(np.mean(sel_effects)),
        mean_best_effect=float(np.mean(best_effects)),
    )


def compute_routing_metrics_from_paired_records(
    critic: FourOutcomeTransferCritic,
    records: list[dict[str, Any]],
    memory_pool: dict[str, dict],
    *,
    build_input_fn: Any = None,
) -> RoutingMetrics:
    """Compute routing metrics from paired records (observational data).

    Groups records by (task_id, receiver_agent_id), treating each group
    as one candidate set. True effect per candidate is defined as
    Y_share - Y_withhold (from the share/withhold outcomes).

    Parameters
    ----------
    critic : fitted critic.
    records : list of paired record dicts.
    memory_pool : memory pool dict keyed by memory_id.
    build_input_fn : optional callable(record, memory_pool) →
        CandidateExposureInput. If None, uses
        ``transfer_features.build_training_data_from_records``.

    Returns
    -------
    RoutingMetrics.
    """
    from collections import defaultdict

    from smtr.core.types import AgentProfile, ReceiverState
    from smtr.router.transfer_features import (
        build_routing_card_from_pool_entry,
    )

    def _default_build_input(rec: dict, pool: dict) -> CandidateExposureInput:
        mem = pool.get(rec["candidate_memory_id"])
        if mem is None:
            return None
        card = build_routing_card_from_pool_entry(mem)
        receiver = AgentProfile(
            agent_id=rec.get("receiver_agent_id", ""),
            role=rec.get("receiver_role", "unknown"),
            capabilities=tuple(rec.get("receiver_capabilities", [])),
            model_name=rec.get("receiver_model_name"),
            tool_names=tuple(rec.get("receiver_tool_names", [])),
        )
        state = ReceiverState(
            task_id=rec["task_id"],
            scenario=rec.get("scenario", "database"),
            task_instruction=rec.get("task_instruction", ""),
            receiver=receiver,
            subtask=rec.get("subtask"),
            environment_signature=tuple(rec.get("environment_signature", [])),
            local_context_summary=rec.get("local_context_summary", ""),
            team_context_summary=rec.get("team_context_summary", ""),
        )
        return CandidateExposureInput(
            receiver_state=state, candidate_card=card
        )

    builder = build_input_fn or _default_build_input

    # Group by (task_id, receiver_agent_id).
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        key = (str(rec.get("task_id", "")), rec.get("receiver_agent_id", ""))
        if key[0] and key[1]:
            groups[key].append(rec)

    all_candidates: list[list[CandidateExposureInput]] = []
    all_effects: list[list[float]] = []

    for key, group_recs in groups.items():
        cands: list[CandidateExposureInput] = []
        effs: list[float] = []
        for rec in group_recs:
            inp = builder(rec, memory_pool)
            if inp is None:
                continue
            # Effect = Y_share - Y_withhold (0 or 1 or -1).
            y_share = int(rec.get("share", {}).get("team_success", 0))
            y_withhold = int(rec.get("withhold", {}).get("team_success", 0))
            effect = y_share - y_withhold
            cands.append(inp)
            effs.append(float(effect))
        if cands:
            all_candidates.append(cands)
            all_effects.append(effs)

    return evaluate_memory_selection(all_candidates, all_effects, critic)
