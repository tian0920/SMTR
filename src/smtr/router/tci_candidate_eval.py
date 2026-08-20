"""TCI Candidate Ranking Evaluator (Task 7).

Evaluates critic's ability to rank candidates by transfer effect.

Metrics:
  1. Spearman correlation ρ(score, effect)
  2. Top-1 Effect Hit: argmax(score) == argmax(effect)
  3. Regret: effect* - effect(selected)

This evaluates the critic's practical utility for routing decisions,
not just pairwise ranking accuracy.

Forbidden:
  - Modifying router policy
  - Modifying candidate generation
  - Score fusion
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from smtr.core.types import CandidateExposureInput
from smtr.router.transfer_critic import FourOutcomeTransferCritic


@dataclass(frozen=True)
class CandidateRankingResult:
    """Result of ranking one candidate set.

    Attributes
    ----------
    candidate_ids : list[str]
        Memory IDs in the candidate set.
    scores : list[float]
        Critic scores (tau_hat = q10 - q01) for each candidate.
    effects : list[float]
        True transfer effects for each candidate.
    selected_id : str
        Memory ID of argmax(score).
    selected_effect : float
        True effect of the selected candidate.
    best_effect : float
        Maximum true effect in the candidate set.
    regret : float
        best_effect - selected_effect (≥ 0).
    is_top1_hit : bool
        Selected candidate is argmax(effect).
    spearman_rho : float
        Rank correlation between scores and effects.
        NaN if all effects or scores are identical.
    """

    candidate_ids: list[str]
    scores: list[float]
    effects: list[float]
    selected_id: str
    selected_effect: float
    best_effect: float
    regret: float
    is_top1_hit: bool
    spearman_rho: float


@dataclass(frozen=True)
class CandidateRankingMetrics:
    """Aggregate ranking metrics over multiple candidate sets.

    Attributes
    ----------
    n_sets : int
        Number of candidate sets evaluated.
    mean_spearman : float
        Mean Spearman correlation (excluding NaN).
    top1_hit_rate : float
        P(argmax(score) == argmax(effect)).
    mean_regret : float
        Mean regret per selection.
    mean_selected_effect : float
        Mean true effect of selected candidates.
    mean_best_effect : float
        Mean best effect per candidate set.
    """

    n_sets: int
    mean_spearman: float
    top1_hit_rate: float
    mean_regret: float
    mean_selected_effect: float
    mean_best_effect: float


def evaluate_candidate_effect_ranking(
    critic: FourOutcomeTransferCritic,
    candidate_sets: list[list[CandidateExposureInput]],
    effect_sets: list[list[float]],
) -> CandidateRankingMetrics:
    """Evaluate critic's candidate ranking against true effects.

    Parameters
    ----------
    critic : fitted critic with predict() method.
    candidate_sets : list of candidate sets.
    effect_sets : list of true effect vectors (same length).

    Returns
    -------
    CandidateRankingMetrics with aggregate statistics.
    """
    if len(candidate_sets) != len(effect_sets):
        raise ValueError(
            "candidate_sets and effect_sets must have same length"
        )

    results: list[CandidateRankingResult] = []
    for cands, effs in zip(candidate_sets, effect_sets):
        result = _rank_one_set(critic, cands, effs)
        if result is not None:
            results.append(result)

    if not results:
        return CandidateRankingMetrics(
            n_sets=0,
            mean_spearman=0.0,
            top1_hit_rate=0.0,
            mean_regret=0.0,
            mean_selected_effect=0.0,
            mean_best_effect=0.0,
        )

    n = len(results)
    top1_hits = sum(1 for r in results if r.is_top1_hit)
    regrets = [r.regret for r in results]
    sel_effects = [r.selected_effect for r in results]
    best_effects = [r.best_effect for r in results]

    # Spearman: exclude NaN (all-identical cases).
    valid_rhos = [
        r.spearman_rho for r in results if not np.isnan(r.spearman_rho)
    ]
    mean_rho = float(np.mean(valid_rhos)) if valid_rhos else 0.0

    return CandidateRankingMetrics(
        n_sets=n,
        mean_spearman=mean_rho,
        top1_hit_rate=top1_hits / n,
        mean_regret=float(np.mean(regrets)),
        mean_selected_effect=float(np.mean(sel_effects)),
        mean_best_effect=float(np.mean(best_effects)),
    )


def _rank_one_set(
    critic: FourOutcomeTransferCritic,
    candidates: list[CandidateExposureInput],
    effects: list[float],
) -> CandidateRankingResult | None:
    """Rank one candidate set and compute metrics.

    Returns None if candidates is empty.
    """
    if not candidates:
        return None

    scores = []
    for cand in candidates:
        pred = critic.predict(cand)
        score = pred.q10_positive_transfer - pred.q01_negative_transfer
        scores.append(score)

    scores_arr = np.array(scores)
    effects_arr = np.array(effects, dtype=float)

    selected_idx = int(np.argmax(scores_arr))
    best_idx = int(np.argmax(effects_arr))

    selected_effect = float(effects_arr[selected_idx])
    best_effect = float(effects_arr[best_idx])
    regret = max(0.0, best_effect - selected_effect)

    # Spearman correlation.
    if len(np.unique(scores_arr)) < 2 or len(np.unique(effects_arr)) < 2:
        rho = float('nan')
    else:
        rho, _ = stats.spearmanr(scores_arr, effects_arr)

    candidate_ids = [
        c.candidate_card.memory_id for c in candidates
    ]

    return CandidateRankingResult(
        candidate_ids=candidate_ids,
        scores=scores,
        effects=effects,
        selected_id=candidate_ids[selected_idx],
        selected_effect=selected_effect,
        best_effect=best_effect,
        regret=regret,
        is_top1_hit=(selected_idx == best_idx),
        spearman_rho=float(rho),
    )
