"""TCI Synthetic Candidate Evaluation (Task 8).

Isolates domain mismatch by constructing candidate sets directly from
intervention contrasts:
  [original_memory, perturbed_memory, random_memory]

Tests whether the critic correctly selects argmax(effect) from this
controlled set where the critic was trained on these exact examples.

This addresses the question: "Does the critic fail on routing because
of domain mismatch (observational vs intervention candidates) or because
it doesn't learn the ranking signal?"

Forbidden:
  - Modifying router policy
  - Modifying candidate generation
  - Score fusion
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    ReceiverState,
)
from smtr.intervention.intervention_contrast import InterventionContrast
from smtr.router.tci_candidate_eval import (
    CandidateRankingMetrics,
    evaluate_candidate_effect_ranking,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import (
    build_routing_card_from_pool_entry,
)


@dataclass(frozen=True)
class SyntheticEvalResult:
    """Result of synthetic candidate evaluation.

    Attributes
    ----------
    n_contrasts : int
        Number of intervention contrasts evaluated.
    top1_hit_rate : float
        P(critic selects argmax effect from [orig, pert, random]).
    mean_regret : float
        Mean regret per contrast.
    correct_selections : int
        Number of contrasts where critic selected best candidate.
    """

    n_contrasts: int
    top1_hit_rate: float
    mean_regret: float
    correct_selections: int


def evaluate_synthetic_candidates(
    critic: FourOutcomeTransferCritic,
    contrasts: list[InterventionContrast],
    tci_inputs: list[tuple[
        CandidateExposureInput,
        CandidateExposureInput,
        int,
        str,
    ]],
    memory_pool: dict[str, dict],
    paired_records: list[dict[str, Any]],
    *,
    include_random: bool = True,
    seed: int = 7,
) -> SyntheticEvalResult:
    """Evaluate critic on synthetic candidate sets from contrasts.

    For each contrast, constructs a candidate set:
      [original_memory, perturbed_memory, random_memory]

    with known effects:
      [effect_orig, effect_pert, effect_random]

    where effect_random is sampled from {-1, 0, +1} uniformly.

    Parameters
    ----------
    critic : fitted critic.
    contrasts : list of InterventionContrast.
    tci_inputs : pre-built CandidateExposureInput pairs (same order).
    memory_pool : memory pool dict.
    paired_records : paired records for receiver context.
    include_random : if True, adds a random memory to each set.
    seed : random seed for reproducibility.

    Returns
    -------
    SyntheticEvalResult with hit rate and regret.
    """
    if len(contrasts) != len(tci_inputs):
        raise ValueError(
            f"contrasts ({len(contrasts)}) must match "
            f"tci_inputs ({len(tci_inputs)})"
        )

    rng = np.random.default_rng(seed)

    candidate_sets: list[list[CandidateExposureInput]] = []
    effect_sets: list[list[float]] = []

    for contrast, (inp_orig, inp_pert, direction, ct) in zip(
        contrasts, tci_inputs
    ):
        effect_orig = contrast.y_original - contrast.y0
        effect_pert = contrast.y_perturbed - contrast.y0

        cands = [inp_orig, inp_pert]
        effs = [float(effect_orig), float(effect_pert)]

        if include_random:
            # Add a random memory from the pool.
            mem_ids = list(memory_pool.keys())
            if mem_ids:
                random_mem_id = rng.choice(mem_ids)
                mem_entry = memory_pool[random_mem_id]
                try:
                    random_card = build_routing_card_from_pool_entry(
                        mem_entry
                    )
                    random_inp = CandidateExposureInput(
                        receiver_state=inp_orig.receiver_state,
                        candidate_card=random_card,
                    )
                    # Random effect from {-1, 0, +1}.
                    random_effect = float(rng.choice([-1, 0, 1]))
                    cands.append(random_inp)
                    effs.append(random_effect)
                except Exception:
                    # Skip if card construction fails.
                    pass

        candidate_sets.append(cands)
        effect_sets.append(effs)

    # Use the standard ranking evaluator.
    metrics = evaluate_candidate_effect_ranking(
        critic, candidate_sets, effect_sets
    )

    correct = int(round(metrics.top1_hit_rate * metrics.n_sets))

    return SyntheticEvalResult(
        n_contrasts=metrics.n_sets,
        top1_hit_rate=metrics.top1_hit_rate,
        mean_regret=metrics.mean_regret,
        correct_selections=correct,
    )
