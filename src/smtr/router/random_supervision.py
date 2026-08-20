"""Random supervision baseline (Codex Task 9).

Generates random memory pairs to test whether TCI augmentation's
effectiveness comes from intervention-specific supervision or
merely from adding more training examples.

The random baseline:
  - Selects pairs of (memory, perturbed_memory) at random.
  - Assigns random directions (+1 or -1 with equal probability).
  - Trains the critic with the same number of augmented examples
    as the TCI condition.

If TCI augmentation outperforms random augmentation, the improvement
is attributable to intervention-specific supervision rather than
just "more data".

Forbidden:
  - Modifying candidate generation.
  - Using real TCI labels in random augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
)
from smtr.router.tci_augmentation import (
    SOURCE_OBSERVATIONAL,
    TCIAugmentedBatch,
)
from smtr.router.transfer_features import (
    build_routing_card_from_pool_entry,
)


# Source type label for random augmentation provenance.
SOURCE_RANDOM_AUGMENTATION: str = "random_augmentation"


def build_random_supervision_pairs(
    *,
    memory_pool: dict[str, dict],
    paired_records: list[dict[str, Any]],
    n_pairs: int,
    seed: int = 7,
) -> list[tuple[CandidateExposureInput,
                CandidateExposureInput,
                int,
                str]]:
    """Generate random memory pairs with random directions.

    Parameters
    ----------
    memory_pool : dict of memory_id → memory entry.
    paired_records : list of paired records used to extract
        receiver contexts.
    n_pairs : number of random pairs to generate.
    seed : random seed for reproducibility.

    Returns
    -------
    List of (input_orig, input_pert, direction, contrast_type)
    where contrast_type="random" and direction is randomly ±1.
    """
    rng = np.random.default_rng(seed)

    # Build receiver context pool from paired records.
    contexts: list[dict[str, Any]] = []
    for rec in paired_records:
        if rec.get("task_id") and rec.get("receiver_agent_id"):
            contexts.append(rec)
    if not contexts:
        return []

    # Build memory card pool.
    cards: list[MemoryRoutingCard] = []
    for mem_id, mem in memory_pool.items():
        try:
            cards.append(build_routing_card_from_pool_entry(mem))
        except Exception:
            continue
    if len(cards) < 2:
        return []

    pairs: list[tuple[CandidateExposureInput,
                      CandidateExposureInput,
                      int,
                      str]] = []

    for _ in range(n_pairs):
        # Random context.
        ctx = contexts[rng.integers(len(contexts))]
        receiver = AgentProfile(
            agent_id=ctx.get("receiver_agent_id", ""),
            role=ctx.get("receiver_role", "unknown"),
            capabilities=tuple(ctx.get("receiver_capabilities", [])),
            model_name=ctx.get("receiver_model_name"),
            tool_names=tuple(ctx.get("receiver_tool_names", [])),
        )
        state = ReceiverState(
            task_id=ctx["task_id"],
            scenario=ctx.get("scenario", "database"),
            task_instruction=ctx.get("task_instruction", ""),
            receiver=receiver,
            subtask=ctx.get("subtask"),
            environment_signature=tuple(
                ctx.get("environment_signature", [])
            ),
            local_context_summary=ctx.get("local_context_summary", ""),
            team_context_summary=ctx.get("team_context_summary", ""),
        )

        # Random pair of distinct memory cards.
        i1, i2 = rng.choice(len(cards), size=2, replace=False)
        inp1 = CandidateExposureInput(
            receiver_state=state, candidate_card=cards[i1]
        )
        inp2 = CandidateExposureInput(
            receiver_state=state, candidate_card=cards[i2]
        )

        # Random direction.
        direction = int(rng.choice([-1, 1]))

        pairs.append((inp1, inp2, direction, "random"))

    return pairs


def build_random_augmentation_examples(
    random_pairs: list[tuple[CandidateExposureInput,
                              CandidateExposureInput,
                              int,
                              str]],
) -> TCIAugmentedBatch:
    """Convert random pairs to augmentation examples (same interface
    as TCI augmentation but with source_type="random_augmentation").

    Each pair produces two examples (positive_transfer + negative_transfer)
    just like TCI augmentation, so the total example count matches.
    """
    inputs: list[CandidateExposureInput] = []
    labels: list[str] = []
    source_types: list[str] = []

    for (inp1, inp2, direction, _) in random_pairs:
        if direction > 0:
            inputs.append(inp1)
            labels.append("positive_transfer")
            source_types.append(SOURCE_RANDOM_AUGMENTATION)
            inputs.append(inp2)
            labels.append("negative_transfer")
            source_types.append(SOURCE_RANDOM_AUGMENTATION)
        else:
            inputs.append(inp2)
            labels.append("positive_transfer")
            source_types.append(SOURCE_RANDOM_AUGMENTATION)
            inputs.append(inp1)
            labels.append("negative_transfer")
            source_types.append(SOURCE_RANDOM_AUGMENTATION)

    return TCIAugmentedBatch(
        inputs=inputs,
        labels=labels,
        source_types=source_types,
    )
