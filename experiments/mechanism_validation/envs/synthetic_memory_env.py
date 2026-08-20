"""Synthetic memory environment for causal benchmark (Task 7).

Generates a minimal environment where:
  - Same memory → different receivers → different effects
  - Ground truth τ(m, r) is fully known and controllable

This is used by the SyntheticCausalValidator to test whether
the model can recover receiver-specific transfer effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SyntheticEnvironment:
    """Minimal synthetic environment for causal validation.
    
    Attributes
    ----------
    n_receivers : int
        Number of synthetic receivers.
    n_memories : int
        Number of synthetic memories.
    seed : int
        Random seed for reproducibility.
    tau_matrix : np.ndarray
        Ground truth τ(m, r) matrix of shape (n_receivers, n_memories).
        Values in {-1, 0, +1}.
    """
    
    n_receivers: int
    n_memories: int
    seed: int = 7
    tau_matrix: np.ndarray | None = None
    
    def __post_init__(self) -> None:
        rng = np.random.RandomState(self.seed)
        if self.tau_matrix is None:
            self.tau_matrix = rng.choice(
                [-1, 0, 1],
                size=(self.n_receivers, self.n_memories),
                p=[0.3, 0.4, 0.3],
            )
    
    def get_effect(self, receiver_id: int, memory_id: int) -> int:
        """Get ground truth effect τ(m, r)."""
        return int(self.tau_matrix[receiver_id, memory_id])
    
    def generate_examples(
        self,
        n_per_pair: int = 5,
    ) -> tuple[list[Any], list[str], list[int]]:
        """Generate synthetic training examples.
        
        Returns
        -------
        inputs : list[CandidateExposureInput]
        labels : list[str]
        true_effects : list[int]
        """
        from smtr.core.types import (
            AgentProfile,
            CandidateExposureInput,
            MemoryRoutingCard,
            ReceiverState,
        )
        
        rng = np.random.RandomState(self.seed + 1)
        inputs = []
        labels = []
        true_effects = []
        
        for r_id in range(self.n_receivers):
            for m_id in range(self.n_memories):
                tau = self.get_effect(r_id, m_id)
                
                for _ in range(n_per_pair):
                    receiver_state = ReceiverState(
                        task_id=f"synthetic_task_{r_id}",
                        scenario=f"synthetic_scenario_{r_id}",
                        task_instruction=f"synthetic task for receiver {r_id}",
                        environment_signature=(f"env_{r_id}",),
                        receiver=AgentProfile(
                            agent_id=f"agent_{r_id}",
                            role="executor",
                            capabilities=(f"cap_{r_id}_a", f"cap_{r_id}_b"),
                            tool_names=(f"tool_{r_id}",),
                            model_name="synthetic_model",
                        ),
                    )
                    
                    memory_card = MemoryRoutingCard(
                        memory_id=f"mem_{m_id}",
                        goal_summary=f"synthetic goal for memory {m_id}",
                        task_tags=(f"tag_{m_id}",),
                        required_tools=(f"tool_{m_id}",),
                        precondition_tags=(f"pre_{m_id}",),
                        environment_constraints=(f"env_{m_id}",),
                        procedure_tags=(f"proc_{m_id}",),
                    )
                    
                    inputs.append(CandidateExposureInput(
                        receiver_state=receiver_state,
                        candidate_card=memory_card,
                    ))
                    
                    if tau > 0:
                        label = "positive_transfer"
                    elif tau < 0:
                        label = "negative_transfer"
                    else:
                        label = rng.choice([
                            "neutral_failure", "neutral_success",
                        ])
                    labels.append(label)
                    true_effects.append(tau)
        
        return inputs, labels, true_effects
