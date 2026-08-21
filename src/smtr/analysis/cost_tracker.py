"""TCI cost tracking infrastructure.

Records the computational cost of TCI validation without affecting
decision logic. Default disabled; enable via config.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class TCICostTracker:
    """Tracks TCI intervention and validation costs.
    
    Records:
      - intervention_count: number of expose/withhold pairs
      - validation_rollouts: actual probe executions
      - candidate_memories_checked: total candidates evaluated
      - validated_memories: candidates that passed TCI gate
      - rejected_memories: candidates that failed TCI gate
      - total_agent_steps: episode counter
      - wall_clock_time: elapsed time since start
    
    Usage:
      tracker = TCICostTracker(enabled=True)
      tracker.record_intervention(memory_id="m1", expose_reward=1.0, withhold_reward=0.5)
      tracker.record_memory_decision(memory_id="m1", delta=0.5, decision="validated")
      tracker.save("cost_history.jsonl")
    """
    
    enabled: bool = False
    intervention_count: int = 0
    validation_rollouts: int = 0
    candidate_memories_checked: int = 0
    validated_memories: int = 0
    rejected_memories: int = 0
    total_agent_steps: int = 0
    
    _history: list[dict] = field(default_factory=list)
    _start_time: float = field(default_factory=time.time)
    
    def record_intervention(
        self,
        memory_id: str,
        expose_reward: float,
        withhold_reward: float,
        episode: int = -1,
    ) -> None:
        """Record one expose/withhold intervention pair."""
        if not self.enabled:
            return
        self.intervention_count += 1
        self.validation_rollouts += 2  # expose + withhold
        self._history.append({
            "type": "intervention",
            "timestamp": datetime.now().isoformat(),
            "episode": episode,
            "memory_id": memory_id,
            "expose_reward": expose_reward,
            "withhold_reward": withhold_reward,
            "delta": expose_reward - withhold_reward,
        })
    
    def record_validation(
        self,
        memory_id: str,
        delta: float,
        decision: str,
        episode: int = -1,
    ) -> None:
        """Record one memory validation decision."""
        if not self.enabled:
            return
        self.candidate_memories_checked += 1
        if decision == "validated":
            self.validated_memories += 1
        elif decision == "rejected":
            self.rejected_memories += 1
        self._history.append({
            "type": "validation",
            "timestamp": datetime.now().isoformat(),
            "episode": episode,
            "memory_id": memory_id,
            "delta": delta,
            "decision": decision,
        })
    
    def record_memory_decision(
        self,
        memory_id: str,
        delta: float,
        decision: str,
        episode: int = -1,
    ) -> None:
        """Alias for record_validation (backward compatibility)."""
        self.record_validation(memory_id, delta, decision, episode)
    
    def step(self) -> None:
        """Increment agent step counter."""
        if not self.enabled:
            return
        self.total_agent_steps += 1
    
    def summary(self) -> dict:
        """Return cost summary statistics."""
        elapsed = time.time() - self._start_time
        return {
            "intervention_count": self.intervention_count,
            "validation_rollouts": self.validation_rollouts,
            "candidate_memories_checked": self.candidate_memories_checked,
            "validated_memories": self.validated_memories,
            "rejected_memories": self.rejected_memories,
            "total_agent_steps": self.total_agent_steps,
            "wall_clock_seconds": round(elapsed, 2),
            "validation_rate": (
                self.validated_memories / self.candidate_memories_checked
                if self.candidate_memories_checked > 0 else 0.0
            ),
        }
    
    def save(self, path: Path | str) -> None:
        """Save cost history to JSONL."""
        if not self.enabled:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in self._history:
                f.write(json.dumps(record) + "\n")
    
    def reset(self) -> None:
        """Reset all counters."""
        self.intervention_count = 0
        self.validation_rollouts = 0
        self.candidate_memories_checked = 0
        self.validated_memories = 0
        self.rejected_memories = 0
        self.total_agent_steps = 0
        self._history.clear()
        self._start_time = time.time()
