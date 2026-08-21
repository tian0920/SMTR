"""Synthetic lifelong environment for long-term memory lifecycle experiments.

One task = one episode. Each task belongs to a skill topic. Success
probability depends on the memories injected into the agent:

    p(success) = clip(base + sum(effects of injected memories), 0.02, 0.98)

Ground truth is fully known: a memory extracted from topic t helps future
tasks of topic t (+helpful_effect), harms them (false/outdated memory),
or does nothing (spurious / off-topic). Off-topic injected memories add a
small distraction cost, so hoarding everything is not free.

The environment is pure numpy and needs no LLM / engine, so 100-episode
runs with 5 seeds complete in seconds while preserving the causal
share/withhold structure of SMTR (TCI probes are honest stochastic
expose-vs-withhold trials on fresh task instances).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

N_TOPICS = 10
BASE_SUCCESS = 0.40
HELPFUL_EFFECT = 0.35
HARMFUL_EFFECT = -0.35
DISTRACTION_PENALTY = 0.02
EFFECT_CLAMP = 0.60
# Fixed validation protocol (not a tuned hyperparameter): expose and
# withhold branches each run this many fresh probe trials.
VALIDATION_TRIALS = 3


@dataclass(frozen=True)
class TaskSample:
    episode: int
    topic: int
    distribution: str = "A"


@dataclass(frozen=True)
class StoredMemory:
    memory_id: str
    topic: int
    content: str
    source_episode: int
    contamination: str  # "none" | "false" | "spurious" | "outdated"
    true_effect: float  # ground truth effect on future same-topic tasks


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    topic: int
    distribution: str
    method: str
    seed: int
    success: bool
    reward: float
    n_injected: int
    injected_ids: tuple[str, ...]


@dataclass
class LifelongEnvironment:
    """Episode sampler + outcome model with known ground truth.

    Two independent RNG streams:
      - ``_task_rng`` seeded only by ``seed``: the task/topic sequence is
        identical across all methods for the same seed (paired design)
      - ``_rng`` seeded by ``seed`` + ``method_seed``: outcomes,
        extraction and TCI probes differ per method
    """

    seed: int = 0
    method_seed: int = 0
    n_topics: int = N_TOPICS
    change_episode: int | None = None
    changed_topics: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self._task_rng = np.random.RandomState(self.seed)
        self._rng = np.random.RandomState(
            (self.seed * 1000 + self.method_seed) % (2**31 - 1)
        )

    # ------------------------------------------------------------------
    # Task sampling
    # ------------------------------------------------------------------
    def sample_task(self, episode: int, topics: tuple[int, ...]) -> TaskSample:
        topic = int(self._task_rng.choice(topics))
        distribution = "A" if self.change_episode is None or episode < self.change_episode else "B"
        return TaskSample(episode=episode, topic=topic, distribution=distribution)

    # ------------------------------------------------------------------
    # Outcome model
    # ------------------------------------------------------------------
    def memory_effect(self, memory: StoredMemory, episode: int) -> float:
        """Effect of one memory at this episode (environment drift aware).

        Only memories created before the environment change are outdated:
        after ``change_episode`` they become harmful on changed topics.
        Memories extracted after the change reflect the new environment.
        """
        if (
            self.change_episode is not None
            and episode >= self.change_episode
            and memory.source_episode < self.change_episode
            and memory.topic in self.changed_topics
            and memory.contamination == "none"
        ):
            # outdated: previously helpful knowledge is now harmful
            return -abs(memory.true_effect)
        return memory.true_effect

    def success_probability(
        self, topic: int, episode: int, injected: list[StoredMemory]
    ) -> float:
        total = BASE_SUCCESS
        for mem in injected:
            if mem.topic == topic:
                total += self.memory_effect(mem, episode)
            else:
                total -= DISTRACTION_PENALTY
        return float(np.clip(total, 0.02, 0.98))

    def execute(
        self, task: TaskSample, injected: list[StoredMemory]
    ) -> tuple[bool, float]:
        prob = self.success_probability(task.topic, task.episode, injected)
        success = bool(self._rng.random() < prob)
        return success, float(success)

    # ------------------------------------------------------------------
    # Experience extraction
    # ------------------------------------------------------------------
    def extract_candidate(
        self, task: TaskSample, contamination_ratio: float
    ) -> StoredMemory:
        """Extract one candidate memory from the episode's experience.

        With probability ``contamination_ratio`` the candidate is
        contaminated (half false, half spurious); otherwise it is a
        genuinely helpful procedure for the task topic.
        """
        roll = self._rng.random()
        if roll < contamination_ratio:
            kind = "false" if self._rng.random() < 0.5 else "spurious"
        else:
            kind = "none"
        return self._make_memory(task, kind)

    def inject_outdated(self, task: TaskSample) -> StoredMemory:
        """Create an outdated-style memory (helpful pre-change, harmful post)."""
        return self._make_memory(task, "outdated")

    def _make_memory(self, task: TaskSample, kind: str) -> StoredMemory:
        if kind == "none" or kind == "outdated":
            true_effect = HELPFUL_EFFECT
            content = f"validated procedure for topic {task.topic}"
        elif kind == "false":
            true_effect = HARMFUL_EFFECT
            content = f"plausible but wrong procedure for topic {task.topic}"
        else:  # spurious: worked once by luck, no future effect
            true_effect = 0.0
            content = f"one-off lucky trick for topic {task.topic}"
        digest = hashlib.sha1(
            f"{task.episode}:{task.topic}:{kind}:{self.seed}".encode()
        ).hexdigest()[:10]
        return StoredMemory(
            memory_id=f"mem_ep{task.episode}_{digest}",
            topic=task.topic,
            content=content,
            source_episode=task.episode,
            contamination=kind if kind != "none" else "none",
            true_effect=true_effect,
        )

    # ------------------------------------------------------------------
    # TCI validation probes (honest expose-vs-withhold trials)
    # ------------------------------------------------------------------
    def tci_probe_delta(self, memory: StoredMemory, episode: int) -> float:
        """Estimate causal utility delta on fresh same-topic task instances.

        Runs VALIDATION_TRIALS expose trials and VALIDATION_TRIALS withhold
        trials; returns mean(expose) - mean(withhold). The outcome model is
        identical to real episodes, so probes are stochastic estimates,
        not ground-truth lookups.
        """
        topic = memory.topic
        expose_reward = 0.0
        withhold_reward = 0.0
        for _ in range(VALIDATION_TRIALS):
            p_expose = self.success_probability(topic, episode, [memory])
            p_withhold = self.success_probability(topic, episode, [])
            expose_reward += float(self._rng.random() < p_expose)
            withhold_reward += float(self._rng.random() < p_withhold)
        return (expose_reward - withhold_reward) / VALIDATION_TRIALS
