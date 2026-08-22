"""Persistent memory lifecycle schema (long-term memory extension, Task 1).

Independent of the existing ``ProcedurePayload`` / ``MemoryRoutingCard``
schemas: a ``PersistentMemoryEntry`` tracks the lifecycle state of one
experience-derived memory through

    candidate -> validated | rejected

without modifying any existing memory interface.

Receiver-conditioned extension (Receiver=3 protocol):
    Adds per-receiver validation tracking so the same memory can be
    validated for one receiver but rejected for another.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MemoryLifecycleStatus = Literal["candidate", "validated", "rejected"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class ValidationRecord(BaseModel):
    """One TCI validation event (P0-3 audit trail)."""

    model_config = ConfigDict(frozen=True)

    episode_id: int
    expose_reward: float
    withhold_reward: float
    delta: float
    decision: str  # "validated" | "rejected" | "suspect"


class ReceiverValidationRecord(BaseModel):
    """One receiver-conditioned TCI validation event.

    Extends ValidationRecord with receiver identity so the same memory
    can have different validation outcomes per receiver.
    """

    model_config = ConfigDict(frozen=True)

    receiver_id: str
    episode_id: int
    expose_reward: float
    withhold_reward: float
    delta: float
    decision: str  # "validated" | "rejected"
    validation_source: str = "receiver_counterfactual_rollout"


class PersistentMemoryEntry(BaseModel):
    """One experience-derived memory with lifecycle state.

    Fields follow the long-term memory spec:
      - ``memory_id``: unique identifier
      - ``content``: memory text (procedure / knowledge)
      - ``source_episode``: episode index that produced this memory
      - ``receiver``: target agent the memory is meant for
      - ``created_step``: global step at creation time
      - ``tci_effect``: latest TCI-estimated treatment effect (delta)
      - ``status``: candidate / validated / rejected
      - ``validation_count``: number of TCI validations performed
      - ``validation_history``: full audit trail of every probe (P0-3)
    """

    model_config = ConfigDict(frozen=True)

    memory_id: str
    content: str
    source_episode: int
    receiver: str
    created_step: int
    tci_effect: float | None = None
    status: MemoryLifecycleStatus = "candidate"
    validation_count: int = 0
    validation_history: tuple[ValidationRecord, ...] = ()
    # Receiver-conditioned extension (backward compatible: all optional)
    receiver_id: str | None = None
    validation_target: str | None = None
    receiver_context: str | None = None
    receiver_validation_history: tuple[ReceiverValidationRecord, ...] = ()
    receiver_decisions: dict[str, str] = Field(default_factory=dict)
    # Authoritative per-receiver lifecycle state.
    # Maps receiver_id → "validated" | "rejected" | "candidate".
    # Legacy ``status`` field is kept for backward compatibility only.
    receiver_status: dict[str, MemoryLifecycleStatus] = Field(default_factory=dict)
    validation_source: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
