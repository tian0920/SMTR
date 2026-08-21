"""Persistent memory lifecycle schema (long-term memory extension, Task 1).

Independent of the existing ``ProcedurePayload`` / ``MemoryRoutingCard``
schemas: a ``PersistentMemoryEntry`` tracks the lifecycle state of one
experience-derived memory through

    candidate -> validated | rejected

without modifying any existing memory interface.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MemoryLifecycleStatus = Literal["candidate", "validated", "rejected"]


def utc_now() -> datetime:
    return datetime.now(UTC)


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
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
