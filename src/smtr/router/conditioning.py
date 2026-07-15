"""Selected-set conditioning policies for sequential routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SelectedSetConditioningPolicy(Protocol):
    @property
    def policy_name(self) -> str:
        ...

    def critic_selected_set(
        self,
        *,
        initial_selected_set: tuple[str, ...],
        current_selected_set: tuple[str, ...],
    ) -> tuple[str, ...]:
        ...


@dataclass(frozen=True)
class DynamicSelectedSetConditioning:
    """Condition the critic on the current accepted selected set."""

    policy_name: str = "dynamic_selected_set"

    def critic_selected_set(
        self,
        *,
        initial_selected_set: tuple[str, ...],
        current_selected_set: tuple[str, ...],
    ) -> tuple[str, ...]:
        del initial_selected_set
        return current_selected_set


@dataclass(frozen=True)
class FrozenInitialSelectedSetConditioning:
    """Condition the critic on the invocation initial selected set."""

    policy_name: str = "frozen_initial_selected_set"

    def critic_selected_set(
        self,
        *,
        initial_selected_set: tuple[str, ...],
        current_selected_set: tuple[str, ...],
    ) -> tuple[str, ...]:
        del current_selected_set
        return initial_selected_set
