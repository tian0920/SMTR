"""Protocols for isolated MARBLE environment instances."""

from __future__ import annotations

from typing import Protocol


class MarbleEnvironmentInstance(Protocol):
    scenario: str
    task_id: str

    def initial_state_digest(self) -> str:
        ...

    def run(
        self,
        *,
        agent_input: object,
        generation_seed: int,
    ) -> object:
        ...

    def close(self) -> None:
        ...


class MarbleEnvironmentFactory(Protocol):
    scenario: str

    def create_isolated(
        self,
        *,
        task: object,
        initial_state_bundle: object,
        branch_id: str,
        workspace: str,
    ) -> MarbleEnvironmentInstance:
        ...
