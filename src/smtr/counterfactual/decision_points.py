import copy
import hashlib
import json
from typing import Any, Protocol

from pydantic import BaseModel

from smtr.counterfactual.schemas import DecisionPoint
from smtr.memory.snapshot import MemoryStoreSnapshot
from smtr.router.candidate_proposer import CandidateProposal


def canonical_digest(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        if isinstance(item, dict):
            return {str(key): normalize(val) for key, val in sorted(item.items())}
        if isinstance(item, list | tuple):
            return [normalize(val) for val in item]
        return item

    encoded = json.dumps(normalize(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


class DecisionPointRecorder(Protocol):
    def record(
        self,
        *,
        graph_state: dict[str, Any],
        environment_snapshot: dict[str, Any],
        receiver_agent_id: str,
        receiver_role: str,
        graph_node: str,
        candidate_proposal: CandidateProposal,
        memory_store_snapshot: MemoryStoreSnapshot,
        run_seed: int,
    ) -> None: ...


class InMemoryDecisionPointRecorder:
    def __init__(self) -> None:
        self.decision_points: list[DecisionPoint] = []

    def record(
        self,
        *,
        graph_state: dict[str, Any],
        environment_snapshot: dict[str, Any],
        receiver_agent_id: str,
        receiver_role: str,
        graph_node: str,
        candidate_proposal: CandidateProposal,
        memory_store_snapshot: MemoryStoreSnapshot,
        run_seed: int,
    ) -> None:
        state_snapshot = copy.deepcopy(graph_state)
        env_snapshot = copy.deepcopy(environment_snapshot)
        capture_index = len(self.decision_points)
        episode_id = str(state_snapshot.get("episode_id", f"episode-{run_seed}"))
        task_id = str(state_snapshot.get("task_id", episode_id))
        digest_payload = {
            "episode_id": episode_id,
            "task_id": task_id,
            "graph_node": graph_node,
            "receiver_agent_id": receiver_agent_id,
            "receiver_role": receiver_role,
            "task_stage": candidate_proposal.request.task_stage,
            "graph_state_snapshot": state_snapshot,
            "environment_snapshot": env_snapshot,
            "candidate_proposal": candidate_proposal,
            "memory_store_snapshot": memory_store_snapshot,
            "run_seed": run_seed,
            "capture_index": capture_index,
        }
        self.decision_points.append(
            DecisionPoint(
                episode_id=episode_id,
                task_id=task_id,
                graph_node=graph_node,
                receiver_agent_id=receiver_agent_id,
                receiver_role=receiver_role,
                task_stage=candidate_proposal.request.task_stage,
                graph_state_snapshot=state_snapshot,
                environment_snapshot=env_snapshot,
                candidate_proposal=candidate_proposal,
                memory_store_snapshot=memory_store_snapshot,
                run_seed=run_seed,
                capture_index=capture_index,
                snapshot_digest=canonical_digest(digest_payload),
            )
        )
