from smtr.memory.repository import SharedMemoryRepository
from smtr.memory.schemas import ExecutionEvidence, MemoryRoutingCard, ProcedurePayload
from smtr.memory.snapshot import MemoryStoreSnapshot


class ReadOnlyCounterfactualStoreError(RuntimeError):
    pass


class ReadOnlyPinnedMemoryView:
    def __init__(
        self,
        repository: SharedMemoryRepository,
        snapshot: MemoryStoreSnapshot,
    ) -> None:
        self._repository = repository
        self._snapshot = snapshot

    def get_routing_cards(self) -> list[MemoryRoutingCard]:
        return self._snapshot.get_routing_cards()

    def get_routing_card(self, memory_id: str) -> MemoryRoutingCard:
        for card in self._snapshot.routing_cards:
            if card.memory_id == memory_id:
                return card
        raise KeyError(memory_id)

    def get_payload(self, memory_id: str, version: int | None = None) -> ProcedurePayload:
        pinned_version = self._snapshot.get_active_version(memory_id)
        if version is not None and version != pinned_version:
            raise KeyError(f"{memory_id} v{version} is not pinned")
        return self._repository.get_payload(memory_id, pinned_version)

    def get_selected_payloads(self, memory_ids: list[str]) -> list[ProcedurePayload]:
        payloads = [self.get_payload(memory_id) for memory_id in memory_ids]
        return sorted(payloads, key=lambda payload: (payload.goal, payload.memory_id))

    def current_revision(self) -> int:
        return self._snapshot.store_revision

    def create_read_snapshot(self) -> MemoryStoreSnapshot:
        return self._snapshot

    def create_memory(self, *args, **kwargs) -> None:
        del args, kwargs
        raise ReadOnlyCounterfactualStoreError("counterfactual memory view is read-only")

    def record_execution_evidence(self, evidence: ExecutionEvidence) -> None:
        del evidence
        raise ReadOnlyCounterfactualStoreError("counterfactual memory view is read-only")
