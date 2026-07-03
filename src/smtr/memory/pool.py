from collections.abc import Iterable

from smtr.memory.schemas import MemoryRoutingCard, ProcedurePayload
from smtr.memory.snapshot import MemoryStoreSnapshot


class SharedMemoryPool:
    def __init__(
        self,
        routing_cards: Iterable[MemoryRoutingCard],
        payloads: Iterable[ProcedurePayload],
    ) -> None:
        self._cards = {card.memory_id: card for card in routing_cards}
        self._payloads = {payload.memory_id: payload for payload in payloads}
        missing_payloads = set(self._cards) - set(self._payloads)
        missing_cards = set(self._payloads) - set(self._cards)
        if missing_payloads or missing_cards:
            raise ValueError(
                "routing cards and payloads must share memory IDs; "
                f"missing_payloads={sorted(missing_payloads)}, "
                f"missing_cards={sorted(missing_cards)}"
            )

    def list_routing_cards(self) -> list[MemoryRoutingCard]:
        return [self._cards[memory_id] for memory_id in sorted(self._cards)]

    def get_routing_cards(self) -> list[MemoryRoutingCard]:
        return self.list_routing_cards()

    def get_routing_card(self, memory_id: str) -> MemoryRoutingCard:
        return self._cards[memory_id]

    def get_payload(self, memory_id: str, version: int | None = None) -> ProcedurePayload:
        payload = self._payloads[memory_id]
        if version is not None and payload.version != version:
            raise KeyError(f"{memory_id} v{version}")
        return payload

    def get_payloads(self, memory_ids: list[str]) -> list[ProcedurePayload]:
        return [self.get_payload(memory_id) for memory_id in memory_ids]

    def get_selected_payloads(self, memory_ids: list[str]) -> list[ProcedurePayload]:
        return self.get_payloads(memory_ids)

    def current_revision(self) -> int:
        return 0

    def create_read_snapshot(self) -> MemoryStoreSnapshot:
        cards = self.list_routing_cards()
        return MemoryStoreSnapshot(
            store_revision=0,
            routing_cards=[card.model_copy(deep=True) for card in cards],
            active_payload_versions={
                card.memory_id: card.active_payload_version for card in cards
            },
        )
