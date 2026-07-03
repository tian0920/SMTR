from pydantic import BaseModel, ConfigDict

from smtr.memory.schemas import MemoryRoutingCard


class MemoryStoreSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    store_revision: int
    routing_cards: list[MemoryRoutingCard]
    active_payload_versions: dict[str, int]

    def get_routing_cards(self) -> list[MemoryRoutingCard]:
        return list(self.routing_cards)

    def get_active_version(self, memory_id: str) -> int:
        return self.active_payload_versions[memory_id]
