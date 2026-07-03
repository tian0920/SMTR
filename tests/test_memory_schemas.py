from smtr.memory.execution_evidence import selected_set_signature
from smtr.memory.schemas import MemoryRoutingCard, ProcedurePayload


def test_selected_set_signature_is_deterministic_and_order_insensitive() -> None:
    assert selected_set_signature(["b", "a"]) == selected_set_signature(["a", "b"])
    assert selected_set_signature([]) == selected_set_signature([])


def test_payload_and_card_are_physically_separate() -> None:
    payload = ProcedurePayload(memory_id="m", goal="goal", steps=["secret step"])
    card = MemoryRoutingCard(memory_id="m", goal_summary="goal")

    assert "secret step" in repr(payload)
    assert "secret step" not in repr(card)
    assert "steps" not in card.model_dump()
