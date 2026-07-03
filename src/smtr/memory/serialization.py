from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def model_to_json(model: BaseModel) -> str:
    return model.model_dump_json()


def model_from_json(model_type: type[T], payload: str) -> T:
    return model_type.model_validate_json(payload)
