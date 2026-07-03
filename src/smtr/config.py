from pydantic import BaseModel, ConfigDict, Field


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int = 7
    top_k: int = Field(default=3, ge=1)

