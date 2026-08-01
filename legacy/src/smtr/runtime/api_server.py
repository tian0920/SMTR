"""FastAPI server for SMTR with OpenAI-compatible endpoints.

Provides:
- POST /v1/chat/completions - OpenAI-compatible chat completion
- POST /smtr/run-pipeline - Run full SMTR pipeline with real LLM
- POST /smtr/demo - Run demo with real LLM
- GET /health - Health check
"""

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="SMTR API", version="0.1.0")

_llm_instance: Any = None
_env_factory: Any = None


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float = 1.0
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""

    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    """Token usage info."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class PipelineRequest(BaseModel):
    """Request to run SMTR pipeline."""

    task: str = "Obtain a target artifact using the valid action sequence."
    seed: int = 7
    top_k: int = 4
    use_real_llm: bool = True
    use_tool_environment: bool = False


class DemoRequest(BaseModel):
    """Request to run SMTR demo."""

    seed: int = 7
    use_real_llm: bool = True


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str | None = None


def get_llm(model_name: str | None = None) -> Any:
    """Get or create LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        from smtr.runtime.real_llm import RealLLM

        _llm_instance = RealLLM(model_name=model_name or "Qwen/Qwen3.5-2B")
    return _llm_instance


def set_llm(llm: Any) -> None:
    """Set the LLM instance (for dependency injection)."""
    global _llm_instance
    _llm_instance = llm


def set_env_factory(factory: Any) -> None:
    """Set the environment factory (for dependency injection)."""
    global _env_factory
    _env_factory = factory


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_llm_instance is not None,
        model_name=getattr(_llm_instance, "_model_name", None) if _llm_instance else None,
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI-compatible chat completion endpoint."""
    llm = get_llm(request.model)

    prompt = "\n".join(f"{msg.role}: {msg.content}" for msg in request.messages)

    try:
        response_text = llm._generate(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
    )


@app.post("/smtr/demo")
async def run_demo_endpoint(request: DemoRequest) -> dict[str, Any]:
    """Run SMTR demo endpoint."""
    from smtr.runtime.graph import run_demo

    llm = get_llm() if request.use_real_llm else None
    env_factory = _env_factory if _env_factory else None

    try:
        state = run_demo(seed=request.seed, llm=llm, env_factory=env_factory)
        return {
            "task": state["task"],
            "team_success": state.get("team_success"),
            "team_reward": state.get("team_reward"),
            "router_trace_count": len(state.get("router_trace", [])),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo failed: {e}") from e


@app.post("/smtr/run-pipeline")
async def run_pipeline(request: PipelineRequest) -> dict[str, Any]:
    """Run full SMTR pipeline endpoint."""
    from smtr.runtime.environment import ToyEnvironment
    from smtr.runtime.graph import run_episode
    from smtr.runtime.tool_environment import ToolEnvironment

    llm = get_llm() if request.use_real_llm else None

    if request.use_tool_environment:

        def env_factory(seed: int) -> Any:
            return ToolEnvironment(seed=seed)
    else:
        env_factory = _env_factory or (lambda seed: ToyEnvironment(seed=seed))

    env = ToyEnvironment(seed=request.seed)
    try:
        state = run_episode(
            seed=request.seed,
            top_k=request.top_k,
            task=request.task,
            environment_observation=env.observe(),
            llm=llm,
            env_factory=env_factory,
        )
        return {
            "task": state["task"],
            "team_success": state.get("team_success"),
            "team_reward": state.get("team_reward"),
            "team_summary": state.get("team_summary"),
            "router_trace_count": len(state.get("router_trace", [])),
            "agent_outputs_keys": list(state.get("agent_outputs", {}).keys()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}") from e


def run_server(host: str = "0.0.0.0", port: int = 8000, **kwargs: Any) -> None:
    """Run the API server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, **kwargs)


if __name__ == "__main__":
    run_server()
