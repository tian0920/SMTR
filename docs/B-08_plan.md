# B-08: Real LLM + Real Tool Environment Integration

## Context

- GPU: Tesla T4, 16GB VRAM, driver 580.126.09
- No torch/transformers/fastapi currently installed
- Current system uses `DeterministicFakeLLM` (hardcoded strategy) and `ToyEnvironment` (state machine)
- Need to add real LLM while keeping the fake LLM as default for existing tests

## Architecture

```
RealLLM (real_llm.py)
  |-- Local mode: transformers + bitsandbytes 8-bit quantization
  |-- API mode: httpx client to remote endpoint
  |-- Interface: plan() + summarize_execution() (same as DeterministicFakeLLM)

ToolEnvironment (tool_environment.py)
  |-- Tool registry with realistic tools (file ops, API calls, search)
  |-- Interface: observe() / snapshot() / restore() / apply() (same as ToyEnvironment)

API Server (api_server.py)
  |-- FastAPI + uvicorn
  |-- POST /v1/chat/completions (OpenAI-compatible)
  |-- POST /smtr/run-pipeline (run full SMTR pipeline)
```

## Task 1: Install dependencies

Update `pyproject.toml` to add optional `[llm]` and `[api]` extras:
```toml
[project.optional-dependencies]
llm = ["torch>=2.1", "transformers>=4.40", "accelerate>=0.28", "bitsandbytes>=0.43"]
api = ["fastapi>=0.111", "uvicorn>=0.29", "httpx>=0.27"]
```

Install: `pip install -e ".[llm,api,dev]"`

## Task 2: Create `src/smtr/runtime/real_llm.py`

`RealLLM` class implementing the same interface as `DeterministicFakeLLM`:

```python
class RealLLM:
    """LLM adapter supporting local model or remote API."""
    
    def __init__(self, *, model_name="Qwen/Qwen3.5-2B",
                 api_base=None, load_in_8bit=True, max_new_tokens=512, temperature=0.1):
        if api_base:
            self._client = httpx.Client(base_url=api_base, timeout=120)
            self._model = None
        else:
            # Local: load with transformers + 8-bit quantization
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", load_in_8bit=load_in_8bit, torch_dtype=torch.float16)
    
    def plan(self, task, observation, visible_payloads) -> dict:
        # Build prompt with task + observation + payload summaries
        # Call model/API, parse JSON response into {plan, explanation}
        
    def summarize_execution(self, results) -> str:
        # Ask LLM to summarize execution results
```

Key design: structured JSON output via system prompt + response format constraint.

## Task 3: Create `src/smtr/runtime/tool_environment.py`

`ToolEnvironment` class implementing `EnvironmentAdapter` protocol:

```python
class ToolEnvironment:
    """Realistic tool environment with tool registry."""
    
    TOOLS = {
        "read_file": {"params": ["path"], "effect": "returns file content"},
        "write_file": {"params": ["path", "content"], "effect": "writes file"},
        "search_web": {"params": ["query"], "effect": "returns results"},
        "run_command": {"params": ["command"], "effect": "returns output"},
        "send_message": {"params": ["recipient", "message"], "effect": "delivers message"},
    }
    
    def observe(self) -> dict:  # Current state + available tools
    def apply(self, action: dict) -> dict:  # Execute tool, return result
    def snapshot(self) -> dict:
    def restore(self, snapshot: dict) -> None:
```

Provides richer actions beyond the 3-step valid_sequence, with realistic state transitions.

## Task 4: Update `src/smtr/runtime/agents.py`

Refactor `run_planner` and `run_executor` to accept configurable LLM and environment:

```python
def run_planner(state, llm=None):
    llm = llm or DeterministicFakeLLM()  # backward compatible
    ...

def run_executor(state, llm=None, env_factory=None):
    llm = llm or DeterministicFakeLLM()
    env = (env_factory or (lambda seed: ToyEnvironment(seed=seed)))(seed=state["run_seed"])
    ...
```

## Task 5: Update `src/smtr/runtime/graph.py`

Add `llm` and `env_factory` parameters to `build_graph`, `run_demo`, `run_episode`:

```python
def build_graph(*, llm=None, env_factory=None, ...):
    # Create closures for planner/executor nodes with injected llm/env
    graph.add_node("planner", lambda state: run_planner(state, llm=llm))
    graph.add_node("executor", lambda state: run_executor(state, llm=llm, env_factory=env_factory))
```

## Task 6: Create `src/smtr/runtime/api_server.py`

FastAPI server with:
- `POST /v1/chat/completions` - OpenAI-compatible endpoint
- `POST /smtr/run-pipeline` - Run full SMTR pipeline with real LLM
- `GET /health` - Health check
- `POST /smtr/demo` - Run demo with real LLM

```python
app = FastAPI(title="SMTR API")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Forward to RealLLM

@app.post("/smtr/run-pipeline")
async def run_pipeline(request: PipelineRequest):
    # Run full SMTR pipeline
```

## Task 7: Update `src/smtr/cli.py`

Add new commands:
- `smtr serve-api` - Start API server
- `smtr demo-real` - Run demo with real LLM
- `smtr collect-counterfactual --use-real-llm` - Collection with real LLM

## Task 8: Create tests

- `tests/test_real_llm.py` - Test RealLLM with mock/local model
- `tests/test_tool_environment.py` - Test ToolEnvironment
- `tests/test_api_server.py` - Test API endpoints

## Task 9: Update documentation

- `todo.md` - Mark B-08 as implemented
- `changelog.md` - Add B-08 entry
- `results.md` - Add test results
- `implementation.md` - Update Priority 6 / Chapter 15 status

## Key Design Decisions

1. **Model**: Qwen/Qwen3.5-2B (2B params, ~2GB FP16 / ~1GB 8-bit, fits T4 16GB easily; fast inference)
2. **Quantization**: bitsandbytes 8-bit (T4 supports FP16 + INT8, not bfloat16)
3. **API compatibility**: OpenAI-compatible `/v1/chat/completions` for easy integration
4. **Backward compatibility**: All existing code continues to use `DeterministicFakeLLM` by default
5. **Lazy imports**: torch/transformers only imported when RealLLM is instantiated (not at module load)
