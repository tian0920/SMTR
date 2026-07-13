# Real Dataset Interface Preparation with Three Benchmarks

## Goal

Make the SMTR codebase ready to ingest data from three real-world agent benchmarks:
1. **tau-bench** (Sierra Research) — tool-agent-user interaction in airline/retail domains
2. **WebArena** (CMU et al.) — realistic web browsing across 5+ websites (812 tasks)
3. **ScienceWorld** (AI2) — text-based science experiment environment (30+ task types, thousands of variations)

Each benchmark maps to SMTR as: **environment** (observable state + actions + outcomes) → **task provider** (generates tasks) → **adapter** (wraps benchmark to satisfy SMTR protocols).

---

## Task 1: Formalize `LLMAdapter` Protocol

**Problem**: `DeterministicFakeLLM` and `RealLLM` share `plan()` + `summarize_execution()` but no Protocol exists.

**File**: `src/smtr/runtime/llm_interface.py` (new)

```python
class LLMAdapter(Protocol):
    def plan(self, task: str, observation: dict, visible_payloads: list[dict]) -> dict: ...
    def summarize_execution(self, results: list[dict]) -> str: ...
```

**Updates**:
- `src/smtr/runtime/fake_llm.py` — `DeterministicFakeLLM` already satisfies this
- `src/smtr/runtime/real_llm.py` — `RealLLM` already satisfies this
- `src/smtr/runtime/agents.py` — type `llm` param as `LLMAdapter | None`

**Risk**: Low — no behavior change.

---

## Task 2: Formalize `TaskProvider` Protocol

**Problem**: `CounterfactualToyTaskProvider` is toy-specific. Need abstract interface for benchmark task providers.

**File**: `src/smtr/counterfactual/task_provider.py`

Add Protocol:
```python
class TaskProvider(Protocol):
    def generate(self, *, scenario: str, seed: int) -> ToyTaskSpec: ...
    def ensure_memories(self, repository: SharedMemoryRepository) -> None: ...
    def evaluation_metadata(self, *, scenario: str, target_memory_id: str,
                           selected_before: list[str], seed: int) -> EvaluationGroupMetadata: ...
```

`CounterfactualToyTaskProvider` already satisfies this.

**Risk**: Very low — pure type annotation.

---

## Task 3: Parameterize `PairedRolloutCollector` for Any Environment

**Problem**: `PairedRolloutCollector._run_branch()` (line 216 of `paired_rollout.py`) hardcodes `ToyEnvironment.clone_from_snapshot()`.

**File**: `src/smtr/counterfactual/paired_rollout.py`

1. Add `env_factory` to `__init__`:
   ```python
   def __init__(self, *, env_factory: Callable[[dict, int], Any] | None = None):
       self._env_factory = env_factory
   ```
2. In `_run_branch()`, use factory if provided, else fall back to `ToyEnvironment`.

**File**: `src/smtr/cli.py` — pass `env_factory` through in `_collect_counterfactual()`.

**Risk**: Low — default behavior unchanged.

---

## Task 4: Generalize Critic Node for Benchmark Outcomes

**Problem**: `run_critic()` in `agents.py` hardcodes `target in inventory` for success. Real benchmarks have different success criteria.

**File**: `src/smtr/runtime/agents.py`

Make critic configurable via a `success_fn` parameter:
```python
def run_critic(state: SMTRState, *, success_fn: Callable[[dict], tuple[bool, float]] | None = None) -> dict:
    if success_fn:
        success, reward = success_fn(state["environment_observation"])
    else:
        # existing toy logic
        target = observation.get("target_artifact")
        success = target in observation.get("inventory", [])
        reward = 1.0 if success else 0.0
```

**File**: `src/smtr/runtime/graph.py` — thread `success_fn` through `build_graph()`.

**Risk**: Low — default behavior unchanged.

---

## Task 5: Add `data_source` Field to Records

**File**: `src/smtr/counterfactual/schemas.py`

Add to `PairedInterventionRecord`:
```python
data_source: str = "toy"  # "toy", "tau_bench", "webarena", "scienceworld", "imported"
```

Add to `EvaluationGroupMetadata`:
```python
data_source: str = "toy"
```

Default `"toy"` ensures backward compatibility.

**Risk**: Low — additive field with default.

---

## Task 6: tau-bench Environment Adapter

**File**: `src/smtr/runtime/tau_bench_env.py` (new)

Wraps tau-bench's airline/retail environment to satisfy `EnvironmentAdapter`:
```python
class TauBenchEnvironmentAdapter:
    """Adapter for tau-bench airline/retail environments."""
    
    def __init__(self, *, domain: str = "retail", seed: int = 0):
        # Lazy import tau_bench to avoid hard dependency
        ...
    
    def observe(self) -> dict:
        # Return current state as dict (tool availability, user state, policy constraints)
        ...
    
    def snapshot(self) -> dict:
        # Full serializable state for paired rollout forking
        ...
    
    def restore(self, snapshot: dict) -> None:
        # Restore from snapshot
        ...
    
    def apply(self, action: dict) -> dict:
        # Execute tool call action, return {ok, action, observation/error}
        ...
```

Key mapping:
- tau-bench **tools** (API calls) → SMTR **actions**
- tau-bench **user simulator state** → part of SMTR **environment_observation**
- tau-bench **task completion** → SMTR **team_success** (via `success_fn`)
- tau-bench **domains** (airline, retail) → SMTR **environment_regime**

**Seed memories**: Create a set of procedural memories for airline/retail domain policies (e.g., "refund policy", "booking change procedure", "cancellation rules") that the router can learn to share/withhold.

**Risk**: Medium — tau-bench requires API keys for user simulator. Mitigate by supporting offline trajectory replay from `historical_trajectories/`.

---

## Task 7: WebArena Environment Adapter

**File**: `src/smtr/runtime/webarena_env.py` (new)

Wraps WebArena's `ScriptBrowserEnv`:
```python
class WebArenaEnvironmentAdapter:
    """Adapter for WebArena web browsing environment."""
    
    def __init__(self, *, config_file: str | None = None, seed: int = 0):
        # Lazy import webarena/browser_env
        ...
    
    def observe(self) -> dict:
        # Return accessibility tree / HTML as dict
        # Include current URL, available links, form fields
        ...
    
    def snapshot(self) -> dict:
        # Full browser state snapshot for forking
        ...
    
    def restore(self, snapshot: dict) -> None:
        ...
    
    def apply(self, action: dict) -> dict:
        # Execute browser action (click, type, navigate, etc.)
        ...
```

Key mapping:
- WebArena **browser actions** (click, type, scroll, navigate) → SMTR **actions**
- WebArena **accessibility tree** → SMTR **environment_observation**
- WebArena **task success** (from evaluation harness) → SMTR **team_success**
- WebArena **websites** (shopping, reddit, gitlab, map) → SMTR **environment_regime**

**Seed memories**: Create procedural memories for web navigation patterns (e.g., "how to search on shopping site", "how to post on reddit", "how to create gitlab issue").

**Risk**: High — WebArena requires self-hosted websites (Docker). Mitigate by:
1. Making the adapter gracefully skip if WebArena is not installed
2. Supporting pre-recorded trajectory import
3. Marking WebArena tests as `@pytest.mark.skip` unless `--webarena` flag passed

---

## Task 8: ScienceWorld Environment Adapter

**File**: `src/smtr/runtime/scienceworld_env.py` (new)

Wraps ScienceWorld's text-based environment:
```python
class ScienceWorldEnvironmentAdapter:
    """Adapter for ScienceWorld text-based science environment."""
    
    def __init__(self, *, task_num: int = 1, simplifications: str = "easy", seed: int = 0):
        # Lazy import scienceworld
        ...
    
    def observe(self) -> dict:
        # Return text observation + available actions as dict
        ...
    
    def snapshot(self) -> dict:
        # Full simulator state for forking
        ...
    
    def restore(self, snapshot: dict) -> None:
        ...
    
    def apply(self, action: dict) -> dict:
        # Execute text action (e.g., "heat object", "pour liquid")
        ...
```

Key mapping:
- ScienceWorld **text actions** → SMTR **actions**
- ScienceWorld **text observations** → SMTR **environment_observation`
- ScienceWorld **task completion score** → SMTR **team_success** / **team_reward**
- ScienceWorld **task types** (boil, melt, conductivity, genetics, etc.) → SMTR **environment_regime**

**Seed memories**: Create procedural memories for science experiment patterns (e.g., "how to measure temperature", "how to test conductivity", "how to grow a plant").

**Risk**: Medium — ScienceWorld requires Java 1.8+. Mitigate by lazy import and graceful fallback.

---

## Task 9: Dataset Loader & Manifest

**File**: `src/smtr/evaluation/dataset_loader.py` (new)

```python
class DatasetManifest(BaseModel):
    name: str
    source: str  # "toy", "tau_bench", "webarena", "scienceworld"
    record_count: int
    schema_version: str
    environment_type: str
    llm_type: str
    collection_date: str
    description: str = ""
    benchmark_config: dict = {}  # domain, task_nums, simplifications, etc.

def load_dataset(path: str | Path) -> tuple[list[PairedInterventionRecord], DatasetManifest | None]:
    """Load a JSONL dataset, validate schema, auto-detect source."""
    ...

def validate_dataset(records: list[PairedInterventionRecord]) -> list[str]:
    """Return validation warnings."""
    ...
```

**Risk**: Low — new file, no existing code changes.

---

## Task 10: Tests

**File**: `tests/test_real_dataset_interfaces.py` (new)

Tests:
1. `LLMAdapter` protocol — both FakeLLM and RealLLM satisfy it
2. `TaskProvider` protocol — `CounterfactualToyTaskProvider` satisfies it
3. `PairedRolloutCollector` with custom env_factory — collect records using a mock environment
4. `data_source` field — default "toy", settable, backward compatible
5. `DatasetManifest` — creation and validation
6. tau-bench adapter — mock test (skip if tau-bench not installed)
7. WebArena adapter — mock test (skip if webarena not installed)
8. ScienceWorld adapter — mock test (skip if scienceworld not installed)
9. `run_critic` with custom `success_fn` — verify configurable success criteria

**File**: `tests/test_tau_bench_env.py` (new, skip if not installed)
**File**: `tests/test_webarena_env.py` (new, skip if not installed)
**File**: `tests/test_scienceworld_env.py` (new, skip if not installed)

---

## Dependencies

```
Task 1 (LLMAdapter Protocol) — independent
Task 2 (TaskProvider Protocol) — independent
Task 3 (env_factory in PairedRolloutCollector) — independent
Task 4 (success_fn in critic) — independent
Task 5 (data_source field) — independent
Task 6 (tau-bench adapter) — depends on Tasks 3, 4, 5
Task 7 (WebArena adapter) — depends on Tasks 3, 4, 5
Task 8 (ScienceWorld adapter) — depends on Tasks 3, 4, 5
Task 9 (Dataset loader) — depends on Task 5
Task 10 (Tests) — depends on all above
```

## Critical Files

1. `src/smtr/counterfactual/paired_rollout.py` — parameterize environment factory
2. `src/smtr/counterfactual/schemas.py` — add data_source field
3. `src/smtr/runtime/agents.py` — generalize critic success function
4. `src/smtr/runtime/graph.py` — thread success_fn through build_graph
5. `src/smtr/counterfactual/task_provider.py` — add TaskProvider Protocol

## Benchmark Installation (Optional Dependencies)

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
tau-bench = ["tau-bench"]
webarena = ["browser-gym", "playwright"]
scienceworld = ["scienceworld"]
all-benchmarks = ["tau-bench", "browser-gym", "playwright", "scienceworld"]
```

## Rejected Alternatives

- **Hard dependency on all benchmarks**: Rejected — makes SMTR heavy; use optional deps + lazy imports.
- **Rewriting evaluation pipeline per benchmark**: Rejected — existing pipeline is benchmark-agnostic; just need adapters.
- **Schema version bump to 2.0**: Rejected — additive field with default doesn't require major version bump.
- **Pre-collecting trajectories from benchmarks**: Rejected for now — first prepare interfaces, then collect data in follow-up.
