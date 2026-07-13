# τ³-Bench ↔ SMTR Minimal Real LLM Integration

## Scope

**Only τ³-bench, retail domain, text half-duplex mode.** WebArena and ScienceWorld are deferred until τ³ paired rollout + critic + real LLM agent are validated.

τ³-bench repo: `sierra-research/tau2-bench` (uses `uv`, Python >=3.12, <3.14).
SMTR Python: 3.12.3 — compatible.

---

## Phase 0: Verify τ³-Bench Installation (no SMTR changes)

**Goal**: Install τ³-bench in an isolated venv, run retail domain tasks, understand the data flow.

**Steps**:
1. Clone `https://github.com/sierra-research/tau2-bench` alongside SMTR workspace
2. `uv sync` (core only — text mode)
3. Run `tau2 run --domain retail --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 1 --num-tasks 3`
4. Inspect output in `data/simulations/` to understand:
   - Task schema (task_id, instruction, domain, metadata)
   - Policy format (domain_policy string)
   - Tool definitions (list of Tool objects with schemas)
   - User simulator behavior (how dialogue turns progress)
   - Evaluation criteria (reward_basis, evaluation_criteria, pass/fail logic)
   - Trajectory format (messages, tool calls, tool results)
5. Verify reproducibility: same task + user-llm + seed → same trajectory

**Deliverable**: A short internal note documenting τ³ task/trajectory/evaluation schemas.

**Risk**: API key requirement for LLM-based user simulator. Mitigate: use `mock` domain first for structural validation, then `retail` with real keys.

---

## Phase 1: SMTRTauAgent — Minimal Agent Loop

**Goal**: Implement SMTR as a τ³ agent plugin. τ³ orchestrator handles user simulation, tool execution, and dialogue; SMTR handles routing + memory injection + LLM generation.

### Task 1: Formalize `LLMAdapter` Protocol (SMTR internal)

**File**: `src/smtr/runtime/llm_interface.py` (new)

```python
class LLMAdapter(Protocol):
    """SMTR-internal LLM abstraction. NOT the τ³ agent interface."""
    def plan(self, task: str, observation: dict, visible_payloads: list[dict]) -> dict: ...
    def summarize_execution(self, results: list[dict]) -> str: ...
```

- `DeterministicFakeLLM` and `RealLLM` already satisfy this — no behavior change
- Update `src/smtr/runtime/agents.py` to type `llm` param as `LLMAdapter | None`

### Task 2: Define `BenchmarkTask` and `OutcomeEvaluator` Protocols

**File**: `src/smtr/counterfactual/benchmark_interface.py` (new)

```python
from typing import Any, Protocol, Iterable
from pydantic import BaseModel

class BenchmarkTask(BaseModel):
    task_id: str
    instruction: str
    domain: str
    metadata: dict[str, Any] = {}

class BenchmarkEpisode(BaseModel):
    task: BenchmarkTask
    seed: int
    episode_id: str

class Outcome(BaseModel):
    success: bool
    reward: float
    metadata: dict[str, Any] = {}

class OutcomeEvaluator(Protocol):
    def evaluate(
        self,
        *,
        task: BenchmarkTask,
        trajectory: list[dict[str, Any]],
        final_state: dict[str, Any],
    ) -> Outcome: ...

class BenchmarkTaskProvider(Protocol):
    def iter_tasks(self, *, split: str) -> Iterable[BenchmarkTask]: ...
    def reset_episode(self, task: BenchmarkTask, *, seed: int) -> BenchmarkEpisode: ...
```

**Key design decisions** (from critique):
- No `ToyTaskSpec` — `BenchmarkTask` is domain-agnostic
- No `ensure_memories()` — memory construction is a separate concern (Phase 2)
- No `evaluation_metadata()` — outcome comes from evaluator, not task provider
- `OutcomeEvaluator` takes task + trajectory + final_state, not just observation

### Task 3: Implement `SMTRTauAgent`

**File**: `src/smtr/runtime/tau3_agent.py` (new)

This is the **core integration point**. SMTR wraps as a τ³ `HalfDuplexAgent`:

```python
# Pseudocode — actual implementation depends on τ³ API discovered in Phase 0
class SMTRTauAgent(HalfDuplexAgent):
    """SMTR routing + memory injection as a τ³ agent plugin."""
    
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        *,
        memory_pool: MemoryPool | None = None,
        router: ProductionSequentialRouter | None = None,
        llm: LLMAdapter | None = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self._memory_pool = memory_pool
        self._router = router
        self._llm = llm or RealLLM(...)
    
    def generate_next_message(self, messages, **kwargs) -> dict:
        # 1. Build SMTR routing state from τ³ conversation context
        # 2. Propose candidate memories from pool
        # 3. ProductionSequentialRouter decides share/withhold
        # 4. Inject selected payloads into LLM context
        # 5. LLM generates next message / tool call
        ...
```

**Integration architecture**:
```
τ³ Orchestrator (retail domain)
  → SMTRTauAgent.generate_next_message(...)
      → build SMTRState from τ³ messages
      → CandidateProposer proposes memories
      → ProductionSequentialRouter decides (critic-guided)
      → inject selected payloads into prompt
      → LLM generates tool call / response
  → τ³ executes tools, advances dialogue
  → τ³ evaluator scores task completion
```

**Constraints**:
- Single executor agent only — no planner/executor split yet
- τ³-bench remains the environment, user simulator, tool executor, and evaluator
- SMTR does NOT reimplement observe/apply — it plugs into τ³'s agent interface

### Task 4: Implement `TauRetailTaskProvider`

**File**: `src/smtr/counterfactual/tau_retail_tasks.py` (new)

Wraps τ³-bench's retail domain task loading:
- `iter_tasks(split="base")` → reads τ³ retail tasks
- `reset_episode(task, seed)` → creates a new episode with given seed
- Maps τ³ task schema → `BenchmarkTask`

### Task 5: Implement `TauOutcomeEvaluator`

**File**: `src/smtr/counterfactual/tau_evaluator.py` (new)

Wraps τ³-bench's official evaluation logic:
- Takes task + trajectory + final_state
- Delegates to τ³'s evaluator for reward computation
- Returns `Outcome(success, reward, metadata)`

### Task 6: Register `SMTRTauAgent` in τ³ Registry

**File**: `src/smtr/runtime/tau3_agent.py` (or separate registration file)

Register the agent factory so τ³ CLI can use it:
```python
registry.register_agent_factory(create_smtr_tau_agent, "smtr_tau_agent")
```

This enables: `tau2 run --domain retail --agent-llm gpt-4.1 --agent smtr_tau_agent ...`

### Task 7: Add `data_source` Field to Records

**File**: `src/smtr/counterfactual/schemas.py`

Add to `PairedInterventionRecord`:
```python
data_source: str = "toy"  # "toy", "tau_bench", "imported"
```

Add to `EvaluationGroupMetadata`:
```python
data_source: str = "toy"
```

Default `"toy"` ensures backward compatibility.

### Task 8: Tests (mock-based)

**File**: `tests/test_tau3_agent.py` (new)
- Mock τ³ agent base class (skip if τ³ not installed)
- Test `SMTRTauAgent.generate_next_message()` routing logic
- Test `TauRetailTaskProvider.iter_tasks()` schema mapping
- Test `TauOutcomeEvaluator.evaluate()` outcome construction
- Test `data_source` field backward compatibility

**File**: `tests/test_benchmark_interface.py` (new)
- Test `BenchmarkTask`, `BenchmarkEpisode`, `Outcome` schemas
- Test Protocol satisfaction

---

## Phase 2: Memory Corpus from Training Trajectories

**Goal**: Construct a small frozen memory pool from τ³ training task trajectories.

### Task 9: `MemoryCorpusBuilder`

**File**: `src/smtr/memory/corpus_builder.py` (new)

```python
class MemoryCorpusBuilder:
    """Extract ProcedurePayloads from historical trajectories."""
    
    def extract_procedures(
        self,
        trajectories: list[EpisodeTrace],
        *,
        min_frequency: int = 2,
    ) -> list[ProcedurePayload]: ...
    
    def build_routing_cards(
        self,
        procedures: list[ProcedurePayload],
    ) -> list[MemoryRoutingCard]: ...
```

**Critical constraint** (memory leakage prevention):
1. Only use **training split / development tasks** for trajectory collection
2. Extract `ProcedurePayload` + `MemoryRoutingCard` from historical trajectories
3. Freeze the memory pool before testing
4. Run router on **held-out tasks** only
5. All methods see the same raw benchmark policy — the only difference is whether they receive shared memory

**No hand-crafted seed memories** from benchmark domain policies (e.g., no manually writing "refund policy" from retail policy.md). That would be policy leakage.

### Task 10: Collect Training Trajectories

Run τ³ retail tasks (training split) with no-memory baseline to collect trajectories. Save as JSONL.

---

## Phase 3: Task-Start Paired Rollout

**Goal**: Run paired rollout at task start (not mid-trajectory) to collect Y^(1) vs Y^(0).

### Task 11: Task-Start Paired Rollout

For each held-out task:
1. Same task initial state + same user simulator config/seed
2. Branch A: share selected memory → run full episode → get Y^(1)
3. Branch B: withhold memory → run full episode → get Y^(0)
4. Compute transfer label from outcome pair

**No snapshot/restore at arbitrary mid-dialogue points.** Only task-start forking.

**Reproducibility concern**: If τ³'s user simulator is not deterministically reproducible from seed alone, either:
- Fix user response trajectories explicitly
- Run multiple trials per branch and aggregate

### Task 12: Integrate with Existing `PairedRolloutCollector`

Adapt `src/smtr/counterfactual/paired_rollout.py` to accept `OutcomeEvaluator` instead of hardcoded `ToyEnvironment` critic. The existing `_run_branch()` logic stays but delegates outcome evaluation to the `OutcomeEvaluator`.

---

## Phase 4 (Future): Multi-Agent + Second Environment

Only after Phases 1-3 are validated:
- Add planner → executor split in SMTRTauAgent
- Consider WebArena / ScienceWorld as second environment
- Generalize `BenchmarkTaskProvider` / `OutcomeEvaluator` into broader framework

---

## Dependencies

```
Phase 0 (verify τ³ install) — independent, no SMTR changes
Phase 1 Tasks 1-2 (Protocols) — independent
Phase 1 Tasks 3-6 (SMTRTauAgent + providers) — depend on Tasks 1-2 + Phase 0
Phase 1 Task 7 (data_source) — independent
Phase 1 Task 8 (tests) — depends on Tasks 1-7
Phase 2 Task 9 (MemoryCorpusBuilder) — depends on Phase 1
Phase 2 Task 10 (collect trajectories) — depends on Phase 1 + Task 9
Phase 3 Task 11 (paired rollout) — depends on Phase 2
Phase 3 Task 12 (integrate PairedRolloutCollector) — depends on Task 11
```

## Critical Files

1. `src/smtr/runtime/tau3_agent.py` (new) — SMTRTauAgent, the core integration
2. `src/smtr/counterfactual/benchmark_interface.py` (new) — BenchmarkTask, OutcomeEvaluator, BenchmarkTaskProvider protocols
3. `src/smtr/counterfactual/schemas.py` — add data_source field
4. `src/smtr/runtime/agents.py` — type LLM param as LLMAdapter
5. `src/smtr/counterfactual/paired_rollout.py` — adapt for OutcomeEvaluator (Phase 3)

## Rejected Alternatives

| Alternative | Why Rejected |
|---|---|
| Wrap τ³ as `EnvironmentAdapter` (observe/apply) | τ³ is an orchestrator, not a forkable env. SMTR should be the agent plugin, not re-implement τ³'s tool loop. |
| Three benchmarks simultaneously | Scope too large. τ³ retail alone is complex enough. WebArena needs self-hosted sites. ScienceWorld has weak SMTR fit. |
| Hand-craft seed memories from retail policy | Memory leakage risk. Must extract from training trajectories only. |
| Mid-trajectory snapshot/restore | τ³ doesn't guarantee arbitrary state forking. Task-start paired rollout is sufficient for first version. |
| `success_fn(observation)` for critic | τ³ evaluation needs task schema + trajectory + evaluation criteria, not just observation. Use `OutcomeEvaluator`. |
| Generalize `TaskProvider` from `ToyTaskSpec` | `ToyTaskSpec` exposes toy assumptions. Use `BenchmarkTask`/`BenchmarkEpisode` instead. |
| Old `tau-bench` (airline/retail) | Deprecated. Use τ³-bench (`tau2-bench` repo) with latest task fixes. |
