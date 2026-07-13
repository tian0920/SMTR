# MARBLE ↔ SMTR Integration Mapping

This document maps MARBLE's architecture to SMTR integration points, documenting the actual API signatures and how each component connects.

## 1. Component Mapping

| MARBLE Component | SMTR Equivalent | Integration File | Strategy |
|---|---|---|---|
| `marble.agent.BaseAgent` | `SMTRTauAgent` | `src/smtr/runtime/marble_agent.py` | `PromptAwareBaseAgent` + `SMTRMarbleAgent` subclass |
| `marble.memory.BaseMemory` (private) | Payload injection target | `src/smtr/runtime/marble_agent.py` | Override `act()` to inject via `_augment_with_private_guidance()` |
| `marble.engine.Engine` | `Tau3PairedRolloutRunner` | `src/smtr/runtime/marble_agent.py` | `SMTRMarbleEngine` subclass + `MarblePairedRolloutRunner` |
| `marble.graph.AgentGraph` | Trace metadata only | N/A | Record topology in trace, NOT in `ContextFingerprint` |
| `marble.evaluator.Evaluator` | `TauOutcome` | `src/smtr/counterfactual/marble_eval.py` | Map milestone metrics → binary success |

## 2. MARBLE BaseAgent API (actual signatures)

### Constructor

```python
BaseAgent.__init__(
    config: Dict,
    env: EnvType,
    shared_memory: Union[SharedMemory, None] = None,
    model: str = "gpt-3.5-turbo",
)
```

**Critical detail**: `BaseAgent.__init__` creates its own `self.memory = BaseMemory()` and `self.shared_memory = SharedMemory()` internally. Any `shared_memory` parameter is ignored. This means SMTR must NOT pass shared_memory — it would have no effect.

### Key methods overridden by SMTR

- **`act(task: str) -> Any`**: Builds prompt inline (lines 201-210 of `base_agent.py`), calls `model_prompting()`, handles tool calls including `new_communication_session`.
- **`_handle_new_communication_session(...)`**: Drives communication loop with `agents[t % 2]` alternating speakers. Constructed per-speaker `communicate_task` prompts.

### What is NOT in BaseAgent

MARBLE's `BaseAgent` does NOT have:
- `_build_base_prompt()`
- `_act_with_augmented_prompt()`
- `_build_communication_prompt()`

These were fictional placeholders in early plan drafts. The actual override strategy copies the real `act()` and `_handle_new_communication_session()` logic and inserts `_augment_with_private_guidance()` calls.

## 3. MARBLE Engine API (actual signatures)

### Constructor

```python
Engine.__init__(config: Config)
```

Initializes: environment → agents → graph → memory → evaluator (in that order).

### Agent initialization

```python
Engine._initialize_agents(agent_configs: List[Dict]) -> List[BaseAgent]
```

Receives `list[dict]`, returns `list[BaseAgent]`. Uses `self.config.llm` as default model.

**SMTR override**: `SMTRMarbleEngine._initialize_agents()` instantiates `SMTRMarbleAgent` for the target receiver and `PromptAwareBaseAgent` for all other agents.

### DB evaluation

```python
Evaluator.evaluate_task_db(task, summary, labels, number_of_labels_pred, root_causes)
```

Stores `metrics["task_evaluation"] = {'root_cause': root_causes, 'predicted': result}` — no LLM call, direct comparison.

## 4. Prompt Injection Architecture

### Why private prompt injection (not SharedMemory)

MARBLE's `BaseAgent.act()` reads from `self.memory` (private), not `self.shared_memory`. Even if we write to SharedMemory, the LLM won't see it during `act()`. SMTR's method constraint requires payload exposure only to the receiver agent's private context.

### PromptAwareBaseAgent for ALL agents

MARBLE's `_handle_new_communication_session()` is driven by the *initiating* agent in a loop — it is NOT called per-agent. If only `SMTRMarbleAgent` overrides it:
- When target initiates: target's payload may leak into the other agent's communication prompt
- When others initiate: target's payload never enters its own communication prompt

**Solution**: `PromptAwareBaseAgent` base class for ALL agents with `render_private_guidance()` hook (returns `""` by default). The communication handler uses `session_current_agent.render_private_guidance()` to inject guidance for the actual speaking agent only.

### Injection points

1. **`act()` prompt**: `_augment_with_private_guidance(prompt)` before `model_prompting()`
2. **Communication prompts**: Per-speaker `communicate_task` augmented with `session_current_agent.render_private_guidance()`
3. **No other LLM call sites** in the current MARBLE codebase

### Information barrier — what must NOT appear

- Routing cards (only payloads)
- Critic's (τ̂, η̂) estimates
- LCB/UCB values
- Other agents' private payloads
- Evaluator / gold labels

## 5. Causal Control via exposure_override

Both share and withhold branches use the same `SMTRMarbleAgent` subclass. The difference is controlled by `exposure_override`:

| Value | Behavior |
|---|---|
| `None` | Run router normally → selects S_K |
| `["m1", "m2"]` | Force that specific set |
| `[]` | Force S_K=∅ (empty) |

This prevents the withhold branch's router from re-selecting memories.

## 6. Evaluation Bridge

### DB environment evaluator output

```python
metrics["task_evaluation"] = {
    'root_cause': ['lock_contention', 'missing_index'],  # gold labels
    'predicted': 'The issue is caused by lock_contention...',  # agent's answer
}
```

### Success heuristic (current)

Check if any root cause label appears as substring in the prediction. This is a simple heuristic to be refined after Task 1c confirms the actual output structure.

### Files

- `src/smtr/counterfactual/marble_eval.py`: `MarbleOutcome`, `extract_marble_outcome()`, `_check_db_success()`
- `src/smtr/counterfactual/marble_paired_rollout.py`: `MarbleBranchResult`, `MarblePairedOutcome`, `MarblePairedRolloutRunner`

## 7. DB Environment Configuration

MARBLE provides pre-configured YAML configs for DB scenarios:
- Location: `MARBLE/marble/configs/test_config_database/`
- Format: `{LLM_MODEL}_{DOMAIN}_{ANOMALY_TYPE}.yaml`
- Example: `gpt-3.5-turbo_E_COMMERCE_LOCK_CONTENTION.yaml`

Config structure includes: `coordinate_mode`, `relationships`, `llm`, `environment` (type: DB, init_sql, anomalies), `task` (content, output_format, labels, root_causes, number_of_labels_pred), `agents` (5 agents with profiles), `memory`, `metrics`, `output`, `engine_planner`.

**Runtime dependency**: Docker (PostgreSQL + Prometheus). Without Docker, DB environment cannot run.

## 8. File Inventory

| File | Purpose | Lines |
|---|---|---|
| `src/smtr/runtime/marble_agent.py` | PromptAwareBaseAgent + SMTRMarbleAgent + SMTRMarbleEngine + data models | ~785 |
| `src/smtr/counterfactual/marble_eval.py` | Evaluation bridge: MARBLE DB → SMTR outcomes | ~135 |
| `src/smtr/counterfactual/marble_paired_rollout.py` | Single-receiver set-level paired evaluation | ~257 |
| `tests/test_marble_agent.py` | 32 tests (23 always-run + 9 MARBLE-dependent) | ~369 |

## 9. Environment Setup

- MARBLE cloned to `/home/ecs-user/MARBLE/`
- MARBLE's `pyproject.toml` relaxed from `python <3.12` to `<3.13` for Python 3.12 compatibility
- Both MARBLE + SMTR installed in `/home/ecs-user/MARBLE/.venv` (Python 3.12)
- MARBLE's `evaluator.py` had a syntax error fixed (misplaced except/else in `parse_research_ratings`)
- Additional dependency `ruamel.yaml` installed (not in MARBLE's pyproject.toml)
