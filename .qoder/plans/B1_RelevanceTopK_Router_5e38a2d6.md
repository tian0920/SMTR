# B1 RelevanceTopK Router Implementation

## Exploration Summary

**Key findings:**
- `MemoryRouter` protocol ([interfaces.py](file:///home/ecs-user/SMTR/src/smtr/router/interfaces.py#L30-L49)) requires `decide_from_proposal()` and `decide()` methods
- `RoutingResult` is defined in [baseline_router.py](file:///home/ecs-user/SMTR/src/smtr/router/baseline_router.py#L8-L16) — shared by all routers
- `RouterDecision.decision_source` is a `Literal` type that needs extending for B1 ([traces.py](file:///home/ecs-user/SMTR/src/smtr/router/traces.py#L40-L46))
- `build_graph()` already accepts `router` param, defaults to `NoMemoryRouter()` ([graph.py](file:///home/ecs-user/SMTR/src/smtr/runtime/graph.py#L157-L170))
- `run_demo()`, `run_episode()`, `run_demo_with_repository()` already accept `router` param (recently added)
- CLI `demo` command has `--seed`, `--db`, `--top-k` but no router mode ([cli.py](file:///home/ecs-user/SMTR/src/smtr/cli.py#L852-L855))
- No existing router factory — needs to be created

---

## Task 1: Extend `RouterDecision.decision_source` Literal

**File:** `src/smtr/router/traces.py`

Add `"relevance_topk_router"` to the `decision_source` Literal type (line 40-46):
```python
decision_source: Literal[
    "fixed_prefix",
    "forced_intervention",
    "frozen_continuation",
    "baseline_router",
    "production_router",
    "relevance_topk_router",  # NEW
] = "baseline_router"
```

---

## Task 2: Implement `RelevanceTopKRouter`

**File:** `src/smtr/router/baselines.py` (new file)

```python
@dataclass(frozen=True)
class RelevanceTopKRouterConfig:
    max_shares_per_invocation: int | None = None

class RelevanceTopKRouter:
    router_name = "RelevanceTopKRouter"
    router_version = "1"
    
    def decide_from_proposal(...) -> RoutingResult:
        # Select top-k candidates by proposer ranking (no critic)
        # Respect max_shares_per_invocation budget
        # Return RoutingResult with decisions
        
    def decide(...) -> tuple[list[RouterDecision], list[str]]:
        # Legacy interface compatibility
```

Key logic:
- `selected = proposal.ranked_candidates[:share_limit]`
- `share_limit = min(len(candidates), config.max_shares_per_invocation)` if configured, else all
- Each selected candidate gets `action="share"`, `reason="relevance_topk_selected"`
- Each exceeded candidate gets `action="withhold"`, `reason="relevance_topk_budget_exceeded"`
- `traversal_order` = proposer ranking order (no shuffle)
- Critic fields (tau_mean, etc.) = `None`

---

## Task 3: Implement `build_router()` factory

**File:** `src/smtr/router/factory.py` (new file)

```python
def build_router(
    mode: Literal["no-memory", "relevance-topk", "learned"],
    *,
    critic_checkpoint: str | Path | None = None,
    max_shares_per_invocation: int | None = None,
    critic_config: SequentialRouterConfig | None = None,
) -> MemoryRouter:
    if mode == "no-memory":
        return NoMemoryRouter()
    elif mode == "relevance-topk":
        return RelevanceTopKRouter(config=RelevanceTopKRouterConfig(
            max_shares_per_invocation=max_shares_per_invocation
        ))
    elif mode == "learned":
        if critic_checkpoint is None:
            raise ValueError("learned mode requires critic_checkpoint")
        critic = FourOutcomeTransferCritic.load(Path(critic_checkpoint))
        return ProductionSequentialRouter(
            critic=critic,
            config=critic_config or SequentialRouterConfig(),
        )
    else:
        raise ValueError(f"unknown router mode: {mode}")
```

---

## Task 4: Write Tests (TDD)

**File:** `tests/test_relevance_topk_router.py` (new file)

### Test 1: B1 selects by relevance rank
- Given candidates `[m3(0.9), m1(0.8), m2(0.4)]`, `max_shares=2`
- Assert selected = `[m3, m1]`, m2 withheld

### Test 2: B1 does not call critic
- Use a fake critic that raises on any call
- Assert B1 completes without accessing critic

### Test 3: B1 trace correctness
- Verify `router_name`, candidate order, selected IDs, proposal ranks, decision reasons

### Test 4: Payload isolation
- Verify B1 only touches routing cards, only selected payloads enter `visible_payloads`

### Test 5: Budget boundary cases
- `max_shares=0` → all withheld
- `max_shares=1` → only top-1 shared
- `max_shares > candidate_count` → all shared
- `max_shares=None` → all shared
- Empty proposal → no decisions

### Test 6: Factory constructs all three modes
- `build_router("no-memory")` → `NoMemoryRouter`
- `build_router("relevance-topk")` → `RelevanceTopKRouter`
- `build_router("learned", critic_checkpoint=...)` → `ProductionSequentialRouter`
- `build_router("learned")` without checkpoint → raises
- `build_router("unknown")` → raises

### Test 7: Regression — existing tests pass
- Run full test suite to verify no regressions

---

## Task 5: Update CLI

**File:** `src/smtr/cli.py`

### 5a: Add `--router-mode` and `--critic-checkpoint` to `demo` parser (line 852-855)
```python
demo_parser.add_argument("--router-mode", choices=["no-memory", "relevance-topk", "learned"], default="no-memory")
demo_parser.add_argument("--critic-checkpoint")
demo_parser.add_argument("--max-shares-per-invocation", type=int)
```

### 5b: Update `_demo()` function (line 100-121)
```python
def _demo(seed, db, top_k, router_mode, critic_checkpoint, max_shares_per_invocation):
    router = build_router(
        mode=router_mode,
        critic_checkpoint=critic_checkpoint,
        max_shares_per_invocation=max_shares_per_invocation,
    )
    # Pass router to run_demo / run_demo_with_repository
```

### 5c: Add same args to `demo-real` parser (line 979-983)

---

## Task 6: Update `router/__init__.py`

**File:** `src/smtr/router/__init__.py`

Export new classes:
```python
from smtr.router.baselines import RelevanceTopKRouter, RelevanceTopKRouterConfig
from smtr.router.factory import build_router
```

---

## Task 7: Run full test suite and verify

```bash
python3 -m pytest tests/test_relevance_topk_router.py -v
python3 -m pytest tests/ -x -q  # full regression
ruff check src/smtr/router/baselines.py src/smtr/router/factory.py
```

---

## Dependencies

```
Task 1 (extend Literal) → Task 2 (implement router)
Task 2 → Task 4 (tests)
Task 3 (factory) → Task 5 (CLI)
Task 1 + Task 2 + Task 3 → Task 6 (__init__.py)
Task 4 + Task 5 + Task 6 → Task 7 (verify)
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| `decision_source` Literal change breaks existing tests | Additive change only; existing values unchanged |
| B1 accidentally calls critic | Test 2 uses error-raising fake critic |
| Factory `learned` mode silently falls back | Explicit `ValueError` when checkpoint missing |
| CLI backward compatibility | Default `--router-mode no-memory` preserves existing behavior |
| `RoutingResult` frozen model serialization | All new fields already optional/None in `RouterDecision` |

---

## Critical Files

1. `src/smtr/router/baselines.py` — new B1 router implementation
2. `src/smtr/router/factory.py` — unified router factory
3. `src/smtr/router/traces.py` — extend `decision_source` Literal
4. `src/smtr/cli.py` — CLI integration
5. `tests/test_relevance_topk_router.py` — comprehensive tests

---

## Rejected Alternatives

1. **Adding B1 to `baseline_router.py`**: Rejected because `baseline_router.py` is specifically for `NoMemoryRouter` and `RoutingResult`. A new file `baselines.py` provides cleaner separation and room for future baselines.

2. **Embedding factory in `graph.py`**: Rejected because `graph.py` is runtime orchestration. Factory belongs in `router/factory.py` to maintain single responsibility.

3. **Using `dataclass` for `RelevanceTopKRouterConfig`**: Considered Pydantic `BaseModel` for consistency with `SequentialRouterConfig`, but `dataclass(frozen=True)` is simpler and sufficient since no validation beyond type is needed. Final decision: use `dataclass` per spec suggestion.
