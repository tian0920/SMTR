# MARBLE Baseline-Ready Interface

> Confirms that all baseline memory controllers implement the frozen
> `BaseMemoryController` interface and can be used directly by the
> MARBLE adapter without code changes.

---

## Interface Compliance

All four baseline controllers inherit from `BaseMemoryController` and
implement the four required abstract methods:

| Controller | Class | extract_memory | update_memory | retrieve_memory | get_statistics |
|-----------|-------|:-:|:-:|:-:|:-:|
| Reflexion | `ReflexionController` | ✓ | ✓ | ✓ | ✓ |
| AGILE | `AgileController` | ✓ | ✓ | ✓ | ✓ |
| Heuristic | `HeuristicMemoryController` | ✓ | ✓ | ✓ | ✓ |
| AgeMem | `AgeMemController` | ✓ | ✓ | ✓ | ✓ |

---

## MARBLE Adapter Integration Points

The MARBLE adapter can call the baseline controllers via:

```python
from smtr.baselines import BaseMemoryController
from smtr.baselines.reflexion import ReflexionController
from smtr.baselines.agile import AgileController
from smtr.baselines.heuristic_memory import HeuristicMemoryController
from smtr.baselines.agemem import AgeMemController

# Instantiate any controller
controller: BaseMemoryController = ReflexionController(top_k=3)

# Standard interface — same for all controllers
candidates = controller.extract_memory(trajectory)
for cand in candidates:
    decision = controller.update_memory(cand, context)
memories = controller.retrieve_memory(query)
stats = controller.get_statistics()
```

---

## Components Requiring No Modification

The following MARBLE components can use baselines without changes:

1. **Memory extraction pipeline** — `extract_memory(trajectory)` accepts
   the same trajectory dict format used by the MARBLE episode runner.

2. **Memory admission** — `update_memory(candidate, context)` returns
   `"store"` / `"discard"` / `"modify"`, which maps directly to the
   MARBLE admission decision.

3. **Memory retrieval** — `retrieve_memory(query)` accepts a `MemoryQuery`
   with topic, episode, and top_k — all available from the MARBLE task context.

4. **Statistics logging** — `get_statistics()` returns a flat dict that can
   be written directly to the MARBLE experiment log.

5. **Baseline policy adapter** — `experiments/lifelong/baseline_policies.py`
   provides the `LifelongPolicy` wrapper that bridges the baseline interface
   to the experiment harness.

---

## MARBLE Experiment Configuration

To use baselines in MARBLE experiments:

```yaml
# In MARBLE experiment config
memory_controller:
  type: reflexion  # or agile, heuristic, agemem
  top_k: 3
  budget: null     # or int for budget-constrained runs
```

The MARBLE adapter reads `memory_controller.type` and instantiates the
corresponding `BaseMemoryController` subclass.

---

## Testing Checklist

Before running MARBLE experiments with baselines:

- [x] All controllers implement `BaseMemoryController` (verified via `isinstance` checks)
- [x] All controllers pass synthetic lifelong smoke tests (5 methods × 100 episodes × 5 seeds)
- [x] Memory ID mapping is consistent (env IDs used throughout)
- [x] No baseline uses TCI, LLM calls, or gradient updates
- [x] Fairness audit: 6/6 checks passed

---

## File Locations

| Component | Path |
|-----------|------|
| Base interface | `src/smtr/baselines/base_memory_controller.py` |
| Reflexion | `src/smtr/baselines/reflexion/reflexion_controller.py` |
| AGILE | `src/smtr/baselines/agile/agile_controller.py` |
| Heuristic | `src/smtr/baselines/heuristic_memory/heuristic_controller.py` |
| AgeMem | `src/smtr/baselines/agemem/agemem_controller.py` |
| Policy adapter | `experiments/lifelong/baseline_policies.py` |
| Config | `configs/baseline_comparison.yaml` |
