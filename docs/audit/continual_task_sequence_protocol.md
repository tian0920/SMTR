# Continual Task Sequence Protocol

**Date**: 2026-08-24
**Status**: DRAFT (pending Phase O config freeze)
**Replaces**: Supervised-style 80/20 train/test split

---

## 1. Rationale

SMTR is **not** a classifier training pipeline. The 80/20 supervised split is
inappropriate because:

1. Memory is accumulated online, not trained offline
2. Task order matters for memory lifecycle (validate → reuse → update)
3. Test tasks should not be isolated from training context
4. The causal intervention (TCI) requires sequential memory persistence

---

## 2. Protocol Definition

```
For each scenario s ∈ SCENARIOS_INCLUDED:
    memory_bank ← ∅
    
    For each task t in deterministic_order(s, seed):
        1. Record: memory_size_before = |memory_bank|
        
        2. Execute episode:
           - Run no_memory / full_memory / smtr_* with same task
           - Record: team_task_score (official metric)
        
        3. If method uses TCI:
           - Propose candidate memories
           - For each candidate m:
             - Run expose/withhold branches
             - Compute delta(m, r) for each receiver r
             - Admit if delta > 0 (any receiver)
           - Record: n_validated, n_rejected
        
        4. Update memory_bank with validated memories
        
        5. Record:
           - task_position (0-indexed position in sequence)
           - task_id
           - seed
           - memory_size_after = |memory_bank|
           - retrieved_memory_count (memories used in this episode)
           - team_task_score (official normalized score)
    
    Reset: memory_bank ← ∅  (between scenarios)
```

---

## 3. Key Properties

### 3.1 Within Scenario: Memory Persists
- Validated memories from task t are available for task t+1, t+2, ...
- This enables cross-task memory reuse (the core TCI mechanism)

### 3.2 Between Scenarios: Memory Resets
- Each scenario is an independent "lifetime"
- Prevents scenario-specific knowledge from contaminating other scenarios
- Matches the MultiAgentBench design (scenarios are independent domains)

### 3.3 Task Order: Deterministic Per Seed
- Same seed → same task order → reproducible results
- Different seeds → different orders → tests order-independence
- Implementation: `hash(scenario + seed)` → shuffle official task pool

### 3.4 No Task Difficulty Filtering
- All official tasks are included regardless of baseline performance
- No cherry-picking based on team_success or official score
- Floor/ceiling effects are analyzed post-hoc (Phase E)

---

## 4. Deterministic Task Ordering

```python
import hashlib

def generate_task_order(scenario: str, seed: int, task_ids: list[str]) -> list[str]:
    """Generate deterministic task ordering for a scenario/seed pair."""
    digest = hashlib.sha256(f"{scenario}:{seed}".encode()).hexdigest()
    # Use first 8 hex chars as sort key prefix
    keyed = [(digest[:8] + tid, tid) for tid in task_ids]
    keyed.sort()
    return [tid for _, tid in keyed]
```

---

## 5. Recorded Fields Per Episode

| Field | Type | Description |
|-------|------|-------------|
| `task_position` | int | 0-indexed position in continual sequence |
| `task_id` | str | Official MultiAgentBench task identifier |
| `seed` | int | Generation seed (0, 1, 2, 3, 4) |
| `scenario` | str | Domain name |
| `method` | str | no_memory / full_memory / smtr_uniform / smtr_receiver |
| `memory_size_before` | int | Memories in bank before this episode |
| `memory_size_after` | int | Memories in bank after this episode |
| `retrieved_memory_count` | int | Memories retrieved/used in this episode |
| `team_task_score` | float | Official normalized Task Score ∈ [0, 1] |
| `evaluator_valid` | bool | Whether official metric was successfully computed |

---

## 6. Prohibited Practices

- ❌ Task difficulty filtering (pre-selecting "medium" tasks)
- ❌ 80/20 or any train/test split
- ❌ Shuffling between methods (all methods see same task order)
- ❌ Cross-scenario memory persistence
- ❌ Task exclusion based on baseline performance

---

## 7. Comparison With Old Protocol

| Aspect | Old Protocol | New Protocol |
|--------|-------------|-------------|
| Task split | 80 train / 20 test | All tasks sequential |
| Memory scope | Per-episode | Within-scenario persistent |
| Task order | Random per episode | Deterministic per seed |
| Evaluation | Binary team_success | Official normalized Task Score |
| Memory lifecycle | No reuse tracking | Full lifecycle (validate → reuse → update) |

---

## 8. Audit Trail

| Check | Status |
|-------|--------|
| Continual sequence replaces train/test split | ✅ DEFINED |
| Memory persists within scenario | ✅ DEFINED |
| Memory resets between scenarios | ✅ DEFINED |
| Deterministic task order per seed | ✅ DEFINED |
| No task difficulty filtering | ✅ ENFORCED |
| Same task order for all methods | ✅ ENFORCED |
