# Reward Pipeline Audit

> Traces the reward computation from agent action to final metric.

---

## 1. Where Does Reward Come From?

### Reward Chain
```
Agent LLM call → SQL query → PostgreSQL execution → 
  Native evaluator (marble_database_evaluate_task_db) → 
  root_cause match → team_success (bool)
```

### Evaluator Details

| Field | Value |
|-------|-------|
| Evaluator name | `marble_database_evaluate_task_db` |
| Type | Rule-based label matching |
| Logic | Compare predicted root causes vs ground-truth labels |
| Output | `success: bool`, `score: float`, `fine_grained: {f1, precision, recall, tp, fp}` |

### Example (task 10, seed 2):
```json
{
  "expected_labels": ["INSERT_LARGE_DATA"],
  "predicted_labels": ["FETCH_LARGE_DATA", "INSERT_LARGE_DATA", "LOCK_CONTENTION", "REDUNDANT_INDEX", "VACUUM"],
  "tp": 1, "fp": 4,
  "f1": 0.3333, "precision": 0.2, "recall": 1.0,
  "success": true
}
```

---

## 2. Is Reward from MARBLE Environment?

**YES**. The reward is determined by the MARBLE native evaluator running on the PostgreSQL database environment.

Evidence:
- `share_audit.json` → `outcome.evaluator_name = "marble_database_evaluate_task_db"`
- `share_audit.json` → `outcome.native_evaluator_executed = true`
- `marble_output.jsonl` contains actual SQL query results from PostgreSQL

---

## 3. How Are Paired Record Outcomes Computed?

Each paired record has two branches:
- **share**: memory injected → LLM runs → evaluator → `share.team_success`
- **withhold**: no memory → LLM runs → evaluator → `withhold.team_success`

Treatment effect: `tau = int(share.team_success) - int(withhold.team_success)`

Labels:
- `positive_transfer`: tau > 0 (share succeeded, withhold failed)
- `negative_transfer`: tau < 0 (share failed, withhold succeeded)
- `neutral_success`: both succeeded
- `neutral_failure`: both failed

---

## 4. How Are Method Rewards Computed?

The offline evaluation (`run_marble_baselines.py`) computes:

```python
method_reward = withhold_mean + sum(tau for selected memories)
```

Where:
- `withhold_mean` = mean of `withhold.team_success` across all candidates in the group
- `sum(tau)` = sum of treatment effects for memories selected by the method policy

---

## 5. Artificial/Cached/Heuristic Reward Check

| Check | Result | Evidence |
|-------|--------|----------|
| Artificial reward | **NOT DETECTED** | Reward comes from real evaluator |
| Lookup table | **NOT DETECTED** | Each outcome is per-edge, per-seed |
| Cached reward | **PARTIALLY** | Outcomes stored in paired records, but originally computed by real evaluator |
| Heuristic score | **NOT DETECTED** | Binary success/failure from label matching |

---

## 6. CRITICAL: Offline Evaluation Layer

The baseline experiment (Task A main results) does **NOT** re-run MARBLE. It uses:

1. Pre-computed paired records (642 valid)
2. Simulates method policies on top
3. Looks up existing outcomes

This means:
- **The underlying outcomes ARE real** (from MARBLE engine)
- **The method selection is simulated** (not re-executed)
- **SMTR-TCI uses ground-truth labels** as TCI proxy (idealized upper bound)

---

## Conclusion

**PASS**: Reward comes from real MARBLE native evaluator on real PostgreSQL database.

**WARNING**: Baseline experiment is offline simulation, not re-execution. SMTR-TCI uses idealized TCI proxy.
