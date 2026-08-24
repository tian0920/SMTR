# Scenario Coverage Decision

**Date**: 2026-08-24
**Status**: FROZEN
**Decision Authority**: Protocol audit, not data-driven

---

## 1. Official MultiAgentBench Scenarios (ACL 2025)

The MultiAgentBench benchmark (ulab-uiuc/MARBLE, ACL 2025) contains exactly
**5 scenarios**, each with **100 official tasks**:

| # | Scenario   | Tasks | Env Class            | Official Metric                           |
|---|------------|-------|----------------------|-------------------------------------------|
| 1 | database   | 100   | DBEnvironment        | root_cause_recall ∈ [0, 1]                |
| 2 | research   | 100   | ResearchEnvironment  | avg(innovation, safety, feasibility) ∈ [1,5] |
| 3 | minecraft  | 100   | MinecraftEnvironment | block_hit_rate ∈ [0, 1]                   |
| 4 | coding     | 100   | CodingEnvironment    | avg(instruction, executability, consistency, quality) ∈ [1,5] |
| 5 | bargaining | 100   | WorldSimulationEnv   | avg(2 roles × 3 aspects) ∈ [1,5]          |

**Source**: `multiagentbench/{scenario}/{scenario}_main.jsonl`

**Total**: 500 official tasks

---

## 2. Scenarios Included

```python
SCENARIOS_INCLUDED = [
    "bargaining",
    "coding",
    "database",
    "minecraft",
    "research",
]
```

**Rationale**: These are ALL official MultiAgentBench scenarios.
No cherry-picking — we include the complete benchmark.

---

## 3. Scenarios Excluded

| Scenario  | Source          | Exclusion Reason |
|-----------|-----------------|------------------|
| werewolf  | MARBLE env only | Not part of MultiAgentBench benchmark. No official task pool (no JSONL). Adversarial/social deduction objective differs fundamentally from cooperative procedural memory evaluation. No stable task-evaluation outcome suitable for causal intervention. |
| web       | MARBLE env only | Not part of MultiAgentBench benchmark. No official task pool. Web navigation is a distinct problem domain requiring different evaluation methodology. |

---

## 4. Known Evaluator Limitations

| Scenario   | Issue | Impact | Mitigation |
|------------|-------|--------|------------|
| coding     | MARBLE engine reads `MARBLE/marble/workspace/solution.py` (relative path). Agents don't write to this path. Evaluator always fails. | All coding episodes → evaluator_failure | Documented; count as INVALID. If >5% failure rate, scenario flagged in Phase F. |
| bargaining | Evaluator LLM sometimes returns unparseable JSON → sentinel -1 values → out-of-range normalized scores | Some episodes → INVALID_OUTCOME | Sentinel detection added; invalid episodes excluded from analysis. |
| research   | Evaluator LLM sometimes fails to parse ratings → empty dict | Some episodes → INVALID_OUTCOME | Empty dict detection added; invalid episodes excluded. |

---

## 5. Frozen Lists

```python
SCENARIOS_INCLUDED = ["bargaining", "coding", "database", "minecraft", "research"]
SCENARIOS_EXCLUDED = ["werewolf", "web"]

EXCLUSION_REASONS = {
    "werewolf": "not_in_benchmark; adversarial_objective; no_official_task_pool",
    "web": "not_in_benchmark; no_official_task_pool; different_evaluation_methodology",
}
```

---

## 6. Prohibited Criteria

The following criteria are **explicitly prohibited** for scenario inclusion/exclusion:

- SMTR performance on the scenario
- Baseline success rate being "too high" or "too low"
- Convenience or computational cost
- Researcher preference based on preliminary results

---

## 7. Audit Trail

| Check | Status |
|-------|--------|
| All official MultiAgentBench scenarios included | ✅ PASS |
| No scenarios excluded based on SMTR performance | ✅ PASS |
| Exclusion reasons are methodological, not empirical | ✅ PASS |
| Frozen lists documented before full experiment | ✅ PASS |
