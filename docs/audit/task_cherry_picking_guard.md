# Task Cherry-Picking Guard

**Date**: 2026-08-24

**Status**: ✅ ESTABLISHED


## 1. Principle

**Formal experiments MUST use the official MultiAgentBench task pool.**

Cherry-picking tasks (e.g., only hard tasks, only memory-sensitive tasks) is
**prohibited** for main results.


## 2. Allowed vs Prohibited

### ✅ Allowed

| Use Case | Task Selection | Status |
|----------|---------------|--------|
| **Pilot runs** | Hard subset (e.g., tasks 51-55) | ✅ OK for pipeline validation |
| **Smoke tests** | Random 5-10 tasks | ✅ OK for quick checks |
| **Debugging** | Specific failing tasks | ✅ OK for root cause analysis |
| **Main experiments** | **Full official pool** (100 tasks × 5 scenarios) | ✅ REQUIRED |

### ❌ Prohibited

| Use Case | Task Selection | Status |
|----------|---------------|--------|
| Main table | Only hard tasks (e.g., 51-100) | ❌ CHERRY-PICKING |
| Main table | Only memory-sensitive tasks | ❌ CHERRY-PICKING |
| Main table | Custom hard tasks (not in official pool) | ❌ CHERRY-PICKING |
| Main table | Tasks selected based on SMTR performance | ❌ CHERRY-PICKING |


## 3. Official Task Pool

| Scenario | JSONL File | Task Count | Task IDs |
|----------|-----------|------------|----------|
| bargaining | `multiagentbench/bargaining/bargaining_main.jsonl` | 100 | 1-100 |
| coding | `multiagentbench/coding/coding_main.jsonl` | 100 | 1-100 |
| database | `multiagentbench/database/database_main.jsonl` | 100 | 1-100 |
| minecraft | `multiagentbench/minecraft/minecraft_main.jsonl` | 100 | 1-100 |
| research | `multiagentbench/research/research_main.jsonl` | 100 | 1-100 |

**Total**: 500 official tasks


## 4. SMTR Current Usage

| Experiment | Task Selection | Status |
|-----------|---------------|--------|
| Difficulty profiling | First 20 per scenario (100 total) | ⚠️ PARTIAL (20% of pool) |
| TCI smoke | First 10 per scenario (50 total) | ⚠️ PARTIAL (10% of pool) |
| Main experiment (planned) | **Full pool** (500 total) | ✅ CORRECT |

**Recommendation**: For main experiments, use all 100 tasks per scenario.
If computational cost is prohibitive, use at least 50 tasks per scenario (250 total)
and report the subset explicitly.


## 5. Reporting Requirements

### In Paper

```markdown
## Experimental Setup

We evaluate on the official MultiAgentBench task pool (Zhu et al., 2025),
which contains 100 tasks per scenario across 5 domains (500 total).

For computational feasibility, we use a stratified sample of N tasks per
scenario (N=XX, total=XXX), selected as the first N tasks from each
official JSONL file. This selection is independent of model performance
and avoids cherry-picking.
```

### In Appendix

```markdown
## Task Selection

All tasks are selected from the official multiagentbench JSONL files:
- multiagentbench/bargaining/bargaining_main.jsonl (tasks 1-N)
- multiagentbench/coding/coding_main.jsonl (tasks 1-N)
- multiagentbench/database/database_main.jsonl (tasks 1-N)
- multiagentbench/minecraft/minecraft_main.jsonl (tasks 1-N)
- multiagentbench/research/research_main.jsonl (tasks 1-N)

No tasks were excluded based on model performance or memory sensitivity.
```


## 6. Pilot vs Main Results

### Pilot (Allowed to use hard subset)

**Purpose**: Validate pipeline mechanics (TCI delta computation, memory injection, etc.)

**Task selection**: Hard subset (e.g., tasks 51-100 with dual root causes)

**Reporting**:
```markdown
## Pilot Validation

We validate the SMTR pipeline on a hard subset (tasks 51-55) to ensure
TCI delta computation works correctly. This is a pilot study and NOT
the main result.
```

### Main Results (Must use full pool)

**Purpose**: Report final performance comparison

**Task selection**: Full official pool (or stratified sample ≥50 per scenario)

**Reporting**:
```markdown
## Main Results

We evaluate all methods on the official MultiAgentBench task pool
(100 tasks × 5 scenarios = 500 total), using 3 seeds per task.
```


## 7. Anti-Cherry-Picking Checklist

Before reporting main results, verify:

- [ ] Tasks selected **before** running SMTR
- [ ] Task selection independent of model performance
- [ ] No tasks excluded based on SMTR delta sign
- [ ] No custom tasks added (only official pool)
- [ ] Task IDs reported in appendix
- [ ] Full pool used (or stratified sample with justification)


## 8. Conclusion

**Rule**: Main experiments use official task pool. Pilots can use subsets.

**Rationale**: Reviewers will reject cherry-picked results. Only full-pool
(or representative sample) results are scientifically valid.
