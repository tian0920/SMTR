# MARBLE Outcome Provenance Audit (Phase 4)

**Date**: 2026-08-24

**Purpose**: Classify all candidate native performance signals by provenance safety.


## Signal Provenance Checklist

### 1. team_success (Binary)

| Question | Answer |
|----------|--------|
| When produced? | After engine simulation completes |
| Who produces it? | SMTR trajectory_collector (inferred from iteration presence) |
| Immediately available? | Yes |
| Needs task ground truth? | No |
| Needs hidden answer? | No |
| Uses future episode info? | No |
| Needs human labels? | No |
| Usable in expose/withhold? | Yes |

**Provenance**: ✅ **SAFE**


### 2. team_success (Native — MARBLE evaluator)

| Question | Answer |
|----------|--------|
| When produced? | Not produced (no native team_success field in output) |
| Who produces it? | N/A |

**Provenance**: ❌ **INVALID** (field doesn't exist in MARBLE output)


### 3. token_usage (Continuous)

| Question | Answer |
|----------|--------|
| When produced? | During engine simulation (cumulative) |
| Who produces it? | MARBLE Evaluator.update() |
| Immediately available? | Yes (in output JSONL) |
| Needs task ground truth? | No |
| Needs hidden answer? | No |
| Uses future episode info? | No |
| Needs human labels? | No |
| Usable in expose/withhold? | Yes |

**Provenance**: ✅ **SAFE** — But not a quality signal (measures effort, not correctness)


### 4. iteration_count (Discrete)

| Question | Answer |
|----------|--------|
| When produced? | During engine simulation |
| Who produces it? | MARBLE Engine loop counter |
| Immediately available? | Yes |
| Needs task ground truth? | No |
| Needs hidden answer? | No |
| Uses future episode info? | No |
| Needs human labels? | No |
| Usable in expose/withhold? | Yes |

**Provenance**: ✅ **SAFE** — Proxy for task difficulty/convergence speed


### 5. summary_length (Continuous proxy)

| Question | Answer |
|----------|--------|
| When produced? | Each iteration (Planner summarize_output) |
| Who produces it? | MARBLE Planner LLM call |
| Immediately available? | Yes (in iteration data) |
| Needs task ground truth? | No |
| Needs hidden answer? | No |
| Uses future episode info? | No |
| Needs human labels? | No |
| Usable in expose/withhold? | Yes |

**Provenance**: ✅ **SAFE** — Very noisy proxy for progress


### 6. task_evaluation (Research domain — LLM-judged)

| Question | Answer |
|----------|--------|
| When produced? | After engine loop completes (final evaluator) |
| Who produces it? | MARBLE Evaluator.evaluate_task_research() (LLM-judged) |
| Immediately available? | ⚠️ YES if evaluator doesn't crash |
| Needs task ground truth? | No (evaluates idea quality, not correctness) |
| Needs hidden answer? | No |
| Uses future episode info? | No |
| Needs human labels? | No |
| Usable in expose/withhold? | Yes |

**Provenance**: ⚠️ **CONDITIONAL** — Evaluator template crashes with BeartypeCallHintReturnViolation in practice. When it works: SAFE (LLM-as-judge, no ground truth).


### 7. task_evaluation (Minecraft — block_hit_rate)

| Question | Answer |
|----------|--------|
| When produced? | After engine loop (read from file) |
| Who produces it? | Minecraft environment (rule-based) |
| Immediately available? | Yes (file I/O) |
| Needs task ground truth? | No (measures block placement accuracy) |
| Needs hidden answer? | No |
| Uses future episode info? | No |
| Needs human labels? | No |
| Usable in expose/withhold? | Yes |

**Provenance**: ✅ **SAFE** — Rule-based, no ground truth required


### 8. task_evaluation (Database — root_cause match)

| Question | Answer |
|----------|--------|
| When produced? | After engine loop (final evaluator) |
| Who produces it? | MARBLE Evaluator.evaluate_task_db() |
| Immediately available? | Yes |
| Needs task ground truth? | ⚠️ YES (root_cause labels) |
| Needs hidden answer? | ⚠️ YES (expected root causes from task config) |
| Uses future episode info? | No |
| Needs human labels? | No (labels in task config) |
| Usable in expose/withhold? | Yes |

**Provenance**: ⚠️ **CONDITIONAL** — Requires ground-truth labels from task config. This is acceptable because labels are part of the task definition (not future information), but it means the signal is not purely behavioral.


### 9. task_evaluation (Bargaining — World Simulation)

| Question | Answer |
|----------|--------|
| When produced? | NEVER (evaluator not called in graph mode) |
| Who produces it? | N/A |

**Provenance**: ❌ **INVALID** — Evaluator not invoked for bargaining domain


### 10. code_quality (Coding — LLM-judged)

| Question | Answer |
|----------|--------|
| When produced? | NEVER in graph mode (star mode only) |
| Who produces it? | N/A |

**Provenance**: ❌ **INVALID** — Only available in star coordinate mode


### 11. planning_scores (Ordinal 1-5)

| Question | Answer |
|----------|--------|
| When produced? | NEVER (commented out in graph mode) |
| Who produces it? | N/A (hardcoded -1) |

**Provenance**: ❌ **INVALID** — All values are -1


### 12. communication_scores (Ordinal 1-5)

| Question | Answer |
|----------|--------|
| When produced? | NEVER (commented out in graph mode) |
| Who produces it? | N/A (hardcoded -1) |

**Provenance**: ❌ **INVALID** — All values are -1


## Summary Classification

| Signal | Domain | Provenance | Notes |
|--------|--------|------------|-------|
| team_success (binary) | All | ✅ SAFE | Currently used |
| token_usage | All | ✅ SAFE | Effort proxy, not quality |
| iteration_count | All | ✅ SAFE | Difficulty proxy |
| summary_length | All | ✅ SAFE | Progress proxy, noisy |
| task_evaluation (minecraft) | Minecraft | ✅ SAFE | Rule-based, continuous |
| task_evaluation (db) | Database | ⚠️ CONDITIONAL | Needs ground-truth labels |
| task_evaluation (research) | Research | ⚠️ CONDITIONAL | Evaluator crashes in practice |
| task_evaluation (bargaining) | Bargaining | ❌ INVALID | Evaluator not called |
| code_quality (coding) | Coding | ❌ INVALID | Star mode only |
| planning_scores | All | ❌ INVALID | Commented out |
| communication_scores | All | ❌ INVALID | Commented out |


## Decision

**Only SAFE signals may enter Phase 5 (BehavioralOutcome API):**
1. ✅ team_success (binary) — baseline
2. ✅ token_usage — effort proxy
3. ✅ iteration_count — difficulty proxy
4. ✅ summary_length — progress proxy
5. ✅ task_evaluation (minecraft) — continuous quality

**CONDITIONAL signals require further investigation:**
- ⚠️ task_evaluation (db) — acceptable if labels are considered part of task definition
- ⚠️ task_evaluation (research) — requires evaluator crash fix

**INVALID signals excluded:**
- ❌ task_evaluation (bargaining) — no evaluator
- ❌ code_quality (coding) — wrong mode
- ❌ planning_scores / communication_scores — disabled
