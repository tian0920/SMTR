# MARBLE Native Evaluation Signal Audit (Static Code Analysis)

**Date**: 2026-08-24

**Scope**: Static audit of MARBLE evaluator source code and SMTR integration

**Files audited**:
- `/home/ecs-user/MARBLE/marble/evaluator/evaluator.py` (619 lines)
- `/home/ecs-user/MARBLE/marble/evaluator/evaluator_prompts.json` (30 lines)
- `/home/ecs-user/MARBLE/marble/engine/engine.py` (1172 lines, graph mode loop)
- `src/smtr/marble/trajectory_collector.py` (532 lines)
- `src/smtr/marble/engine_process.py` (844 lines)
- `src/smtr/marble/outcome/scenarios/database.py` (173 lines)
- `src/smtr/marble/outcome/scenarios/generic.py` (134 lines)


## 1. Per-Scenario Evaluator Inventory

| Scenario | Evaluator Source | Raw Fields | Raw Range | Binary/Ordinal/Continuous | Per-Iteration Available | Final-Only | Online Observable | Oracle Risk | Usable for TCI |
|----------|------------------|------------|-----------|---------------------------|------------------------|------------|-------------------|-------------|----------------|
| **bargaining** | None (World Simulation) | N/A | N/A | N/A | No | N/A | N/A | N/A | ❌ No evaluator called in graph mode |
| **coding** | evaluate_code_quality() | instruction_following, executability, consistency, quality | 1-5 each | Continuous (discrete 1-5) | No (star mode only) | Yes (star mode) | ⚠️ Star mode only | LOW | ⚠️ Graph mode doesn't call it |
| **database** | evaluate_task_db() | root_causes, predicted | Binary match | Binary (subset match) | No | Yes | Yes | LOW (ground-truth labels) | ✅ But already binary |
| **minecraft** | Rule-based (block_hit_rate) | block_hit_rate | 0.0-1.0 | Continuous | No (read from file) | Yes | ⚠️ File I/O | NONE | ✅ Continuous signal |
| **research** | evaluate_task_research() | innovation, safety, feasibility | 1-5 each | Continuous (discrete 1-5) | No | Yes | ⚠️ Template bug crashes | LOW | ⚠️ Crashes in practice |


## 2. Engine Loop Evaluator Calls (Graph Mode)

**Source**: `/home/ecs-user/MARBLE/marble/engine/engine.py` lines 250-490

### Per-Iteration Evaluators (COMMENTED OUT)

```python
# Line 297-298: COMMUNICATION EVALUATOR DISABLED
# communications_str = self._format_communications(iteration_data_communications)
# self.evaluator.evaluate_communication(self.task, communications_str)
self.evaluator.metrics["communication_score"].append(-1)  # Hardcoded -1

# Line 306-316: PLANNING + KPI EVALUATORS DISABLED
# agent_profiles = self._get_agent_profiles()
# ...
# self.evaluator.evaluate_planning(...)
# self.evaluator.evaluate_kpi(...)
self.evaluator.metrics["planning_score"].append(-1)  # Hardcoded -1
```

**Conclusion**: Per-iteration `evaluate_communication`, `evaluate_planning`, and `evaluate_kpi` are **all commented out** in graph mode. They are replaced with hardcoded `-1` placeholders.

### Final Evaluators (ACTIVE but problematic)

```python
# Line 451-454: RESEARCH EVALUATOR (ACTIVE)
if isinstance(self.environment, ResearchEnvironment):
    self.evaluator.evaluate_task_research(self.task, iteration_data_summary)
    summary_data["task_evaluation"] = self.evaluator.metrics["task_evaluation"]

# Line 459-460: WORLD/BARGAINING EVALUATOR (NOT CALLED)
elif self.environment.name == "World Simulation Environment":
    self.evaluator.evaluate_task_world(self.task, iteration_data["summary"])

# Line 465-471: MINECRAFT EVALUATOR (ACTIVE)
elif isinstance(self.environment, MinecraftEnvironment):
    block_hit_rate = json.load(f)[-1]["block_hit_rate"]
    summary_data["task_evaluation"] = block_hit_rate * 5

# Line 472-482: DB EVALUATOR (ACTIVE)
elif self.environment.name == "DB Environment":
    self.evaluator.evaluate_task_db(...)
```

**Critical finding**: The bargaining domain uses `environment.name == "World Simulation Environment"`, but the actual environment type is "Base" (not "World Simulation"). This causes the evaluator to **not be called** for bargaining tasks.


## 3. SMTR Trajectory Collector Score Extraction

**Source**: `src/smtr/marble/trajectory_collector.py` lines 363-416

```python
def _extract_team_success(raw: dict[str, Any]) -> bool:
    # Check for explicit team_success key
    if "team_success" in raw:
        return bool(raw["team_success"])
    # Fallback: check iterations format
    iterations = raw.get("iterations", [])
    if isinstance(iterations, list) and iterations:
        last = iterations[-1]
        if isinstance(last, dict):
            tr = last.get("task_results", [])
            if isinstance(tr, list) and tr:
                return True  # Engine ran → assume success

def _extract_score(raw: dict[str, Any]) -> float:
    if "score" in raw:
        return float(raw["score"])
    # Fallback: planning_scores average
    planning = raw.get("planning_scores", [])
    if isinstance(planning, list) and planning:
        valid = [s for s in planning if isinstance(s, (int, float)) and s >= 0]
        if valid:
            return sum(valid) / len(valid)
    # Ultimate fallback: binary success
    return float(_extract_team_success(raw))  # 0.0 or 1.0
```

**Issue**: Since `planning_scores` is always `[]` or `[-1, -1, ...]`, the score extraction always falls back to `float(team_success)` = {0.0, 1.0}.


## 4. Key Questions Answered

### Q1. Does `team_success` discard information from the original evaluator?

**YES**. The SMTR trajectory_collector infers `team_success` from the presence of iteration data, not from any explicit evaluator signal. This discards:
- Research: innovation/safety/feasibility ratings (1-5 scale)
- Minecraft: block_hit_rate (0.0-1.0 continuous)
- Database: root_cause match precision/recall/F1
- Coding: code quality scores (1-5 scale, but not called in graph mode)
- Bargaining: buyer/seller effectiveness ratings (not called)

### Q2. Is `team_score` real from evaluator, or fallback/constant?

**FALLBACK**. The `_extract_score()` function:
1. Checks for explicit `score` key → **ABSENT** in all outputs
2. Checks `planning_scores` average → **all -1**, filtered out as invalid
3. Falls back to `float(team_success)` → **always 0.0 or 1.0**

### Q3. Does each iteration have an independent score/feedback?

**NO (in graph mode)**. Per-iteration evaluators (`evaluate_communication`, `evaluate_planning`, `evaluate_kpi`) are commented out and replaced with `-1`.

The only per-iteration data available:
- `task_assignments`: dict (agent → task description)
- `task_results`: list of agent outputs (text)
- `summary`: planner-generated summary (text)
- `continue_simulation`: bool (planner decision)
- `communications`: list of inter-agent messages

### Q4. Are there continuous signals available without ground-truth labels?

**YES** (but not currently extracted):
1. **Minecraft**: `block_hit_rate` (0.0-1.0) — rule-based, no ground truth
2. **Research**: `task_evaluation` dict with {innovation, safety, feasibility} (1-5 each) — LLM-judged, no ground truth
3. **Coding**: `code_quality` dict with {instruction_following, executability, consistency, quality} (1-5 each) — LLM-judged, but only in star mode
4. **Token usage**: continuous count, available for all domains
5. **Iteration count**: number of iterations before convergence (implicit task difficulty signal)

### Q5. Do different domains provide different signals?

**YES**:
- **Minecraft**: Continuous (block_hit_rate)
- **Research**: Ordinal (1-5 × 3 dimensions)
- **Database**: Binary (root_cause subset match) + fine-grained (TP/FP/recall/precision/F1 in SMTR)
- **Coding**: Ordinal (1-5 × 4 dimensions, star mode only)
- **Bargaining**: None (evaluator not called)


## 5. Summary of Available Native Signals

| Signal | Type | Domain | Currently Used | Available in Graph Mode | Safe for TCI |
|--------|------|--------|----------------|------------------------|--------------|
| team_success | Binary | All | ✅ Yes | ✅ Yes | ✅ SAFE |
| score (fallback) | Binary {0,1} | All | ✅ Yes | ✅ Yes | ✅ SAFE (but coarse) |
| task_evaluation (research) | Ordinal 1-5 ×3 | Research | ❌ No | ⚠️ Crashes | ⚠️ CONDITIONAL |
| task_evaluation (minecraft) | Continuous 0-5 | Minecraft | ❌ No | ✅ Yes | ✅ SAFE |
| code_quality | Ordinal 1-5 ×4 | Coding | ❌ No | ❌ Star mode only | ❌ INVALID (graph mode) |
| task_evaluation (db) | Binary match | Database | ❌ No | ✅ Yes | ✅ SAFE |
| planning_scores | Ordinal 1-5 | All | ❌ No | ❌ Commented out | ❌ INVALID |
| communication_scores | Ordinal 1-5 | All | ❌ No | ❌ Commented out | ❌ INVALID |
| token_usage | Continuous | All | ❌ No | ✅ Yes | ✅ SAFE |
| iteration_count | Discrete | All | ❌ No | ✅ Yes | ✅ SAFE |
| block_hit_rate | Continuous 0-1 | Minecraft | ❌ No | ✅ Yes | ✅ SAFE |


## 6. Recommendations

1. **Extract `task_evaluation` from raw outputs** when available (minecraft, research if evaluator doesn't crash)
2. **Use `token_usage` and `iteration_count`** as auxiliary continuous signals
3. **For bargaining**: No native signal available; requires custom evaluator or model downgrade
4. **For research**: Investigate evaluator crash (see Phase 9 research/83 analysis)
5. **Do not re-enable commented-out evaluators** without fixing template bugs first
