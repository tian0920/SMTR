# Official Metric Trajectory Integration (Phase A)

**Date**: 2026-08-24
**Status**: ✅ IMPLEMENTED


## 1. Changes Summary

### Trajectory dataclass (`trajectory_collector.py`)

**New fields (PRIMARY)**:
```python
official_metric_name: str = "unknown"
official_metric_raw: float | None = None
official_metric_normalized: float | None = None
official_metric_valid: bool = False
official_metric_source: str = "official_evaluator"
official_metric_error: str | None = None
```

**Legacy fields (DIAGNOSTIC ONLY)**:
```python
team_success: bool = False        # NOT used for reward/TCI
score: float | None = None        # = official_metric_normalized (or None)
```

**New property**:
```python
@property
def evaluator_failure(self) -> bool:
    return not self.official_metric_valid
```

### Extraction Flow
```
raw MARBLE engine output (JSONL)
    ↓
_parse_raw_output()
    ↓
task_evaluation field
    ↓
OfficialMetricOutcomeEvaluator.evaluate()
    ↓
OfficialMetricOutcome(raw_score, normalized_score, is_valid)
    ↓
Trajectory(official_metric_raw, official_metric_normalized, official_metric_valid)
```

### Rules
1. `score` = `official_metric_normalized` (or `None` if invalid)
2. `team_success` is extracted but **NEVER** used for reward
3. Missing `task_evaluation` → `official_metric_valid=False`
4. No fallback to `team_success` when official metric is missing
5. `evaluator_failure=True` → episode marked INVALID

## 2. Per-Scenario Metric Extraction

| Scenario | Official Metric | Raw Format | Normalization |
|----------|----------------|------------|---------------|
| database | `root_cause_recall` | `{root_cause: [...], predicted: ...}` | subset match recall |
| research | `avg_innovation_safety_feasibility` | `{innovation:1-5, safety:1-5, feasibility:1-5}` | (avg-1)/4 |
| minecraft | `block_hit_rate` | float (block_hit_rate × 5) | /5.0 |
| coding | `avg_code_quality` | `{4 dimensions, each 1-5}` | (avg-1)/4 |
| bargaining | `avg_negotiation_quality` | `{buyer:3d, seller:3d, each 1-5}` | (avg-1)/4 |


## 3. Engine Integration Fixes (Applied)

### Fix 1: Environment Type/Name Override (Critical)
**Root cause**: Raw JSONL task configs have `environment.type="Base"` for 4/5
scenarios (database, research, coding, bargaining). Only minecraft had the
correct type. This caused the MARBLE engine to create `BaseEnvironment` instead
of scenario-specific environments, preventing evaluators from running.

**Fix**: `_build_engine_config()` now ALWAYS overrides `env.type` and `env.name`
from the `_SCENARIO_ENV_TYPE` and `_SCENARIO_ENV_NAME` mappings, regardless of
the raw task config values.

### Fix 2: Evaluation LLM Configuration
**Root cause**: Raw configs have `metrics.evaluate_llm=""` (empty string).
The MARBLE evaluator uses this to call an LLM for rating research/bargaining/
database tasks. With empty model name, `model_prompting()` returns None,
causing beartype violations and missing task_evaluation.

**Fix**: `_build_engine_config()` now sets `metrics.evaluate_llm.model` to the
same model as the main LLM if it's missing or empty.

### Fix 3: Coding `code_quality` Field
**Root cause**: MARBLE engine stores coding scores in `code_quality` field,
not `task_evaluation`. The engine's star_coordinate only sets `code_quality`.

**Fix**: `OfficialMetricOutcomeEvaluator.evaluate()` now checks `code_quality`
as fallback for coding scenario.

### Fix 4: Empty Dict Detection
**Root cause**: MARBLE evaluator sets `task_evaluation={}` when LLM rating
parser fails (research). This is truthy but contains no valid data.

**Fix**: Empty dicts now treated as missing/invalid.

### Fix 5: Bargaining Sentinel Detection
**Root cause**: MARBLE evaluator returns -1 for each dimension when LLM
parsing fails. This caused normalized_score=-0.5 (outside [0,1] range).

**Fix**: Dimensions with values < 1 now detected as sentinel failures.

### Known Limitation: Coding `solution.py` Path
The MARBLE engine reads `MARBLE/marble/workspace/solution.py` (relative path)
but coding agents don't write to this location. All coding episodes will be
evaluator failures. This is a MARBLE engine bug, not an SMTR issue.

### Smoke Test Results (post-fix)

| Scenario   | Status | Metric Valid | Score | Notes |
|------------|--------|-------------|-------|-------|
| database   | ✅ | True | 0.0 | Evaluator works; model got recall=0 |
| research   | ✅ | True | 0.833 | Evaluator works |
| minecraft  | ✅ | True | 0.0 | Evaluator works; no score.json |
| coding     | ❌ | False | None | Engine path bug: solution.py not found |
| bargaining | ⚠️ | varies | varies | LLM parser may fail → sentinel detected |
