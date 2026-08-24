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


## 3. Upstream Fix Required

The MARBLE engine's `environment.name` check fails for database and bargaining
because the JSONL files have empty `environment.name`. This means `task_evaluation`
will be `None` for these scenarios until the upstream is fixed.

**Workaround**: The `TrajectoryCollector._build_engine_config()` normalizes
`environment.type` but does NOT set `environment.name`. The upstream MARBLE
engine must be patched or the JSONL files updated.

See: `docs/audit/multiagentbench_ceiling_root_cause.md` Section 3.
