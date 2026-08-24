# Binary-Success Ceiling Manifest (Frozen)

**Date**: 2026-08-24

**Status**: FROZEN — Do not overwrite or modify these results.

**Purpose**: Preserve the baseline binary-success evaluation results before any evaluator modification.


## 1. Current Model

- **Backbone**: qwen3-30b-a3b
- **API endpoint**: DashScope MaaS (ap-southeast-1)
- **Thinking mode**: disabled (SMTR_LLM_ENABLE_THINKING=false)
- **Coordinate mode**: graph (default for all 5 domains)


## 2. Current Task Distribution

- **5 MARBLE domains**: bargaining, coding, database, minecraft, research
- **Tasks per domain**: 20 (tasks 1-100 across 5 domains)
- **Seeds**: [0, 1] per task
- **Total episodes profiled**: 200


## 3. Difficulty Profiling Results

**Commit**: 7c2cbbe

**Result directory**: results/marble/difficulty_profile/

| Metric | Value |
|--------|-------|
| Total tasks | 100 |
| Easy (reward > 0.9) | 99 (99%) |
| Medium (0.5 < reward ≤ 0.9) | 0 (0%) |
| Hard (reward ≤ 0.5) | 1 (1%) |
| Apparent hard task | research/83 (reward=0.0) |

**Interpretation**: Severe ceiling effect. 99% of tasks achieve perfect binary success without memory.


## 4. Online TCI Smoke Test Results

**Commit**: f659847

**Result directories**:

- results/marble/online_pilot_tci_smoke2/
- results/marble/pilot_hard_tci/

| Metric | Value |
|--------|-------|
| expose_success | 100% |
| withhold_success | 100% |
| delta | 0 |
| MOR (Memory Opportunity Rate) | 0 |
| Validated memories | 0 |
| Cross-episode reuse | 0 |

**Interpretation**: TCI correctly rejects all candidates because binary success is identical across expose/withhold branches.


## 5. Conclusion

**Binary task-success signal suffers severe ceiling effect.**

The current SMTR pipeline extracts only:
- `team_success`: binary (True/False)
- `score`: fallback to `float(team_success)` = {0.0, 1.0}

This coarse signal provides no discrimination for TCI causal effect measurement.

**Root causes identified** (to be detailed in Phase 1 audit):
1. MARBLE graph-mode engine comments out per-iteration evaluators (planning, communication, KPI)
2. Task-specific evaluators (research, db, minecraft, coding) are either not called or crash
3. SMTR trajectory_collector falls back to binary success for score


## 6. Frozen Artifacts

| Artifact | Path | Commit |
|----------|------|--------|
| Difficulty episode CSV | results/marble/difficulty_profile/difficulty_episode.csv | 7c2cbbe |
| Difficulty summary | results/marble/difficulty_profile/difficulty_summary.csv | 7c2cbbe |
| Task ranking | results/marble/difficulty_profile/task_difficulty_ranking.csv | 7c2cbbe |
| TCI smoke2 results | results/marble/online_pilot_tci_smoke2/ | f659847 |
| Hard pilot TCI results | results/marble/pilot_hard_tci/ | 7c2cbbe |
| Hard pilot baseline | results/marble/pilot_hard_baseline/ | 7c2cbbe |
| Go/No-Go decision | docs/audit/go_no_go_decision.md | 7c2cbbe |


## 7. Immutability Constraint

**These results must NOT be overwritten, re-run, or modified.**

Any subsequent evaluator changes must produce NEW result directories with distinct naming.

Comparison against this frozen baseline is required for all future Go/No-Go decisions.
