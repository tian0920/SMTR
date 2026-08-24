# MARBLE Iteration Signal Analysis (Phase 3)

**Total episodes with iteration data**: 198
**Total iteration records**: 1468

## Score Trajectory Summary

Episodes with multi-iteration progress: 177
Episodes with growing summaries: 69
Episodes with shrinking summaries: 99

## Key Finding: Per-Iteration Evaluator Signals

**All per-iteration evaluator signals are DISABLED in MARBLE graph mode.**

The MARBLE engine graph_coordinate() loop has the following evaluators **commented out**:
- `evaluate_communication()` → replaced with hardcoded `-1`
- `evaluate_planning()` → replaced with hardcoded `-1`
- `evaluate_kpi()` → replaced with hardcoded `-1`

This means NO continuous per-iteration signal is available from the MARBLE evaluator.

## Available Iteration-Level Signals

| Signal | Type | Source | Usable? |
|--------|------|--------|---------|
| summary_length | Continuous (proxy) | Planner summary text length | ✅ Yes (proxy) |
| n_task_results | Discrete | Number of agent task completions | ✅ Yes |
| n_messages | Discrete | Inter-agent communication count | ✅ Yes |
| n_errors | Discrete | Error keyword count in output | ✅ Yes (proxy) |
| continue_simulation | Binary | Planner termination decision | ✅ Yes |
| token_usage (global) | Continuous | Total token consumption | ✅ Yes |
| planning_scores | Ordinal 1-5 | Per-iteration evaluator | ❌ No (hardcoded -1) |
| communication_scores | Ordinal 1-5 | Per-iteration evaluator | ❌ No (hardcoded -1) |

## Can We Define Delta(m,r) = P_expose(m,r) - P_withhold(r)?

**With binary success (current)**: No. P is always {0, 1}, ceiling effect.

**With iteration-level proxies**:
- P = summary_length_delta (last_iter - first_iter): Possible, but noisy
- P = token_usage: Possible, but not directly related to task quality
- P = n_errors: Possible, inverse proxy for quality

**With native final evaluator signals** (if available):
- P = task_evaluation (research): {innovation, safety, feasibility} 1-5 → average
- P = task_evaluation (minecraft): block_hit_rate * 5
- P = task_evaluation (database): root_cause recall (0.0, 0.5, 1.0)

**Recommendation**: Use native final evaluator signals where available (minecraft, database),
and iteration-level proxies (summary_length trajectory) for domains without native evaluators.