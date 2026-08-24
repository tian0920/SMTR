# Phase 6: Go/No-Go Decision

**Date**: 2026-08-24

**Decision**: **NO-GO** — Do not proceed to full 5-domain run.


## Evidence Summary

### 1. Difficulty Profiling (200 episodes)

| Metric | Value |
|--------|-------|
| Total tasks profiled | 100 |
| Easy tasks (reward > 0.9) | 99 (99%) |
| Medium tasks (0.5 < reward ≤ 0.9) | 0 (0%) |
| Hard tasks (reward ≤ 0.5) | 1 (1%) |

**Conclusion**: Severe ceiling effect. 99/100 tasks achieve 100% success with no_memory baseline.

### 2. Hard Baseline Pilot (Phase 3)

| Task | Baseline Reward | Failure Rate | Verdict |
|------|-----------------|--------------|---------|
| research/83 | 0.0 | 100% | PASS (margin exists) |

**Caveat**: research/83 execution time was ~38s (vs typical 200-600s), suggesting possible evaluator crash rather than genuine task difficulty.

### 3. TCI Activation (Phase 4)

| Metric | Threshold | Actual | Pass? |
|--------|-----------|--------|-------|
| MOR > 5% | > 0.05 | 0.0000 | **FAIL** |
| Validated memories | > 0 | 0 | **FAIL** |
| Cross-episode reuse | ≥ 1 | 0 | **FAIL** |

**Verdict**: 0/3 Go criteria met.


## Root Cause Analysis

### Why 99% Easy Tasks?

1. **qwen3-30b-a3b capability**: The model is sufficiently capable to solve nearly all MARBLE tasks (single root cause, structured reasoning)
2. **MARBLE task design**: Tasks 1-50 (single root cause) and 51-100 (dual root cause) were designed for smaller LLMs or more complex scenarios
3. **Evaluator robustness**: Even with evaluator crash patches, most tasks pass evaluation

### Why research/83 Fails?

- **Execution time**: 38s (anomalously short vs 200-600s for other tasks)
- **Hypothesis**: Evaluator template bug causes crash before scoring
- **Evidence**: Both seeds (0, 1) fail identically with team_success=False
- **Action needed**: Inspect raw MARBLE output for research/83 to confirm


## Go/No-Go Criteria Checklist

| Criterion | Required | Actual | Status |
|-----------|----------|--------|--------|
| MOR > 5% | Yes | 0.00% | ❌ FAIL |
| Validated memories > 0 | Yes | 0 | ❌ FAIL |
| Cross-episode reuse ≥ 1 | Yes | 0 | ❌ FAIL |
| Hard baseline reward < 0.9 | Yes | 0.0 | ✅ PASS (but 1 task only) |

**Final verdict**: 1/4 criteria met → **NO-GO**


## Recommended Actions

### Option A: Investigate research/83 (Low effort)

1. Inspect raw MARBLE engine output for research/83
2. Confirm if evaluator crash or genuine task failure
3. If crash: fix evaluator, re-run profiling
4. If genuine: only 1 hard task exists → insufficient for statistical power

### Option B: Use harder MARBLE tasks (Medium effort)

1. Check if MARBLE has tasks 101+ or custom difficulty scenarios
2. Design multi-step reasoning tasks requiring memory
3. Add adversarial distractor information

### Option C: Use weaker LLM (Medium effort)

1. Switch to qwen3-7b or similar smaller model
2. Re-run difficulty profiling
3. Expect more medium/hard tasks

### Option D: Redefine success criteria (High effort, risky)

1. Change from binary success to partial credit scoring
2. Measure "quality of reasoning" not just "correct answer"
3. Requires significant evaluator redesign


## Conclusion

**The current MARBLE + qwen3-30b-a3b configuration does not exhibit memory formation opportunity.**

The system achieves near-perfect performance without memory, leaving no room for causal transfer (Δ = 0). Proceeding to full 5-domain run would consume ~28+ hours of compute with no expected positive signal.

**Recommendation**: Pursue Option A (investigate research/83) or Option B/C (harder tasks or weaker model) before re-evaluating Go/No-Go.
