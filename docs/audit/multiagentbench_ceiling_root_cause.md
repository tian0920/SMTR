# MultiAgentBench Ceiling Effect Root-Cause Audit

**Date**: 2026-08-24

**Status**: ✅ COMPLETED


## 1. Executive Summary

**Ceiling Effect Root Cause**: SMTR is **NOT reading** the official `task_evaluation` field from MARBLE engine output.

| Metric | SMTR Uses | Official | Status |
|--------|-----------|----------|--------|
| **Task Score** | Binary `team_success` (0/1) | Continuous `task_evaluation` (0-100%) | ❌ **WRONG** |
| **Result** | 99/100 = 100% success | Variance (34%-84% in paper) | 🔴 **CEILING EFFECT** |

**Classification of 100 Profiling Tasks** (200 episodes):

| Category | Count | Proportion | Description |
|----------|-------|------------|-------------|
| **A. Genuinely Solved** | 0 | 0% | None — official metrics not computed |
| **B. Partial Success → Success** | 198 | 99% | Binary `team_success=1` but no official TS |
| **C. Evaluator Too Coarse** | 0 | 0% | N/A — evaluators not running |
| **D. Evaluator Fallback** | 0 | 0% | N/A — no fallback path |
| **E. Execution Failure** | 2 | 1% | research/83 (both seeds, exit_code=1) |
| **F. Unknown** | 0 | 0% | All classified |

**Verdict**: 99% of tasks show **artificial ceiling** because SMTR uses binary heuristic instead of official continuous scores.


## 2. Root Cause Chain

### Step 1: MARBLE Engine Computes `task_evaluation`

The MARBLE engine **DOES** compute official metrics per scenario:

```python
# marble/engine/engine.py (graph mode)

# Research: Uses isinstance() check → SHOULD run
if isinstance(self.environment, ResearchEnvironment):
    self.evaluator.evaluate_task_research(self.task, iteration_data_summary)
    summary_data["task_evaluation"] = self.evaluator.metrics["task_evaluation"]

# Minecraft: Uses isinstance() check → DOES run
elif isinstance(self.environment, MinecraftEnvironment):
    block_hit_rate = json.load(f)[-1]["block_hit_rate"]
    summary_data["task_evaluation"] = block_hit_rate * 5

# Database: Uses name check → FAILS (name is empty)
elif self.environment.name == "DB Environment":  # ❌ name="" in JSONL
    self.evaluator.evaluate_task_db(...)
    summary_data["task_evaluation"] = ...

# Bargaining: Uses name check → FAILS (name is empty)
elif self.environment.name == "World Simulation Environment":  # ❌ name="" in JSONL
    self.evaluator.evaluate_task_world(...)
    summary_data["task_evaluation"] = ...
```

**Issue 1**: Database and bargaining evaluators **never run** because `environment.name` is empty in JSONL files.

**Issue 2**: Research evaluator **crashes** due to missing API key (research/83).

**Issue 3**: Minecraft evaluator **does run** but SMTR doesn't read it.


### Step 2: SMTR TrajectoryCollector Ignores `task_evaluation`

The `TrajectoryCollector` in `trajectory_collector.py`:

```python
# src/smtr/marble/trajectory_collector.py

def _extract_team_success(self, iterations: list[dict]) -> bool:
    """Infer success from final iteration summary."""
    final_summary = iterations[-1].get("summary", "")
    return "success" in final_summary.lower()  # ❌ HEURISTIC

def _extract_score(self, run_dict: dict) -> float:
    """Fallback chain for score."""
    score = run_dict.get("score")
    if score is not None:
        return float(score)
    
    planning_scores = run_dict.get("planning_scores", [])
    if planning_scores and all(s != -1 for s in planning_scores):
        return sum(planning_scores) / len(planning_scores)
    
    team_success = run_dict.get("team_success", False)
    return float(team_success)  # ❌ BINARY FALLBACK
```

**Critical Issue**: `TrajectoryCollector` **NEVER** reads `task_evaluation` field!

It only reads:
- `score` (not present)
- `planning_scores` (all -1 in graph mode)
- `team_success` (inferred from summary text)

**Result**: Falls back to binary `float(team_success)` → {0.0, 1.0}.


### Step 3: Binary Heuristic → Ceiling Effect

The `_extract_team_success()` heuristic:
- Checks if "success" appears in final iteration summary
- Most summaries contain "successfully completed" or similar phrases
- Result: 198/200 episodes → `team_success=True` → `score=1.0`

**But**: This is **NOT** the official metric! The official `task_evaluation` field contains:
- Research: {innovation: 1-5, safety: 1-5, feasibility: 1-5}
- Minecraft: `block_hit_rate` × 5 (continuous [0, 5])
- Database: {root_cause: [...], predicted: ...}
- Bargaining: {buyer: {...}, seller: {...}}
- Coding: {instruction_following: 1-5, executability: 1-5, ...}


## 3. Per-Scenario Root Cause Analysis

### database (40 episodes, 20 tasks × 2 seeds)

| Check | Result |
|-------|--------|
| Official evaluator ran? | ❌ **NO** |
| Why? | `environment.name == "DB Environment"` check fails (name="") |
| SMTR `team_success` | 40/40 = 100% |
| Official `task_evaluation` | 0/40 available (None) |
| **Root Cause** | **D. Evaluator not triggered** + **B. Binary heuristic** |

**Evidence**:
```python
# JSONL: environment.name = ""
# Engine check: self.environment.name == "DB Environment"  # ❌ FAILS
```

**What SHOULD happen**:
- Compute root cause recall: `|predicted ∩ ground_truth| / |ground_truth|`
- Example scores: 0.34, 0.53, 0.45 (from paper Table 1)


### research (40 episodes, 20 tasks × 2 seeds)

| Check | Result |
|-------|--------|
| Official evaluator ran? | ❌ **NO** (crashed) |
| Why? | Missing API key → LLM-as-judge fails |
| SMTR `team_success` | 38/40 = 95% |
| Official `task_evaluation` | 0/40 available (None) |
| **Root Cause** | **E. Execution failure** (research/83) + **B. Binary heuristic** |

**Evidence**:
- research/83: Both seeds have `exit_code=1`, `duration=38.77s` (too fast = crash)
- Other 38 episodes: `team_success=True` but no `task_evaluation`

**What SHOULD happen**:
- LLM-judged ratings: {innovation, safety, feasibility} 1-5
- Average → scale to 0-100%
- Example scores: 80.87%, 80.80%, 80.00% (from paper Table 1)


### minecraft (40 episodes, 20 tasks × 2 seeds)

| Check | Result |
|-------|--------|
| Official evaluator ran? | ✅ **YES** (in engine) |
| SMTR read it? | ❌ **NO** |
| SMTR `team_success` | 40/40 = 100% |
| Official `task_evaluation` | 0/40 in SMTR data (but computed in engine) |
| **Root Cause** | **B. Binary heuristic** (SMTR ignores available signal) |

**Evidence**:
```python
# Engine DOES compute: summary_data["task_evaluation"] = block_hit_rate * 5
# But TrajectoryCollector doesn't read it
# Result: SMTR uses team_success heuristic instead
```

**What SHOULD happen**:
- Read `block_hit_rate` directly
- Scale to 0-100%: `block_hit_rate * 100`
- Example scores: 6.12%, 0.21%, 9.15%, 33.60% (from paper Table 1)


### coding (40 episodes, 20 tasks × 2 seeds)

| Check | Result |
|-------|--------|
| Official evaluator ran? | ❌ **NO** (star mode only) |
| Why? | Graph mode doesn't call `evaluate_code_quality()` |
| SMTR `team_success` | 40/40 = 100% |
| Official `task_evaluation` | 0/40 available (None) |
| **Root Cause** | **D. Evaluator not triggered** + **B. Binary heuristic** |

**Evidence**:
```python
# Engine: code_quality evaluation only in star mode
if self.environment.name == "Coding Environment":  # star mode
    self.evaluator.evaluate_code_quality(...)
# Graph mode: No coding evaluator call
```

**What SHOULD happen**:
- LLM-judged ratings: {instruction_following, executability, consistency, quality} 1-5
- Average → scale to 0-100%
- Example scores: 59.90%, 62.10%, 56.60%, 65.10% (from paper Table 1)


### bargaining (40 episodes, 20 tasks × 2 seeds)

| Check | Result |
|-------|--------|
| Official evaluator ran? | ❌ **NO** |
| Why? | `environment.name == "World Simulation Environment"` check fails (name="") |
| SMTR `team_success` | 40/40 = 100% |
| Official `task_evaluation` | 0/40 available (None) |
| **Root Cause** | **D. Evaluator not triggered** + **B. Binary heuristic** |

**Evidence**:
```python
# JSONL: environment.name = ""
# Engine check: self.environment.name == "World Simulation Environment"  # ❌ FAILS
```

**What SHOULD happen**:
- LLM-judged ratings: {buyer, seller} × {effectiveness, progress, interaction} 1-5
- Average → scale to 0-100%
- Example scores: 72.81%, 72.13%, 73.15%, 74.47% (from paper Table 1)


## 4. research/83 Special Investigation

### Classification: **E. Execution Failure**

**Evidence**:
| Field | Value |
|-------|-------|
| `exit_code` | 1 (both seeds) |
| `engine_duration_seconds` | 38.77s (seed=0), 38.78s (seed=1) |
| `team_success` | False (both seeds) |
| `score` | 0.0 (both seeds) |
| `task_evaluation` | None |

**Root Cause Chain**:
1. Research evaluator uses `isinstance(self.environment, ResearchEnvironment)` → **SHOULD run**
2. Evaluator calls `model_prompting()` → LLM-as-judge
3. Missing `OPENAI_API_KEY` → litellm fails → returns None
4. `parse_research_ratings(None)` → beartype violation
5. Engine crashes → `exit_code=1`

**Conclusion**: research/83 is **NOT a genuine failure** — it's an **engine/evaluator crash**.

**Action**: Exclude research/83 from difficulty analysis. It should not count as "hard task".


## 5. Ceiling Effect Taxonomy

### Primary Root Causes

| ID | Root Cause | Affected Scenarios | Proportion |
|----|-----------|-------------------|------------|
| **R1** | SMTR ignores `task_evaluation` field | ALL 5 scenarios | 100% |
| **R2** | Database/bargaining evaluators not triggered | database, bargaining | 40% |
| **R3** | Coding evaluator only in star mode | coding | 20% |
| **R4** | Research evaluator crashes | research | 20% |
| **R5** | Binary heuristic for `team_success` | ALL 5 scenarios | 100% |

### Secondary Root Causes

| ID | Root Cause | Description |
|----|-----------|-------------|
| **S1** | JSONL `environment.name` is empty | Database/bargaining evaluators never triggered |
| **S2** | Graph mode disables evaluators | Planning/communication scores all -1 |
| **S3** | Missing API key in subprocess | Research evaluator crashes |


## 6. Comparison with Paper Results

### Paper Table 1 (Official TS Scores)

| Model | Research | Minecraft | Database | Coding | Bargaining | Average |
|-------|----------|-----------|----------|--------|------------|---------|
| Llama-3.1-8B | 80.87% | 6.12% | 34.00% | 59.90% | 72.81% | 50.74% |
| Llama-3.1-70B | 80.80% | 0.21% | 53.00% | 62.10% | 72.13% | 53.65% |
| Llama-3.3-70B | 80.00% | 9.15% | 28.50% | 56.60% | 73.15% | 49.48% |
| GPT-3.5-turbo | 70.20% | 5.05% | 45.00% | 55.50% | 71.67% | 49.48% |
| GPT-4o-mini | **84.13%** | **33.60%** | **45.00%** | **65.10%** | **74.47%** | **60.46%** |

### SMTR Profiling Results (Binary `team_success`)

| Model | Research | Minecraft | Database | Coding | Bargaining | Average |
|-------|----------|-----------|----------|--------|------------|---------|
| qwen3-30b-a3b | 95%* | 100% | 100% | 100% | 100% | **99%** |

*Excluding research/83 (crash)

**Discrepancy**: 
- Paper shows **34%-84%** variance across scenarios
- SMTR shows **99%** success (binary ceiling)
- **Root cause**: SMTR uses wrong metric!


## 7. Solutions

### Immediate Fix (Priority 1)

**Read official `task_evaluation` field in `TrajectoryCollector`**:

```python
def _extract_score(self, run_dict: dict) -> float:
    # Priority 1: Official task_evaluation
    task_eval = run_dict.get("task_evaluation")
    if task_eval is not None:
        return self._scale_task_evaluation(task_eval, run_dict.get("scenario"))
    
    # Priority 2: Planning scores (if available)
    planning_scores = run_dict.get("planning_scores", [])
    if planning_scores and all(s != -1 for s in planning_scores):
        return sum(planning_scores) / len(planning_scores)
    
    # Priority 3: Binary fallback (last resort)
    team_success = run_dict.get("team_success", False)
    return float(team_success)

def _scale_task_evaluation(self, task_eval: Any, scenario: str) -> float:
    """Scale official task_evaluation to [0, 1] range."""
    if scenario == "minecraft":
        # task_eval = block_hit_rate * 5 → scale to [0, 1]
        return float(task_eval) / 5.0
    elif scenario in ["research", "coding", "bargaining"]:
        # task_eval = {dim1: 1-5, dim2: 1-5, ...} → average → scale
        if isinstance(task_eval, dict):
            values = [v for v in task_eval.values() if isinstance(v, (int, float))]
            if values:
                avg = sum(values) / len(values)
                return (avg - 1) / 4  # Scale [1, 5] → [0, 1]
        return 0.0
    elif scenario == "database":
        # task_eval = {root_cause: [...], predicted: ...} → compute recall
        # (requires external computation)
        return 0.0  # Placeholder
    return 0.0
```


### Upstream Fix (Priority 2)

**Fix JSONL `environment.name` fields**:

```python
# multiagentbench/database/database_main.jsonl
{
  "environment": {
    "type": "Base",
    "name": "DB Environment",  # ← ADD THIS
    ...
  }
}

# multiagentbench/bargaining/bargaining_main.jsonl
{
  "environment": {
    "type": "Base",
    "name": "World Simulation Environment",  # ← ADD THIS
    ...
  }
}
```

**Or**: Change engine to use `isinstance()` checks (like research/minecraft).


### Environment Fix (Priority 3)

**Fix research evaluator API key issue**:

```python
# Ensure OPENAI_API_KEY is propagated to MARBLE subprocess
# See: docs/audit/research_83_failure_analysis.md
```


## 8. Conclusion

```
CEILING_EFFECT_ROOT_CAUSE = 
    PRIMARY: SMTR ignores official task_evaluation field (R1)
    SECONDARY: Evaluators not triggered (R2, R3, R4) + Binary heuristic (R5)

SOLUTION = 
    1. Read task_evaluation in TrajectoryCollector
    2. Fix environment.name in JSONL files
    3. Propagate API key to subprocess

EXPECTED_IMPACT = 
    Before: 99/100 = 100% success (binary ceiling)
    After: 34%-84% variance (official TS scores)
    
VERDICT = ✅ Ceiling effect is ARTIFICIAL — caused by wrong metric extraction
```

**Priority**: 🔴 **CRITICAL** — This is the single most important fix for SMTR!
