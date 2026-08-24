# research/83 Failure Analysis (Phase 9)

**Date**: 2026-08-24

**Classification**: **B. Evaluator Failure + C. Engine Failure**

**Verdict**: research/83 is NOT a genuine hard task. It must NOT be counted as a hard task.


## Evidence

### 1. Raw Engine Output

**File**: `/tmp/smtr_traj_d53j9wnm/e40dd839756ba516289e69eb/marble_output.jsonl`

```json
{
  "iterations": [],        // EMPTY — 0 iterations
  "final_output": "",      // EMPTY — no output
  "token_usage": N/A       // No tokens consumed
}
```

### 2. Stderr (engine log)

```
beartype.roar.BeartypeCallHintReturnViolation: 
  Function marble.llms.model_prompting.model_prompting() return "None" 
  violates type hint list[litellm.types.utils.Message], 
  as <class "builtins.NoneType"> "None" not instance of list.
```

**Location**: `engine_planner.py` line 376, called from `engine.py` line 519 (`star_coordinate`)

### 3. Stdout (engine log)

```
Attempt 1 failed: litellm.InternalServerError: 
  InternalServerError: OpenAIException - Missing credentials. 
  Please pass an api_key...
...
Attempt 5 failed: litellm.InternalServerError: ...
Failed to execute 'model_prompting' after 5 retries.
```

**Root cause**: OPENAI_API_KEY not propagated to MARBLE engine subprocess.

### 4. Execution Metrics

| Metric | research/83 | research/88 (normal) |
|--------|-------------|---------------------|
| Execution time | ~38s | ~285s |
| Iterations | 0 | 5 |
| Token usage | 0 | 194,655 |
| team_success | False | True |
| coordinate_mode | star | star |

### 5. Failure Chain

```
1. OPENAI_API_KEY missing in subprocess
   ↓
2. litellm.completion() fails 5 times → returns None
   ↓
3. model_prompting() returns None (violates beartype type hint)
   ↓
4. beartype raises BeartypeCallHintReturnViolation
   ↓
5. star_coordinate() catches exception, writes empty output
   ↓
6. iterations=[] → _extract_team_success() returns False
   ↓
7. team_success=False, reward=0.0
```


## Classification Rationale

| Category | Evidence | Match? |
|----------|----------|--------|
| A. Genuine agent failure | No agent work was done (0 iterations) | ❌ No |
| B. Evaluator failure | Beartype type violation in engine_planner | ✅ Yes |
| C. Engine failure | API key not propagated to subprocess | ✅ Yes |
| D. Unknown | Root cause fully identified | ❌ No |


## Action

**research/83 MUST NOT be counted as a hard task.**

The reward=0 result is entirely caused by infrastructure failure (missing API key in subprocess), not task difficulty.

This is consistent with the observation that ALL other research tasks (19/20) achieve reward=1.0, confirming that the research domain is easy for qwen3-30b-a3b when the engine runs correctly.
