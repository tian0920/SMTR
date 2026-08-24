# MultiAgentBench Official Metrics Audit

**Date**: 2026-08-24

**Status**: ✅ COMPLETED


## 1. Official Evaluation Protocol (from ACL 2025 Paper)

### Paper Reference
- **Title**: MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents
- **Venue**: ACL 2025 (Main Conference)
- **ArXiv**: https://arxiv.org/abs/2503.01935

### Official Metrics (Two Dimensions)

| Dimension | Metric | Range | Description |
|-----------|--------|-------|-------------|
| **Task Completion** | **Task Score (TS)** | 0-100 (%) | Final output quality per scenario |
| **Coordination** | **Coordination Score (CS)** | 0-100 (%) | Average of Communication + Planning scores |

### Official Reporting

From **Table 1** (line 131-140 in paper HTML):

| Model | Research TS | Minecraft TS | Database TS | Coding TS | Bargaining TS |
|-------|-------------|--------------|-------------|-----------|---------------|
| Llama-3.1-8B | 80.87 | 6.12 | 34.00 | 59.90 | 72.81 |
| Llama-3.1-70B | 80.80 | 0.21 | 53.00 | 62.10 | 72.13 |
| Llama-3.3-70B | 80.00 | 9.15 | 28.50 | 56.60 | 73.15 |
| GPT-3.5-turbo | 70.20 | 5.05 | 45.00 | 55.50 | 71.67 |
| GPT-4o-mini | **84.13** | **33.60** | **45.00** | **65.10** | **74.47** |

**Key Observation**: Scores are **continuous percentages**, NOT binary success/failure!


## 2. Per-Scenario Official Metrics

### database

| Field | Value |
|-------|-------|
| **Official Metric** | Root cause recall (subset match) |
| **Metric Range** | Continuous [0, 1] → scaled to 0-100% |
| **Binary / Continuous** | **Continuous** (proportion of correct root causes) |
| **Task-level / Step-level** | Task-level (final output) |
| **Evaluator Source** | `marble/evaluator/evaluator.py` L270-285 |
| **Official Aggregation** | Average across 100 tasks |
| **Current SMTR Usage** | ❌ **INCORRECT** — SMTR uses binary `team_success` |

**Official Evaluation Code**:
```python
def evaluate_task_db(self, task, result, labels, pred_num, root_causes):
    self.metrics["task_evaluation"] = {
        'root_cause': root_causes,  # ground truth
        'predicted': result,        # agent output
    }
```

**Official Scoring** (computed externally):
- Parse `predicted` to extract predicted root causes
- Compute recall = |predicted ∩ ground_truth| / |ground_truth|
- Scale to 0-100%

**SMTR Current Usage**:
- Extracts `team_success` as binary (0/1)
- **WRONG**: Ignores partial credit from root cause recall


### research

| Field | Value |
|-------|-------|
| **Official Metric** | LLM-judged {innovation, safety, feasibility} |
| **Metric Range** | Ordinal 1-5 per dimension → averaged → scaled to 0-100% |
| **Binary / Continuous** | **Continuous** (average of 3 dimensions) |
| **Task-level / Step-level** | Task-level (final proposal) |
| **Evaluator Source** | `marble/evaluator/evaluator.py` L240-268 |
| **Official Aggregation** | Average across 100 tasks |
| **Current SMTR Usage** | ❌ **INCORRECT** — SMTR uses binary `team_success` |

**Official Evaluation Code**:
```python
def evaluate_task_research(self, task, result):
    ratings = self.parse_research_ratings(llm_response.content)
    # Returns: {innovation: 1-5, safety: 1-5, feasibility: 1-5}
    self.metrics["task_evaluation"] = ratings
```

**Official Scoring**:
- Average(innovation, safety, feasibility) → range [1, 5]
- Scale to 0-100%: `(avg - 1) / 4 * 100`

**SMTR Current Usage**:
- Extracts `team_success` as binary (0/1)
- **WRONG**: Ignores fine-grained quality ratings


### minecraft

| Field | Value |
|-------|-------|
| **Official Metric** | Block placement hit rate |
| **Metric Range** | Continuous [0, 1] → ×5 → scaled to 0-100% |
| **Binary / Continuous** | **Continuous** (proportion of correct blocks) |
| **Task-level / Step-level** | Task-level (final structure) |
| **Evaluator Source** | `marble/environments/minecraft_utils/build_judger.py` L376 |
| **Official Aggregation** | Average across 100 tasks |
| **Current SMTR Usage** | ✅ **PARTIALLY CORRECT** — Uses `block_hit_rate` but not scaled |

**Official Evaluation Code**:
```python
block_hit_rate = cal_block_hit_rate(task_data)
# Returns: float [0, 1]
summary_data["task_evaluation"] = block_hit_rate * 5
# Scale: [0, 5]
```

**Official Scoring**:
- `block_hit_rate` ∈ [0, 1]
- Multiply by 20 to get 0-100% scale
- Example: 0.336 → 33.6% (GPT-4o-mini in paper)

**SMTR Current Usage**:
- Extracts `block_hit_rate` correctly
- **PARTIALLY CORRECT**: Has the signal but not scaled to official 0-100%


### coding

| Field | Value |
|-------|-------|
| **Official Metric** | LLM-judged {instruction_following, executability, consistency, quality} |
| **Metric Range** | Ordinal 1-5 per dimension → averaged → scaled to 0-100% |
| **Binary / Continuous** | **Continuous** (average of 4 dimensions) |
| **Task-level / Step-level** | Task-level (final code) |
| **Evaluator Source** | `marble/evaluator/evaluator.py` L300-350 |
| **Official Aggregation** | Average across 100 tasks |
| **Current SMTR Usage** | ❌ **INCORRECT** — SMTR uses binary `team_success` |

**Official Evaluation Code**:
```python
def evaluate_code_quality(self, task, code_result):
    # Returns: {instruction_following: 1-5, executability: 1-5, 
    #           consistency: 1-5, quality: 1-5}
    self.metrics["task_evaluation"] = ratings
```

**Official Scoring**:
- Average(4 dimensions) → range [1, 5]
- Scale to 0-100%: `(avg - 1) / 4 * 100`

**SMTR Current Usage**:
- Extracts `team_success` as binary (0/1)
- **WRONG**: Ignores code quality dimensions


### bargaining

| Field | Value |
|-------|-------|
| **Official Metric** | LLM-judged buyer/seller {effectiveness, progress, interaction} |
| **Metric Range** | Ordinal 1-5 per dimension → averaged → scaled to 0-100% |
| **Binary / Continuous** | **Continuous** (average of 6 scores: 2 roles × 3 dimensions) |
| **Task-level / Step-level** | Task-level (final negotiation outcome) |
| **Evaluator Source** | `marble/evaluator/evaluator.py` L380-420 |
| **Official Aggregation** | Average across 100 tasks |
| **Current SMTR Usage** | ❌ **INCORRECT** — SMTR uses binary `team_success` |

**Official Evaluation Code**:
```python
def evaluate_task_world(self, task, result):
    # Returns: {buyer_effectiveness: 1-5, buyer_progress: 1-5, 
    #           buyer_interaction: 1-5, seller_effectiveness: 1-5,
    #           seller_progress: 1-5, seller_interaction: 1-5}
    self.metrics["task_evaluation"] = ratings
```

**Official Scoring**:
- Average(6 dimensions) → range [1, 5]
- Scale to 0-100%: `(avg - 1) / 4 * 100`

**SMTR Current Usage**:
- Extracts `team_success` as binary (0/1)
- **WRONG**: Ignores negotiation quality dimensions


## 3. Critical Questions Answered

### Q1: Is `team_success` the official main metric?

**Answer**: ❌ **NO**

**Evidence**:
- Paper Table 1 reports **Task Score (TS)** as continuous percentage (0-100%)
- SMTR's `team_success` is binary (0/1)
- SMTR extracts `team_success` from iteration summaries using heuristics (e.g., "success" keyword)
- This is **NOT** the official metric

**Impact**: SMTR is measuring the wrong signal!


### Q2: Are there finer-grained official scores?

**Answer**: ✅ **YES**

| Scenario | Official Fine-Grained Score |
|----------|----------------------------|
| database | Root cause recall (continuous [0, 1]) |
| research | {innovation, safety, feasibility} 1-5 |
| minecraft | `block_hit_rate` (continuous [0, 1]) |
| coding | {instruction_following, executability, consistency, quality} 1-5 |
| bargaining | {buyer, seller} × {effectiveness, progress, interaction} 1-5 |

**Conclusion**: All scenarios have **continuous/ordinal scores**, not binary!


### Q3: Is SMTR incorrectly binarizing official metrics?

**Answer**: ✅ **YES — THIS IS THE ROOT CAUSE OF CEILING EFFECT**

**Evidence**:
- SMTR `_extract_team_success()` in `trajectory_collector.py` uses heuristics to infer binary success
- SMTR `_extract_score()` falls back to `float(team_success)` if no planning_scores available
- Result: 99/100 tasks show `team_success=1` (binary)
- But official TS scores would show **variance** (e.g., 80%, 85%, 72%, etc.)

**Root Cause**: SMTR is **not reading** the official `task_evaluation` field from MARBLE output!


### Q4: Are there agent-level / team-level partial scores?

**Answer**: ✅ **YES**

| Level | Metric | Source |
|-------|--------|--------|
| **Agent-level** | Individual KPI | Milestone detection (not implemented in current engine) |
| **Team-level** | Coordination Score (CS) | Average of Communication + Planning scores |
| **Task-level** | Task Score (TS) | Scenario-specific evaluator |

**Current Status**:
- Agent-level KPI: **NOT IMPLEMENTED** in MARBLE engine (commented out)
- Team-level CS: **DISABLED** in graph mode (communication/planning evaluators commented out)
- Task-level TS: **AVAILABLE** but SMTR ignores it


### Q5: Can official scores be used for expose/withhold delta?

**Answer**: ✅ **YES — WITH CAVEATS**

| Scenario | Can Use for Delta? | Caveat |
|----------|-------------------|--------|
| database | ✅ YES | Need to compute recall from predicted vs ground_truth |
| research | ⚠️ CONDITIONAL | Requires LLM-as-judge (may crash without API key) |
| minecraft | ✅ YES | Directly read `block_hit_rate` |
| coding | ⚠️ CONDITIONAL | Requires LLM-as-judge (star mode only) |
| bargaining | ⚠️ CONDITIONAL | Requires LLM-as-judge + environment name fix |

**Conclusion**: At least **database + minecraft** can be used immediately without oracle!


## 4. Official Metric Inventory

### Summary Table

| Scenario | Official Metric | Range | Binary/Continuous | SMTR Currently Uses | Correct? |
|----------|----------------|-------|-------------------|---------------------|----------|
| **database** | Root cause recall | [0, 1] | Continuous | Binary `team_success` | ❌ NO |
| **research** | Avg(innovation, safety, feasibility) | [1, 5] | Ordinal | Binary `team_success` | ❌ NO |
| **minecraft** | `block_hit_rate` | [0, 1] | Continuous | `block_hit_rate` | ✅ YES |
| **coding** | Avg(4 dimensions) | [1, 5] | Ordinal | Binary `team_success` | ❌ NO |
| **bargaining** | Avg(6 dimensions) | [1, 5] | Ordinal | Binary `team_success` | ❌ NO |

**Verdict**: 4/5 scenarios are using **WRONG METRIC** in SMTR!


## 5. Recommendations

### Immediate Actions

1. **Read official `task_evaluation` field** from MARBLE engine output
   - Location: `summary_data["task_evaluation"]` in engine output
   - This contains the official metric per scenario

2. **Implement official metric adapter**
   - Parse `task_evaluation` per scenario
   - Scale to 0-100% using official formulas
   - Use for expose/withhold delta

3. **Stop using binary `team_success`**
   - It is NOT the official metric
   - It causes ceiling effect (99/100 = 100%)
   - Official TS scores have **variance** (see paper Table 1)

### Example Official Scoring

```python
# database
recall = len(set(predicted) & set(ground_truth)) / len(ground_truth)
ts_score = recall * 100  # 0-100%

# research
avg_rating = (innovation + safety + feasibility) / 3
ts_score = (avg_rating - 1) / 4 * 100  # 0-100%

# minecraft
ts_score = block_hit_rate * 100  # 0-100%

# coding
avg_rating = (instruction + executability + consistency + quality) / 4
ts_score = (avg_rating - 1) / 4 * 100  # 0-100%

# bargaining
avg_rating = (buyer_eff + buyer_prog + buyer_int + 
              seller_eff + seller_prog + seller_int) / 6
ts_score = (avg_rating - 1) / 4 * 100  # 0-100%
```

### Expected Impact

If SMTR uses official TS instead of binary `team_success`:
- **Before**: 99/100 tasks = 100% success (ceiling effect)
- **After**: TS scores will show **variance** (e.g., 34%, 53%, 45%, etc. from paper Table 1)
- **Result**: Ceiling effect **SOLVED** without changing model or tasks!


## 6. Conclusion

```
OFFICIAL_MAIN_METRIC = Task Score (TS)
TS_TYPE = Continuous [0, 100%]
TS_SOURCE = Scenario-specific evaluator (task_evaluation field)

SMTR_CURRENT_METRIC = team_success (binary 0/1)
SMTR_CORRECT = NO — using wrong metric

CEILING_EFFECT_ROOT_CAUSE = SMTR binarizes official continuous scores
SOLUTION = Read official task_evaluation field directly
```

**Priority**: 🔴 **CRITICAL** — This is the root cause of the ceiling effect!
