# Official Metric Backbone Profile — Unified Report

**Date**: 2026-08-26
**Model**: qwen3-30b-a3b (no_memory baseline)
**Episodes**: 200 (5 scenarios × 20 tasks × 2 seeds)
**Phases covered**: A→B→C→D→E→F→M→N

---

## Executive Summary

The official MultiAgentBench Task Score pipeline has been fully integrated
and validated against the current backbone (qwen3-30b-a3b). Out of 200 episodes,
**196 (98%) produced valid official metric scores**.

**Key finding**: Under the official continuous metric, the backbone is
**NOT saturated at the ceiling** (no scenario has fraction_at_max ≥ 80%),
but **three scenarios (coding, database, minecraft) have near-zero within-scenario
variance**, making them unsuitable for detecting memory effects.

**Decision**: ❌ **BACKBONE_NO_GO** — 1/5 GO criteria failed.
Only 1 of 5 scenarios has within-scenario std ≥ 0.05 (threshold: ≥ 3).

**Recommended next step**: Phase L — backbone difficulty sweep, focusing on
finding task difficulty levels that produce non-trivial score distributions
for coding, database, and minecraft scenarios.

---

## 1. Per-Scenario Statistics

| Scenario | N | Valid Rate | Mean | Std | Min | Max | Q25 | Q75 | At Max | At Min |
|----------|---|-----------|------|-----|-----|-----|-----|-----|--------|--------|
| bargaining | 40 | 100.0% | 0.722 | 0.091 | 0.583 | 0.958 | 0.667 | 0.792 | 0.0% | 0.0% |
| coding | 40 | 100.0% | 0.317 | 0.029 | 0.312 | 0.500 | 0.312 | 0.312 | 0.0% | 0.0% |
| database | 40 | 100.0% | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% |
| minecraft | 40 | 100.0% | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% |
| research | 40 | 90.0% | 0.852 | 0.049 | 0.667 | 0.917 | 0.833 | 0.917 | 0.0% | 0.0% |

### Scenario Diagnosis

**Bargaining** (mean=0.722, std=0.091): ✅ **GOOD** — Healthy spread across
[0.583, 0.958]. The evaluator captures nuanced differences in negotiation
quality (buyer + seller × 3 dimensions = 6 ratings averaged). This is the
best scenario for detecting memory effects.

**Coding** (mean=0.317, std=0.029): ⚠️ **LOW VARIANCE** — Nearly all tasks
score 0.312 (the minimum non-zero value: instruction_following=1,
executability=1, consistency=1, quality=2). The evaluator is very strict and
the LLM-generated code doesn't pass execution checks, causing a floor effect.
One outlier at 0.500 shows some differentiation is possible.

**Database** (mean=0.000, std=0.000): ❌ **FLOOR** — All episodes score 0.
The evaluator checks exact SQL query match against ground truth; the LLM
never generates correct queries for these tasks. Zero variance makes this
scenario useless for memory effect detection.

**Minecraft** (mean=0.000, std=0.000): ❌ **FLOOR** — All episodes score 0.
The minecraft evaluator checks for specific items/actions in the game world
(score.json). The LLM agents cannot interact with the Minecraft environment
effectively enough to produce any score.

**Research** (mean=0.852, std=0.049): ⚠️ **HIGH CEILING** — Most tasks score
0.833 or 0.917 (the two dominant score levels). 4 episodes had invalid
evaluator output (research/18: task_evaluation missing for both seeds).
The evaluator produces a narrow band of scores — the LLM does "good enough"
on most research tasks but rarely achieves perfect scores.

---

## 2. Headroom Distribution

| Scenario | Headroom>5% | Headroom>10% | Headroom>20% |
|----------|-------------|--------------|--------------|
| bargaining | 95.0% | 95.0% | 85.0% |
| coding | 100.0% | 100.0% | 100.0% |
| database | 100.0% | 100.0% | 100.0% |
| minecraft | 100.0% | 100.0% | 100.0% |
| research | 100.0% | 72.2% | 2.8% |

**Interpretation**:
- **Bargaining**: 85% of episodes have >20% headroom → ample room for
  memory to improve scores. Best candidate for TCI pilot.
- **Coding**: 100% headroom>20% — but the floor effect means improvement
  would need to come from a fundamentally different code generation approach,
  not from memory retrieval.
- **Database/Minecraft**: 100% headroom but scores are identically 0.
  These scenarios require a backbone that can at least attempt the tasks.
- **Research**: Only 2.8% have >20% headroom. The model already performs
  near ceiling — memory improvement would be marginal.

---

## 3. Saturation Assessment: Official Metric vs Binary Heuristic

| Scenario | Binary team_success | Official Mean Score | Saturated (official)? |
|----------|:-------------------:|:-------------------:|:--------------------:|
| bargaining | 100% | 0.722 | ❌ NO |
| coding | 100% | 0.317 | ❌ NO |
| database | 100% | 0.000 | ❌ NO |
| minecraft | 100% | 0.000 | ❌ NO |
| research | 90% | 0.852 | ❌ NO |

**Critical insight**: Under the old binary `team_success` heuristic,
4/5 scenarios showed 100% success → appeared fully saturated.
Under the official continuous metric:
- Bargaining: actually 0.722 (28% headroom, NOT saturated)
- Coding: actually 0.317 (68% headroom, NOT saturated)
- Research: actually 0.852 (15% headroom, partially saturated)

The official metric **reveals the true difficulty landscape** that the
binary heuristic was masking.

---

## 4. Backbone GO/NO-GO Decision

| # | Criterion | Value | Threshold | Pass? |
|---|-----------|-------|-----------|:-----:|
| 1 | Valid evaluator rate | 98.0% | ≥ 95% | ✅ |
| 2 | Overall score variance (cross-scenario) | 0.1265 | > 0 | ✅ |
| 3 | Scenarios with within-scenario std ≥ 0.05 | 1/5 | ≥ 3 | ❌ |
| 4 | Avg fraction with headroom > 10% | 93.4% | ≥ 20% | ✅ |
| 5 | Max fraction_at_max | 0.0% | < 80% | ✅ |

**Decision**: ❌ **BACKBONE_NO_GO** (4/5 criteria passed, criterion #3 failed)

### Why criterion #3 failed

Within-scenario std ≥ 0.05 is the measure of whether individual tasks within
a scenario produce meaningfully different scores (necessary for detecting
memory effects). Results:

| Scenario | Within-scenario std | ≥ 0.05? |
|----------|:------------------:|:-------:|
| bargaining | 0.091 | ✅ |
| coding | 0.029 | ❌ |
| database | 0.000 | ❌ |
| minecraft | 0.000 | ❌ |
| research | 0.049 | ❌ |

Only **bargaining** passes. The other 4 scenarios have either zero variance
(database, minecraft: all scores = 0) or near-zero variance (coding: floor
effect at 0.312; research: ceiling clustering at 0.833-0.917).

This means that even with perfect memory, we could not detect improvement
in 4/5 scenarios with this backbone — the signal would be lost in noise.

---

## 5. MARBLE Engine Bugs Found and Fixed

During Phase D profiling, **10 engine integration bugs** were discovered and
patched in the MARBLE codebase (`/home/ecs-user/MARBLE`):

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| 1 | env type="Base" for all scenarios | Raw JSONL configs have generic type | `_SCENARIO_ENV_TYPE` mapping override |
| 2 | evaluate_llm="" (empty string) | Raw configs missing evaluation model | Set to main LLM model |
| 3 | coding: `code_quality` not in `task_evaluation` | Engine stores coding scores separately | Fallback to `code_quality` field |
| 4 | research: empty dict `{}` | Evaluator parse failure → truthy empty dict | Empty dict detection |
| 5 | bargaining: sentinel -1 → normalized=-0.5 | Evaluator parse failure → default ratings | Sentinel detection (`any(v < 1)`) |
| 6 | bargaining: buyer never evaluated | Engine only called seller_prompt | Separate buyer + seller evaluation |
| 7 | JSON parser greedy match | `re.search(r'\{[\s\S]*\}')` matches too much | Balanced-brace scanner |
| 8 | coding: model_name=gpt-3.5-turbo | Tool schema default; agent LLM copies it | Override from env.config['llm'] |
| 9 | coding/engine: wrong solution.py path | Hardcoded `MARBLE/marble/workspace/` | Changed to relative `workspace/` |
| 10 | evaluator: max_token_num=512 | Truncates verbose evaluation before JSON | Increased to 4096 |

**Impact**: These fixes improved evaluator success from **0/200 → 196/200 (98%)**.

---

## 6. Phase M: Scenario Coverage Decision

**Frozen scenario list** (see `docs/audit/scenario_coverage_decision.md`):
- **INCLUDED**: bargaining, coding, database, minecraft, research
- **EXCLUDED**: werewolf (adversarial, not in standard benchmark), web (not in benchmark)

Rationale: These 5 scenarios constitute the official MultiAgentBench
(ACL 2025) evaluation suite.

---

## 7. Phase N: Continual Task Sequence Protocol

**Protocol** (see `docs/audit/continual_task_sequence_protocol.md`):
- Replaces 80/20 train/test split (incompatible with online learning)
- Same-scenario tasks: memory persists across episodes
- Cross-scenario boundary: memory resets
- Deterministic task ordering per seed

---

## 8. Next Steps: Phase L — Backbone Difficulty Sweep

Given BACKBONE_NO_GO, the protocol requires Phase L:

**Goal**: Find task difficulty levels where 3+ scenarios have within-scenario
std ≥ 0.05, enabling memory effect detection.

**Strategy**:
1. **Bargaining** — ✅ Already usable (std=0.091). Keep as-is.
2. **Coding** — Need easier tasks or a more capable backbone where code
   quality scores vary across [0.2, 0.8] instead of clustering at 0.312.
3. **Database** — Need tasks where the LLM can produce at least partially
   correct SQL (non-zero recall). Current tasks are too hard.
4. **Minecraft** — Need tasks achievable without domain-specific tools/knowledge.
   Current tasks produce zero score universally.
5. **Research** — Near ceiling; consider harder research tasks or accept
   that this scenario has limited headroom for improvement.

**Alternative path**: If backbone sweep is too costly, proceed with
**bargaining-only TCI pilot** (Phase H), using the 40 bargaining episodes
as the validated signal-rich scenario.

---

## 9. Reproducibility

### Data
- `results/marble/official_metric_profile/episode_scores.csv` (200 rows)
- `results/marble/official_metric_profile/task_summary.csv` (100 tasks)
- `results/marble/official_metric_profile/scenario_summary.csv` (5 scenarios)
- `results/marble/official_metric_profile/profiling_v2.log` (full log)

### Code
- SMTR: commit `702ccac` (latest)
- MARBLE: patched in-place at `/home/ecs-user/MARBLE`
- Scripts: `experiments/marble_receiver3/run_official_metric_profile.py`,
  `analysis/analyze_official_metric_profile.py`

### Environment
- `source scripts/env_dashscope.sh`
- Model: `openai/qwen3-30b-a3b`
- Total wall-clock: ~11 hours
- API cost: ~200 episodes × avg 200s runtime
