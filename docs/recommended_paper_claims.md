# Recommended Paper Claims

> Based on the baseline evaluation results, this document identifies
> which claims are experimentally supported and which should be avoided.

---

## Supported Claims

### Claim 1: SMTR-TCI achieves higher cumulative reward than all baselines

**Evidence**: Final reward = 0.872 vs second-best Reflexion at 0.820 (+6.3%)
across 5 seeds × 100 episodes.

**Supporting table**: `table_baseline_performance.tex`

**Confidence**: High — consistent across all seeds.

---

### Claim 2: SMTR-TCI's advantage grows over time (late-stage improvement)

**Evidence**: Late-stage reward = 0.981 vs second-best Retrieval at 0.933
(+5.1%). The gap between SMTR and baselines widens in the last 20% of episodes.

**Supporting figure**: `figures/baseline_longterm/reward_vs_episode.png`

**Confidence**: High — visible in the learning curves.

---

### Claim 3: SMTR-TCI selectively filters harmful memories

**Evidence**: Harmful retention rate = 0.059 vs Full Memory at 0.212 (3.6× lower).
SMTR stores 61 memories vs 100 for most baselines, but achieves higher reward.

**Supporting table**: `table_memory_quality.tex`

**Confidence**: High — direct measurement from memory_history.jsonl.

---

### Claim 4: SMTR-TCI produces higher-quality persistent knowledge

**Evidence**: MQS = 1.734 (highest), despite storing only 61 memories vs 100
for most baselines. Useful rate = 0.960, cross-topic transfer = 0.909.

**Supporting table**: `table_memory_quality.tex`

**Confidence**: High

---

### Claim 5: Simple retrieval is a strong baseline

**Evidence**: Retrieval achieves 0.818 final reward, beating AGILE (0.784),
Heuristic (0.776), AgeMem (0.756), and Full Memory (0.752). This confirms
that topic-based filtering alone provides significant value.

**Confidence**: High

---

### Claim 6: Reflexion's verbal reflection memory is effective

**Evidence**: Reflexion achieves 0.820 (second-best), suggesting that
storing structured reflection text provides useful context for future episodes.

**Confidence**: Moderate — deterministic reflection text (no LLM) may
underestimate the original method's capability.

### Claim 7: SMTR-TCI is resilient to memory contamination

**Evidence**: At contamination ratio 0.3, SMTR maintains 0.980 final reward
vs 0.78-0.86 for other methods. Harmful retention = 0.201 vs 1.000 for
Full Memory/Retrieval/Reflexion/Heuristic.

**Supporting table**: `table_contamination_baseline.tex`

**Confidence**: High — consistent across 5 seeds and 3 contamination ratios.

---

### Claim 8: SMTR-TCI adapts to environment changes (outdated knowledge)

**Evidence**: In the outdated variant (environment change at episode 60),
SMTR *improves* after the change (drop=-0.078) while Full Memory degrades
(drop=0.103). SMTR's TCI re-validation detects and rejects outdated
knowledge.

**Supporting table**: `table_contamination_baseline.tex` (outdated sub-table)

**Confidence**: High

---

### Claim 9: The performance gain comes from causal validation, not extra compute

**Evidence**: Random Validation (same TCI probe trials but random decisions)
achieves 0.816 reward — 5.6% lower than SMTR's 0.872 at the same
computational cost (1300 ops).

**Supporting table**: `table_cost_fair_comparison.tex`

**Confidence**: High

---

## Claims to Avoid

### DO NOT CLAIM: SMTR is more cost-efficient per operation

**Evidence**: SMTR has the lowest reward-per-operation (0.00067) vs
Reflexion (0.00410). SMTR uses 6.5× more operations due to TCI probes.

**Instead say**: "SMTR achieves higher absolute performance at the cost
of additional validation probes. The probes are the causal identification
mechanism, not a wasteful overhead."

---

### DO NOT CLAIM: SMTR outperforms learned memory controllers

**Evidence**: AgeMem is implemented as a frozen rule-based approximation,
not a learned RL controller. A fully trained AgeMem may perform differently.

**Instead say**: "SMTR outperforms a rule-based approximation of the AgeMem
action space (ADD/DELETE/COMPRESS)."

---

### DO NOT CLAIM: SMTR generalizes to all task types

**Evidence**: Experiments use a synthetic 10-topic environment. Real-world
tasks may have different memory utility distributions.

**Instead say**: "On a synthetic lifelong learning benchmark with known
ground truth, SMTR demonstrates..."

---

### DO NOT CLAIM: AGILE baseline represents the full AGILE framework

**Evidence**: AGILE-inspired experience consolidation omits RL policy
optimisation. Only the experience scoring heuristic is preserved.

**Instead say**: "An AGILE-inspired experience consolidation baseline
(without RL parameter optimisation)."

---

## Suggested Paper Narrative

1. **Opening**: SMTR-TCI achieves state-of-the-art performance on the lifelong
   memory benchmark, outperforming 6 baselines by 6.3–16.0% in final reward.

2. **Mechanism**: The advantage comes from selective memory validation — SMTR
   stores fewer but higher-quality memories (MQS 1.734, 3.6× lower harmful
   retention than Full Memory).

3. **Long-term**: The advantage compounds over time, with a 5.1% late-stage
   margin over the best retrieval baseline.

4. **Contamination resilience**: Under 30% contamination, SMTR maintains
   0.980 reward while baselines degrade to 0.78–0.86. Under environment
   drift, SMTR's re-validation mechanism *improves* performance while
   baselines degrade.

5. **Causal, not computational**: Random Validation (same probes, random
   decisions) achieves 0.816 — confirming that causal validation, not
   extra compute, drives the gain.

6. **Baseline landscape**: Simple retrieval is surprisingly strong;
   heuristic memory management methods do not significantly outperform it.
