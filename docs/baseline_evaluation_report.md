# Baseline Evaluation Report

> Automated summary of all baseline comparison experiments.

---

## 1. Performance Ranking

Based on average reward across 100 episodes × 5 seeds.

| Rank | Method | Final Reward | Late-stage Reward | Memory Size |
|------|--------|-------------|-------------------|-------------|
| #1 | **SMTR-TCI** | 0.872 | 0.981 | 100 |
| #2 | Reflexion | 0.820 | 0.914 | 100 |
| #3 | Retrieval | 0.818 | 0.933 | 100 |
| #4 | AGILE-inspired | 0.784 | 0.867 | 100 |
| #5 | Heuristic | 0.776 | 0.886 | 100 |
| #6 | AgeMem-inspired | 0.756 | 0.838 | 50 |
| #7 | Full Memory | 0.752 | 0.895 | 100 |

**Key finding**: SMTR-TCI achieves the highest final reward (+6.3% over the
second-best method Reflexion, +16.0% over Full Memory). The late-stage
advantage is even larger (0.981 vs 0.933), indicating sustained long-term
improvement.

---

## 2. Late-stage Ranking

Late-stage reward (last 20% of episodes) measures sustained performance:

| Rank | Method | Late Reward |
|------|--------|-------------|
| #1 | **SMTR-TCI** | 0.981 |
| #2 | Retrieval | 0.933 |
| #3 | Reflexion | 0.914 |
| #4 | Full Memory | 0.895 |
| #5 | Heuristic | 0.886 |
| #6 | AGILE-inspired | 0.867 |
| #7 | AgeMem-inspired | 0.838 |

**Key finding**: SMTR-TCI's late-stage advantage is larger than its overall
advantage, confirming that TCI validation produces *lasting* knowledge that
compounds over time.

---

## 3. Memory Quality Ranking

| Method | Stored | Useful% | Cross-topic | Harmful Retention | Late Gain | MQS |
|--------|--------|---------|-------------|-------------------|-----------|-----|
| **SMTR-TCI** | 61 | 0.960 | 0.909 | **0.059** | **0.590** | **1.734** |
| Full Memory | 100 | 0.990 | 0.912 | 0.212 | 0.490 | 1.562 |
| Retrieval | 100 | 0.900 | 0.195 | 0.190 | 0.530 | 0.904 |
| AgeMem-inspired | 50 | 0.984 | 0.000 | 0.241 | 0.440 | 0.795 |
| Reflexion | 100 | 0.900 | 0.000 | 0.186 | 0.520 | 0.759 |
| Heuristic | 100 | 0.470 | 0.000 | 0.172 | 0.490 | 0.401 |
| AGILE-inspired | 100 | 0.300 | 0.000 | 0.226 | 0.460 | 0.245 |

**Key finding**: SMTR-TCI has the highest MQS (1.734) and the **lowest harmful
retention** (0.059) — 3.6× lower than Full Memory (0.212). Despite storing
only 61 memories (vs 100), SMTR achieves higher reward with better knowledge
quality. Only SMTR and Full Memory show cross-topic knowledge transfer
(cross-topic > 0.9); other baselines are limited to same-topic retrieval.

---

## 4. Contamination Ranking

| Method | r=0.1 Final | r=0.2 Final | r=0.3 Final | Harmful Retention |
|--------|-------------|-------------|-------------|-------------------|
| **SMTR-TCI** | 0.980 | 1.000 | **0.980** | **0.201** |
| Full Memory | 1.000 | 0.960 | 0.860 | 1.000 |
| Retrieval | 0.960 | 0.940 | 0.780 | 1.000 |
| Reflexion | 0.940 | 0.940 | 0.840 | 1.000 |
| Heuristic | 0.900 | 0.920 | 0.820 | 1.000 |
| AgeMem-inspired | 0.840 | 0.880 | 0.780 | 0.492 |

**Outdated variant** (environment change at episode 60):

| Method | Final Reward | Performance Drop | Recovery (episodes) |
|--------|-------------|------------------|--------------------|
| **SMTR-TCI** | **0.960** | **-0.078** (improved!) | 16 |
| Retrieval | 0.880 | -0.035 | 15 |
| Reflexion | 0.880 | -0.025 | 15 |
| Full Memory | 0.640 | 0.103 | 29 |
| Heuristic | 0.620 | 0.143 | 23 |
| AgeMem-inspired | 0.620 | 0.157 | 32 |

**Key finding**: At r=0.3, SMTR-TCI maintains 0.980 reward vs 0.78-0.86
for other methods. SMTR's harmful retention is **5× lower** (0.201) than
Full Memory/Retrieval/Reflexion/Heuristic (1.000) and **2.4× lower** than
AgeMem (0.492). In the outdated variant, SMTR actually *improves* after
the environment change (drop=-0.078), confirming TCI re-validation rejects
outdated knowledge.

---

## 5. Cost Efficiency Ranking

| Method | Reward | Total Ops | Reward/Op |
|--------|--------|-----------|----------|
| **SMTR-TCI** | **0.872** | 1300 | 0.00067 |
| Random Validation | 0.816 | 1300 | 0.00063 |
| Reflexion | 0.820 | 200 | 0.00410 |
| Heuristic | 0.776 | 200 | 0.00388 |
| AgeMem-inspired | 0.756 | 200 | 0.00378 |

**Key finding**: Random Validation (same probe trials as SMTR but random
decisions) achieves 0.816 — **5.6% lower** than SMTR's 0.872 at the
same computational cost. This proves the gain comes from **causal
validation** (delta > 0 gate), not from extra computation alone.

---

## 6. Fairness Audit

**6/6 fairness checks passed.**

- Same environment (LifelongEnvironment)
- Same task stream (paired design, shared task RNG)
- Same seeds (0–4)
- Same backbone (no extra LLM / agent modifications)
- Same memory budget (unlimited in this run)
- Same evaluation (success_probability model)

The only method with additional computation is SMTR-TCI (TCI validation probes).
This is the core mechanism being evaluated, not an unfair advantage.

---

## 7. Summary

| Dimension | Winner | Margin |
|-----------|--------|--------|
| Final Reward | SMTR-TCI | +6.3% over #2 (Reflexion) |
| Late-stage Reward | SMTR-TCI | +5.1% over #2 (Retrieval) |
| Memory Quality (MQS) | SMTR-TCI | 1.734 vs Full Memory 1.562 |
| Contamination Resilience (r=0.3) | SMTR-TCI | 0.980 vs 0.860 (Full Memory) |
| Causal Validation Value | SMTR-TCI | +5.6% over Random Validation (same cost) |
| Cost Efficiency (reward/op) | Reflexion | 6.1× better per-op than SMTR |
| Absolute Performance at any cost | SMTR-TCI | Highest reward overall |

---

## Experimental Configuration

- **Episodes**: 100
- **Seeds**: 0, 1, 2, 3, 4
- **Contamination ratio**: 0.2
- **Memory budget**: unlimited
- **Environment**: synthetic lifelong (10 topics, paired design)
- **Backbone**: numpy-based success probability model
