# Claim-Evidence Matrix

> Systematic mapping of every paper claim to its supporting evidence.
> Each claim is rated by confidence based on experimental results.

---

## Claim 1: TCI Forms More Reliable Persistent Knowledge

| Field | Value |
|-------|-------|
| **Claim** | TCI-validated knowledge produces higher long-term reward than all baseline memory controllers |
| **Evidence** | MARBLE main experiment (118 groups, 7 methods, 3 seeds) |
| **Metric** | Mean method reward |
| **Result** | SMTR=0.670 vs best baseline (Heuristic)=0.415, delta=+61.2% |
| **Statistical** | Bootstrap 95% CI: SMTR [0.542, 0.788] vs Heuristic [0.288, 0.542] — no overlap |
| **Experiment** | `results/marble/main/baseline_results.csv` |
| **Table** | `paper/tables/table_marble_main.tex` |
| **Confidence** | **HIGH** — large effect size, non-overlapping CIs |

---

## Claim 2: TCI Reduces Harmful Memory Accumulation

| Field | Value |
|-------|-------|
| **Claim** | SMTR-TCI achieves zero harmful memory retention across contamination ratios |
| **Evidence** | MARBLE contamination experiment (ratios 0.1, 0.2, 0.3) |
| **Metric** | Harmful memory retention rate |
| **Result** | SMTR: 0.000 at all ratios; Full Memory: 0.602 at r=0.3; Retrieval: 0.339 at r=0.3 |
| **Experiment** | `results/marble/contamination/contamination_results.csv` |
| **Table** | `paper/tables/table_marble_contamination.tex` |
| **Confidence** | **HIGH** — zero harmful retention at all noise/contamination levels |

---

## Claim 3: TCI Works Across Different Multi-agent Configurations

| Field | Value |
|-------|-------|
| **Claim** | SMTR's improvement is not a single-domain artifact |
| **Evidence** | Domain-wise analysis (5 domains by agent count) |
| **Metric** | Per-domain mean reward |
| **Result** | SMTR wins 5/5 domains: solo=0.444, small=0.783, medium=0.579, large=0.821, complex=0.600 |
| **Experiment** | `results/marble/domain_analysis/domain_wise_results.csv` |
| **Table** | `paper/tables/table_marble_domain.tex` |
| **Figure** | `figures/marble_domain_performance.pdf` |
| **Confidence** | **HIGH** — wins all 5 domains |

---

## Claim 4: TCI Does Not Require a Perfect Reward Oracle

| Field | Value |
|-------|-------|
| **Claim** | TCI degrades gracefully under noisy reward observations |
| **Evidence** | Noise robustness experiment (sigma 0.0–0.3) |
| **Metric** | Method reward under noise; harmful retention |
| **Result** | SMTR at σ=0.3: reward=0.670, harmful=0.000; Random Validation: reward=0.373, harmful=0.161 |
| **Experiment** | `results/noise_robustness/noise_summary.json` |
| **Table** | `paper/tables/table_noise_robustness.tex` |
| **Figure** | `figures/noise_robustness.pdf` |
| **Confidence** | **HIGH** — reward drops only 3.7% at σ=0.3; harmful retention stays at 0 |

---

## Claim 5: Validated Knowledge Transfers Better Than Unvalidated

| Field | Value |
|-------|-------|
| **Claim** | TCI-validated memories have higher per-injection treatment effect |
| **Evidence** | MARBLE main experiment: treatment effect per injected memory |
| **Metric** | total_tau / n_injected (precision of injection) |
| **Result** | SMTR: 0.3 positive per injection; baselines: 0.2 per injection |
| **Experiment** | `results/marble/main/baseline_results.csv` |
| **Table** | `paper/tables/table_marble_quality.tex` |
| **Confidence** | **MEDIUM** — effect exists but small absolute difference |

---

## Claim 6: Receiver-Conditioned Knowledge Utility

| Field | Value |
|-------|-------|
| **Claim** | Memory utility depends on receiver identity |
| **Evidence** | Receiver analysis (agent1, 303 records) |
| **Metric** | Per-receiver mean tau, disagreement rate |
| **Result** | agent1: mean_tau=-0.006, positive_rate=6.1%, negative_rate=6.7% |
| **Experiment** | `results/marble/main/receiver_conditioned_results.csv` |
| **Confidence** | **LOW** — only 1 receiver (agent1) available; multi-receiver data needed |

---

## Claim 7: Cost Efficiency (from Synthetic Baselines)

| Field | Value |
|-------|-------|
| **Claim** | SMTR achieves higher reward at equal computational cost |
| **Evidence** | Synthetic cost comparison (SMTR vs Random Validation) |
| **Metric** | Reward at equal operations budget |
| **Result** | SMTR=0.872 vs Random Validation=0.816 (same 1300 ops) |
| **Experiment** | `results/baseline_cost_comparison/` |
| **Table** | `paper/tables/table_cost_fair_comparison.tex` |
| **Confidence** | **HIGH** — synthetic experiment with full seeds |

---

## Claim 8: Memory Quality Score Superiority (from Synthetic Baselines)

| Field | Value |
|-------|-------|
| **Claim** | SMTR achieves the highest Memory Quality Score (MQS) |
| **Evidence** | Synthetic memory quality analysis |
| **Metric** | MQS = useful_rate × (1 + transfer_gain) / (1 + harmful_retention) |
| **Result** | SMTR MQS=1.734 > Full Memory MQS=1.562 |
| **Experiment** | `results/memory_quality/memory_quality.csv` |
| **Table** | `paper/tables/table_memory_quality.tex` |
| **Confidence** | **HIGH** — synthetic experiment with full seeds |

---

## Summary

| # | Claim | Confidence | Key Risk |
|---|-------|-----------|----------|
| 1 | TCI forms reliable knowledge | HIGH | — |
| 2 | TCI reduces harmful accumulation | HIGH | — |
| 3 | TCI works across domains | HIGH | — |
| 4 | TCI doesn't need perfect oracle | HIGH | — |
| 5 | Validated > unvalidated transfer | MEDIUM | Small absolute gap |
| 6 | Receiver-conditioned utility | LOW | Single receiver only |
| 7 | Cost efficiency | HIGH | Synthetic only |
| 8 | Memory quality | HIGH | Synthetic only |
