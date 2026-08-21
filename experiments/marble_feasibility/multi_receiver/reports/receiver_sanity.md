# Multi-Receiver Heterogeneity Sanity Report

**Verdict: PASS**

- Records: 6 total, 6 valid
- Receivers: ['agent1', 'agent2', 'agent3']
- Tasks: ['1']

## Receiver Effect Variance

| (task, memory) | agent1 | agent2 | agent3 |
|---|---|---|---|
| 1::database_1_harmful | 0 | 0 | 0 |
| 1::database_1_helpful | 0 | 1 | -1 |

**receiver_effect_variance = 0.3333**

## Global vs Receiver-conditioned

- Global τ(m) MAE: 0.5000
- Receiver-conditioned τ(m,r) MAE: n/a (single seed per receiver)
- SMTR beats global: None

## Checks

- [PASS] Records span >= 2 receivers: receivers=['agent1', 'agent2', 'agent3']
- [PASS] receiver_effect_variance > 0: variance=0.3333
- [PASS] at least one memory shows tau(m,r1) != tau(m,r2): heterogeneous=1/2
- [N/A] receiver-conditioned tau(m,r) MAE <= global tau(m) MAE: receiver-conditioned MAE n/a (need >=2 seeds per receiver); global_mae=0.5000