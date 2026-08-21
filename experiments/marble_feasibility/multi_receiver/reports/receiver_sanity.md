# Multi-Receiver Heterogeneity Sanity Report

**Verdict: FAIL**

- Records: 2 total, 0 valid
- Receivers: []
- Tasks: []

## Receiver Effect Variance

| (task, memory) |  |
|---|

**receiver_effect_variance = 0.0000**

## Global vs Receiver-conditioned

Skipped: only 0 valid records

## Checks

- [FAIL] Records span >= 2 receivers: receivers=[]
- [FAIL] receiver_effect_variance > 0: variance=0.0000
- [FAIL] at least one memory shows tau(m,r1) != tau(m,r2): heterogeneous=0/0
- [FAIL] receiver-conditioned tau(m,r) MAE <= global tau(m) MAE: skipped (insufficient data)