# B0/B1/M0 Full Scenario Experiment Report

**Critic checkpoint**: `checkpoints/critic_pi3_v22.joblib`  
**Episodes per scenario**: 20  
**Task seeds**: [0, 1, 2, 3, 4]  
**Generation seeds**: [0, 1]  
**Traversal seeds**: [0, 1, 2]  
**Top-k**: 4, **Max shares/invocation**: 3  

## Cross-Scenario Summary

| Scenario | B0 SR | B1 SR | M0 SR | B1 NegTR | M0 NegTR | B1 PosTR | M0 PosTR | M0 ShareRate |
|----------|-------|-------|-------|----------|----------|----------|----------|--------------|
| Positive Transfer | 0.000 | 0.000 | 0.200 | 0.0% | 0.0% | 0.0% | 20.0% | 20.0% |
| Negative Transfer | 1.000 | 0.000 | 1.000 | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Neutral (Success) | 1.000 | 1.000 | 1.000 | 0.0% | 0.0% | 0.0% | 0.0% | 30.0% |
| Neutral (Failure) | 0.000 | 0.000 | 0.000 | 0.0% | 0.0% | 0.0% | 0.0% | 30.0% |
| Prefix-Sensitive | 0.000 | 0.000 | 0.000 | 0.0% | 0.0% | 0.0% | 0.0% | 50.0% |
| Flip: Pos→Neg | 1.000 | 0.000 | 1.000 | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Flip: Neg→Pos | 0.000 | 0.000 | 0.000 | 0.0% | 0.0% | 0.0% | 0.0% | 26.7% |
| Flip: Neu→Neg | 0.000 | 0.000 | 0.200 | 0.0% | 0.0% | 0.0% | 20.0% | 13.3% |
| Flip: Neu→Pos | 1.000 | 0.000 | 0.800 | 100.0% | 20.0% | 0.0% | 0.0% | 13.3% |

## Per-Scenario Details

### Positive Transfer (`positive`)

*Elapsed: 10.19s*

**B0 (NoMemoryRouter)**:
- Episodes: 40
- Success rate: 0.000
- Avg selected: 0.0
- All-withhold rate: 1.000

**B1 (RelevanceTopKRouter)**:
- Episodes: 40
- Success rate: 0.000
- Avg selected: 7.0
- All-withhold rate: 0.000
- Positive transfer: 0.000
- Negative transfer: 0.000
- Neutral success: 0.000
- Neutral failure: 1.000
- Success delta vs B0: 0.000

**M0 (ProductionSequentialRouter)**:
- Episodes: 120
- Success rate: 0.200
- Avg selected: 1.4
- All-withhold rate: 0.600
- Positive transfer: 0.200
- Negative transfer: 0.000
- Neutral success: 0.000
- Neutral failure: 0.800
- Success delta vs B0: 0.200
- Share decision rate: 0.200
- τ-LCB rejection rate: 0.000
- Neg-risk UCB rejection rate: 0.450
- Budget rejection rate: 0.017
- Low-support rejection rate: 0.000
- Other rejection reasons: {'tau_lcb_nonpositive': 480}

**M0 vs B1**:
- Success difference: 0.200
- Neg-transfer diff: 0.000
- Pos-transfer diff: 0.200
- Avg selected diff: -5.6

**Bootstrap 95% CI**:
- b0_success_rate: mean=0.000 [0.000, 0.000]
- b1_success_rate: mean=0.000 [0.000, 0.000]
- m0_success_rate: mean=0.199 [0.075, 0.350]

---

### Negative Transfer (`negative`)

*Elapsed: 10.09s*

**B0 (NoMemoryRouter)**:
- Episodes: 40
- Success rate: 1.000
- Avg selected: 0.0
- All-withhold rate: 1.000

**B1 (RelevanceTopKRouter)**:
- Episodes: 40
- Success rate: 0.000
- Avg selected: 7.0
- All-withhold rate: 0.000
- Positive transfer: 0.000
- Negative transfer: 1.000
- Neutral success: 0.000
- Neutral failure: 0.000
- Success delta vs B0: -1.000

**M0 (ProductionSequentialRouter)**:
- Episodes: 120
- Success rate: 1.000
- Avg selected: 0.0
- All-withhold rate: 1.000
- Positive transfer: 0.000
- Negative transfer: 0.000
- Neutral success: 1.000
- Neutral failure: 0.000
- Success delta vs B0: 0.000
- Share decision rate: 0.000
- τ-LCB rejection rate: 0.000
- Neg-risk UCB rejection rate: 0.000
- Budget rejection rate: 0.000
- Low-support rejection rate: 0.000
- Other rejection reasons: {'tau_lcb_nonpositive': 1440}

**M0 vs B1**:
- Success difference: 1.000
- Neg-transfer diff: -1.000
- Pos-transfer diff: 0.000
- Avg selected diff: -7.0

**Bootstrap 95% CI**:
- b0_success_rate: mean=1.000 [1.000, 1.000]
- b1_success_rate: mean=0.000 [0.000, 0.000]
- m0_success_rate: mean=1.000 [1.000, 1.000]

---

### Neutral (Success) (`neutral_success`)

*Elapsed: 10.21s*

**B0 (NoMemoryRouter)**:
- Episodes: 40
- Success rate: 1.000
- Avg selected: 0.0
- All-withhold rate: 1.000

**B1 (RelevanceTopKRouter)**:
- Episodes: 40
- Success rate: 1.000
- Avg selected: 7.0
- All-withhold rate: 0.000
- Positive transfer: 0.000
- Negative transfer: 0.000
- Neutral success: 1.000
- Neutral failure: 0.000
- Success delta vs B0: 0.000

**M0 (ProductionSequentialRouter)**:
- Episodes: 120
- Success rate: 1.000
- Avg selected: 1.6
- All-withhold rate: 0.000
- Positive transfer: 0.000
- Negative transfer: 0.000
- Neutral success: 1.000
- Neutral failure: 0.000
- Success delta vs B0: 0.000
- Share decision rate: 0.300
- τ-LCB rejection rate: 0.000
- Neg-risk UCB rejection rate: 0.000
- Budget rejection rate: 0.000
- Low-support rejection rate: 0.000
- Other rejection reasons: {'tau_lcb_nonpositive': 1008}

**M0 vs B1**:
- Success difference: 0.000
- Neg-transfer diff: 0.000
- Pos-transfer diff: 0.000
- Avg selected diff: -5.4

**Bootstrap 95% CI**:
- b0_success_rate: mean=1.000 [1.000, 1.000]
- b1_success_rate: mean=1.000 [1.000, 1.000]
- m0_success_rate: mean=1.000 [1.000, 1.000]

---

### Neutral (Failure) (`neutral_failure`)

*Elapsed: 10.14s*

**B0 (NoMemoryRouter)**:
- Episodes: 40
- Success rate: 0.000
- Avg selected: 0.0
- All-withhold rate: 1.000

**B1 (RelevanceTopKRouter)**:
- Episodes: 40
- Success rate: 0.000
- Avg selected: 7.0
- All-withhold rate: 0.000
- Positive transfer: 0.000
- Negative transfer: 0.000
- Neutral success: 0.000
- Neutral failure: 1.000
- Success delta vs B0: 0.000

**M0 (ProductionSequentialRouter)**:
- Episodes: 120
- Success rate: 0.000
- Avg selected: 1.8
- All-withhold rate: 0.000
- Positive transfer: 0.000
- Negative transfer: 0.000
- Neutral success: 0.000
- Neutral failure: 1.000
- Success delta vs B0: 0.000
- Share decision rate: 0.300
- τ-LCB rejection rate: 0.000
- Neg-risk UCB rejection rate: 0.000
- Budget rejection rate: 0.017
- Low-support rejection rate: 0.000
- Other rejection reasons: {'tau_lcb_nonpositive': 984}

**M0 vs B1**:
- Success difference: 0.000
- Neg-transfer diff: 0.000
- Pos-transfer diff: 0.000
- Avg selected diff: -5.2

**Bootstrap 95% CI**:
- b0_success_rate: mean=0.000 [0.000, 0.000]
- b1_success_rate: mean=0.000 [0.000, 0.000]
- m0_success_rate: mean=0.000 [0.000, 0.000]

---

### Prefix-Sensitive (`prefix_sensitive`)

*Elapsed: 10.2s*

**B0 (NoMemoryRouter)**:
- Episodes: 40
- Success rate: 0.000
- Avg selected: 0.0
- All-withhold rate: 1.000

**B1 (RelevanceTopKRouter)**:
- Episodes: 40
- Success rate: 0.000
- Avg selected: 5.0
- All-withhold rate: 0.000
- Positive transfer: 0.000
- Negative transfer: 0.000
- Neutral success: 0.000
- Neutral failure: 1.000
- Success delta vs B0: 0.000

**M0 (ProductionSequentialRouter)**:
- Episodes: 120
- Success rate: 0.000
- Avg selected: 2.4
- All-withhold rate: 0.000
- Positive transfer: 0.000
- Negative transfer: 0.000
- Neutral success: 0.000
- Neutral failure: 1.000
- Success delta vs B0: 0.000
- Share decision rate: 0.500
- τ-LCB rejection rate: 0.000
- Neg-risk UCB rejection rate: 0.000
- Budget rejection rate: 0.017
- Low-support rejection rate: 0.000
- Other rejection reasons: {'tau_lcb_nonpositive': 696}

**M0 vs B1**:
- Success difference: 0.000
- Neg-transfer diff: 0.000
- Pos-transfer diff: 0.000
- Avg selected diff: -2.6

**Bootstrap 95% CI**:
- b0_success_rate: mean=0.000 [0.000, 0.000]
- b1_success_rate: mean=0.000 [0.000, 0.000]
- m0_success_rate: mean=0.000 [0.000, 0.000]

---

### Flip: Pos→Neg (`flip_pos_to_neg`)

*Elapsed: 10.18s*

**B0 (NoMemoryRouter)**:
- Episodes: 40
- Success rate: 1.000
- Avg selected: 0.0
- All-withhold rate: 1.000

**B1 (RelevanceTopKRouter)**:
- Episodes: 40
- Success rate: 0.000
- Avg selected: 8.0
- All-withhold rate: 0.000
- Positive transfer: 0.000
- Negative transfer: 1.000
- Neutral success: 0.000
- Neutral failure: 0.000
- Success delta vs B0: -1.000

**M0 (ProductionSequentialRouter)**:
- Episodes: 120
- Success rate: 1.000
- Avg selected: 0.0
- All-withhold rate: 1.000
- Positive transfer: 0.000
- Negative transfer: 0.000
- Neutral success: 1.000
- Neutral failure: 0.000
- Success delta vs B0: 0.000
- Share decision rate: 0.000
- τ-LCB rejection rate: 0.000
- Neg-risk UCB rejection rate: 0.000
- Budget rejection rate: 0.000
- Low-support rejection rate: 0.000
- Other rejection reasons: {'tau_lcb_nonpositive': 1440}

**M0 vs B1**:
- Success difference: 1.000
- Neg-transfer diff: -1.000
- Pos-transfer diff: 0.000
- Avg selected diff: -8.0

**Bootstrap 95% CI**:
- b0_success_rate: mean=1.000 [1.000, 1.000]
- b1_success_rate: mean=0.000 [0.000, 0.000]
- m0_success_rate: mean=1.000 [1.000, 1.000]

---

### Flip: Neg→Pos (`flip_neg_to_pos`)

*Elapsed: 9.91s*

**B0 (NoMemoryRouter)**:
- Episodes: 40
- Success rate: 0.000
- Avg selected: 0.0
- All-withhold rate: 1.000

**B1 (RelevanceTopKRouter)**:
- Episodes: 40
- Success rate: 0.000
- Avg selected: 8.0
- All-withhold rate: 0.000
- Positive transfer: 0.000
- Negative transfer: 0.000
- Neutral success: 0.000
- Neutral failure: 1.000
- Success delta vs B0: 0.000

**M0 (ProductionSequentialRouter)**:
- Episodes: 120
- Success rate: 0.000
- Avg selected: 2.6
- All-withhold rate: 0.600
- Positive transfer: 0.000
- Negative transfer: 0.000
- Neutral success: 0.000
- Neutral failure: 1.000
- Success delta vs B0: 0.000
- Share decision rate: 0.267
- τ-LCB rejection rate: 0.000
- Neg-risk UCB rejection rate: 0.367
- Budget rejection rate: 0.050
- Low-support rejection rate: 0.000
- Other rejection reasons: {'tau_lcb_nonpositive': 456}

**M0 vs B1**:
- Success difference: 0.000
- Neg-transfer diff: 0.000
- Pos-transfer diff: 0.000
- Avg selected diff: -5.4

**Bootstrap 95% CI**:
- b0_success_rate: mean=0.000 [0.000, 0.000]
- b1_success_rate: mean=0.000 [0.000, 0.000]
- m0_success_rate: mean=0.000 [0.000, 0.000]

---

### Flip: Neu→Neg (`flip_neu_to_neg`)

*Elapsed: 10.09s*

**B0 (NoMemoryRouter)**:
- Episodes: 40
- Success rate: 0.000
- Avg selected: 0.0
- All-withhold rate: 1.000

**B1 (RelevanceTopKRouter)**:
- Episodes: 40
- Success rate: 0.000
- Avg selected: 8.0
- All-withhold rate: 0.000
- Positive transfer: 0.000
- Negative transfer: 0.000
- Neutral success: 0.000
- Neutral failure: 1.000
- Success delta vs B0: 0.000

**M0 (ProductionSequentialRouter)**:
- Episodes: 120
- Success rate: 0.200
- Avg selected: 1.2
- All-withhold rate: 0.600
- Positive transfer: 0.200
- Negative transfer: 0.000
- Neutral success: 0.000
- Neutral failure: 0.800
- Success delta vs B0: 0.200
- Share decision rate: 0.133
- τ-LCB rejection rate: 0.000
- Neg-risk UCB rejection rate: 0.117
- Budget rejection rate: 0.017
- Low-support rejection rate: 0.000
- Other rejection reasons: {'tau_lcb_nonpositive': 1056}

**M0 vs B1**:
- Success difference: 0.200
- Neg-transfer diff: 0.000
- Pos-transfer diff: 0.200
- Avg selected diff: -6.8

**Bootstrap 95% CI**:
- b0_success_rate: mean=0.000 [0.000, 0.000]
- b1_success_rate: mean=0.000 [0.000, 0.000]
- m0_success_rate: mean=0.199 [0.075, 0.350]

---

### Flip: Neu→Pos (`flip_neu_to_pos`)

*Elapsed: 9.96s*

**B0 (NoMemoryRouter)**:
- Episodes: 40
- Success rate: 1.000
- Avg selected: 0.0
- All-withhold rate: 1.000

**B1 (RelevanceTopKRouter)**:
- Episodes: 40
- Success rate: 0.000
- Avg selected: 8.0
- All-withhold rate: 0.000
- Positive transfer: 0.000
- Negative transfer: 1.000
- Neutral success: 0.000
- Neutral failure: 0.000
- Success delta vs B0: -1.000

**M0 (ProductionSequentialRouter)**:
- Episodes: 120
- Success rate: 0.800
- Avg selected: 1.2
- All-withhold rate: 0.600
- Positive transfer: 0.000
- Negative transfer: 0.200
- Neutral success: 0.800
- Neutral failure: 0.000
- Success delta vs B0: -0.200
- Share decision rate: 0.133
- τ-LCB rejection rate: 0.000
- Neg-risk UCB rejection rate: 0.117
- Budget rejection rate: 0.017
- Low-support rejection rate: 0.000
- Other rejection reasons: {'tau_lcb_nonpositive': 1056}

**M0 vs B1**:
- Success difference: 0.800
- Neg-transfer diff: -0.800
- Pos-transfer diff: 0.000
- Avg selected diff: -6.8

**Bootstrap 95% CI**:
- b0_success_rate: mean=1.000 [1.000, 1.000]
- b1_success_rate: mean=0.000 [0.000, 0.000]
- m0_success_rate: mean=0.799 [0.675, 0.901]

---

## Analysis

### Overall Average Success Rates

| Method | Avg Success Rate |
|--------|-----------------|
| B0 | 0.444 |
| B1 | 0.111 |
| M0 | 0.467 |

### Transfer Occurrence

| Metric | B1 | M0 |
|--------|----|----|
| Scenarios with positive transfer | 0/9 | 2/9 |
| Scenarios with negative transfer | 3/9 | 1/9 |
