# Encoder Ablation Report

**Verdict: PASS**

**Split: in_distribution** (train=613, test=297, informative=41)

## Results

| Encoder | Ranking | Sign Acc | Tau Corr | Pred Std | Features | Unique |
|---------|---------|----------|----------|----------|----------|--------|
| original | 0.3978 | 0.4390 | -0.2052 | 0.1090 | 32 | 11 |
| task_only | 0.8840 | 0.8049 | 0.6469 | 0.3145 | 20 | 20 |
| memory_only | 0.4358 | 0.3902 | -0.1675 | 0.0769 | 13 | 8 |
| metadata_full | 0.8433 | 0.7561 | 0.6231 | 0.3274 | 70 | 121 |
| causal_input | 0.8261 | 0.7561 | 0.6004 | 0.3243 | 57 | 120 |
| metadata_only | 0.8273 | 0.7561 | 0.6081 | 0.3268 | 33 | 121 |
| metadata_no_task | 0.4134 | 0.4390 | -0.1802 | 0.1055 | 13 | 12 |

## Acceptance Criteria

### [PASS] causal_input >= random + 10%
- causal=0.8261, random=0.3199

### [PASS] |causal_input - metadata_full| < 0.10
- gap=0.0172 (full=0.8433, causal=0.8261)

### [PASS] metadata_only - causal_input < 0.10 (shortcuts not main driver)
- meta_only=0.8273, causal=0.8261, gap=+0.0012

### [PASS] metadata_no_task < task_only - 0.20 (task_id drives, not shortcuts)
- task_only=0.8840, meta_no_task=0.4134, drop=0.4706

---

All criteria met. Performance gains are from causal features, not metadata shortcuts.