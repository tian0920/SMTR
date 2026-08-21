# MARBLE Causal Signal Audit Report

**Conclusion: SIGNAL_EXISTS**

Oracle ranking = 1.0000 > 0.7, Current SMTR ranking = 0.6989 > 0.65. Both oracle and current representation succeed.

## 1. Intervention Stability

| Metric | Value |
|--------|-------|
| ICC | 0.1653 |
| Verdict | BORDERLINE |
| Exact agreement | 77.52% |
| Groups | 307 |
| Unstable | 69 |

## 2. Oracle Probe

| Metric | Value |
|--------|-------|
| Ranking | 1.0000 |
| Pearson r | 0.9711 |
| Sign accuracy | 0.1275 |
| Verdict | PASS |

## 3. Representation Probe

| Feature Set | Ranking |
|-------------|---------|
| A_current_smtr | 0.6989 |
| B_plus_memory_meta | 0.6955 |
| C_plus_execution | 1.0000 |

## 4. Case Analysis

### positive_transfer

- Cases: 10
- Prediction change rate: 100.00%
- Tasks: ['21', '19', '11', '2', '10']

### negative_transfer

- Cases: 10
- Prediction change rate: 100.00%
- Tasks: ['26', '3', '19']

### neutral_success

- Cases: 10
- Prediction change rate: 30.00%
- Tasks: ['16', '13']

### neutral_failure

- Cases: 10
- Prediction change rate: 20.00%
- Tasks: ['10', '100', '11']

## Next Steps

- MARBLE is suitable for SMTR.
- Proceed to scale experiments.
