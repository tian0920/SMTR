# Memory Intervention Experiment Report

## Summary
- Total runs: 96
- Valid: 96, Invalid: 0, Failed: 0
- Validity rate: 100.0%

## Memory Type Effects

| Memory Type | N Cases | Mean Score Δ | Mean F1 Effect | Positive | Negative | Neutral |
|---|---|---|---|---|---|---|
| beneficial | 24 | -0.0833 | -0.0183 | 5 | 8 | 11 |
| irrelevant | 24 | -0.0833 | +0.0115 | 7 | 6 | 11 |
| conflicting | 24 | +0.0417 | +0.0210 | 7 | 7 | 10 |
| role_mismatched | 24 | +0.0833 | +0.0143 | 6 | 4 | 14 |

## Diagnostic Thresholds
- Positive cases (F1 effect > 0): 25 (need ≥8)
- Negative cases (F1 effect < 0): 25 (need ≥8)
- Positive task coverage: 8 tasks: ['51', '52', '58', '59', '67', '73', '85', '93'] (need ≥3)
- Negative task coverage: 8 tasks: ['51', '52', '58', '59', '67', '73', '85', '93'] (need ≥3)

## Per-Memory-Type Analysis

### beneficial
- Mean F1 effect: -0.0183
- Direction: 5↑ 8↓ 11→

### irrelevant
- Mean F1 effect: +0.0115
- Direction: 7↑ 6↓ 11→

### conflicting
- Mean F1 effect: +0.0210
- Direction: 7↑ 7↓ 10→

### role_mismatched
- Mean F1 effect: +0.0143
- Direction: 6↑ 4↓ 14→

Beneficial avg F1 effect: -0.0183 (target: > 0)
Conflicting avg F1 effect: +0.0210 (target: < 0)

## Overall Assessment: BELOW THRESHOLD

Thresholds met: False
