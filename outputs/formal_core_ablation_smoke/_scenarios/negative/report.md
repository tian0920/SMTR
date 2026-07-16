# Formal SMTR Core Ablation

SMTR shares when `tau_mean > 0` and `negative_risk_mean <= epsilon`.

## Main Results

| Method | Success | PosTR | NegTR | NetTR | Opportunity Capture | Safety Preservation | Exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 1.000 | - | - | - | - | - | 0.000 |
| B1-Matched | 0.500 | 0.000 | 0.500 | -0.500 | - | 0.500 | 3.500 |
| B1-Top1 | 0.000 | 0.000 | 1.000 | -1.000 | - | 0.000 | 3.000 |
| EffectOnly-SMTR | 0.750 | 0.000 | 0.250 | -0.250 | - | 0.750 | 1.500 |
| RiskOnly-SMTR | 1.000 | 0.000 | 0.000 | 0.000 | - | 1.000 | 0.000 |
| SMTR | 1.000 | 0.000 | 0.000 | 0.000 | - | 1.000 | 0.000 |
| Static-SMTR | 1.000 | 0.000 | 0.000 | 0.000 | - | 1.000 | 0.000 |

## Notes

EffectOnly removes the risk condition. RiskOnly removes the effect condition. Static-SMTR keeps the SMTR gate but freezes critic selected-set conditioning at invocation start.
