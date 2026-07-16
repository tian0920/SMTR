# Formal SMTR Core Ablation

SMTR shares when `tau_mean > 0` and `negative_risk_mean <= epsilon`.

## Main Results

| Method | Success | PosTR | NegTR | NetTR | Opportunity Capture | Safety Preservation | Exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.000 | - | - | - | - | - | 0.000 |
| B1-Matched | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | 3.500 |
| B1-Top1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | 3.000 |
| EffectOnly-SMTR | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | - | 5.000 |
| RiskOnly-SMTR | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | 3.500 |
| SMTR | 0.250 | 0.250 | 0.000 | 0.250 | 0.250 | - | 3.500 |
| Static-SMTR | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | 3.500 |

## Notes

EffectOnly removes the risk condition. RiskOnly removes the effect condition. Static-SMTR keeps the SMTR gate but freezes critic selected-set conditioning at invocation start.
