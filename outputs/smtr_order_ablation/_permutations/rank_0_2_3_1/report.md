# Formal SMTR Core Ablation

SMTR shares when `tau_mean > 0` and `negative_risk_mean <= epsilon`.

## Main Results

| Method | Success | PosTR | NegTR | NetTR | Opportunity Capture | Safety Preservation | Exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 1.000 | - | - | - | - | - | 0.000 |
| SMTR | 1.000 | 0.000 | 0.000 | 0.000 | - | 1.000 | 0.000 |
| Static-SMTR | 1.000 | 0.000 | 0.000 | 0.000 | - | 1.000 | 0.000 |

## Notes

EffectOnly removes the risk condition. RiskOnly removes the effect condition. Static-SMTR keeps the SMTR gate but freezes critic selected-set conditioning at invocation start.
