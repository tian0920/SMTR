# Baseline Fairness Report

**Results directory**: results/baseline_comparison/formation
**Checks passed**: 6/6

## Fairness Checks

| # | Check | Status |
|---|-------|--------|
| 1 | Same environment (config exists) | PASS |
| 2 | Same episode count (parity) | PASS |
| 3 | Same task sequence (paired design) | PASS |
| 4 | No extra information access | PASS (see notes) |
| 5 | Same backbone (LifelongEnvironment) | PASS |
| 6 | Same memory budget | PASS |

## Methods Audited

- AgeMem-inspired (`agemem`)
- AGILE-inspired (`agile`)
- Full Memory (`full_memory`)
- Heuristic (`heuristic`)
- Reflexion (`reflexion`)
- Retrieval (`retrieval`)
- SMTR-TCI (`smtr_tci`)

## Warnings & Notes

- INFO: SMTR-TCI has additional information: TCI validation probes (expose/withhold trials) — this is the core mechanism being evaluated, not unfair advantage

## Conclusion

**6/6 fairness checks passed.**
All baselines share the same environment, task stream, seeds, evaluation model, and memory budget. SMTR-TCI uses TCI validation probes (additional computation), which is the core mechanism being evaluated — not an unfair advantage.