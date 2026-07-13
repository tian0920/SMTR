# Ablation Experiment Results

**Methods**: B0, B1-Top1, B1-Top3, B1-Matched, A1-NoSet, M0-Full
**Scenarios**: 9
**Episodes per scenario**: 20 (5 task seeds x 2 gen seeds x 3 trav seeds)
**Critic**: critic_pi3_v22 (M0-Full, A1-NoSet)

## Cross-Scenario Summary

| Scenario | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
|----------|-----|---------|---------|------------|----------|---------|
| positive | 0.00 | 1.00 | 0.00 | 0.50 | 0.00 | 0.20 |
| negative | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| neutral_success | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |
| neutral_failure | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| prefix_sensitive | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| flip_pos_to_neg | 1.00 | 1.00 | 0.00 | 0.50 | 0.00 | 1.00 |
| flip_neg_to_pos | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| flip_neu_to_neg | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 |
| flip_neu_to_pos | 1.00 | 1.00 | 0.00 | 0.50 | 0.00 | 0.80 |

## Average Success Rates

- **B0**: 0.444
- **B1-Top1**: 0.556
- **B1-Top3**: 0.111
- **B1-Matched**: 0.278
- **A1-NoSet**: 0.000
- **M0-Full**: 0.467

## Negative Transfer Rates

| Scenario | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
|----------|---------|---------|------------|----------|---------|
| positive | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| negative | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| neutral_success | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| neutral_failure | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| prefix_sensitive | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| flip_pos_to_neg | 0.00 | 1.00 | 0.50 | 1.00 | 0.00 |
| flip_neg_to_pos | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| flip_neu_to_neg | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| flip_neu_to_pos | 0.00 | 1.00 | 0.50 | 1.00 | 0.20 |

## Candidate-Level Diagnostics

| Scenario | Method | Recall@K | Router+Recall | Harmful Rej | Precision |
|----------|--------|----------|---------------|-------------|-----------|
| positive | M0-Full | 1.00 | 0.40 | - | 1.00 |
| positive | A1-NoSet | - | - | - | - |
| positive | B1-Top3 | 1.00 | 1.00 | - | 1.00 |
| negative | M0-Full | 1.00 | - | 1.00 | - |
| negative | A1-NoSet | - | - | - | - |
| negative | B1-Top3 | 1.00 | - | 0.00 | 0.00 |
| neutral_success | M0-Full | - | - | - | 0.00 |
| neutral_success | A1-NoSet | - | - | - | - |
| neutral_success | B1-Top3 | - | - | - | 0.00 |
| neutral_failure | M0-Full | - | - | - | 0.00 |
| neutral_failure | A1-NoSet | - | - | - | - |
| neutral_failure | B1-Top3 | - | - | - | 0.00 |
| prefix_sensitive | M0-Full | 1.00 | 1.00 | - | 1.00 |
| prefix_sensitive | A1-NoSet | - | - | - | - |
| prefix_sensitive | B1-Top3 | 1.00 | 1.00 | - | 1.00 |
| flip_pos_to_neg | M0-Full | 1.00 | - | 1.00 | - |
| flip_pos_to_neg | A1-NoSet | - | - | - | - |
| flip_pos_to_neg | B1-Top3 | 1.00 | - | 1.00 | - |
| flip_neg_to_pos | M0-Full | 1.00 | 0.40 | - | 1.00 |
| flip_neg_to_pos | A1-NoSet | - | - | - | - |
| flip_neg_to_pos | B1-Top3 | 1.00 | 1.00 | - | 1.00 |
| flip_neu_to_neg | M0-Full | - | - | - | - |
| flip_neu_to_neg | A1-NoSet | - | - | - | - |
| flip_neu_to_neg | B1-Top3 | - | - | - | - |
| flip_neu_to_pos | M0-Full | - | - | - | - |
| flip_neu_to_pos | A1-NoSet | - | - | - | - |
| flip_neu_to_pos | B1-Top3 | - | - | - | - |

## Prefix Formation Trace

| Scenario | Method | Prefix Recall | Prefix Sel Rate | Success+Prefix | Success-Prefix |
|----------|--------|---------------|-----------------|----------------|----------------|
| prefix_sensitive | M0-Full | 1.00 | 1.00 | 0.00 | 0.00 |
| prefix_sensitive | A1-NoSet | 0.00 | 0.00 | 0.00 | 0.00 |
| flip_pos_to_neg | M0-Full | 0.00 | 0.00 | 0.00 | 1.00 |
| flip_pos_to_neg | A1-NoSet | 0.00 | 0.00 | 0.00 | 0.00 |
| flip_neg_to_pos | M0-Full | 0.00 | 0.00 | 0.00 | 0.00 |
| flip_neg_to_pos | A1-NoSet | 0.00 | 0.00 | 0.00 | 0.00 |
| flip_neu_to_neg | M0-Full | 0.00 | 0.00 | 0.00 | 0.20 |
| flip_neu_to_neg | A1-NoSet | 0.00 | 0.00 | 0.00 | 0.00 |
| flip_neu_to_pos | M0-Full | 0.00 | 0.00 | 0.00 | 0.80 |
| flip_neu_to_pos | A1-NoSet | 0.00 | 0.00 | 0.00 | 0.00 |

## Key Findings

1. **B1-Top1 vs B1-Top3**: Top1 avg SR=0.556, Top3 avg SR=0.111. Restricting budget to 1 reduces negative transfer from indiscriminate sharing.
2. **B1-Matched**: avg SR=0.278. Budget-matched relevance baseline controls for share count confound.
3. **A1-NoSet vs M0-Full**: A1 avg SR=0.000, M0 avg SR=0.467. Delta measures the value of selected-set conditioning.

## Output Files

- Base directory: `outputs/ablation_all_scenarios/`
- Per-scenario dirs: `outputs/ablation_all_scenarios/<scenario>/`
  - `runs.jsonl`: per-run records
  - `summary.json`: method summaries
  - `config.json`: experiment configuration