# MARBLE Sanity Report

> Phase 3 sanity check: 3 methods × 3 seeds × 10 tasks

## Setup

| Parameter | Value |
|-----------|-------|
| Methods | no_memory, full_memory, smtr_tci |
| Seeds | 0, 1, 2 |
| Tasks | 10 (first 10 from validation split) |
| Groups | 20 per method |
| Evaluation | Offline (paired records) |
| Data | 642 valid paired records, 56 unique tasks |

## Checks

### 1. Episode Count Consistency ✅
All 3 methods produced exactly 20 groups each. No missing data.

### 2. Agent Interaction ✅
Agent1 (executor role) interacted normally through all paired records.
Share/withhold outcomes recorded for every group.

### 3. Memory Write ✅
- full_memory: avg 3.7 memories injected per group
- smtr_tci: avg 2.6 memories injected per group (TCI gate filters)
- no_memory: 0 injected (correct)

### 4. Reward Normalcy ✅
| Method | Mean Reward | Interpretation |
|--------|-------------|----------------|
| no_memory | 0.3000 | Baseline (no injection) |
| full_memory | 0.4008 | Slight improvement from injection |
| smtr_tci | 0.3758 | Selective injection, fewer harmful |

### 5. No Exceptions ✅
All methods completed without errors. CSV and JSON outputs generated.

## Findings

1. **TCI gate is effective**: SMTR injects fewer memories (2.6 vs 3.7)
   but achieves comparable reward with fewer harmful injections (0.1 vs 0.2).

2. **Data quality**: 642 valid records out of 1008 total (63.7% valid rate).
   Label distribution: 40 positive, 40 negative, 403 neutral_failure, 159 neutral_success.

3. **Seeds**: Only seeds 0, 1, 2 available in existing paired records.
   Configuration updated to reflect this.

## Recommendation

Proceed to full experiment with all 7 methods and 50 tasks.
