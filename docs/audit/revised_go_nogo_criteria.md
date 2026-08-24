# Revised Go/No-Go Criteria

**Date**: 2026-08-24

**Status**: 🔄 DRAFT (pending ablation results)


## 1. Motivation

The previous Go/No-Go criteria were too simplistic:
- ❌ "baseline success < 0.9" — doesn't account for metric granularity
- ❌ "MOR ≥ 5%" — arbitrary threshold, may not apply to all benchmarks
- ❌ "validated > 0" — too weak (should require more)

The revised criteria are more robust and aligned with the official MultiAgentBench metrics.


## 2. Revised Criteria

### GO if ALL of the following are met:

| # | Criterion | Formula | Threshold | Rationale |
|---|-----------|---------|-----------|-----------|
| 1 | **Outcome variance** | Var(official_metric) > 0 | > 0 | Tasks are not all identical |
| 2 | **Pairwise discrimination** | P(expose ≠ withhold) | ≥ 5% | TCI can distinguish branches |
| 3 | **Positive delta exists** | count(δ > 0) | > 0 | Memory helps at least some tasks |
| 4 | **Negative delta exists** | count(δ < 0) | > 0 | Memory harms some tasks (realistic) |
| 5 | **Cross-episode reuse** | count(reuse > 0) | > 0 | Memories transfer across episodes |

### NO-GO if ANY of the following:

| # | Failure Mode | Check | Action |
|---|-------------|-------|--------|
| F1 | Zero variance | All tasks have same score | Fix backbone (P5) or tasks |
| F2 | Zero discrimination | All δ = 0 | Fix outcome adapter (P4) or TCI |
| F3 | All positive | No δ < 0 | Suspicious — check for bias |
| F4 | All negative | No δ > 0 | Memory not helping — fix method |
| F5 | No reuse | Memories never transfer | Fix memory selection (SMTR) |


## 3. Detailed Definitions

### Criterion 1: Outcome Variance

```python
import numpy as np

scores = [outcome.normalized_score for outcome in all_episodes]
variance = np.var(scores)

# GO if variance > 0 (tasks have different difficulty)
assert variance > 0, "All tasks have identical scores — check backbone"
```

**Rationale**: If all tasks have the same score, the benchmark is either too easy
(ceiling) or too hard (floor). The backbone sweep (P5) should find a model with
30%-80% success rate, which implies non-zero variance.


### Criterion 2: Pairwise Discrimination Rate (PDR)

```python
deltas = [outcome.oriented_delta for outcome in all_paired_episodes]
n_nonzero = sum(1 for d in deltas if d != 0)
n_total = len(deltas)
pdr = n_nonzero / n_total if n_total > 0 else 0.0

# GO if PDR >= 5% (at least 5% of pairs show different outcomes)
assert pdr >= 0.05, f"PDR too low: {pdr:.1%} — TCI cannot distinguish branches"
```

**Rationale**: If all expose/withhold pairs have identical scores, TCI delta is
always zero. This could mean:
- Outcome metric is too coarse (binary)
- Memory injection has no effect
- TCI computation is broken

**Note**: We lowered the threshold from "MOR ≥ 5%" to "PDR ≥ 5%" because:
- MOR (Memory Opportunity Rate) = P(δ > 0) — only counts positive deltas
- PDR (Pairwise Discrimination Rate) = P(δ ≠ 0) — counts both positive and negative
- PDR is more general and doesn't assume memory always helps


### Criterion 3: Positive Delta Exists

```python
n_positive = sum(1 for d in deltas if d > 0)

# GO if at least one positive delta (memory helps)
assert n_positive > 0, "No positive deltas — memory never helps"
```

**Rationale**: If all deltas are ≤ 0, memory is never beneficial. This could mean:
- Memory selection is broken (SMTR selects wrong memories)
- Memory injection is broken (memories not actually injected)
- Outcome metric is insensitive to memory benefit


### Criterion 4: Negative Delta Exists

```python
n_negative = sum(1 for d in deltas if d < 0)

# GO if at least one negative delta (memory sometimes harms)
assert n_negative > 0, "No negative deltas — suspicious (memory always helps?)"
```

**Rationale**: If all deltas are ≥ 0, this is suspicious. In reality:
- Some memories are harmful (misleading, outdated, irrelevant)
- Some tasks don't benefit from memory
- Some memory injections cause confusion

**Note**: This is a **sanity check**, not a hard requirement. If all deltas are
genuinely positive (memory always helps), this could be valid — but it's rare.


### Criterion 5: Cross-Episode Reuse

```python
reuse_counts = [memory.reuse_count for memory in all_memories]
n_reused = sum(1 for c in reuse_counts if c > 1)

# GO if at least one memory is reused across episodes
assert n_reused > 0, "No memories reused — memory selection not generalizing"
```

**Rationale**: SMTR's value proposition is **cross-episode memory reuse**. If
memories are never reused (each memory used only once), then:
- Memory selection is not generalizing
- SMTR is equivalent to per-episode optimization
- No long-term learning benefit


## 4. Comparison with Previous Criteria

| Criterion | Previous | Revised | Change |
|-----------|----------|---------|--------|
| Baseline difficulty | success < 0.9 | variance > 0 | More general |
| Memory opportunity | MOR ≥ 5% | PDR ≥ 5% | Counts both + and - |
| Validation | validated > 0 | positive_delta > 0 | More specific |
| Reuse | reuse > 0 | reuse > 0 | Same |
| **New** | — | negative_delta > 0 | Sanity check |

**Key Changes**:
1. Replaced "baseline success < 0.9" with "variance > 0" — more robust
2. Replaced "MOR ≥ 5%" with "PDR ≥ 5%" — more general (counts negative deltas)
3. Added "negative_delta > 0" — sanity check (memory shouldn't always help)


## 5. Decision Matrix

| Variance | PDR | Positive δ | Negative δ | Reuse | Decision |
|----------|-----|------------|------------|-------|----------|
| ✅ | ✅ | ✅ | ✅ | ✅ | **GO** |
| ❌ | — | — | — | — | **NO-GO** (fix backbone) |
| ✅ | ❌ | — | — | — | **NO-GO** (fix outcome/TCI) |
| ✅ | ✅ | ❌ | — | — | **NO-GO** (fix memory selection) |
| ✅ | ✅ | ✅ | ❌ | — | **WARN** (suspicious, investigate) |
| ✅ | ✅ | ✅ | ✅ | ❌ | **NO-GO** (fix reuse) |


## 6. Expected Values (from Paper)

Based on MultiAgentBench paper Table 1:

| Model | Avg TS | Variance | Expected PDR |
|-------|--------|----------|--------------|
| Llama-3.1-8B | 50.74% | High (6.12%-80.87%) | ~30-50% |
| GPT-4o-mini | 60.46% | High (33.60%-84.13%) | ~20-40% |
| qwen3-30b-a3b (current) | 99% (binary) | **Zero** (ceiling) | ~0% |

**Prediction**: After fixing the outcome adapter (P4) and running the backbone
sweep (P5), we expect:
- Variance > 0 (non-saturated backbone)
- PDR ≥ 5% (official metric has granularity)
- Positive and negative deltas exist (memory helps/harms)
- Reuse > 0 (SMTR generalizes across episodes)


## 7. Implementation

### Pre-Flight Check Script

```python
def check_go_nogo(results: list[PairedResult]) -> dict:
    """Check all Go/No-Go criteria."""
    
    # Criterion 1: Outcome variance
    scores = [r.score for r in results if r.score is not None]
    variance = np.var(scores) if scores else 0.0
    c1_pass = variance > 0
    
    # Criterion 2: Pairwise discrimination
    deltas = [r.delta for r in results if r.delta is not None]
    n_nonzero = sum(1 for d in deltas if d != 0)
    pdr = n_nonzero / len(deltas) if deltas else 0.0
    c2_pass = pdr >= 0.05
    
    # Criterion 3: Positive delta
    n_positive = sum(1 for d in deltas if d > 0)
    c3_pass = n_positive > 0
    
    # Criterion 4: Negative delta
    n_negative = sum(1 for d in deltas if d < 0)
    c4_pass = n_negative > 0
    
    # Criterion 5: Cross-episode reuse
    reuse_counts = [m.reuse_count for m in all_memories]
    n_reused = sum(1 for c in reuse_counts if c > 1)
    c5_pass = n_reused > 0
    
    return {
        "go": all([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass]),
        "variance": variance,
        "pdr": pdr,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_reused": n_reused,
        "criteria": {
            "variance > 0": c1_pass,
            "pdr >= 5%": c2_pass,
            "positive_delta > 0": c3_pass,
            "negative_delta > 0": c4_pass,
            "reuse > 0": c5_pass,
        },
    }
```


## 8. Conclusion

**Revised Go/No-Go**:
1. ✅ Outcome variance > 0 (non-saturated backbone)
2. ✅ Pairwise discrimination rate ≥ 5% (TCI can distinguish branches)
3. ✅ Positive delta exists (memory helps)
4. ✅ Negative delta exists (memory sometimes harms — sanity check)
5. ✅ Cross-episode reuse > 0 (SMTR generalizes)

**Status**: 🔄 DRAFT — Pending backbone sweep (P5) and ablation results

**Next Steps**:
1. Run backbone difficulty sweep (P5)
2. Select non-saturated backbone
3. Run TCI ablation with official metrics (P4)
4. Apply Go/No-Go criteria
5. If GO → proceed to main experiment (P7)
6. If NO-GO → diagnose failure mode and fix
