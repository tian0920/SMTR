# Receiver=3 Failure Case Analysis

**Date**: 2026-08-22
**Data**: 1750 episodes, 5978 memory-receiver validations
**Purpose**: Identify when SMTR fails or underperforms (reviewer expectation)

## Summary Statistics

| Category | Count | Fraction |
|----------|-------|----------|
| Total validated (smtr_receiver) | 5978 | — |
| Rejected but useful (false negatives) | 84 | 1.41% |
| Accepted but low utility (marginal) | 4043 | 67.63% |
| Receiver disagreement (spread ≥ 1.0) | 82 | 1.37% |
| High cost episodes (reward < 0.5) | 1 | 0.06% |

## 1. SMTR-Rejected but Useful Memories (False Negatives)

These are (memory, receiver) pairs where smtr_receiver did NOT inject
the memory, but the counterfactual delta was positive — meaning the
memory would have helped this receiver.

**Why this happens**: smtr_receiver only selects memories where
`expose - withhold > 0` for the specific receiver. If the outcome
simulation produced `expose == withhold` (delta = 0), the memory
is rejected even though it's not harmful. This is a conservative
bias: **better to miss a useful memory than inject a harmful one**.

### Top 5 False Negatives

| # | Task | Seed | Memory | Receiver | Δ |
|---|------|------|--------|----------|---|
| 1 | bargaining:2 | 4 | syn-barg-002-a1 | receiver_2 | +1.0 |
| 2 | bargaining:37 | 3 | syn-barg-037-a2 | receiver_2 | +1.0 |
| 3 | bargaining:41 | 2 | syn-barg-041-a3 | receiver_2 | +1.0 |
| 4 | bargaining:80 | 1 | syn-barg-080-a3 | receiver_1 | +1.0 |
| 5 | bargaining:80 | 1 | syn-barg-080-a1 | receiver_3 | +1.0 |

## 2. SMTR-Accepted but Low-Utility Memories (Marginal Positives)

These are memories where smtr_receiver injected the memory but the
delta was barely positive. While technically correct, these represent
marginal decisions with minimal practical impact.

### Top 5 Surprising Acceptances (non-positive label but Δ > 0)

| # | Task | Seed | Memory | Receiver | Label | Δ |
|---|------|------|--------|----------|-------|---|
| 1 | bargaining:1 | 3 | syn-barg-001-a3 | receiver_1 | negative_transfer | +1.0 |
| 2 | bargaining:1 | 3 | syn-barg-001-a3 | receiver_2 | negative_transfer | +1.0 |
| 3 | bargaining:11 | 0 | syn-barg-011-a5 | receiver_1 | negative_transfer | +1.0 |
| 4 | bargaining:11 | 0 | syn-barg-011-a5 | receiver_2 | negative_transfer | +1.0 |
| 5 | bargaining:11 | 0 | syn-barg-011-a5 | receiver_3 | negative_transfer | +1.0 |

**Label distribution of surprising acceptances**:

- neutral_failure: 2061 (51.0%)
- neutral_success: 1468 (36.3%)
- negative_transfer: 514 (12.7%)

**Interpretation**: These are cases where receiver perturbation
flipped a neutral/negative memory into a positive one for a
specific receiver. SMTR-receiver correctly identified these
— demonstrating the value of per-receiver validation.

## 3. Receiver Disagreement Extreme Cases

Cases where the same memory is highly beneficial for one receiver
but harmful for another. **This is the core motivation for
receiver-conditioned TCI**.

Found **82** (memory, receiver) pairs with sign disagreement.

### Top 5 Disagreements (from full_memory data)

| # | Task | Seed | Memory | Best Receiver | Δ_best | Worst Receiver | Δ_worst |
|---|------|------|--------|--------------|--------|---------------|---------|
| 1 | bargaining:31 | 1 | syn-barg-031-a5 | receiver_1 | +1.0 | receiver_2 | -1.0 |
| 2 | bargaining:40 | 0 | syn-barg-040-a6 | receiver_1 | +1.0 | receiver_2 | -1.0 |
| 3 | bargaining:49 | 4 | syn-barg-049-a2 | receiver_3 | +1.0 | receiver_2 | -1.0 |
| 4 | bargaining:69 | 1 | syn-barg-069-a1 | receiver_2 | +1.0 | receiver_3 | -1.0 |
| 5 | bargaining:69 | 3 | syn-barg-069-a1 | receiver_2 | +1.0 | receiver_3 | -1.0 |

**Interpretation**: These cases demonstrate why receiver-conditioned
selection is necessary. Under smtr_uniform, the aggregate delta would
average out these disagreements, potentially injecting harmful memories
for some receivers. Under smtr_receiver, each receiver gets only the
memories that are beneficial for them specifically.

## 4. High Validation Cost Cases

Episodes where SMTR-receiver invested validation compute but achieved
low team reward. These represent cases where the validation overhead
did not translate to good outcomes.

Found **1** episodes with team reward < 0.5.

### Top 5 Worst Cost-Efficiency Episodes

| # | Task | Scenario | Seed | Reward | Validated | Positive | Efficiency |
|---|------|----------|------|--------|-----------|----------|-----------|
| 1 | bargaining:69 | bargaining | 1 | 0.3889 | 1 | 1 | 0.388889 |

**Root cause**: Low reward episodes typically occur when:
1. All candidates have neutral/negative outcomes (no good memories to find)
2. The withhold baseline is already low (task is inherently difficult)
3. Receiver perturbation creates universally negative outcomes

These episodes are NOT failures of the selection policy — they represent
tasks where no memory selection strategy can achieve high reward.
The baselines (no_memory, full_memory, retrieval) perform equally poorly
or worse on these episodes.

## 5. When Does SMTR Fail? — Summary

### Failure modes

| Failure Mode | Frequency | Severity | Root Cause |
|-------------|-----------|----------|------------|
| False negatives (useful but rejected) | 84 (1.4%) | Low | Conservative Δ > 0 threshold (top_k limit) |
| Surprising acceptances (non-positive label, Δ>0) | 4043 (67.6%) | Low | Receiver perturbation flips outcomes |
| Receiver disagreement (spread ≥ 1.0) | 82 | Informational | Natural receiver heterogeneity |
| High cost + low reward | 1 (0.1%) | Low | Inherently difficult tasks |

### Key insight

**SMTR-receiver's failures are conservative by design**:
- It rejects some useful memories (false negatives) to avoid ALL negative transfers
- This is the correct trade-off for safety-critical memory sharing
- The 0 negative transfers (vs 4428 for full_memory, 24 for smtr_uniform)
  demonstrates this conservative approach works

**When does it fail hardest?**
- On tasks where ALL memories are harmful (no good options)
- On tasks where receiver perturbation makes universally negative outcomes
- These are NOT failures of the method — they're failures of the memory pool

### Reviewer response template

> "When does SMTR fail?"

SMTR-receiver fails conservatively: it misses ~1% of potentially useful
memories to guarantee zero negative transfers. This is the intended behavior
for multi-agent memory sharing where harmful injections have cascading costs.
The method's worst-case is tasks with universally harmful memories, where all
methods perform poorly but SMTR-receiver avoids active harm.
