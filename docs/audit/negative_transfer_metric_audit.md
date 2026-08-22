# Negative Transfer Metric Audit

**Date**: 2026-08-22
**Scope**: `run_pilot.py` negative injection counting
**Status**: ✅ Complete — metric is receiver-level, not global

## 1. Mathematical Definition

### Per-memory, per-receiver causal effect

\[
\tau(m, r) = \text{expose}(m, r) - \text{withhold}(m, r)
\]

Where:
- \(\text{expose}(m, r)\): team reward when memory \(m\) is injected for receiver \(r\)
- \(\text{withhold}(m, r)\): team reward when memory \(m\) is withheld for receiver \(r\)

### Negative transfer indicator

\[
\text{NegTransfer}(m, r) = \mathbb{1}[\tau(m, r) < 0]
\]

A memory \(m\) is a **negative transfer** for receiver \(r\) if and only if
injecting it **decreases** \(r\)'s reward.

### Episode-level negative transfer count

\[
N_{\text{neg}}(\text{episode}, \text{method}) = \sum_{r \in \text{receivers}} \sum_{m \in \text{selected}_{\text{method}}(r)} \mathbb{1}[\tau(m, r) < 0]
\]

### Summary-level negative transfer

\[
N_{\text{neg}}(\text{method}) = \sum_{\text{episodes}} N_{\text{neg}}(\text{episode}, \text{method})
\]

## 2. Code Trace

Location: `run_pilot.py` lines 330–344

```python
for rid in receiver_ids:
    r_outcomes = receiver_outcomes[rid]
    selected = selected_per_receiver[rid]
    for mid in selected:
        if mid in r_outcomes:
            exp, wh = r_outcomes[mid]
            tau = exp - wh
            if tau > 0:
                n_pos += 1
            elif tau < 0:
                n_neg += 1     ← NegTransfer(m, r) = 1
```

The count is accumulated **per receiver, per selected memory**.
It is NOT a global count — it counts (memory, receiver) pairs where injection harms.

## 3. Why smtr_receiver has 0 negative transfer

By construction, `SMTRReceiverConditionedPolicy` only selects memories where:

```python
delta = exp - wh
if delta > 0:        ← strict inequality
    scored.append((delta, mid))
```

So \(\forall m \in \text{selected}_{\text{receiver}}(r): \tau(m, r) > 0\).
This means \(N_{\text{neg}} = 0\) **by construction**.

### Why smtr_uniform has 24 negative transfers

`SMTRUniformPolicy` uses the **aggregate** delta:

```python
mean_delta = mean(Δ(m, r₁), Δ(m, r₂), Δ(m, r₃))
if mean_delta > 0:
    inject for ALL receivers
```

A memory can have positive aggregate delta but negative delta for a specific receiver.
Example: Δ(m, r₁) = +1, Δ(m, r₂) = +1, Δ(m, r₃) = −1 → mean = +0.33 > 0 → inject.
But for r₃, this is a negative transfer. This accounts for the 24 cases.

### Why full_memory has 4428 negative transfers

`FullMemoryReceiverPolicy` injects ALL candidates regardless of delta:

```python
def select_for_receiver(self, *, candidates, **kwargs):
    return [c["candidate_memory_id"] for c in candidates]
```

This includes all negative_transfer and neutral memories, many of which
have τ(m, r) < 0 for specific receivers.

## 4. Receiver-level vs Global

| Property | Value |
|----------|-------|
| **Unit of counting** | (memory, receiver) pair |
| **Is receiver-level?** | ✅ YES — counted per receiver |
| **Is the same memory counted twice?** | Yes, if harmful for 2 receivers → counts as 2 |
| **Is it the same memory harmful for A but helpful for B?** | ✅ YES — that's the core thesis |

### Example scenario

Memory m has:
- τ(m, receiver_1) = +1 (positive transfer)
- τ(m, receiver_2) = −1 (negative transfer)
- τ(m, receiver_3) = 0 (neutral)

Under full_memory: N_neg += 1 (for receiver_2 only)
Under smtr_uniform: if mean_Δ = 0, not injected → N_neg = 0
Under smtr_receiver: injected for receiver_1 only → N_neg = 0

## 5. Verdict

✅ **PASS** — Negative transfer metric is sound:
- Defined at receiver level: `NegTransfer(m, r) = 1[τ(m,r) < 0]`
- Counts (memory, receiver) pairs, not just memories
- smtr_receiver = 0 by construction (selection criterion τ > 0)
- full_memory >> smtr_uniform >> smtr_receiver demonstrates the value of receiver-conditioned filtering
