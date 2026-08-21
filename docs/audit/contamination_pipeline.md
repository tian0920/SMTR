# Contamination Pipeline Audit

> Verifies that the zero contamination retention result is genuine.

---

## 1. How Is Contamination Defined?

In the contamination experiment (`run_contamination.py`):

```python
# Label neutral_failure/neutral_success records as "contaminated"
neutral_records = [r for r in group_records if r.get("label") in ("neutral_failure", "neutral_success")]
n_contaminate = int(len(neutral_records) * ratio)
```

Contamination is **simulated** by randomly marking a fraction of neutral memories as contaminated.

---

## 2. Are Contaminated Memories Actually Injected?

### Baseline methods (full_memory, retrieval):

**YES** — full_memory injects ALL candidates including contaminated ones:
```python
if method == "full_memory":
    selected = group_records  # ALL records, including contaminated
```

**YES** — retrieval injects top-k by rank, which may include contaminated:
```python
elif method == "retrieval":
    ranked = sorted(group_records, key=lambda r: r.get("candidate_rank", 0))
    selected = ranked[:3]
```

### SMTR-TCI:

**NO** — SMTR only selects `positive_transfer` memories:
```python
elif method == "smtr_tci":
    selected = [r for r in group_records if r.get("label") == "positive_transfer"]
```

Since contaminated memories are labeled `neutral_failure` or `neutral_success`, they are **never** selected by SMTR.

---

## 3. Does Receiver See Contaminated Memory?

The contamination experiment is **offline simulation** — it does NOT re-run MARBLE. It checks whether each method's selection policy would include contaminated memories.

For baselines: YES, they would include contaminated memories (in simulation).
For SMTR: NO, it would not (in simulation).

---

## 4. Does SMTR Have Extra Ground Truth Information?

**YES — this is the critical finding.**

SMTR's selection rule `label == "positive_transfer"` uses the **ground-truth four-outcome label** which is only available AFTER both share and withhold branches have been executed.

In a real deployment:
- TCI would need to predict whether a memory causes positive transfer
- This prediction would have errors
- Some contaminated memories might pass the gate

In this experiment:
- SMTR has **perfect hindsight** — it knows exactly which memories are beneficial
- This makes zero contamination retention a **tautological result**, not a learned behavior

---

## 5. Is Baseline Fair?

| Method | Selection Rule | Uses Ground Truth? |
|--------|---------------|-------------------|
| full_memory | inject all | No |
| retrieval | top-k by rank | No (uses candidate_rank) |
| **smtr_tci** | label == "positive_transfer" | **YES** |

**Assessment**: The comparison is **asymmetric**. SMTR uses post-hoc ground truth labels while baselines use only features available at decision time.

---

## 6. Contamination Audit Summary

| Check | Result |
|-------|--------|
| Contamination injection simulated | YES (for baselines) |
| Receiver sees contamination | YES (in simulation, for baselines) |
| SMTR uses ground truth | **YES — idealized proxy** |
| Baseline uses ground truth | NO |
| Fair comparison | **NO — asymmetric information** |

---

## Conclusion

**WARNING**: The zero contamination retention result is **correct but trivially guaranteed** by the experimental design. SMTR selects only `positive_transfer` memories by definition, so it can never select contaminated (`neutral_*`) memories.

**For paper**: This should be presented as "TCI's design principle prevents contamination by construction" rather than "TCI learned to avoid contamination."

**Recommendation**: Run a real TCI model (with prediction errors) to show contamination reduction under realistic conditions.
