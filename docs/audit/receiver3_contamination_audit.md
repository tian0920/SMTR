# Receiver=3 Contamination Claim Audit

> Verifies the contamination propagation experiment is not biased by
> design, and calibrates the paper claim wording.
>
> Audit date: 2026-08-22

---

## 1. Claim Calibration

| Before (avoid) | After (use) |
|----------------|-------------|
| "SMTR **eliminates** contamination" | "Receiver-conditioned validation **substantially reduces** harmful knowledge propagation in multi-agent sharing" |

Rationale: residual contamination is non-zero (1.96% at ratio 0.3).
"Eliminates" is factually wrong and reviewer-fatal.

---

## 2. Contamination Source Audit

### 2.1 Harmful memory sources

Contaminated memories are drawn from REAL paired records, by type:

| Contamination type | Source label | Pool size | Rationale |
|--------------------|-------------|-----------|-----------|
| false_procedural | neutral_failure | 403 | Procedure that failed; sharing it spreads wrong method |
| spurious_success | neutral_success | 159 | Success without causal contribution; over-trusted |
| outdated | neutral_* + negative_transfer | 602 | No-longer-applicable or harmful experience |

At ratio ρ, a random ρ-fraction of the source pool is flagged
contaminated (deterministic seeding: `det_seed(task, seed, type, ratio)`).

### 2.2 Injection mechanism

```
source agent experience
      │
      ▼
memory pool (per task group, real paired records)
      │
      ├── ratio ρ flagged as contaminated (invisible to all methods)
      │
      ▼
method selection policy ──► per-receiver injection ──► team outcome
```

All methods see the SAME pool; none sees the contamination flag.

### 2.3 Receiver exposure

Each receiver is evaluated independently with its own simulated
outcome model. Contamination propagation depth = fraction of receivers
that received ≥1 contaminated memory.

### 2.4 SMTR rejection point

SMTR-receiver rejects a (memory, receiver) pair iff the MEASURED
Δ(m,r) ≤ 0. The rejection happens at the TCI gate, using only
counterfactual rollout outcomes — never the contamination flag.

### 2.5 Baseline propagation path

- **full_memory**: no filter → propagates everything
- **retrieval**: rank top-3 → propagates whatever ranks high

---

## 3. Pre-Filtering Check (critical)

**Question**: is contamination being filtered out by experiment
design before methods ever see it?

**Evidence — contaminated memory injection rates at ratio 0.3:**

| Method | Contaminated injected | Available | Rate |
|--------|----------------------|-----------|------|
| full_memory | 561 | 561 | **100.0%** |
| retrieval | 330 | 561 | 58.8% |
| smtr_uniform | 69 | 561 | 12.3% |
| smtr_receiver | 24 | 561 | **4.3%** |

Key observations:

1. **Contaminated memories ARE injectable** — full_memory injects
   100% of them. The design does not pre-filter.
2. **SMTR-receiver's 24 residual injections prove the gate is
   measurement-based, not label-based.** If the gate secretly saw the
   contamination flag, residuals would be exactly 0. The 24 residuals
   are contaminated memories whose measured Δ(m,r) happened to be
   positive — the gate honestly follows its measurement.
3. The reduction (100% → 4.3%) comes from one mechanism only:
   contaminated memories tend to have non-positive measured Δ(m,r),
   and the gate rejects non-positive Δ.

**Verdict: PASS** — no pre-filtering; the claim is earned by the gate.

---

## 4. Residual Contamination Analysis

Why does 4.3% remain?

- Contamination is a SEMANTIC flag (false/outdated content), but TCI
  measures OUTCOME. A "false procedural" memory can still correlate
  with success in the measured rollout (e.g., the receiver succeeds
  despite it), giving Δ > 0.
- This is an honest limitation: **counterfactual validation catches
  outcome-harmful memories, not semantically-false ones that happen
  not to hurt in the validation window.**

Paper framing: "TCI reduces propagation of outcome-harmful knowledge;
semantically false memories with neutral measured effects remain a
limitation requiring content-level verification."

---

## 5. Main Contamination Results (for paper)

At contamination ratio 0.3 (averaged over 3 types, 5 seeds):

| Method | Team reward | Contam. rate | Prop. depth |
|--------|-------------|--------------|-------------|
| full_memory | 0.0245 | 45.8% | 45.8% |
| retrieval | 0.0662 | 25.3% | 25.3% |
| smtr_uniform | 0.4461 | 5.6% | 5.6% |
| **smtr_receiver** | **0.4461** | **2.0%** | **2.0%** |

Significance (paired, n=408): p = 2.34e-06 for receiver < uniform.

---

## Overall Verdict

| Check | Result |
|-------|--------|
| Contamination not pre-filtered | **PASS** (full_memory injects 100%) |
| Gate blind to contamination flag | **PASS** (4.3% residual proves it) |
| Claim wording calibrated | **PASS** ("substantially reduces", not "eliminates") |
| Statistical significance | **PASS** (p=2.34e-06) |

**Overall: PASS** — the contamination claim is defensible with the
calibrated wording.
