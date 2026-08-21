# Receiver=3 Baseline Fairness Audit

> Verifies that all baseline methods are fairly supported in the
> receiver=3 experiment setting.

---

## Method Fairness Matrix

| Method | Shared Memory? | Same Scope? | Receiver Visible? | Same Budget? | Receiver Conditioned? | Verdict |
|--------|---------------|-------------|-------------------|-------------|----------------------|---------|
| no_memory | N/A | N/A | N/A | N/A | N/A | PASS |
| full_memory | Yes (all) | Yes (all candidates) | Yes (same set) | Yes | No (uniform) | PASS |
| retrieval | Yes (top-k) | Yes (same candidates) | Yes (same top-k) | Yes (top_k=3) | No (score-based) | PASS |
| smtr_uniform | Yes (validated) | Yes (delta>0) | Yes (same set) | Yes | No (aggregate delta) | PASS |
| smtr_receiver | Yes (validated) | Yes (per-receiver delta>0) | Yes (per-receiver) | Yes | **Yes** | PASS |

---

## Detailed Audit

### 1. No Memory

**PASS** — Withhold baseline. No memory injected for any receiver.

### 2. Full Memory

**PASS** — All candidate memories injected for all receivers.
- Same candidate pool available to all receivers
- No selective filtering
- Budget: unlimited (all candidates)

### 3. Retrieval

**PASS** — Top-k by candidate_rank for all receivers.
- Same ranking function applied to all receivers
- top_k=3 budget applied uniformly
- No receiver-specific filtering

### 4. SMTR-Uniform (TCI aggregate)

**PASS** — Memories validated by aggregate delta across all receivers.
- Single accept/reject decision per memory
- Same validated set applied to all receivers
- Budget: only validated memories

### 5. SMTR-Receiver (TCI per-receiver)

**PASS** — Memories validated per-receiver by individual delta.
- Different accept/reject decisions per receiver
- Each receiver gets their own validated set
- Budget: only validated memories per receiver

---

## Fairness Concerns

### Concern 1: SMTR-receiver has 3× validation budget

**Assessment**: WARNING — acknowledged in cost analysis (Phase 8).

SMTR-receiver runs 3 validations per memory (one per receiver),
while SMTR-uniform runs 1. This gives SMTR-receiver more information
but at higher cost.

**Mitigation**: Cost analysis (Phase 8) reports reward/cost ratio.
The paper should clearly state the cost multiplier.

### Concern 2: Baseline methods are not receiver-conditioned

**Assessment**: PASS — this is the experimental design.

Baselines (full_memory, retrieval) intentionally do not use
receiver-conditioned selection. This is the control group that
demonstrates the value of receiver-conditioned TCI.

### Concern 3: Simulated receiver outcomes

**Assessment**: WARNING — receiver heterogeneity is modeled via
perturbation, not from real multi-agent runs.

The perturbation model is deterministic and based on real paired
records, but does not capture full agent behavioral diversity.

**Mitigation**: Paper should note this as a limitation and suggest
real multi-agent validation as future work.

---

## Overall Verdict

| Check | Result |
|-------|--------|
| All methods use same candidate pool | PASS |
| Budget constraints applied fairly | PASS |
| SMTR cost multiplier disclosed | WARNING |
| Receiver outcomes from real data | WARNING (simulated) |
| No hidden advantages | PASS |

**Overall: PASS with WARNINGS**

The experimental design is fair. All methods have access to the same
candidate memories and are evaluated on the same episodes. The only
difference is the selection strategy, which is the intended experimental
variable.
