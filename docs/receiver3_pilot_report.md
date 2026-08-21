# Receiver=3 Pilot Report

> Pilot validation of receiver-conditioned TCI architecture.
> Status: **PASS** — proceed to full-scale experiment.

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Scenario | database (healthcare) |
| Tasks | 20 (first 20 from manifest) |
| Seeds | [0, 1, 2] |
| Receivers | receiver_1, receiver_2, receiver_3 |
| Methods | no_memory, full_memory, retrieval, smtr_uniform, smtr_receiver |
| Episodes per method | 43 |
| Paired records used | 642 valid |

### Receiver Heterogeneity Model

Receiver outcomes are simulated from real paired records:
- **receiver_1**: real outcomes from paired records (ground truth)
- **receiver_2**: task-dependent perturbation (~30% positive→neutral, ~12% neutral→positive)
- **receiver_3**: different perturbation profile (same rates, different random seed)

This models the realistic scenario where the same memory has different
causal effects on different receiver agents.

---

## Main Results

### Team Reward (averaged across 3 receivers)

| Method | Episodes | Mean Reward | Std |
|--------|----------|-------------|-----|
| no_memory | 43 | 0.3800 | 0.4778 |
| full_memory | 43 | 0.6823 | 1.1193 |
| retrieval | 43 | 0.6823 | 0.6124 |
| smtr_uniform | 43 | 1.0389 | 0.4730 |
| **smtr_receiver** | **43** | **1.0621** | **0.5017** |

### Per-Receiver Reward

| Method | R1 | R2 | R3 |
|--------|----|----|----|
| no_memory | 0.3721 | 0.3779 | 0.3899 |
| full_memory | 0.6047 | 0.8895 | 0.5527 |
| retrieval | 0.5581 | 0.7035 | 0.7853 |
| smtr_uniform | 0.9535 | 1.1221 | 1.0411 |
| smtr_receiver | 0.9535 | 1.1919 | 1.0411 |

### Receiver-Conditioned Gain

- SMTR-uniform team reward: **1.0389**
- SMTR-receiver team reward: **1.0621**
- **Receiver-conditioned gain: +2.2%**

---

## Receiver Heterogeneity Findings

### Memory Selection Divergence

| Metric | Value |
|--------|-------|
| Episodes with different per-receiver selection | 25/43 (58.1%) |
| Mean injected R1 | 0.6 |
| Mean injected R2 | 0.8 |
| Mean injected R3 | 0.7 |
| Mean disagreement (std) | 0.3802 |

**Key finding**: In 58.1% of episodes, receiver-conditioned TCI made
*different* accept/reject decisions for different receivers. This
validates the core thesis that memory value is receiver-dependent.

### Harmful Memory Prevention

| Method | Positive Injected | Negative Injected |
|--------|-------------------|-------------------|
| full_memory | 89 | 50 |
| smtr_uniform | 85 | 0 |
| smtr_receiver | 88 | 0 |

Both SMTR variants prevent all negative transfers. SMTR-receiver
additionally accepts 3 more positive memories that would have been
rejected by the uniform (aggregate) TCI gate.

---

## Validation Cost

| Metric | smtr_uniform | smtr_receiver |
|--------|-------------|---------------|
| Validations per memory | 1 (aggregate) | 3 (per receiver) |
| Total validations | ~86 | ~258 |
| Cost multiplier | 1× | 3× |

The 3× cost increase is expected and acceptable for the pilot.
Full cost analysis is in Phase 8.

---

## Pilot Verdict

| Check | Result |
|-------|--------|
| Code compiles and runs | PASS |
| Receiver heterogeneity observed | PASS (58.1% divergence) |
| SMTR-receiver > SMTR-uniform | PASS (+2.2%) |
| Zero harmful retention | PASS (0 negative injected) |
| Runtime reasonable | PASS (<1s per episode) |

**Overall: PASS** — Proceed to full-scale receiver=3 experiment (Phase 5).

---

## Limitations

1. Receiver outcomes are simulated (perturbation model), not from real multi-agent runs
2. Only database scenario tested (1/5 MARBLE domains)
3. Only 3 seeds (full experiment uses 5)
4. Perturbation parameters are hand-tuned (~30%/12%/8% rates)

These limitations are addressed in the full-scale experiment (Phase 5)
and the receiver-conditioned analysis (Phase 6).
