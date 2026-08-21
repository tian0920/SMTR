# Receiver-Conditioned TCI Information Audit

> Verifies that receiver-conditioned TCI decisions do NOT rely on
> leaked ground-truth labels, future rewards, or task answers.
>
> Audit date: 2026-08-22
> Scope: `src/smtr/memory/receiver_intervention.py`,
>        `src/smtr/memory/consolidation.py`,
>        `experiments/marble_receiver3/pilot/run_pilot.py` (policies)

---

## 1. Information Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│ ENVIRONMENT (simulator / MARBLE engine)                          │
│                                                                  │
│  paired_records.jsonl                                            │
│    ├─ share.team_success      ──┐                                │
│    ├─ withhold.team_success   ──┼─► simulate_receiver_outcome()  │
│    └─ label (4-outcome)       ──┘      │                         │
│        ▲                               │                         │
│        │ used ONLY here                ▼                         │
│        │ (perturbation rates)   (expose, withhold) per receiver  │
│        │ = the "physics" of            │                         │
│          the simulated world           │                         │
└────────┼───────────────────────────────┼────────────────────────┘
         │                               │
         ╳ NOT passed to policies        │
                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ DECISION LAYER (SMTR policies + TCI gate)                        │
│                                                                  │
│  SMTRReceiverConditionedPolicy.select_for_receiver()             │
│    inputs:  candidate_memory_id, receiver_outcomes[rid][mid]     │
│    rule:    delta(m,r) = expose - withhold > 0 → select          │
│                                                                  │
│  SMTRUniformPolicy.select_for_receiver()                         │
│    inputs:  candidate_memory_id, receiver_outcomes (all rids)    │
│    rule:    mean_r delta(m,r) > 0 → select for ALL receivers     │
│                                                                  │
│  MemoryAdmissionController.admit_for_receiver()                  │
│    inputs:  reward_expose, reward_withhold (measured rollouts)   │
│    rule:    delta > 0 → validated, else rejected                 │
│                                                                  │
│  ReceiverInterventionEvaluator.evaluate()                        │
│    inputs:  paired_outcomes or outcome_fn(memory, rid, state)    │
│    rule:    delta > 0 → validated                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Decision Input Audit

### 2.1 Allowed inputs (present in decision layer)

| Input | Source | Verdict |
|-------|--------|---------|
| `reward_expose` / `expose_reward` | counterfactual rollout outcome | ALLOWED |
| `reward_withhold` / `withhold_reward` | counterfactual rollout outcome | ALLOWED |
| `receiver_id` | agent identity | ALLOWED |
| `receiver_state` / `receiver_context` | optional context | ALLOWED |
| `candidate_memory_id` | memory identity | ALLOWED |

### 2.2 Forbidden inputs (checked by AST scan of policy classes)

AST scan of `SMTRReceiverConditionedPolicy` and `SMTRUniformPolicy`
found exactly these string-literal field accesses:

- `'candidate_memory_id'` — memory identity only

**Not accessed anywhere in the decision layer:**

| Forbidden field | Accessed? |
|-----------------|-----------|
| `label` (ground-truth 4-outcome) | **NO** |
| `task_instruction` / task answer | **NO** |
| `share.team_success` directly | **NO** (only via measured outcome tuples) |
| future reward / next-episode data | **NO** |

Note: the `candidates` list passed to policies *does* carry a `label`
key (inherited from the baseline runner interface), but AST analysis
confirms no SMTR policy reads it. Baseline `SMTRPolicy` in
`run_marble_baselines.py` DOES read `label` — that variant is the
explicitly documented **oracle upper bound** and is never used in the
receiver=3 pipeline.

**Verdict: PASS** — decision inputs are counterfactual-rollout
rewards only; no label leakage into SMTR-receiver decisions.

---

## 3. Non-Oracle Confirmation

The user's status table asked: 非 oracle TCI — 需要确认.

| Aspect | Status |
|--------|--------|
| SMTR-receiver decisions use measured Δ(m,r) | CONFIRMED non-oracle |
| SMTR-uniform decisions use measured mean Δ | CONFIRMED non-oracle |
| Perturbation simulator uses `label` | Environment-side only (like MARBLE's evaluator uses task answers) |
| Oracle variant (`SMTRPolicy` in marble_baselines) | Separate, explicitly labeled, NOT in receiver=3 pipeline |

**Verdict: PASS** — the receiver=3 pipeline is non-oracle. The
simulator's use of labels is equivalent to a benchmark harness using
ground truth to *score* runs, not to inform decisions.

---

## 4. Issues Found and Fixed

### Issue 1 — Cross-process non-determinism (FAIL → FIXED)

`hash((task_id, mid, receiver_id))` was used to seed receiver
perturbations. Python string hashing is salted per process
(PYTHONHASHSEED), verified empirically:

```
run1: hash(...) = 1421955481
run2: hash(...) = 945521200   ← DIFFERENT
```

**Fix**: introduced `det_seed()` using `zlib.crc32(repr(tuple(parts)))`,
replaced all 3 occurrences (pilot, main, contamination). Verified:
two independent processes now produce identical `main_episodes.csv`
(md5 `c3420aa9...` both runs).

### Issue 2 — Tautological "100% decision accuracy" (WARNING → REFRAMED)

The previous "Receiver TCI accuracy: 100%" compared each
receiver-conditioned decision against the very delta it was derived
from (`decision = delta > 0`). This is **self-consistency by
construction**, not independent accuracy.

**Fix**: metric renamed and decomposed:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Uniform TCI per-receiver alignment | 46.6% | Aggregate decision wrong for 53.4% of (memory, receiver) pairs |
| Uniform false accepts (harm) | 213 / 399 (53.4%) | Harmful injections uniform TCI would make |
| Uniform false rejects (loss) | 0 / 399 | Uniform is conservative here (mean>0 only when all-positive-ish) |
| Receiver self-consistency | 100% | By construction — labeled as such, NOT claimed as accuracy |

The meaningful claim is now: **uniform TCI mis-decides 53.4% of
individual (memory, receiver) pairs; receiver conditioning removes
this aggregation error by construction of measuring Δ(m,r) directly.**

---

## 5. Remaining Caveats (disclosed)

1. **Simulated receiver heterogeneity.** receiver_2/receiver_3
   outcomes are deterministic perturbations of real agent1 outcomes,
   not independent MARBLE runs. Perturbation rates (30% downgrade of
   positive_transfer, 12% upgrade of neutral) are modeling
   assumptions. Real multi-agent rollouts remain future work.
2. **Perturbation rates tuned by hand.** The disagreement level
   (84.2%) depends on these rates. A sensitivity check is recommended
   before publication.
3. **Binary outcomes.** team_success ∈ {0,1}, so delta ∈ {-1,0,1};
   continuous rewards would give richer Δ(m,r) structure.

---

## Overall Verdict

| Check | Result |
|-------|--------|
| No label leakage into decisions | **PASS** |
| No future information | **PASS** |
| Non-oracle TCI | **PASS** |
| Cross-process reproducibility | **PASS** (after fix) |
| Metric honesty (no tautological accuracy claim) | **PASS** (after fix) |

**Overall: PASS** — receiver-conditioned TCI decisions are driven
solely by measured per-receiver counterfactual outcomes.
