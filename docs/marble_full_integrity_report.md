# MARBLE Full Integrity Report

> Final comprehensive integrity assessment for all MARBLE experiment results.
> Determines whether current results can be used in paper submission.

---

## Verdict Summary

### Overall: CONDITIONAL PASS

The underlying MARBLE data is **real and authentic**. The experiments are built on genuine multi-agent LLM execution on real PostgreSQL databases. However, the **experiment design** has limitations that must be addressed before paper submission.

| Component | Verdict | Details |
|-----------|---------|---------|
| MARBLE engine execution | **PASS** | Real LLM, real DB, real evaluator |
| Paired records authenticity | **PASS** | 642 valid records from real engine runs |
| LLM model calls | **PASS** | ~5,826 calls to qwen3-30b-a3b |
| Reward source | **PASS** | Native evaluator label matching |
| Trajectory files | **PASS** | 1,260 audit files, all verifiable |
| Config integrity | **PASS** | No fake/mock/cache/offline flags |
| Benchmark source | **PASS** | Official MARBLE, unmodified |
| Runtime reasonableness | **PASS** | Mean 6.4s per branch |
| Offline evaluation layer | **WARNING** | Method policies simulated, not re-executed |
| SMTR-TCI ground-truth proxy | **WARNING** | Uses labels as oracle (idealized upper bound) |
| Scenario coverage | **WARNING** | 1/5 scenarios (database only) |
| Receiver coverage | **WARNING** | 1/3 receivers (agent1 only) |
| Seed coverage | **WARNING** | 3/5 seeds |
| Contamination result | **WARNING** | Zero retention is tautological |
| Score uniformity | **WARNING** | All scores=0.600, baseline methods indistinguishable |

---

## 1. What Is Real

### 1.1 MARBLE Engine Runs
- **210 control groups** across 70 database tasks × 3 seeds
- **1,260 share branches** attempted, **971 succeeded** with real engine execution
- **642 valid paired records** with both share and withhold outcomes
- **19,929 on-disk files** totaling 202.4 MB

### 1.2 LLM Execution
- Model: `qwen3-30b-a3b` via Alibaba Cloud MaaS
- Real SQL queries against PostgreSQL healthcare database
- Natural language reasoning from 5 agents per task
- Token usage: ~6,275 per branch
- Execution timestamps: 2026-08-14T14:31:xx UTC

### 1.3 Evaluation
- Native evaluator `marble_database_evaluate_task_db`
- Ground-truth root cause label matching
- Binary team_success based on predicted vs expected labels
- F1, precision, recall computed for each branch

### 1.4 Paired Design
- Shared `InitialStateBundle` across share/withhold branches
- Digests verify identical initial conditions
- Memory visibility audit confirms injection isolation
- Control reuse: same control serves multiple candidates

---

## 2. What Has Limitations

### 2.1 Offline Evaluation (Not Re-Execution)

**Issue**: `run_marble_baselines.py` does NOT re-run MARBLE for each method. It:
1. Reads pre-computed paired records
2. Simulates each method's selection policy
3. Looks up existing outcomes

**Impact**: Method behavior is simulated. The actual LLM response to injected memories is from the original share/withhold runs, not re-evaluated per method.

**Mitigation**: The underlying outcomes are real. The simulation is a valid approximation if we assume method behavior only affects WHICH memories are injected, not HOW the LLM responds to them.

### 2.2 SMTR-TCI Ground-Truth Proxy

**Issue**: SMTR-TCI selects memories where `label == "positive_transfer"`. This label is only known AFTER both branches execute.

**Impact**: SMTR has **perfect hindsight**. In real deployment, TCI would predict transfer type with errors.

**Consequence**:
- The +61.2% improvement is an **upper bound** on TCI performance
- Zero contamination retention is **tautological** (by definition, positive_transfer memories are not contaminated)
- The comparison is **asymmetric** (baselines use decision-time features, SMTR uses post-hoc labels)

### 2.3 Limited Coverage

| Dimension | Available | Used | Coverage |
|-----------|-----------|------|----------|
| Scenarios | 5 | 1 | 20% |
| Tasks (database) | 100 | 50 | 50% |
| Receivers | 3 | 1 | 33% |
| Seeds | 5 | 3 | 60% |

### 2.4 Score Uniformity

All 642 valid candidate records have `candidate_score = 0.600`. This makes retrieval, heuristic, and agemem methods nearly identical (all select by rank when scores are equal).

---

## 3. What Cannot Be Claimed

Based on this audit, the following claims from `docs/claim_evidence_matrix.md` need revision:

| Claim | Current Confidence | Revised Assessment |
|-------|-------------------|-------------------|
| SMTR +61% improvement | HIGH | **UPPER BOUND** — present as oracle ceiling |
| Zero harmful retention | HIGH | **TAUTOLOGICAL** — guaranteed by selection rule |
| Works across domains | HIGH | **LIMITED** — 1/5 scenarios, domain = agent count |
| No perfect oracle needed | HIGH | **VALID** — noise experiment shows graceful degradation |
| Receiver-conditioned utility | LOW | **STILL LOW** — single receiver only |

---

## 4. Detailed Audit Artifacts

### CSV Files
| File | Rows | Description |
|------|------|-------------|
| `results/audit/episode_statistics.csv` | 21 | Per-method, per-seed episode counts |
| `results/audit/trajectory_audit.csv` | 1,260 | Branch-level audit (real_engine, success, evaluator) |
| `results/audit/seed_check.csv` | 7 | Seed coverage per method (WARNING: 3/5) |
| `results/audit/runtime_analysis.csv` | 1,260 | Runtime per branch (mean 6.4s, range 4-10s) |

### Audit Documents
| File | Verdict |
|------|---------|
| `docs/audit/model_execution.md` | PASS (real LLM) |
| `docs/audit/config_audit.md` | PASS (no fake flags) |
| `docs/audit/reward_pipeline.md` | PASS (real evaluator) |
| `docs/audit/contamination_pipeline.md` | WARNING (tautological) |
| `docs/audit/marble_dataset_check.md` | PASS (official MARBLE) |

---

## 5. Can These Results Enter the Paper?

### YES — With Caveats

The results CAN be used in the paper if:

1. **SMTR-TCI is labeled as "Oracle Upper Bound"**: Clearly state that the method uses ground-truth labels as a proxy for perfect TCI validation. This represents the ceiling of what TCI could achieve, not its current performance.

2. **Coverage limitations are disclosed**: 1/5 scenarios, 1/3 receivers, 3/5 seeds.

3. **Offline evaluation is explained**: The method comparison uses simulated policies on real paired outcomes, not re-executed MARBLE runs.

4. **Contamination claim is reframed**: "TCI's design principle prevents contamination by construction" rather than "TCI learned to avoid contamination."

### NO — Without These Changes

The results CANNOT be presented as:
- "SMTR outperforms baselines by 61%" (without "upper bound" qualifier)
- "Zero contamination across all settings" (without "by design" qualifier)
- "Works across diverse multi-agent domains" (only 1 scenario tested)

---

## 6. Recommended Next Steps

### Priority 1: Expand Coverage (Required for Strong Paper)

| Action | Effort | Impact |
|--------|--------|--------|
| Run coding + research scenarios | ~500 LLM calls each | Cross-domain validity |
| Execute agent2, agent3 receivers | ~420 new branches | Multi-receiver evidence |
| Add seeds 3, 4 | ~840 new branches | Statistical strength |

### Priority 2: Real TCI Model (Required for Main Claim)

| Action | Effort | Impact |
|--------|--------|--------|
| Train TCI classifier on paired data | 1-2 days | Replace oracle proxy with real prediction |
| Evaluate with prediction errors | ~100 simulations | Realistic contamination results |
| Report TCI precision/recall | 1 hour | Quantify gate quality |

### Priority 3: Cross-Backbone Validation (Nice to Have)

| Backbone | Purpose | Cost |
|----------|---------|------|
| GPT-4.1 / Claude Sonnet | Commercial baseline | $$$ |
| Llama-3.1-70B / Qwen2.5-72B | Open-source validation | $$ |
| Qwen3-30b-a3b (current) | Reference | Already done |

---

## 7. Paper Evidence Chain (Revised)

```
Synthetic Lifelong (why TCI works)
  └─ MQS=1.734, reward=0.872 — REAL execution, full seeds ✓

Baseline Comparison (why existing approaches insufficient)
  └─ Full Memory harmful_retention=1.0 — REAL execution ✓
  └─ All baselines: harmful>0, reward<0.872 — REAL execution ✓

MARBLE Real Environment (realistic multi-agent validation)
  └─ Real engine: 971 branches, real LLM, real PostgreSQL ✓
  └─ Paired outcomes: 642 valid records ✓
  └─ Method comparison: OFFLINE SIMULATION ⚠
  └─ SMTR reward: ORACLE UPPER BOUND ⚠
  └─ Coverage: 1/5 scenarios, 1/3 receivers ⚠

Noise Robustness (not an oracle)
  └─ σ=0.3: reward=0.670, harmful=0.000 — SIMULATED noise on real data ✓

Cost Analysis (efficient)
  └─ Same ops → +6.9% — SYNTHETIC experiment ✓

Analysis (why memories persist)
  └─ TCI design prevents contamination BY CONSTRUCTION ⚠
```

Legend: ✓ = real execution/validated, ⚠ = has caveats
