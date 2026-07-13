# Ablation Experiment — Implementation Details

## Method Definitions

| Method | Display Label | Router Class | Critic | Feature Block | Share Budget | Selected Set | Pairwise |
|--------|--------------|--------------|--------|---------------|-------------|--------------|----------|
| `b0_no_memory` | B0 | NoMemoryRouter | None | None | zero | No | No |
| `b1_top1` | B1-Top1 | RelevanceTopKRouter | None | None | fixed 1 | No | No |
| `b1_top3` | B1-Top3 | RelevanceTopKRouter | None | None | fixed 3 | No | No |
| `b1_matched` | B1-Matched | BudgetMatchedTopKRouter | None | None | validation-matched | No | No |
| `smtr` | SMTR | ProductionSequentialRouter | critic_full_gate_ablation_v1 | full | fixed 3 | Yes | Yes |
| `effect_only_smtr` | EffectOnly-SMTR | ProductionSequentialRouter | critic_full_gate_ablation_v1 | full | fixed 3 | Yes | Yes |

Formal SMTR shares a candidate memory when:

```text
tau_mean > 0 and negative_risk_mean <= negative_risk_budget
```

LCB/UCB confidence-bound routing is no longer the default formal method. The
optional Robust-SMTR extension lives in `smtr.robust` and must be invoked
explicitly.

## Modified Files

### Core Infrastructure

| File | Change |
|------|--------|
| `src/smtr/experiment/methods.py` | **New** — Method registry with `MethodSpec`, `METHOD_REGISTRY`, `build_default_specs()` |
| `src/smtr/experiment/schemas.py` | Added `VALID_METHOD_IDS`, `METHOD_ID_TO_REGISTRY`, `methods`/`negative_risk_budget`/`budget_manifest_path` to `ExperimentConfig` |
| `src/smtr/experiment/runner.py` | Refactored to support arbitrary method list via `_build_router_for_method()` |
| `src/smtr/experiment/summary.py` | Updated bootstrap CI to group formal SMTR comparisons; canonical reason mapping |
| `src/smtr/router/factory.py` | Added `build_smtr_router()` formal factory |
| `src/smtr/router/baselines.py` | Added `BudgetMatchedTopKRouter`, `BudgetManifestConfig`; `proposal_rank`/`proposal_score` in decisions |
| `src/smtr/router/traces.py` | Added `proposal_rank`, `proposal_score` fields to `RouterDecision` |
| `src/smtr/cli.py` | Added `--methods`, `--negative-risk-budget`, `--budget-manifest-path` to `run-experiment`/`compare-routers` |

### Diagnostics

| File | Change |
|------|--------|
| `src/smtr/experiment/candidate_diagnostics.py` | **New** — Candidate-level ground-truth diagnostics (Recall@K, router recall, precision) |
| `src/smtr/experiment/prefix_trace.py` | **New** — Prefix formation trace for prefix-sensitive and flip scenarios |

### Training & Manifest

| File | Change |
|------|--------|
| `scripts/train_a1_critic.py` | **New** — Train A1-NoSet critic with `feature_block="context_plus_candidate"` |
| `scripts/run_ablation_experiments.py` | **New** — Run all 6 methods across 9 scenarios |
| `outputs/budget_manifest.json` | **Generated** — B1-Matched budget distribution from M0 validation |

### Tests

| File | Tests |
|------|-------|
| `tests/test_rejection_reason_mapping.py` | Canonical reason mapping, sum invariants |
| `tests/test_method_registry.py` | Registry completeness, spec validation |
| `tests/test_b1_topk_variants.py` | B1-Top1 ≤1, B1-Top3 ≤3, proposal_rank/score |
| `tests/test_b1_matched.py` | Manifest frozen, budget sampling, no test leakage |
| `tests/test_a1_no_selected_set.py` | Feature audit, checkpoint loading, metadata |
| `tests/test_candidate_diagnostics.py` | Recall@K, router recall, harmful rejection |
| `tests/test_prefix_formation_trace.py` | Prefix recall, selection rate, success rates |

## A1-NoSet Definition

**Ablation**: Remove selected-set conditioning from the critic.

$$f(o, m, S) \rightarrow f(o, m)$$

The critic still uses:
- Task/context features (task tags, receiver role, environment facts)
- Candidate features (goal, preconditions, postconditions, environment)

But **excludes**:
- Selected-set size, selected cards, selected memory aggregate features
- Candidate-prefix pairwise interaction features
- Prefix conflict/complementarity features

**Implementation**: Uses `HashingTransferFeatureEncoder(feature_block="context_plus_candidate")` which filters tokens via `_include_token()`:
```python
if self.feature_block == "context_plus_candidate":
    return not is_selected and not is_interaction
```

**Training**: Same records, split, seeds, hyperparameters as M0 (critic_pi3_v22):
- Records: `data/paired_records_pi3_v22.jsonl` (200 records)
- Split: seed=7, test_fraction=0.2 → 160 train, 40 test
- Bootstrap: n_bootstrap=31, LogisticRegression base models
- Checkpoint: `checkpoints/critic_no_selected_set_v1.joblib`

## B1-Matched Budget Matching

**Purpose**: Control for share count confound — ensure B1 doesn't just perform worse because it shares more.

**Mechanism**: Invocation-level budget distribution matching.

1. Run M0 on validation set (existing `outputs/b0_b1_m0_all_scenarios/`)
2. Count per-invocation share counts: P(|S|=0), P(|S|=1), P(|S|=2), P(|S|=3)
3. Save as immutable manifest (`outputs/budget_manifest.json`)
4. B1-Matched samples budget from this distribution per invocation
5. Selects top-budget candidates by relevance ranking

**Manifest**:
```json
{
  "method": "M0",
  "max_shares_per_invocation": 3,
  "count_distribution": {"0": 0.5111, "1": 0.2593, "2": 0.1333, "3": 0.0963},
  "total_invocations": 3240,
  "seed": 7
}
```

**No test leakage**: Budget is fixed before testing; B1-Matched never reads test-set M0 outcomes.

## Rejection Reason Fix

**Problem**: `tau_lcb_nonpositive` was mapped to `other` in summary statistics because `summary.py` checked for `"tau_lcb_below_threshold"` but `causal_gate.py` emitted `"tau_lcb_nonpositive"`.

**Fix**: Canonical reason mapping in `summary.py`:
```python
_REASON_MAP = {
    "accepted": "shared",
    "tau_lcb_nonpositive": "tau_lcb_nonpositive",
    "negative_risk_ucb_exceeds_epsilon": "negative_risk_ucb_exceeded",
    "budget_exhausted": "share_budget_exceeded",
    "low_support": "low_support",
    ...
}
```

**Invariant**: `share_count + rejection_count == total_decisions` and `share_rate + sum(rejection_rates) == 1`.

## CLI Usage

### Run all 6 methods on a single scenario:
```bash
python -m smtr compare-routers \
  --db data/smtr_memory_v2.sqlite \
  --critic-checkpoint checkpoints/critic_pi3_v22.joblib \
  --a1-critic-checkpoint checkpoints/critic_no_selected_set_v1.joblib \
  --budget-manifest-path outputs/budget_manifest.json \
  --methods B0 B1-Top1 B1-Top3 B1-Matched A1-NoSet M0-Full \
  --scenario positive \
  --episodes 20 \
  --task-seeds 0 1 2 3 4 \
  --generation-seeds 0 1 \
  --traversal-seeds 0 1 2 \
  --output-dir outputs/ablation_positive \
  --overwrite
```

### Run all scenarios:
```bash
python scripts/run_ablation_experiments.py
```

### Train A1 critic:
```bash
python scripts/train_a1_critic.py
```

## Test Commands

```bash
# All tests
python -m pytest tests/ --ignore=tests/test_real_llm.py -q

# New ablation tests only
python -m pytest tests/test_method_registry.py tests/test_b1_topk_variants.py \
  tests/test_b1_matched.py tests/test_a1_no_selected_set.py \
  tests/test_candidate_diagnostics.py tests/test_prefix_formation_trace.py \
  tests/test_rejection_reason_mapping.py -v
```

**Result**: 544 passed, 12 skipped, 2 xfailed.
