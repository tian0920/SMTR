# Ablation Methods and Diagnostics Implementation Plan

## Task 1: Fix Rejection Reason Statistics (Section 1)

**Root cause**: `summary.py` line 170 checks `reason == "tau_lcb_below_threshold"` but `causal_gate.py` line 24 emits `"tau_lcb_nonpositive"`. This causes all tau-LCB rejections to fall into `other_reason_counts`.

**Files to modify**:
- `src/smtr/experiment/summary.py` — Fix `_add_m0_rejection_metrics()`:
  - Change `"tau_lcb_below_threshold"` to `"tau_lcb_nonpositive"`
  - Add canonical reason mapping function `canonicalize_reason(reason) -> str` with the 7 standard categories: `shared`, `tau_lcb_nonpositive`, `negative_risk_ucb_exceeded`, `low_support`, `share_budget_exceeded`, `no_critic_available`, `other`
  - Map legacy reasons: `"tau_lcb_below_threshold"` -> `"tau_lcb_nonpositive"`, `"negative_risk_ucb_exceeds_epsilon"` -> `"negative_risk_ucb_exceeded"`, `"budget_exhausted"` -> `"share_budget_exceeded"`, `"accepted"` / `"critic_guided_share"` / `"epsilon_exploration"` -> `"shared"`
- `src/smtr/router/traces.py` — Add `proposal_rank` and `proposal_score` fields to `RouterDecision` (needed for Section 2)
- `tests/test_rejection_reason_mapping.py` — New test file:
  - Test `tau_lcb_nonpositive` correctly counted as LCB rejection
  - Test all decision reasons sum to total decisions
  - Test `share_rate + sum(rejection_rates) == 1`

## Task 2: Method Registry (Section 7)

**New file**: `src/smtr/experiment/methods.py`

```python
METHOD_REGISTRY = {
    "b0_no_memory": MethodSpec(
        router_class="NoMemoryRouter", critic_checkpoint=None,
        feature_block=None, share_budget_policy="zero",
        gate_policy="none", uses_selected_set=False,
        uses_pairwise_interactions=False,
    ),
    "b1_top1": MethodSpec(
        router_class="RelevanceTopKRouter", critic_checkpoint=None,
        feature_block=None, share_budget_policy="fixed_1",
        gate_policy="none", uses_selected_set=False,
        uses_pairwise_interactions=False,
    ),
    "b1_top3": MethodSpec(..., share_budget_policy="fixed_3", ...),
    "b1_matched": MethodSpec(
        router_class="RelevanceTopKRouter", ...,
        share_budget_policy="validation_matched", ...
    ),
    "a1_no_selected_set": MethodSpec(
        router_class="ProductionSequentialRouter",
        feature_block="context_plus_candidate",
        uses_selected_set=False, uses_pairwise_interactions=False,
    ),
    "m0_full": MethodSpec(
        router_class="ProductionSequentialRouter",
        feature_block="full",
        uses_selected_set=True, uses_pairwise_interactions=True,
    ),
}
```

- Update `ExperimentConfig.method` to accept method_id from registry instead of hardcoded `"B0"/"B1"/"M0"`
- Update `ComparisonRunRecord.method` to use `Literal["B0", "B1-Top1", "B1-Top3", "B1-Matched", "A1-NoSet", "M0-Full"]`
- Update `ExperimentSummary` to hold a dict of method summaries instead of fixed b0/b1/m0 fields
- Update `runner.py` to build routers from method registry specs
- Update `summary.py` to compute summaries per method_id

## Task 3: B1-Top1 and B1-Top3 (Section 2)

No new router class needed — reuse `RelevanceTopKRouter` with different `max_shares_per_invocation`:
- B1-Top1: `max_shares_per_invocation=1`
- B1-Top3: `max_shares_per_invocation=3`

**Files to modify**:
- `src/smtr/router/baselines.py` — Add `proposal_rank` and `proposal_score` to each `RouterDecision` (1-based rank, proposer score)
- `src/smtr/experiment/runner.py` — Build B1-Top1/B1-Top3 routers from method registry
- `tests/test_b1_topk_variants.py` — New tests:
  - B1-Top1 selects at most 1 per invocation
  - B1-Top3 selects at most 3 per invocation
  - Both use same proposer, no critic calls
  - Trace records `proposal_rank` and `proposal_score`

## Task 4: B1-Matched (Section 3)

**New file**: `src/smtr/experiment/budget_manifest.py`

- `ShareBudgetManifest` — immutable manifest storing M0's per-invocation share count distribution from a validation run:
  ```python
  class ShareBudgetManifest(BaseModel):
      method: str = "M0"
      max_shares_per_invocation: int
      count_distribution: dict[str, float]  # {"0": 0.60, "1": 0.15, ...}
      validation_split_digest: str | None
      critic_checkpoint_digest: str | None
      seed: int
  ```
- `build_manifest_from_m0_runs(runs)` — compute distribution from M0 validation runs
- `sample_budget(manifest, rng)` — sample an invocation budget from the distribution

**B1-Matched router logic**:
- Still uses `RelevanceTopKRouter` but with per-invocation budget sampled from manifest
- At each invocation: `budget = sample_budget(manifest, rng_for_this_invocation)`, then select top-`budget` by relevance
- The manifest is loaded once at experiment start from a pre-computed file
- RNG is seeded deterministically per (episode_id, generation_seed) to ensure reproducibility

**How to generate the manifest**: Run M0 on the validation set first (or use existing experiment data), extract per-invocation share counts, compute distribution, save as JSON.

**Files to create/modify**:
- `src/smtr/experiment/budget_manifest.py` — New
- `src/smtr/router/baselines.py` — Add `BudgetMatchedTopKRouter` variant (or add `budget_fn` parameter to `RelevanceTopKRouter`)
- `src/smtr/experiment/runner.py` — Wire B1-Matched with manifest
- `tests/test_b1_matched.py`:
  - B1-Matched does not read test M0 outcome
  - Fixed manifest is reproducible
  - Budget sampling respects distribution

## Task 5: A1 — SMTR without Selected Set (Section 4)

**Feature block verification**: The existing `context_plus_candidate` block in `transfer_features.py` (line 158-159) already excludes `selected_*` and `interaction_*` tokens. This is exactly what A1 needs.

**Training A1 critic**:
- Use same training records (paired_records JSONL), same train/val/test split, same bootstrap seeds, same hyperparameters
- Only change: `HashingTransferFeatureEncoder(feature_block="context_plus_candidate")`
- Save as `checkpoints/critic_no_selected_set_v1.joblib`

**New file**: `scripts/train_a1_critic.py`
- Load same paired records as M0 training
- Use same split (from metadata)
- Train with `feature_block="context_plus_candidate"`
- Save checkpoint + metadata JSON

**Files to modify**:
- `src/smtr/router/factory.py` — Accept `feature_block` parameter, pass to encoder
- `src/smtr/experiment/runner.py` — Wire A1 method
- `tests/test_a1_no_selected_set.py`:
  - A1 feature tokens contain no `selected_*`, `interaction_*` prefixes
  - A1 checkpoint loads in ProductionSequentialRouter
  - A1 uses same training split as M0

## Task 6: Candidate-Level Diagnostics (Section 5)

**New file**: `src/smtr/experiment/candidate_diagnostics.py`

Compute per-scenario, per-decision-point metrics using counterfactual ground truth:

1. **Candidate positive Recall@K**: Is the positive-transfer target memory in proposer top-K?
2. **Candidate negative Recall@K**: Is the negative-transfer target memory in proposer top-K?
3. **Router positive recall**: P(share target | target positive, target in candidates)
4. **Harmful-memory rejection**: P(withhold target | target negative, target in candidates)
5. **Positive transfer precision**: #positive_shared / #total_shared
6. **Neutral exposure rate**: #neutral_shared / #total_shared

**Ground truth source**: Use `CounterfactualToyTaskProvider` scenario metadata to determine each memory's true transfer class (positive/negative/neutral) — not scenario name guessing.

**Integration**: Add to `ComparisonRunner.run()` after main loop, using collected runs + scenario metadata.

**Files to create/modify**:
- `src/smtr/experiment/candidate_diagnostics.py` — New
- `src/smtr/experiment/schemas.py` — Add `CandidateDiagnosticsSummary` to `ExperimentSummary`
- `tests/test_candidate_diagnostics.py`:
  - Recall@K correct on hand-crafted fixture
  - Router positive recall and harmful rejection correct

## Task 7: Prefix Formation Trace (Section 6)

**New file**: `src/smtr/experiment/prefix_trace.py`

For prefix-sensitive and flip scenarios, record per-target routing chain:
- Which prefix memories were required
- Whether they entered candidates, their ranks, traversal order
- Whether they were selected before the target
- Critic prediction on the target
- Target action and ground truth region

**Summary metrics**:
- `prefix_candidate_recall`
- `prefix_order_success_rate`
- `prefix_selection_success_rate`
- `target_evaluated_under_correct_prefix_rate`
- `success_given_correct_prefix`
- `success_without_correct_prefix`

**Files to create/modify**:
- `src/smtr/experiment/prefix_trace.py` — New
- `src/smtr/experiment/schemas.py` — Add `PrefixTraceSummary`
- `tests/test_prefix_formation_trace.py`:
  - Correctly identifies: prefix not recalled, prefix order wrong, prefix rejected, target evaluated under correct prefix

## Task 8: Integration and Runner Update

**Files to modify**:
- `src/smtr/experiment/runner.py` — Major refactor:
  - Accept list of method_ids instead of hardcoded B0/B1/M0
  - Build each router from method registry
  - Run all methods per episode with same fairness guarantees
  - Collect candidate diagnostics and prefix traces
- `src/smtr/experiment/schemas.py` — Update `ExperimentConfig` to accept `methods: list[str]`
- `src/smtr/cli.py` — Update `compare-routers` to accept `--methods` flag

## Task 9: Tests and Acceptance (Sections 8-9)

**New test files**:
- `tests/test_rejection_reason_mapping.py` (Task 1)
- `tests/test_b1_topk_variants.py` (Task 3)
- `tests/test_b1_matched.py` (Task 4)
- `tests/test_a1_no_selected_set.py` (Task 5)
- `tests/test_candidate_diagnostics.py` (Task 6)
- `tests/test_prefix_formation_trace.py` (Task 7)
- `tests/test_method_registry.py` — Test registry completeness and validation

**Acceptance verification**:
1. CLI can build and run all 6 methods
2. A1 is independently trained no-selected-set critic
3. B1-Matched budget only from validation set
4. Rejection reasons no longer contradictory
5. Per-decision candidate-level diagnostics output
6. Prefix scenarios output prefix formation trace
7. All tests pass, no regression

## Execution Order

1. Task 1 (rejection fix) — unblocks accurate stats everywhere
2. Task 2 (method registry) — foundation for all new methods
3. Task 3 (B1-Top1/Top3) — simple config change, validates registry
4. Task 4 (B1-Matched) — needs M0 validation data
5. Task 5 (A1 training) — can run in parallel with Task 4
6. Task 6 (candidate diagnostics) — needs scenarios working
7. Task 7 (prefix trace) — needs scenarios working
8. Task 8 (integration) — wire everything together
9. Task 9 (tests + smoke) — final validation
