# Next-Round Ablation Experiment

## Gap Analysis: What Exists vs What's Needed

| Spec Section | Status | Work Required |
|---|---|---|
| Section 1: Pre-experiment audit | NONE | New audit script |
| Section 2: Experiment config (40 eps, 5 trav seeds) | Runner exists | New run script with updated config |
| Section 3: Main results table | summary.py exists | Extend with new metrics |
| Section 4: Fair budget comparison (paired bootstrap CI) | Basic bootstrap exists | New paired comparison module |
| Section 5: Selected-set ablation (prefix matched-pair) | prefix_trace.py exists | New matched-pair audit module |
| Section 6: Bottleneck decomposition (5-stage funnel) | NONE | New module |
| Section 7: Rejection reason analysis (matched cases) | Canonical reasons exist | New matched-case audit module |
| Section 8: Statistical spec (group bootstrap) | Basic CI exists | Extend for paired comparisons |
| Section 9: Output files | writer.py exists | Extend with 6 new output files |
| Section 10: Interpretation criteria | NONE | Encode in report generator |
| Section 11: Final reply | NONE | Report template |

**Critical audit finding**: M0 metadata shows `train_class_distribution: {positive: 35, negative: 37, ...}` but A1 shows `{positive: 34, negative: 35, ...}`. Both claim `seed=7, test_fraction=0.2, training_records_digest=paired_records_pi3_v22.jsonl`. The `group_split` function splits by `episode_id` deterministically, so the split *should* be identical. The class distribution discrepancy needs investigation -- it may indicate different record loading or a metadata error. The audit script must verify this.

---

## Task 1: Pre-Experiment Audit Script

**New file**: `scripts/audit_next_ablation.py`

Print a structured audit report covering all 10 items from Section 1:

1. `git rev-parse HEAD` + `git status --short`
2. M0 checkpoint: path, SHA-256, feature_block (from metadata), train/test counts, split seed, critic_version
3. A1 checkpoint: path, SHA-256, confirm `selected_set_features_enabled=false`, confirm same `training_records_digest`
4. **Split verification**: Re-run `group_split(seed=7, test_fraction=0.2)` on `paired_records_pi3_v22.jsonl`, compare train/test episode_ids against metadata class distributions. If mismatch, HALT.
5. Budget manifest: validation_split_digest, critic_checkpoint_digest, count_distribution, confirm seed
6. Memory DB: path, record count, snapshot digest
7. Proposer: class name, configuration
8. top_k=4, max_shares_per_invocation=3
9. M0 and A1 gate: both use `FourOutcomeTransferCritic` with same gate logic
10. Run `pytest tests/test_rejection_reason_mapping.py -q` and report pass/fail

**Gate checks**:
- If A1 and M0 don't share the same training split -> HALT
- If B1-Matched manifest was derived from test M0 results -> HALT (check `validation_split_digest` != test split)

---

## Task 2: Bottleneck Funnel Module

**New file**: `src/smtr/experiment/bottleneck_funnel.py`

For each positive-transfer or prefix-dependent scenario, compute a 5-stage funnel:

```
Stage 1 (Proposer recall): positive_target_recall_at_4, required_prefix_recall_at_4
Stage 2 (Traversal/order): required_prefix_traversed_before_target_rate
Stage 3 (Prefix selection): required_prefix_selected_rate
Stage 4 (Target routing): target_share_given_correct_prefix, target_withhold_given_negative_prefix
Stage 5 (Execution): task_success_given_correct_target_selection
```

Input: runs.jsonl records + scenario ground-truth maps from `candidate_diagnostics.py`.

Output: per-scenario, per-method funnel with count and rate at each stage.

Key function:
```python
def compute_bottleneck_funnel(
    runs: list[dict], *, scenario: str, method: str
) -> BottleneckFunnelResult
```

Reuse `SCENARIO_TARGET_MEMORY`, `SCENARIO_PREFIX_MEMORIES`, `SCENARIO_TARGET_EFFECT` from `candidate_diagnostics.py`.

---

## Task 3: Paired Comparison Module

**New file**: `src/smtr/experiment/paired_comparisons.py`

Compute paired group bootstrap 95% CI for method pairs:
- M0-Full vs B1-Top1
- M0-Full vs B1-Top3
- M0-Full vs B1-Matched
- M0-Full vs A1-NoSet

Bootstrap unit: base episode = (scenario, task_seed, generation_seed).
Multiple traversal seeds belong to the same group.

For each pair, compute:
- success_difference (mean delta)
- negative_transfer_difference
- positive_transfer_difference
- average_selected_difference
- 95% paired bootstrap CI (using `numpy.percentile` on bootstrap deltas)

Key function:
```python
def compute_paired_comparisons(
    runs: list[dict], *, config: dict
) -> dict[str, PairedComparisonResult]
```

---

## Task 4: Prefix Matched-Pair Audit

**New file**: `src/smtr/experiment/prefix_matched_pair.py`

For prefix-sensitive and flip scenarios, compare M0-Full vs A1-NoSet on the same base episodes:

```
delta_tau correlation: corr(M0_tau_mean - A1_tau_mean, ground_truth_tau)
delta_tau MAE: mean|M0_tau - A1_tau - ground_truth_tau|
effect_direction_accuracy: fraction where sign(M0-A1) matches sign(ground_truth)
transfer_region_flip_accuracy: per-scenario flip accuracy
  positive_to_negative_accuracy
  negative_to_positive_accuracy
  neutral_to_negative_accuracy
  neutral_to_positive_accuracy
```

Uses per-episode critic predictions from `prefix_trace.py` traces.

Key function:
```python
def compute_prefix_matched_pair_audit(
    runs: list[dict], *, scenario: str
) -> PrefixMatchedPairResult
```

---

## Task 5: Rejection Reason Matched-Case Analysis

**New file**: `src/smtr/experiment/rejection_analysis.py`

For M0-Full and A1-NoSet:
1. Compute per-reason proportions (shared, tau_lcb_nonpositive, negative_risk_ucb_exceeded, low_support, share_budget_exceeded, other). Verify sum = 1.
2. Find matched discordant cases:
   - A1 share, M0 withhold (same episode, same candidate)
   - A1 withhold, M0 share
3. For each matched case, report:
   - candidate card metadata (candidate_memory_ids, positions)
   - selected-before-target IDs
   - A1 prediction (tau_mean, tau_lcb, negative_risk_ucb)
   - M0 prediction (same fields)
   - ground_truth effect
   - final task outcome

Key function:
```python
def compute_rejection_analysis(
    m0_runs: list[dict], a1_runs: list[dict], *, scenario: str
) -> RejectionAnalysisResult
```

---

## Task 6: Extend Writer for New Output Files

**File**: `src/smtr/experiment/writer.py`

Add methods to write the new output files required by Section 9:
- `write_decisions(records)` -> `decisions.jsonl` (one record per router decision with full metadata)
- `write_prefix_traces(traces)` -> `prefix_traces.jsonl`
- `write_scenario_slices(slices)` -> `scenario_slices.json`
- `write_bottleneck_funnel(funnel)` -> `bottleneck_funnel.json`
- `write_paired_comparisons(comparisons)` -> `paired_comparisons.json`

---

## Task 7: New Experiment Runner Script

**New file**: `scripts/run_next_ablation.py`

Orchestrates the full experiment:
1. Run audit (inline, not subprocess)
2. Configure: episodes=40, task_seeds=[0..4], gen_seeds=[0,1], trav_seeds=[0..4] (or [0..3] as smoke test), top_k=4, max_shares=3
3. Output to `outputs/next_ablation/`
4. Run all 6 methods x 9 scenarios using `ComparisonRunner`
5. Compute: candidate diagnostics, prefix traces, bottleneck funnel, paired comparisons, rejection analysis, prefix matched-pair audit
6. Write all output files per Section 9
7. Generate `report.md` with all 10 required sections

Config adjustment: 40 episodes = 5 task seeds x 2 gen seeds x 4 traversal seeds. But `traversal_seeds=[0,1,2,3]` gives 40. If cost is too high, fall back to `traversal_seeds=[0,1,2]` (20 base episodes x 3 trav seeds = 60 runs per method per scenario).

---

## Task 8: Report Generator

**New file**: `src/smtr/experiment/report_generator.py`

Generate `outputs/next_ablation/report.md` with the 10 required sections:
1. Experiment setup (config table, checkpoint info)
2. Fairness check (audit results)
3. Main results (per-scenario table + macro average)
4. Matched-budget comparison (M0 vs B1-Matched with CI)
5. M0 vs A1 (per-scenario selected-set ablation)
6. Prefix matched-pair audit (delta-tau metrics)
7. Proposer/router/execution funnel (5-stage bottleneck)
8. Rejection reason (per-method proportions + matched cases)
9. Representative failure cases (top 3 worst episodes)
10. Cautious conclusions (following Section 10 interpretation criteria)

---

## Task 9: Tests

New test files:
- `tests/test_bottleneck_funnel.py` -- funnel computation on fixture data
- `tests/test_paired_comparisons.py` -- bootstrap CI on synthetic data
- `tests/test_prefix_matched_pair.py` -- delta-tau on fixture data
- `tests/test_rejection_analysis.py` -- matched-case detection on fixture data

All existing tests must continue to pass.

---

## Task 10: Run Experiment and Generate Outputs

1. Run audit script, verify all checks pass
2. Execute full experiment (9 scenarios x 6 methods x 40 episodes)
3. Verify all output files exist in `outputs/next_ablation/`
4. Review report.md for completeness

---

## Execution Order

1. Task 1 (audit script)
2. Task 2 (bottleneck funnel)
3. Task 3 (paired comparisons)
4. Task 4 (prefix matched-pair audit)
5. Task 5 (rejection analysis)
6. Task 6 (extend writer)
7. Task 7 (new runner script)
8. Task 8 (report generator)
9. Task 9 (tests, verify all pass)
10. Task 10 (run experiment, generate outputs)
