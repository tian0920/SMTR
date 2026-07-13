# B0/B1/M0 Comparison Experiment Runner

## Architecture Overview

Create a new `src/smtr/experiment/` package containing the comparison runner, schemas, and summary computation. Reuse existing isolation mechanisms (`copy.deepcopy()` for state, `ReadOnlyPinnedMemoryView` for memory). Add a `compare-routers` CLI subcommand.

### Key Design Decisions

- **Reuse existing patterns**: `PairedRolloutCollector._run_branch()` pattern (deepcopy state + clone env + ReadOnlyPinnedMemoryView) for isolation
- **No modifications to router algorithms**: B0, B1, M0 remain unchanged
- **Single proposer per episode**: Build proposer once, reuse across all three methods
- **Streaming writes**: Write runs.jsonl incrementally (append per run)
- **Bootstrap via numpy**: Episode-level group resampling

---

## Task 1: Create experiment schemas (`src/smtr/experiment/__init__.py` + schemas)

Create `src/smtr/experiment/__init__.py` (empty) and `src/smtr/experiment/schemas.py`.

**`ExperimentConfig`** (Pydantic BaseModel):
- `db_path`, `critic_checkpoint`, `episodes`, `task_seeds`, `generation_seeds`, `traversal_seeds`, `top_k`, `max_shares_per_invocation`, `output_dir`, `overwrite`, `bootstrap_seed`, `bootstrap_n`

**`ComparisonRunRecord`** (Pydantic BaseModel):
- `experiment_id`, `episode_id`, `task_instance_id`, `method` (B0/B1/M0), `router_name`, `task_seed`, `environment_seed`, `generation_seed`, `traversal_seed` (null for B0/B1), `memory_snapshot_id`, `environment_snapshot_digest`, `candidate_memory_ids`, `selected_memory_ids`, `selected_count`, `team_success`, `failure_reason`, `policy_level_transfer_label`, `runtime_seconds`, `router_trace` (list of dicts with method-specific fields)

**`MethodSummary`** (Pydantic BaseModel):
- Per-method: `episode_count`, `success_rate`, `avg_selected_size`, `median_selected_size`, `all_withhold_rate`, `avg_candidate_count`, `mean_runtime`
- Transfer: `positive_transfer_rate`, `negative_transfer_rate`, `neutral_success_rate`, `neutral_failure_rate`, `success_delta_vs_b0`
- M0 rejection: `share_decision_rate`, `tau_lcb_rejection_rate`, `negative_risk_ucb_rejection_rate`, `share_budget_rejection_rate`, `low_support_rejection_rate`, `no_critic_rejection_rate`, `other_reason_counts`

**`ExperimentSummary`** (Pydantic BaseModel):
- `b0`, `b1`, `m0` (MethodSummary each), `m0_vs_b1` (comparison dict), `bootstrap_ci` (dict), `experiment_invalid` (bool), `invalid_reason` (str|null)

---

## Task 2: Create the comparison runner (`src/smtr/experiment/runner.py`)

**`ComparisonRunner`** class:

```python
class ComparisonRunner:
    def __init__(self, config: ExperimentConfig): ...
    def run(self) -> ExperimentSummary: ...
```

### Core loop:

```
for episode_idx in range(episodes):
    task_seed = config.task_seeds[episode_idx % len(config.task_seeds)]
    
    # 1. Generate task + environment (shared across methods)
    env = ToyEnvironment(seed=task_seed)
    env_snapshot = env.snapshot()
    env_snapshot_digest = canonical_digest(env_snapshot)
    
    # 2. Set up memory repository (shared snapshot)
    repository = SQLiteSharedMemoryRepository(config.db_path)
    seed_repository(repository)
    memory_snapshot = repository.create_read_snapshot()
    memory_snapshot_id = str(memory_snapshot.store_revision)
    
    # 3. Build proposer (shared)
    proposer = DeterministicHybridCandidateProposer()
    
    for gen_seed in config.generation_seeds:
        # 4. Run B0 (deterministic, once per episode group)
        b0_state = self._run_method(
            method="B0", router=NoMemoryRouter(),
            env_snapshot=env_snapshot, memory_snapshot=memory_snapshot,
            proposer=proposer, generation_seed=gen_seed, task_seed=task_seed,
        )
        
        # 5. Run B1 (deterministic, once per episode group)
        b1_state = self._run_method(
            method="B1", router=RelevanceTopKRouter(...),
            env_snapshot=env_snapshot, memory_snapshot=memory_snapshot,
            proposer=proposer, generation_seed=gen_seed, task_seed=task_seed,
        )
        
        # 6. Run M0 for each traversal seed
        for trav_seed in config.traversal_seeds:
            m0_state = self._run_method(
                method="M0", router=ProductionSequentialRouter(...),
                env_snapshot=env_snapshot, memory_snapshot=memory_snapshot,
                proposer=proposer, generation_seed=gen_seed, task_seed=task_seed,
                traversal_seed=trav_seed,
            )
    
    # 7. Compute transfer labels for B1 and M0 vs B0
    # 8. Write run records to runs.jsonl
```

### `_run_method()` — isolated execution:

```python
def _run_method(self, *, method, router, env_snapshot, memory_snapshot, 
                proposer, generation_seed, task_seed, traversal_seed=None):
    # Deep copy state (isolation)
    state = copy.deepcopy(initial_state(...))
    env = ToyEnvironment.clone_from_snapshot(env_snapshot, seed=generation_seed)
    memory_view = ReadOnlyPinnedMemoryView(repository, memory_snapshot)
    
    # Build and run graph
    app = build_graph(
        memory_pool=memory_view, proposer=proposer, router=router,
        config=RuntimeConfig(seed=generation_seed, top_k=config.top_k),
    )
    result = app.invoke(state)
    return result
```

### Transfer label computation:

```python
def _compute_transfer_label(b0_success, method_success):
    # method=1 (B1/M0), B0=0 → positive_transfer
    # method=0, B0=1 → negative_transfer
    # method=1, B0=1 → neutral_success
    # method=0, B0=0 → neutral_failure
```

### Error handling:
- Wrap each run in try/except
- Record failures in `errors.jsonl` and as run records with `failure_reason`
- Never silently drop from denominator

---

## Task 3: Create summary computation (`src/smtr/experiment/summary.py`)

**`compute_summary(runs, config) -> ExperimentSummary`**:
- Group runs by method
- Compute per-method metrics (success rate, avg/median selected size, etc.)
- Compute transfer labels for B1 and M0
- Compute M0 rejection reason counts from router_trace decisions
- Compute M0-vs-B1 differences
- Check for `no_critic_available` in M0 → mark experiment invalid

**`compute_bootstrap_ci(runs, config) -> dict`**:
- Bootstrap unit: `(task_instance_id, generation_seed)` group
- Resample groups with replacement (numpy RNG with `bootstrap_seed`)
- For each bootstrap sample, compute success rates for B1 and M0
- Return 2.5th and 97.5th percentiles for each metric

---

## Task 4: Create output writer (`src/smtr/experiment/writer.py`)

**`ExperimentWriter`** class:
- `__init__(output_dir, overwrite)`: Create dir, check existence
- `write_config(config)`: Write `config.json`
- `append_run(record)`: Append to `runs.jsonl`
- `append_error(error_record)`: Append to `errors.jsonl`
- `write_summary(summary)`: Write `summary.json`
- `load_runs() -> list[ComparisonRunRecord]`: Parse runs.jsonl

---

## Task 5: Add CLI subcommand (`src/smtr/cli.py`)

Add `compare-routers` subparser with all required args:
```
--db, --critic-checkpoint, --episodes, --task-seeds (nargs='+'),
--generation-seeds (nargs='+'), --traversal-seeds (nargs='+'),
--top-k, --max-shares-per-invocation, --output-dir, --overwrite
```

Add `_compare_routers(args)` dispatch function that:
1. Validates args (checkpoint exists, episodes > 0, top_k > 0, etc.)
2. Creates `ExperimentConfig`
3. Creates `ComparisonRunner` and calls `run()`
4. Prints summary to stdout

---

## Task 6: Write tests (`tests/test_compare_routers.py`)

15 test categories:

1. **Same task/env/memory snapshot**: Assert all three methods receive identical task, env observation, and memory snapshot digest
2. **State isolation**: Verify B0/B1/M0 runs don't mutate shared state (check env snapshot unchanged after each run)
3. **Transfer label mapping**: Test all 4 combinations of (method_success, b0_success)
4. **B0 selected set always empty**: Assert `selected_count == 0` for all B0 runs
5. **B1 selects by proposer order**: Verify B1 selected IDs match top-k of proposer ranking
6. **M0 loads and calls critic**: Verify M0 router_trace contains critic fields (tau_mean, etc.)
7. **M0 trace has critic prediction fields**: Verify tau_mean/tau_lcb/tau_ucb present in M0 decisions
8. **No `no_critic_available` in learned mode**: Assert M0 decisions don't have this reason
9. **JSONL output parseable**: Write and re-read runs.jsonl, verify valid JSON
10. **Summary consistent with runs**: Compute summary from runs.jsonl, compare with summary.json
11. **No overwrite without flag**: Verify existing output dir raises error without `--overwrite`
12. **Reproducibility**: Same config + seeds → identical runs.jsonl
13. **Runtime failure recorded**: Inject failing router, verify error in runs.jsonl and errors.jsonl
14. **Payload isolation**: Unselected payloads not in agent visible_payloads
15. **No regression**: Existing tests still pass

---

## Task 7: Run tests and fix issues

Run `python3 -m pytest tests/test_compare_routers.py -v`, fix any failures.

---

## Task 8: Run smoke experiment

```bash
python3 -m smtr.cli compare-routers \
  --db data/memory_compare_smoke.sqlite \
  --critic-checkpoint checkpoints/critic_pi0.joblib \
  --episodes 20 \
  --task-seeds 0 \
  --generation-seeds 0 \
  --traversal-seeds 0 1 2 \
  --top-k 4 \
  --max-shares-per-invocation 3 \
  --output-dir outputs/b0_b1_m0_smoke
```

Report: actual command, checkpoint, memory snapshot, B0/B1/M0 success rates, B1/M0 negative transfer, avg selected set size, M0 rejection reasons, errors, output file paths.

---

## Task 9: Full test suite + ruff

Run full test suite to verify no regressions. Run ruff for code quality.

---

## Dependencies

- Task 1 (schemas) → Tasks 2, 3, 4
- Task 2 (runner) → Task 5 (CLI), Task 6 (tests)
- Task 3 (summary) → Task 2 (runner), Task 5 (CLI)
- Task 4 (writer) → Task 2 (runner)
- Task 5 (CLI) → Task 7 (run tests)
- Task 6 (tests) → Task 7 (run tests)
- Task 7 → Task 8 (smoke)
- Task 8 → Task 9 (full suite)

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| `run_episode()` modifies memory store | Use `ReadOnlyPinnedMemoryView` (proven in paired_rollout.py) |
| Deep copy of SMTRState is incomplete | SMTRState is a TypedDict with plain dicts/lists — deepcopy works |
| M0 critic not loaded properly | Fail fast if checkpoint missing; test critic fields present |
| Bootstrap CI incorrect | Use episode-level groups, not individual runs |
| Circular imports in new experiment module | Import from existing modules only; no reverse dependencies |
| ToyEnvironment not deterministic across clones | Use `clone_from_snapshot()` with explicit seed |

## Critical Files

1. `src/smtr/experiment/runner.py` — core comparison runner
2. `src/smtr/experiment/schemas.py` — Pydantic models for config/records/summary
3. `src/smtr/experiment/summary.py` — metrics and bootstrap CI computation
4. `src/smtr/cli.py` — CLI integration
5. `tests/test_compare_routers.py` — 15 test categories
