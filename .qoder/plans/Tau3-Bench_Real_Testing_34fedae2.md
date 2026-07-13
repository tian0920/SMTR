# Tau3-Bench Real Testing Integration

## Problem

SMTR uses system Python; tau3-bench has its own venv at `/home/ecs-user/tau2-bench/.venv/`. The existing runner (`run_smtr_tau3.py`) uses outdated tau3 API calls. No real end-to-end test has been completed.

## Phase 1: Environment Bridge

Make `smtr` importable from tau2-bench's venv so a single script can use both.

**1.1** Install SMTR as editable into tau2-bench's venv:
```bash
/home/ecs-user/tau2-bench/.venv/bin/pip install -e /home/ecs-user/SMTR
```
This gives tau2-bench's Python access to both `tau2` and `smtr.*`.

**1.2** Verify dual imports work:
```bash
/home/ecs-user/tau2-bench/.venv/bin/python3 -c "import tau2; import smtr; print('OK')"
```

## Phase 2: Fix Outdated API in `run_smtr_tau3.py`

The current `run_tau3_eval()` function uses non-existent imports. Fix to match actual tau3 API:

**2.1** Replace imports in `run_tau3_eval()` (line 96-98):
- Old: `from tau2.domains.retail.environment import get_environment, get_tasks`
- New: `from tau2.runner.build import build_environment, build_agent, build_user`
- Old: `from tau2.orchestrator import Orchestrator`
- New: `from tau2.orchestrator.orchestrator import Orchestrator`
- Old: `from tau2.user import create_user_simulator`
- New: use `build_user("user_simulator", env, task, llm=...)`

**2.2** Fix task loading:
- Replace `get_tasks()` with loading from `data/tau2/domains/retail/tasks.json` + `split_tasks.json` for split awareness.

**2.3** Fix agent creation:
- Use `build_agent("llm_agent", env, llm=..., llm_args=...)` for the baseline (no SMTR memory) branch.
- Use `SMTRTauAgent` for the memory-augmented branch.

**2.4** Fix orchestrator usage:
- Correct constructor params: `Orchestrator(domain=, agent=, user=, environment=, task=, max_steps=, seed=)`.
- `run_simulation(orchestrator)` returns the simulation result.

## Phase 3: End-to-End Baseline Test (No Memory)

Before testing SMTR memory injection, confirm the tau3 pipeline works with a plain LLM agent.

**3.1** Run 3 retail dev/test tasks with the baseline `llm_agent` using qwen_remote config:
```bash
/home/ecs-user/tau2-bench/.venv/bin/python3 run_smtr_tau3.py \
  --domain retail --num-tasks 3 \
  --agent-llm "qwen3.5-plus" \
  --agent-llm-args '{"api_base": "...", "api_key": "..."}' \
  --user-llm "qwen3.5-plus" \
  --output-dir outputs/tau3_baseline
```

**3.2** Verify results JSON is produced with reward scores.

## Phase 4: SMTR Memory-Augmented Test

**4.1** Build a memory pool from collected trajectories (existing data in `data/tau3_*` directories) or create a small test pool from training task patterns.

**4.2** Run paired evaluation: same tasks, compare baseline vs SMTR-augmented agent. Use `Tau3PairedRolloutRunner` or the updated `run_smtr_tau3.py` with `--memory-pool`.

**4.3** Collect and compare rewards between branches.

## Phase 5: Results & Documentation

**5.1** Save results to `outputs/tau3_real_test/`.
**5.2** Update `results.md`, `changelog.md`, and `implementation.md` with real testing outcomes.

## Key Files

| File | Action |
|------|--------|
| `run_smtr_tau3.py` | Fix tau3 API calls, update agent/env construction |
| `src/smtr/runtime/tau3_agent.py` | Verify compatibility with current tau3 API (may need minor fixes) |
| `src/smtr/counterfactual/tau3_paired_rollout.py` | Verify `_run_branch()` uses correct Orchestrator API |
| `conf/llm_test_config.json` | Reference for LLM configs (qwen_remote) |

## Risks

- tau3-bench API may have subtle differences from what `SMTRTauAgent` expects (message types, tool format). Will debug iteratively.
- LLM API key/rate limits for qwen_remote. Will start with 2-3 tasks.
- `smtr` pip install into tau2 venv may pull conflicting dependency versions. Will check before committing.
