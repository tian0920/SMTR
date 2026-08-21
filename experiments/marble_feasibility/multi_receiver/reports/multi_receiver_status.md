# Multi-Receiver Configuration Status Report (Task 3)

**Status: IMPLEMENTATION COMPLETE — LIVE COLLECTION BLOCKED (expired LLM API key)**

## What was delivered

### 1. `configs/marble_3receiver.yaml`
- receiver_count: 3 (agent0 / agent1 / agent2), each with receiver_id, role, memory access
- Minimal validation scale: Tasks=3, Receivers=3, memory pool=100, Seeds=[0,1,2]
- Sanity thresholds: `receiver_effect_variance > 0`, SMTR (receiver-conditioned) beats global

### 2. `collect_interventions.py` — multi-receiver aware
- Records are grouped by `receiver_agent_id`; the receiver dimension genuinely varies
- Informative resampling is done per-receiver independently (per-receiver label balance)
- New output: `data/receiver_statistics.json` (per-receiver counts / informative ratio)
- Regression-tested on existing single-receiver data: identical output (agent1: 500 pairs, 50% informative)

### 3. `multi_receiver/collect_multi_receiver.py`
- Runs the real MARBLE engine for each (task, receiver, seed) group:
  one shared no-memory control + one share branch per candidate memory
- Identical helpful/harmful memory payloads across receivers → any
  tau(m,r1) != tau(m,r2) is attributable to the receiver
- Output schema kept compatible with the feasibility pipeline:
  `{ task, receiver, memory, Y_expose, Y_withhold }`

### 4. `multi_receiver/run_receiver_sanity.py`
- Check 1: `receiver_effect_variance` across tau(m,r1..r3) for the same memory;
  requires > 0 (heterogeneity)
- Check 2: Global tau(m) vs receiver-conditioned tau(m,r) leave-one-out MAE;
  expects SMTR <= Global
- Logic validated on synthetic heterogeneous data:
  variance=0.1667 detected, recv_mae=0.333 < global_mae=0.600, smtr_beats_global=True

## Live collection attempt

Ran: `task=1 (database), receivers=agent0/1/2, seed=0, helpful+harmful memories`

Result: engine launches and Docker DB environment comes up, but all LLM
calls fail with `litellm.BadRequestError: Invalid API-key provided`.

Root cause: both available keys are expired:
- `sk-74ff...198d` (scripts/run_full_q30b_experiment.sh)
- `sk-86cc...92fd` (conf/llm_test_config.json)

The 2 completed pairs carry `invalid_reason: shared_control_invalid:real_marble_engine_not_executed`
(initial_state_match=True and memory_intervention_verified=True confirm the
non-LLM parts of the pipeline are correct).

## Receiver heterogeneity verdict

**Cannot be determined yet** — requires a valid LLM API key.
Once a key is available:

```bash
cd /home/ecs-user/SMTR
export MARBLE_ROOT=/home/ecs-user/MARBLE
export MARBLE_LLM_MODEL=qwen3-30b-a3b
export DASHSCOPE_API_KEY=<valid_key>
export OPENAI_API_KEY=<valid_key>
export OPENAI_BASE_URL=https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
export SMTR_LLM_ENABLE_THINKING=false

# 1) minimal smoke (1 task, 3 receivers, 1 seed)
python experiments/marble_feasibility/multi_receiver/collect_multi_receiver.py --task-ids 1 --seeds 0

# 2) full validation scale (3 tasks, 3 receivers, seeds 0 1 2)
python experiments/marble_feasibility/multi_receiver/collect_multi_receiver.py --task-count 3 --seeds 0 1 2

# 3) sanity checks
python experiments/marble_feasibility/multi_receiver/run_receiver_sanity.py
```

## Decision tree (per task spec)

- Encoder ablation: **PASS** (task_id is causal driver; no metadata shortcut)
- Leakage audit: **PASS** (no non-causal identity shortcut)
- Receiver heterogeneity: **PENDING** (blocked on API key)

→ If heterogeneity is confirmed after re-run: proceed to formal ICLR main experiments.
→ If receivers show no difference: adjust paper claims (do not emphasize receiver-conditioned).
