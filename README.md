# SMTR — RIMA: Receiver-conditioned Intervention-based Memory Admission

Research codebase for **RIMA**, a canonical method for cross-agent
procedural memory transfer in multi-agent systems, evaluated on the
[MARBLE](https://github.com/sakanaai/MARBLE) multi-agent benchmark.

The repository package name stays `smtr` for continuity; the formal
method, experiments, configs, results and docs live under the unified
`rima/` namespace (`experiments/rima/`, `configs/rima/`,
`results/rima/`, `docs/rima/`, `src/smtr/rima/`).

## The RIMA Pipeline

```text
Historical experience
  → Shared Memory Pool  M^t            (memories from tasks < t only)
  → Receiver-conditioned retrieval  C_r^t
  → Frozen transfer critic  tau_hat(m, r | x_t)
  → Receiver-specific admission  A_r^t(m) = I[tau_hat > 0]
  → Receiver knowledge state  K_r      (per-receiver, never broadcast)
  → Future task reuse
```

Key properties:

- **Official continuous outcome.** The estimand is defined on the
  normalized official MARBLE Task Score per scenario
  (`src/smtr/rima/outcome.py`). Team success is diagnostic metadata
  only and never affects admission.
- **Two-potential-outcome transfer critic.** The critic estimates
  `tau(m, r | x) = E[Y(1) - Y(0) | m, r, x]` with separate
  expose/withhold heads, is trained on matched offline interventions
  with a task-level split, and is **frozen** before any continual
  evaluation (`src/smtr/router/official_score_transfer_critic.py`).
- **Admission is exactly `tau_hat > 0`** (Eq. 8). No risk gate, no
  epsilon/eta thresholds in the formal path.
- **Multi-memory admission.** All positive-tau memories are admitted;
  there is no top-1 restriction.
- **Multi-receiver simultaneous execution** with receiver-specific
  knowledge; memory unions are never broadcast.
- **Self-transfer excluded** (`source(m) == receiver`), counted
  separately, and never admitted.
- **Temporal invariant.** A memory can only influence tasks strictly
  after its origin task; current-task memories never affect the
  current task.
- **Fail-closed everywhere.** Invalid outcomes yield `delta = None`
  (never `0`), are excluded from training and counted, and the formal
  decision source is hard-checked to be `frozen_transfer_critic`.

## Repository Layout (formal path)

| Path | Purpose |
|---|---|
| `src/smtr/rima/` | outcome, features, admission, split, critic validation, metrics |
| `src/smtr/memory/` | shared memory pool, receiver knowledge, sanitizer |
| `src/smtr/router/official_score_transfer_critic.py` | canonical critic |
| `experiments/rima/train_critic.py` | Stage B–C: split audit + train + freeze |
| `experiments/rima/run_mechanism_eval.py` | matched expose/withhold mechanism evidence |
| `experiments/rima/run_continual_main.py` | canonical continual runner (6 methods) |
| `configs/rima/continual_protocol.yaml` | single canonical protocol config |
| `tests/rima/` | canonical integrity test suite (20 invariants) |
| `docs/rima/` | metric system and protocol documentation |
| `docs/experiment_lineage/rima_canonical_migration.md` | generation lineage (v1 → prototype → canonical) |

Methods compared by the canonical runner (same backbone, task order,
retrieval budget, candidate pool, context budget and seeds):
`no_memory`, `full_memory`, `retrieval`, `reflexion`,
`rima_uniform` (receiver-agnostic critic ablation),
`rima_receiver` (full RIMA).

## Quick Start

```bash
# 0. integrity suite (no LLM calls): RIMA_CANONICAL_INTEGRITY = PASS
python -m pytest tests/rima/ tests/marble/test_procedural_memory_sanitization.py -q

# 1. train + freeze critics (requires intervention records + LLM creds)
python experiments/rima/train_critic.py --records <records.json> --out artifacts/rima/critics/

# 2. mechanism evaluation (offline interventions only)
python experiments/rima/run_mechanism_eval.py ...

# 3. canonical continual experiment
python experiments/rima/run_continual_main.py \
    --scenario bargaining --method rima_receiver \
    --critic-receiver artifacts/rima/critics/receiver_critic.joblib
```

Outputs land in `results/rima/`.

## Metric System

Primary metric: **Official Task Score**. Memory quantities
(`memory_bank_size`, `validated_memory_count`, `candidate_count`) are
diagnostic-only. Intervention cost is reported separately from online
inference cost. See [`docs/rima/metrics.md`](docs/rima/metrics.md).

## Legacy Generations

Earlier research generations are preserved, not deleted:

- **SMTR-v1** (single receiver / single exposure / binary team
  outcome) — code under `src/smtr/router/transfer_critic.py`,
  `src/smtr/marble/` and legacy tests; original README archived at
  [`docs/archive/smtr_v1_method.md`](docs/archive/smtr_v1_method.md);
  retained only for controlled ablations and historical reproducibility.
- **Online receiver-3 prototype** (observed-delta oracle admission) —
  remains as an oracle upper bound; `observed_delta` is hard-blocked
  from the formal admission path.

Lineage, freeze points and result reuse policy:
[`docs/experiment_lineage/rima_canonical_migration.md`](docs/experiment_lineage/rima_canonical_migration.md).
