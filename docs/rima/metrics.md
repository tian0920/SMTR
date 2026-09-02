# RIMA Canonical Metric System (Phase 23–26)

Authoritative metric definitions for all formal RIMA experiments.
Implemented in [`src/smtr/rima/metrics.py`](../../src/smtr/rima/metrics.py)
and emitted by the canonical runner
(`experiments/rima/run_continual_main.py`, block `rima_metrics` /
`cost_report`).

## 1. Metric hierarchy (Phase 23)

| Tier | Metric | Definition |
|---|---|---|
| **Primary** | Official Task Score | normalized official MARBLE metric per scenario (`root_cause_recall` / `avg_innovation_safety_feasibility` / `block_hit_rate` / `avg_code_quality` / `avg_negotiation_quality`), averaged over **valid** outcomes |
| Secondary | Coordination Score | diagnostic companion, never a primary claim |
| Long-term | Cumulative Task Score | sum of valid task scores over the continual run |
| Long-term | Late-stage Task Score | mean of the last 5 valid task scores |
| Memory | admission rate | n_admitted / n_formal_candidates (self-transfer & invalid excluded from denominator) |
| Memory | cross-task reuse rate | fraction of admitted memories reused on ≥2 distinct tasks |
| Memory | receiver-specific reuse rate | mean over memories of (admitting receivers / deciding receivers) |
| Memory | harmful admission rate | fraction of admitted memories with observed delta < 0 (only where mechanism evaluation attached an observed delta) |
| Memory | receiver disagreement rate | fraction of memories decided for ≥2 receivers whose status differs across receivers |
| Critic | tau prediction correlation | Pearson(tau_hat, observed_delta) on mechanism-eval pairs |
| Critic | sign accuracy | fraction of pairs with sign(tau_hat) == sign(observed_delta) |
| Cost | see §3 | offline vs online, reported separately |

## 2. Memory quantity is NOT a success metric (Phase 24)

`memory_bank_size`, `validated_memory_count`, `candidate_count` are
**diagnostic-only**. They are reported under a dedicated
`diagnostic_only` block in every run summary.

The paper must not claim "more validated memories = better". The claim
target is **future team performance** (Official Task Score on future
tasks).

## 3. Intervention cost protocol (Phase 25)

Interventions (matched expose/withhold executions) happen **only**
during:

- critic training (`experiments/rima/train_critic.py`), and
- mechanism evaluation (`experiments/rima/run_mechanism_eval.py`).

They **never** happen during formal inference/admission. Every run
summary therefore reports cost in two separate blocks:

- `offline_intervention_cost`: intervention collection episodes + critic
  training time;
- `online_rima_inference_cost`: frozen-critic inference only (wall time
  per task, extra tokens; `intervention_episodes_in_formal_path = 0`
  invariant).

This separation is a key advantage of RIMA over oracle online TCI, and
must be preserved in all tables/figures.

## 4. Intervention budget experiment (Phase 26)

The existing 25% / 50% / 75% / 100% budget sweeps are retained but
re-labeled:

- **Old label**: execution budget (formal execution restricted).
- **Canonical label**: **critic supervision budget** — the fraction of
  matched intervention pairs used to *train the critic*. Formal
  execution is never budgeted.

The reported chain is:

```
intervention (supervision) budget → critic quality → final continual performance
```

Legacy budget artifacts under `outputs/tci_smtr_budget_*` remain valid
as historical evidence under the old framing; new runs must use the
canonical label.

## 5. Fail-closed handling of invalid outcomes (Phase 27)

All formal outcome objects carry `is_valid`:

- invalid ⇒ score/delta = `None` (never `0`);
- invalid outcomes are excluded from training pairs and counted
  (`n_invalid_pairs_excluded`, `n_invalid`);
- invalid outcomes are reported separately in mechanism reports;
- an invalid outcome must never be converted into a silent `rejected`.

## 6. Risk gate removed from the formal path (Phase 28)

The paper's admission rule (Eq. 8) is exactly:

```
A_r^t(m) = I[tau_hat(m, r | x_t) > 0]
```

No `epsilon_star`, no calibrated `eta` threshold, no risk gate exists
in the formal path (`src/smtr/rima/`, `experiments/rima/`,
`src/smtr/router/official_score_transfer_critic.py`). Verified: zero
occurrences of `epsilon_star` / `eta_cal` / `risk_gate` in those
modules.

The epsilon/eta machinery belongs to the SMTR-v1 generation and lives
only in the legacy modules (`src/smtr/router/transfer_critic.py`,
`src/smtr/marble/formal_protocol.py`, `src/smtr/marble/training.py`,
`src/smtr/marble/paired_evaluation.py`), which are deprecation-marked
and retained solely for controlled ablations / historical
reproducibility (see `docs/experiment_lineage/rima_canonical_migration.md`).
