# RIMA Canonical Migration — Experiment Lineage

**Freeze point**: `2833b164de4071d8bcb6c0919a37b16e238be81d`
**Tag**: `pre-rima-canonical-refactor`
**Frozen on**: 2026-08-31
**Previous tag**: `v0.2-marble-database-runtime-audited` (`3572066c`)

## Purpose

This document freezes the experiment lineage before the RIMA canonical
refactor. All historical experiments and results are PRESERVED, but they
are demoted to `legacy` / controlled-ablation status and MUST NOT enter
new RIMA main tables.

## Three Generations

### A. SMTR-v1 (legacy)

- Single receiver, single memory exposure (`S = empty`)
- Binary team success as outcome
- Four-outcome critic `q_theta = [q00,q01,q10,q11]`, `tau_hat = q10 - q01`
- Admission: `tau_hat > 0 AND eta <= epsilon*` (risk gate)
- Code: `src/smtr/router/transfer_critic.py` (kept, renamed to
  `BinaryFourOutcomeTransferCritic` semantics), `legacy/`
- Status: **legacy / controlled ablation only**. Renamed experiment:
  `RIMA-Binary` (appendix comparison). Must NOT be the MultiAgentBench
  main method.

### B. Online receiver3 oracle prototype (legacy)

- Real online expose/withhold intervention per task
- Observed delta directly decides admission (oracle)
- Code: `experiments/marble_receiver3/run_online_main.py` and related
- Results: `results/marble/bargaining_tci_mechanism_pilot/` (pilot data,
  mechanism-evidence only)
- Status: **oracle upper bound / mechanism evidence only**. Observed-delta
  admission is FORBIDDEN in the formal evaluation path because the
  counterfactual outcome is unavailable at admission time.

### C. RIMA canonical (formal)

- Receiver-conditioned Intervention-based Memory Admission
- Historical matched interventions -> frozen receiver-conditioned
  continuous potential-outcome critic -> persistent multi-receiver memory
  admission
- Formal pipeline:
  `M^t (shared pool) -> C_r^t (receiver-conditioned retrieval) ->
  frozen transfer critic -> tau_hat(m,r|x_t) -> admission I[tau_hat > 0]
  -> receiver-specific K_r -> future task reuse`
- Code: `src/smtr/memory/shared_memory_pool.py`,
  `src/smtr/memory/receiver_knowledge.py`,
  `src/smtr/router/official_score_transfer_critic.py`,
  `experiments/rima/run_continual_main.py` (new)
- Outcome: normalized official MultiAgentBench Task Score in [0,1]
- Admission: `decision_source == "frozen_transfer_critic"` ONLY

## Formal Evaluation Path Prohibitions

In the RIMA canonical evaluation path the following are forbidden:

- observed expose/withhold delta directly deciding admission
- `team_success` fallback (diagnostic metadata only)
- current-task-generated memory reused in the current task
- single-memory-only router (top-1 admission limit)
- receiver memory broadcast (union -> all receivers)
- self-transfer (`source(m) == receiver`)
- oracle outcome
- ground-truth leakage (answers, scores, hidden state in payloads)

## Result Reuse Policy

| Generation | Results kept? | In RIMA main tables? |
|---|---|---|
| A. SMTR-v1 | yes (`legacy/`, archived results) | NO |
| B. Online oracle prototype | yes (mechanism evidence / upper bound) | NO |
| C. RIMA canonical | yes (`results/rima/`) | YES |

## Directory Mapping (post-refactor)

```
experiments/rima/       # canonical runner
configs/rima/           # canonical configs (final.yaml)
results/rima/           # canonical results
docs/rima/              # canonical docs/audit
tests/rima/             # integrity test suite
legacy/                 # SMTR-v1, online_oracle_tci, offline_paired (preserved)
```

## Refactor Status Log

| Phase | Status | Evidence |
|---|---|---|
| 0 | DONE | tag `pre-rima-canonical-refactor` @ `2833b164` |
| 1-22 | DONE | `src/smtr/rima/`, `experiments/rima/`, critic freeze + sha256 |
| 23-26 | DONE | `src/smtr/rima/metrics.py` + `docs/rima/metrics.md`; runner emits `rima_metrics`/`cost_report` |
| 27 | DONE | fail-closed in outcome/critic; integrity invariant #10 |
| 28 | DONE | zero `epsilon_star`/`eta_cal`/`risk_gate` in formal path; legacy modules retained for ablation only |
| 29 | DONE | `configs/rima/`, `docs/rima/`, `results/rima/` unified |
| 30 | DONE | README rewritten for RIMA; old README → `docs/archive/smtr_v1_method.md` |
| 31 | DONE | `tests/rima/` 28 tests + sanitizer suite, all PASS (RIMA_CANONICAL_INTEGRITY = PASS) |
| 32-37 | PENDING | mechanism pilot → continual pilot → cross-domain → protocol freeze |
| 38-40 | PENDING | paper sync, RQ structure, final cleanup |

## Phase 32 Preflight Findings (2026-09-01)

Bargaining no_memory preflight (1 task, seed 0) debugging chain:

1. **Per-agent LLM hardcoding** — raw task JSONLs set per-agent `llm`
   (gpt-4o / gpt-3.5-turbo); MARBLE engine prefers per-agent value over
   global config, so every agent call hit `litellm.NotFoundError: Model
   not exist` on DashScope. Fixed in `_build_engine_config` by forcing all
   agents onto the configured model.
2. **Engine timeout** — default 900s killed the engine mid-run before
   `marble_output.jsonl` was written (fail-closed correctly recorded
   INVALID). Default raised to 1800s; `--engine-timeout` CLI flag added.
3. **Evaluator JSON parsing** — seller-side ratings parsed to -1 sentinels
   once (buyer 4/5/5 OK). Root cause hypothesis: `enable_thinking=true`
   default inflates evaluator responses toward the 4096-token cap.
   Mitigations: thinking OFF by default (override via
   `SMTR_LLM_ENABLE_THINKING`) + up-to-2-retry patch on
   `evaluate_task_world` keeping the best per-role rating.
4. **Credential blocker (resolved)** — the original key was a CN-account key
   that got rejected (401). The replacement key `sk-13c9…` belongs to an
   **international** account: it only authenticates against
   `dashscope-intl.aliyuncs.com/compatible-mode/v1`. `.env` now sets
   `DASHSCOPE_BASE_URL` accordingly; engine env mapping picks it up
   automatically. Verified: preflight 3/3 valid (100%), no evaluator
   retries needed once thinking mode was disabled.

## Phase 32 Pilot Pipeline (launched 2026-09-01)

Canonical chain (all formal invariants enforced):

1. **Stage A** `experiments/rima/collect_training_interventions.py`
   (new): no_memory source episodes → sanitized pool →
   `MatchedInterventionCollector(purpose=TRAINING_COLLECTION)` matched
   expose/withhold on later tasks; historical-only candidates;
   self-transfer excluded; incremental `intervention_records.json` +
   `candidates.json` + `source_agents.json`.
   Smoke: 2/2 pairs valid, self-transfer excluded. Full collection
   running: tasks 1-5, 3 candidates x 3 receivers (~4-5 h).
2. **Stage B-C** `train_critic.py --records results/rima/stage_a/...`
   → frozen `critic_receiver.joblib` + `critic_uniform.joblib` with
   task-level split audit.
3. **Phase 32 mechanism eval** `run_mechanism_eval.py --task-offset 5
   --limit-per-scenario 10` (held-out tasks 6-15, disjoint from
   training). Fixed: `agent_ids` injection for `select_receivers`,
   dict→`CandidateMemory` conversion, `"*"` wildcard candidate pool.
   GO conditions: valid ≥ 95%, tau sign accuracy > random, positive
   and negative observed deltas exist, receiver disagreement exists.

