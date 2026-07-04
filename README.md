# SMTR

Shared Memory Transfer Router is a deterministic research prototype for studying
when procedural memories should be exposed to agents in a multi-agent runtime.

## Shared Memory Pool

The memory pool has two physical representations:

- `ProcedurePayload`: the full procedure, including sensitive ordered `steps`.
- `MemoryRoutingCard`: router-visible metadata used for candidate retrieval and
  future share/withhold decisions.

Candidate proposal and router traces only receive routing cards. Payloads are
loaded through `get_selected_payloads()` after the router has produced final
`selected_memory_ids`. With the current `NoMemoryRouter`, every candidate is
withheld, so agents receive no payload steps.

## Candidate Proposal

`DeterministicHybridCandidateProposer` performs high-recall, non-causal
retrieval over routing cards only. The score is:

```text
score = 0.45 * goal_similarity
      + 0.15 * task_tag_overlap
      + 0.25 * environment_compatibility
      + 0.15 * receiver_compatibility
```

Goal similarity is deterministic token Jaccard. Environment compatibility checks
required and forbidden facts; explicit conflicts are penalized but retained so
the candidate set stays high-recall. Receiver compatibility uses role match
first, then capability overlap.

## Router

`NoMemoryRouter` is the safe baseline. The learned online algorithm is
implemented by `ProductionSequentialRouter`:

```text
candidate proposal
-> random traversal of the proposed candidates
-> sequential set-conditioned gating
-> payload-only exposure after selection is complete
```

Candidate proposal is a replaceable high-recall module; its rank is retained as
proposal metadata, but the sequential gate uses a reproducible random traversal
order and does not learn or optimize memory ordering. At each step, the router
predicts the candidate's conditional effect from the task/agent/environment
context, the candidate routing card, and the cards already accepted into `S`.
The default accept rule is strict:

```text
share iff LCB(tau_hat) > 0 and UCB(eta_hat) <= epsilon
```

The online router predicts these quantities with the transfer critic. It does
not run a real share/withhold counterfactual experiment for every online
candidate.

## SQLite Store

The local store uses SQLite tables for payload versions, routing cards, raw
execution evidence, and store metadata. It supports immutable read snapshots
containing routing cards and active versions, but no payload steps.

```bash
pip install -e .
python -m smtr.cli seed-memories --db data/smtr_memory.sqlite
python -m smtr.cli list-memories --db data/smtr_memory.sqlite
python -m smtr.cli demo --seed 7 --db data/smtr_memory.sqlite --top-k 4
```

`list-memories` prints routing metadata and execution alpha/beta counts. It does
not print payload steps.

## Paired Counterfactual Rollout

The paired collector is an offline training-data tool. It is not called by the
production `demo` path. For one captured pre-route decision point, it resumes two
branches from the same graph state, environment snapshot, memory snapshot,
candidate order, selected-memory prefix, and seed:

- share branch: expose the target candidate memory payload.
- withhold branch: withhold that same target memory payload.

The only intended branch difference is target memory exposure. A
`ReadOnlyPinnedMemoryView` fixes active payload versions from the captured
snapshot and rejects writes, so branch execution cannot update execution
evidence, routing cards, payload versions, or store revision.

The current continuation policy is frozen no-share. Candidate order is
randomized for training coverage and matched to online traversal semantics, not
optimized. The collected JSONL records contain four-outcome transfer labels and
do not serialize procedure payload steps.

```bash
python -m smtr.cli collect-counterfactual \
  --db data/smtr_memory.sqlite \
  --episodes 40 \
  --seed 7 \
  --top-k 4 \
  --scenario-mix balanced \
  --target-policy scenario-designated \
  --prefix-mode stratified \
  --max-prefix-size 2 \
  --output data/paired_records.jsonl

python -m smtr.cli inspect-paired-records \
  --input data/paired_records.jsonl \
  --show-prefix-distribution
```

`selected_before` is the already accepted memory prefix `S_{t-1}` for the
receiving agent. It is part of both paired branches: the share branch receives
`S ∪ {m}` and the withhold branch receives `S`. Prefix sampling is randomized
training-time exploration, not memory-order optimization. Candidate retrieval
order remains the proposer relevance order; traversal order is a separate
random nuisance variable used for paired data coverage.

Version 1.1 paired records freeze router-visible card metadata in
`RoutingFeatureSnapshot` objects for the target memory and selected prefix
memories. Training never reconstructs features from live cards, because those
cards may later drift as evidence is ingested.

Paired evidence ingestion is explicit and separate from raw procedure execution
evidence:

```bash
python -m smtr.cli ingest-paired-evidence \
  --db data/smtr_memory.sqlite \
  --input data/paired_records.jsonl
```

The four-outcome critic estimates `q00`, `q01`, `q10`, and `q11`, where `q01` is
negative-transfer probability and `q10 - q01` is the conditional marginal
effect. It reports bootstrap intervals for `tau`, an upper confidence bound for
negative risk, and a nearest-neighbor support diagnostic.

```bash
python -m smtr.cli train-transfer-critic \
  --input data/paired_records.jsonl \
  --output checkpoints/transfer_critic.joblib \
  --seed 7 \
  --n-bootstrap 31 \
  --test-fraction 0.2

python -m smtr.cli evaluate-transfer-critic \
  --input data/paired_records.jsonl \
  --checkpoint checkpoints/transfer_critic.joblib \
  --output outputs/transfer_critic_eval.json
```

## Policy-Aware Iteration

Transfer effects are tied to the frozen continuation policy used after the
target candidate. Mixing records from different continuation policies changes
the estimand. The offline loop is:

```text
pi0 no-share -> D0 -> C0
C0 -> pi1 critic-guided continuation policy
pi1 -> D1 -> C1
```

`C0` is used only to construct `pi1`; it is not an estimator of effects under
`pi1`. `C1` is trained only from records collected under `pi1`.

Policy manifests fingerprint the frozen continuation policy and checkpoint
dependencies. Round manifests bind records to one base memory snapshot and store
revision, blocking temporal leakage from future evidence. Future agent candidate
proposals may differ between share and withhold branches after the target
intervention; that is a legitimate downstream causal effect. The continuation
policy, checkpoint, pinned snapshot, and seed derivation remain fixed.

```bash
python -m smtr.cli create-no-share-policy --output policies/pi0_no_share.json
python -m smtr.cli build-critic-continuation-policy \
  --critic checkpoints/critic_pi0.joblib \
  --tau-lcb-threshold 0.0 \
  --negative-risk-ucb-threshold 0.20 \
  --reject-low-support \
  --output policies/pi1_critic_sequential.json
python -m smtr.cli evaluate-transfer-critic \
  --input data/paired_records_pi1_v12.jsonl \
  --checkpoint checkpoints/critic_pi1.joblib \
  --split-suite strict \
  --output outputs/critic_pi1_strict_eval.json
```

Strict evaluation uses group holdouts such as scenario family, environment
regime, target memory family, and prefix structure family. Shortcut diagnostics
report when a grouping nearly determines the transfer class. Learned policies
remain offline-only in this stage and are not loaded by the default demo.

## Risk-Constrained Exploration

The next offline round can build a frozen risk-constrained exploratory
continuation policy from an existing critic. The policy manifest records the
source critic checksum, source estimand policy fingerprint, candidate traversal
version, proposer version, feature encoder schema, and exploration thresholds.
Collection records use schema v1.3 and include continuation behavior metadata
for downstream router decisions: decision mode, share probability, exploration
eligibility/selection, support distance, support threshold, and candidate
scores.

```bash
python -m smtr.cli build-exploratory-continuation-policy \
  --critic checkpoints/critic_pi1.joblib \
  --safe-negative-risk-ucb-threshold 0.20 \
  --hard-negative-risk-veto-ucb 0.35 \
  --exploration-round-probability 0.30 \
  --output policies/pi2_explore.json
python -m smtr.cli collect-counterfactual \
  --db data/smtr_memory.sqlite \
  --episodes 480 \
  --seed 29 \
  --top-k 6 \
  --scenario-design factorial \
  --factorial-balance stratified \
  --target-policy uniform \
  --prefix-mode stratified \
  --max-prefix-size 2 \
  --continuation-policy-manifest policies/pi2_explore.json \
  --round-id pi2 \
  --round-index 2 \
  --require-min-continuation-share-rate 0.10 \
  --require-max-continuation-share-rate 0.35 \
  --require-max-hard-risk-share-rate 0.01 \
  --output data/paired_records_pi2_v13.jsonl
```

The factorial toy environment attaches evaluation metadata such as environment
regime, factor-combination id, surface variant id, mechanism group id, target
memory family, and prefix structure family. These fields are for splitting and
auditing only; the transfer feature leakage scanner verifies that evaluation
labels, branch outcomes, payloads, steps, and split identifiers are absent from
critic features.

```bash
python -m smtr.cli validate-collection-quality \
  --input data/paired_records_pi2_v13.jsonl \
  --min-continuation-share-rate 0.10 \
  --max-continuation-share-rate 0.35 \
  --max-hard-risk-share-rate 0.01 \
  --require-factorial-diversity \
  --output outputs/pi2_collection_quality.json
python -m smtr.cli scan-transfer-feature-leakage \
  --input data/paired_records_pi2_v13.jsonl \
  --output outputs/feature_leakage_scan_pi2.json
python -m smtr.cli audit-feature-blocks \
  --input data/paired_records_pi2_v13.jsonl \
  --split-suite compositional \
  --output outputs/feature_block_audit_pi2.json
python -m smtr.cli evaluate-transfer-critic \
  --input data/paired_records_pi2_v13.jsonl \
  --checkpoint checkpoints/critic_pi2.joblib \
  --split-suite compositional \
  --output outputs/critic_pi2_compositional_eval.json
```

Risk-constrained exploratory policies are still data-collection policies, not
production routers. The default `demo` path continues to use `NoMemoryRouter`.

## Current Research Boundary

This stage implements structured procedural memories, versioned storage,
candidate retrieval, payload isolation, deterministic procedure writing, raw
procedure execution evidence, offline paired counterfactual labels,
policy-specific critics, and offline risk-constrained exploration diagnostics.
Procedure execution success is not a memory transfer effect. Raw success/failure
contexts are not positive/negative transfer contexts.

Future stages will deploy a sequential causal share/withhold router, refine
procedures, compose meta-procedures, and connect real multi-agent runtimes.
