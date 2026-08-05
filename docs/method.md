# SMTR Method

## 1. Problem: Cross-Agent Procedural Memory Exposure

In multi-agent systems, agents accumulate procedural memories from successful task execution. A natural question arises: should a memory produced by one agent (writer) be exposed to another agent (receiver)?

This is fundamentally different from single-agent memory selection because:
- The writer and receiver may have different roles, capabilities, and tools
- A procedure that helped the writer may harm the receiver (negative transfer)
- The decision must be receiver-specific, not memory-global

## 2. Why This Differs from Single-Agent Causal Memory Selection

Single-agent CMI (Causal Memory Intervention) asks: "Does this memory help *me*?"

Cross-agent exposure asks: "Does this memory, written by agent $w$, help agent $r$ in their current context?"

The key distinction is the **writer-receiver mismatch**: a memory's value depends on the interaction between who wrote it and who receives it.

## 3. Memory Representation: Payload + Routing Card

Each memory is split into:
- **Payload**: Full procedure steps, preconditions, postconditions (private)
- **Routing Card**: Goal summary, task tags, environment constraints, transfer hints, writer profile (public)

The router only sees the routing card. The payload is only revealed after a share decision.

## 4. Candidate Proposal: Card-Only Retrieval

For each receiver state, retrieve candidate memories using:
- Task similarity (receiver instruction vs card goal/tags)
- Environment compatibility
- Writer-receiver role/capability compatibility

Candidates include `match_type`: matched_writer_receiver, mismatched_writer_receiver, cross_task_same_group, cross_task_cross_group.

## 5. Action Space (v1)

The v1 action space is fixed to single-memory exposure:

$$A(o_r) \in \{\varnothing, m_1, \ldots, m_K\}$$

Each receiver is exposed to **at most one** candidate memory per episode. The selected-memory prefix $S$ is fixed to $S = \varnothing$; estimands are therefore written $\tau(m \mid o_r)$ and $\eta(m \mid o_r)$, never $\tau(m \mid o_r, S)$. The `selected_prefix_cards` interface exists for compatibility only and never influences features or decisions.

## 6. Candidate-Level Paired Intervention

For each candidate memory $m$ and receiver state:
- **Share branch**: Run MARBLE task with $m$'s payload injected
- **Withhold branch**: Run same task without $m$'s payload

Hold constant: task, seed, environment snapshot, receiver state, all other inputs.

This produces a four-outcome label:
- positive_transfer: share succeeds, withhold fails
- negative_transfer: share fails, withhold succeeds
- neutral_success: both succeed
- neutral_failure: both fail

### Core-valid pairs

A paired record enters critic training, calibration and evaluation only if it passes the single validity predicate `is_core_valid_pair`: same target task, receiver, generation seed, initial environment state and agent/tool configuration across branches; the only treatment difference is the candidate memory exposure; both branches carry MARBLE native team outcomes; the target receiver sees the memory in the share branch and not in the withhold branch; non-target agents never see the payload.

Invalid pairs are excluded, never relabelled as failure samples (an invalid environment, missing evaluator, or visibility failure produces no transfer label), and are reported via `invalid_pair_rate` with per-reason counts.

### Split isolation

**Target identity never crosses splits**: `task_id` (target task), `target_trajectory_id` (the receiver's execution trajectory under evaluation), treatment edges `(task_id, receiver_agent_id, candidate_memory_id)` and `edge_id` are each disjoint across train/validation/test. **Memory provenance may legitimately recur**: memories are extracted from train trajectories only (`memory_source_split == "train"`), so the same train-derived memory — and its `memory_source_trajectory_id` — may serve candidates in both validation and test; this is legal reuse, not leakage.

Critic training uses train edges; q01 calibration and epsilon selection use validation edges only (once, never re-applied); final evaluation uses test edges once. The split audit computes `target_task_overlap`, `target_trajectory_overlap`, `treatment_edge_overlap`, `non_train_memory_sources`, `self_transfer_edges` (target task == memory source task), `test_used_for_calibration` as fatal checks, reports legal provenance reuse (`shared_train_memory_provenance_count`, `memory_source_trajectory_reuse`) as statistics, and derives `split_integrity_passed` from those results — never assumed.

The audit artifact (`schema_version: smtr_split_audit_v2`) binds every audited file by SHA-256 digest: dataset manifest, split manifest, memory pool, the three per-split paired-record files and the critic checkpoint. A formal end-to-end evaluation must bind such an artifact and re-verifies, before any episode runs, that the audit passed, that calibration and ε selection used only the validation split, and that every digest still matches the file actually consumed.

## 7. Four-Outcome Transfer Critic

Train an ensemble classifier predicting:
- $q_{00} = P(\text{neutral\_failure})$
- $q_{01} = P(\text{negative\_transfer})$
- $q_{10} = P(\text{positive\_transfer})$
- $q_{11} = P(\text{neutral\_success})$

Feature blocks: task context, receiver marginal, writer marginal, writer-receiver interaction, memory card. There is no prefix block: $S = \varnothing$ in v1.

## 8. Receiver-Specific Exposure Router

Decision rule:
$$\hat{\tau} = q_{10} - q_{01}, \quad \hat{\eta} = q_{01}$$

Share candidate if $\hat{\tau} > 0$ and $\hat{\eta}_{\text{calibrated}} \leq$ budget.
Share at most one memory per receiver (the safe candidate with highest $\hat{\tau}$).

**Eta trace schema**: each `RouterDecision` stores both `eta_raw` ($q_{01}$ from the critic) and `eta_calibrated` (q01 after the single validation-split calibration). The router gate compares `eta_calibrated` against the risk budget; `eta_raw` is retained for diagnostics only. Calibration is applied **exactly once**, at routing time. Downstream consumers — including the risk–utility curve — read the calibrated value from the trace and MUST NOT invoke the calibrator again; in formal mode a trace row missing `eta_calibrated` is an error, not a silent re-calibration.

This is a **receiver-specific exposure mask**, not a global memory filter.

## 9. MARBLE Evaluation

**Generation seed protocol** (enforced inside the function API, not only the CLI): a formal run requires at least 5 unique generation seeds, a pilot at least 3; duplicates are deduplicated and the unique count is validated before any critic load or episode. The result records `generation_seeds`, `unique_seed_count`, `minimum_required_seed_count` and `seed_protocol_passed`.

**Split-audit binding**: a formal end-to-end evaluation requires a verified split-audit artifact (see §6). Before any critic load or MARBLE episode, the evaluation validates the artifact's schema version, integrity verdict, calibration/ε-selection splits and all file digests against the files it will actually consume, and aborts on any mismatch. The result records `split_audit_verified`, `split_audit_path`, `split_audit_digest` and `split_integrity_passed`.

Metrics:
- Team success rate
- Positive/negative transfer rate
- Harmful exposure rejection rate
- Writer-receiver mismatch share rate
- Same-memory different-receiver decision count
- Receiver-specific quarantine pair count

Outcome scope (v1): the only reliable supervision is the MARBLE team-level
outcome, so the formal claim is:

> SMTR controls cross-agent memory exposure using team-level transfer outcomes.

No receiver-local evaluator exists in v1; local metrics are reported as
`null` (never 0) and no local–team divergence claim is made. Local outcomes
are a future extension only.
