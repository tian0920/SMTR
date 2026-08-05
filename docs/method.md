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

Share candidate if $\hat{\tau} > 0$ and $\hat{\eta} \leq$ budget.
Share at most one memory per receiver (the safe candidate with highest $\hat{\tau}$).

This is a **receiver-specific exposure mask**, not a global memory filter.

## 9. MARBLE Evaluation

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
