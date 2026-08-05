# Ablation Plan (清单 P0-2 / 十一)

## Formal Main-Table Methods

| ID | Method | Definition |
|----|--------|------------|
| B0 | NoMemory | Never expose any memory. |
| B1 | SemanticTop1 | Top-1 by task-memory semantic similarity only; ignores writer, receiver, role, capability, environment and the transfer critic; always exposes top-1. |
| B2 | RoleAwareTop1 | Top-1 by task relevance + receiver role/capability compatibility; no paired intervention labels, no critic; always exposes top-1. |
| B3 | GlobalTransferCritic | Paired transfer supervision, but without writer identity, receiver identity, writer/receiver roles and writer-receiver pair/interaction features; learns only global memory/context transfer tendency. |
| B4 | SMTR-no-pair-interaction | Writer and receiver marginal features kept; writer_role→receiver_role pair tokens, compatibility and interaction features removed. |
| B5 | SMTR-no-risk | Full SMTR critic; share iff `tau_hat > 0`; ignores `eta_hat` and `epsilon_star`. |
| — | SMTR | Full writer-receiver-conditioned critic; share iff `tau_hat > 0` and calibrated `eta_hat <= epsilon_star`. |

Removed from the formal table:

- **AllShare** — in the v1 single-memory action space it is behaviorally
  identical to a top-1 heuristic baseline.
- **FactualSuccess** — no reliable memory-level historical aggregates exist.

## Distinctness Requirement

Any two main-table methods must be able to produce different actions; no
two methods may be behaviorally identical on all inputs. Each ablation
removes exactly one method component, and method names match their feature
block / router rule.

## Ablation Goals

### B0 vs SMTR
Does selective sharing beat no sharing?

### B1 vs B2 vs SMTR
Does receiver-aware and label-trained routing beat semantic similarity and
role heuristics?

### B3 vs SMTR
Do writer/receiver identity and pair features matter beyond a global
transfer tendency?

### B4 vs SMTR
Do writer-receiver interaction features matter (marginals kept)?

### B5 vs SMTR
Does the calibrated risk gate (`eta_hat <= epsilon_star`) matter?

## Feature Block Ablation

- `full`: all blocks (SMTR);
- `no_pair_interaction`: writer/receiver marginals, no interaction (B4);
- `memory_task_only`: task/env/memory card only (B3);
- `no_receiver`: no receiver identity/profile or interaction.
