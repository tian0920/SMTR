# Receiver=3 Experiment Protocol

> Frozen protocol document for all MARBLE receiver=3 experiments.
> All subsequent experiments MUST read from `configs/marble_receiver3_main.yaml`.

---

## Memory Flow Architecture

```
Source Agent (agent1)
    │
    ├─ produces candidate memory m
    │
    ▼
TCI Validation Gate (receiver-conditioned)
    │
    ├─ Δ(m, receiver_1) > 0 ?  → retain for r1
    ├─ Δ(m, receiver_2) > 0 ?  → retain for r2
    ├─ Δ(m, receiver_3) > 0 ?  → retain for r3
    │
    ▼
Shared Persistent Memory Bank
    │
    ├─ m_validated_for = {receiver_1, receiver_3}
    │
    ▼
Receiver Agents
    ├─ receiver_1: receives m (validated)
    ├─ receiver_2: does NOT receive m (rejected for r2)
    └─ receiver_3: receives m (validated)
```

---

## Key Design Principles

### 1. Source → Receiver Flow Only
- Memory flows from source agent TO receivers
- Receivers do NOT share memories with each other
- Source agent does NOT receive its own memories back

### 2. Receiver-Conditioned TCI
- TCI decision is per-receiver: Δ(m, r) = expose(m, r) - withhold(m, r)
- Same memory can be validated for receiver A but rejected for receiver B
- No shared threshold — pure sign-of-delta rule

### 3. Shared Persistent Memory Bank
- All validated memories stored in one bank
- Each entry tagged with `receiver_id` and `validation_target`
- `validation_history` records per-receiver expose/withhold outcomes

### 4. Fair Baseline Comparison
All methods share:
- Same task sequence
- Same seeds
- Same memory budget
- Same paired evaluation

---

## Experiment Matrix

| Phase | Tasks | Seeds | Methods | Receivers |
|-------|-------|-------|---------|-----------|
| Pilot | 5 | [0] | 4 | 3 |
| Main | 50 | [0,1,2,3,4] | 7 | 3 |
| Contamination | 50 | [0,1,2,3,4] | 3 | 3 |

---

## Methods

| Method | Description | Receiver-conditioned? |
|--------|-------------|----------------------|
| no_memory | No memory injection | No |
| full_memory | Inject all available memories | No (shared to all) |
| retrieval | Top-k by score | No (same retrieval) |
| reflexion | Store reflections, retrieve by recency | No |
| heuristic | Score × rank importance | No |
| agemem | Diversity-aware selection | No |
| **smtr_tci** | TCI-validated, per-receiver | **Yes** |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| official_score | MARBLE native evaluator |
| late_stage_reward | Last 20% episodes |
| memory_quality | MQS (useful × transfer / harmful) |
| receiver_transfer | Per-receiver τ |
| receiver_disagreement | P(decision_i ≠ decision_j) |
| contamination_propagation | Harmful retention per receiver |

---

## Commit Plan

1. **Commit 1**: Protocol + audit (Phase 0-1)
2. **Commit 2**: Receiver-conditioned TCI implementation (Phase 2-3)
3. **Commit 3**: Pilot experiment + report (Phase 4)
4. **Commit 4**: Full MARBLE main (Phase 5)
5. **Commit 5**: Analysis + tables + reports (Phase 6-10)
