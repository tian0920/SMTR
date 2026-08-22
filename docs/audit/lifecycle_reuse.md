# Lifecycle Reuse Audit

**Status:** PASS  
**Date:** 2026-08-22  
**Scope:** How retrieved memories are used in agent context/action

---

## 1. Objective

Verify that the reuse path (retrieved memory → agent prompt/context →
action) does not expose receiver-rejected memories to that receiver.

## 2. Reuse Paths

### 2.1 Receiver=3 Experiment (Offline Evaluation)

The Receiver=3 experiment does NOT inject memories into live agent
context at runtime. It evaluates offline using paired records:

```python
# experiments/marble_receiver3/pilot/run_pilot.py
class SMTRReceiverConditionedPolicy:
    def select_for_receiver(self, *, receiver_id, candidates,
                            receiver_outcomes, **kwargs):
        r_outcomes = receiver_outcomes.get(receiver_id, {})
        scored = []
        for c in candidates:
            mid = c["candidate_memory_id"]
            if mid in r_outcomes:
                exp, wh = r_outcomes[mid]
                delta = exp - wh
                if delta > 0:  # ← Only validated memories
                    scored.append((delta, mid))
        scored.sort(reverse=True)
        return [mid for _, mid in scored[:self._top_k]]
```

**Analysis:**
- Only memories with `delta > 0` for the specific receiver are selected.
- Receiver-rejected memories (delta ≤ 0) are excluded from selection.
- Selected memory IDs are used to compute team_reward, not injected
  into any agent prompt.

**Status: PASS** — Receiver-rejected memories cannot enter the
receiver's decision context.

### 2.2 Lifelong Experiment (Single-Agent)

```python
# experiments/lifelong/methods.py
def select_memories(self, task):
    validated = self.bank.retrieve_validated()
    return [self._meta[e.memory_id] for e in validated
            if topic_affinity(...)]
```

**Analysis:**
- Single-agent context — all memories go to the same agent.
- No multi-receiver concern.

**Status: N/A** — Single-agent pipeline.

### 2.3 MARBLE Baseline Adapter

```python
# src/smtr/marble/memory_adapter.py
selected = self.controller.retrieve_memory(query)
return candidates_to_memory_payloads(selected)
```

**Analysis:**
- Baseline controllers are single-agent.
- No receiver conditioning is applied.

**Status: N/A** — Baseline single-agent pipeline.

## 3. Receiver-Validated Priority

**Q: Are receiver-validated memories prioritized?**

In the Receiver=3 experiment, memory selection scores by delta value
(highest first). This means:
- Memories with larger positive Δ(m, r) are preferred.
- This provides a natural prioritization of high-value memories.

**Status: PASS** — Receiver-validated memories are prioritized by
causal utility (delta ranking).

## 4. Receiver-Rejected Memory Exposure

**Q: Can a receiver-rejected memory be used by that receiver?**

| Path | Can expose rejected memory? |
|------|---------------------------|
| `SMTRReceiverConditionedPolicy` | NO — only delta > 0 selected |
| `retrieve_validated()` (legacy, single-agent) | N/A — no multi-receiver |
| `get_receiver_validated_memories()` | NO — only validated entries |

**Status: PASS**

## 5. Conclusion

**Verdict: PASS**

1. The Receiver=3 reuse path correctly excludes receiver-rejected memories.
2. Memory selection prioritizes by causal utility (delta ranking).
3. No receiver-rejected memory can enter the rejecting receiver's context
   through any documented path.
