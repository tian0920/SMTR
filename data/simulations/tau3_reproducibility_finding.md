# τ³-Bench Reproducibility Finding

**Date**: 2026-07-04  
**Model**: qwen3.5-plus (via OpenAI-compatible API)  
**Domain**: mock  
**Seed**: 42

## Experiment

Ran the same mock domain task twice with identical configuration:
- Same agent-llm, user-llm, seed=42
- Compared message-by-message trajectories

## Results

| Metric | Run A | Run B |
|--------|-------|-------|
| Reward | 1.0 | 1.0 |
| Message count | 8 | 8 |
| Tool calls | get_users, create_task | get_users, create_task |
| Tool arguments | identical | identical |

## Differences Found

1. **Timestamps**: Different wall-clock times (expected)
2. **User simulator content**: Different wording for same task
   - Run A: "I need help creating a task for an upcoming meeting..."
   - Run B: "I need help creating a task for an upcoming meeting..." (slightly different phrasing)
3. **Agent responses**: Different wording but same actions
4. **Tool call IDs**: Different random UUIDs
5. **Token usage**: Different completion tokens (257 vs 156 for user simulator)

## Conclusion

**τ³-bench with Qwen model is stochastic at the message level, but functionally consistent.**

The user simulator does NOT produce identical trajectories even with the same seed. This is likely due to:
- API-based LLMs having non-deterministic sampling
- Temperature > 0 in user simulator

## Implication for Paired Rollout

For reliable causal estimates τ(m|o,S), we should use one of:

1. **Multi-trial estimation**: Run each branch multiple times, estimate E[Y^(1) - Y^(0)]
2. **User trajectory replay**: Record user messages from Branch A, replay them in Branch B
3. **Outcome-level comparison**: Compare binary success/failure rather than message-level trajectories

**Decision**: Use **outcome-level comparison** (Task 3.9 design) since τ³-bench provides official reward evaluation. Multi-trial estimation can be added later for variance reduction.
