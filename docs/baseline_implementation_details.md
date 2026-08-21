# Baseline Implementation Details

> For each baseline, this document records the original paper idea,
> the SMTR implementation mapping, the memory decision mechanism,
> additional computation, and trainable parameters.
>
> **Purpose**: allow reviewers to verify that each baseline is a
> faithful adaptation of the published method within the constraints
> of a fair comparison framework.

---

## Summary Table

| Method | Original Principle | SMTR Adaptation | Key Difference from Original |
|--------|-------------------|-----------------|------------------------------|
| Full Memory | N/A (trivial baseline) | All experiences stored permanently; all injected | None — this is the standard "no selection" baseline |
| Retrieval | Standard top-k semantic retrieval | Topic-affinity scored retrieval; top-k=3 | Uses synthetic topic tags instead of embedding similarity |
| Reflexion | Verbal reflection from trajectory feedback → episodic memory buffer | Deterministic reflection text from success/failure + topic; all stored | **No LLM reflection generator** — replaced by deterministic text template. No multi-turn self-reflection loop |
| AGILE | Experience replay + RL policy optimisation | AGILE-inspired experience consolidation, **without RL parameter optimization**. Score = reward + 0.3·novelty + 0.2·consequence; budget eviction | **No gradient update, no policy training** — only the experience consolidation heuristic is preserved |
| Heuristic | Importance = 0.5·recency + 0.3·usage + 0.2·retrieval_success | Identical formula; FIFO-style eviction when over budget | No LLM-based importance estimation — uses observable statistics only |
| AgeMem | Learned RL controller for ADD / KEEP / DELETE / COMPRESS actions | AgeMem-inspired memory controller abstraction, **without learned policy training**. Frozen rule-based policy with deterministic thresholds | **No RL training, no learned policy** — only the action space and feature inputs are preserved as a rule-based approximation |
| SMTR-TCI | Causal memory validation via TCI (Treatment-on-the-Created-Information) | delta > 0 gate on expose vs withhold trials; double-negative rejection rule | None — this is the full SMTR method |

---

## Detailed Per-Baseline Disclosure

### 1. Full Memory

- **Paper**: N/A (standard baseline)
- **Memory type**: Verbatim experience
- **Selection principle**: All stored memories are injected into every future episode
- **Memory decision**: Store everything, never discard
- **Additional computation**: 0
- **Trainable parameters**: 0
- **TCI usage**: None

### 2. Retrieval

- **Paper**: Standard information retrieval baseline
- **Memory type**: Verbatim experience
- **Selection principle**: Store all, retrieve top-k=3 by topic affinity
- **Memory decision**: Store all; retrieve by `topic_affinity(memory_topic, task_topic) > 0`, sorted descending
- **Additional computation**: 0 (topic tag comparison only)
- **Trainable parameters**: 0
- **TCI usage**: None

### 3. Reflexion (NeurIPS 2023)

- **Original paper**: *Reflexion: Language Agents with Verbal Reinforcement Learning*
  - Official code: https://github.com/noahshinn/reflexion
  - Core loop: `trajectory → failure/success feedback → LLM reflection → verbal memory → future prompt`
- **SMTR adaptation**:
  - `extract_memory()`: deterministic reflection text encoding success/failure, topic, episode
  - `update_memory()`: all reflections stored unconditionally (matching original paper's episodic memory buffer)
  - `retrieve_memory()`: topic-filtered, most recent first
- **Differences from original**:
  - **No LLM reflection generator**: the original uses GPT-4 to generate free-form verbal reflections. Our implementation uses a deterministic template (`"Episode N: task on topic T succeeded/failed. Strategy: ... Continue/avoid this approach."`). This removes LLM cost and stochasticity while preserving the *memory format* (natural-language reflection) and the *unconditional storage* policy.
  - **No multi-turn self-reflection loop**: the original Reflexion allows multiple reflection rounds per episode. We use a single reflection per episode.
  - **No binary evaluator / sliding window**: the original uses a binary evaluator to decide when to reflect. We reflect after every episode.
- **Memory decision mechanism**: Unconditional store
- **Additional computation**: 0
- **Trainable parameters**: 0
- **TCI usage**: None

### 4. AGILE (NeurIPS 2024)

- **Original paper**: *AGILE: A Novel Reinforcement Learning Framework of LLM Agents*
  - Official code: https://github.com/bytarnish/AGILE
  - Core idea: `trajectory → experience buffer → RL optimisation → improved action policy`
- **SMTR adaptation**:
  - AGILE-inspired experience consolidation, **without RL parameter optimization**
  - `extract_memory()`: extracts (state, action, outcome, lesson) from trajectory
  - `update_memory()`: experience score = `reward + 0.3 × novelty + 0.2 × consequence`; evict lowest-scored when over budget
  - `retrieve_memory()`: topic-filtered, ranked by experience score
- **Differences from original**:
  - **No RL gradient update**: AGILE's core contribution is a reinforcement learning loop that updates the agent policy. We do not modify any LLM parameters or agent policy — only the *experience consolidation* mechanism is preserved.
  - **No experience replay buffer with RL sampling**: the original samples from the experience buffer for policy gradient updates. We use the buffer only for storage and retrieval.
  - **Score weights are fixed**: `0.3` for novelty and `0.2` for consequence are default values, not tuned.
- **Memory decision mechanism**: Score-based eviction (lowest experience_score evicted first)
- **Additional computation**: 0
- **Trainable parameters**: 0
- **TCI usage**: None

### 5. Heuristic Memory (ACL 2026)

- **Original paper**: *How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior*
  - Core idea: `memory → importance_score → keep / delete`
  - `importance = 0.5 × recency + 0.3 × usage_frequency + 0.2 × retrieval_success`
- **SMTR adaptation**:
  - `extract_memory()`: identical to Full Memory extraction
  - `update_memory()`: store all; evict lowest importance-scored when over budget
  - `retrieve_memory()`: importance-ranked, topic-filtered
- **Differences from original**:
  - Weights `0.5 / 0.3 / 0.2` are taken directly from the paper heuristic; not re-tuned.
  - Recency is normalised by latest episode; usage is capped at 5; retrieval success uses a 0.5 prior.
- **Memory decision mechanism**: Importance-scored eviction
- **Additional computation**: 0
- **Trainable parameters**: 0
- **TCI usage**: None

### 6. AgeMem (ACL 2026)

- **Original paper**: *Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents*
  - Core idea: agent learns to `write / retrieve / forget / compress` via an RL controller
- **SMTR adaptation**:
  - AgeMem-inspired memory controller abstraction, **without learned policy training**
  - `extract_memory()`: identical extraction
  - `update_memory()`: frozen rule-based policy decides ADD / DELETE / COMPRESS based on (age, usage, reward, topic_count)
  - `retrieve_memory()`: topic-filtered, most recent first
- **Differences from original**:
  - **No RL controller training**: the original trains a learned policy to decide memory actions. We use a frozen rule-based policy with fixed thresholds: `age_delete_threshold=50`, `usage_delete_threshold=0`, `reward_compress_threshold=0.3`.
  - **No TCI delta or future reward intervention**: only current observable features are used.
  - **Action space preserved**: ADD, KEEP, DELETE, COMPRESS are all implemented.
- **Memory decision mechanism**: Rule-based ADD/DELETE/COMPRESS with periodic stale sweep
- **Additional computation**: 0
- **Trainable parameters**: 0
- **TCI usage**: None

### 7. SMTR-TCI (Full Method)

- **Original paper**: This work
- **Memory type**: TCI-validated persistent knowledge
- **Selection principle**: Causal-utility gated (`delta = reward_expose − reward_withhold > 0 → validated`)
- **Memory decision mechanism**:
  - Every candidate undergoes TCI validation (expose vs withhold probe trials)
  - `delta > 0` → validated; else rejected
  - Re-validation of existing validated memories on same-topic episodes
  - Double-negative confirmation rule: 2 consecutive non-positive probes → reject
- **Additional computation**: `2 × VALIDATION_TRIALS = 6` probe trials per candidate + re-validation probes
- **Trainable parameters**: 0 (threshold-free: delta > 0)
- **TCI usage**: Yes (core mechanism)

---

## Fairness Summary

All baselines share:
- Same `LifelongEnvironment` (paired design, shared task RNG)
- Same seeds (0–4)
- Same memory budget (`capacity` parameter)
- Same evaluation model (`success_probability`)
- Zero additional LLM calls, training steps, or learnable parameters

The only method with additional computation is SMTR-TCI, which uses TCI validation probes.
This is the core method under evaluation, not an unfair advantage — the probes are the
*mechanism* being tested, not extra resources.
