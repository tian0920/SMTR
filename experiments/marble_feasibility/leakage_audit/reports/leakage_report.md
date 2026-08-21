# Metadata Leakage Audit Report (v2)

**Verdict: PASS**

## Key Insight

Encoder ablation proved task_id is a **causal driver** (task_only=0.88, metadata_no_task=0.41). This audit isolates whether the remaining metadata fields (rank, score, source, memory_id) constitute non-causal shortcuts.

## Results

| Model | Ranking | Notes |
|-------|---------|-------|
| Metadata-only (no task_id) | 0.4134 | rank+score+source+mem_id only |
| Shuffled ALL metadata | 0.6118 | including task_id shuffled |
| Full SMTR | 0.8273 | all features |
| Without candidate_score | 0.8273 | full minus score |
| Task-only (reference) | 0.8840 | task_id only |
| Full metadata (with task) | 0.8273 | task+rank+score+source+mem_id |
| SMTR reference (ablation) | 0.8433 | from encoder ablation |

## Acceptance Criteria

### [PASS] metadata_only (no task_id) < SMTR - 0.10
- meta_only_no_task=0.4134, SMTR=0.8433, threshold=0.7433

### [PASS] Shuffle ALL metadata drop >= 0.20
- full_meta=0.8273, shuffled=0.6118, drop=0.2155

### [PASS] Remove candidate_score drop < 0.10
- full=0.8273, no_score=0.8273, drop=0.0000

## Interpretation

- **task_id is causal**: True (task_only=0.8840)
- **metadata without task is weak**: True (meta_no_task=0.4134)
- **shuffle destroys signal**: True (drop=0.2155)
- **candidate_score not critical**: True (drop=0.0000)

---

**Conclusion**: No significant metadata leakage detected. Performance gains come from causal task context, not identity shortcuts.