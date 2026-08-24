# MARBLE Benchmark Identity Audit

**Date**: 2026-08-24

**Status**: ✅ VERIFIED


## 1. Current Benchmark Identity

### Upstream Repository

- **URL**: https://github.com/ulab-uiuc/MARBLE
- **Local path**: `/home/ecs-user/MARBLE`
- **Git remote**: `origin https://github.com/ulab-uiuc/MARBLE.git`
- **Current commit**: `2b1ca0d` (main branch)


### Paper & Venue

- **Title**: MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents
- **Authors**: Kunlun Zhu et al. (University of Illinois Urbana-Champaign)
- **Venue**: ACL 2025 (Main Conference)
- **ArXiv**: https://arxiv.org/abs/2503.01935
- **GitHub Org**: https://github.com/MultiagentBench (forked from ulab-uiuc/MARBLE)


### Full Name

**MARBLE** = **M**ulti-**A**gent Coo**R**dination **B**ackbone with **L**LM **E**ngine

This is the **code framework** for the MultiAgentBench benchmark.


## 2. Supported Scenarios

| Scenario | Environment Class | JSONL File | Task Count |
|----------|------------------|------------|------------|
| **bargaining** | `world_env.py` (World Simulation) | `multiagentbench/bargaining/bargaining_main.jsonl` | 100 |
| **coding** | `coding_env.py` (Coding) | `multiagentbench/coding/coding_main.jsonl` | 100 |
| **database** | `db_env.py` (DB) | `multiagentbench/database/database_main.jsonl` | 100 |
| **minecraft** | `minecraft_env.py` (Minecraft) | `multiagentbench/minecraft/minecraft_main.jsonl` | 100 |
| **research** | `research_env.py` (Research) | `multiagentbench/research/research_main.jsonl` | 100 |

**Total**: 5 scenarios × 100 tasks = **500 official tasks**


## 3. Official Evaluation Metrics

| Scenario | Official Evaluator | Metric Type | Range |
|----------|-------------------|-------------|-------|
| **database** | `Evaluator.evaluate_task_db()` | Root cause recall (subset match) | Binary (0/1) |
| **research** | `Evaluator.evaluate_task_research()` | LLM-judged {innovation, safety, feasibility} | Ordinal 1-5 |
| **minecraft** | Rule-based `block_hit_rate` | Block placement accuracy | Continuous [0, 1] |
| **coding** | `Evaluator.evaluate_code_quality()` | LLM-judged {instruction_following, executability, consistency, quality} | Ordinal 1-5 |
| **bargaining** | `Evaluator.evaluate_task_world()` | LLM-judged {effectiveness, progress, interaction} for buyer/seller | Ordinal 1-5 |

**Source**: `/home/ecs-user/MARBLE/marble/evaluator/evaluator.py` (lines 170-619)


## 4. Official Task Split

The MultiAgentBench paper uses **all 100 tasks per scenario** (500 total) for evaluation.

- **SMTR current usage**: 20 tasks per scenario (100 total) = **20% of official pool**
- **SMTR task selection**: First 20 tasks from each `{scenario}_main.jsonl`
- **Official protocol**: Use full task pool, multiple seeds


## 5. Identity Verification

### Question: Which MARBLE is this?

**Answer**: ✅ **A. ulab-uiuc/MARBLE**

| Check | Evidence |
|-------|----------|
| Repository URL | `https://github.com/ulab-uiuc/MARBLE.git` |
| README title | "MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents" |
| Author | Kunlun Zhu (UIUC) |
| Paper venue | ACL 2025 |
| Scenarios | bargaining, coding, database, minecraft, research |
| Task structure | `multiagentbench/{scenario}/{scenario}_main.jsonl` |

### Question: Is this Samujjalborah/MARBLE (Bioinformatics)?

**Answer**: ❌ **NO**

| Check | Evidence |
|-------|----------|
| No bioinformatics code | `grep -r "bioinformatics" /home/ecs-user/MARBLE` → 0 matches |
| No genomics/proteomics | No related environments or tasks |
| Different author | Samujjalborah ≠ Kunlun Zhu |
| Different venue | Bioinformatics paper ≠ ACL 2025 |

**Conclusion**: No confusion risk. Current codebase is unambiguously **ulab-uiuc/MARBLE** (MultiAgentBench).


## 6. SMTR Code References Audit

### Correct References (ulab-uiuc/MARBLE)

- ✅ `src/smtr/marble/task_loader.py` line 3: "Loads tasks from `{marble_root}/multiagentbench/{scenario}/{scenario}_main.jsonl`"
- ✅ All evaluator paths point to `/home/ecs-user/MARBLE/marble/evaluator/`
- ✅ All environment paths point to `/home/ecs-user/MARBLE/marble/environments/`

### Incorrect References (Samujjalborah/MARBLE)

- ✅ **NONE FOUND** — No references to bioinformatics, genomics, or Samujjalborah

**Verdict**: ✅ All SMTR code correctly references ulab-uiuc/MARBLE (MultiAgentBench)


## 7. Summary

```
CURRENT_BENCHMARK = A

A = ulab-uiuc/MARBLE (MultiAgentBench)
    Multi-Agent CooRdination Backbone with LLM Engine
    ACL 2025 Main Conference
    5 scenarios × 100 tasks = 500 total

B = Samujjalborah/MARBLE (Bioinformatics)
    NOT USED — zero references in SMTR codebase
```


## 8. Recommendations

1. **No action required** — Benchmark identity is correct
2. **Documentation**: Add explicit citation to MultiAgentBench paper in SMTR README
3. **Task coverage**: Consider using full 100-task pool per scenario (currently 20%)
