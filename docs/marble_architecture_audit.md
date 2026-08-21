# MARBLE Architecture Audit

> Analysis-only audit of the SMTR↔MARBLE integration layer.
> **No code was modified.**

---

## 1. MARBLE Task Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MARBLE Task Flow                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐   │
│  │  Task Provider│───>│ Environment     │───>│ Agent Execution  │   │
│  │  (manifest)   │    │ (Docker DB)     │    │ (MARBLE Engine)  │   │
│  └──────────────┘    └─────────────────┘    └──────────────────┘   │
│         │                      │                      │            │
│         │                      │                      ▼            │
│         │                      │              ┌──────────────────┐  │
│         │                      │              │  Trajectory       │  │
│         │                      │              │  (messages,       │  │
│         │                      │              │   tool calls)     │  │
│         │                      │              └────────┬─────────┘  │
│         │                      │                       │            │
│         │                      │                       ▼            │
│         │                      │              ┌──────────────────┐  │
│         │                      │              │ Memory Extraction │  │
│         │                      │              │ (procedure        │  │
│         │                      │              │  payload)         │  │
│         │                      │              └────────┬─────────┘  │
│         │                      │                       │            │
│         │                      │                       ▼            │
│         │                      │              ┌──────────────────┐  │
│         │                      │              │ Memory Injection  │  │
│         │                      │              │ (MarbleMemory     │  │
│         │                      │              │  Injector)        │  │
│         │                      │              └────────┬─────────┘  │
│         │                      │                       │            │
│         │                      │                       ▼            │
│         │                      │              ┌──────────────────┐  │
│         │                      │              │ Receiver Agent    │  │
│         │                      │              │ Execution         │  │
│         │                      │              └────────┬─────────┘  │
│         │                      │                       │            │
│         │                      │                       ▼            │
│         │                      │              ┌──────────────────┐  │
│         │                      │              │ Outcome Evaluation│  │
│         │                      │              │ (root_cause match)│  │
│         │                      │              └──────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Detailed Steps

1. **Task Loading**: `MarbleTaskProvider` reads `dataset.json` manifest (500 database
   tasks) and `splits.json` (train/validation/test partitions).

2. **Environment Setup**: `MarbleDatabaseEnvironment` materialises an
   `InitialStateBundle` into a Docker workspace (`init.sql`, `task.json`,
   `marble_config.yaml`).

3. **Agent Execution**: `run_marble_engine_process()` invokes the real
   MARBLE Engine as a subprocess with LLM routing (qwen3-30b-a3b via
   DASHSCOPE).

4. **Trajectory**: The engine produces `marble_output.jsonl` containing
   agent messages, tool calls, and database interactions.

5. **Memory Extraction**: Writer agent trajectories → procedural memory
   payloads (`procedure`, `preconditions`, `postconditions`, `provenance`).

6. **Memory Injection**: `MarbleMemoryInjector.build_injection()` creates
   memory payloads injected into receiver agents' `private_memory_payloads`.

7. **Paired Evaluation**: Share vs withhold branches run on the same
   (task, receiver, seed) — outcome determined by root-cause prediction
   matching ground truth.

---

## 2. Current Support

| Dimension | Status |
|-----------|--------|
| **Agent count** | 1–5 agents per task (MARBLE native: agent1..agent5) |
| **Receiver count** | 1 (single-receiver pilot) or 3 (multi-receiver config) |
| **Role definitions** | `executor` (read_write memory access) |
| **Communication protocol** | Memory payload injection via `private_memory_payloads` in agent input YAML |
| **Episode length** | Per-task: one MARBLE engine invocation = one task execution |
| **LLM backbone** | qwen3-30b-a3b (thinking mode OFF) via DASHSCOPE API |
| **Scenario** | `database` (PostgreSQL diagnostic tasks) |
| **Task count** | 500 tasks in frozen dataset manifest |
| **Splits** | train / validation / test (500 records) |
| **Docker isolation** | Per-branch workspace with `init.sql` + `task.json` |

---

## 3. Memory Controller Insertion Points

### 3.1 Where Memory Decisions Happen

```
Writer Agent ──trajectory──> Memory Extraction ──candidate──> ???
                                                               │
                          ┌─────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  Memory Controller  │  <-- INSERTION POINT
              │  (decision gate)    │
              └─────────┬───────────┘
                        │
                   validated?
                   /         \
                 yes          no
                  │            │
                  ▼            ▼
          Memory Pool    Discarded
                  │
                  ▼
          ┌───────────────┐
          │  Retrieval     │  <-- INSERTION POINT
          │  (top-k,       │
          │   topic-match) │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │  Injection     │
          │  (payload into │
          │   receiver)    │
          └───────────────┘
```

### 3.2 Current Memory Pipeline

| Stage | Component | File |
|-------|-----------|------|
| Extraction | `pilot_candidates` / trajectory → `procedure payload` | `src/smtr/marble/pilot_candidates.py` |
| Routing Card | `build_routing_card_from_pool_entry()` | `src/smtr/router/transfer_features.py` |
| Critic | `FourOutcomeTransferCritic` | `src/smtr/router/transfer_critic.py` |
| Router | `SMTRExposureRouter` / `NoMemoryRouter` / `SemanticTop1Router` | `src/smtr/router/` |
| Injection | `MarbleMemoryInjector.build_injection()` | `src/smtr/marble/memory_injection.py` |
| Evaluation | `DatabaseOutcomeEvaluator` | `src/smtr/marble/outcome/scenarios/database.py` |

### 3.3 Baseline Controller Insertion Point

The `BaseMemoryController` interface (`extract_memory`, `update_memory`,
`retrieve_memory`) maps to the MARBLE pipeline as follows:

| Baseline Method | MARBLE Mapping |
|-----------------|----------------|
| `extract_memory(trajectory)` | Called after writer agent execution, before routing |
| `update_memory(candidate, context)` | Replaces the TCI critic's validate/reject decision |
| `retrieve_memory(query)` | Replaces the router's candidate scoring + selection |
| `get_statistics()` | Logging only, no pipeline effect |

The adapter layer (`src/smtr/marble/memory_adapter.py`, Phase 2) bridges:
- MARBLE trajectory dict → `BaseMemoryController.extract_memory()` input
- `BaseMemoryController.retrieve_memory()` output → `MarbleMemoryInjector` payloads

---

## 4. Differences from Lifelong Experiment Interface

| Aspect | Synthetic Lifelong | MARBLE Real |
|--------|-------------------|-------------|
| **Environment** | numpy-based `LifelongEnvironment` (success probability model) | Real MARBLE Engine subprocess (Docker + PostgreSQL) |
| **Episodes** | 100 sequential episodes per seed | Per-task independent execution (no sequential episode dependency) |
| **Topics** | 10 synthetic topics with cross-topic affinity | Database root-cause categories (INSERT_LARGE_DATA, LOCK_CONTENTION, etc.) |
| **Memory** | `StoredMemory` frozen dataclass (topic, content, contamination, true_effect) | Procedural payload dict (procedure, preconditions, postconditions, provenance) |
| **Injection** | Direct list of `StoredMemory` objects → `success_probability()` | `MarbleMemoryInjector` → YAML config → engine subprocess |
| **Evaluation** | `reward = float(success)` via stochastic model | `MarbleOutcome` via root-cause prediction matching |
| **Ground truth** | Known (`true_effect`, `contamination` field) | Unknown (must be inferred from paired records) |
| **Paired design** | Shared task RNG across methods | Shared `InitialStateBundle` + `generation_seed` |
| **Multi-agent** | 1 writer + 3 receivers (synthetic) | 1 writer + 1–5 receivers (real engine) |
| **Contamination** | Explicit `false` / `spurious` / `outdated` labels | Implicit (wrong procedure for task, cross-scenario mismatch) |
| **Cost per run** | ~0 seconds (numpy) | ~30-60 seconds per branch (LLM + Docker) |
| **Memory format** | `content: str` (template text) | `payload: dict` (structured procedure) |

### Critical Interface Gaps

1. **No episode loop**: MARBLE tasks are independent executions, not a
   sequential episode stream. "Late-stage" must be defined by task ordering,
   not time progression.

2. **No explicit contamination labels**: Synthetic experiments know
   `contamination=none/false/spurious`. MARBLE must infer from paired
   outcomes (share_outcome vs withhold_outcome → four-outcome label).

3. **Topic → root-cause mapping**: Synthetic `topic_affinity()` maps to
   task similarity in MARBLE. The adapter must map database root-cause
   categories to "topic-like" affinity groups.

4. **Memory budget**: Synthetic experiments use `capacity` parameter.
   MARBLE tasks have no explicit memory budget — the router decides
   which memories to inject (top-k or critic-gated).

5. **LLM dependency**: Each MARBLE branch requires a real LLM API call.
   Baseline experiments are expensive (~30s × 2 branches × 500 tasks × 5
   seeds = ~416 hours for one method). Scale-down to validation subsets
   is mandatory.

---

## 5. Existing Infrastructure

### Already Available

- `MarblePairedBranchRunner`: shared-control paired execution
- `MarbleMemoryInjector`: memory payload injection
- `DatabaseOutcomeEvaluator`: root-cause based evaluation
- `MarbleTaskProvider`: frozen manifest task loading
- Multi-receiver config: `configs/marble_3receiver.yaml`
- Multi-receiver collection: `experiments/marble_feasibility/multi_receiver/collect_multi_receiver.py`
- Receiver sanity analysis: `experiments/marble_feasibility/multi_receiver/run_receiver_sanity.py`
- Paired record format: `{task, receiver, memory, Y_expose, Y_withhold}`
- Baseline controllers: `ReflexionController`, `AgileController`, `HeuristicMemoryController`, `AgeMemController`
- Synthetic multi-agent: `experiments/lifelong/run_multi_agent.py`

### Needs to be Created (Phase 1–10)

- `configs/marble_baseline.yaml` — unified baseline experiment config
- `src/smtr/marble/memory_adapter.py` — `BaseMemoryController` ↔ MARBLE bridge
- Baseline experiment runner for MARBLE
- Receiver-conditioned analysis scripts
- Contamination propagation experiment
- Statistical analysis and table generation

---

## 6. Constraints for Baseline Experiments

1. **LLM cost**: Each branch ≈ 30-60s. Must limit to validation-scale
   subsets (3-10 tasks × 3 receivers × 5 seeds).

2. **Docker availability**: MARBLE engine requires Docker slots.
   `DockerSlotPool` manages parallel execution.

3. **Fairness**: All methods must share:
   - Same `InitialStateBundle` per (task, seed)
   - Same `generation_seed`
   - Same receiver agent configuration
   - Same evaluation (`DatabaseOutcomeEvaluator`)

4. **No modification to**:
   - `MarblePairedBranchRunner` execution logic
   - `MarbleMemoryInjector` payload format
   - `DatabaseOutcomeEvaluator` scoring
   - `engine_process.py` subprocess invocation

5. **Scale**: Start with sanity check (3 tasks, 1 seed), then scale
   to validation (10 tasks, 5 seeds).

---

## 7. File Map

| Component | Path |
|-----------|------|
| Task provider | `src/smtr/marble/task_provider.py` |
| Branch runner | `src/smtr/marble/branch_runner.py` |
| Memory injection | `src/smtr/marble/memory_injection.py` |
| Engine process | `src/smtr/marble/engine_process.py` |
| Environment (DB) | `src/smtr/marble/environment/scenarios/database.py` |
| Outcome (DB) | `src/smtr/marble/outcome/scenarios/database.py` |
| Paired evaluation | `src/smtr/marble/paired_evaluation.py` |
| Paired records | `src/smtr/marble/paired_records.py` |
| Isolation bundles | `src/smtr/marble/environment/isolation.py` |
| Multi-receiver | `experiments/marble_feasibility/multi_receiver/` |
| Synthetic multi-agent | `experiments/lifelong/run_multi_agent.py` |
| 3-receiver config | `configs/marble_3receiver.yaml` |
| Dataset manifest | `artifacts/marble/manifests/dataset.json` (500 tasks) |
| Split manifest | `artifacts/marble/manifests/splits.json` |
| Baseline controllers | `src/smtr/baselines/` |
