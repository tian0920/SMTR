# SMTR 代码结构审计（Task 0）

> 目的：为「长期 memory lifecycle」扩展（experience → candidate → TCI validation →
> consolidation/rejection → persistent memory bank → future task transfer）定位现有代码中
> 的 memory extraction / retrieval / receiver assignment / TCI intervention / evaluation
> pipeline，并给出最小修改点。**本文档只做分析，不修改任何代码。**

## 1. 顶层结构

```
src/smtr/
├── core/types.py                 # 核心类型别名
├── memory/                       # memory schema / pool / 存储（★ Task 1 落点）
├── marble/                       # MARBLE 真实引擎管道（采集/抽取/配对/评估）
├── intervention/                 # TCI 扰动干预算子与执行（★ Task 2 输入来源）
├── router/                       # critic / TCI ranker / gate / 路由决策
├── counterfactual/               # 离线配对反事实收集器
├── evaluation/                   # 离线评估套件（gates / splits / metrics）
├── cli.py                        # 顶层入口，转发给 marble.cli
└── schemas.py / config.py

experiments/
├── marble_feasibility/           # 可行性 + encoder ablation + leakage audit + multi-receiver
└── mechanism_validation/         # 机制验证（含合成因果环境 synthetic_memory_env.py）

configs/                          # 运行配置（当前仅 marble_3receiver.yaml）
scripts/                          # 实验启动/分析脚本
tests/                            # 按包镜像的单元测试（tests/memory/ 等）
artifacts/marble/                 # 所有 MARBLE 产物（受路径守卫约束）
```

## 2. 五大关键组件定位

### 2.1 Memory Extraction（记忆抽取）

| 项 | 位置 |
|----|------|
| 抽取函数 | [`src/smtr/marble/real_data.py`](../src/smtr/marble/real_data.py) `extract_procedural_memories()`（L127） |
| 数据模型 | `ExtractedMemory`（L101）：routing_card + payload + provenance |
| 落盘/读取 | `write_memory_pool()` / `load_memory_pool()`（L1160/L1174） |
| CLI | `smtr.marble.cli extract-database-memories`（trajectory-index → memory pool） |

- 输入：writer-agent 成功轨迹（`RealDatabaseTrajectory`，来自 `collect-database-trajectories`）
- 每个 agent slice（≥ min_actions）产生一条过程性 memory；procedure 来自真实
  action/tool 顺序；来源 agent 只记录为 `MemoryProvenance`（Writer-Agnostic 约束）
- **现状：抽取即终点，没有任何 validated/rejected 状态字段**

### 2.2 Memory Retrieval（检索/候选构建）

| 项 | 位置 |
|----|------|
| 打分 | `real_data.py::_score_memory_for_recipient()`（L732）——**仅用 metadata**（task_sim + requirement satisfaction 均值），不调用 outcome |
| 候选构建 | `build_cross_task_candidates()`（L553）→ `DatabaseCandidateManifest` |
| 分层 cohort | semantic_top / receiver_compatible / receiver_incompatible / cross_receiver_anchor |
| CLI | `build-database-candidates`、`build-budgeted-candidates` |
| 补充 | `router/candidate_proposer.py`（在线候选提议） |

### 2.3 Receiver Assignment（接收方分配）

| 项 | 位置 |
|----|------|
| receiver 条目加载 | `real_data.py::load_receiver_entries()`（L1183） |
| 锚点分配 | `_select_anchor_assignments()`（L855）：跨 ≥2 receivers 的 anchor memory |
| 治疗边（treatment edge） | `(target_task_id, receiver_agent_id, candidate_memory_id)` 三元组，`marble/real_pairs.py` |
| 运行时注入 | `marble/memory_injection.py`（share/withhold agent 输入 + runtime shim 写入指定 agent 的 BaseMemory）；`marble/branch_runner.py`（shared-control 配对执行） |
| 运行时可见性 | `marble/runtime_visibility_audit.py`（已验证 MARBLE database 任务 agent 编号为 1-based：agent1..agent5） |

### 2.4 TCI Intervention（干预实验）

| 项 | 位置 |
|----|------|
| 扰动算子 | `intervention/transfer_perturbation.py`（单字段扰动，硬不变量校验） |
| 扰动执行 | `intervention/perturbation_runner.py`（只跑扰动分支，复用 Y_0 / Y_original） |
| 干预指标 | `intervention/perturbation_analysis.py`（Flip Rate / Harmful Flip / Benefit Flip / Support Gain） |
| 对比构建 | `intervention/contrast_builder.py` → `InterventionContrast`（带 direction ∈ {-1,0,+1}） |
| TCI 监督 | `router/tci_supervision.py`（TCI 对 → 软标签二分类样本，混入 critic 训练） |
| CLI | `build-transfer-perturbations` → `run-transfer-perturbations` → `analyze-transfer-perturbations` → `build-intervention-contrasts` → `train-tci-ranker` / `evaluate-tci-*` |

**关键**：TCI 已经产出每条 (m, m~, direction) 的因果方向判定——这正是
Task 2 admission gate（δ = reward_expose − reward_withhold）所需的信号源。

### 2.5 Evaluation Pipeline（评估管道）

| 阶段 | 位置 | CLI |
|------|------|-----|
| critic 训练 | `marble/training.py`（paired records → `CandidateExposureInput` → `FourOutcomeTransferCritic`，支持 `--tci-contrasts` 蒸馏） | `train-critic` |
| 路由级评估（不跑引擎） | `marble/router_evaluation.py` + `router/smtr_gate.py` / `causal_gate.py` | `run-paired-decision-evaluation` |
| 端到端（真引擎） | `marble/end_to_end_evaluation.py`（router 选定 memory 注入后真实执行） | `run-marble-evaluation` |
| 配对因果效应 | `marble/paired_causal_evaluation.py`（τ = Y_expose − Y_withhold） | — |
| 离线评估套件 | `evaluation/`（ablation_gates / split_audit / metrics / group_effects / leakage_scanner） | `audit-splits` / `integrity-audit` |
| 可行性/机制验证 | `experiments/marble_feasibility/*`、`experiments/mechanism_validation/run_validation.py` | 直接 `python` 运行 |

## 3. 当前数据流图

```
[collect-database-trajectories]          MARBLE 真引擎跑 train split 任务
        │  trajectories.jsonl
        ▼
[extract-database-memories]  ── real_data.extract_procedural_memories
        │  memory_pool.jsonl        (candidate memory，无 lifecycle 状态)
        ▼
[build-database-candidates]  ── _score_memory_for_recipient (metadata-only)
        │  candidate_manifest.json  (receiver-conditioned 候选集)
        ▼
[generate-database-paired-records] ── branch_runner shared-control 执行
        │  paired_records.jsonl     (每个 treatment edge: Y_expose / Y_withhold → τ)
        ├──► [build-transfer-perturbations] → [run-transfer-perturbations]
        │        → [analyze-transfer-perturbations] → [build-intervention-contrasts]
        │                │  intervention_contrasts.jsonl (TCI direction)
        │                ├──► train-tci-ranker / evaluate-tci-*
        │                └──► train-critic --tci-contrasts (TCI 蒸馏进 critic)
        ▼
[train-critic]  ── FourOutcomeTransferCritic (flat / opportunity_factorized)
        ▼
[run-paired-decision-evaluation]  ── router gate 决策（share/withhold）离线评估
        ▼
[run-marble-evaluation]  ── 真引擎端到端：router 选 memory → 注入 → 执行
        ▼
[audit-splits / integrity-audit]  ── split 隔离与完整性审计
```

**缺口（lifecycle 视角）**：
1. memory 一经抽取即为"永久候选"，无 `candidate → validated/rejected` 状态机
2. TCI direction 只用于 ranker/critic 蒸馏，**不回写 memory 状态**
3. 无跨 episode 的 persistent bank；每个实验独立构建 memory pool
4. 无污染注入、无长期 episode 序列、无跨分布迁移协议

## 4. 实验入口脚本与配置位置

### 实验入口
| 入口 | 用途 |
|------|------|
| `src/smtr/cli.py` → `src/smtr/marble/cli.py` | 全部 MARBLE 管道命令（唯一正式 CLI） |
| `experiments/marble_feasibility/run_feasibility.py` | 可行性主流程 |
| `experiments/marble_feasibility/collect_interventions.py` | 干预对采集（已支持多 receiver） |
| `experiments/marble_feasibility/multi_receiver/collect_multi_receiver.py` | 3-receiver 真引擎采集 |
| `experiments/marble_feasibility/{encoder_ablation,leakage_audit}/run_*.py` | Task 1/2 验证（已 PASS） |
| `experiments/mechanism_validation/run_validation.py` | 机制验证（含 `envs/synthetic_memory_env.py` 合成环境） |
| `scripts/run_full_q30b_experiment.sh` 等 | 真实 LLM 实验脚本 |
| `run_regression.sh` | 测试回归（排除 29 个 legacy 破损文件） |

### 配置位置
| 配置 | 位置 |
|------|------|
| 多 receiver | `configs/marble_3receiver.yaml` |
| 可行性 | `experiments/marble_feasibility/config.yaml` |
| 机制验证 | `experiments/mechanism_validation/configs/mechanism_default.yaml` |
| LLM | `conf/llm_test_config.json` + 环境变量（DASHSCOPE_API_KEY 等） |

## 5. 最小修改点（Task 1–10 落点建议）

| Task | 落点 | 修改性质 |
|------|------|----------|
| Task 1 PersistentMemory | **新增** `src/smtr/memory/persistent_memory.py` + `memory_schema.py`；测试放 `tests/memory/`。`memory/__init__.py` 现有 4 个导出**保持不变**（只增不改） | 纯新增 |
| Task 2 Admission Gate | **新增** `src/smtr/memory/consolidation.py`（MemoryAdmissionController）。输入直接复用 paired_records 的 Y_expose/Y_withhold（τ），规则 δ>0，无新阈值 | 纯新增 |
| Task 3 Lifelong runner | **新增** `experiments/lifelong/run_lifelong.py`。推荐基于 `mechanism_validation/envs/synthetic_memory_env.py` 扩展成长期环境（可控 τ、零引擎成本、可复现）；不触碰 marble CLI | 纯新增 |
| Task 4 Formation | `experiments/lifelong/` 内新增实验对比逻辑；复用 Task 1 bank 的统计接口 | 纯新增 |
| Task 5 Contamination | **新增** `experiments/contamination/contamination_generator.py`，输出复用 `ExtractedMemory` schema | 纯新增 |
| Task 6 Transfer | `experiments/lifelong/` 或独立目录；A/B 分布由环境采样参数定义（seed 控制，不新增超参） | 纯新增 |
| Task 7 Multi-agent | 复用 `configs/marble_3receiver.yaml` 的 3-receiver 设定（heterogeneity 已 PASS：variance=0.333） | 纯新增 |
| Task 8 Budget | 复用 bank 容量参数（10/50/100），无 adaptive threshold | 纯新增 |
| Task 9 Launcher | **新增** `scripts/run_all_longterm.py` | 纯新增 |
| Task 10 Report | **新增** `docs/longterm_experiment_report.md` | 纯新增 |

### 兼容性保证
- **不改动** `src/smtr/marble/`、`src/smtr/router/`、`src/smtr/intervention/` 任何现有函数签名
- **不改动** `memory/__init__.py` 既有导出与现有 schema（`ProcedurePayload`、`MemoryRoutingCard` 保持 frozen）
- receiver=1 主实验（现有 paired records / critic / e2e 评估链）完全不受影响；receiver=3 仅通过新实验路径扩展
- 回归基线：`bash run_regression.sh`（当前通过）

### 风险提示
1. **真引擎成本**：MARBLE 真引擎每分支 ~3–9 分钟 + LLM 费用。Task 3–8 的 episode 级实验
   （100 episodes × 多方法 × 5 seeds）若全部跑真引擎不可行 → 建议主实验用合成环境
   （ground-truth τ 已知），真引擎仅做 Task 7 小规模验证。
2. **artifact 路径守卫**：`assert_marble_artifact_path` 要求产物写入 `artifacts/marble/` 下
   且从项目根目录运行；lifelong 实验产物建议放 `artifacts/marble/outputs/lifelong*` 或
   `results/lifelong/`（后者不在守卫范围内，需确认走哪条路径）。
3. **task_id 非全局唯一**：5 个 scenario 各有独立 id 空间；任何按 task_id 索引的逻辑必须
   携带 scenario（multi_receiver 采集已踩过此坑）。
