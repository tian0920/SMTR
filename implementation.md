# SMTR 项目交接说明

## 核查标记说明

本文件已按 `method.md` 的研究主线核查并标注：

* <span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>：绿色，属于 `method.md` 核心方法，且当前代码大框架已实现。
* <span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#ea580c;color:#ffffff;font-weight:700">SMTR 必需 未实现/不完全</span>：橙色，属于 `method.md` 核心方法，但当前代码未实现、默认路径未接入，或实现与方法定义不完全一致。
* <span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#6b7280;color:#ffffff;font-weight:700">非必要细节</span> + ~~删除线~~：灰色/删除线，工程交接或扩展实验中有用，但不是 `method.md` 第一版方法主线所必需的细节；按要求保留原文但划掉。

## 1. 项目目标

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>

项目名称：**SMTR — Shared Memory Transfer Router**

目标是在多 Agent 系统中研究：

> 某条 shared procedural memory 是否应该暴露给某个正在执行任务的 agent？

我们不把问题定义为传统 retrieval，即“哪条 memory 与任务最相似”，而是定义为状态条件下的因果迁移问题：

$$\tau^\pi(m \mid o,S)=
\mathbb{E}
\left[
Y^{(1),\pi}-Y^{(0),\pi}
\mid o,S,m
\right]
$$

其中：

* (m)：候选 procedural memory；
* (o)：当前任务、agent、环境状态、任务阶段、局部上下文等；
* (S)：当前 agent 已经被注入的 memory 集合；
* ($Y^{(1),\pi}$)：暴露 (m) 时的团队任务结果；
* ($Y^{(0),\pi}$)：不暴露 (m) 时的团队任务结果；
* ($\pi$)：target memory 之后继续处理其他候选 memory 的冻结 continuation policy。

我们尤其关注负迁移：

$$\eta^\pi(m \mid o,S)=\Pr
\left(
Y^{(1),\pi}=0,;
Y^{(0),\pi}=1
\mid o,S,m
\right)
$$

也就是：给 agent 看了 memory 反而导致失败，而不给它看本可成功。

---

## 2. 核心研究假设

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>

系统中的 shared memory 不应被视为天然有用或天然无用。其作用取决于：

* 接收 memory 的 agent；
* 当前任务及子任务阶段；
* 当前环境状态与工具约束；
* agent 已有的局部上下文；
* 已经注入的 memory 前缀集合 (S)；【多轮情况下，前面轮次已经有注入的memory】
* 后续 memory routing policy。

因此，memory 的价值不是全局常数：

$\tau(m)$

而是：

$$
\tau^\pi(m \mid o,S)
$$
这也是项目与普通 semantic retrieval、role-aware routing、procedure-success prediction 的主要区别。

---

## 3. 系统控制流

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>

系统分为两个严格分离的阶段：

```text
Shared Memory Pool
        ↓
Candidate Proposer
        ↓
Top-K Routing Cards
        ↓
Router / Sequential Selector
        ↓
Selected Memory IDs
        ↓
Load only selected ProcedurePayloads
        ↓
Inject payloads into receiving agent context
        ↓
Agent execution
        ↓
Team outcome + trajectory logging
```

### Candidate proposal

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>

Candidate proposer 的职责是高召回筛选，不做因果决策。

当前使用 deterministic hybrid score：

$$
0.45\cdot \text{goal similarity}
+
0.15\cdot \text{task tag overlap}
+
0.25\cdot \text{environment compatibility}
+
0.15\cdot \text{receiver compatibility}
$$

candidate proposer 只能读取 `MemoryRoutingCard`，绝不能读取完整 procedure payload。【MemoryRoutingCard是指memory的前缀，而不是memory内容本身】

### Router

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>

Router 是唯一拥有 `share / withhold` 决策权的组件。

```text
share:
  将 procedure payload 注入 receiving agent context

withhold:
  不注入 payload
```

agent 不负责判断自己是否应该看到某条 memory。agent 只负责执行，并为 router 提供 outcome supervision。

---

## 4. 最重要的数据隔离原则

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>

每条 memory 被拆为两个对象：

```text
ProcedurePayload
  - goal
  - preconditions
  - steps
  - postconditions
  - version
  - writer / source metadata

MemoryRoutingCard
  - goal summary
  - task tags
  - environment constraints
  - receiver compatibility
  - raw execution evidence
  - paired transfer evidence
  - router-visible statistics
```

关键约束：

1. `ProcedurePayload.steps` 是敏感内容。
2. Candidate proposer 不能读取 payload。
3. Router 在选择前不能让执行 agent 接触 payload。
4. 未被选中的 payload 不得出现在：

   * agent local context；
   * global graph state；
   * router trace；
   * paired record；
   * feature encoder 输入。
5. paired record 中只保存 immutable card-level snapshot，不保存 payload steps。

这是保证 `share / withhold` 作为有效 treatment 的基础。

---

## 5. 当前代码架构

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>核心分层与 `method.md` 一致：memory pool / candidate proposer / sequential router / receiving agent / counterfactual collection / critic。  
~~以下完整目录清单是工程交接细节，不是方法论文主线的必要组成。~~

核心目录如下：

```text
src/smtr/
  runtime/
    graph.py
    state.py
    agents.py
    environment.py
    fake_llm.py
    real_llm.py
    tool_environment.py
    api_server.py
    delegation_topology.py

  memory/
    schemas.py
    store.py
    repository.py
    pool.py
    snapshot.py
    serialization.py
    procedure_writer.py
    execution_evidence.py
    paired_transfer_evidence.py
    seed_memories.py
    refinement.py
    meta_procedure.py

  router/
    candidate_proposer.py
    baseline_router.py
    interfaces.py
    traces.py
    transfer_features.py
    transfer_critic.py
    transfer_evaluation.py
    sequential_router.py
    safety_guard.py
    off_policy_correction.py

  counterfactual/
    schemas.py
    candidate_traversal.py
    continuation_policy.py
    decision_points.py
    prefix_sampler.py
    interaction_boundary_sampler.py
    forced_router.py
    paired_rollout.py
    policy_round.py
    record_writer.py
    snapshot.py
    task_provider.py

  policy/
    schemas.py
    manifests.py
    fingerprints.py
    no_share_policy.py
    critic_sequential_policy.py
    exploratory_policy.py
    online_refresh.py

  evaluation/
    splitters.py
    temporal_integrity.py
    shortcut_diagnostics.py
    feature_ablation.py
    interaction_audit.py
    compositional_splits.py
    leakage_scanner.py
    logging.py
    group_effects.py
    stale_propagation.py
```

当前 runtime 使用固定 LangGraph workflow：

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>该流程实现了“Router 决策后 agent 执行并记录反馈”的因果时序。

```text
START
  ↓
pre_route_planner
  ↓
planner
  ↓
pre_route_executor
  ↓
executor
  ↓
pre_route_critic
  ↓
critic
  ↓
END
```

每个 `pre_route_<agent>` 节点都会：

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#ea580c;color:#ffffff;font-weight:700">SMTR 必需 不完全</span>该流程总体必要且已实现；但当前默认 runtime 调用 router 时没有把 `cards_by_id/context` 传给 production sequential router，因此 learned router 默认接入仍不完整。

```text
1. 读取 task、环境、agent local context
2. 从 memory pool 获取 routing cards
3. candidate proposer 生成 Top-K 候选
4. router 对候选进行 share / withhold
5. 仅加载 selected IDs 对应 payload
6. 将 selected payload 注入当前 agent local context
7. 记录 router trace
```

---

## 6. 已完成模块

### 6.1 LangGraph 多 Agent runtime

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>需要一个 receiving-agent 执行环境来承接 router 暴露的 payload 并产生 outcome。

已实现：

* planner → executor → critic；
* ~~deterministic fake LLM；~~
* ~~deterministic ToyEnvironment；~~
* agent local context 隔离；
* snapshot / restore；
* 统一 router trace；
* `NoMemoryRouter` baseline；
* <span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#ea580c;color:#ffffff;font-weight:700">SMTR 必需 不完全</span>默认 demo 不加载 learned router。

### 6.2 Versioned procedural memory store

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>需要持久化 payload/card 分离、证据、snapshot 与版本，以保证 paired rollout 和 temporal integrity。

当前使用 SQLite，包含：

```text
memory_payload_versions
memory_routing_cards
execution_evidence
paired_transfer_evidence
memory_store_metadata
```

已支持：

* payload versioning；
* routing card 与 payload 隔离；
* raw execution evidence；
* paired transfer evidence；
* bounded FIFO context buffers；
* read-only memory snapshots；
* pinned active payload version；
* store revision；
* snapshot digest。

### 6.3 Raw execution evidence

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>必要，因为 `method.md` 明确区分 procedure execution success 与 share-vs-withhold transfer effect。

原始 procedure execution success 与 transfer effect 被严格区分。

raw execution evidence 更新：

$$
\alpha \leftarrow \alpha+1
$$

或：

$$
\beta \leftarrow \beta+1
$$

但它不等价于 positive / negative transfer。

### 6.4 Paired counterfactual rollout

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>这是 `method.md` 中负迁移风险与四结果监督的核心识别机制。

已实现：

```text
share branch:
  S ∪ {m}

withhold branch:
  S
```

两分支必须共享：

* graph state snapshot；
* environment snapshot；
* candidate order；
* selected prefix；
* memory snapshot；
* continuation policy；
* router checkpoint；
* seed derivation rule。

两分支唯一被强制改变的变量是 target memory 是否被注入。

四类标签：

```text
Yshare=1, Ywithhold=0 → positive
Yshare=0, Ywithhold=1 → negative
Yshare=1, Ywithhold=1 → neutral_success
Yshare=0, Ywithhold=0 → neutral_failure
```

### 6.5 Prefix-conditioned collection

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>必要，因为研究对象是 `tau(m | o,S)`，非空 prefix 是检验 selected-set conditioning 的关键。

已经支持：

$$
\tau(m\mid o,S)
$$

其中 (S) 可以非空。

prefix sampling 支持：

```text
empty
uniform
stratified
```

prefix memory 必须：

* 出现在 target 前；
* 不包含 target；
* 默认无明确 environment conflict；
* receiver-compatible；
* 位于 pinned memory snapshot 中。

### 6.6 Four-outcome transfer critic

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>必要；当前把 transfer critic 与 harm critic 合并为四结果分类器，能导出 `tau` 与 `eta`。

当前 critic 输出：

$$
q_{00},q_{01},q_{10},q_{11}
$$

其中：

```text
q00: neutral failure
q01: negative transfer
q10: positive transfer
q11: neutral success
```

并导出：

$$
\tau=q_{10}-q_{01}
$$

$$
\eta=q_{01}
$$

当前实现：

* hashing feature encoder；
* bootstrap logistic-regression ensemble；
* uncertainty intervals；
* support-distance diagnostic；
* policy-specific checkpoint metadata；
* strict training guard。

### 6.7 Policy-aware collection rounds

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>必要；`method.md` 的 `tau^pi` 依赖冻结 continuation policy，不能混合不同 estimand。

当前已实现：

```text
pi0 -> C0 -> pi1 -> D1 -> C1 -> pi2_explore -> D2 -> C2
```

重点原则：

> 一个 critic 只能估计其对应 frozen continuation policy 下的 effect。

不允许把不同 policy fingerprint 的 paired records 混合训练。

### 6.8 Candidate–prefix pairwise interaction encoder (A-01)

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>必要；这是当前代码对 `SetEnc(S)` / candidate-prefix interaction 的具体实现方式。  
<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#ea580c;color:#ffffff;font-weight:700">SMTR 必需 不完全</span>它是 hashing-token 的工程实现，不是显式神经 SetEnc；且由于 routing card 不含 `strategy`，无法覆盖所有 interaction 维度。

在 `router/transfer_features.py` 中扩展了 `HashingTransferFeatureEncoder.tokens()`，
使编码器能够表达 candidate 与 selected-set 成员之间的 pairwise 交互，而不仅仅是
「selected-set 是否包含某类 memory」的聚合统计。

新增辅助函数与常量：

- `INTERACTION_SIGNALS`：8 个信号名称（env_agree / env_conflict / forbidden_conflict
  / precond_postcond_overlap / postcond_postcond_overlap / role_overlap /
  capability_overlap / task_tag_overlap）。
- `_pair_interaction_signals(candidate, selected)`：单对交互信号。
- `_pairwise_interaction_tokens(candidate, selected_cards)`：对所有 pair 做
  permutation-invariant 聚合（mean/max/min 分桶 + conflict/compatibility/pair count）。

交互 token 仅在 `full` block 中保留；`_include_token()` 会排除 `interaction_*`
前缀的 token 出 `context_only / candidate_only / selected_set_only /
context_plus_candidate` 这些消融子块，使 `full_model_gain_over_best_single_block`
能够直接量化交互特征的边际贡献。

由于 `RoutingFeatureSnapshot` 刻意不含 `strategy` 字段以防机制泄漏
（`test_card_feature_snapshots.py` 断言 `strategy` 不得出现在 routing-card 序列化中），
A-01.7（`candidate_strategy` vs `prefix_strategy`）**无法实现**。

### 6.9 Interaction-boundary prefix sampler (S2 / A-07)

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>作为离线 paired rollout 数据采集机制是必要的辅助细节，用于覆盖 `tau(m|o,S)` 的边界和 effect-flip 区域。  
<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#ea580c;color:#ffffff;font-weight:700">SMTR 必需 不完全</span>它不属于 production router；`method.md` 要求测试时报告随机 permutation 均值/方差，当前这里强调的是定向采样而非完整随机顺序评估。

新增 `counterfactual/interaction_boundary_sampler.py`，**仅用于离线采集**，不进入
production router。该采样器按「是否可能引起 effect-flip」给候选前缀打分，而非按
prefix size 平衡性。

两个互补评分成分：

- **结构评分**（A-07.1/2）：复用 6.8 的 `_pair_interaction_signals`，按 env_conflict /
  forbidden_conflict / precond_postcond_overlap 等信号打分（冲突权重 2.0，重叠
  权重 0.25–1.0）。
- **critic 评分钩子**（A-07.4/5/6）：可注入 `critic_scorer`，按空前缀 vs 候选前缀的
  预测 disagreement + ensemble uncertainty + τ̂ 近零打分。机制无关——不需要 payload
  strategy。

采样规则：取 top-k（按 score 降序）交互信号 > 0 的前缀；如果所有候选得分为 0，
回退空前缀，不注入噪声前缀。

A-07.3（target vs prefix action strategy 相反）**无法实现**，与 A-01.7 同根因。

配合 CLI `--scenario-mix interaction`（将 `prefix_sensitive` 纳入采集）与
`--prefix-mode interaction-boundary`，该采样器可以定向采集 lock-vs-recover 的
effect-flip 样本（`positive → neutral_failure` 等四类）。

### 6.10 Production sequential router

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#ea580c;color:#ffffff;font-weight:700">SMTR 必需 未实现/不完全</span>必要；这是 `method.md` 最终在线策略

```text
share iff LCB(tau_hat) > 0 and UCB(eta_hat) <= epsilon
```

的承载模块。当前代码已有 `ProductionSequentialRouter`，能按候选顺序维护 selected set
并调用 critic，但与 `method.md` 仍有三处不完全一致：

* 当前线上遍历的是 `proposal.ranked_candidates`，不是默认随机 permutation；
* 当前 production router 的正迁移门控主要用 `tau_mean > threshold`，不是严格 `tau_lcb > 0`；
* 当前 runtime 默认调用 `decide_from_proposal()` 时没有传入 `cards_by_id/context`，learned router 直接接入默认 workflow 时会缺少 critic 输入。

`FrozenCriticSequentialContinuationPolicy` 已实现更接近方法公式的 LCB/UCB 判定，但它主要用于
离线 collection / frozen continuation policy，而不是默认 production runtime。

---

## ~~7. 已生成的重要产物~~

~~<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#6b7280;color:#ffffff;font-weight:700">非必要细节</span>以下 checkpoint、output、fingerprint 是实验复现与交接材料，不是 `method.md` 第一版方法定义的必要组成。~~

```text
policies/pi2_explore.json
policies/pi2_explore_v2.json
data/paired_records_pi2_v13.jsonl
data/paired_records_pi2_s2_v14.jsonl
data/paired_records_pi2_s4_v15.jsonl
checkpoints/critic_pi2.joblib
checkpoints/critic_pi2_interaction.joblib
checkpoints/critic_pi2_s2.joblib
checkpoints/critic_pi2_s4.joblib

outputs/pi2_collection_quality.json
outputs/feature_leakage_scan_pi2.json
outputs/feature_block_audit_pi2.json
outputs/feature_block_audit_pi2_interaction.json
outputs/feature_block_audit_pi2_s2.json
outputs/feature_block_audit_pi2_s4.json
outputs/prefix_sensitivity_pi2.json
outputs/prefix_sensitivity_pi2_interaction.json
outputs/prefix_sensitivity_pi2_s2.json
outputs/prefix_sensitivity_pi2_s4.json
outputs/candidate_substitution_pi2.json
outputs/candidate_substitution_pi2_interaction.json
outputs/candidate_substitution_pi2_s2.json
outputs/candidate_substitution_pi2_s4.json
outputs/critic_pi2_compositional_eval.json
```

`*_pi2*` = 老 pi2 数据 + 老 critic（无交互编码）；`*_pi2_interaction*` = 老 pi2 数据
+ A-01 编码器 critic；`*_pi2_s2*` = S2 interaction-boundary 数据 + A-01 编码器 critic。

~~关键 policy fingerprint：~~

```text
pi2_explore:
18563e8540256d53ae44d91fbd9e847e48ebabbfa34dde3380bfa1f62e3ac87f
```

~~其 source critic estimand policy fingerprint：~~

```text
452e7fe06f37f34e79f5282879a80e16e6551539fc7a62b298609e08442ea373
```

---

## ~~8. 最新 D2 / C2 结果~~

~~<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#6b7280;color:#ffffff;font-weight:700">非必要细节</span>以下结果用于记录阶段性实验状态，但不属于方法实现本身；方法只要求定义并报告相关指标。~~

### ~~D2 collection~~

```text
records: 480

positive: 66
negative: 40
neutral_success: 136
neutral_failure: 238

prefix size:
  |S|=0: 162
  |S|=1: 160
  |S|=2: 158

continuation share rate: 0.2637
hard-risk continuation share rate: 0.0
```

说明当前 exploratory policy 已经产生真实的非零 continuation share，而非退化为 no-share policy。

continuation trace 中：

```text
safe_exploit: 3689
boundary_explore: 0
risk_veto: 0
hard_ood_veto: 0
budget_exhausted: 1320
```

### ~~C2 基础性能~~

```text
test accuracy: 0.9063
macro F1: 0.8779
log loss: 0.2968
```

compositional split accuracy：

```text
environment: 0.8938
episode: 0.9063
factor_combination: 0.9182
prefix: 0.8333
scenario: 1.0
surface: 0.9063
target family: 0.8133
```

### ~~Feature block audit~~

```text
full model macro F1: 0.8779
best single block:
  context_plus_candidate macro F1: 0.7818

full-model gain: +0.0961
```

这说明 selected-memory set 相关特征确实增加了预测信息，但尚不足以证明它正确学习了 interaction direction。

### ~~Interaction audit~~

```text
prefix sensitivity direction accuracy: 0.50
candidate substitution direction accuracy: 0.78
```

这是当前最重要的问题。

### ~~A-01 / S2 改造后的最新结果~~

三个阶段的对比（old = pi2 基线；A-01 = 仅编码器改进；S2 = 编码器 + 定向采集）：

| 指标 | old(pi2) | A-01(pi2) | S2 |
|---|---|---|---|
| **prefix sensitivity direction accuracy** | 0.50 | 0.54 | **0.65** |
| prefix Δτ MAE | 0.1618 | 0.1577 | **0.1039** |
| feature block: full macro F1 | 0.8779 | 0.8693 | **0.9610** |
| full-model gain over best single block | +0.0961 | +0.0874 | **+0.2228** |
| candidate substitution direction accuracy | 0.78 | 0.78 | N/A（matched_pair=0） |

S2 已让关键指标 prefix sensitivity direction accuracy **明显高于 0.50**，初步达到
建议的验收标准；feature-block full macro F1 与 full-model gain 大幅提升；
但引入 candidate-substitution 审计覆盖率回归（详见 Section 9 与 14）。

### ~~S3 强化后的 prefix audit 指标~~

S3（A-08/A-09/A-10）为交互审计新增了比 direction accuracy 敏锐得多的指标。
对同一批 critic 重跑 `audit-prefix-sensitivity`：

| 新增指标 | old(pi2) | A-01(pi2) | S2 |
|---|---|---|---|
| delta correlation（A-10.1） | 0.085 | 0.253 | **0.876** |
| delta MAE（A-10.2） | 0.192 | 0.158 | **0.104** |
| transfer-region flip accuracy（A-10.3） | 0.714 | 0.714 | **1.00** |
| positive→neutral / negative→neutral / neutral→positive detection | – / – / 0.69 | – / – / 0.69 | 1.0 / 1.0 / – |

- **delta correlation** 是最敏锐的信号（0.085→0.253→0.876），远比 direction accuracy
  （0.52→0.54→0.65）更能区分三个阶段。
- transition detection 揭示不同数据集实际覆盖的 flip 类型不同：pi2 以 neutral→positive
  为主，S2 以 positive/negative→neutral 为主；`positive→negative` 在三个数据集中均为 0 对。
  **注：ToyEnvironment 增强后已修复——新数据 pos→neg = 13 对（dir_acc=1.0）。**
- 注：old(pi2) 重跑 direction accuracy 为 0.52（历史记录 0.50），因旧 critic 的 pickle 实例
  在已演进的编码器（含交互 token）下预测，与原始训练特征存在偏差（详见 Section 9）。

### ~~S4：boundary exploration 修复后的结果~~

S4（A-11/A-12）修复了 exploratory continuation policy 中 boundary_explore 恒为 0 的 bug：
`trigger` 复用了驱动 safe_exploit 的同一个 `score`，使 boundary 分支数学上不可能命中（详见 Section 9.3）。
修复：独立 `trigger_score`（不放宽风险阈值），policy_version 1→2。以相同 S2 设置重采 480 条（仅换 v2 策略）：

| 继续行为 | S2（v1） | S4（v2） |
|---|---|---|
| **boundary_explore share count** | 0 | **1344** |
| safe_exploit share count | 2828 | 1416 |
| continuation share rate | 0.3106 | 0.3031 |
| **hard_risk_share_rate** | 0.0 | **0.0**（未变） |

下游（S4 数据 vs S2）：prefix dir_acc 0.65→0.64；delta correlation 0.876→0.821；
full macro F1 0.961→0.901；full-model gain +0.2228→+0.2507。S4 主目标已达成（boundary_explore 从 0
升至 1344、风险未放宽）；下游指标与 S2 大体相当。

---

## 9. 当前最关键的研究问题

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#ea580c;color:#ffffff;font-weight:700">SMTR 必需 未实现/不完全</span>本节对应 `method.md` 的核心尚未完全闭环处：selected-set conditional effect、interaction boundary、scenario leakage / shortcut、真实环境外推。

### 9.1 Prefix interaction 没有被充分学到

**进展**：A-01（交互编码器）将该指标从 0.50 提升至 0.54；S2（定向采集 effect-flip 数据）
进一步提升至 0.65，已明显高于 0.50。核心问题尚未彻底解决（候选 substitute 审计
matched_pair 回归为 0，仍需 Section 14 的更严格验收），但已初步达标。

项目最核心的目标是：

$$
\tau(m\mid o,S)
$$

但当前 prefix sensitivity direction accuracy 为：

```text
0.50
```

接近随机水平。

这说明 critic 对以下问题仍不可靠：

> 同一条 target memory，在不同已选 memory prefix 下，正迁移为什么会消失、变弱或转为负迁移？

当前 critic 已经能较好区分 target candidate 的作用：

```text
candidate substitution direction accuracy: 0.78
```

但仍不能可靠地识别 target–prefix interaction。

这比整体 classification accuracy 更值得优先解决。

### 9.2 Selected-set encoder 过于聚合

当前 selected-set representation 主要依赖 hashing token 和 aggregate statistics。

它很可能丢失以下信息：

* 哪一条 prefix memory 与 target memory 冲突；
* target precondition 是否被 prefix 改写；
* target postcondition 是否被 prefix 阻断；
* target strategy 与 prefix strategy 是否互补；
* environment constraint 是否冲突；
* 角色和能力条件是否被 prefix 改变。

目前的 `selected_set_only` / aggregate encoding 很可能无法表达 candidate–prefix 的配对关系。

### 9.3 Boundary exploration 没有真正触发 `[已修复：S4]`

虽然 `pi2_explore` 有 boundary exploration 机制，但早期结果是：

```text
boundary_explore = 0
```

**根因（S4/A-11）**：`decide()` 中 `trigger = score < exploration_round_probability` 复用了
驱动 `safe_exploit` 门控的同一个 `score`。safe_exploit 抢走所有 `score < 0.375`，故 boundary
分支只在 `score ≥ 0.375` 运行，而 trigger 需 `score < 0.30`——永为假。已通过独立 `trigger_score`
修复（不放宽风险阈值，policy_version 1→2），boundary_explore 从 0 升至 1344（详见 Section 8）。

早期所有 continuation share 都来自 `safe_exploit`，意味着数据对高置信区域偏置，缺少
near-threshold、effect flip、uncertain interaction 等边界数据；S4 修复后已引入 boundary 探索样本。

### ~~9.4 Scenario split 仍然异常高~~

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>已通过 ToyEnvironment 增强解决。

```text
scenario split macro_f1 = 0.732（原 1.0）
```

修复措施：将 flip 场景合并到基础场景的 scenario_family、添加隐藏扰动机制（perturbation_offset
从 context fingerprint 排除）、增加环境 regime 变化。scenario_family 现在包含混合 transfer class。

### ~~9.5 当前环境仍是 deterministic toy environment~~

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>ToyEnvironment 已增强，支持前缀–目标交互、flip 场景、上下文模式、资源追踪、隐藏扰动。

现阶段验证的是系统设计和因果数据管线，不是对真实 LLM agent 或真实工具环境的最终有效性证明。

尚未覆盖：

* 真实模型采样噪声；
* 工具 API 版本变化；
* 动态网页或数据库状态；
* 多 agent delegation；
* agent communication failure；
* memory writing / correction；
* 高阶 memory interaction。

---

## 10. 接手后最优先的工作

### Priority 1：重构 selected-set interaction encoder `[已实现：A-01]`

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>必要；对应 `method.md` 的 Set Encoder 与 `tau(m|o,S)`。

**已实现状态**：详见 6.8。prefix sensitivity direction accuracy 从 0.50 提升到 0.54（仅编码器），
进一步配合 S2 定向采集达到 0.65。A-01.7 因 `strategy` 字段被刻意排除在 routing card 之外，
无法实现。

不要马上部署 learned router。

建议新增 candidate–prefix pairwise interaction feature：

```text
candidate_required_env vs prefix_required_env
candidate_forbidden_env vs prefix_forbidden_env
candidate_precondition vs prefix_postcondition
candidate_postcondition vs prefix_postcondition
candidate_role vs prefix_role
candidate_capability vs prefix_capability
candidate_strategy vs prefix_strategy
candidate task tags vs prefix task tags
```

对每个 target–prefix pair 生成 feature，再做 permutation-invariant aggregation：

```text
mean
max
min
count
conflict count
compatibility count
```

重点不是只表示：

```text
selected set contains recover-like memories
```

而是表示：

```text
target recover conflicts with prefix lock
```

### Priority 2：定向采集 interaction-disagreement 数据 `[已实现：S2]`

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>必要；对应 paired rollout 在 selected-prefix 边界处的覆盖。

**已实现状态**：详见 6.9。`counterfactual/interaction_boundary_sampler.py` 实现
结构评分 + critic 评分钩子（A-07.3 因 `strategy` 不在 routing card，同 A-01.7 无法实现）；
配合 `--scenario-mix interaction` 纳入 `prefix_sensitive` 场景。480 条 S2 数据已生成
（`data/paired_records_pi2_s2_v14.jsonl`），prefix sensitivity direction accuracy 从 0.54 跃升到 0.65。
但 candidate-substitution 审计在 S2 数据上 matched_pair_count=0（覆盖率回归），需 Section 14 补充平衡采集或改进审计。

不要只平衡 prefix size。

应主动收集：

```text
positive -> neutral
positive -> negative
negative -> neutral
neutral -> positive
```

也就是：

$$
\tau(m\mid o,\varnothing)
\neq
\tau(m\mid o,S)
$$

的样本。

可增加一个仅用于离线 collection 的 interaction-boundary sampler，优先选择：

* target 与 prefix 有 precondition conflict；
* prefix 会改变 target 的 environment applicability；
* target 和 prefix action strategy 相反；
* 当前 critic 对不同 prefix 的 prediction disagreement 大；
* ensemble uncertainty 大；
* (\hat\tau) 接近零。

### Priority 3：改进 prefix audit 指标 `[已实现：S3]`

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>必要；对应 `method.md` 建议报告 order/prefix sensitivity 与 transfer-region boundary。

**已实现状态**：`evaluation/interaction_audit.py` 现额外返回 `delta_correlation`（A-10.1）、
`delta_mae`（A-10.2）、`transfer_region_flip_accuracy`（A-10.3）与 `flip_detection`
（A-10.4–7：positive→neutral / positive→negative / negative→neutral / neutral→positive），
同时保留原有 direction accuracy 以保兼容。结果见 Section 8「S3 强化后的 prefix audit 指标」。

当前只看 direction accuracy 不够。

建议增加：

$$
\Delta\tau_{\text{true}}=
\tau(m\mid o,S_1)-\tau(m\mid o,S_2)
$$

$$
\Delta\hat{\tau}=
\hat{\tau}(m\mid o,S_1)-\hat{\tau}(m\mid o,S_2)
$$

并报告：

```text
delta correlation
delta MAE
transfer-region flip accuracy
positive-to-neutral detection
positive-to-negative detection
negative-to-neutral detection
neutral-to-positive detection
```

### Priority 4：让 boundary exploration 真正覆盖边界 `[已实现：S4]`

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>作为训练数据采集策略必要；但 production router 的最终决策仍应保持风险约束。

**已实现状态**：根因是 `trigger` 复用了驱动 safe_exploit 的 `score`（非下面 5 条假设）。
修复采用独立 `trigger_score`（`policy/exploratory_policy.py`），**不放宽风险阈值**（尊重下方建议），
policy_version 1→2。boundary_explore 从 0 升至 1344，hard-risk share rate 保持 0.0（详见 Section 8）。
A-12 的 offline boundary sampler 已由 S2 的 `interaction_boundary_sampler.py` 提供。

早期 exploration policy 的 boundary branch 未触发。

需要检查：

* boundary band 是否太窄；
* safe exploit 是否提前消耗 share budget；
* candidate eligibility 是否过严；
* critic uncertainty 是否过低；
* support threshold 是否不合理。

建议不要简单放宽最终风险阈值，而是单独建立 offline boundary sampler。

### ~~Priority 5：增加真实环境前的压力测试 ✅已实现（S5）~~

~~<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#6b7280;color:#ffffff;font-weight:700">非必要细节</span>压力测试有助于增强实验可信度，但不是 `method.md` 第一版方法核心；`method.md` 明确不以构建新 benchmark 为主贡献。~~

在接入真实 LLM environment 前，至少应补充：

* 多种 tool regime； ✅
* 工具版本变化； ✅
* 权限差异； ✅
* agent role 变化； ✅
* stale procedure； ✅
* conflicting procedure； ✅
* redundant procedure； ✅
* incomplete procedure； ✅
* receiver capability mismatch。 ✅

---

## 11. 绝对不要破坏的 invariants

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>这些 invariants 是 `method.md` 的因果解释成立条件，必须保留。

以下约束是项目因果解释成立的前提。

### Payload isolation

未选中的 procedure payload 不得泄漏。

禁止将 `steps` 写入：

```text
candidate proposal
router trace
paired record
feature encoder
global graph state
unselected agent context
```

### Paired branch isolation

share / withhold branch 必须：

```text
same graph state snapshot
same environment snapshot
same memory snapshot
same candidate order
same selected prefix
same policy fingerprint
same node seed rule
```

仅 target memory exposure 可以被强制改变。

### Memory-store immutability during collection

同一 policy round 内：

```text
memory store revision 必须固定
```

禁止：

```text
paired evidence ingestion
execution evidence update
payload update
card update
memory creation
```

这些只能发生在 round finalized 后。

### Policy-specific estimand

不得混合不同 continuation policy 的 records 训练一个 critic。

必须保持：

```text
D0 -> C0 estimates tau^pi0
D1 -> C1 estimates tau^pi1
D2 -> C2 estimates tau^pi2
```

### Feature leakage prevention

下列字段不得进入 critic features：

```text
memory_id
payload version as categorical token
steps
payload
transfer_class
y_share
y_withhold
team_reward
scenario_family
environment_regime
prefix_structure_family
factor_combination_id
surface_variant_id
mechanism_group_id
branch label
```

---

## ~~12. 常用命令~~

~~<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#6b7280;color:#ffffff;font-weight:700">非必要细节</span>以下命令是工程复现说明，不是方法本身；保留但视为非方法必要内容。~~

### ~~测试~~

```bash
pytest -q
ruff check .
```

当前状态：

```text
pytest: 325 passed, 2 xfailed
ruff: All checks passed
```

基线 58 → A-01 交互编码 +3（61）→ S2 interaction-boundary 采样器 +3（64）
→ S3 audit 指标 +4（68）→ S4 boundary 修复 +3（71）→ S5 压力测试 +19（90）
→ S6 验收标准 +21（109 passed，2 xfailed）→ B-08 真实 LLM +36（145 passed，2 xfailed）
→ S7 全部实现 +156（301 passed，2 xfailed）→ S8/S9 invariants +24（325 passed，2 xfailed）。

存在 1 条 StarletteDeprecationWarning（来自 fastapi testclient），与项目功能无关。
之前的 3 条 sklearn FutureWarning（`multi_class="auto"`）已通过移除该参数解决（S8/T-20）。

### ~~生成 / 检查 paired data~~

```bash
python -m smtr.cli collect-counterfactual \
  --db data/smtr_memory.sqlite \
  --episodes 480 \
  --seed 29 \
  --top-k 6 \
  --scenario-design factorial \
  --factorial-balance stratified \
  --target-policy uniform \
  --prefix-mode stratified \
  --max-prefix-size 2 \
  --continuation-policy-manifest policies/pi2_explore.json \
  --round-id pi2 \
  --round-index 2 \
  --require-min-continuation-share-rate 0.10 \
  --require-max-continuation-share-rate 0.35 \
  --require-max-hard-risk-share-rate 0.01 \
  --output data/paired_records_pi2_v13.jsonl
```

```bash
python -m smtr.cli inspect-paired-records \
  --input data/paired_records_pi2_v13.jsonl \
  --show-prefix-distribution \
  --show-factor-coverage \
  --show-continuation-behavior
```

### ~~训练 critic~~

```bash
python -m smtr.cli train-transfer-critic \
  --input data/paired_records_pi2_v13.jsonl \
  --output checkpoints/critic_pi2.joblib \
  --seed 29 \
  --n-bootstrap 31 \
  --test-fraction 0.2 \
  --require-policy-fingerprint 18563e8540256d53ae44d91fbd9e847e48ebabbfa34dde3380bfa1f62e3ac87f
```

### ~~审计~~

```bash
python -m smtr.cli scan-transfer-feature-leakage \
  --input data/paired_records_pi2_v13.jsonl \
  --output outputs/feature_leakage_scan_pi2.json
```

```bash
python -m smtr.cli audit-feature-blocks \
  --input data/paired_records_pi2_v13.jsonl \
  --seed 29 \
  --n-bootstrap 31 \
  --split-suite compositional \
  --output outputs/feature_block_audit_pi2.json
```

```bash
python -m smtr.cli audit-prefix-sensitivity \
  --input data/paired_records_pi2_v13.jsonl \
  --checkpoint checkpoints/critic_pi2.joblib \
  --output outputs/prefix_sensitivity_pi2.json
```

```bash
python -m smtr.cli audit-candidate-substitution \
  --input data/paired_records_pi2_v13.jsonl \
  --checkpoint checkpoints/critic_pi2.joblib \
  --output outputs/candidate_substitution_pi2.json
```

### ~~S2 定向采集（interaction-boundary）~~

```bash
python -m smtr.cli collect-counterfactual \
  --db data/smtr_memory.sqlite \
  --episodes 480 \
  --seed 7 \
  --top-k 4 \
  --scenario-mix interaction \
  --target-policy scenario-designated \
  --prefix-mode interaction-boundary \
  --max-prefix-size 2 \
  --continuation-policy-manifest policies/pi2_explore.json \
  --boundary-critic-checkpoint checkpoints/critic_pi2_interaction.joblib \
  --output data/paired_records_pi2_s2_v14.jsonl \
  --round-id pi2_s2
```

其中 `--boundary-critic-checkpoint` 是可选的 A-07.4/5/6 critic 评分钩子；
省略则仅使用结构评分。

---

## 13. 不建议立刻做的事情

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>这些禁忌基本都对应 `method.md` 的非目标或因果识别前提。

不要立刻：

```text
- 把 critic_pi2 接入默认 production router
- 用当前 critic 宣称已经学到 prefix interaction
- 用 0.90 左右的 accuracy 宣称泛化成功
- ~~忽略 scenario split = 1.0 的异常~~（已通过 ToyEnvironment 增强解决，F1=0.732）
- 把 raw execution success 当作 transfer effect
- 混合 pi0/pi1/pi2 数据训练一个统一 critic
- 在 collection round 内写 paired evidence
- 让 candidate proposer 或 router 读取 payload steps
```

---

## 14. 建议的下一阶段验收标准

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#ea580c;color:#ffffff;font-weight:700">SMTR 必需 未实现/不完全</span>这些验收项是把研究原型升级为可支撑论文主张的必要条件；其中部分已通过，部分仍未完成。

在考虑 production sequential router 前，至少应达到：

```text
1. Prefix sensitivity direction accuracy 明显高于 0.50；
2. 能稳定识别 positive -> neutral 与 positive -> negative flip；
3. boundary exploration 不再为 0；
4. selected-set interaction encoder 显著优于 aggregate-only baseline；
5. scenario split 不再异常接近 1.0，或能明确解释其原因；
6. target family、environment regime、prefix family 均具有足够 label diversity；
7. feature leakage scanner 继续为 0 violations；
8. strict compositional OOD 下仍有合理性能；
9. high-risk continuation exposure rate 保持接近 0；
10. 默认 demo 仍未被 exploratory policy 污染。
```

**当前进展**（S6 验收标准测试全部执行，19 passed / 2 xfailed）：

- 第 1 项（prefix direction accuracy）：**已通过**（S4 后 0.64，delta correlation 0.821）。
- 第 2 项（稳定识别 flip）：**已通过**。pos→neu 1.0(9对), neg→neu 1.0(9对),
  **pos→neg 1.0(13对)**（ToyEnvironment 增强后新增 flip 场景与前缀记忆，覆盖缺口已修复）。
- 第 3 项（boundary exploration 不再为 0）：**已通过**（S4 后 1344）。
- 第 4 项（interaction encoder 优于 baseline）：**已通过**（full-model gain +0.2507，F1 0.901）。
- 第 5 项（scenario split 不接近 1.0）：**已通过**（ToyEnvironment 增强后 scenario_family F1=0.732，
  远低于 0.999 阈值；通过合并 flip 场景到基础场景、添加隐藏扰动机制实现）。
- 第 6 项（label diversity）：**已通过**（target 3类, regime 3类, prefix 3类, 4类 transfer class）。
- 第 7 项（feature leakage = 0）：**已通过**（0 violations）。
- 第 8 项（compositional OOD）：**已通过**（episode F1=0.901, factor F1=0.905, prefix F1=0.785）。
- 第 9 项（high-risk ≈ 0）：**已通过**（S4 重采后仍为 0.0）。
- 第 10 项（demo 未被污染）：**已通过**（默认 demo 无 frozen_continuation 决策）。

---

## ~~15. S7 实现状态~~

~~<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#6b7280;color:#ffffff;font-weight:700">非必要细节</span>S7 中若干工程扩展超出 `method.md` 第一版范围，尤其 off-policy correction、high-order group effects、meta-procedure、memory refinement、真实 LLM、delegation topology、stale propagation 等；保留为工程路线记录。~~

所有 S7 任务已实现：

```text
- [x] production sequential router (B-01)
  - src/smtr/router/sequential_router.py (17 tests)
- [x] runtime safety guard 与 fallback router (B-02)
  - src/smtr/router/safety_guard.py (22 tests)
- [x] ~~online policy refresh / active data acquisition (B-03)~~
  - src/smtr/policy/online_refresh.py (15 tests)
- [x] ~~off-policy correction (B-04)~~
  - src/smtr/router/off_policy_correction.py (25 tests)
- [x] ~~high-order group effects (B-05)~~
  - src/smtr/evaluation/group_effects.py (13 tests)
- [x] ~~meta-procedure composition (B-06)~~
  - src/smtr/memory/meta_procedure.py (18 tests)
- [x] ~~memory refinement / contradiction repair (B-07)~~
  - src/smtr/memory/refinement.py (12 tests)
- [x] ~~real LLM + real tool environment (B-08)~~
  - src/smtr/runtime/real_llm.py (9), tool_environment.py (19), api_server.py (8) = 36 tests
- [x] ~~real multi-agent delegation topology (B-09)~~
  - src/smtr/runtime/delegation_topology.py (20 tests)
- [x] ~~stale memory propagation experiments (B-10)~~
  - src/smtr/evaluation/stale_propagation.py (14 tests)
```

总测试数: 325 passed, 2 xfailed

## 16. S8/S9 实现状态

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#16a34a;color:#ffffff;font-weight:700">SMTR 必需 已实现</span>S8/S9 中的 invariants 测试必要；它们对应 payload isolation、paired branch isolation、temporal integrity、policy-specific estimand 和 feature leakage prevention。

S8（技术债）和 S9（invariants）测试已完成：

```text
- [x] ~~T-20 sklearn multi_class 替换（已确认：参数已从代码中移除）~~
- [x] T-21 Payload isolation（4 tests）
- [x] T-22 Paired branch isolation（3 tests）
- [x] T-23 Memory store immutability（3 tests）
- [x] T-24 Policy-specific estimand（2 tests）
- [x] T-25 Feature leakage prevention（8 tests）
- [x] Invariant cross-checks（4 tests）
```

测试文件：`tests/test_invariants.py`（24 tests）

## 17. 一句话总结

<span style="display:inline-block;padding:2px 7px;border-radius:4px;background-color:#ea580c;color:#ffffff;font-weight:700">SMTR 必需 不完全</span>总结准确，但应补充：production 默认路径、严格 LCB/UCB 决策、随机 permutation 测试与真实泛化仍未完全闭环。

当前项目已经完成了一个较完整的研究原型：

> 从 payload-isolated shared procedural memory、candidate proposal、paired counterfactual supervision、policy-aware continuation 到 four-outcome transfer critic 的端到端管线。

接手后最高优先级——强化 candidate–prefix interaction representation（A-01）、
定向采集 effect-flip 数据（S2），并用更严格的 interaction audit 验证——已经取得
实质性进展：

> critic 学习 τ(m|o,S) 的 prefix 方向正确性 0.50 → 0.54 → 0.65，明显高于随机。

但彻底结论（是否已经真正学到 interaction boundary）仍需继续推进：S3 改进审计指标、
S4 让 boundary exploration 真正覆盖边界、解决 candidate-substitution 审计覆盖率回归。
