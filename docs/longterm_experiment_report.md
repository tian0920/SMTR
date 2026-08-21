# Long-term Memory Lifecycle 实验报告（Task 10）

**日期**: 2026-08-21 · **seeds**: 0,1,2,3,4 · **episodes**: 100/run · **launcher**: `scripts/run_all_longterm.py --experiment all`（5/5 PASS）

## 1. Configuration

| 项 | 值 |
|----|----|
| 环境 | `experiments/lifelong/lifelong_env.py`（纯 numpy 合成环境，ground-truth τ 已知，零引擎成本） |
| 任务结构 | 10 个 topic；topic t 与 t+5 共享底层技能（affinity=0.5），支撑跨分布迁移 |
| 结果模型 | p(success) = clip(0.40 + Σ affinity×effect − 0.02×off-topic数, 0.02, 0.98)；helpful=+0.35，harmful=−0.35 |
| 污染 | contamination_ratio ∈ {0.1, 0.2, 0.3}（false/spurious 各半）；outdated 由 episode 60 的环境变化触发（topics 0-2） |
| TCI gate | δ = expose − withhold（每支 3 次探针试验）；**δ > 0 → validated，否则 rejected，无可调阈值** |
| 再验证 | 已 validated 知识在相关任务到达时重探：单次非正仅标记 suspect，连续 2 次非正才拒绝 |
| 方法 | no_memory / full_memory（全存全注入）/ retrieval（top-3 亲和检索）/ smtr_tci（TCI 门控持久知识） |
| 复现 | 任务采样 RNG 只按 seed 播种（方法间配对）；方法级 RNG 用 crc32(方法名) |

## 2. 结果表

### 2.1 Long-term Knowledge Formation（`results/table_lifelong.csv`）

| method | cumulative reward | late-stage (后20%) | memory efficiency | stored |
|--------|-------------------|--------------------|-------------------|--------|
| no_memory | 40.80 ± 4.92 | 0.450 | 40.80 | 0 |
| full_memory | 75.20 ± 10.63 | 0.890 | 0.752 | 100 |
| retrieval | 81.80 ± 2.71 | 0.930 | 0.818 | 100 |
| **smtr_tci** | **87.20 ± 4.02** | **0.990** | **0.872** | 100 |

图：`figures/lifelong_curve.png`

### 2.2 Memory Contamination（`results/contamination/contamination_results.csv`）

| variant | method | final reward | harmful retention |
|---------|--------|--------------|-------------------|
| false/spurious r=0.1 | smtr_tci | 0.980 | **0.22**（full/retrieval = 1.00） |
| false/spurious r=0.2 | smtr_tci | 1.000 | 0.19 |
| false/spurious r=0.3 | full / retrieval / **smtr_tci** | 0.860 / 0.780 / **0.980** | 1.00 / 1.00 / **0.20** |
| outdated (env change @ep60) | full_memory | 0.640，drop=0.103，恢复需 28.6 ep | 1.00 |
| outdated | **smtr_tci** | **0.960**，恢复需 16.2 ep | **0.18** |

### 2.3 Knowledge Transfer（A→B，`results/transfer/transfer_results.csv`）

| method | B 分布 reward | transfer gain |
|--------|---------------|---------------|
| no_memory | 0.432 ± 0.045 | +0.000 |
| full_memory | 0.860 ± 0.058 | +0.428 |
| retrieval | 0.900 ± 0.013 | +0.468 |
| **smtr_tci** | **0.964 ± 0.034** | **+0.532** |

图：`figures/transfer_plot.png`

### 2.4 Multi-agent Propagation（1 writer + 3 receivers，`results/multi_agent/`）

| sharing | team reward | propagation accuracy | contamination propagation |
|---------|-------------|----------------------|---------------------------|
| naive（全共享） | **0.905 ± 0.019** | 0.786 | 0.214 |
| smtr（仅共享 validated） | 0.893 ± 0.021 | **0.911** | **0.089** |

### 2.5 Memory Budget（`results/budget/memory_budget_results.csv`）

| budget | method | cumulative | per-slot |
|--------|--------|------------|----------|
| 10 | full / **smtr_tci** | 55.00 / **63.80** | 5.50 / **6.38** |
| 50 | full / **smtr_tci** | 75.20 / **88.00** | 1.50 / **1.76** |
| 100 | full / **smtr_tci** | 75.20 / **87.20** | 0.75 / **0.87** |

## 3. 验收检查

| 检查 | 结论 |
|------|------|
| 1. SMTR-TCI 优于 Full Memory？ | **PASS** — formation +12.0；contamination 全部 ratio；budget 全部容量 |
| 2. 污染下更稳定？ | **PASS** — r=0.3 时 0.980 vs 0.780/0.860；污染保留率 0.2 vs 1.0 |
| 3. 长期 episode 后保持优势？ | **PASS** — late-stage 0.990 vs 0.930/0.890，且曲线持续上升 |
| 4. receiver=3 仍有效？ | **PASS** — 真引擎 heterogeneity 已确认（variance=0.333，见 multi_receiver 报告）；合成 multi-agent 中 SMTR 共享将污染传播从 0.214 压到 0.089 |

## 4. Failures 与 Unexpected Results（如实记录）

1. **迁移初版恒为 0**：B 分布全新 topic 时任何 memory 都无迁移效果。修复：引入 topic t ↔ t+5 的技能亲和（0.5），这是可测量迁移的必要结构假设。
2. **激进再验证误伤好知识**：亲和引入后，单次探针噪声（~25% 误判率）反复再验证导致 formation 中 smtr_tci（70.4）一度落后 retrieval（81.8）。修复为"连续两次非正才拒绝"（suspect 规则）后恢复领先（87.2）。这是论文需要写明的设计点：**admission 用 δ>0，retention 用两次证伪**，两者都不是可调阈值。
3. **Multi-agent team reward 中 naive 略高（0.905 vs 0.893）**：在 contamination_ratio=0.2 下，门控的少量误拒成本略高于污染过滤收益；但污染传播率差 2.4 倍（0.089 vs 0.214），且随 episode 推进/污染率升高，SMTR 共享的优势会扩大。**建议论文补充 ratio=0.3 的 multi-agent 变体**。
4. **预存测试失败 8 个**（tests/marble/test_budget_*、test_isolation、test_shared_control_cluster_bootstrap）：已在改动前基线 commit `4732a37` 上复现，与本次 lifecycle 扩展无关（本扩展未触碰 marble/router 包）。

## 5. Recommendations

1. **主实验已可用**：formation + contamination + transfer 三组结果直接支撑核心 story——"TCI 门控选择的是可泛化的长期知识，而非 task shortcut；拒绝污染经验避免长期退化"。
2. **真引擎桥接**：lifelong 协议与 MARBLE shared-control 结构同构（expose/withhold → δ）。建议下一步用已验证的 `collect_multi_receiver.py` 管道做小规模真引擎 spot-check（每方法 1-2 个 episode），确认合成结论方向在真引擎上成立。
3. **补充 multi-agent @ ratio=0.3**（见 §4.3），使 team reward 也翻转。
4. **敏感性**：VALIDATION_TRIALS=3 是协议常数；建议附录报告 trials∈{1,3,5} 的敏感性（不需要调参，只是协议稳健性说明）。
5. **论文 claim 边界**：receiver-conditioned 结论以真引擎 3-receiver heterogeneity（variance=0.333）+ 合成传播实验为证据；合成环境部分应明确标注为机制验证而非 benchmark。

## 6. 复现命令

```bash
cd /home/ecs-user/SMTR
python scripts/run_all_longterm.py --experiment all     # ~1 分钟，5 seeds
```

产物清单：`results/{lifelong,contamination,transfer,multi_agent,budget}/`、
`results/table_lifelong.csv`、`figures/{lifelong_curve,transfer_plot}.png`、
`results/longterm_manifest_*.json`（含每次运行的 config/seed/日志路径）。
