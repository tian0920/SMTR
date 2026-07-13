# SMTR 消融实验结果（第二轮）

## 1. 实验设置

| Component | Detail |
|-----------|--------|
| **Git commit** | `69fccfb0221a` |
| **Workspace clean** | False |
| **Methods** | B0, B1-Top1, B1-Top3, B1-Matched, A1-NoSet, M0-Full |
| **Scenarios** | 9 counterfactual toy-task scenarios |
| **Episodes per scenario** | 40 (5 task seeds × 2 gen seeds × 4 trav seeds) |
| **Critic (M0-Full)** | `critic_pi3_v22` |
| **Critic (A1-NoSet)** | `critic_no_selected_set_v1` |
| **top_k** | 4 |
| **max_shares_per_invocation** | 3 |
| **M0 checkpoint SHA** | `9b357815ba2e7b3a...` |
| **A1 checkpoint SHA** | `3a096642f16e4588...` |
| **A1 uses_selected_set** | True |

## 2. 公平性检查

- **Split 验证**: A1 元数据匹配 = True
- **Note**: M0 metadata may differ due to versioning; A1 is canonical
- **M0/A1 same gate**: True
- **Rejection reason test**: 通过
- **Budget manifest source**: validation split (not test)

## 3. 主要结果

| Scenario | Method | Success | PosTR | NegTR | Avg Selected | Avg Selected/Ep |
|----------|--------|--------:|------:|------:|-------------:|----------------:|
| positive | B0 | 0.00 | — | — | 0.0 | 0.0 |
| positive | B1-Top1 | 1.00 | 1.00 | 0.00 | 1.0 | 1.0 |
| positive | B1-Top3 | 0.00 | 0.00 | 0.00 | 7.0 | 7.0 |
| positive | B1-Matched | 0.50 | 0.50 | 0.00 | 1.5 | 1.5 |
| positive | A1-NoSet | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 |
| positive | M0-Full | 0.20 | 0.20 | 0.00 | 1.4 | 1.4 |
| negative | B0 | 1.00 | — | — | 0.0 | 0.0 |
| negative | B1-Top1 | 0.00 | 0.00 | 1.00 | 1.0 | 1.0 |
| negative | B1-Top3 | 0.00 | 0.00 | 1.00 | 7.0 | 7.0 |
| negative | B1-Matched | 0.50 | 0.00 | 0.50 | 1.5 | 1.5 |
| negative | A1-NoSet | 0.00 | 0.00 | 1.00 | 0.0 | 0.0 |
| negative | M0-Full | 1.00 | 0.00 | 0.00 | 0.0 | 0.0 |
| neutral_success | B0 | 1.00 | — | — | 0.0 | 0.0 |
| neutral_success | B1-Top1 | 1.00 | 0.00 | 0.00 | 1.0 | 1.0 |
| neutral_success | B1-Top3 | 1.00 | 0.00 | 0.00 | 7.0 | 7.0 |
| neutral_success | B1-Matched | 1.00 | 0.00 | 0.00 | 1.5 | 1.5 |
| neutral_success | A1-NoSet | 0.00 | 0.00 | 1.00 | 0.0 | 0.0 |
| neutral_success | M0-Full | 1.00 | 0.00 | 0.00 | 1.6 | 1.6 |
| neutral_failure | B0 | 0.00 | — | — | 0.0 | 0.0 |
| neutral_failure | B1-Top1 | 0.00 | 0.00 | 0.00 | 1.0 | 1.0 |
| neutral_failure | B1-Top3 | 0.00 | 0.00 | 0.00 | 7.0 | 7.0 |
| neutral_failure | B1-Matched | 0.00 | 0.00 | 0.00 | 1.5 | 1.5 |
| neutral_failure | A1-NoSet | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 |
| neutral_failure | M0-Full | 0.00 | 0.00 | 0.00 | 1.8 | 1.8 |
| prefix_sensitive | B0 | 0.00 | — | — | 0.0 | 0.0 |
| prefix_sensitive | B1-Top1 | 1.00 | 1.00 | 0.00 | 1.0 | 1.0 |
| prefix_sensitive | B1-Top3 | 0.00 | 0.00 | 0.00 | 5.0 | 5.0 |
| prefix_sensitive | B1-Matched | 0.00 | 0.00 | 0.00 | 1.5 | 1.5 |
| prefix_sensitive | A1-NoSet | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 |
| prefix_sensitive | M0-Full | 0.00 | 0.00 | 0.00 | 2.4 | 2.4 |
| flip_pos_to_neg | B0 | 1.00 | — | — | 0.0 | 0.0 |
| flip_pos_to_neg | B1-Top1 | 1.00 | 0.00 | 0.00 | 3.0 | 3.0 |
| flip_pos_to_neg | B1-Top3 | 0.00 | 0.00 | 1.00 | 8.0 | 8.0 |
| flip_pos_to_neg | B1-Matched | 1.00 | 0.00 | 0.00 | 2.0 | 2.0 |
| flip_pos_to_neg | A1-NoSet | 0.00 | 0.00 | 1.00 | 0.0 | 0.0 |
| flip_pos_to_neg | M0-Full | 1.00 | 0.00 | 0.00 | 0.0 | 0.0 |
| flip_neg_to_pos | B0 | 0.00 | — | — | 0.0 | 0.0 |
| flip_neg_to_pos | B1-Top1 | 0.00 | 0.00 | 0.00 | 3.0 | 3.0 |
| flip_neg_to_pos | B1-Top3 | 0.00 | 0.00 | 0.00 | 8.0 | 8.0 |
| flip_neg_to_pos | B1-Matched | 0.00 | 0.00 | 0.00 | 2.0 | 2.0 |
| flip_neg_to_pos | A1-NoSet | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 |
| flip_neg_to_pos | M0-Full | 0.00 | 0.00 | 0.00 | 2.6 | 2.6 |
| flip_neu_to_neg | B0 | 0.00 | — | — | 0.0 | 0.0 |
| flip_neu_to_neg | B1-Top1 | 0.00 | 0.00 | 0.00 | 3.0 | 3.0 |
| flip_neu_to_neg | B1-Top3 | 0.00 | 0.00 | 0.00 | 8.0 | 8.0 |
| flip_neu_to_neg | B1-Matched | 0.00 | 0.00 | 0.00 | 2.0 | 2.0 |
| flip_neu_to_neg | A1-NoSet | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 |
| flip_neu_to_neg | M0-Full | 0.20 | 0.20 | 0.00 | 1.2 | 1.2 |
| flip_neu_to_pos | B0 | 1.00 | — | — | 0.0 | 0.0 |
| flip_neu_to_pos | B1-Top1 | 1.00 | 0.00 | 0.00 | 3.0 | 3.0 |
| flip_neu_to_pos | B1-Top3 | 0.00 | 0.00 | 1.00 | 8.0 | 8.0 |
| flip_neu_to_pos | B1-Matched | 1.00 | 0.00 | 0.00 | 2.0 | 2.0 |
| flip_neu_to_pos | A1-NoSet | 0.00 | 0.00 | 1.00 | 0.0 | 0.0 |
| flip_neu_to_pos | M0-Full | 0.80 | 0.00 | 0.20 | 1.2 | 1.2 |

### 宏平均

| Method | Avg Success | Avg PosTR | Avg NegTR | Avg Selected |
|--------|-----------:|----------:|----------:|-------------:|
| B0 | 0.444 | 0.000 | 0.000 | 0.0 |
| B1-Top1 | 0.556 | 0.222 | 0.111 | 1.9 |
| B1-Top3 | 0.111 | 0.000 | 0.333 | 7.2 |
| B1-Matched | 0.444 | 0.056 | 0.056 | 1.7 |
| A1-NoSet | 0.000 | 0.000 | 0.444 | 0.0 |
| M0-Full | 0.467 | 0.044 | 0.022 | 1.4 |

### 补充指标

| Scenario | Method | Neutral Success Rate | Neutral Failure Rate | All Withhold Rate | Runtime/Ep |
|----------|--------|---------------------:|---------------------:|------------------:|-----------:|
| positive | B0 | — | — | 1.00 | 0.015s |
| positive | B1-Top1 | 0.00 | 0.00 | 0.00 | 0.016s |
| positive | B1-Top3 | 0.00 | 1.00 | 0.00 | 0.015s |
| positive | B1-Matched | 0.00 | 0.50 | 0.00 | 0.015s |
| positive | A1-NoSet | 0.00 | 1.00 | 1.00 | 0.012s |
| positive | M0-Full | 0.00 | 0.80 | 0.60 | 0.051s |
| negative | B0 | — | — | 1.00 | 0.014s |
| negative | B1-Top1 | 0.00 | 0.00 | 0.00 | 0.014s |
| negative | B1-Top3 | 0.00 | 0.00 | 0.00 | 0.015s |
| negative | B1-Matched | 0.50 | 0.00 | 0.00 | 0.014s |
| negative | A1-NoSet | 0.00 | 0.00 | 1.00 | 0.012s |
| negative | M0-Full | 1.00 | 0.00 | 1.00 | 0.051s |
| neutral_success | B0 | — | — | 1.00 | 0.014s |
| neutral_success | B1-Top1 | 1.00 | 0.00 | 0.00 | 0.015s |
| neutral_success | B1-Top3 | 1.00 | 0.00 | 0.00 | 0.015s |
| neutral_success | B1-Matched | 1.00 | 0.00 | 0.00 | 0.014s |
| neutral_success | A1-NoSet | 0.00 | 0.00 | 1.00 | 0.012s |
| neutral_success | M0-Full | 1.00 | 0.00 | 0.00 | 0.052s |
| neutral_failure | B0 | — | — | 1.00 | 0.014s |
| neutral_failure | B1-Top1 | 0.00 | 1.00 | 0.00 | 0.015s |
| neutral_failure | B1-Top3 | 0.00 | 1.00 | 0.00 | 0.016s |
| neutral_failure | B1-Matched | 0.00 | 1.00 | 0.00 | 0.015s |
| neutral_failure | A1-NoSet | 0.00 | 1.00 | 1.00 | 0.013s |
| neutral_failure | M0-Full | 0.00 | 1.00 | 0.00 | 0.050s |
| prefix_sensitive | B0 | — | — | 1.00 | 0.014s |
| prefix_sensitive | B1-Top1 | 0.00 | 0.00 | 0.00 | 0.016s |
| prefix_sensitive | B1-Top3 | 0.00 | 1.00 | 0.00 | 0.016s |
| prefix_sensitive | B1-Matched | 0.00 | 1.00 | 0.00 | 0.015s |
| prefix_sensitive | A1-NoSet | 0.00 | 1.00 | 1.00 | 0.013s |
| prefix_sensitive | M0-Full | 0.00 | 1.00 | 0.00 | 0.054s |
| flip_pos_to_neg | B0 | — | — | 1.00 | 0.014s |
| flip_pos_to_neg | B1-Top1 | 1.00 | 0.00 | 0.00 | 0.015s |
| flip_pos_to_neg | B1-Top3 | 0.00 | 0.00 | 0.00 | 0.015s |
| flip_pos_to_neg | B1-Matched | 1.00 | 0.00 | 0.00 | 0.015s |
| flip_pos_to_neg | A1-NoSet | 0.00 | 0.00 | 1.00 | 0.012s |
| flip_pos_to_neg | M0-Full | 1.00 | 0.00 | 1.00 | 0.051s |
| flip_neg_to_pos | B0 | — | — | 1.00 | 0.014s |
| flip_neg_to_pos | B1-Top1 | 0.00 | 1.00 | 0.00 | 0.014s |
| flip_neg_to_pos | B1-Top3 | 0.00 | 1.00 | 0.00 | 0.015s |
| flip_neg_to_pos | B1-Matched | 0.00 | 1.00 | 0.00 | 0.014s |
| flip_neg_to_pos | A1-NoSet | 0.00 | 1.00 | 1.00 | 0.013s |
| flip_neg_to_pos | M0-Full | 0.00 | 1.00 | 0.60 | 0.050s |
| flip_neu_to_neg | B0 | — | — | 1.00 | 0.014s |
| flip_neu_to_neg | B1-Top1 | 0.00 | 1.00 | 0.00 | 0.015s |
| flip_neu_to_neg | B1-Top3 | 0.00 | 1.00 | 0.00 | 0.015s |
| flip_neu_to_neg | B1-Matched | 0.00 | 1.00 | 0.00 | 0.014s |
| flip_neu_to_neg | A1-NoSet | 0.00 | 1.00 | 1.00 | 0.013s |
| flip_neu_to_neg | M0-Full | 0.00 | 0.80 | 0.60 | 0.051s |
| flip_neu_to_pos | B0 | — | — | 1.00 | 0.015s |
| flip_neu_to_pos | B1-Top1 | 1.00 | 0.00 | 0.00 | 0.015s |
| flip_neu_to_pos | B1-Top3 | 0.00 | 0.00 | 0.00 | 0.016s |
| flip_neu_to_pos | B1-Matched | 1.00 | 0.00 | 0.00 | 0.015s |
| flip_neu_to_pos | A1-NoSet | 0.00 | 0.00 | 1.00 | 0.012s |
| flip_neu_to_pos | M0-Full | 0.80 | 0.00 | 0.60 | 0.051s |

## 4. 公平预算比较（配对组 Bootstrap 95% CI）

| Comparison | Success Diff | 95% CI | NegTR Diff | 95% CI | PosTR Diff | 95% CI | Avg Selected Diff | 95% CI |
|------------|-------------:|--------|-----------:|--------|-----------:|--------|------------------:|--------|
| M0-Full vs B1-Top1 | -0.089 | [-0.114, -0.067] | +0.000 | [+0.000, +0.000] | +0.000 | [+0.000, +0.000] | -0.538 | [-0.761, -0.287] |
| M0-Full vs B1-Top3 | +0.355 | [+0.332, +0.379] | +0.000 | [+0.000, +0.000] | +0.000 | [+0.000, +0.000] | -5.869 | [-6.122, -5.626] |
| M0-Full vs B1-Matched | +0.022 | [-0.000, +0.044] | +0.000 | [+0.000, +0.000] | +0.000 | [+0.000, +0.000] | -0.361 | [-0.657, -0.062] |
| M0-Full vs A1-NoSet | +0.467 | [+0.444, +0.492] | +0.000 | [+0.000, +0.000] | +0.000 | [+0.000, +0.000] | +1.354 | [+1.125, +1.604] |

**匹配预算结论**: M0-Full vs B1-Matched 选择数量差异 = -0.361 (95% CI [-0.657, -0.062]). 成功率差异 = +0.022.
M0 和 B1-Matched 表现相近——优势可能来自保守选择，而非迁移感知。

## 5. Selected-Set 消融（M0-Full vs A1-NoSet）

| Scenario | Metric | M0-Full | A1-NoSet | Delta |
|----------|--------|--------:|---------:|------:|
| prefix_sensitive | Success | 0.00 | 0.00 | +0.00 |
| prefix_sensitive | Positive Transfer | 0.00 | 0.00 | +0.00 |
| prefix_sensitive | Negative Transfer | 0.00 | 0.00 | +0.00 |
| flip_pos_to_neg | Success | 1.00 | 0.00 | +1.00 |
| flip_pos_to_neg | Positive Transfer | 0.00 | 0.00 | +0.00 |
| flip_pos_to_neg | Negative Transfer | 0.00 | 1.00 | -1.00 |
| flip_neg_to_pos | Success | 0.00 | 0.00 | +0.00 |
| flip_neg_to_pos | Positive Transfer | 0.00 | 0.00 | +0.00 |
| flip_neg_to_pos | Negative Transfer | 0.00 | 0.00 | +0.00 |
| flip_neu_to_neg | Success | 0.20 | 0.00 | +0.20 |
| flip_neu_to_neg | Positive Transfer | 0.20 | 0.00 | +0.20 |
| flip_neu_to_neg | Negative Transfer | 0.00 | 0.00 | +0.00 |
| flip_neu_to_pos | Success | 0.80 | 0.00 | +0.80 |
| flip_neu_to_pos | Positive Transfer | 0.00 | 0.00 | +0.00 |
| flip_neu_to_pos | Negative Transfer | 0.20 | 1.00 | -0.80 |

## 6. 前缀匹配对审计

| Scenario | N Paired | Delta-Tau Corr | Delta-Tau MAE | Direction Acc |
|----------|---------:|---------------:|--------------:|--------------:|
| prefix_sensitive | 80 | — | 1.000 | 0.000 |
| flip_pos_to_neg | 80 | — | 1.253 | 0.000 |
| flip_neg_to_pos | 80 | — | 0.775 | 1.000 |
| flip_neu_to_neg | 80 | — | 1.000 | 0.000 |
| flip_neu_to_pos | 80 | — | 1.000 | 0.000 |

### 翻转类型准确率

| Scenario | Accuracy |
|----------|---------:|
| flip_pos_to_neg | 0.800 |
| flip_neg_to_pos | 0.200 |
| flip_neu_to_neg | 1.000 |
| flip_neu_to_pos | 0.000 |

## 7. 瓶颈漏斗（Proposer → Router → Execution）

### positive

| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
|-------|--------:|--------:|--------:|--------:|--------:|--------:|
| ground_truth_opportunity | 80/100% | 80/100% | 80/100% | 80/100% | 320/100% | 320/100% |
| target_prefix_recalled | 80/100% | 80/100% | 80/100% | 80/100% | 0/0% | 320/100% |
| correct_traversal_order | 80/100% | 80/100% | 80/100% | 80/100% | 0/0% | 320/100% |
| correct_prefix_selected | 80/100% | 80/100% | 80/100% | 80/100% | 0/0% | 320/100% |
| target_correctly_routed | 0/0% | 80/100% | 80/100% | 0/0% | 0/0% | 128/40% |
| task_succeeds | 0/0% | 80/100% | 0/0% | 0/0% | 0/0% | 64/20% |

### negative

| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
|-------|--------:|--------:|--------:|--------:|--------:|--------:|
| ground_truth_opportunity | 80/100% | 80/100% | 80/100% | 80/100% | 320/100% | 320/100% |
| target_prefix_recalled | 80/100% | 80/100% | 80/100% | 80/100% | 0/0% | 320/100% |
| correct_traversal_order | 80/100% | 80/100% | 80/100% | 80/100% | 0/0% | 320/100% |
| correct_prefix_selected | 80/100% | 80/100% | 80/100% | 80/100% | 0/0% | 320/100% |
| target_correctly_routed | 80/100% | 0/0% | 0/0% | 80/100% | 0/0% | 320/100% |
| task_succeeds | 80/100% | 0/0% | 0/0% | 40/50% | 0/0% | 320/100% |

### neutral_success

| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
|-------|--------:|--------:|--------:|--------:|--------:|--------:|
| ground_truth_opportunity | 80/100% | 80/100% | 80/100% | 80/100% | 320/100% | 320/100% |
| target_prefix_recalled | 80/100% | 80/100% | 80/100% | 80/100% | 0/0% | 320/100% |
| correct_traversal_order | 80/100% | 80/100% | 80/100% | 80/100% | 0/0% | 320/100% |
| correct_prefix_selected | 80/100% | 80/100% | 80/100% | 80/100% | 0/0% | 320/100% |
| target_correctly_routed | 80/100% | 80/100% | 80/100% | 80/100% | 0/0% | 320/100% |
| task_succeeds | 80/100% | 80/100% | 80/100% | 80/100% | 0/0% | 320/100% |

### neutral_failure

| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
|-------|--------:|--------:|--------:|--------:|--------:|--------:|
| ground_truth_opportunity | 80/100% | 80/100% | 80/100% | 80/100% | 320/100% | 320/100% |
| target_prefix_recalled | 80/100% | 80/100% | 80/100% | 80/100% | 0/0% | 320/100% |
| correct_traversal_order | 80/100% | 80/100% | 80/100% | 80/100% | 0/0% | 320/100% |
| correct_prefix_selected | 80/100% | 80/100% | 80/100% | 80/100% | 0/0% | 320/100% |
| target_correctly_routed | 80/100% | 80/100% | 80/100% | 80/100% | 0/0% | 320/100% |
| task_succeeds | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |

### prefix_sensitive

| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
|-------|--------:|--------:|--------:|--------:|--------:|--------:|
| ground_truth_opportunity | 80/100% | 80/100% | 80/100% | 80/100% | 320/100% | 320/100% |
| target_prefix_recalled | 80/100% | 80/100% | 80/100% | 80/100% | 0/0% | 320/100% |
| correct_traversal_order | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| correct_prefix_selected | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| target_correctly_routed | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| task_succeeds | 0/0% | 80/100% | 0/0% | 0/0% | 0/0% | 0/0% |

### flip_pos_to_neg

| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
|-------|--------:|--------:|--------:|--------:|--------:|--------:|
| ground_truth_opportunity | 80/100% | 80/100% | 80/100% | 80/100% | 320/100% | 320/100% |
| target_prefix_recalled | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| correct_traversal_order | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| correct_prefix_selected | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| target_correctly_routed | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| task_succeeds | 80/100% | 80/100% | 0/0% | 80/100% | 0/0% | 320/100% |

### flip_neg_to_pos

| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
|-------|--------:|--------:|--------:|--------:|--------:|--------:|
| ground_truth_opportunity | 80/100% | 80/100% | 80/100% | 80/100% | 320/100% | 320/100% |
| target_prefix_recalled | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| correct_traversal_order | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| correct_prefix_selected | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| target_correctly_routed | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| task_succeeds | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |

### flip_neu_to_neg

| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
|-------|--------:|--------:|--------:|--------:|--------:|--------:|
| ground_truth_opportunity | 80/100% | 80/100% | 80/100% | 80/100% | 320/100% | 320/100% |
| target_prefix_recalled | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| correct_traversal_order | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| correct_prefix_selected | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| target_correctly_routed | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| task_succeeds | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 64/20% |

### flip_neu_to_pos

| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
|-------|--------:|--------:|--------:|--------:|--------:|--------:|
| ground_truth_opportunity | 80/100% | 80/100% | 80/100% | 80/100% | 320/100% | 320/100% |
| target_prefix_recalled | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| correct_traversal_order | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| correct_prefix_selected | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| target_correctly_routed | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |
| task_succeeds | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% | 0/0% |

## 8. 拒绝原因分析

### 逐方法比例

| Scenario | Method | Shared | τ_LCB | Neg Risk | Low Support | Budget | Other | Sum |
|----------|--------|-------:|------:|---------:|------------:|-------:|------:|----:|
| positive | M0-Full | 0.200 | 0.333 | 0.450 | 0.000 | 0.017 | 0.000 | 1.000 |
| positive | A1-NoSet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| negative | M0-Full | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| negative | A1-NoSet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| neutral_success | M0-Full | 0.300 | 0.700 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| neutral_success | A1-NoSet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| neutral_failure | M0-Full | 0.300 | 0.683 | 0.000 | 0.000 | 0.017 | 0.000 | 1.000 |
| neutral_failure | A1-NoSet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| prefix_sensitive | M0-Full | 0.500 | 0.483 | 0.000 | 0.000 | 0.017 | 0.000 | 1.000 |
| prefix_sensitive | A1-NoSet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| flip_pos_to_neg | M0-Full | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| flip_pos_to_neg | A1-NoSet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| flip_neg_to_pos | M0-Full | 0.267 | 0.317 | 0.367 | 0.000 | 0.050 | 0.000 | 1.000 |
| flip_neg_to_pos | A1-NoSet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| flip_neu_to_neg | M0-Full | 0.133 | 0.733 | 0.117 | 0.000 | 0.017 | 0.000 | 1.000 |
| flip_neu_to_neg | A1-NoSet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| flip_neu_to_pos | M0-Full | 0.133 | 0.733 | 0.117 | 0.000 | 0.017 | 0.000 | 1.000 |
| flip_neu_to_pos | A1-NoSet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### 匹配不一致案例

- A1 share, M0 withhold: **0** cases
- A1 withhold, M0 share: **0** cases

## 9. 代表性失败案例

负迁移 episode 总数: 0


## 10. 谨慎结论

### M0 vs B1-Matched（迁移感知路由的价值）
- M0-Full 和 B1-Matched 表现相近 (diff = +0.022).
- 此前的优势可能来自保守选择，而非迁移感知。

### M0 vs A1-NoSet（Selected-Set 条件化的价值）
- M0-Full 优于 A1-NoSet +0.467.
- 优势集中在前缀/翻转场景中——支持 selected-set 条件化。

### 瓶颈诊断
- **Proposer 召回率低** (avg 0.33). 优先级：改进 proposer，而非 critic。
- 路由器正召回率足够 (0.60).

### 尚未解决的限制

- Toy environment: results may not generalize to real multi-agent domains.
- A1 critic trained with potentially different split than M0 (versioning artifact).
- Flip scenarios test encoder robustness, not routing per se.
- 40 episodes per scenario may have wide CIs for rare events.
# invalidated_by_evaluation_refactor

本报告由旧的 episode 级 schema 和旧 prefix matched-pair 分析生成，不再作为正式机制验证结果使用。
