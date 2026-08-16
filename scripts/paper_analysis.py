#!/usr/bin/env python3
"""Paper experiment analysis: Exp1 main results, Exp2 anatomy, Exp4 ablation."""
import json
import random
import math
from collections import Counter, defaultdict
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────
all_records = []
with open("artifacts/marble/outputs/q30b_full_resume/paired_records.jsonl") as f:
    for line in f:
        if line.strip():
            all_records.append(json.loads(line))

# Filter to valid records only (real MARBLE engine executed)
records = [r for r in all_records if r.get("valid", False)]

# Load old eval traces (38-task proof-of-concept)
with open("artifacts/marble/outputs/q30b_full/eval_full/traces.json") as f:
    old_traces = json.load(f)
with open("artifacts/marble/outputs/q30b_full/eval_full/result_table.json") as f:
    old_results = json.load(f)

print(f"Total records: {len(all_records)} (valid: {len(records)})")
print(f"Tasks: {len(set(r['task_id'] for r in records))}")

# ── Helper functions ───────────────────────────────────────────────────
def get_outcomes(rec):
    """Return (share_success, withhold_success) as ints (0/1)."""
    s = int(bool(rec.get("share", {}).get("team_success", False)))
    w = int(bool(rec.get("withhold", {}).get("team_success", False)))
    return (s, w)

def label_of(rec):
    return rec.get("label", "neutral_failure")

def edge_key(rec):
    return (rec["task_id"], rec["receiver_agent_id"], rec["candidate_memory_id"])

def task_receiver_key(rec):
    return (rec["task_id"], rec["receiver_agent_id"])

# ── Aggregate per edge ────────────────────────────────────────────────
edge_data = defaultdict(lambda: {
    "share_outcomes": [], "withhold_outcomes": [],
    "labels": [], "seeds": [], "task_id": None, "memory_id": None
})

for r in records:
    ek = edge_key(r)
    s, w = get_outcomes(r)
    edge_data[ek]["share_outcomes"].append(s)
    edge_data[ek]["withhold_outcomes"].append(w)
    edge_data[ek]["labels"].append(label_of(r))
    edge_data[ek]["seeds"].append(r.get("generation_seed", 0))
    edge_data[ek]["task_id"] = r["task_id"]
    edge_data[ek]["memory_id"] = r["candidate_memory_id"]

# Per-edge aggregated metrics
edge_agg = {}
for ek, d in edge_data.items():
    share_rate = sum(d["share_outcomes"]) / len(d["share_outcomes"])
    withhold_rate = sum(d["withhold_outcomes"]) / len(d["withhold_outcomes"])
    # Majority label
    label_counts = Counter(d["labels"])
    majority_label = label_counts.most_common(1)[0][0]
    edge_agg[ek] = {
        "share_rate": share_rate,
        "withhold_rate": withhold_rate,
        "majority_label": majority_label,
        "n_seeds": len(d["seeds"]),
        "task_id": d["task_id"],
        "memory_id": d["memory_id"],
    }

# ── Group edges by task-receiver ─────────────────────────────────────
tr_edges = defaultdict(list)
for ek, agg in edge_agg.items():
    trk = (agg["task_id"], ek[1])  # (task_id, receiver_agent_id)
    tr_edges[trk].append((ek, agg))

# ═══════════════════════════════════════════════════════════════════════
# EXP 2: Transfer Anatomy
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP 2: TRANSFER ANATOMY")
print("="*60)

# 2a. Four-Quadrant Distribution
print("\n── 2a. Four-Quadrant Distribution ──")
label_counts = Counter(r.get("label", "?") for r in records)
total = len(records)
q_map = {
    "neutral_failure": ("q00", "Neutral Failure"),
    "negative_transfer": ("q01", "Negative Transfer"),
    "positive_transfer": ("q10", "Positive Transfer"),
    "neutral_success": ("q11", "Neutral Success"),
}
quad_dist = []
for label, (qcode, qname) in q_map.items():
    c = label_counts.get(label, 0)
    frac = c / total * 100
    quad_dist.append({"quadrant": qcode, "label": label, "count": c, "fraction": round(frac, 1)})
    print(f"  {qcode} ({qname}): {c:4d} ({frac:5.1f}%)")

# Edge-level label (majority vote)
edge_labels = Counter(a["majority_label"] for a in edge_agg.values())
print(f"\n  Edge-level (majority vote):")
for label, (qcode, qname) in q_map.items():
    c = edge_labels.get(label, 0)
    frac = c / len(edge_agg) * 100
    print(f"  {qcode} ({qname}): {c:4d} edges ({frac:5.1f}%)")

# 2b. Oracle Ceiling Analysis
print("\n── 2b. Oracle Ceiling Analysis ──")

# For each (task, receiver) episode, compute oracle-optimal policy
oracle_episodes = []
b0_episodes = []
semantic_episodes = []

for trk, edges in tr_edges.items():
    # Withhold outcome (same for all candidates in same episode)
    withhold_success = edges[0][1]["withhold_rate"]
    
    # Oracle: pick best candidate or withhold
    best_share = max(e[1]["share_rate"] for e in edges)
    oracle_success = max(best_share, withhold_success)
    
    # b0: always withhold
    b0_success = withhold_success
    
    # semantic_top1: average across all candidates (since semantic scores are tied)
    avg_share = sum(e[1]["share_rate"] for e in edges) / len(edges)
    semantic_success = avg_share
    
    oracle_episodes.append({"task_id": trk[0], "success": oracle_success})
    b0_episodes.append({"task_id": trk[0], "success": b0_success})
    semantic_episodes.append({"task_id": trk[0], "success": semantic_success})

oracle_rate = sum(e["success"] for e in oracle_episodes) / len(oracle_episodes)
b0_rate = sum(e["success"] for e in b0_episodes) / len(b0_episodes)
semantic_rate = sum(e["success"] for e in semantic_episodes) / len(semantic_episodes)

print(f"  Oracle ceiling: {oracle_rate:.4f}")
print(f"  b0_no_memory:   {b0_rate:.4f}")
print(f"  semantic_top1:  {semantic_rate:.4f}")
print(f"  Oracle gap (SMTR potential): {oracle_rate - b0_rate:.4f}")

# 2c. Difficulty Stratification
print("\n── 2c. Difficulty Stratification ──")

# Control success rate = withhold success rate per task
task_control = defaultdict(list)
for trk, edges in tr_edges.items():
    task_id = trk[0]
    wh = edges[0][1]["withhold_rate"]
    task_control[task_id].append(wh)

task_difficulty = {}
for task_id, rates in task_control.items():
    avg_control = sum(rates) / len(rates)
    if avg_control > 0.667:
        difficulty = "easy"
    elif avg_control > 0.333:
        difficulty = "medium"
    else:
        difficulty = "hard"
    task_difficulty[task_id] = {
        "control_rate": avg_control,
        "difficulty": difficulty,
    }

# Compute per-difficulty metrics with bootstrap CIs
for diff in ["easy", "medium", "hard"]:
    diff_tasks = [t for t, d in task_difficulty.items() if d["difficulty"] == diff]
    diff_episodes_oracle = [e for e in oracle_episodes if e["task_id"] in diff_tasks]
    diff_episodes_b0 = [e for e in b0_episodes if e["task_id"] in diff_tasks]
    diff_episodes_sem = [e for e in semantic_episodes if e["task_id"] in diff_tasks]
    
    if not diff_episodes_b0:
        continue
    
    o_rate = sum(e["success"] for e in diff_episodes_oracle) / len(diff_episodes_oracle)
    b_rate = sum(e["success"] for e in diff_episodes_b0) / len(diff_episodes_b0)
    s_rate = sum(e["success"] for e in diff_episodes_sem) / len(diff_episodes_sem)
    
    # Bootstrap CI for oracle gap
    oracle_by_t = {e["task_id"]: e["success"] for e in diff_episodes_oracle}
    b0_by_t = {e["task_id"]: e["success"] for e in diff_episodes_b0}
    random.seed(42)
    boot_gaps = []
    for _ in range(1000):
        sampled = [random.choice(diff_tasks) for _ in range(len(diff_tasks))]
        g = sum(oracle_by_t[t] - b0_by_t[t] for t in sampled) / len(sampled)
        boot_gaps.append(g)
    boot_gaps.sort()
    gap_ci_lo = boot_gaps[int(0.025 * len(boot_gaps))]
    gap_ci_hi = boot_gaps[int(0.975 * len(boot_gaps))]
    gap_pval = sum(1 for g in boot_gaps if g <= 0) / len(boot_gaps)
    
    print(f"  {diff:8s} (n={len(diff_tasks):2d} tasks): "
          f"b0={b_rate:.3f}, semantic={s_rate:.3f}, oracle={o_rate:.3f}, "
          f"gap={o_rate-b_rate:+.3f} [{gap_ci_lo:+.3f},{gap_ci_hi:+.3f}] p={gap_pval:.3f}")

# ═══════════════════════════════════════════════════════════════════════
# EXP 4: Ablation Study
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP 4: ABLATION STUDY")
print("="*60)

# Compute per-episode outcomes for different policies
# For each (task, receiver), we need:
# - withhold outcome (same for all)
# - best share outcome among candidates

# First, compute semantic_top1 more carefully:
# Pick the candidate with best average share outcome (since semantic scores are tied)
# This is actually the "best-candidate" policy, not semantic

# For semantic_top1 simulation: pick a random candidate (since all have equal semantic score)
random.seed(42)

def compute_policy_rate(episodes_data):
    """Compute mean success rate from episode data."""
    if not episodes_data:
        return 0.0
    return sum(e["success"] for e in episodes_data) / len(episodes_data)

# Oracle gate: for each episode, pick the best candidate;
# share only if that candidate's share_rate > withhold_rate
oracle_gate_episodes = []
oracle_gate_neg_exposed = 0
oracle_gate_neg_total = 0

for trk, edges in tr_edges.items():
    withhold_success = edges[0][1]["withhold_rate"]
    # Find best share candidate
    best_edge, best_agg = max(edges, key=lambda x: x[1]["share_rate"])
    
    # Track negative transfer exposure
    for ek, agg in edges:
        if agg["majority_label"] == "negative_transfer":
            oracle_gate_neg_total += 1
    
    if best_agg["share_rate"] > withhold_success:
        oracle_gate_episodes.append({"task_id": trk[0], "success": best_agg["share_rate"], "action": "share"})
        # Check if chosen candidate is negative transfer
        if best_agg["majority_label"] == "negative_transfer":
            oracle_gate_neg_exposed += 1
    else:
        oracle_gate_episodes.append({"task_id": trk[0], "success": withhold_success, "action": "withhold"})

oracle_gate_rate = compute_policy_rate(oracle_gate_episodes)
oracle_share_count = sum(1 for e in oracle_gate_episodes if e["action"] == "share")
oracle_share_rate = oracle_share_count / len(oracle_gate_episodes)
oracle_neg_exposure = oracle_gate_neg_exposed / max(oracle_gate_neg_total, 1)
print(f"\n  oracle_gate: {oracle_gate_rate:.4f} (share_rate={oracle_share_rate:.3f})")
print(f"  oracle_gate neg transfer exposure: {oracle_neg_exposure:.3f} ({oracle_gate_neg_exposed}/{oracle_gate_neg_total})")

# Semantic neg transfer exposure: always shares -> 100% exposure
semantic_neg_total = sum(1 for a in edge_agg.values() if a["majority_label"] == "negative_transfer")
print(f"  semantic neg transfer exposure: 1.000 (all {semantic_neg_total} neg edges exposed)")

# Random gate: semantic picks a candidate, then randomly withhold with prob p
# Test multiple p values
# Estimate SMTR withhold rate from old eval: ~83% withheld
print("\n  Random gate ablation (varying p):")
random_results = {}
for p in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
    n_trials = 200
    trial_rates = []
    for trial in range(n_trials):
        episode_successes = []
        for trk, edges in tr_edges.items():
            withhold_success = edges[0][1]["withhold_rate"]
            # Pick first candidate (semantic top-1 among equals)
            share_success = edges[0][1]["share_rate"]
            if random.random() > p:
                episode_successes.append(share_success)
            else:
                episode_successes.append(withhold_success)
        trial_rates.append(sum(episode_successes) / len(episode_successes))
    mean_rate = sum(trial_rates) / len(trial_rates)
    random_results[f"p={p:.1f}"] = mean_rate
    print(f"    p={p:.1f} (withhold prob): {mean_rate:.4f}")

# Optimal random gate (p that maximizes success)
best_p = max(random_results, key=random_results.get)
print(f"\n  Best random gate: {best_p} -> {random_results[best_p]:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# EXP 1: Main Results (combining old eval + oracle analysis)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP 1: MAIN RESULTS TABLE")
print("="*60)

# Old eval results (38 tasks, proof-of-concept)
print("\n── Old eval (38 tasks, proof-of-concept) ──")
for m in old_results:
    method = m["method"]
    sr = m.get("paired_policy_success_rate", "?")
    neg = m.get("negative_transfer_exposure_rate", "?")
    share = m.get("share_rate", "?")
    print(f"  {method:25s}: success={sr}, neg_transfer={neg}, share_rate={share}")

# New analysis (70 tasks, full dataset)
print("\n── Full analysis (70 tasks, valid records) ──")
print(f"  {'Method':25s} {'Success':>8} {'Share%':>8} {'NegTransfer':>12}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*12}")
print(f"  {'b0_no_memory':25s} {b0_rate:8.4f} {0.0:8.3f} {0.0:12.3f}")
print(f"  {'semantic_top1':25s} {semantic_rate:8.4f} {1.0:8.3f} {1.0:12.3f}")
print(f"  {'oracle_gate':25s} {oracle_gate_rate:8.4f} {oracle_share_rate:8.3f} {oracle_neg_exposure:12.3f}")
print(f"  {'random_gate(p=0.5)':25s} {random_results['p=0.5']:8.4f} {0.5:8.3f} {'-':>12}")
best_p_val = random_results[best_p]
print(f"  {best_p.replace('p=','random_gate(')+')':25s} {best_p_val:8.4f} {1-float(best_p.split('=')[1]):8.3f} {'-':>12}")

# ── Save neg transfer exposure rates ──
neg_transfer_edges = [(ek, agg) for ek, agg in edge_agg.items() if agg["majority_label"] == "negative_transfer"]
print(f"\n  Negative transfer edges: {len(neg_transfer_edges)}")

# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STATISTICAL ANALYSIS")
print("="*60)

# Task-level bootstrap CI for oracle vs b0
def cluster_bootstrap_ci(episodes, n_bootstrap=1000, seed=42):
    """Cluster bootstrap CI grouped by task_id."""
    random.seed(seed)
    # Group by task
    task_data = defaultdict(list)
    for e in episodes:
        task_data[e["task_id"]].append(e["success"])
    
    task_ids = sorted(task_data.keys())
    n = len(task_ids)
    
    boot_means = []
    for _ in range(n_bootstrap):
        sampled_tasks = [random.choice(task_ids) for _ in range(n)]
        boot_values = []
        for t in sampled_tasks:
            boot_values.extend(task_data[t])
        boot_means.append(sum(boot_values) / len(boot_values))
    
    boot_means.sort()
    ci_lower = boot_means[int(0.025 * n_bootstrap)]
    ci_upper = boot_means[int(0.975 * n_bootstrap)]
    point = sum(e["success"] for e in episodes) / len(episodes)
    return point, ci_lower, ci_upper

# Compute CIs
for name, episodes in [("b0", b0_episodes), ("semantic", semantic_episodes), 
                         ("oracle", oracle_episodes), ("oracle_gate", oracle_gate_episodes)]:
    pt, lo, hi = cluster_bootstrap_ci(episodes)
    print(f"  {name:20s}: {pt:.4f} [{lo:.4f}, {hi:.4f}]")

# Paired difference tests
print("\n── Paired difference tests ──")
random.seed(42)
task_ids_all = sorted(set(e["task_id"] for e in b0_episodes))

# Build per-task success lookup for each method
oracle_by_task = {e["task_id"]: e["success"] for e in oracle_gate_episodes}
b0_by_task = {e["task_id"]: e["success"] for e in b0_episodes}
sem_by_task = {e["task_id"]: e["success"] for e in semantic_episodes}

def paired_diff_test(method_a, method_b, name_a, name_b, task_ids, n_boot=10000):
    """Bootstrap paired difference test."""
    random.seed(42)
    observed = sum(method_a[t] - method_b[t] for t in task_ids) / len(task_ids)
    boot_diffs = []
    for _ in range(n_boot):
        sampled = [random.choice(task_ids) for _ in range(len(task_ids))]
        d = sum(method_a[t] - method_b[t] for t in sampled) / len(sampled)
        boot_diffs.append(d)
    boot_diffs.sort()
    p_value = sum(1 for d in boot_diffs if d <= 0) / n_boot
    ci_lo = boot_diffs[int(0.025 * n_boot)]
    ci_hi = boot_diffs[int(0.975 * n_boot)]
    print(f"  {name_a} - {name_b}: diff={observed:+.4f}, p={p_value:.4f}, 95%CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]")
    return {"diff": observed, "p_value": p_value, "ci_95": [ci_lo, ci_hi]}

stat_tests = {}
stat_tests["oracle_vs_b0"] = paired_diff_test(oracle_by_task, b0_by_task, "oracle_gate", "b0", task_ids_all)
stat_tests["oracle_vs_semantic"] = paired_diff_test(oracle_by_task, sem_by_task, "oracle_gate", "semantic", task_ids_all)
stat_tests["semantic_vs_b0"] = paired_diff_test(sem_by_task, b0_by_task, "semantic", "b0", task_ids_all)

# ═══════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════
output_dir = Path("artifacts/marble/outputs/q30b_paper/paper_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    "dataset": {
        "total_records": len(records),
        "total_tasks": len(set(r["task_id"] for r in records)),
        "total_edges": len(edge_agg),
        "total_episodes": len(tr_edges),
        "label_distribution": dict(label_counts),
    },
    "exp2_quadrant_distribution": quad_dist,
    "exp2_oracle_ceiling": {
        "oracle_rate": oracle_rate,
        "b0_rate": b0_rate,
        "semantic_rate": semantic_rate,
        "oracle_gap": oracle_rate - b0_rate,
    },
    "exp2_difficulty_stratification": {
        diff: {
            "n_tasks": len([t for t, d in task_difficulty.items() if d["difficulty"] == diff]),
            "b0_rate": compute_policy_rate([e for e in b0_episodes if e["task_id"] in [t for t, d in task_difficulty.items() if d["difficulty"] == diff]]),
            "semantic_rate": compute_policy_rate([e for e in semantic_episodes if e["task_id"] in [t for t, d in task_difficulty.items() if d["difficulty"] == diff]]),
            "oracle_rate": compute_policy_rate([e for e in oracle_episodes if e["task_id"] in [t for t, d in task_difficulty.items() if d["difficulty"] == diff]]),
        }
        for diff in ["easy", "medium", "hard"]
    },
    "exp4_ablation": {
        "b0_no_memory": {"success_rate": b0_rate, "share_rate": 0.0, "neg_transfer_exposure": 0.0},
        "semantic_top1": {"success_rate": semantic_rate, "share_rate": 1.0, "neg_transfer_exposure": 1.0},
        "oracle_gate": {"success_rate": oracle_gate_rate, "share_rate": oracle_share_rate, "neg_transfer_exposure": oracle_neg_exposure},
        "random_gate_sweep": random_results,
        "best_random_gate": best_p,
    },
    "old_eval_38tasks": {m["method"]: m for m in old_results},
    "statistical_tests": stat_tests,
}

(output_dir / "paper_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nResults saved to {output_dir / 'paper_results.json'}")
