#!/usr/bin/env python3
"""Re-run and analyze the SMTR mechanism with clean features (SMTR-v1).

Steps:
  1. Load q30b_full_resume paired records + rebuilt memory pool
  2. Train a FourOutcomeTransferCritic with feature_block="full" (clean)
  3. Run SMTR tau-only routing on all records
  4. Analyze tau/eta distributions, routing decisions, four-outcome labels

Usage:
    python scripts/run_and_analyze_smtr.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smtr.core.types import CandidateExposureInput
from smtr.router.transfer_features import (
    HashingTransferFeatureEncoder,
    load_paired_records_with_metadata,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic, LABEL_TO_INDEX
from smtr.router.exposure_router import SMTRExposureRouter
from smtr.counterfactual.edge_keys import (
    edge_equal_sample_weights,
    group_records_by_control_family,
    group_records_by_edge,
)


# ── Configuration ─────────────────────────────────────────────────────
PAIRED_RECORDS = Path("artifacts/marble/outputs/q30b_full_resume/paired_records.jsonl")
MEMORY_POOL = Path("artifacts/marble/real_data/database_v1/memory_pool.jsonl")
SEED = 7
N_BOOTSTRAP = 31
N_FEATURES = 512
FEATURE_BLOCK = "full"
TEST_FRACTION = 0.2


def group_split_records(
    records: list[dict], *, seed: int, test_fraction: float,
) -> tuple[list[dict], list[dict]]:
    """Split records by control_family_id (or edge_id) for grouped CV."""
    family_ids = sorted({
        rec.get("control_family_id", rec.get("edge_id", ""))
        for rec in records
    })
    rng = random.Random(seed)
    rng.shuffle(family_ids)
    test_count = max(1, int(round(len(family_ids) * test_fraction)))
    test_ids = set(family_ids[:test_count])
    train = [r for r in records if r.get("control_family_id", r.get("edge_id", "")) not in test_ids]
    test = [r for r in records if r.get("control_family_id", r.get("edge_id", "")) in test_ids]
    if not train and test:
        train, test = test, []
    return train, test


def main() -> None:
    print("=" * 70)
    print("SMTR-v1 Mechanism Re-run & Analysis")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────
    print("\n[1] Loading data...")
    training_data = load_paired_records_with_metadata(PAIRED_RECORDS, MEMORY_POOL)
    print(f"  Total training examples (core-valid, features constructible): {len(training_data)}")

    # Split into train/test by control family
    all_records = [rec for _, _, rec in training_data]
    train_records, test_records = group_split_records(
        all_records, seed=SEED, test_fraction=TEST_FRACTION,
    )
    train_set = set(id(r) for r in train_records)
    test_set = set(id(r) for r in test_records)
    train_triples = [(inp, lbl, rec) for inp, lbl, rec in training_data if id(rec) in train_set]
    test_triples = [(inp, lbl, rec) for inp, lbl, rec in training_data if id(rec) in test_set]
    print(f"  Train split: {len(train_triples)}, Test split: {len(test_triples)}")

    # Label distribution
    train_labels = [lbl for _, lbl, _ in train_triples]
    test_labels = [lbl for _, lbl, _ in test_triples]
    print(f"\n  Train label distribution:")
    for k, v in sorted(Counter(train_labels).items()):
        pct = v / len(train_labels) * 100 if train_labels else 0
        print(f"    {k}: {v} ({pct:.1f}%)")
    if test_labels:
        print(f"\n  Test label distribution:")
        for k, v in sorted(Counter(test_labels).items()):
            pct = v / len(test_labels) * 100 if test_labels else 0
            print(f"    {k}: {v} ({pct:.1f}%)")

    # ── 2. Train critic ───────────────────────────────────────────────
    print(f"\n[2] Training critic (feature_block={FEATURE_BLOCK}, n_bootstrap={N_BOOTSTRAP})...")
    critic = FourOutcomeTransferCritic(
        n_features=N_FEATURES,
        n_bootstrap=N_BOOTSTRAP,
        feature_block=FEATURE_BLOCK,
        seed=SEED,
    )
    train_inputs = [item for item, _, _ in train_triples]
    train_labels_list = [lbl for _, lbl, _ in train_triples]
    train_records_list = [rec for _, _, rec in train_triples]

    sample_weights = edge_equal_sample_weights(train_records_list)
    bootstrap_clusters = group_records_by_control_family(train_records_list)

    critic.fit(
        train_inputs,
        train_labels_list,
        coverage_mode="pilot",
        sample_weights=sample_weights,
        bootstrap_clusters=bootstrap_clusters,
    )
    print(f"  Critic trained: {critic.n_bootstrap} bootstrap models")
    print(f"  Coverage report: {critic.coverage_report}")

    # ── 3. Feature audit ──────────────────────────────────────────────
    print(f"\n[3] Feature audit...")
    encoder = critic.encoder
    sample_inputs = train_inputs[:min(50, len(train_inputs))]
    all_tokens = []
    for item in sample_inputs:
        all_tokens.extend(encoder.tokens(item))
    prefixes = set()
    for tok in all_tokens:
        prefixes.add(tok.lower().split(":", 1)[0])
    print(f"  Feature block: {encoder.feature_block}")
    print(f"  n_features: {encoder.n_features}")
    print(f"  Unique token prefixes: {sorted(prefixes)}")
    forbidden = {"writer", "writer_role", "wr_pair", "source_agent", "memory_source_agent"}
    leaked = prefixes & forbidden
    if leaked:
        print(f"  WARNING: forbidden leakage detected: {leaked}")
    else:
        print(f"  No writer/provenance leakage detected ✓")

    # ── 4. Run SMTR routing on ALL records ────────────────────────────
    print(f"\n[4] Running SMTR tau-only routing...")

    decisions_all = []
    for i, (inp, lbl, rec) in enumerate(training_data):
        pred = critic.predict(inp)
        tau_hat = pred.tau_hat
        q00 = pred.q00_neutral_failure
        q01 = pred.q01_negative_transfer
        q10 = pred.q10_positive_transfer
        q11 = pred.q11_neutral_success
        action = "share" if tau_hat > 0 else "withhold"
        decisions_all.append({
            "task_id": rec.get("task_id", ""),
            "receiver_agent_id": rec.get("receiver_agent_id", ""),
            "candidate_memory_id": rec.get("candidate_memory_id", ""),
            "generation_seed": rec.get("generation_seed", 0),
            "action": action,
            "tau_hat": tau_hat,
            "q00": q00, "q01": q01, "q10": q10, "q11": q11,
            "eta_raw": q01,
            "label": lbl,
            "edge_id": rec.get("edge_id", ""),
            "control_family_id": rec.get("control_family_id", ""),
        })

    print(f"  Total decisions: {len(decisions_all)}")

    # ── 5. Analysis ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ANALYSIS RESULTS")
    print(f"{'='*70}")

    # 5a. Routing decisions
    share_count = sum(1 for d in decisions_all if d["action"] == "share")
    withhold_count = len(decisions_all) - share_count
    print(f"\n[5a] Routing Decisions:")
    print(f"  Share:    {share_count} ({share_count/len(decisions_all)*100:.1f}%)")
    print(f"  Withhold: {withhold_count} ({withhold_count/len(decisions_all)*100:.1f}%)")

    # 5b. Tau distribution
    taus = np.array([d["tau_hat"] for d in decisions_all])
    print(f"\n[5b] Tau (τ = q10 - q01) Distribution:")
    print(f"  Mean:   {taus.mean():+.4f}")
    print(f"  Std:    {taus.std():.4f}")
    print(f"  Min:    {taus.min():+.4f}")
    print(f"  Max:    {taus.max():+.4f}")
    print(f"  Median: {np.median(taus):+.4f}")
    positive_tau = sum(1 for t in taus if t > 0)
    print(f"  τ > 0:  {positive_tau} ({positive_tau/len(taus)*100:.1f}%)")
    print(f"  τ ≤ 0:  {len(taus)-positive_tau} ({(len(taus)-positive_tau)/len(taus)*100:.1f}%)")

    # 5c. Eta (q01) distribution — diagnostic only
    etas = np.array([d["eta_raw"] for d in decisions_all])
    print(f"\n[5c] Eta (η = q01, diagnostic) Distribution:")
    print(f"  Mean:   {etas.mean():.4f}")
    print(f"  Std:    {etas.std():.4f}")
    print(f"  Min:    {etas.min():.4f}")
    print(f"  Max:    {etas.max():.4f}")
    print(f"  Median: {np.median(etas):.4f}")

    # 5d. Four-outcome probability distributions
    q00s = np.array([d["q00"] for d in decisions_all])
    q01s = etas
    q10s = np.array([d["q10"] for d in decisions_all])
    q11s = np.array([d["q11"] for d in decisions_all])
    print(f"\n[5d] Four-Outcome Probabilities (mean ± std):")
    print(f"  q00 (neutral_failure):  {q00s.mean():.4f} ± {q00s.std():.4f}")
    print(f"  q01 (negative_transfer): {q01s.mean():.4f} ± {q01s.std():.4f}")
    print(f"  q10 (positive_transfer): {q10s.mean():.4f} ± {q10s.std():.4f}")
    print(f"  q11 (neutral_success):   {q11s.mean():.4f} ± {q11s.std():.4f}")

    # 5e. Routing vs ground truth
    print(f"\n[5e] Routing Decision vs Ground Truth Label:")
    label_by_action = defaultdict(Counter)
    for d in decisions_all:
        label_by_action[d["action"]][d["label"]] += 1
    for action in ["share", "withhold"]:
        total = sum(label_by_action[action].values())
        print(f"\n  Action={action} (n={total}):")
        for label, count in sorted(label_by_action[action].items()):
            pct = count / total * 100 if total else 0
            print(f"    {label}: {count} ({pct:.1f}%)")

    # 5f. Tau by label
    print(f"\n[5f] Tau Distribution by Ground Truth Label:")
    by_label = defaultdict(list)
    for d in decisions_all:
        by_label[d["label"]].append(d["tau_hat"])
    for label in ["positive_transfer", "negative_transfer", "neutral_success", "neutral_failure"]:
        vals = by_label.get(label, [])
        if vals:
            arr = np.array(vals)
            print(f"  {label:25s}: n={len(arr):3d}, τ_mean={arr.mean():+.4f}, "
                  f"τ_std={arr.std():.4f}, τ>0={sum(1 for v in arr if v>0)}/{len(arr)}")

    # 5g. Eta by label
    print(f"\n[5g] Eta (q01) Distribution by Ground Truth Label:")
    eta_by_label = defaultdict(list)
    for d in decisions_all:
        eta_by_label[d["label"]].append(d["eta_raw"])
    for label in ["positive_transfer", "negative_transfer", "neutral_success", "neutral_failure"]:
        vals = eta_by_label.get(label, [])
        if vals:
            arr = np.array(vals)
            print(f"  {label:25s}: n={len(arr):3d}, η_mean={arr.mean():.4f}, "
                  f"η_std={arr.std():.4f}")

    # 5h. Edge-level analysis
    print(f"\n[5h] Edge-Level Analysis:")
    edge_decisions = defaultdict(list)
    for d in decisions_all:
        edge_decisions[d["edge_id"]].append(d)
    consistent_edges = 0
    mixed_edges = 0
    for edge_id, dlist in edge_decisions.items():
        actions = set(d["action"] for d in dlist)
        if len(actions) == 1:
            consistent_edges += 1
        else:
            mixed_edges += 1
    print(f"  Total edges: {len(edge_decisions)}")
    print(f"  Consistent (all share or all withhold): {consistent_edges}")
    print(f"  Mixed (some share, some withhold): {mixed_edges}")

    # 5i. Receiver-level analysis
    print(f"\n[5i] Receiver-Level Analysis:")
    recv_decisions = defaultdict(list)
    for d in decisions_all:
        recv_decisions[d["receiver_agent_id"]].append(d)
    for recv_id in sorted(recv_decisions.keys()):
        dlist = recv_decisions[recv_id]
        n_share = sum(1 for d in dlist if d["action"] == "share")
        n_total = len(dlist)
        mean_tau = np.mean([d["tau_hat"] for d in dlist])
        mean_eta = np.mean([d["eta_raw"] for d in dlist])
        print(f"  {recv_id}: {n_share}/{n_total} shared, "
              f"τ_mean={mean_tau:+.4f}, η_mean={mean_eta:.4f}")

    # 5j. Test-set performance
    if test_triples:
        print(f"\n[5j] Test-Set Critic Performance:")
        test_preds = []
        for inp, lbl, rec in test_triples:
            pred = critic.predict(inp)
            pred_label = ["neutral_failure", "negative_transfer", "positive_transfer", "neutral_success"][
                int(np.argmax([pred.q00_neutral_failure, pred.q01_negative_transfer,
                              pred.q10_positive_transfer, pred.q11_neutral_success]))
            ]
            test_preds.append((pred_label, lbl))
        correct = sum(1 for p, t in test_preds if p == t)
        print(f"  Accuracy: {correct}/{len(test_preds)} ({correct/len(test_preds)*100:.1f}%)")
        from sklearn.metrics import classification_report
        y_true = [t for _, t in test_preds]
        y_pred = [p for p, _ in test_preds]
        labels_order = ["neutral_failure", "negative_transfer", "positive_transfer", "neutral_success"]
        print(classification_report(y_true, y_pred, labels=labels_order, digits=3, zero_division=0))

    # 5k. SMTR selectivity
    print(f"\n[5k] SMTR Selectivity Analysis:")
    shared_labels = [d["label"] for d in decisions_all if d["action"] == "share"]
    withheld_labels = [d["label"] for d in decisions_all if d["action"] == "withhold"]
    if shared_labels:
        shared_pos = sum(1 for l in shared_labels if l == "positive_transfer")
        shared_neg = sum(1 for l in shared_labels if l == "negative_transfer")
        print(f"  Shared memories label breakdown:")
        print(f"    positive_transfer:  {shared_pos}/{len(shared_labels)} ({shared_pos/len(shared_labels)*100:.1f}%)")
        print(f"    negative_transfer:  {shared_neg}/{len(shared_labels)} ({shared_neg/len(shared_labels)*100:.1f}%)")
    if withheld_labels:
        withheld_pos = sum(1 for l in withheld_labels if l == "positive_transfer")
        withheld_neg = sum(1 for l in withheld_labels if l == "negative_transfer")
        print(f"  Withheld memories label breakdown:")
        print(f"    positive_transfer:  {withheld_pos}/{len(withheld_labels)} ({withheld_pos/len(withheld_labels)*100:.1f}%)")
        print(f"    negative_transfer:  {withheld_neg}/{len(withheld_labels)} ({withheld_neg/len(withheld_labels)*100:.1f}%)")

    # Transfer safety
    if shared_labels:
        negative_share_rate = shared_neg / len(shared_labels)
        print(f"\n  Transfer safety: {negative_share_rate*100:.1f}% of shared memories are negative_transfer")
    # Transfer coverage
    total_pos = sum(1 for d in decisions_all if d["label"] == "positive_transfer")
    pos_shared = sum(1 for d in decisions_all if d["label"] == "positive_transfer" and d["action"] == "share")
    if total_pos > 0:
        print(f"  Transfer coverage: {pos_shared}/{total_pos} ({pos_shared/total_pos*100:.1f}%) of positive_transfer memories are shared")

    # 5l. Tau separation: how well does tau separate positive from negative transfer?
    print(f"\n[5l] Tau Separation Quality:")
    pos_taus = [d["tau_hat"] for d in decisions_all if d["label"] == "positive_transfer"]
    neg_taus = [d["tau_hat"] for d in decisions_all if d["label"] == "negative_transfer"]
    if pos_taus and neg_taus:
        pos_arr = np.array(pos_taus)
        neg_arr = np.array(neg_taus)
        print(f"  Positive transfer τ: mean={pos_arr.mean():+.4f}, std={pos_arr.std():.4f}")
        print(f"  Negative transfer τ: mean={neg_arr.mean():+.4f}, std={neg_arr.std():.4f}")
        # Separation = difference in means / pooled std
        pooled_std = np.sqrt((pos_arr.var() + neg_arr.var()) / 2)
        if pooled_std > 0:
            separation = (pos_arr.mean() - neg_arr.mean()) / pooled_std
            print(f"  Cohen's d (separation): {separation:.4f}")
        # Sign accuracy: P(τ>0 | positive) and P(τ≤0 | negative)
        pos_correct = sum(1 for t in pos_taus if t > 0)
        neg_correct = sum(1 for t in neg_taus if t <= 0)
        print(f"  P(τ>0 | positive_transfer): {pos_correct}/{len(pos_taus)} ({pos_correct/len(pos_taus)*100:.1f}%)")
        print(f"  P(τ≤0 | negative_transfer): {neg_correct}/{len(neg_taus)} ({neg_correct/len(neg_taus)*100:.1f}%)")

    print(f"\n{'='*70}")
    print("Analysis complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
