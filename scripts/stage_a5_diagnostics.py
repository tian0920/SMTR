"""Stage A.5 comprehensive diagnostic analysis.

Phases D, F, G, H, J:
- D: Transfer-state distribution audit
- F: Critic diagnostics (confusion matrix, OVR metrics, PR-AUC, ROC-AUC)
- G: tau_hat distribution analysis by true state
- H: eta_hat distribution analysis by true state (diagnostic only)
- J: Baseline decision overlap analysis

SMTR-v1: eta (= q01) is diagnostic only; epsilon* is no longer a routing
parameter.  Phase I (SMTR vs SMTR-no-risk disagreement) has been removed
because smtr_no_risk is no longer a separate method.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def paired_label(rec):
    """Extract four-outcome label from a paired record."""
    return rec.get("label", rec.get("paired_label", "unknown"))


LABEL_MAP = {
    "neutral_failure": "q00",
    "negative_transfer": "q01",
    "positive_transfer": "q10",
    "neutral_success": "q11",
}


def classify_label(label):
    return LABEL_MAP.get(label, label)


def phase_d(paired_records, split_name="val"):
    """Phase D: Transfer-state distribution audit."""
    print(f"\n{'='*60}")
    print(f"Phase D: Transfer-State Distribution Audit ({split_name})")
    print(f"{'='*60}")

    labels = Counter(classify_label(paired_label(r)) for r in paired_records)
    total = len(paired_records)
    valid = sum(1 for r in paired_records if r.get("valid", True))

    print(f"\nTotal records: {total}")
    print(f"Valid records: {valid}")
    print(f"\nFour-state distribution:")
    for state in ["q00", "q01", "q10", "q11"]:
        count = labels.get(state, 0)
        frac = count / total if total else 0
        print(f"  {state}: {count:4d} ({frac*100:5.1f}%)")

    n_informative = labels.get("q01", 0) + labels.get("q10", 0)
    print(f"\nInformative records (q01 + q10): {n_informative}")
    print(f"  positive_transfer (q10): {labels.get('q10', 0)}")
    print(f"  negative_transfer (q01): {labels.get('q01', 0)}")
    print(f"  informative fraction: {n_informative/total:.3f}" if total else "")

    # Minimum data requirement check
    q10 = labels.get("q10", 0)
    q01 = labels.get("q01", 0)
    ideal = q10 >= 20 and q01 >= 20
    acceptable = q10 >= 10 and q01 >= 10
    print(f"\nData requirement check:")
    print(f"  Ideal (>=20/20): {'PASS' if ideal else 'FAIL'}")
    print(f"  Minimum (>=10/10): {'PASS' if acceptable else 'FAIL'}")
    if not acceptable:
        print(f"  WARNING: Below minimum. Do not judge critic by macro-F1.")

    return labels


def phase_f(traces, paired_records, split_name="val"):
    """Phase F: Critic diagnostics."""
    print(f"\n{'='*60}")
    print(f"Phase F: Critic Diagnostics ({split_name})")
    print(f"{'='*60}")

    # Build outcome lookup
    outcome_by_key = {}
    for rec in paired_records:
        key = (
            str(rec.get("task_id", "")),
            int(rec.get("generation_seed", 0)),
            str(rec.get("receiver_agent_id", "")),
            str(rec.get("candidate_memory_id", "")),
        )
        outcome_by_key[key] = paired_label(rec)

    # Use SMTR traces for critic diagnostics (full critic predictions)
    smtr_traces = traces.get("smtr", [])
    matched = []
    for t in smtr_traces:
        key = (
            str(t.get("task_id", "")),
            int(t.get("generation_seed", 0)),
            str(t.get("receiver_agent_id", "")),
            str(t.get("candidate_memory_id", "")),
        )
        true_label = outcome_by_key.get(key)
        if true_label:
            matched.append({
                "true_label": classify_label(true_label),
                "tau_hat": t.get("tau_hat", 0.0),
                "eta_raw": t.get("eta_raw", 0.0),
                "eta_calibrated": t.get("eta_calibrated", t.get("eta_raw", 0.0)),
                "action": t.get("action", "withhold"),
            })

    print(f"\nMatched traces: {len(matched)}")

    # F1: Four-state confusion matrix
    print(f"\n--- F1: Four-State Confusion Matrix ---")
    from sklearn.metrics import confusion_matrix, classification_report

    y_true = [m["true_label"] for m in matched]
    # Predicted class from four-outcome probabilities
    # tau_hat = q10 - q01, eta_raw = q01
    # Reconstruct approximate q values
    y_pred_class = []
    for m in matched:
        tau = m["tau_hat"]
        eta = m["eta_raw"]
        q01 = eta
        q10 = tau + eta
        q00 = max(0, 1 - q01 - q10 - 0.1)  # approximate
        q11 = max(0, 1 - q00 - q01 - q10)
        probs = [q00, q01, q10, q11]
        classes = ["q00", "q01", "q10", "q11"]
        y_pred_class.append(classes[np.argmax(probs)])

    labels_order = ["q00", "q01", "q10", "q11"]
    cm = confusion_matrix(y_true, y_pred_class, labels=labels_order)
    print(f"{'':>8} {'q00':>6} {'q01':>6} {'q10':>6} {'q11':>6}")
    for i, row_label in enumerate(labels_order):
        row = "  ".join(f"{cm[i][j]:6d}" for j in range(4))
        print(f"{row_label:>8} {row}")

    # F2: Per-class metrics
    print(f"\n--- F2: Per-Class Metrics ---")
    print(classification_report(y_true, y_pred_class, labels=labels_order,
                                 digits=3, zero_division=0))

    return matched, y_true, y_pred_class


def phase_f_ovr(matched, y_true):
    """Phase F3/F4: One-vs-rest metrics."""
    print(f"\n--- F3: Positive-Transfer (q10) One-vs-Rest ---")
    y_binary_pos = np.array([1 if y == "q10" else 0 for y in y_true])
    tau_scores = np.array([m["tau_hat"] for m in matched])

    from sklearn.metrics import (
        precision_recall_curve, roc_auc_score, average_precision_score,
        precision_score, recall_score, f1_score
    )

    # Use tau_hat as score for positive transfer (higher tau = more likely q10)
    _ovr_metrics("q10", y_binary_pos, tau_scores, "tau_hat")

    print(f"\n--- F4: Harmful-Transfer (q01) One-vs-Rest ---")
    y_binary_neg = np.array([1 if y == "q01" else 0 for y in y_true])
    eta_scores = np.array([m["eta_raw"] for m in matched])

    # Use eta_hat as score for harmful transfer (higher eta = more likely q01)
    _ovr_metrics("q01", y_binary_neg, eta_scores, "eta_raw")

    return y_binary_pos, y_binary_neg, tau_scores, eta_scores


def _ovr_metrics(state_name, y_binary, scores, score_name):
    from sklearn.metrics import (
        precision_recall_curve, roc_auc_score, average_precision_score,
        precision_score, recall_score, f1_score
    )

    prevalence = y_binary.mean()
    print(f"  Class prevalence: {y_binary.sum()}/{len(y_binary)} = {prevalence:.3f}")

    # Threshold at median score
    threshold = np.median(scores)
    y_pred = (scores > threshold).astype(int)
    p = precision_score(y_binary, y_pred, zero_division=0)
    r = recall_score(y_binary, y_pred, zero_division=0)
    f1 = f1_score(y_binary, y_pred, zero_division=0)
    print(f"  At median threshold ({threshold:.4f}):")
    print(f"    Precision: {p:.3f}")
    print(f"    Recall:    {r:.3f}")
    print(f"    F1:        {f1:.3f}")

    # AUC metrics
    n_pos = y_binary.sum()
    n_neg = len(y_binary) - n_pos
    if n_pos > 0 and n_neg > 0 and len(set(scores)) > 1:
        roc = roc_auc_score(y_binary, scores)
        pr_auc = average_precision_score(y_binary, scores)
        print(f"  ROC-AUC:  {roc:.3f}")
        print(f"  PR-AUC:   {pr_auc:.3f} (baseline: {prevalence:.3f})")
        if pr_auc > prevalence * 1.5:
            print(f"  Signal detected: PR-AUC > 1.5× prevalence")
        else:
            print(f"  Weak signal: PR-AUC close to prevalence baseline")
    else:
        print(f"  AUC metrics: insufficient data (pos={n_pos}, neg={n_neg})")


def phase_g(matched):
    """Phase G: Analyze tau_hat distributions."""
    print(f"\n{'='*60}")
    print(f"Phase G: tau_hat Distribution Analysis")
    print(f"{'='*60}")

    by_state = defaultdict(list)
    for m in matched:
        by_state[m["true_label"]].append(m["tau_hat"])

    print(f"\n--- G1: tau_hat by true transfer state ---")
    print(f"{'State':>6} {'Count':>6} {'Mean':>8} {'Median':>8} {'Std':>8} {'P25':>8} {'P50':>8} {'P75':>8}")
    for state in ["q00", "q01", "q10", "q11"]:
        vals = by_state.get(state, [])
        if vals:
            arr = np.array(vals)
            print(f"{state:>6} {len(arr):6d} {arr.mean():8.4f} {np.median(arr):8.4f} "
                  f"{arr.std():8.4f} {np.percentile(arr, 25):8.4f} {np.percentile(arr, 50):8.4f} "
                  f"{np.percentile(arr, 75):8.4f}")

    # G2: Ranking check
    print(f"\n--- G2: tau ranking check ---")
    q10_tau = by_state.get("q10", [])
    q01_tau = by_state.get("q01", [])
    if q10_tau and q01_tau:
        mean_q10 = np.mean(q10_tau)
        mean_q01 = np.mean(q01_tau)
        print(f"  mean(tau | q10) = {mean_q10:.4f}")
        print(f"  mean(tau | q01) = {mean_q01:.4f}")
        print(f"  Difference: {mean_q10 - mean_q01:.4f}")
        print(f"  Correct ordering (q10 > q01): {'YES' if mean_q10 > mean_q01 else 'NO'}")

        # AUC
        from sklearn.metrics import roc_auc_score
        y = np.array([1]*len(q10_tau) + [0]*len(q01_tau))
        scores = np.array(q10_tau + q01_tau)
        if len(set(scores)) > 1:
            auc = roc_auc_score(y, scores)
            print(f"  AUC(q10 vs q01 using tau_hat): {auc:.3f}")
            if auc > 0.6:
                print(f"  Ranking signal: MODERATE")
            elif auc > 0.55:
                print(f"  Ranking signal: WEAK")
            else:
                print(f"  Ranking signal: NONE (random level)")

    # G3: Sign accuracy
    print(f"\n--- G3: Sign accuracy ---")
    if q10_tau:
        p_pos = np.mean(np.array(q10_tau) > 0)
        print(f"  P(tau_hat > 0 | q10) = {p_pos:.3f} (should be HIGH)")
    if q01_tau:
        p_neg = np.mean(np.array(q01_tau) > 0)
        print(f"  P(tau_hat > 0 | q01) = {p_neg:.3f} (should be LOW)")
    if q10_tau and q01_tau:
        if abs(p_pos - p_neg) < 0.1:
            print(f"  VERDICT: tau critic has NOT learned transfer direction")
        else:
            print(f"  VERDICT: tau shows directional signal (delta={p_pos-p_neg:.3f})")


def phase_h(matched):
    """Phase H: Analyze eta_hat distributions (diagnostic only)."""
    print(f"\n{'='*60}")
    print(f"Phase H: eta_hat Distribution Analysis")
    print(f"{'='*60}")

    by_state = defaultdict(list)
    for m in matched:
        by_state[m["true_label"]].append(m["eta_calibrated"])

    print(f"\n--- H1: eta_hat (calibrated) by true transfer state ---")
    print(f"{'State':>6} {'Count':>6} {'Mean':>8} {'Median':>8} {'Std':>8} {'P25':>8} {'P50':>8} {'P75':>8}")
    for state in ["q00", "q01", "q10", "q11"]:
        vals = by_state.get(state, [])
        if vals:
            arr = np.array(vals)
            print(f"{state:>6} {len(arr):6d} {arr.mean():8.4f} {np.median(arr):8.4f} "
                  f"{arr.std():8.4f} {np.percentile(arr, 25):8.4f} {np.percentile(arr, 50):8.4f} "
                  f"{np.percentile(arr, 75):8.4f}")

    # H2: Harmful discrimination
    print(f"\n--- H2: Harmful discrimination ---")
    q01_eta = by_state.get("q01", [])
    non_q01_eta = []
    for state in ["q00", "q10", "q11"]:
        non_q01_eta.extend(by_state.get(state, []))

    if q01_eta and non_q01_eta:
        mean_q01 = np.mean(q01_eta)
        mean_rest = np.mean(non_q01_eta)
        print(f"  mean(eta | q01) = {mean_q01:.4f}")
        print(f"  mean(eta | non-q01) = {mean_rest:.4f}")
        print(f"  Correct ordering (q01 > non-q01): {'YES' if mean_q01 > mean_rest else 'NO'}")

        from sklearn.metrics import roc_auc_score, average_precision_score
        y = np.array([1]*len(q01_eta) + [0]*len(non_q01_eta))
        scores = np.array(q01_eta + non_q01_eta)
        if len(set(scores)) > 1:
            auc = roc_auc_score(y, scores)
            pr_auc = average_precision_score(y, scores)
            prevalence = len(q01_eta) / len(y)
            print(f"  AUC(q01 vs rest using eta_hat): {auc:.3f}")
            print(f"  PR-AUC(q01 vs rest): {pr_auc:.3f} (baseline: {prevalence:.3f})")

    # H3: epsilon diagnostic analysis (no longer a routing gate)
    print(f"\n--- H3: eta diagnostic summary ---")
    print(f"  SMTR-v1: eta (= q01) is diagnostic only; no epsilon* routing gate.")
    if q01_eta:
        print(f"  mean(eta | q01) = {np.mean(q01_eta):.4f}")
        print(f"  max(eta | q01)  = {np.max(q01_eta):.4f}")
    if non_q01_eta:
        print(f"  mean(eta | non-q01) = {np.mean(non_q01_eta):.4f}")


def phase_j(traces):
    """Phase J: Baseline decision overlap analysis."""
    print(f"\n{'='*60}")
    print(f"Phase J: Baseline Decision Overlap")
    print(f"{'='*60}")

    semantic = traces.get("semantic_top1", [])
    compatible = traces.get("receiver_compatible_top1", [])

    def episode_key(t):
        return (
            str(t.get("task_id", "")),
            int(t.get("generation_seed", 0)),
            str(t.get("receiver_agent_id", "")),
        )

    # Group by episode to see which memory each method selects
    sem_by_ep = defaultdict(set)
    comp_by_ep = defaultdict(set)
    for t in semantic:
        if t.get("action") == "share":
            sem_by_ep[episode_key(t)].add(t.get("candidate_memory_id", ""))
    for t in compatible:
        if t.get("action") == "share":
            comp_by_ep[episode_key(t)].add(t.get("candidate_memory_id", ""))

    all_eps = set(sem_by_ep.keys()) | set(comp_by_ep.keys())
    disagree = 0
    same = 0
    for ep in all_eps:
        sem_mem = sem_by_ep.get(ep, set())
        comp_mem = comp_by_ep.get(ep, set())
        if sem_mem == comp_mem:
            same += 1
        else:
            disagree += 1

    print(f"\n--- J1: Semantic vs Compatible disagreement ---")
    total = same + disagree
    print(f"  Episodes with share action: {total}")
    print(f"  Same selection: {same} ({same/total*100:.1f}%)" if total else "")
    print(f"  Different selection: {disagree} ({disagree/total*100:.1f}%)" if total else "")

    if disagree == 0:
        print(f"\n  VERDICT: Compatibility signal produces NO independent routing behavior.")
        print(f"  Current candidate pool: semantic and compatibility rankings converge.")
    elif total > 0 and disagree / total < 0.05:
        print(f"\n  VERDICT: Near-zero disagreement. Small sample coincidence possible.")
    else:
        print(f"\n  VERDICT: Meaningful disagreement exists ({disagree/total*100:.1f}%).")

    # J2: Per-episode selection detail
    print(f"\n--- J2: Selection detail (first 10 episodes) ---")
    for i, ep in enumerate(sorted(all_eps)[:10]):
        sem_mem = sem_by_ep.get(ep, set())
        comp_mem = comp_by_ep.get(ep, set())
        status = "SAME" if sem_mem == comp_mem else "DIFF"
        print(f"  [{status}] task={ep[0]} seed={ep[1]} agent={ep[2]}")
        print(f"    semantic: {sem_mem or 'none'}")
        print(f"    compatible: {comp_mem or 'none'}")


def main():
    paired_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/marble/outputs/effect_check/stageA_paired_val/paired_records.jsonl"
    traces_path = sys.argv[2] if len(sys.argv) > 2 else "artifacts/marble/outputs/effect_check/stageA_paired_eval/traces.json"
    critic_path = sys.argv[3] if len(sys.argv) > 3 else "artifacts/marble/outputs/effect_check/stageA_critic"
    split_name = sys.argv[4] if len(sys.argv) > 4 else "val"

    print(f"Loading data...")
    print(f"  Paired records: {paired_path}")
    print(f"  Traces: {traces_path}")
    print(f"  Critic: {critic_path}")

    paired_records = load_jsonl(paired_path)
    traces = json.load(open(traces_path))

    # Load critic (diagnostic only; no epsilon* in SMTR-v1)
    try:
        from smtr.router.transfer_critic import FourOutcomeTransferCritic
        _critic = FourOutcomeTransferCritic.load(Path(critic_path))
        print(f"  Critic loaded (diagnostic only; no epsilon* in SMTR-v1)")
    except Exception as e:
        print(f"  Could not load critic: {e}")

    # Phase D
    labels = phase_d(paired_records, split_name)

    # Phase F
    matched, y_true, y_pred_class = phase_f(traces, paired_records, split_name)
    phase_f_ovr(matched, y_true)

    # Phase G
    phase_g(matched)

    # Phase H
    phase_h(matched)

    # Phase J
    phase_j(traces)

    print(f"\n{'='*60}")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
