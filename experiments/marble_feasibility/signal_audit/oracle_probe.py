"""Audit 2: Oracle Feature Test.

Tests whether ANY model, given complete execution information, can predict
the transfer effect τ = Y₁ - Y₀.  This is a diagnostic experiment.

Oracle features (allowed):
  - Task: task_id, expected labels, root causes, score from control audit
  - Memory: routing card fields, source_task_content hash
  - Receiver: role (constant in current data)
  - Execution: fine_grained evaluator output from control/share audits
  - Candidate: source, rank

Model: Simple MLP (input → 128 → 64 → τ)
Evaluation: Pearson r, sign accuracy, pairwise ranking (5-fold CV).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _load_all_paired_records(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    for raw in paths:
        p = _PROJECT_ROOT / raw
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    records.append(json.loads(line))
    return records


def _load_memory_pool(path: str) -> dict[str, dict]:
    memories = {}
    p = _PROJECT_ROOT / path
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                m = json.loads(line)
                memories[m.get("memory_id", "")] = m
    return memories


def _get_tau(record: dict) -> int:
    y1 = 1 if record.get("share", {}).get("team_success") else 0
    y0 = 1 if record.get("withhold", {}).get("team_success") else 0
    return y1 - y0


def _load_audit_file(path: Path) -> dict | None:
    """Load an audit JSON file, returning None if missing."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _extract_oracle_features(
    record: dict,
    memory_pool: dict[str, dict],
    output_base: Path,
) -> np.ndarray | None:
    """Extract rich oracle features from a single record.

    Returns None if critical audit files are missing.
    """
    features: list[float] = []

    # ── Task features ──
    task_id = str(record.get("task_id", ""))
    # One-hot task_id (hash to reduce dimensionality)
    task_hash = hash(task_id) % 20
    for i in range(20):
        features.append(1.0 if i == task_hash else 0.0)

    # ── Load control audit for fine_grained ──
    ctrl_path_str = record.get("control_artifact_path", "")
    ctrl_audit = None
    if ctrl_path_str:
        ctrl_audit = _load_audit_file(Path(ctrl_path_str))

    if ctrl_audit:
        outcome = ctrl_audit.get("audit", {}).get("outcome", {})
        fg = outcome.get("fine_grained", {}) or {}
        score = outcome.get("score", 0) or 0
        features.append(float(score))

        # Expected labels (hash to 10 bins)
        expected = fg.get("expected_labels", [])
        exp_hash_vec = [0.0] * 10
        for lab in expected:
            exp_hash_vec[hash(lab) % 10] = 1.0
        features.extend(exp_hash_vec)

        # Predicted labels
        predicted = fg.get("predicted_labels", [])
        pred_hash_vec = [0.0] * 10
        for lab in predicted:
            pred_hash_vec[hash(lab) % 10] = 1.0
        features.extend(pred_hash_vec)

        # F1, precision, recall
        features.append(float(fg.get("f1", 0) or 0))
        features.append(float(fg.get("precision", 0) or 0))
        features.append(float(fg.get("recall", 0) or 0))
        features.append(float(fg.get("tp", 0) or 0))
        features.append(float(fg.get("fp", 0) or 0))
    else:
        features.extend([0.0] * 26)  # pad: score(1)+exp(10)+pred(10)+metrics(5)

    # ── Load share audit for fine_grained ──
    if ctrl_path_str:
        ctrl_dir = Path(ctrl_path_str).parent
        shares_dir = ctrl_dir.parent / "shares"
        edge_id = record.get("edge_id", "")
        share_audit_path = shares_dir / edge_id / "share_audit.json"
        share_audit = _load_audit_file(share_audit_path)
    else:
        share_audit = None

    if share_audit:
        s_outcome = share_audit.get("outcome", {})
        s_fg = s_outcome.get("fine_grained", {}) or {}
        s_score = s_outcome.get("score", 0) or 0
        features.append(float(s_score))

        s_expected = s_fg.get("expected_labels", [])
        s_exp_vec = [0.0] * 10
        for lab in s_expected:
            s_exp_vec[hash(lab) % 10] = 1.0
        features.extend(s_exp_vec)

        s_predicted = s_fg.get("predicted_labels", [])
        s_pred_vec = [0.0] * 10
        for lab in s_predicted:
            s_pred_vec[hash(lab) % 10] = 1.0
        features.extend(s_pred_vec)

        features.append(float(s_fg.get("f1", 0) or 0))
        features.append(float(s_fg.get("precision", 0) or 0))
        features.append(float(s_fg.get("recall", 0) or 0))
        features.append(float(s_fg.get("tp", 0) or 0))
        features.append(float(s_fg.get("fp", 0) or 0))
    else:
        features.extend([0.0] * 26)  # pad: score(1)+exp(10)+pred(10)+metrics(5)

    # ── Memory features ──
    mem_id = record.get("candidate_memory_id", "")
    mem = memory_pool.get(mem_id, {})
    rc = mem.get("routing_card", {})

    # Memory base group (2 groups)
    mem_base = "-".join(mem_id.split("-")[:2]) if "-" in mem_id else mem_id
    mem_base_hash = hash(mem_base) % 5
    for i in range(5):
        features.append(1.0 if i == mem_base_hash else 0.0)

    # Source agent (5 agents)
    src_agent = record.get("memory_source_agent_id", "")
    agent_hash = hash(src_agent) % 5
    for i in range(5):
        features.append(1.0 if i == agent_hash else 0.0)

    # Routing card features (mostly constant but included for completeness)
    features.append(float(len(rc.get("required_tools", []))))
    features.append(float(len(rc.get("required_capabilities", []))))
    features.append(float(len(rc.get("task_tags", []))))
    features.append(float(len(rc.get("execution_role_tags", []))))
    features.append(float(len(rc.get("environment_constraints", []))))
    features.append(float(rc.get("evidence_count", 0)))

    # ── Candidate features ──
    features.append(float(record.get("candidate_rank", 0)))
    features.append(float(record.get("candidate_score", 0)))
    src = record.get("candidate_source", "")
    src_hash = hash(src) % 3
    for i in range(3):
        features.append(1.0 if i == src_hash else 0.0)

    # ── Label overlap features (oracle: is memory related to task?) ──
    if ctrl_audit and share_audit:
        c_fg = (ctrl_audit.get("audit", {}).get("outcome", {})
                .get("fine_grained", {}) or {})
        s_fg2 = share_audit.get("outcome", {}).get("fine_grained", {}) or {}

        c_exp = set(c_fg.get("expected_labels", []))
        s_exp = set(s_fg2.get("expected_labels", []))
        c_pred = set(c_fg.get("predicted_labels", []))
        s_pred = set(s_fg2.get("predicted_labels", []))

        # Does the expected task label change between control and treatment?
        features.append(1.0 if c_exp != s_exp else 0.0)
        # Does predicted label change?
        features.append(1.0 if c_pred != s_pred else 0.0)
        # Overlap between expected and predicted
        if c_exp:
            features.append(len(c_exp & c_pred) / max(len(c_exp), 1))
        else:
            features.append(0.0)
        if s_exp:
            features.append(len(s_exp & s_pred) / max(len(s_exp), 1))
        else:
            features.append(0.0)
    else:
        features.extend([0.0] * 4)

    # ── Task source snapshot features (from bundle.json) ──
    if ctrl_path_str:
        ctrl_dir = Path(ctrl_path_str).parent
        bundle_path = ctrl_dir / "control" / "bundle.json"
        bundle = _load_audit_file(bundle_path)
        if bundle:
            snap = bundle.get("task_source_snapshot", {})
            features.append(float(snap.get("agent_count", 0)))
            features.append(float(snap.get("relationship_count", 0)))
            features.append(float(snap.get("number_of_labels_pred", 0)))
            root_causes = snap.get("root_causes", [])
            rc_vec = [0.0] * 10
            for lab in root_causes:
                rc_vec[hash(lab) % 10] = 1.0
            features.extend(rc_vec)
            labels = snap.get("labels", [])
            lab_vec = [0.0] * 10
            for lab in labels:
                lab_vec[hash(lab) % 10] = 1.0
            features.extend(lab_vec)
        else:
            features.extend([0.0] * 23)
    else:
        features.extend([0.0] * 23)

    return np.array(features, dtype=float)


def _pairwise_ranking(pred_tau: np.ndarray, true_tau: np.ndarray,
                      n_samples: int, rng: np.random.RandomState) -> float:
    """Compute pairwise ranking accuracy on informative pairs."""
    n = len(pred_tau)
    if n < 2:
        return 0.5

    informative_idx = [i for i in range(n) if true_tau[i] != 0]
    if len(informative_idx) < 2:
        return 0.5

    correct = 0
    total = 0
    pairs = rng.randint(0, len(informative_idx), size=(n_samples, 2))
    for a, b in pairs:
        i, j = informative_idx[a], informative_idx[b]
        if i == j:
            continue
        if true_tau[i] != true_tau[j]:
            total += 1
            if (pred_tau[i] > pred_tau[j]) == (true_tau[i] > true_tau[j]):
                correct += 1

    return correct / max(total, 1)


def main() -> None:
    config = _load_config()
    data_cfg = config["data"]
    oracle_cfg = config["audit"]["oracle"]

    print("=" * 60)
    print("Audit 2: Oracle Feature Test")
    print("=" * 60)

    # ── Load data ──
    all_records = _load_all_paired_records(data_cfg["all_paired_splits"])
    valid = [r for r in all_records if r.get("valid", False)]
    memory_pool = _load_memory_pool(data_cfg["memory_pool_path"])
    output_base = _PROJECT_ROOT / data_cfg["output_base"]

    print(f"\n  Valid records: {len(valid)}")
    print(f"  Memory pool: {len(memory_pool)}")

    # ── Extract features ──
    print("\n  Extracting oracle features...")
    X_list = []
    y_list = []
    valid_records = []

    for r in valid:
        feat = _extract_oracle_features(r, memory_pool, output_base)
        if feat is not None:
            X_list.append(feat)
            y_list.append(_get_tau(r))
            valid_records.append(r)

    X = np.array(X_list)
    y = np.array(y_list, dtype=float)
    print(f"  Features shape: {X.shape}")
    print(f"  τ distribution: pos={int((y > 0).sum())}, "
          f"neg={int((y < 0).sum())}, zero={int((y == 0).sum())}")

    # ── 5-fold cross-validation ──
    seed = oracle_cfg["seed"]
    n_folds = oracle_cfg["n_folds"]
    rng = np.random.RandomState(seed)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_rankings = []
    fold_pearsons = []
    fold_sign_accs = []

    print(f"\n  Running {n_folds}-fold cross-validation...")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train MLP
        mlp = MLPRegressor(
            hidden_layer_sizes=(oracle_cfg["hidden_size"], 64),
            max_iter=500,
            random_state=seed + fold_idx,
            early_stopping=True,
            validation_fraction=0.15,
            learning_rate="adaptive",
        )
        mlp.fit(X_train_s, y_train)

        # Predict
        pred = mlp.predict(X_test_s)

        # Pearson r
        if np.std(pred) > 0 and np.std(y_test) > 0:
            r = float(np.corrcoef(pred, y_test)[0, 1])
        else:
            r = 0.0
        fold_pearsons.append(r)

        # Sign accuracy
        sign_true = np.sign(y_test)
        sign_pred = np.sign(pred)
        sign_acc = float(np.mean(sign_true == sign_pred))
        fold_sign_accs.append(sign_acc)

        # Pairwise ranking (informative only)
        ranking = _pairwise_ranking(pred, y_test, 5000, rng)
        fold_rankings.append(ranking)

        print(f"    Fold {fold_idx + 1}: "
              f"ranking={ranking:.4f}, r={r:.4f}, sign_acc={sign_acc:.4f}")

    # ── Aggregate ──
    mean_ranking = float(np.mean(fold_rankings))
    mean_pearson = float(np.mean(fold_pearsons))
    mean_sign = float(np.mean(fold_sign_accs))

    ranking_thr = oracle_cfg["ranking_pass_threshold"]
    verdict = "PASS" if mean_ranking > ranking_thr else "FAIL"

    print(f"\n  ── Oracle Probe Results ──")
    print(f"  Mean ranking: {mean_ranking:.4f} (threshold >{ranking_thr})")
    print(f"  Mean Pearson r: {mean_pearson:.4f}")
    print(f"  Mean sign accuracy: {mean_sign:.4f}")
    print(f"  Verdict: {verdict}")

    if verdict == "PASS":
        interpretation = ("SIGNAL_EXISTS — MARBLE contains learnable "
                          "transfer signal.  Problem is representation.")
    else:
        interpretation = ("ENVIRONMENT_MISMATCH — MARBLE current subset "
                          "does not contain stable transfer structure.")

    print(f"  Interpretation: {interpretation}")

    # ── Save ──
    out_dir = _THIS_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "audit": "oracle_probe",
        "verdict": verdict,
        "interpretation": interpretation,
        "ranking": round(mean_ranking, 4),
        "pearson_r": round(mean_pearson, 4),
        "sign_accuracy": round(mean_sign, 4),
        "n_features": X.shape[1],
        "n_records": len(y),
        "tau_distribution": {
            "positive": int((y > 0).sum()),
            "negative": int((y < 0).sum()),
            "neutral": int((y == 0).sum()),
        },
        "fold_rankings": [round(r, 4) for r in fold_rankings],
        "fold_pearsons": [round(r, 4) for r in fold_pearsons],
        "fold_sign_accs": [round(r, 4) for r in fold_sign_accs],
        "threshold": ranking_thr,
    }

    out_path = out_dir / "oracle_probe.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")

    print(f"\n{'=' * 60}")
    print(f"  RESULT: {verdict}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
