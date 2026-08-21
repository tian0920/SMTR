"""Audit 3: Current Representation Probe.

Compares three feature sets to diagnose where the representation fails:

  Feature A: Current SMTR features (HashingTransferFeatureEncoder)
  Feature B: SMTR + memory metadata (source_agent, source_task, base group)
  Feature C: SMTR + memory metadata + execution outcome features

Uses sklearn LogisticRegression with 5-fold CV on each feature set.
Compares ranking, sign accuracy, and Pearson r across A/B/C.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
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
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _encode_feature_a(record: dict, memory_pool: dict) -> list[float]:
    """Feature A: current SMTR-like features (metadata only, no payload)."""
    feats: list[float] = []

    # Task context
    task_id = str(record.get("task_id", ""))
    task_hash = hash(task_id) % 20
    for i in range(20):
        feats.append(1.0 if i == task_hash else 0.0)

    # Memory routing card features
    mem_id = record.get("candidate_memory_id", "")
    mem = memory_pool.get(mem_id, {})
    rc = mem.get("routing_card", {})
    feats.append(float(len(rc.get("required_tools", []))))
    feats.append(float(len(rc.get("required_capabilities", []))))
    feats.append(float(len(rc.get("task_tags", []))))
    feats.append(float(len(rc.get("execution_role_tags", []))))
    feats.append(float(len(rc.get("environment_constraints", []))))
    feats.append(float(rc.get("evidence_count", 0)))

    # Procedure type hash
    proc_hash = hash(rc.get("procedure_type", "")) % 5
    for i in range(5):
        feats.append(1.0 if i == proc_hash else 0.0)

    # Length bucket hash
    len_hash = hash(rc.get("procedure_length_bucket", "")) % 4
    for i in range(4):
        feats.append(1.0 if i == len_hash else 0.0)

    # Receiver (constant in current data)
    recv = record.get("receiver_agent_id", "")
    recv_hash = hash(recv) % 3
    for i in range(3):
        feats.append(1.0 if i == recv_hash else 0.0)

    # Candidate
    feats.append(float(record.get("candidate_rank", 0)))
    feats.append(float(record.get("candidate_score", 0)))
    src_hash = hash(record.get("candidate_source", "")) % 3
    for i in range(3):
        feats.append(1.0 if i == src_hash else 0.0)

    return feats


def _encode_feature_b(record: dict, memory_pool: dict) -> list[float]:
    """Feature B: SMTR + memory metadata."""
    feats = _encode_feature_a(record, memory_pool)

    # Add memory metadata
    mem_id = record.get("candidate_memory_id", "")

    # Memory base group
    mem_base = "-".join(mem_id.split("-")[:2]) if "-" in mem_id else mem_id
    base_hash = hash(mem_base) % 5
    for i in range(5):
        feats.append(1.0 if i == base_hash else 0.0)

    # Source agent
    src_agent = record.get("memory_source_agent_id", "")
    agent_hash = hash(src_agent) % 5
    for i in range(5):
        feats.append(1.0 if i == agent_hash else 0.0)

    # Source task
    src_task = str(record.get("memory_source_task_id", ""))
    stask_hash = hash(src_task) % 5
    for i in range(5):
        feats.append(1.0 if i == stask_hash else 0.0)

    # Memory content hash (source_task_content)
    mem = memory_pool.get(mem_id, {})
    content = mem.get("source_task_content", "")
    content_hash = hash(content[:200]) % 5
    for i in range(5):
        feats.append(1.0 if i == content_hash else 0.0)

    return feats


def _encode_feature_c(record: dict, memory_pool: dict,
                      output_base: Path) -> list[float]:
    """Feature C: SMTR + memory metadata + execution outcome."""
    feats = _encode_feature_b(record, memory_pool)

    # Add execution outcome features from control audit
    ctrl_path_str = record.get("control_artifact_path", "")
    ctrl_audit = None
    if ctrl_path_str:
        ctrl_audit = _load_audit_file(Path(ctrl_path_str))

    if ctrl_audit:
        outcome = ctrl_audit.get("audit", {}).get("outcome", {})
        fg = outcome.get("fine_grained", {}) or {}

        feats.append(float(outcome.get("score", 0) or 0))
        feats.append(float(fg.get("f1", 0) or 0))
        feats.append(float(fg.get("precision", 0) or 0))
        feats.append(float(fg.get("recall", 0) or 0))
        feats.append(float(fg.get("tp", 0) or 0))
        feats.append(float(fg.get("fp", 0) or 0))

        # Expected labels hash
        exp = fg.get("expected_labels", [])
        exp_vec = [0.0] * 10
        for lab in exp:
            exp_vec[hash(lab) % 10] = 1.0
        feats.extend(exp_vec)

        # Predicted labels hash
        pred = fg.get("predicted_labels", [])
        pred_vec = [0.0] * 10
        for lab in pred:
            pred_vec[hash(lab) % 10] = 1.0
        feats.extend(pred_vec)

        # Failure reason
        fr = outcome.get("failure_reason", "")
        fr_hash = hash(fr) % 5
        for i in range(5):
            feats.append(1.0 if i == fr_hash else 0.0)
    else:
        feats.extend([0.0] * 41)

    # Share audit outcome
    if ctrl_path_str:
        ctrl_dir = Path(ctrl_path_str).parent
        shares_dir = ctrl_dir.parent / "shares"
        edge_id = record.get("edge_id", "")
        share_path = shares_dir / edge_id / "share_audit.json"
        share_audit = _load_audit_file(share_path)
    else:
        share_audit = None

    if share_audit:
        s_outcome = share_audit.get("outcome", {})
        s_fg = s_outcome.get("fine_grained", {}) or {}

        feats.append(float(s_outcome.get("score", 0) or 0))
        feats.append(float(s_fg.get("f1", 0) or 0))
        feats.append(float(s_fg.get("precision", 0) or 0))
        feats.append(float(s_fg.get("recall", 0) or 0))
        feats.append(float(s_fg.get("tp", 0) or 0))
        feats.append(float(s_fg.get("fp", 0) or 0))

        s_exp = s_fg.get("expected_labels", [])
        s_exp_vec = [0.0] * 10
        for lab in s_exp:
            s_exp_vec[hash(lab) % 10] = 1.0
        feats.extend(s_exp_vec)

        s_pred = s_fg.get("predicted_labels", [])
        s_pred_vec = [0.0] * 10
        for lab in s_pred:
            s_pred_vec[hash(lab) % 10] = 1.0
        feats.extend(s_pred_vec)

        # Label overlap between control and share
        if ctrl_audit:
            c_fg = (ctrl_audit.get("audit", {}).get("outcome", {})
                    .get("fine_grained", {}) or {})
            c_exp = set(c_fg.get("expected_labels", []))
            s_exp_set = set(s_exp)
            c_pred = set(c_fg.get("predicted_labels", []))
            s_pred_set = set(s_pred)
            feats.append(1.0 if c_exp != s_exp_set else 0.0)
            feats.append(1.0 if c_pred != s_pred_set else 0.0)
        else:
            feats.extend([0.0, 0.0])
    else:
        feats.extend([0.0] * 43)

    return feats


def _pairwise_ranking(pred_tau: np.ndarray, true_tau: np.ndarray,
                      n_samples: int, rng: np.random.RandomState) -> float:
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


def _evaluate_feature_set(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int,
    seed: int,
) -> dict:
    """Evaluate a feature set with 5-fold CV using Ridge regression."""
    rng = np.random.RandomState(seed)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_rankings = []
    fold_pearsons = []
    fold_sign_accs = []
    fold_tau_stds = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Use Ridge (L2-regularized linear) for τ prediction
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)

        # Tau std
        fold_tau_stds.append(float(np.std(pred)))

        # Pearson
        if np.std(pred) > 0 and np.std(y_test) > 0:
            r = float(np.corrcoef(pred, y_test)[0, 1])
        else:
            r = 0.0
        fold_pearsons.append(r)

        # Sign accuracy
        sign_true = np.sign(y_test)
        sign_pred = np.sign(pred)
        fold_sign_accs.append(float(np.mean(sign_true == sign_pred)))

        # Ranking
        ranking = _pairwise_ranking(pred, y_test, 5000, rng)
        fold_rankings.append(ranking)

    return {
        "name": name,
        "n_features": X.shape[1],
        "ranking": round(float(np.mean(fold_rankings)), 4),
        "ranking_std": round(float(np.std(fold_rankings)), 4),
        "pearson_r": round(float(np.mean(fold_pearsons)), 4),
        "sign_accuracy": round(float(np.mean(fold_sign_accs)), 4),
        "tau_pred_std": round(float(np.mean(fold_tau_stds)), 4),
        "fold_rankings": [round(r, 4) for r in fold_rankings],
    }


def main() -> None:
    config = _load_config()
    data_cfg = config["data"]
    repr_cfg = config["audit"]["representation"]

    print("=" * 60)
    print("Audit 3: Current Representation Probe")
    print("=" * 60)

    # ── Load data ──
    all_records = _load_all_paired_records(data_cfg["all_paired_splits"])
    valid = [r for r in all_records if r.get("valid", False)]
    memory_pool = _load_memory_pool(data_cfg["memory_pool_path"])
    output_base = _PROJECT_ROOT / data_cfg["output_base"]

    print(f"\n  Valid records: {len(valid)}")

    # ── Extract 3 feature sets ──
    y = np.array([_get_tau(r) for r in valid], dtype=float)

    print("\n  Encoding Feature A (SMTR current)...")
    X_a = np.array([_encode_feature_a(r, memory_pool) for r in valid])

    print("  Encoding Feature B (SMTR + memory metadata)...")
    X_b = np.array([_encode_feature_b(r, memory_pool) for r in valid])

    print("  Encoding Feature C (SMTR + memory + execution outcome)...")
    X_c = np.array([_encode_feature_c(r, memory_pool, output_base)
                    for r in valid])

    print(f"\n  Feature A shape: {X_a.shape}")
    print(f"  Feature B shape: {X_b.shape}")
    print(f"  Feature C shape: {X_c.shape}")
    print(f"  τ: pos={int((y > 0).sum())}, neg={int((y < 0).sum())}, "
          f"zero={int((y == 0).sum())}")

    # ── Evaluate each ──
    seed = repr_cfg["seed"]
    n_folds = repr_cfg["n_folds"]

    results = {}
    for name, X_set in [("A_current_smtr", X_a),
                        ("B_plus_memory_meta", X_b),
                        ("C_plus_execution", X_c)]:
        print(f"\n  Evaluating {name}...")
        res = _evaluate_feature_set(
            name, X_set, y, n_folds=n_folds, seed=seed,
        )
        results[name] = res
        print(f"    ranking={res['ranking']:.4f} ± {res['ranking_std']:.4f}, "
              f"r={res['pearson_r']:.4f}, sign={res['sign_accuracy']:.4f}, "
              f"τ_std={res['tau_pred_std']:.4f}")

    # ── Verdict ──
    r_a = results["A_current_smtr"]["ranking"]
    r_b = results["B_plus_memory_meta"]["ranking"]
    r_c = results["C_plus_execution"]["ranking"]

    print(f"\n  ── Comparison ──")
    print(f"  A (current SMTR):     ranking = {r_a:.4f}")
    print(f"  B (+ memory meta):    ranking = {r_b:.4f} (Δ = {r_b - r_a:+.4f})")
    print(f"  C (+ execution):      ranking = {r_c:.4f} (Δ = {r_c - r_a:+.4f})")

    if r_c > 0.65:
        verdict = "REPRESENTATION_FAILURE"
        interpretation = ("Execution features help → current SMTR representation "
                          "is the bottleneck.  Need richer features.")
    elif r_a > 0.65:
        verdict = "REPRESENTATION_FAILURE"
        interpretation = "Current SMTR features work. Unexpected result."
    else:
        verdict = "ENVIRONMENT_MISMATCH"
        interpretation = ("No feature set achieves ranking >0.65.  "
                          "MARBLE signal is too weak or noisy for any model.")

    print(f"\n  Verdict: {verdict}")
    print(f"  Interpretation: {interpretation}")

    # ── Save ──
    out_dir = _THIS_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "audit": "representation_probe",
        "verdict": verdict,
        "interpretation": interpretation,
        "feature_sets": results,
    }
    out_path = out_dir / "representation_probe.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")

    print(f"\n{'=' * 60}")
    print(f"  RESULT: {verdict}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
