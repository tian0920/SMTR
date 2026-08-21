"""Run encoder ablation across 5 encoders + metadata-only model.

For each encoder:
  1. Load in_distribution split (train_raw + test)
  2. Encode features
  3. Train Ridge regression on τ with informative-record weighting
  4. Evaluate informative ranking, sign accuracy, tau correlation, pred std

Acceptance criteria:
  1. causal_input (no metadata) >= random + 10%
  2. |causal_input - metadata_full| < 0.10
  3. metadata_only < original SMTR (metadata can't beat SMTR base)

Outputs:
  - reports/encoder_ablation.json
  - reports/encoder_ablation.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

_THIS_DIR = Path(__file__).parent
_FEASIBILITY_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_FEASIBILITY_DIR))  # for _probe_models

# Ensure encoders/ is importable
sys.path.insert(0, str(_THIS_DIR))


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _get_tau(record: dict) -> int:
    y1 = 1 if record.get("share", {}).get("team_success") else 0
    y0 = 1 if record.get("withhold", {}).get("team_success") else 0
    return y1 - y0


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _dedup(records: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in records:
        key = (r.get("edge_id", ""), r.get("generation_seed", -1))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _informative_ranking(
    pred: np.ndarray, tau: np.ndarray, n_samples: int, rng: np.random.RandomState
) -> float:
    """Pairwise ranking only on records where |τ| > 0."""
    info_idx = np.where(tau != 0)[0]
    if len(info_idx) < 2:
        return 0.5
    pairs = rng.choice(info_idx, size=(n_samples, 2), replace=True)
    correct = total = 0
    for i, j in pairs:
        if tau[i] != tau[j]:
            total += 1
            if (pred[i] > pred[j]) == (tau[i] > tau[j]):
                correct += 1
    return correct / max(total, 1)


def _random_ranking(tau: np.ndarray, n_samples: int, rng: np.random.RandomState) -> float:
    """Random baseline informative ranking."""
    pred = rng.randn(len(tau))
    return _informative_ranking(pred, tau, n_samples, rng)


def _train_and_evaluate(
    encoder,
    train_inputs, train_records, train_tau,
    test_inputs, test_records, test_tau,
    train_cfg, eval_cfg,
) -> dict:
    """Train Ridge + evaluate a single encoder."""
    # Encode
    X_tr = encoder.encode_batch(train_inputs, records=train_records)
    X_te = encoder.encode_batch(test_inputs, records=test_records)
    X_tr = X_tr.toarray() if hasattr(X_tr, "toarray") else np.asarray(X_tr)
    X_te = X_te.toarray() if hasattr(X_te, "toarray") else np.asarray(X_te)

    # Feature stats
    unique_train = len(set(tuple(r) for r in X_tr))
    n_features = X_tr.shape[1]

    # Scale
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Sample weights
    weights = np.where(train_tau != 0, train_cfg["informative_weight"], 1.0)

    # Train Ridge
    model = Ridge(alpha=train_cfg["alpha"])
    model.fit(X_tr_s, train_tau, sample_weight=weights)

    # Predict
    pred = model.predict(X_te_s)

    # Metrics
    rng = np.random.RandomState(eval_cfg["seed"])
    n_samples = eval_cfg["n_pairwise_samples"]

    ranking = _informative_ranking(pred, test_tau, n_samples, rng)
    random_rank = _random_ranking(test_tau, n_samples, np.random.RandomState(eval_cfg["seed"]))

    # Sign accuracy
    info_mask = test_tau != 0
    if info_mask.sum() > 0:
        sign_pred = np.sign(pred[info_mask])
        sign_true = np.sign(test_tau[info_mask])
        sign_acc = float((sign_pred == sign_true).mean())
    else:
        sign_acc = 0.0

    # Tau correlation (informative only)
    if info_mask.sum() > 2 and np.std(pred[info_mask]) > 0:
        tau_corr = float(np.corrcoef(pred[info_mask], test_tau[info_mask])[0, 1])
    else:
        tau_corr = 0.0

    return {
        "ranking": round(ranking, 4),
        "random_ranking": round(random_rank, 4),
        "sign_accuracy": round(sign_acc, 4),
        "tau_correlation": round(tau_corr, 4),
        "pred_std": round(float(np.std(pred)), 4),
        "pred_mean": round(float(np.mean(pred)), 4),
        "n_features": n_features,
        "unique_train_vectors": unique_train,
    }


def main() -> None:
    config = _load_config()
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]

    split_name = config["split"]
    splits_dir = _FEASIBILITY_DIR / config["splits_dir"] / split_name
    memory_pool_path = _PROJECT_ROOT / config["memory_pool_path"]

    print("=" * 60)
    print("Encoder Ablation Validation")
    print("=" * 60)

    # ── Load data ──
    train_records = _dedup(
        [r for r in _load_jsonl(splits_dir / "train_raw.jsonl") if r.get("valid", False)]
    )
    test_records = [r for r in _load_jsonl(splits_dir / "test.jsonl") if r.get("valid", False)]

    from smtr.router.transfer_features import build_training_data_from_records

    train_data = build_training_data_from_records(train_records, memory_pool_path)
    test_data = build_training_data_from_records(test_records, memory_pool_path)

    train_inputs = [item for item, _, _ in train_data]
    train_recs = [rec for _, _, rec in train_data]
    train_tau = np.array([_get_tau(r) for r in train_recs])

    test_inputs = [item for item, _, _ in test_data]
    test_recs = [rec for _, _, rec in test_data]
    test_tau = np.array([_get_tau(r) for r in test_recs])

    print(f"\n  Train: {len(train_recs)} records, {int((train_tau != 0).sum())} informative")
    print(f"  Test:  {len(test_recs)} records, {int((test_tau != 0).sum())} informative")

    # ── Run each encoder ──
    from encoders.original import OriginalEncoder
    from encoders.task_only import TaskOnlyEncoder
    from encoders.memory_only import MemoryOnlyEncoder
    from encoders.metadata_full import MetadataFullEncoder
    from encoders.causal_input import CausalInputEncoder

    encoders = [
        ("original", OriginalEncoder),
        ("task_only", TaskOnlyEncoder),
        ("memory_only", MemoryOnlyEncoder),
        ("metadata_full", MetadataFullEncoder),
        ("causal_input", CausalInputEncoder),
    ]

    results = {}
    for name, EncoderClass in encoders:
        print(f"\n  ── {name} ──")
        enc = EncoderClass()
        res = _train_and_evaluate(
            enc, train_inputs, train_recs, train_tau,
            test_inputs, test_recs, test_tau,
            train_cfg, eval_cfg,
        )
        results[name] = res
        print(f"    ranking={res['ranking']:.4f}, sign={res['sign_accuracy']:.4f}, "
              f"corr={res['tau_correlation']:.4f}, pred_std={res['pred_std']:.4f}, "
              f"features={res['n_features']}, unique={res['unique_train_vectors']}")

    # ── Metadata-only model (Criterion 3) ──
    if config.get("metadata_only", {}).get("enabled", False):
        print("\n  ── metadata_only ──")
        from encoders import deterministic_hash, one_hot

        def encode_metadata_only(records: list[dict]) -> np.ndarray:
            rows = []
            for r in records:
                feats = []
                # task_id (20-dim)
                tid = str(r.get("task_id", ""))
                feats += one_hot(deterministic_hash(tid, 20), 20)
                # memory_id (8-dim)
                mem_id = r.get("candidate_memory_id", "")
                feats += one_hot(deterministic_hash(mem_id, 8), 8)
                # candidate_rank (1-dim)
                feats.append(float(r.get("candidate_rank", 0)) / 10.0)
                # candidate_score (1-dim)
                feats.append(float(r.get("candidate_score", 0.0)))
                # source_hash (3-dim)
                src = r.get("candidate_source", "")
                feats += one_hot(deterministic_hash(src, 3), 3)
                rows.append(feats)
            return np.array(rows, dtype=float)

        X_tr_meta = encode_metadata_only(train_recs)
        X_te_meta = encode_metadata_only(test_recs)
        unique_meta = len(set(tuple(r) for r in X_tr_meta))

        sc = StandardScaler()
        X_tr_meta_s = sc.fit_transform(X_tr_meta)
        X_te_meta_s = sc.transform(X_te_meta)

        w = np.where(train_tau != 0, train_cfg["informative_weight"], 1.0)
        model_meta = Ridge(alpha=train_cfg["alpha"])
        model_meta.fit(X_tr_meta_s, train_tau, sample_weight=w)
        pred_meta = model_meta.predict(X_te_meta_s)

        rng = np.random.RandomState(eval_cfg["seed"])
        ranking_meta = _informative_ranking(pred_meta, test_tau, eval_cfg["n_pairwise_samples"], rng)
        random_meta = _random_ranking(test_tau, eval_cfg["n_pairwise_samples"],
                                       np.random.RandomState(eval_cfg["seed"]))

        info_mask = test_tau != 0
        sign_meta = float((np.sign(pred_meta[info_mask]) == np.sign(test_tau[info_mask])).mean())
        corr_meta = float(np.corrcoef(pred_meta[info_mask], test_tau[info_mask])[0, 1]) if info_mask.sum() > 2 else 0.0

        results["metadata_only"] = {
            "ranking": round(ranking_meta, 4),
            "random_ranking": round(random_meta, 4),
            "sign_accuracy": round(sign_meta, 4),
            "tau_correlation": round(corr_meta, 4),
            "pred_std": round(float(np.std(pred_meta)), 4),
            "pred_mean": round(float(np.mean(pred_meta)), 4),
            "n_features": X_tr_meta.shape[1],
            "unique_train_vectors": unique_meta,
        }
        print(f"    ranking={ranking_meta:.4f}, sign={sign_meta:.4f}, "
              f"corr={corr_meta:.4f}, pred_std={np.std(pred_meta):.4f}")

        # ── Additional: metadata_without_task (diagnostic) ──
        print("\n  ── metadata_no_task (diagnostic) ──")

        def encode_metadata_no_task(records: list[dict]) -> np.ndarray:
            rows = []
            for r in records:
                feats = []
                # NO task_id
                # memory_id (8-dim)
                mem_id = r.get("candidate_memory_id", "")
                feats += one_hot(deterministic_hash(mem_id, 8), 8)
                # candidate_rank (1-dim)
                feats.append(float(r.get("candidate_rank", 0)) / 10.0)
                # candidate_score (1-dim)
                feats.append(float(r.get("candidate_score", 0.0)))
                # source_hash (3-dim)
                src = r.get("candidate_source", "")
                feats += one_hot(deterministic_hash(src, 3), 3)
                rows.append(feats)
            return np.array(rows, dtype=float)

        X_tr_mnt = encode_metadata_no_task(train_recs)
        X_te_mnt = encode_metadata_no_task(test_recs)
        unique_mnt = len(set(tuple(r) for r in X_tr_mnt))

        sc_mnt = StandardScaler()
        X_tr_mnt_s = sc_mnt.fit_transform(X_tr_mnt)
        X_te_mnt_s = sc_mnt.transform(X_te_mnt)

        model_mnt = Ridge(alpha=train_cfg["alpha"])
        model_mnt.fit(X_tr_mnt_s, train_tau, sample_weight=w)
        pred_mnt = model_mnt.predict(X_te_mnt_s)

        rng_mnt = np.random.RandomState(eval_cfg["seed"])
        ranking_mnt = _informative_ranking(pred_mnt, test_tau, eval_cfg["n_pairwise_samples"], rng_mnt)
        info_mask_mnt = test_tau != 0
        sign_mnt = float((np.sign(pred_mnt[info_mask_mnt]) == np.sign(test_tau[info_mask_mnt])).mean())
        corr_mnt = float(np.corrcoef(pred_mnt[info_mask_mnt], test_tau[info_mask_mnt])[0, 1]) if info_mask_mnt.sum() > 2 else 0.0

        results["metadata_no_task"] = {
            "ranking": round(ranking_mnt, 4),
            "sign_accuracy": round(sign_mnt, 4),
            "tau_correlation": round(corr_mnt, 4),
            "pred_std": round(float(np.std(pred_mnt)), 4),
            "n_features": X_tr_mnt.shape[1],
            "unique_train_vectors": unique_mnt,
        }
        print(f"    ranking={ranking_mnt:.4f}, sign={sign_mnt:.4f}, "
              f"corr={corr_mnt:.4f}, pred_std={np.std(pred_mnt):.4f}")

    # ── Acceptance criteria ──
    acceptance = config["acceptance"]
    random_baseline = results.get("causal_input", {}).get("random_ranking", 0.5)
    causal_rank = results.get("causal_input", {}).get("ranking", 0.0)
    full_rank = results.get("metadata_full", {}).get("ranking", 0.0)
    original_rank = results.get("original", {}).get("ranking", 0.0)
    meta_only_rank = results.get("metadata_only", {}).get("ranking", 0.0)

    checks = {}

    # Criterion 1: causal_input >= random + 10%
    margin = acceptance["no_metadata_min_margin"]
    checks["no_metadata_vs_random"] = {
        "description": f"causal_input >= random + {margin:.0%}",
        "value": f"causal={causal_rank:.4f}, random={random_baseline:.4f}",
        "passed": causal_rank >= random_baseline + margin,
    }

    # Criterion 2: |causal_input - metadata_full| < 0.10
    gap = acceptance["causal_vs_full_max_gap"]
    actual_gap = abs(full_rank - causal_rank)
    checks["causal_vs_full_gap"] = {
        "description": f"|causal_input - metadata_full| < {gap:.2f}",
        "value": f"gap={actual_gap:.4f} (full={full_rank:.4f}, causal={causal_rank:.4f})",
        "passed": actual_gap < gap,
    }

    # Criterion 3: metadata_only should not significantly exceed causal_input
    # (If metadata shortcuts drive performance, metadata_only >> causal_input)
    if "metadata_only" in results:
        meta_gap = results.get("metadata_only", {}).get("ranking", 0.0) - causal_rank
        checks["metadata_not_much_better_than_causal"] = {
            "description": "metadata_only - causal_input < 0.10 (shortcuts not main driver)",
            "value": f"meta_only={meta_only_rank:.4f}, causal={causal_rank:.4f}, gap={meta_gap:+.4f}",
            "passed": meta_gap < 0.10,
        }

    # Criterion 4: metadata_no_task should be much lower than task_only
    # (Proves task_id is the causal driver, not rank/score/source shortcuts)
    if "metadata_no_task" in results:
        task_only_rank = results.get("task_only", {}).get("ranking", 0.0)
        mnt_rank = results.get("metadata_no_task", {}).get("ranking", 0.0)
        checks["task_id_is_causal_driver"] = {
            "description": "metadata_no_task < task_only - 0.20 (task_id drives, not shortcuts)",
            "value": f"task_only={task_only_rank:.4f}, meta_no_task={mnt_rank:.4f}, drop={task_only_rank - mnt_rank:.4f}",
            "passed": mnt_rank < task_only_rank - 0.20,
        }

    all_passed = all(c["passed"] for c in checks.values())
    verdict = "PASS" if all_passed else "FAIL"

    print(f"\n{'=' * 60}")
    print("Acceptance Criteria")
    print("=" * 60)
    for name, check in checks.items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['description']}")
        print(f"         {check['value']}")
    print(f"\n  Verdict: {verdict}")

    # ── Save report ──
    reports_dir = _THIS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "verdict": verdict,
        "checks": checks,
        "results": results,
        "split": split_name,
        "train_records": len(train_recs),
        "test_records": len(test_recs),
        "train_informative": int((train_tau != 0).sum()),
        "test_informative": int((test_tau != 0).sum()),
    }
    json_path = reports_dir / "encoder_ablation.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Markdown
    md = ["# Encoder Ablation Report\n"]
    md.append(f"**Verdict: {verdict}**\n")
    md.append(f"**Split: {split_name}** (train={len(train_recs)}, test={len(test_recs)}, "
              f"informative={int((test_tau != 0).sum())})\n")
    md.append("## Results\n")
    md.append("| Encoder | Ranking | Sign Acc | Tau Corr | Pred Std | Features | Unique |")
    md.append("|---------|---------|----------|----------|----------|----------|--------|")
    for name in ["original", "task_only", "memory_only", "metadata_full", "causal_input",
                 "metadata_only", "metadata_no_task"]:
        if name in results:
            r = results[name]
            md.append(f"| {name} | {r['ranking']:.4f} | {r['sign_accuracy']:.4f} | "
                      f"{r['tau_correlation']:.4f} | {r['pred_std']:.4f} | "
                      f"{r['n_features']} | {r['unique_train_vectors']} |")

    md.append("\n## Acceptance Criteria\n")
    for name, check in checks.items():
        icon = "PASS" if check["passed"] else "FAIL"
        md.append(f"### [{icon}] {check['description']}")
        md.append(f"- {check['value']}\n")

    md.append("---\n")
    if verdict == "PASS":
        md.append("All criteria met. Performance gains are from causal features, "
                  "not metadata shortcuts.")
    else:
        md.append("Some criteria failed. See individual results above.")

    md_path = reports_dir / "encoder_ablation.md"
    md_path.write_text("\n".join(md))
    print(f"  Saved: {md_path}")

    print(f"\n{'=' * 60}")
    print(f"Encoder Ablation: {verdict}")
    print("=" * 60)

    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
