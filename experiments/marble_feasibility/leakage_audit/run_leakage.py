"""Metadata Leakage Audit (v2 — corrected experimental design).

Tests whether EnhancedEncoder exploits non-causal identity shortcuts.

Key insight from encoder ablation:
  task_only = 0.8840 (task_id is causal driver)
  metadata_no_task = 0.4134 (metadata without task is near random)
  → task_id is a CAUSAL feature, not a shortcut.

Experiment A: Metadata-only WITHOUT task_id (rank + score + source + memory_id)
  → This isolates suspected shortcuts from the causal task signal.
  → If ranking > SMTR - 10%: severe shortcut concern.

Experiment B: Full shuffle (permute ALL metadata including task_id)
  → Performance should collapse to near-random.

Experiment C: Remove candidate_score from full model
  → candidate_score is the highest-risk field.

PASS criteria:
  1. metadata_only (no task) < SMTR - 10%
  2. shuffle drop >= 20%
  3. remove candidate_score drop < 10%
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

_THIS_DIR = Path(__file__).parent
_FEASIBILITY_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _FEASIBILITY_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_FEASIBILITY_DIR))

import hashlib


def _deterministic_hash(s: str, mod: int) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % mod


def _one_hot(index: int, dim: int) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(dim)]


def _get_tau(record: dict) -> int:
    y1 = 1 if record.get("share", {}).get("team_success") else 0
    y0 = 1 if record.get("withhold", {}).get("team_success") else 0
    return y1 - y0


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _dedup(records: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in records:
        key = (r.get("edge_id", ""), r.get("generation_seed", -1))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _informative_ranking(pred, tau, n_samples, rng):
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


def _train_eval(X_tr, X_te, y_tr, y_te, alpha=0.01, w_info=5.0, seed=42):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    w = np.where(y_tr != 0, w_info, 1.0)
    model = Ridge(alpha=alpha)
    model.fit(X_tr_s, y_tr, sample_weight=w)
    pred = model.predict(X_te_s)
    rng = np.random.RandomState(seed)
    ranking = _informative_ranking(pred, y_te, 5000, rng)
    return ranking, pred


def _encode_metadata_no_task(records: list[dict]) -> np.ndarray:
    """Experiment A: metadata WITHOUT task_id.
    memory_id(8) + rank(1) + score(1) + source(3) = 13-dim.
    Tests: do shortcut fields alone predict tau?
    """
    rows = []
    for r in records:
        feats = []
        # NO task_id
        mem_id = r.get("candidate_memory_id", "")
        feats += _one_hot(_deterministic_hash(mem_id, 8), 8)
        feats.append(float(r.get("candidate_rank", 0)) / 10.0)
        feats.append(float(r.get("candidate_score", 0.0)))
        src = r.get("candidate_source", "")
        feats += _one_hot(_deterministic_hash(src, 3), 3)
        rows.append(feats)
    return np.array(rows, dtype=float)


def _encode_full_metadata(records: list[dict]) -> np.ndarray:
    """Full metadata: task(20) + memory_id(8) + rank(1) + score(1) + source(3) = 33-dim."""
    rows = []
    for r in records:
        feats = []
        tid = str(r.get("task_id", ""))
        feats += _one_hot(_deterministic_hash(tid, 20), 20)
        mem_id = r.get("candidate_memory_id", "")
        feats += _one_hot(_deterministic_hash(mem_id, 8), 8)
        feats.append(float(r.get("candidate_rank", 0)) / 10.0)
        feats.append(float(r.get("candidate_score", 0.0)))
        src = r.get("candidate_source", "")
        feats += _one_hot(_deterministic_hash(src, 3), 3)
        rows.append(feats)
    return np.array(rows, dtype=float)


def _encode_shuffled_all(records: list[dict], rng: np.random.RandomState) -> np.ndarray:
    """Experiment B: shuffle ALL metadata fields including task_id.
    Performance should collapse to near-random."""
    task_ids = [str(r.get("task_id", "")) for r in records]
    mem_ids = [r.get("candidate_memory_id", "") for r in records]
    sources = [r.get("candidate_source", "") for r in records]
    ranks = [float(r.get("candidate_rank", 0)) for r in records]
    scores = [float(r.get("candidate_score", 0.0)) for r in records]

    rng.shuffle(task_ids)
    rng.shuffle(mem_ids)
    rng.shuffle(sources)
    rng.shuffle(ranks)
    rng.shuffle(scores)

    rows = []
    for i in range(len(records)):
        feats = []
        feats += _one_hot(_deterministic_hash(task_ids[i], 20), 20)
        feats += _one_hot(_deterministic_hash(mem_ids[i], 8), 8)
        feats.append(ranks[i] / 10.0)
        feats.append(scores[i])
        feats += _one_hot(_deterministic_hash(sources[i], 3), 3)
        rows.append(feats)
    return np.array(rows, dtype=float)


def _encode_full_smtr(records: list[dict]) -> np.ndarray:
    """SMTR full: task(20) + rank(1) + score(1) + source(3) + mem_base(5) + mem_id(8) = 38-dim."""
    rows = []
    for r in records:
        feats = []
        tid = str(r.get("task_id", ""))
        feats += _one_hot(_deterministic_hash(tid, 20), 20)
        feats.append(float(r.get("candidate_rank", 0)) / 10.0)
        feats.append(float(r.get("candidate_score", 0.0)))
        src = r.get("candidate_source", "")
        feats += _one_hot(_deterministic_hash(src, 3), 3)
        mem_id = r.get("candidate_memory_id", "")
        mem_base = "-".join(mem_id.split("-")[:2]) if "-" in mem_id else mem_id
        feats += _one_hot(_deterministic_hash(mem_base, 5), 5)
        feats += _one_hot(_deterministic_hash(mem_id, 8), 8)
        rows.append(feats)
    return np.array(rows, dtype=float)


def _encode_no_score(records: list[dict]) -> np.ndarray:
    """Experiment C: Full SMTR WITHOUT candidate_score = 37-dim."""
    rows = []
    for r in records:
        feats = []
        tid = str(r.get("task_id", ""))
        feats += _one_hot(_deterministic_hash(tid, 20), 20)
        feats.append(float(r.get("candidate_rank", 0)) / 10.0)
        # NO candidate_score
        src = r.get("candidate_source", "")
        feats += _one_hot(_deterministic_hash(src, 3), 3)
        mem_id = r.get("candidate_memory_id", "")
        mem_base = "-".join(mem_id.split("-")[:2]) if "-" in mem_id else mem_id
        feats += _one_hot(_deterministic_hash(mem_base, 5), 5)
        feats += _one_hot(_deterministic_hash(mem_id, 8), 8)
        rows.append(feats)
    return np.array(rows, dtype=float)


def _encode_task_only(records: list[dict]) -> np.ndarray:
    """Reference: task_id only (20-dim)."""
    rows = []
    for r in records:
        tid = str(r.get("task_id", ""))
        rows.append(_one_hot(_deterministic_hash(tid, 20), 20))
    return np.array(rows, dtype=float)


def main() -> None:
    print("=" * 60)
    print("Metadata Leakage Audit (v2)")
    print("=" * 60)

    splits_dir = _FEASIBILITY_DIR / "splits" / "in_distribution"
    train_records = _dedup(
        [r for r in _load_jsonl(splits_dir / "train_raw.jsonl") if r.get("valid", False)]
    )
    test_records = [r for r in _load_jsonl(splits_dir / "test.jsonl") if r.get("valid", False)]
    y_tr = np.array([_get_tau(r) for r in train_records])
    y_te = np.array([_get_tau(r) for r in test_records])

    print(f"\n  Train: {len(train_records)}, Test: {len(test_records)}")
    print(f"  Informative: train={int((y_tr != 0).sum())}, test={int((y_te != 0).sum())}")

    # ── Experiment A: Metadata-only WITHOUT task_id ──
    print("\n  ── A: Metadata-only (NO task_id) ──")
    X_tr_a = _encode_metadata_no_task(train_records)
    X_te_a = _encode_metadata_no_task(test_records)
    rank_a, _ = _train_eval(X_tr_a, X_te_a, y_tr, y_te)
    print(f"    ranking = {rank_a:.4f}")

    # ── Experiment B: Full shuffle (ALL fields including task_id) ──
    print("\n  ── B: Full shuffle (all metadata) ──")
    rng_b = np.random.RandomState(42)
    X_tr_b = _encode_shuffled_all(train_records, rng_b)
    X_te_b = _encode_shuffled_all(test_records, np.random.RandomState(42))
    rank_b, _ = _train_eval(X_tr_b, X_te_b, y_tr, y_te)
    print(f"    ranking = {rank_b:.4f}")

    # ── Experiment C: Full vs Without candidate_score ──
    print("\n  ── C: Full SMTR model ──")
    X_tr_full = _encode_full_smtr(train_records)
    X_te_full = _encode_full_smtr(test_records)
    rank_full, _ = _train_eval(X_tr_full, X_te_full, y_tr, y_te)
    print(f"    ranking = {rank_full:.4f}")

    print("\n  ── C: Without candidate_score ──")
    X_tr_ns = _encode_no_score(train_records)
    X_te_ns = _encode_no_score(test_records)
    rank_ns, _ = _train_eval(X_tr_ns, X_te_ns, y_tr, y_te)
    print(f"    ranking = {rank_ns:.4f}")

    # ── References ──
    print("\n  ── References ──")
    X_tr_to = _encode_task_only(train_records)
    X_te_to = _encode_task_only(test_records)
    rank_to, _ = _train_eval(X_tr_to, X_te_to, y_tr, y_te)
    print(f"    task_only = {rank_to:.4f}")

    X_tr_fm = _encode_full_metadata(train_records)
    X_te_fm = _encode_full_metadata(test_records)
    rank_fm, _ = _train_eval(X_tr_fm, X_te_fm, y_tr, y_te)
    print(f"    full_metadata (with task) = {rank_fm:.4f}")

    smtr_rank = 0.8433  # from encoder_ablation metadata_full
    print(f"    SMTR reference (ablation metadata_full) = {smtr_rank:.4f}")

    # ── Acceptance criteria ──
    checks = {}

    # 1. metadata_only (no task) < SMTR - 10%
    checks["metadata_only_below_smtr"] = {
        "description": "metadata_only (no task_id) < SMTR - 0.10",
        "value": f"meta_only_no_task={rank_a:.4f}, SMTR={smtr_rank:.4f}, threshold={smtr_rank - 0.10:.4f}",
        "passed": rank_a < smtr_rank - 0.10,
    }

    # 2. Full shuffle drops >= 20% from full_metadata
    shuffle_drop = rank_fm - rank_b
    checks["shuffle_drop"] = {
        "description": "Shuffle ALL metadata drop >= 0.20",
        "value": f"full_meta={rank_fm:.4f}, shuffled={rank_b:.4f}, drop={shuffle_drop:.4f}",
        "passed": shuffle_drop >= 0.20,
    }

    # 3. Remove candidate_score: drop < 10%
    score_drop = rank_full - rank_ns
    checks["remove_score_drop"] = {
        "description": "Remove candidate_score drop < 0.10",
        "value": f"full={rank_full:.4f}, no_score={rank_ns:.4f}, drop={score_drop:.4f}",
        "passed": abs(score_drop) < 0.10,
    }

    all_passed = all(c["passed"] for c in checks.values())
    verdict = "PASS" if all_passed else "FAIL"

    print(f"\n{'=' * 60}")
    print("Leakage Audit Results")
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
        "results": {
            "metadata_only_no_task": round(rank_a, 4),
            "shuffled_all": round(rank_b, 4),
            "full_smtr": round(rank_full, 4),
            "no_candidate_score": round(rank_ns, 4),
            "task_only": round(rank_to, 4),
            "full_metadata_with_task": round(rank_fm, 4),
            "smtr_reference": smtr_rank,
        },
        "interpretation": {
            "task_id_is_causal": rank_to > 0.80,
            "metadata_without_task_is_weak": rank_a < 0.55,
            "shuffle_destroys_signal": rank_b < rank_fm - 0.15,
            "candidate_score_not_critical": abs(score_drop) < 0.05,
        },
    }
    json_path = reports_dir / "leakage_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Markdown
    md = ["# Metadata Leakage Audit Report (v2)\n"]
    md.append(f"**Verdict: {verdict}**\n")
    md.append("## Key Insight\n")
    md.append("Encoder ablation proved task_id is a **causal driver** (task_only=0.88, "
              "metadata_no_task=0.41). This audit isolates whether the remaining metadata "
              "fields (rank, score, source, memory_id) constitute non-causal shortcuts.\n")
    md.append("## Results\n")
    md.append("| Model | Ranking | Notes |")
    md.append("|-------|---------|-------|")
    md.append(f"| Metadata-only (no task_id) | {rank_a:.4f} | rank+score+source+mem_id only |")
    md.append(f"| Shuffled ALL metadata | {rank_b:.4f} | including task_id shuffled |")
    md.append(f"| Full SMTR | {rank_full:.4f} | all features |")
    md.append(f"| Without candidate_score | {rank_ns:.4f} | full minus score |")
    md.append(f"| Task-only (reference) | {rank_to:.4f} | task_id only |")
    md.append(f"| Full metadata (with task) | {rank_fm:.4f} | task+rank+score+source+mem_id |")
    md.append(f"| SMTR reference (ablation) | {smtr_rank:.4f} | from encoder ablation |")

    md.append("\n## Acceptance Criteria\n")
    for name, check in checks.items():
        icon = "PASS" if check["passed"] else "FAIL"
        md.append(f"### [{icon}] {check['description']}")
        md.append(f"- {check['value']}\n")

    md.append("## Interpretation\n")
    interp = report["interpretation"]
    md.append(f"- **task_id is causal**: {interp['task_id_is_causal']} (task_only={rank_to:.4f})")
    md.append(f"- **metadata without task is weak**: {interp['metadata_without_task_is_weak']} "
              f"(meta_no_task={rank_a:.4f})")
    md.append(f"- **shuffle destroys signal**: {interp['shuffle_destroys_signal']} "
              f"(drop={rank_fm - rank_b:.4f})")
    md.append(f"- **candidate_score not critical**: {interp['candidate_score_not_critical']} "
              f"(drop={score_drop:.4f})")

    md.append("\n---\n")
    if verdict == "PASS":
        md.append("**Conclusion**: No significant metadata leakage detected. "
                  "Performance gains come from causal task context, not identity shortcuts.")
    else:
        md.append("**Conclusion**: Some leakage concerns detected. Review individual criteria.")

    md_path = reports_dir / "leakage_report.md"
    md_path.write_text("\n".join(md))
    print(f"  Saved: {md_path}")

    print(f"\n{'=' * 60}")
    print(f"Leakage Audit: {verdict}")
    print("=" * 60)

    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
