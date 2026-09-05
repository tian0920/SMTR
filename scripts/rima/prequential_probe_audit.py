"""P1-2/3/4: prequential probe audit (injection-independent).

Evaluates critic prediction quality from ``probe_events.jsonl`` using
the PRE-probe prediction vs the post-probe observed causal effect::

    e_t = |predicted_mu_pre_probe - observed_tau|

This metric is defined for every valid probe and does NOT depend on
scored execution injection (P1-2).

Outputs ``prequential_probe_audit.json`` in the stream directory:

* P1-2 transfer MAE overall + early/middle/late windows.
* P1-3 prequential metrics (MAE, RMSE, Spearman, Pearson, sign
  accuracy, positive precision/recall) overall and per critic version.
* P1-4 LCB decomposition: fraction(mu>0), fraction(LCB>0), mean
  penalty beta*sigma, and the mean-failure vs uncertainty-failure
  split.

Usage::

    python scripts/rima/prequential_probe_audit.py <stream_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _valid_pairs(probe_events: list[dict]) -> list[dict[str, float]]:
    """Probe events carrying both a pre-probe mu and an observed tau."""
    pairs = []
    for ev in probe_events:
        mu = ev.get("predicted_mu_pre_probe")
        tau = ev.get("observed_tau")
        if mu is None or tau is None:
            continue
        pairs.append(
            {
                "mu": float(mu),
                "tau": float(tau),
                "sigma": ev.get("predicted_sigma_pre_probe"),
                "lcb": ev.get("predicted_lcb_pre_probe"),
                "version": ev.get("critic_version_pre_probe", 1),
                "task_position": ev.get("task_position"),
            }
        )
    return pairs


def _prequential_metrics(pairs: list[dict[str, float]]) -> dict[str, Any]:
    """P1-3: prediction-quality metrics for a set of (mu, tau) pairs."""
    n = len(pairs)
    if n == 0:
        return {"n": 0}

    mu = np.array([p["mu"] for p in pairs])
    tau = np.array([p["tau"] for p in pairs])
    errors = mu - tau

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    spearman = spearmanr(mu, tau).statistic if n >= 2 else None
    pearson = pearsonr(mu, tau).statistic if n >= 2 and np.std(mu) > 0 and np.std(tau) > 0 else None

    pred_pos = mu > 0.0
    obs_pos = tau > 0.0
    sign_accuracy = float(np.mean(pred_pos == obs_pos))
    tp = int(np.sum(pred_pos & obs_pos))
    fp = int(np.sum(pred_pos & ~obs_pos))
    fn = int(np.sum(~pred_pos & obs_pos))
    positive_precision = tp / (tp + fp) if (tp + fp) > 0 else None
    positive_recall = tp / (tp + fn) if (tp + fn) > 0 else None

    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "spearman": None if spearman is None else float(spearman),
        "pearson": None if pearson is None else float(pearson),
        "sign_accuracy": sign_accuracy,
        "positive_precision": positive_precision,
        "positive_recall": positive_recall,
    }


def _lcb_decomposition(pairs: list[dict[str, float]]) -> dict[str, Any]:
    """P1-4: mean failure vs uncertainty failure decomposition."""
    n = len(pairs)
    if n == 0:
        return {"n": 0}

    mu_pos = sum(1 for p in pairs if p["mu"] > 0.0)
    lcb_values = [p["lcb"] for p in pairs if p["lcb"] is not None]
    lcb_pos = sum(1 for v in lcb_values if v > 0.0)

    sigma_values = [p["sigma"] for p in pairs if p["sigma"] is not None]
    penalty_values = [
        p["mu"] - p["lcb"]
        for p in pairs
        if p["lcb"] is not None
    ]

    mean_failure = sum(1 for p in pairs if p["mu"] <= 0.0)
    uncertainty_failure = sum(
        1 for p in pairs if p["mu"] > 0.0 and (p["lcb"] is not None and p["lcb"] <= 0.0)
    )

    return {
        "n": n,
        "fraction_mu_positive": mu_pos / n,
        "fraction_lcb_positive": lcb_pos / len(lcb_values) if lcb_values else None,
        "mean_sigma": _mean(sigma_values),
        "mean_penalty_beta_sigma": _mean(penalty_values),
        "mean_failure_count": mean_failure,
        "uncertainty_failure_count": uncertainty_failure,
        "mean_failure_fraction": mean_failure / n,
        "uncertainty_failure_fraction": uncertainty_failure / n,
    }


def _split_windows(n_tasks: int) -> tuple[range, range, range]:
    third = max(1, n_tasks // 3)
    return range(0, third), range(third, n_tasks - third), range(n_tasks - third, n_tasks)


def prequential_probe_audit(stream_dir: str | Path) -> dict[str, Any]:
    stream_dir = Path(stream_dir)
    probe_events = _load_jsonl(stream_dir / "probe_events.jsonl")
    task_records = _load_jsonl(stream_dir / "tasks.jsonl")
    n_tasks = len(task_records)

    pairs = _valid_pairs(probe_events)

    result: dict[str, Any] = {
        "stream_dir": str(stream_dir),
        "n_probe_events": len(probe_events),
        "n_valid_pairs": len(pairs),
        "pre_probe_fields_present": len(pairs) > 0,
    }

    if not pairs:
        result["status"] = "INSUFFICIENT_DATA"
        result["note"] = (
            "No probe events carry predicted_mu_pre_probe; runs before the "
            "P1-1 fix cannot be audited prequentially."
        )
        out_path = stream_dir / "prequential_probe_audit.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"INSUFFICIENT_DATA: no pre-probe predictions in {stream_dir}")
        return result

    # --- P1-2: Transfer MAE overall + windows (E/M/L) ---
    abs_errors = [abs(p["mu"] - p["tau"]) for p in pairs]
    transfer_mae = {"mae": float(np.mean(abs_errors)), "n": len(pairs)}

    positions = [p["task_position"] for p in pairs if p["task_position"] is not None]
    if n_tasks > 0 and positions:
        early, middle, late = _split_windows(n_tasks)
        for name, window in (("early", early), ("middle", middle), ("late", late)):
            win_pairs = [
                p for p in pairs
                if p["task_position"] is not None and p["task_position"] in window
            ]
            transfer_mae[name] = (
                {
                    "mae": float(
                        np.mean([abs(p["mu"] - p["tau"]) for p in win_pairs])
                    ),
                    "n": len(win_pairs),
                }
                if win_pairs
                else None
            )
    result["transfer_mae"] = transfer_mae

    # --- P1-3: prequential metrics overall + per critic version ---
    result["prequential_overall"] = _prequential_metrics(pairs)
    by_version: dict[int, list[dict[str, float]]] = {}
    for p in pairs:
        by_version.setdefault(int(p["version"]), []).append(p)
    result["prequential_by_critic_version"] = {
        f"v{v}": _prequential_metrics(vp) for v, vp in sorted(by_version.items())
    }

    # --- P1-4: LCB decomposition overall + per critic version ---
    result["lcb_decomposition"] = _lcb_decomposition(pairs)
    result["lcb_decomposition_by_critic_version"] = {
        f"v{v}": _lcb_decomposition(vp) for v, vp in sorted(by_version.items())
    }

    result["status"] = "OK"

    out_path = stream_dir / "prequential_probe_audit.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Prequential probe audit for {stream_dir}")
    print(f"  probe_events={len(probe_events)} valid_pairs={len(pairs)}")
    print(f"  Transfer MAE: {transfer_mae['mae']:.4f} (n={transfer_mae['n']})")
    for name in ("early", "middle", "late"):
        entry = transfer_mae.get(name)
        if entry:
            print(f"    {name}: mae={entry['mae']:.4f} (n={entry['n']})")
    overall = result["prequential_overall"]
    print(
        f"  Overall: rmse={overall['rmse']:.4f} "
        f"spearman={overall['spearman']} sign_acc={overall['sign_accuracy']:.3f}"
    )
    lcb = result["lcb_decomposition"]
    print(
        f"  LCB: frac(mu>0)={lcb['fraction_mu_positive']:.3f} "
        f"frac(LCB>0)={lcb['fraction_lcb_positive']} "
        f"mean_penalty={lcb['mean_penalty_beta_sigma']}"
    )
    print(f"\nWrote {out_path}")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/rima/prequential_probe_audit.py <stream_dir>")
        sys.exit(1)
    prequential_probe_audit(sys.argv[1])
