"""Train models at each intervention budget level.

For each budget ratio and seed:
  - budget > 0: Train MLP on intervention subset → predict τ(m,r)
  - budget = 0: Outcome-only baseline → predict Y_withhold from (m,r)

Model architecture: MLP [128, 64] (same as receiver heterogeneity).
Input: concat(memory_embedding, receiver_embedding) = 32-dim.
Loss: MSE via Adam solver.

Saves one model per (ratio, seed) combination.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent / "src"))
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent))


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
    label: str = "Model",
) -> MLPRegressor:
    """Train MLP regressor with early stopping."""
    model_cfg = config["model"]
    train_cfg = config["training"]

    if len(X_train) < 10:
        # Too few samples — return a trivial model.
        model = MLPRegressor(
            hidden_layer_sizes=(8,),
            max_iter=1,
            random_state=config["environment"]["seed"],
        )
        model.fit(X_train, y_train)
        print(f"  [{label}] Too few samples ({len(X_train)}), "
              f"trivial model trained")
        return model

    # For small datasets, disable early stopping to avoid premature
    # convergence due to noisy validation metrics.
    use_early_stopping = len(X_train) >= 400
    if use_early_stopping:
        val_frac = min(0.15, 25 / max(len(X_train), 1))
    else:
        val_frac = 0.0  # No validation split; train on all data.

    X_tr, X_val, y_tr, y_val = (
        train_test_split(
            X_train, y_train,
            test_size=val_frac,
            random_state=config["environment"]["seed"],
        )
        if val_frac > 0
        else (X_train, None, y_train, None)
    )

    # Scale regularization for small sample sizes.
    base_alpha = train_cfg["weight_decay"]
    alpha = base_alpha * min(1.0, len(X_train) / 500.0)

    model = MLPRegressor(
        hidden_layer_sizes=(
            model_cfg["hidden_dim_1"],
            model_cfg["hidden_dim_2"],
        ),
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate="adaptive",
        learning_rate_init=train_cfg["lr"],
        max_iter=train_cfg["epochs"],
        early_stopping=use_early_stopping,
        validation_fraction=val_frac if use_early_stopping else 0.0,
        n_iter_no_change=train_cfg["patience"],
        batch_size=min(train_cfg["batch_size"], len(X_tr)),
        random_state=config["environment"]["seed"],
        verbose=False,
    )
    model.fit(X_tr, y_tr)

    if np.std(y_tr) > 1e-8:
        train_corr = float(np.corrcoef(model.predict(X_tr), y_tr)[0, 1])
    else:
        train_corr = 0.0
    print(f"  [{label}] Converged at iter {model.n_iter_}, "
          f"train corr={train_corr:.4f}")
    return model


def main() -> None:
    config = _load_config()
    budget_cfg = config["budget"]

    ratios = budget_cfg["ratios"]
    seeds = budget_cfg["seeds"]

    print("=" * 60)
    print("Intervention Budget — Model Training")
    print("=" * 60)

    # Load data.
    artifacts_dir = _THIS_DIR / "artifacts"
    train_data = np.load(artifacts_dir / "train_data.npz")
    budget_masks = np.load(artifacts_dir / "budget_masks.npz")

    X_all = train_data["combined_features"]   # (n, 32)
    tau_all = train_data["tau_true"]           # (n,)
    y_withhold_all = train_data["y_withhold"]  # (n,)

    print(f"  Total training samples: {len(X_all)}")
    print(f"  Input dim: {X_all.shape[1]}")
    print(f"  Budget ratios: {ratios}, Seeds: {seeds}\n")

    for ratio in ratios:
        for seed in seeds:
            mask_key = f"mask_{ratio}_{seed}"
            mask = budget_masks[mask_key].astype(bool)
            n_intervention = int(mask.sum())

            label = f"budget={ratio:.2f}_seed={seed}"
            print(f"  Training {label} "
                  f"({n_intervention} intervention samples)...")

            if ratio <= 0.0:
                # Outcome-only baseline: predict Y_withhold from (m,r).
                X_train = X_all
                y_train = y_withhold_all
                model_type = "outcome_only"
            else:
                # SMTR: predict tau from intervention subset.
                X_train = X_all[mask]
                y_train = tau_all[mask]
                model_type = "smtr"

            model = _train_mlp(X_train, y_train, config, label)

            # Save.
            ckpt_name = f"model_budget_{ratio:.2f}_seed_{seed}.joblib"
            joblib.dump(model, artifacts_dir / ckpt_name)

            # Quick check on a small sample.
            sample_preds = model.predict(X_all[:20])
            if model_type == "smtr" and np.std(tau_all[:20]) > 1e-8:
                sample_corr = float(
                    np.corrcoef(sample_preds, tau_all[:20])[0, 1]
                )
            else:
                sample_corr = float("nan")
            print(f"    → Saved {ckpt_name} "
                  f"(sample corr={sample_corr:.4f})\n")

    # Also save model metadata.
    meta = {
        "ratios": ratios,
        "seeds": seeds,
        "model_type": {
            f"{ratio}_{seed}": (
                "outcome_only" if ratio <= 0.0 else "smtr"
            )
            for ratio in ratios
            for seed in seeds
        },
    }
    import json
    with open(artifacts_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("All models trained and saved.")
    print("Done.")


if __name__ == "__main__":
    main()
