"""Train global baseline model (memory-only, no receiver info).

Model: τ̂(m) — predicts transfer effect from memory embedding only.
Architecture: MLP with hidden layers [128, 64].
Loss: MSE(τ̂, τ_true) via Adam solver with early stopping.

This model CANNOT distinguish receiver-specific effects.
If the SMTR hypothesis holds (τ(m,r₁) ≠ τ(m,r₂)),
this model should perform significantly worse than the
receiver-conditioned model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent / "src"))
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent))


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _train_mlp(
    X_train: np.ndarray,
    tau_train: np.ndarray,
    config: dict,
    label: str = "Model",
) -> MLPRegressor:
    """Train MLP with MSE loss via sklearn MLPRegressor."""
    model_cfg = config["model"]
    train_cfg = config["training"]

    # Use 15% of training data for validation (early stopping).
    X_tr, X_val, tau_tr, tau_val = train_test_split(
        X_train, tau_train,
        test_size=0.15,
        random_state=config["environment"]["seed"],
    )

    model = MLPRegressor(
        hidden_layer_sizes=(
            model_cfg["hidden_dim_1"],
            model_cfg["hidden_dim_2"],
        ),
        activation="relu",
        solver="adam",
        alpha=train_cfg["weight_decay"],
        learning_rate="adaptive",
        learning_rate_init=train_cfg["lr"],
        max_iter=train_cfg["epochs"],
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=train_cfg["patience"],
        batch_size=min(train_cfg["batch_size"], len(X_tr)),
        random_state=config["environment"]["seed"],
        verbose=False,
    )

    model.fit(X_tr, tau_tr)

    train_corr = np.corrcoef(model.predict(X_tr), tau_tr)[0, 1]
    val_corr = np.corrcoef(model.predict(X_val), tau_val)[0, 1]
    print(f"  [{label}] Converged at iteration {model.n_iter_}")
    print(f"  [{label}] Train Pearson: {train_corr:.4f}")
    print(f"  [{label}] Val Pearson:   {val_corr:.4f}")

    return model


def main() -> None:
    config = _load_config()

    print("=" * 60)
    print("Receiver Heterogeneity Stress Test — Global Model Training")
    print("=" * 60)

    # Load data.
    artifacts_dir = _THIS_DIR / "artifacts"
    train_data = np.load(artifacts_dir / "train_data.npz")

    X_train = train_data["mem_features"]  # (n, 16)
    tau_train = train_data["tau_true"]    # (n,)

    print(f"  Training samples: {len(X_train)}")
    print(f"  Input dim (memory only): {X_train.shape[1]}")
    print(f"  τ distribution: "
          f"+{(tau_train > 0).sum()}/-{(tau_train < 0).sum()}")

    # Train model.
    print("\n  Training...")
    model = _train_mlp(X_train, tau_train, config, "Global")

    # Save checkpoint.
    import joblib
    ckpt_path = artifacts_dir / "global_model.joblib"
    joblib.dump(model, ckpt_path)
    print(f"\n  Saved: {ckpt_path}")

    # Quick train accuracy.
    preds = model.predict(X_train)
    train_corr = float(np.corrcoef(preds, tau_train)[0, 1])
    train_sign_acc = float(
        (np.sign(preds) == np.sign(tau_train)).mean()
    )
    print(f"  Full train Pearson: {train_corr:.4f}")
    print(f"  Full train Sign accuracy: {train_sign_acc:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
