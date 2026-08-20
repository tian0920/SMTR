"""Train SMTR receiver-conditioned model τ̂(m, r).

Model: τ̂(m, r) — predicts transfer effect from memory + receiver.
Architecture: MLP with hidden layers [128, 64].
Loss: MSE(τ̂, τ_true) via Adam solver with early stopping.

This model can learn receiver-specific effects because it has
access to both memory and receiver embeddings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent / "src"))
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent))

from train_global import _load_config, _train_mlp


def main() -> None:
    config = _load_config()

    print("=" * 60)
    print("Receiver Heterogeneity Stress Test — SMTR Model Training")
    print("=" * 60)

    # Load data.
    artifacts_dir = _THIS_DIR / "artifacts"
    train_data = np.load(artifacts_dir / "train_data.npz")

    X_train = train_data["combined_features"]  # (n, 32)
    tau_train = train_data["tau_true"]          # (n,)

    print(f"  Training samples: {len(X_train)}")
    print(f"  Input dim (memory + receiver): {X_train.shape[1]}")
    print(f"  τ distribution: "
          f"+{(tau_train > 0).sum()}/-{(tau_train < 0).sum()}")

    # Train model.
    print("\n  Training...")
    model = _train_mlp(X_train, tau_train, config, "SMTR")

    # Save checkpoint.
    import joblib
    ckpt_path = artifacts_dir / "receiver_model.joblib"
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
