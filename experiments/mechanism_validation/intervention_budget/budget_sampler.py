"""Budget sampler: create intervention masks for different budget ratios.

For each budget ratio r and seed s, generates:
  - intervention_mask: boolean array of length n_train
    True = sample has paired (expose + withhold) intervention
    False = sample only has observational (Y_withhold)

Key constraints:
  - Test set is FIXED across all budgets (no dataset difference).
  - Different seeds produce different masks for robustness.

Saves:
  artifacts/budget_masks.npz
    Keys: mask_{ratio}_{seed} for each combination
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent / "src"))
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent))


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def generate_intervention_mask(
    n_samples: int,
    budget_ratio: float,
    seed: int,
    tau_true: np.ndarray | None = None,
) -> np.ndarray:
    """Generate boolean intervention mask with stratified sampling.

    When tau_true is provided, stratifies by sign to ensure balanced
    representation of positive and negative effects in each budget.

    Args:
        n_samples: Total number of training samples.
        budget_ratio: Fraction of samples with intervention [0, 1].
        seed: Random seed for reproducibility.
        tau_true: Optional ground truth tau for stratification.

    Returns:
        Boolean array of shape (n_samples,).
        True = intervention available.
    """
    if budget_ratio <= 0.0:
        return np.zeros(n_samples, dtype=bool)
    if budget_ratio >= 1.0:
        return np.ones(n_samples, dtype=bool)

    rng = np.random.RandomState(seed)
    n_intervention = int(n_samples * budget_ratio)
    mask = np.zeros(n_samples, dtype=bool)

    if tau_true is not None:
        # Stratified sampling: balance τ+ and τ− in the intervention set.
        pos_idx = np.where(tau_true > 0)[0]
        neg_idx = np.where(tau_true < 0)[0]
        n_pos = int(n_intervention * len(pos_idx) / n_samples)
        n_neg = n_intervention - n_pos
        # Clamp to available.
        n_pos = min(n_pos, len(pos_idx))
        n_neg = min(n_neg, len(neg_idx))
        sel_pos = rng.choice(pos_idx, size=n_pos, replace=False)
        sel_neg = rng.choice(neg_idx, size=n_neg, replace=False)
        indices = np.concatenate([sel_pos, sel_neg])
    else:
        indices = rng.choice(n_samples, size=n_intervention, replace=False)

    mask[indices] = True
    return mask


def main() -> None:
    config = _load_config()
    budget_cfg = config["budget"]
    n_train = config["data"]["n_train"]

    ratios = budget_cfg["ratios"]
    seeds = budget_cfg["seeds"]

    print("=" * 60)
    print("Intervention Budget — Budget Sampler")
    print("=" * 60)
    print(f"  Training samples: {n_train}")
    print(f"  Budget ratios: {ratios}")
    print(f"  Seeds: {seeds}")

    # Load training data for stratification.
    artifacts_dir = _THIS_DIR / "artifacts"
    train_data = np.load(artifacts_dir / "train_data.npz")
    tau_true = train_data["tau_true"]

    save_dict: dict[str, np.ndarray] = {}

    for ratio in ratios:
        for seed in seeds:
            mask = generate_intervention_mask(
                n_train, ratio, seed, tau_true=tau_true,
            )
            key = f"mask_{ratio}_{seed}"
            save_dict[key] = mask
            n_int = int(mask.sum())
            # Report stratification balance.
            if n_int > 0:
                n_pos = int((tau_true[mask] > 0).sum())
                n_neg = n_int - n_pos
                balance = f"(τ+={n_pos}, τ-={n_neg})"
            else:
                balance = "(outcome-only)"
            print(f"  ratio={ratio:.2f}, seed={seed}: "
                  f"{n_int}/{n_train} interventions "
                  f"({n_int/n_train:.1%}) {balance}")

    # Save.
    out_dir = _THIS_DIR / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "budget_masks.npz"
    np.savez(out_path, **save_dict)
    print(f"\n  Saved: {out_path}")
    print(f"  Total masks: {len(save_dict)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
