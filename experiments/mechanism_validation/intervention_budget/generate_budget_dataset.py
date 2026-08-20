"""Generate budget dataset using the receiver heterogeneity environment.

Reuses the same ground truth:
  τ(m,r) = sign(z_m^T W z_r)

Generates:
  - env_data.npz: embeddings + W + tau_matrix
  - train_data.npz: 1000 training samples
  - test_data.npz: 400 test samples (FIXED across all budgets)

The environment parameters are identical to receiver_heterogeneity
to ensure comparability.
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


def _generate_embeddings(
    n_memories: int,
    n_receivers: int,
    embedding_dim: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate and normalize memory/receiver embeddings."""
    z_memory = rng.randn(n_memories, embedding_dim).astype(np.float32)
    z_receiver = rng.randn(n_receivers, embedding_dim).astype(np.float32)
    z_memory /= np.linalg.norm(z_memory, axis=1, keepdims=True)
    z_receiver /= np.linalg.norm(z_receiver, axis=1, keepdims=True)
    return z_memory, z_receiver


def _generate_effect_matrix(
    z_memory: np.ndarray,
    z_receiver: np.ndarray,
    embedding_dim: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute τ(m,r) = sign(z_m^T W z_r)."""
    W = rng.randn(embedding_dim, embedding_dim).astype(np.float32)
    scores = z_memory @ W @ z_receiver.T
    tau_matrix = np.sign(scores).astype(np.float32)
    tau_matrix[tau_matrix == 0] = 1.0
    return W, tau_matrix


def _sample_data(
    n_samples: int,
    n_memories: int,
    n_receivers: int,
    z_memory: np.ndarray,
    z_receiver: np.ndarray,
    tau_matrix: np.ndarray,
    noise_std: float,
    rng: np.random.RandomState,
) -> dict:
    """Sample (memory, receiver) pairs with ground truth labels."""
    memory_idx = rng.randint(0, n_memories, size=n_samples)
    receiver_idx = rng.randint(0, n_receivers, size=n_samples)

    tau_true = tau_matrix[memory_idx, receiver_idx]
    noise = rng.randn(n_samples).astype(np.float32) * noise_std

    y_withhold = rng.binomial(1, 0.5, size=n_samples).astype(np.float32)
    y_expose = np.clip(y_withhold + tau_true + noise, 0.0, 1.0)

    mem_features = z_memory[memory_idx]
    recv_features = z_receiver[receiver_idx]
    combined = np.concatenate(
        [mem_features, recv_features], axis=1,
    ).astype(np.float32)

    return {
        "memory_idx": memory_idx,
        "receiver_idx": receiver_idx,
        "mem_features": mem_features,
        "recv_features": recv_features,
        "combined_features": combined,
        "tau_true": tau_true,
        "y_withhold": y_withhold,
        "y_expose": y_expose,
        "noise": noise,
    }


def main() -> None:
    config = _load_config()
    env_cfg = config["environment"]
    data_cfg = config["data"]

    seed = env_cfg["seed"]
    rng = np.random.RandomState(seed)

    n_memories = env_cfg["n_memories"]
    n_receivers = env_cfg["n_receivers"]
    embedding_dim = env_cfg["embedding_dim"]
    noise_std = env_cfg["noise_std"]
    n_train = data_cfg["n_train"]
    n_test = data_cfg["n_test"]

    print("=" * 60)
    print("Intervention Budget — Dataset Generation")
    print("=" * 60)

    # Generate embeddings.
    z_memory, z_receiver = _generate_embeddings(
        n_memories, n_receivers, embedding_dim, rng,
    )

    # Generate effect matrix.
    W, tau_matrix = _generate_effect_matrix(
        z_memory, z_receiver, embedding_dim, rng,
    )

    # Effect statistics.
    total = n_memories * n_receivers
    n_pos = int((tau_matrix > 0).sum())
    n_neg = int((tau_matrix < 0).sum())
    print(f"  Memories: {n_memories}, Receivers: {n_receivers}")
    print(f"  Effect matrix: {total} pairs "
          f"(+{n_pos}, -{n_neg})")

    # Heterogeneous memories.
    mixed = sum(
        1 for m in range(n_memories)
        if (tau_matrix[m] > 0).any() and (tau_matrix[m] < 0).any()
    )
    print(f"  Heterogeneous memories: {mixed}/{n_memories}")

    # Sample train/test.
    print(f"\n  Sampling {n_train} train + {n_test} test...")
    train_data = _sample_data(
        n_train, n_memories, n_receivers,
        z_memory, z_receiver, tau_matrix, noise_std, rng,
    )
    test_data = _sample_data(
        n_test, n_memories, n_receivers,
        z_memory, z_receiver, tau_matrix, noise_std, rng,
    )

    # Save.
    out_dir = _THIS_DIR / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "env_data.npz",
        z_memory=z_memory,
        z_receiver=z_receiver,
        W=W,
        tau_matrix=tau_matrix,
    )
    np.savez(out_dir / "train_data.npz", **train_data)
    np.savez(out_dir / "test_data.npz", **test_data)

    print(f"\n  Saved to: {out_dir}")
    print(f"    env_data.npz   (embeddings + W + tau_matrix)")
    print(f"    train_data.npz ({n_train} samples)")
    print(f"    test_data.npz  ({n_test} samples)")

    # Distributions.
    for name, data in [("Train", train_data), ("Test", test_data)]:
        tau = data["tau_true"]
        print(f"  {name} τ: +{(tau > 0).sum()}/-{(tau < 0).sum()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
