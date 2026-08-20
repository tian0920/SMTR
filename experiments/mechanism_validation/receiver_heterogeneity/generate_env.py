"""Generate synthetic environment for receiver heterogeneity stress test.

Ground truth design:
  - Memory embedding: z_m ∈ R^16
  - Receiver embedding: z_r ∈ R^16
  - Interaction matrix: W ∈ R^(16×16)
  - Effect: τ(m,r) = sign(z_m^T W z_r)
  - Noise: ε ~ N(0, noise_std)
  - Outcomes: Y_withhold ~ Bernoulli(0.5)
              Y_expose = clip(Y_withhold + τ + ε, 0, 1)

The bilinear form z_m^T W z_r creates receiver-specific effects:
  - Same memory m can have τ(m,r₁)=+1 but τ(m,r₂)=-1
  - This is exactly the SMTR core hypothesis.
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
    config_path = _THIS_DIR / "config.yaml"
    with open(config_path) as f:
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
    # L2 normalize so bilinear products are bounded.
    z_memory /= np.linalg.norm(z_memory, axis=1, keepdims=True)
    z_receiver /= np.linalg.norm(z_receiver, axis=1, keepdims=True)
    return z_memory, z_receiver


def _generate_effect_matrix(
    z_memory: np.ndarray,
    z_receiver: np.ndarray,
    embedding_dim: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute τ(m,r) = sign(z_m^T W z_r) and return W, tau_matrix."""
    W = rng.randn(embedding_dim, embedding_dim).astype(np.float32)
    # Bilinear scores: shape (n_memories, n_receivers)
    scores = z_memory @ W @ z_receiver.T
    tau_matrix = np.sign(scores).astype(np.float32)
    # Handle exact zeros (very rare with continuous W).
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

    # Y_withhold ~ Bernoulli(0.5), Y_expose = clip(Y_withhold + tau + noise)
    y_withhold = rng.binomial(1, 0.5, size=n_samples).astype(np.float32)
    y_expose = np.clip(y_withhold + tau_true + noise, 0.0, 1.0)

    # Model input features
    mem_features = z_memory[memory_idx]           # (n, 16)
    recv_features = z_receiver[receiver_idx]      # (n, 16)
    combined = np.concatenate(
        [mem_features, recv_features], axis=1,
    ).astype(np.float32)                          # (n, 32)

    return {
        "memory_idx": memory_idx,
        "receiver_idx": receiver_idx,
        "mem_features": mem_features,
        "recv_features": recv_features,
        "combined_features": combined,
        "tau_true": tau_true,
        "y_withhold": y_withhold,
        "y_expose": y_expose,
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
    print("Receiver Heterogeneity Stress Test — Environment Generation")
    print("=" * 60)

    # Generate embeddings.
    z_memory, z_receiver = _generate_embeddings(
        n_memories, n_receivers, embedding_dim, rng,
    )

    # Generate effect matrix.
    W, tau_matrix = _generate_effect_matrix(
        z_memory, z_receiver, embedding_dim, rng,
    )

    # Print effect matrix statistics.
    n_pos = int((tau_matrix > 0).sum())
    n_neg = int((tau_matrix < 0).sum())
    n_zero = int((tau_matrix == 0).sum())
    total = n_memories * n_receivers
    print(f"  Memories: {n_memories}, Receivers: {n_receivers}")
    print(f"  Effect matrix: {total} pairs")
    print(f"    Positive: {n_pos} ({n_pos/total:.1%})")
    print(f"    Negative: {n_neg} ({n_neg/total:.1%})")
    print(f"    Zero:     {n_zero} ({n_zero/total:.1%})")

    # Per-memory heterogeneity: how many memories have mixed effects?
    mixed_count = 0
    for m in range(n_memories):
        row = tau_matrix[m]
        if (row > 0).any() and (row < 0).any():
            mixed_count += 1
    print(f"  Memories with heterogeneous effects: "
          f"{mixed_count}/{n_memories}")

    # Sample train and test data.
    print(f"\n  Sampling {n_train} train + {n_test} test examples...")
    train_data = _sample_data(
        n_train, n_memories, n_receivers,
        z_memory, z_receiver, tau_matrix, noise_std, rng,
    )
    test_data = _sample_data(
        n_test, n_memories, n_receivers,
        z_memory, z_receiver, tau_matrix, noise_std, rng,
    )

    # Save artifacts.
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
    print(f"    env_data.npz  (embeddings + W + tau_matrix)")
    print(f"    train_data.npz ({n_train} samples)")
    print(f"    test_data.npz  ({n_test} samples)")

    # Print train/test label distributions.
    train_pos = (train_data["tau_true"] > 0).sum()
    test_pos = (test_data["tau_true"] > 0).sum()
    print(f"\n  Train τ distribution: +{train_pos}/-{n_train - train_pos}")
    print(f"  Test  τ distribution: +{test_pos}/-{n_test - test_pos}")
    print("\nDone.")


if __name__ == "__main__":
    main()
