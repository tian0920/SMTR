"""TCI pairwise ranker — offline only (Tasks 13-14).

Simple linear ranker with pairwise logistic loss:
  L = log(1 + exp(-d * (s_m - s_m~)))

Implemented with numpy (torch-free) to match project constraints.
Does NOT modify transfer_critic.py or router decision rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass
class TCIRankerConfig:
    """Configuration for TCI ranker training."""

    feature_dim: int = 128
    learning_rate: float = 0.01
    n_epochs: int = 50
    seed: int = 7


@dataclass
class TCIRankerCheckpoint:
    """Saved TCI ranker state."""

    weights: np.ndarray  # shape (feature_dim,)
    bias: float = 0.0
    feature_dim: int = 128
    config: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save checkpoint to disk."""
        joblib.dump(
            {
                "weights": self.weights,
                "bias": self.bias,
                "feature_dim": self.feature_dim,
                "config": self.config,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> TCIRankerCheckpoint:
        """Load checkpoint from disk."""
        data = joblib.load(str(path))
        return cls(
            weights=data["weights"],
            bias=data["bias"],
            feature_dim=data["feature_dim"],
            config=data.get("config", {}),
        )


def score_memory(
    features: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> float:
    """Compute transfer score: s(m) = w^T x + b.

    Parameters
    ----------
    features : shape (feature_dim,) or (n, feature_dim)
    weights : shape (feature_dim,)
    bias : scalar

    Returns
    -------
    Scalar score or array of scores.
    """
    return float(np.dot(features, weights) + bias)


def tci_pairwise_loss(
    score_original: np.ndarray,
    score_perturbed: np.ndarray,
    direction: np.ndarray,
) -> float:
    """Compute pairwise logistic loss.

    L = log(1 + exp(-d * (s_m - s_m~)))

    Parameters
    ----------
    score_original : shape (n,) — s(m) for original memory
    score_perturbed : shape (n,) — s(m~) for perturbed memory
    direction : shape (n,) — +1 or -1

    Returns
    -------
    Mean loss over all pairs.
    """
    margin = direction * (score_original - score_perturbed)
    # Numerically stable log(1 + exp(-x))
    loss = np.where(
        margin > 0,
        np.log1p(np.exp(-margin)),
        -margin + np.log1p(np.exp(margin)),
    )
    return float(np.mean(loss))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class TCIRanker:
    """Offline linear ranker trained with pairwise TCI supervision.

    Model: s(m) = w^T * features(m) + b
    Loss: pairwise logistic — log(1 + exp(-d * (s_m - s_m~)))

    Does NOT modify transfer_critic.py.
    Does NOT modify router decision rules.
    """

    def __init__(self, config: TCIRankerConfig | None = None) -> None:
        self.config = config or TCIRankerConfig()
        rng = np.random.RandomState(self.config.seed)
        self.weights = rng.randn(self.config.feature_dim) * 0.01
        self.bias = 0.0

    def score(self, features: np.ndarray) -> np.ndarray:
        """Score one or more memories.

        Parameters
        ----------
        features : shape (feature_dim,) or (n, feature_dim)

        Returns
        -------
        Scalar or array of scores.
        """
        if features.ndim == 1:
            return np.dot(features, self.weights) + self.bias
        return features @ self.weights + self.bias

    def train(
        self,
        features_original: np.ndarray,
        features_perturbed: np.ndarray,
        directions: np.ndarray,
    ) -> dict[str, Any]:
        """Train ranker with pairwise gradient descent.

        Parameters
        ----------
        features_original : shape (n, feature_dim)
        features_perturbed : shape (n, feature_dim)
        directions : shape (n,) — +1 or -1

        Returns
        -------
        Training history dict with final loss.
        """
        n = len(directions)
        if n == 0:
            return {"final_loss": 0.0, "n_pairs": 0}

        lr = self.config.learning_rate
        diff = features_original - features_perturbed  # (n, d)

        history: list[float] = []
        for epoch in range(self.config.n_epochs):
            s_orig = self.score(features_original)
            s_pert = self.score(features_perturbed)
            margin = directions * (s_orig - s_pert)
            # Gradient of pairwise logistic loss
            sig = _sigmoid(-margin)  # (n,)
            # d L / d w = mean(-d * sig * diff)
            grad_w = (-directions * sig) @ diff / n
            grad_b = float(np.mean(-directions * sig))

            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

            loss = tci_pairwise_loss(s_orig, s_pert, directions)
            history.append(loss)

        final_loss = history[-1] if history else 0.0
        return {"final_loss": final_loss, "n_pairs": n, "loss_history": history}

    def train_with_validation(
        self,
        train_features_original: np.ndarray,
        train_features_perturbed: np.ndarray,
        train_directions: np.ndarray,
        valid_features_original: np.ndarray,
        valid_features_perturbed: np.ndarray,
        valid_directions: np.ndarray,
        *,
        epochs: int = 100,
    ) -> dict[str, Any]:
        """Train with early stopping on held-out validation set.

        Tracks train_loss, valid_accuracy, valid_margin per epoch.
        Restores best weights (by max valid_pairwise_accuracy) at the end.

        Parameters
        ----------
        train_features_original : shape (n_train, feature_dim)
        train_features_perturbed : shape (n_train, feature_dim)
        train_directions : shape (n_train,)
        valid_features_original : shape (n_valid, feature_dim)
        valid_features_perturbed : shape (n_valid, feature_dim)
        valid_directions : shape (n_valid,)
        epochs : number of training epochs

        Returns
        -------
        History dict with train_loss, valid_accuracy, valid_margin,
        best_epoch, best_valid_accuracy.
        """
        n_train = len(train_directions)
        n_valid = len(valid_directions)
        if n_train == 0:
            return {
                "train_loss": [],
                "valid_accuracy": [],
                "valid_margin": [],
                "best_epoch": -1,
                "best_valid_accuracy": 0.0,
            }

        lr = self.config.learning_rate
        diff_train = train_features_original - train_features_perturbed

        train_losses: list[float] = []
        valid_accuracies: list[float] = []
        valid_margins: list[float] = []

        best_valid_acc = -1.0
        best_weights = self.weights.copy()
        best_bias = self.bias
        best_epoch = -1

        for epoch in range(epochs):
            # ── Forward pass on train ──
            s_orig_train = self.score(train_features_original)
            s_pert_train = self.score(train_features_perturbed)
            margin_train = train_directions * (s_orig_train - s_pert_train)
            sig_train = _sigmoid(-margin_train)
            grad_w = (-train_directions * sig_train) @ diff_train / n_train
            grad_b = float(np.mean(-train_directions * sig_train))

            # ── Gradient step ──
            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

            # ── Train loss ──
            train_loss = tci_pairwise_loss(
                s_orig_train, s_pert_train, train_directions
            )
            train_losses.append(train_loss)

            # ── Validation metrics ──
            if n_valid > 0:
                s_orig_v = self.score(valid_features_original)
                s_pert_v = self.score(valid_features_perturbed)
                margin_v = valid_directions * (s_orig_v - s_pert_v)
                correct_v = float(np.mean((margin_v > 0).astype(float)))
                mean_margin_v = float(np.mean(margin_v))
            else:
                correct_v = 0.0
                mean_margin_v = 0.0

            valid_accuracies.append(correct_v)
            valid_margins.append(mean_margin_v)

            if correct_v > best_valid_acc:
                best_valid_acc = correct_v
                best_weights = self.weights.copy()
                best_bias = self.bias
                best_epoch = epoch

        # Restore best state.
        self.weights = best_weights
        self.bias = best_bias

        return {
            "train_loss": train_losses,
            "valid_accuracy": valid_accuracies,
            "valid_margin": valid_margins,
            "best_epoch": best_epoch,
            "best_valid_accuracy": best_valid_acc,
        }

    def to_checkpoint(self) -> TCIRankerCheckpoint:
        """Export current state as checkpoint."""
        return TCIRankerCheckpoint(
            weights=self.weights.copy(),
            bias=self.bias,
            feature_dim=self.config.feature_dim,
            config={
                "learning_rate": self.config.learning_rate,
                "n_epochs": self.config.n_epochs,
                "seed": self.config.seed,
            },
        )

    @classmethod
    def from_checkpoint(cls, ckpt: TCIRankerCheckpoint) -> TCIRanker:
        """Load ranker from checkpoint."""
        config = TCIRankerConfig(
            feature_dim=ckpt.feature_dim,
            learning_rate=ckpt.config.get("learning_rate", 0.01),
            n_epochs=ckpt.config.get("n_epochs", 50),
            seed=ckpt.config.get("seed", 7),
        )
        ranker = cls(config)
        ranker.weights = ckpt.weights.copy()
        ranker.bias = ckpt.bias
        return ranker
