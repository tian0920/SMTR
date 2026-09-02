"""Four-outcome transfer critic for cross-agent memory exposure.

Supports two modes:
  - ``flat``: original four-class multinomial logistic regression.
  - ``opportunity_factorized``: three binary heads (baseline, rescue, damage)
    whose predictions are combined into the four-outcome distribution.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from smtr.core.types import (
    CandidateExposureInput,
    TransferPrediction,
    TransferPredictionDistribution,
)
from smtr.counterfactual.edge_keys import (
    TreatmentEdgeKey,
    treatment_edge_key,
)
from smtr.router.transfer_calibration import (
    DEFAULT_EPSILONS,
    Q01Calibrator,
    build_edge_calibration_examples,
    build_edge_threshold_examples,
    select_epsilon_edge_level,
)
from smtr.router.transfer_coverage import (
    count_outcome_edges,
    validate_transfer_label_coverage,
)
from smtr.router.transfer_features import HashingTransferFeatureEncoder

CLASS_ORDER = ["q00", "q01", "q10", "q11"]
LABEL_TO_INDEX = {
    "neutral_failure": 0,
    "negative_transfer": 1,
    "positive_transfer": 2,
    "neutral_success": 3,
}

_VALID_CRITIC_MODES = frozenset({"flat", "opportunity_factorized"})

# Task 1: Explicit critic training modes for the final integration.
# ``observational``: L = L_obs (original flat/factorized, backward compatible).
# ``tci_augmented``: L = L_obs + L_TCI (alpha=1 fixed, TCI examples appended
# with obs_weight=1 per observational example, tci_weight=1 per TCI example).
# Factorized critic + TCI is intentionally excluded: ablation showed
# factorized intervention ranking = 0.1842 (worse than random).
_VALID_CRITIC_TRAINING_MODES: frozenset[str] = frozenset({
    "observational",
    "tci_augmented",
    "tci_full",
})
TCI_SCHEMA_VERSION: str = "v1"

# Effect classes for absolute transfer value supervision.
EFFECT_CLASSES: tuple[int, ...] = (-1, 0, 1)


@dataclass
class FactorizedCriticMember:
    """One bootstrap member of the opportunity-factorized critic."""

    baseline_model: Any
    rescue_model: Any
    damage_model: Any


@dataclass(frozen=True)
class FactorizedDiagnostics:
    """Per-head predicted probabilities for one candidate exposure."""

    baseline_success: float
    rescue_given_failure: float
    damage_given_success: float


@dataclass
class TCIValueHead:
    """Absolute transfer effect predictor (Task 3).

    Predicts τ(m) ∈ {-1, 0, +1} from memory features φ(m).

    Uses sklearn LogisticRegression with class_weight="balanced" for
    the three-class problem. No neural networks, transformers, or
    attention (forbidden).

    Attributes
    ----------
    model : LogisticRegression
        Fitted classifier with classes=[-1, 0, 1].
    n_examples : int
        Number of training examples used.
    """

    model: Any
    n_examples: int = 0

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict effect class for each row in features.

        Parameters
        ----------
        features : (n, n_features) array

        Returns
        -------
        (n,) array of predicted effects in {-1, 0, 1}.
        """
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability distribution over effect classes.

        Returns
        -------
        (n, 3) array of probabilities for classes [-1, 0, 1].
        """
        return self.model.predict_proba(features)

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """Raw decision function values.

        Returns
        -------
        (n, 3) array of raw logits (one per class).
        """
        return self.model.decision_function(features)


class FourOutcomeTransferCritic:
    """TCI-SMTR: Transfer-Critical Intervention-guided critic.

    .. deprecated:: RIMA canonical refactor (2026-08-31)
        Demoted to **controlled ablation only** (experiment name
        ``RIMA-Binary``). The binary four-outcome critic with team_success
        outcome is NOT the formal MultiAgentBench method. The formal
        method is :class:`smtr.router.official_score_transfer_critic.
        OfficialScoreTransferCritic` (continuous potential-outcome critic
        on official Task Score). See
        ``docs/experiment_lineage/rima_canonical_migration.md``.

    Ensemble of logistic regression critics predicting four transfer
    outcomes. The critic output ``s_θ(m) = q10(m) - q01(m)`` is the
    **transfer utility score** estimating E[Y_m - Y_0].

    Training objective (unified, no separate heads):
      L = L_obs + L_rank + L_τ

    where:
      - L_obs: observational four-outcome classification (P(Y))
      - L_rank: pairwise ranking from TCI contrasts (τ(m) > τ(m̃))
      - L_τ: absolute transfer effect prediction (P(τ))

    All three losses share the same four-class output. Fixed weights:
    obs_weight=1, rank_weight=1, value_weight=1 (no lambda search).

    Outputs: q00=P(neutral_failure), q01=P(negative_transfer),
             q10=P(positive_transfer), q11=P(neutral_success)
    """

    def __init__(
        self,
        *,
        n_features: int = 512,
        n_bootstrap: int = 31,
        feature_block: str = "full",
        seed: int = 7,
        critic_mode: str = "flat",
    ) -> None:
        if critic_mode not in _VALID_CRITIC_MODES:
            raise ValueError(
                f"unknown critic_mode: {critic_mode!r}; "
                f"valid: {sorted(_VALID_CRITIC_MODES)}"
            )
        self.critic_mode = critic_mode
        self.n_features = n_features
        self.n_bootstrap = n_bootstrap
        self.feature_block = feature_block
        self.seed = seed
        self.encoder = HashingTransferFeatureEncoder(
            n_features=n_features, feature_block=feature_block
        )
        self.members: list[LogisticRegression] = []
        self.factorized_members: list[FactorizedCriticMember] = []
        self.head_support_report: dict[str, Any] | None = None
        self._fitted = False
        self.coverage_report: dict[str, Any] | None = None
        self.q01_calibrator: Q01Calibrator | None = None
        self.epsilon_star: float | None = None
        self.risk_calibration: dict[str, Any] | None = None
        self.calibration_split: str | None = None
        self.epsilon_selection_split: str | None = None
        self.validation_edge_count: int | None = None
        self.calibration_unit: str | None = None
        self.epsilon_selection_unit: str | None = None
        # 清单 P0-2 (3.6): training provenance bound into the checkpoint so
        # the split audit can verify which data files produced this critic.
        self.training_split: str | None = None
        self.train_record_digest: str | None = None
        self.validation_record_digest: str | None = None
        self.memory_pool_digest: str | None = None
        # 清单 Shared-Control 第16.1节: budget / shared-control provenance
        # bound into every checkpoint.
        self.training_budget_policy: str | None = None
        self.training_budget_requested: float | None = None
        self.training_budget_realized: float | None = None
        self.parent_train_candidate_manifest_digest: str | None = None
        self.budget_train_candidate_manifest_digest: str | None = None
        self.shared_control_definition_version: str | None = None
        self.loss_weighting_unit: str | None = None
        # 清单 Writer-Agnostic 第十章: method-schema metadata bound into
        # every formal checkpoint (writer-agnostic conditioning contract).
        self.method_schema_metadata: dict[str, Any] | None = None
        self.bootstrap_cluster_unit: str | None = None
        self.adaptive_sampling_used: bool = False
        self.adaptive_stopping_used: bool = False
        # 清单 Fixed-Budget 第9/10章: effective training-subset digest and
        # structured budget metadata blocks for every checkpoint.
        self.effective_train_record_digest: str | None = None
        self.effective_train_edge_count: int | None = None
        self.budget_policy_metadata: dict[str, Any] | None = None
        self.training_support_metadata: dict[str, Any] | None = None
        self.training_artifact_digests: dict[str, Any] | None = None
        # 清单 Formal Protocol §2: seed protocol metadata bound into every
        # checkpoint so downstream stages verify the same protocol.
        self.seed_protocol_metadata: dict[str, Any] | None = None
        self._edge_calibration_examples: list | None = None
        # TCI distillation provenance: number of TCI examples appended
        # during fit(); zero for observational-only critics (and for
        # old checkpoints loaded before distillation was added).
        self.tci_distillation_n_examples: int = 0
        self.tci_distillation_alpha: float | None = None
        self.tci_distillation_metrics: dict[str, Any] | None = None
        # Task 1: formal training mode label (checkpoint provenance).
        # Defaults to "observational" for old checkpoints and for
        # fit() calls with tci_inputs=None.
        self.training_mode: str = "observational"
        self.n_observational_examples: int = 0
        self.n_tci_examples: int = 0
        self.tci_schema_version: str | None = None
        # Task 3/6: TCI value head for absolute transfer effect prediction.
        # None for observational and tci_augmented modes.
        self.tci_value_head: TCIValueHead | None = None
        self.tci_rank_examples: int = 0
        self.tci_value_examples: int = 0

    def fit(
        self,
        inputs: list[CandidateExposureInput],
        labels: list[str],
        *,
        records: list[dict[str, Any]] | None = None,
        coverage_mode: str = "pilot",
        sample_weights: np.ndarray | None = None,
        bootstrap_clusters: dict | None = None,
        tci_inputs: list[tuple[
            CandidateExposureInput,
            CandidateExposureInput,
            int,
            str,
        ]] | None = None,
        tci_alpha: float = 1.0,
        tci_effect_batch: Any | None = None,
    ) -> None:
        """Train bootstrap ensemble on paired record features.

        Dispatches to ``_fit_flat`` or ``_fit_factorized`` based on
        ``self.critic_mode``. Flat mode ignores ``records``.

        Optional TCI distillation supervision (observational+tci mode):
        when ``tci_inputs`` is provided, the TCI pairs are converted to
        soft-labeled binary classification examples and appended to the
        observational training data with weight ``tci_alpha / n_tci``.
        When ``tci_inputs`` is None, behaviour is identical to the
        original ``observational`` training. Default: observational.

        Optional TCI value supervision (tci_value_augmented mode):
        when ``tci_effect_batch`` is provided (alongside ``tci_inputs``),
        a separate value head is trained to predict absolute transfer
        effect τ(m) ∈ {-1, 0, +1}. Training mode becomes
        ``tci_value_augmented``. Default: None.
        """
        if self.critic_mode == "flat":
            return self._fit_flat(
                inputs, labels,
                coverage_mode=coverage_mode,
                sample_weights=sample_weights,
                bootstrap_clusters=bootstrap_clusters,
                tci_inputs=tci_inputs,
                tci_alpha=tci_alpha,
                tci_effect_batch=tci_effect_batch,
            )
        if self.critic_mode == "opportunity_factorized":
            if records is None:
                raise ValueError(
                    "opportunity_factorized mode requires records="
                )
            return self._fit_factorized(
                inputs, records,
                coverage_mode=coverage_mode,
                sample_weights=sample_weights,
                bootstrap_clusters=bootstrap_clusters,
            )
        raise ValueError(f"unknown critic_mode: {self.critic_mode!r}")

    def _fit_flat(
        self,
        inputs: list[CandidateExposureInput],
        labels: list[str],
        *,
        coverage_mode: str = "pilot",
        sample_weights: np.ndarray | None = None,
        bootstrap_clusters: dict | None = None,
        tci_inputs: list[tuple[
            CandidateExposureInput,
            CandidateExposureInput,
            int,
            str,
        ]] | None = None,
        tci_alpha: float = 1.0,
        tci_effect_batch: Any | None = None,
    ) -> None:
        """Flat four-outcome training (TCI-SMTR unified critic).

        Training modes:
          - ``observational`` (default): L = L_obs.
          - ``tci_augmented``: L = L_obs + L_rank (pairwise ranking).
          - ``tci_full``: L = L_obs + L_rank + L_τ (ranking + effect).

        TCI effect supervision is unified into the same critic: effect
        labels {-1, 0, +1} are mapped to the four-outcome space and
        appended as additional training examples. No separate value head.
        """
        # Build observational feature matrix.
        X_obs = self.encoder.encode_batch(inputs)
        y_obs = np.array([LABEL_TO_INDEX[lb] for lb in labels])
        n_obs = len(labels)

        # TCI rank augmentation: pairwise ranking examples.
        X_tci_rank = None
        y_tci_rank = None
        if tci_inputs:
            from smtr.router.tci_augmentation import (
                build_tci_augmentation_examples,
            )
            batch = build_tci_augmentation_examples(tci_inputs)
            if batch.inputs:
                X_tci_rank = self.encoder.encode_batch(batch.inputs)
                y_tci_rank = np.array(
                    [LABEL_TO_INDEX[lb] for lb in batch.labels]
                )
        n_rank = X_tci_rank.shape[0] if X_tci_rank is not None else 0

        # TCI effect augmentation: absolute effect → 4-class labels.
        # effect=+1 → positive_transfer, effect=-1 → negative_transfer,
        # effect= 0 → neutral_success.
        X_tci_effect = None
        y_tci_effect = None
        n_effect = 0
        has_effect = (
            tci_effect_batch is not None
            and hasattr(tci_effect_batch, 'n_examples')
            and tci_effect_batch.n_examples > 0
            and tci_inputs is not None
        )
        if has_effect:
            from smtr.router.transfer_target import (
                build_effect_targets,
            )
            effect_targets = build_effect_targets(
                tci_effect_batch, tci_inputs=tci_inputs,
            )
            if effect_targets:
                effect_inputs = [t.input for t in effect_targets]
                effect_labels = [t.unified_label for t in effect_targets]
                X_tci_effect = self.encoder.encode_batch(effect_inputs)
                y_tci_effect = np.array(
                    [LABEL_TO_INDEX[lb] for lb in effect_labels]
                )
                n_effect = X_tci_effect.shape[0]

        # Combine rank + effect into unified TCI block.
        from scipy import sparse as sp_sparse
        X_tci = None
        y_tci = None
        tci_n = 0
        if X_tci_rank is not None and X_tci_effect is not None:
            if sp_sparse.issparse(X_tci_rank) or sp_sparse.issparse(
                X_tci_effect
            ):
                X_tci = sp_sparse.vstack(
                    [X_tci_rank, X_tci_effect]
                ).tocsr()
            else:
                X_tci = np.vstack([X_tci_rank, X_tci_effect])
            y_tci = np.concatenate([y_tci_rank, y_tci_effect])
            tci_n = X_tci.shape[0]
        elif X_tci_rank is not None:
            X_tci = X_tci_rank
            y_tci = y_tci_rank
            tci_n = n_rank
        elif X_tci_effect is not None:
            X_tci = X_tci_effect
            y_tci = y_tci_effect
            tci_n = n_effect

        # Set training_mode.
        # Priority: tci_full > tci_augmented > observational.
        if n_effect > 0 and n_rank > 0:
            self.training_mode = "tci_full"
            self.tci_schema_version = TCI_SCHEMA_VERSION
        elif n_rank > 0:
            self.training_mode = "tci_augmented"
            self.tci_schema_version = TCI_SCHEMA_VERSION
        else:
            self.training_mode = "observational"
            self.tci_schema_version = None
        self.n_observational_examples = n_obs
        self.n_tci_examples = tci_n
        self.tci_rank_examples = n_rank
        self.tci_value_examples = n_effect

        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype=float)
            if len(sample_weights) != len(labels):
                raise ValueError(
                    "sample_weights must have one entry per training record"
                )
        if bootstrap_clusters is not None and tci_n > 0:
            bootstrap_clusters = None

        unique_classes = np.unique(y_obs)
        if len(unique_classes) < 2:
            raise ValueError(
                "training data must contain at least two transfer outcome classes"
            )

        report = validate_transfer_label_coverage(labels, mode=coverage_mode)
        report.update(count_outcome_edges(inputs, labels))
        self.coverage_report = report

        required_classes = set(unique_classes.tolist())

        # Validate that bootstrap clusters partition every usable row.
        if bootstrap_clusters is not None:
            n_rows = len(inputs)
            flat = [
                idx
                for cluster in bootstrap_clusters.values()
                for idx in cluster
            ]
            if any(idx < 0 or idx >= n_rows for idx in flat):
                raise ValueError(
                    "bootstrap clusters contain row indices out of range"
                )
            if len(set(flat)) != len(flat):
                raise ValueError(
                    "bootstrap clusters overlap (duplicate row indices)"
                )
            if set(flat) != set(range(n_rows)):
                raise ValueError(
                    "bootstrap clusters must partition every training "
                    "record row"
                )

        rng = np.random.default_rng(self.seed)
        self.members = []
        for _ in range(self.n_bootstrap):
            if bootstrap_clusters is not None:
                idx = _cluster_bootstrap_with_full_coverage(
                    y_obs, bootstrap_clusters, required_classes, rng
                )
            else:
                idx = _bootstrap_with_full_coverage(
                    y_obs, required_classes, rng
                )
            if idx is None:
                continue
            X_boot = X_obs[idx]
            y_boot = y_obs[idx]
            w_boot = None if sample_weights is None else sample_weights[idx]

            # Append unified TCI examples (rank + effect).
            # Each TCI example gets weight 1/n_tci (total TCI weight=1).
            if X_tci is not None:
                if sp_sparse.issparse(X_boot) or sp_sparse.issparse(X_tci):
                    X_boot = sp_sparse.vstack([X_boot, X_tci]).tocsr()
                else:
                    X_boot = np.vstack([X_boot, X_tci])
                y_boot = np.concatenate([y_boot, y_tci])
                w_tci = np.full(tci_n, 1.0 / tci_n)
                if w_boot is not None:
                    w_boot = np.concatenate([w_boot, w_tci])
                else:
                    w_obs = np.ones(len(idx), dtype=float)
                    w_boot = np.concatenate([w_obs, w_tci])

            clf = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced" if w_boot is None else None,
            )
            clf.fit(X_boot, y_boot, sample_weight=w_boot)
            self.members.append(clf)
        if not self.members:
            raise ValueError("no bootstrap member covered all required classes")
        self._fitted = True
        self.tci_distillation_n_examples = tci_n

    def _fit_factorized(
        self,
        inputs: list[CandidateExposureInput],
        records: list[dict[str, Any]],
        *,
        coverage_mode: str = "pilot",
        sample_weights: np.ndarray | None = None,
        bootstrap_clusters: dict | None = None,
    ) -> None:
        """Opportunity-factorized training: three binary heads."""
        from smtr.router.opportunity_training import (
            apply_family_multiplicities,
            bootstrap_family_multiplicities,
            build_opportunity_training_data,
        )

        opp = build_opportunity_training_data(inputs, records)
        self.head_support_report = opp.support_report

        # ---- Formal mode: fail-fast if any head lacks both classes ----
        for head_name, ds in [
            ("baseline", opp.baseline),
            ("rescue", opp.rescue),
            ("damage", opp.damage),
        ]:
            if len(ds.inputs) == 0:
                if head_name in ("rescue", "damage"):
                    # Allowed: no opportunity for this head.
                    continue
                raise ValueError(
                    f"formal mode: {head_name} head has no training data"
                )
            unique = set(ds.targets.tolist())
            if head_name == "baseline" and len(unique) < 2:
                raise ValueError(
                    f"formal mode: {head_name} head lacks class diversity: "
                    f"{unique}"
                )

        # ---- Bootstrap: shared family multiplicities ----
        # Use baseline family_ids as the canonical family set.
        all_family_ids = opp.baseline.family_ids
        rng = np.random.default_rng(self.seed)
        self.factorized_members = []
        max_attempts = self.n_bootstrap * 20

        for attempt in range(max_attempts):
            if len(self.factorized_members) >= self.n_bootstrap:
                break
            mult = bootstrap_family_multiplicities(all_family_ids, rng)

            # Apply to baseline.
            b_idx, b_w = apply_family_multiplicities(opp.baseline, mult)
            if len(b_idx) == 0:
                continue
            b_targets = opp.baseline.targets[b_idx]
            if len(set(b_targets.tolist())) < 2:
                continue  # skip: baseline lacks both classes

            # Apply to rescue.
            r_idx, r_w = apply_family_multiplicities(opp.rescue, mult)
            # Apply to damage.
            d_idx, d_w = apply_family_multiplicities(opp.damage, mult)

            # ---- Fit baseline head ----
            X_b = self.encoder.encode_baseline_batch(
                [opp.baseline.inputs[i].receiver_state for i in b_idx]
            )
            baseline_clf = LogisticRegression(
                max_iter=1000, solver="lbfgs", class_weight=None,
            )
            baseline_clf.fit(X_b, b_targets, sample_weight=b_w)

            # ---- Fit rescue head (may be empty) ----
            rescue_clf = None
            if len(r_idx) > 0:
                r_targets = opp.rescue.targets[r_idx]
                if len(set(r_targets.tolist())) >= 2:
                    X_r = self.encoder.encode_batch(
                        [opp.rescue.inputs[i] for i in r_idx]
                    )
                    rescue_clf = LogisticRegression(
                        max_iter=1000, solver="lbfgs", class_weight=None,
                    )
                    rescue_clf.fit(X_r, r_targets, sample_weight=r_w)

            # ---- Fit damage head (may be empty) ----
            damage_clf = None
            if len(d_idx) > 0:
                d_targets = opp.damage.targets[d_idx]
                if len(set(d_targets.tolist())) >= 2:
                    X_d = self.encoder.encode_batch(
                        [opp.damage.inputs[i] for i in d_idx]
                    )
                    damage_clf = LogisticRegression(
                        max_iter=1000, solver="lbfgs", class_weight=None,
                    )
                    damage_clf.fit(X_d, d_targets, sample_weight=d_w)

            # Skip member if rescue or damage had opportunity but
            # lacked class diversity.
            if len(r_idx) > 0 and rescue_clf is None:
                continue
            if len(d_idx) > 0 and damage_clf is None:
                continue

            self.factorized_members.append(FactorizedCriticMember(
                baseline_model=baseline_clf,
                rescue_model=rescue_clf,
                damage_model=damage_clf,
            ))

        if len(self.factorized_members) < self.n_bootstrap:
            raise ValueError(
                f"could not fit {self.n_bootstrap} factorized members; "
                f"got {len(self.factorized_members)} after {max_attempts} "
                f"attempts. Check head support."
            )

        # Also run flat coverage report for checkpoint compatibility.
        labels = [
            f"q{int(r['share']['team_success'])}{int(r['withhold']['team_success'])}"
            .replace("q00", "neutral_failure")
            .replace("q01", "negative_transfer")
            .replace("q10", "positive_transfer")
            .replace("q11", "neutral_success")
            for r in records
        ]
        from smtr.marble.paired_outcomes import paired_record_label
        labels = [paired_record_label(r) for r in records]
        report = validate_transfer_label_coverage(labels, mode=coverage_mode)
        report.update(count_outcome_edges(inputs, labels))
        self.coverage_report = report
        self._fitted = True

    def predict(self, item: CandidateExposureInput) -> TransferPrediction:
        """Predict four-outcome distribution for a candidate exposure."""
        probs = self._member_probs(item).mean(axis=0)
        return TransferPrediction(
            q00_neutral_failure=float(probs[0]),
            q01_negative_transfer=float(probs[1]),
            q10_positive_transfer=float(probs[2]),
            q11_neutral_success=float(probs[3]),
        )

    def _member_probs(self, item: CandidateExposureInput) -> np.ndarray:
        """Per-bootstrap-member four-outcome probabilities, shape (M, 4)."""
        if not self._fitted:
            raise RuntimeError("critic not fitted")
        if self.critic_mode == "opportunity_factorized":
            return self._factorized_member_probs(item)
        X = self.encoder.encode_one(item)
        member_probs = []
        for clf in self.members:
            p = clf.predict_proba(X)[0]
            # Align to 4 classes
            full_p = np.zeros(4)
            for i, c in enumerate(clf.classes_):
                full_p[int(c)] = p[i]
            member_probs.append(full_p)
        return np.asarray(member_probs)

    def _factorized_member_probs(
        self, item: CandidateExposureInput
    ) -> np.ndarray:
        """Per-member four-outcome probs from factorized heads, shape (M, 4)."""
        X_b = self.encoder.encode_baseline_one(item.receiver_state)
        X_full = self.encoder.encode_one(item)
        member_probs = []
        for member in self.factorized_members:
            b = _binary_prob(member.baseline_model, X_b)
            g = (
                _binary_prob(member.rescue_model, X_full)
                if member.rescue_model is not None
                else 0.0
            )
            h = (
                _binary_prob(member.damage_model, X_full)
                if member.damage_model is not None
                else 0.0
            )
            q00 = (1.0 - b) * (1.0 - g)
            q01 = b * h
            q10 = (1.0 - b) * g
            q11 = b * (1.0 - h)
            # Numerical invariant: no softmax, strict conservation.
            assert q00 >= 0 and q01 >= 0 and q10 >= 0 and q11 >= 0
            assert abs(q00 + q01 + q10 + q11 - 1.0) < 1e-8
            member_probs.append(np.array([q00, q01, q10, q11]))
        return np.asarray(member_probs)

    def predict_factorized_diagnostics(
        self, item: CandidateExposureInput
    ) -> FactorizedDiagnostics:
        """Per-head mean predictions for debugging (factorized mode only)."""
        if self.critic_mode != "opportunity_factorized":
            raise RuntimeError(
                "predict_factorized_diagnostics requires opportunity_factorized mode"
            )
        X_b = self.encoder.encode_baseline_one(item.receiver_state)
        X_full = self.encoder.encode_one(item)
        b_vals, g_vals, h_vals = [], [], []
        for member in self.factorized_members:
            b_vals.append(_binary_prob(member.baseline_model, X_b))
            g_vals.append(
                _binary_prob(member.rescue_model, X_full)
                if member.rescue_model is not None
                else 0.0
            )
            h_vals.append(
                _binary_prob(member.damage_model, X_full)
                if member.damage_model is not None
                else 0.0
            )
        return FactorizedDiagnostics(
            baseline_success=float(np.mean(b_vals)),
            rescue_given_failure=float(np.mean(g_vals)),
            damage_given_success=float(np.mean(h_vals)),
        )

    def predict_distribution(
        self, item: CandidateExposureInput
    ) -> TransferPredictionDistribution:
        """Ensemble-mean prediction plus bootstrap member uncertainty (清单第九章).

        tau_lower is the 0.10 quantile of member taus and eta_upper the
        0.90 quantile of member etas; member etas use the validation-fitted
        q01 calibrator when available so the risk bound matches the SMTR
        decision rule.
        """
        member_probs = self._member_probs(item)
        probs = member_probs.mean(axis=0)
        mean = TransferPrediction(
            q00_neutral_failure=float(probs[0]),
            q01_negative_transfer=float(probs[1]),
            q10_positive_transfer=float(probs[2]),
            q11_neutral_success=float(probs[3]),
        )
        member_tau = member_probs[:, 2] - member_probs[:, 1]
        member_eta = member_probs[:, 1]
        if self.q01_calibrator is not None:
            member_eta = self.q01_calibrator.predict(member_eta)
        return TransferPredictionDistribution(
            mean=mean,
            tau_std=float(member_tau.std()),
            eta_std=float(member_eta.std()),
            tau_lower=float(np.quantile(member_tau, 0.10)),
            eta_upper=float(np.quantile(member_eta, 0.90)),
        )

    def predict_batch(self, items: list[CandidateExposureInput]) -> list[TransferPrediction]:
        """Predict for a batch."""
        return [self.predict(item) for item in items]

    def predict_calibrated(self, item: CandidateExposureInput) -> TransferPrediction:
        """Prediction with calibrated negative-transfer risk (清单第三章).

        The SMTR decision rule compares the *calibrated* eta against the
        validation-selected epsilon_star; raw eta must never drive sharing.
        """
        raw = self.predict(item)
        return raw.model_copy(
            update={"eta_hat_calibrated": self.calibrate_negative_risk(raw.eta_hat_raw)}
        )

    def calibrate_negative_risk(self, eta_raw: float) -> float:
        """Calibrated q01 for one raw risk estimate.

        Falls back to the raw probability only when no validation-fitted
        calibrator exists (e.g. unit-test stubs); production checkpoints
        always carry one after ``calibrate_q01``.
        """
        if self.q01_calibrator is None:
            return float(eta_raw)
        return float(self.q01_calibrator.predict(np.array([float(eta_raw)]))[0])

    def calibrate_q01(
        self,
        inputs: list[CandidateExposureInput],
        labels: list[str],
        records: list[dict[str, Any]] | None = None,
        *,
        split_name: str = "validation",
        epsilons=DEFAULT_EPSILONS,
        delta: float = 0.10,
    ) -> dict[str, Any]:
        """Fit q01 calibration and select epsilon on validation treatment edges."""

        if not self._fitted:
            raise RuntimeError("critic not fitted")

        if split_name != "validation":
            raise ValueError(
                "q01 calibration and epsilon selection must use "
                f"the validation split, got {split_name!r}"
            )

        if records is None:
            raise ValueError(
                "edge-level calibration requires paired records; "
                "record-level calibration is not part of formal SMTR"
            )

        if len(inputs) != len(labels) or len(inputs) != len(records):
            raise ValueError(
                "inputs, labels and records must have identical lengths"
            )

        if not inputs:
            raise ValueError("validation data is empty")

        predictions = self.predict_batch(inputs)

        edge_predictions: dict[
            TreatmentEdgeKey,
            list[dict[str, float]],
        ] = defaultdict(list)

        for record, prediction in zip(records, predictions):
            edge_key = treatment_edge_key(record)

            edge_predictions[edge_key].append({
                "predicted_q01":
                    float(prediction.q01_negative_transfer),
                "predicted_tau":
                    float(
                        prediction.q10_positive_transfer
                        - prediction.q01_negative_transfer
                    ),
            })

        predictions_by_edge: dict[
            TreatmentEdgeKey,
            dict[str, float],
        ] = {}

        for edge_key, rows in edge_predictions.items():
            predictions_by_edge[edge_key] = {
                "predicted_q01": float(
                    np.mean([
                        row["predicted_q01"]
                        for row in rows
                    ])
                ),
                "predicted_tau": float(
                    np.mean([
                        row["predicted_tau"]
                        for row in rows
                    ])
                ),
            }

        calibration_examples = build_edge_calibration_examples(
            records=records,
            predictions_by_edge=predictions_by_edge,
        )

        if not calibration_examples:
            raise ValueError(
                "no validation treatment edges available for calibration"
            )

        # One edge is one calibration unit.
        calibration_weights = np.ones(
            len(calibration_examples),
            dtype=float,
        )

        self.q01_calibrator = Q01Calibrator().fit(
            np.asarray([
                example.predicted_q01
                for example in calibration_examples
            ], dtype=float),
            np.asarray([
                example.empirical_eta
                for example in calibration_examples
            ], dtype=float),
            sample_weight=calibration_weights,
        )

        threshold_examples = build_edge_threshold_examples(
            calibration_examples,
            self.q01_calibrator,
        )

        selection = select_epsilon_edge_level(
            examples=threshold_examples,
            candidate_epsilons=[
                float(value)
                for value in epsilons
            ],
            max_negative_exposure_rate=float(delta),
        )

        selection.update({
            "calibration_unit": "treatment_edge",
            "calibration_split": "validation",
            "calibration_method":
                self.q01_calibrator.method,
            "calibration_status":
                self.q01_calibrator.calibration_status,
            "calibration_edge_count":
                len(calibration_examples),
            "epsilon_selection_unit":
                "treatment_edge",
            "epsilon_selection_split":
                "validation",
            "validation_edge_count":
                len(calibration_examples),
            "risk_delta":
                float(delta),
        })

        self.epsilon_star = float(
            selection["epsilon_star"]
        )
        self.risk_calibration = selection
        self.calibration_split = "validation"
        self.epsilon_selection_split = "validation"
        self.validation_edge_count = len(
            calibration_examples
        )
        self.calibration_unit = "treatment_edge"
        self.epsilon_selection_unit = "treatment_edge"

        return selection

    def calibrated_q01(self, pred: TransferPrediction) -> float:
        """Calibrated negative-transfer probability for one prediction."""
        return self.calibrate_negative_risk(pred.q01_negative_transfer)

    def save(self, path: Path) -> None:
        """Save critic checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        import sklearn
        joblib.dump(
            {
                "members": self.members,
                "n_features": self.n_features,
                "n_bootstrap": self.n_bootstrap,
                "feature_block": self.feature_block,
                "seed": self.seed,
                "encoder": self.encoder,
                "schema_version": self.encoder.schema_version,
                "sklearn_version": sklearn.__version__,
                "method_version": "1.0",
                "coverage_report": self.coverage_report,
                "q01_calibrator": self.q01_calibrator,
                "epsilon_star": self.epsilon_star,
                "risk_calibration": self.risk_calibration,
                # 清单 P0-8: checkpoint must record where calibration and
                # epsilon selection happened.
                "calibration_split": self.calibration_split,
                "epsilon_selection_split": self.epsilon_selection_split,
                "validation_edge_count": self.validation_edge_count,
                "calibration_unit": self.calibration_unit,
                "calibration_method": (
                    self.q01_calibrator.method
                    if self.q01_calibrator is not None
                    else "unfitted"
                ),
                "calibration_status": (
                    self.q01_calibrator.calibration_status
                    if self.q01_calibrator is not None
                    else "unfitted"
                ),
                "calibration_edge_count": self.validation_edge_count,
                "epsilon_selection_unit": self.epsilon_selection_unit,
                "epsilon_validation_edge_count": self.validation_edge_count,
                # 清单 P0-2 (3.6): training provenance digests.
                "training_split": self.training_split,
                "train_record_digest": self.train_record_digest,
                "validation_record_digest": self.validation_record_digest,
                "memory_pool_digest": self.memory_pool_digest,
                # 清单 Shared-Control 第16.1节: budget / shared-control
                # provenance for every checkpoint.
                "training_budget_policy": self.training_budget_policy,
                "training_budget_requested": self.training_budget_requested,
                "training_budget_realized": self.training_budget_realized,
                "parent_train_candidate_manifest_digest": (
                    self.parent_train_candidate_manifest_digest
                ),
                "budget_train_candidate_manifest_digest": (
                    self.budget_train_candidate_manifest_digest
                ),
                "shared_control_definition_version": (
                    self.shared_control_definition_version
                ),
                "loss_weighting_unit": self.loss_weighting_unit,
                "bootstrap_cluster_unit": self.bootstrap_cluster_unit,
                "adaptive_sampling_used": self.adaptive_sampling_used,
                "adaptive_stopping_used": self.adaptive_stopping_used,
                # 清单 Fixed-Budget 第9/10章: effective-subset digest and
                # structured budget provenance blocks.
                "effective_train_record_digest": (
                    self.effective_train_record_digest
                ),
                "effective_train_edge_count": (
                    self.effective_train_edge_count
                ),
                "budget_policy": self.budget_policy_metadata,
                "training_support": self.training_support_metadata,
                "artifact_digests": self.training_artifact_digests,
                # 清单 Writer-Agnostic 第十章: method-schema metadata.
                "method_schema_metadata": self.method_schema_metadata,
                # 清单 Formal Protocol §2: seed protocol metadata.
                "seed_protocol_metadata": self.seed_protocol_metadata,
                # Counterfactual Opportunity v1: factorized state.
                "critic_mode": self.critic_mode,
                "factorization_version": (
                    "counterfactual_opportunity_v1"
                    if self.critic_mode == "opportunity_factorized"
                    else None
                ),
                "factorized_members": self.factorized_members,
                "head_support_report": self.head_support_report,
                # TCI Distillation provenance (Task 6).
                "tci_distillation_n_examples": (
                    self.tci_distillation_n_examples
                ),
                "tci_distillation_alpha": self.tci_distillation_alpha,
                "tci_distillation_metrics": self.tci_distillation_metrics,
                # Task 1/4: formal training mode + example counts.
                "training_mode": self.training_mode,
                "n_observational_examples": self.n_observational_examples,
                "n_tci_examples": self.n_tci_examples,
                "tci_schema_version": self.tci_schema_version,
                # Task 3/6: TCI value head checkpoint.
                "tci_value_head": self.tci_value_head,
                "tci_rank_examples": self.tci_rank_examples,
                "tci_value_examples": self.tci_value_examples,
                "effect_classes": list(EFFECT_CLASSES),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> FourOutcomeTransferCritic:
        """Load critic from checkpoint."""
        data = joblib.load(path)
        critic = cls(
            n_features=data["n_features"],
            n_bootstrap=data["n_bootstrap"],
            feature_block=data["feature_block"],
            seed=data["seed"],
            critic_mode=data.get("critic_mode", "flat"),
        )
        critic.members = data["members"]
        critic.encoder = data["encoder"]
        critic.coverage_report = data.get("coverage_report")
        critic.q01_calibrator = data.get("q01_calibrator")
        critic.epsilon_star = data.get("epsilon_star")
        critic.risk_calibration = data.get("risk_calibration")
        critic.calibration_split = data.get("calibration_split")
        critic.epsilon_selection_split = data.get("epsilon_selection_split")
        critic.validation_edge_count = data.get("validation_edge_count")
        critic.calibration_unit = data.get("calibration_unit")
        critic.epsilon_selection_unit = data.get("epsilon_selection_unit")
        critic.training_split = data.get("training_split")
        critic.train_record_digest = data.get("train_record_digest")
        critic.validation_record_digest = data.get("validation_record_digest")
        critic.memory_pool_digest = data.get("memory_pool_digest")
        critic.training_budget_policy = data.get("training_budget_policy")
        critic.training_budget_requested = data.get("training_budget_requested")
        critic.training_budget_realized = data.get("training_budget_realized")
        critic.parent_train_candidate_manifest_digest = data.get(
            "parent_train_candidate_manifest_digest"
        )
        critic.budget_train_candidate_manifest_digest = data.get(
            "budget_train_candidate_manifest_digest"
        )
        critic.shared_control_definition_version = data.get(
            "shared_control_definition_version"
        )
        critic.loss_weighting_unit = data.get("loss_weighting_unit")
        critic.bootstrap_cluster_unit = data.get("bootstrap_cluster_unit")
        critic.adaptive_sampling_used = bool(
            data.get("adaptive_sampling_used", False)
        )
        critic.adaptive_stopping_used = bool(
            data.get("adaptive_stopping_used", False)
        )
        critic.effective_train_record_digest = data.get(
            "effective_train_record_digest"
        )
        critic.effective_train_edge_count = data.get(
            "effective_train_edge_count"
        )
        critic.budget_policy_metadata = data.get("budget_policy")
        critic.training_support_metadata = data.get("training_support")
        critic.training_artifact_digests = data.get("artifact_digests")
        critic.method_schema_metadata = data.get("method_schema_metadata")
        critic.seed_protocol_metadata = data.get("seed_protocol_metadata")
        critic.factorized_members = data.get("factorized_members", [])
        critic.head_support_report = data.get("head_support_report")
        critic.tci_distillation_n_examples = int(
            data.get("tci_distillation_n_examples", 0)
        )
        critic.tci_distillation_alpha = data.get("tci_distillation_alpha")
        critic.tci_distillation_metrics = data.get("tci_distillation_metrics")
        # Task 1/4: training mode + example counts (defaults for old checkpoints).
        critic.training_mode = data.get("training_mode", "observational")
        critic.n_observational_examples = int(
            data.get("n_observational_examples", 0)
        )
        critic.n_tci_examples = int(data.get("n_tci_examples", 0))
        critic.tci_schema_version = data.get("tci_schema_version")
        # Task 3/6: TCI value head (defaults for old checkpoints).
        critic.tci_value_head = data.get("tci_value_head")
        critic.tci_rank_examples = int(data.get("tci_rank_examples", 0))
        critic.tci_value_examples = int(data.get("tci_value_examples", 0))
        critic._fitted = True
        return critic


def _cluster_bootstrap_with_full_coverage(
    y: np.ndarray,
    clusters: dict,
    required_classes: set[int],
    rng: np.random.Generator,
    *,
    max_attempts: int = 10,
) -> np.ndarray | None:
    """Cluster bootstrap: resample clusters, keep all member rows together.

    Each draw selects clusters with replacement; drawing a cluster adds
    *all* of its rows, so one cluster's rows never split across a member.
    Under shared controls a cluster is a task-receiver control family.
    Returns None when no draw covers the required classes.
    """
    cluster_keys = list(clusters.keys())
    for _ in range(max_attempts):
        chosen = rng.choice(
            len(cluster_keys), size=len(cluster_keys), replace=True
        )
        idx: list[int] = []
        for pos in chosen:
            idx.extend(clusters[cluster_keys[pos]])
        idx_arr = np.asarray(idx, dtype=int)
        if required_classes.issubset(set(np.unique(y[idx_arr]).tolist())):
            return idx_arr
    return None


def _bootstrap_with_full_coverage(
    y: np.ndarray,
    required_classes: set[int],
    rng: np.random.Generator,
    *,
    max_attempts: int = 10,
) -> np.ndarray | None:
    """Resample until the bootstrap draw covers all required classes.

    Returns None after ``max_attempts`` failures so the caller can skip the
    member; padding missing classes with zero probability is forbidden.
    """
    for _ in range(max_attempts):
        idx = _stratified_bootstrap_indices(y, rng)
        if required_classes.issubset(set(np.unique(y[idx]).tolist())):
            return idx
    return None


def _stratified_bootstrap_indices(
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stratified bootstrap ensuring all classes are represented."""
    sampled: list[int] = []
    for cls in np.unique(y):
        cls_indices = np.flatnonzero(y == cls)
        sampled.extend(
            rng.choice(
                cls_indices,
                size=len(cls_indices),
                replace=True,
            ).tolist()
        )
    sampled_array = np.asarray(sampled)
    rng.shuffle(sampled_array)
    return sampled_array


def _binary_prob(model: Any, X: Any) -> float:
    """P(target=1) from a binary LogisticRegression model."""
    p = model.predict_proba(X)[0]
    # Find the index of class 1.
    classes = model.classes_
    idx1 = list(classes).index(1) if 1 in classes else None
    if idx1 is None:
        # Model only saw one class; return 0 or 1.
        return float(classes[0] == 1)
    return float(p[idx1])


# ---------------------------------------------------------------------------
# RIMA canonical refactor (2026-08-31): semantic alias.
#
# Phase 3 renames the four-outcome binary critic to ``RIMA-Binary`` semantics
# (controlled ablation only). The original class name is kept so that legacy
# imports and historical experiments do not break; the formal method is
# ``OfficialScoreTransferCritic``.
# ---------------------------------------------------------------------------
BinaryFourOutcomeTransferCritic = FourOutcomeTransferCritic
"""Semantic alias (Phase 3): binary four-outcome critic = ``RIMA-Binary``.

Controlled-ablation only; NOT the MultiAgentBench main method.
"""
