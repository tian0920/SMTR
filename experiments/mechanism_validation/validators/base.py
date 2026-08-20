"""Base validator class for mechanism validation experiments.

All validators inherit from this base and implement the validate() method.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ValidationResult:
    """Result of a single validation experiment.
    
    Attributes
    ----------
    name : str
        Name of the validation test.
    passed : bool
        Whether the test passed acceptance criteria.
    metrics : dict[str, Any]
        Detailed metrics from the experiment.
    message : str
        Human-readable summary message.
    duration_seconds : float
        Wall-clock time for the experiment.
    artifacts : dict[str, Any]
        Additional data artifacts (e.g., per-model results).
    """
    
    name: str
    passed: bool
    metrics: dict[str, Any]
    message: str
    duration_seconds: float = 0.0
    artifacts: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "name": self.name,
            "passed": self.passed,
            "metrics": self.metrics,
            "message": self.message,
            "duration_seconds": round(self.duration_seconds, 2),
            "artifacts": self.artifacts,
        }


class BaseValidator(ABC):
    """Abstract base class for mechanism validators.
    
    Each validator implements a specific mechanism test:
    - Contrast necessity
    - Receiver conditioning
    - Rank loss necessity
    - Memory shuffle
    - Source leakage
    - Synthetic causal benchmark
    
    Validators are independent and can run in any order.
    """
    
    def __init__(
        self,
        config: dict[str, Any],
        project_root: Path,
    ) -> None:
        """Initialize validator with configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary from YAML config.
        project_root : Path
            Root directory of the project.
        """
        self.config = config
        self.project_root = project_root
        self._data_cache: dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        """Return validator name (for reports)."""
        return self.__class__.__name__
    
    @property
    def data_config(self) -> dict[str, Any]:
        """Return data configuration section."""
        return self.config.get("data", {})
    
    @property
    def model_config(self) -> dict[str, Any]:
        """Return model configuration section."""
        return self.config.get("model", {})
    
    @property
    def training_config(self) -> dict[str, Any]:
        """Return training configuration section."""
        return self.config.get("training", {})
    
    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path against project root."""
        path = Path(relative_path)
        if not path.is_absolute():
            path = self.project_root / path
        return path
    
    def _get_train_path(self) -> Path:
        """Get path to training paired records."""
        return self._resolve_path(
            self.data_config["paired_records"]["train"]
        )
    
    def _get_validation_path(self) -> Path:
        """Get path to validation paired records."""
        return self._resolve_path(
            self.data_config["paired_records"]["validation"]
        )
    
    def _get_test_path(self) -> Path:
        """Get path to test paired records."""
        return self._resolve_path(
            self.data_config["paired_records"]["test"]
        )
    
    def _get_pool_path(self) -> Path:
        """Get path to memory pool."""
        return self._resolve_path(
            self.data_config["memory_pool"]
        )
    
    def _get_contrasts_path(self) -> Path:
        """Get path to intervention contrasts."""
        return self._resolve_path(
            self.data_config["interventions"]["contrasts"]
        )
    
    def _get_perturbations_path(self) -> Path:
        """Get path to perturbations manifest."""
        return self._resolve_path(
            self.data_config["interventions"]["perturbations"]
        )
    
    def _load_paired_records(self, path: Path) -> list[tuple]:
        """Load paired records with metadata.
        
        Returns list of (CandidateExposureInput, label, record).
        """
        cache_key = f"records:{path}"
        if cache_key not in self._data_cache:
            from smtr.router.transfer_features import (
                load_paired_records_with_metadata,
            )
            self._data_cache[cache_key] = load_paired_records_with_metadata(
                path, self._get_pool_path()
            )
        return self._data_cache[cache_key]
    
    def _load_pool(self) -> dict:
        """Load memory pool as dict keyed by memory_id."""
        cache_key = "pool"
        if cache_key not in self._data_cache:
            pool: dict = {}
            pool_path = self._get_pool_path()
            for line in pool_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    m = json.loads(line)
                    pool[m["memory_id"]] = m
            self._data_cache[cache_key] = pool
        return self._data_cache[cache_key]
    
    def _load_tci_tuples(self) -> list[tuple]:
        """Load TCI tuples for evaluation."""
        cache_key = "tci_tuples"
        if cache_key not in self._data_cache:
            from smtr.marble.training import _build_tci_inputs_for_critic
            self._data_cache[cache_key] = _build_tci_inputs_for_critic(
                tci_contrasts_path=self._get_contrasts_path(),
                perturbations_manifest_path=self._get_perturbations_path(),
                paired_records_path=self._get_train_path(),
                memory_pool_path=self._get_pool_path(),
            )
        return self._data_cache[cache_key]
    
    def _get_common_train_kwargs(self) -> dict:
        """Get common kwargs for train_critic."""
        return dict(
            train_records_path=self._get_train_path(),
            validation_records_path=self._get_validation_path(),
            test_records_path=self._get_test_path(),
            memory_pool_path=self._get_pool_path(),
            seed=self.model_config.get("seed", 7),
            n_bootstrap=self.model_config.get("n_bootstrap", 11),
            n_features=self.model_config.get("n_features", 512),
            feature_block=self.model_config.get("feature_block", "full"),
            coverage_mode=self.training_config.get("coverage_mode", "pilot"),
        )
    
    @abstractmethod
    def validate(self) -> ValidationResult:
        """Run the validation experiment.
        
        Returns
        -------
        ValidationResult
            Result containing metrics, pass/fail, and artifacts.
        """
        ...
    
    def _evaluate_pairwise(self, critic, tci_tuples) -> dict:
        """Evaluate pairwise ranking accuracy on TCI tuples.
        
        Parameters
        ----------
        critic : FourOutcomeTransferCritic
            Trained critic.
        tci_tuples : list
            List of (input_orig, input_pert, direction, contrast_type).
            
        Returns
        -------
        dict with pairwise_accuracy, pairwise_margin, n_pairs,
        and per-contrast_type breakdown.
        """
        from smtr.router.tci_supervision import evaluate_tci_loss_on_critic
        return evaluate_tci_loss_on_critic(critic, tci_tuples)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file.
    
    Parameters
    ----------
    config_path : Path
        Path to YAML config file.
        
    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_json_report(
    result: ValidationResult,
    output_path: Path,
) -> None:
    """Save validation result as JSON.
    
    Parameters
    ----------
    result : ValidationResult
        Validation result to save.
    output_path : Path
        Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


def save_json_results(
    results: list[ValidationResult],
    output_path: Path,
) -> None:
    """Save multiple validation results as JSON.
    
    Parameters
    ----------
    results : list[ValidationResult]
        List of validation results.
    output_path : Path
        Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {r.name: r.to_dict() for r in results}
    
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            import numpy as np
            if isinstance(o, (np.bool_,)):
                return bool(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            return super().default(o)
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)


def format_markdown_report(
    results: list[ValidationResult],
) -> str:
    """Format validation results as Markdown report.
    
    Parameters
    ----------
    results : list[ValidationResult]
        List of validation results.
        
    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = [
        "# SMTR Mechanism Validation Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "---",
        "",
    ]
    
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)
    
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        lines.append(f"## {r.name}")
        lines.append("")
        lines.append(f"**Status:** {status}")
        lines.append("")
        lines.append(f"**Message:** {r.message}")
        lines.append("")
        lines.append(f"**Duration:** {r.duration_seconds:.2f}s")
        lines.append("")
        
        if r.metrics:
            lines.append("**Metrics:**")
            lines.append("")
            for key, value in r.metrics.items():
                if isinstance(value, float):
                    lines.append(f"- {key}: {value:.4f}")
                else:
                    lines.append(f"- {key}: {value}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"**Overall:** {passed_count}/{total_count} tests passed")
    lines.append("")
    
    if passed_count == total_count:
        lines.append("### ✅ MECHANISM VERIFIED")
        lines.append("")
        lines.append("All mechanism validation tests passed. "
                    "SMTR core mechanism is validated.")
    else:
        lines.append("### ❌ MECHANISM NOT FULLY VERIFIED")
        lines.append("")
        failed = [r.name for r in results if not r.passed]
        lines.append(f"Failed tests: {', '.join(failed)}")
        lines.append("")
        lines.append("**Action required:** Review failed tests and "
                    "consider modifying theory/estimand before "
                    "scaling experiments.")
    
    return "\n".join(lines)
