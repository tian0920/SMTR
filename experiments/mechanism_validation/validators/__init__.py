"""SMTR Mechanism Validation Validators.

Each validator tests a specific aspect of the SMTR mechanism:
- contrast_test: Intervention contrast necessity
- receiver_test: Receiver conditioning
- ranking_test: Rank loss necessity
- leakage_test: Source identity leakage
- shuffle_test: Memory randomization
- synthetic_test: Synthetic causal benchmark
"""

from .base import BaseValidator, ValidationResult, load_config

from .contrast_test import ContrastNecessityValidator
from .receiver_test import ReceiverConditioningValidator
from .ranking_test import RankLossValidator
from .leakage_test import SourceLeakageValidator
from .shuffle_test import MemoryShuffleValidator
from .synthetic_test import SyntheticCausalValidator

__all__ = [
    "BaseValidator",
    "ValidationResult",
    "load_config",
    "ContrastNecessityValidator",
    "ReceiverConditioningValidator",
    "RankLossValidator",
    "SourceLeakageValidator",
    "MemoryShuffleValidator",
    "SyntheticCausalValidator",
]
