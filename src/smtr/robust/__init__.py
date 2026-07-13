"""Optional Robust-SMTR extension.

Importing this package is required to construct Robust-SMTR routers. The
formal SMTR router and factory do not import this package by default.
"""

from smtr.robust.config import RobustSMTRGateConfig
from smtr.robust.factory import build_robust_smtr_router
from smtr.robust.robust_gate import RobustSMTRGate

__all__ = [
    "RobustSMTRGate",
    "RobustSMTRGateConfig",
    "build_robust_smtr_router",
]
