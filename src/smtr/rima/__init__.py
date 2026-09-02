"""RIMA canonical components.

RIMA = Receiver-conditioned Intervention-based Memory Admission.

This package contains the formal (canonical) implementation aligned with the
RIMA paper. Legacy components (SMTR-v1 binary critic, online oracle
prototype) remain in ``src/smtr/router`` and ``src/smtr/memory`` but are
demoted to controlled-ablation status; see
``docs/experiment_lineage/rima_canonical_migration.md``.
"""

__all__ = [
    "RIMA_ESTIMAND",
    "FORMAL_DECISION_SOURCE",
    "FORBIDDEN_DECISION_SOURCES",
]

#: Paper Eq. 4: tau(m, a_r | x_t) = E[Y(1) - Y(0) | m, a_r, x_t]
#: where Y is the normalized official MultiAgentBench Task Score in [0, 1].
RIMA_ESTIMAND = (
    "tau(m,a_r|x_t) = E[Y(1)-Y(0)|m,a_r,x_t]; "
    "Y = normalized official MultiAgentBench Task Score in [0,1]"
)

#: The ONLY decision source allowed in the formal admission path.
FORMAL_DECISION_SOURCE = "frozen_transfer_critic"

#: Decision sources forbidden in the formal path (fail-closed invariant).
FORBIDDEN_DECISION_SOURCES = frozenset(
    {
        "observed_delta",
        "team_success",
        "oracle",
        "ground_truth",
    }
)
