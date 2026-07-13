from smtr.router.baseline_router import NoMemoryRouter
from smtr.router.candidate_proposer import DeterministicHybridCandidateProposer

__all__ = [
    "DeterministicHybridCandidateProposer",
    "NoMemoryRouter",
]


def __getattr__(name: str):
    """Lazy imports for RelevanceTopKRouter and build_router to avoid circular imports."""
    if name == "RelevanceTopKRouter":
        from smtr.router.baselines import RelevanceTopKRouter

        return RelevanceTopKRouter
    if name == "RelevanceTopKRouterConfig":
        from smtr.router.baselines import RelevanceTopKRouterConfig

        return RelevanceTopKRouterConfig
    if name == "build_router":
        from smtr.router.factory import build_router

        return build_router
    if name == "RouterMode":
        from smtr.router.factory import RouterMode

        return RouterMode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

