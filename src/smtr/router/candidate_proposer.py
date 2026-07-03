import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from smtr.memory.schemas import FactValue, MemoryRoutingCard
from smtr.router.traces import CandidateTrace

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "with",
}


class CandidateRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: str
    task_stage: str
    receiver_agent_id: str
    receiver_role: str
    receiver_capabilities: list[str] = Field(default_factory=list)
    environment_observation: dict[str, FactValue] = Field(default_factory=dict)
    local_context_summary: str = ""
    top_k: int = 5
    seed: int = 0


class CandidateScore(CandidateTrace):
    pass


class CandidateProposal(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: CandidateRequest
    ranked_candidates: list[CandidateScore]
    pool_revision: int


def _tokens(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _task_tags(task: str) -> set[str]:
    tokens = _tokens(task)
    tags = set(tokens)
    if {"artifact", "target"} & tokens:
        tags.add("artifact")
    if {"plan", "sequence", "ordered"} & tokens:
        tags.add("ordered-actions")
    if {"execute", "action", "actions", "tool"} & tokens:
        tags.add("tool-chain")
    if {"judge", "verify", "check", "critic"} & tokens:
        tags.add("verification")
    return tags


def _as_fact_observation(raw: dict[str, Any]) -> dict[str, FactValue]:
    observation: dict[str, FactValue] = {}
    for key, value in raw.items():
        if isinstance(value, str | bool | int | float):
            observation[key] = value
    tags = raw.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            observation[f"tag:{str(tag).lower()}"] = True
    return observation


def _fact_matches(observation: dict[str, FactValue], key: str, expected: FactValue) -> bool | None:
    if key not in observation:
        return None
    return observation[key] == expected


class DeterministicHybridCandidateProposer:
    proposer_name = "DeterministicHybridCandidateProposer"
    proposer_version = "1"

    def propose_from_cards(
        self,
        *,
        request: CandidateRequest,
        cards: list[MemoryRoutingCard],
        pool_revision: int,
    ) -> CandidateProposal:
        request_tags = _task_tags(request.task)
        task_tokens = _tokens(request.task)
        receiver_capabilities = set(request.receiver_capabilities)
        scored: list[CandidateScore] = []

        for card in cards:
            goal_similarity = _jaccard(task_tokens, _tokens(card.goal_summary))
            card_tags = set(card.task_tags)
            task_tag_overlap = len(request_tags & card_tags) / max(1, len(request_tags | card_tags))

            matched_required = 0
            explicit_conflict = False
            explanations: list[str] = []
            for key, expected in card.required_environment_facts.items():
                match = _fact_matches(request.environment_observation, key, expected)
                if match is True:
                    matched_required += 1
                elif match is False:
                    explicit_conflict = True
                    explanations.append(f"required fact mismatch: {key}")

            for key, forbidden in card.forbidden_environment_facts.items():
                if _fact_matches(request.environment_observation, key, forbidden) is True:
                    explicit_conflict = True
                    explanations.append(f"forbidden fact present: {key}")

            environment_compatibility = matched_required / max(
                1, len(card.required_environment_facts)
            )
            if explicit_conflict:
                environment_compatibility = max(0.0, environment_compatibility - 0.60)

            if request.receiver_role in card.compatible_receiver_roles:
                receiver_compatibility = 1.0
            elif receiver_capabilities & set(card.compatible_receiver_capabilities):
                receiver_compatibility = 0.5
            else:
                receiver_compatibility = 0.0

            total_score = (
                0.45 * goal_similarity
                + 0.15 * task_tag_overlap
                + 0.25 * environment_compatibility
                + 0.15 * receiver_compatibility
            )
            if not explanations:
                explanations.append("high-recall deterministic card-only score")
            scored.append(
                CandidateScore(
                    memory_id=card.memory_id,
                    total_score=round(total_score, 8),
                    goal_similarity=round(goal_similarity, 8),
                    task_tag_overlap=round(task_tag_overlap, 8),
                    environment_compatibility=round(environment_compatibility, 8),
                    receiver_compatibility=round(receiver_compatibility, 8),
                    explicit_environment_conflict=explicit_conflict,
                    score_explanation=explanations,
                )
            )

        ranked = sorted(
            scored,
            key=lambda item: (
                -item.total_score,
                item.explicit_environment_conflict,
                item.memory_id,
            ),
        )[: request.top_k]
        return CandidateProposal(
            request=request,
            ranked_candidates=ranked,
            pool_revision=pool_revision,
        )

    def propose(
        self,
        *,
        task: str,
        receiver_agent: str,
        environment_observation: dict[str, Any],
        cards: list[MemoryRoutingCard],
        top_k: int,
        seed: int,
    ) -> list[CandidateTrace]:
        request = CandidateRequest(
            task=task,
            task_stage="legacy",
            receiver_agent_id=receiver_agent,
            receiver_role=receiver_agent,
            receiver_capabilities=[],
            environment_observation=_as_fact_observation(environment_observation),
            local_context_summary="",
            top_k=top_k,
            seed=seed,
        )
        return list(
            self.propose_from_cards(
                request=request,
                cards=cards,
                pool_revision=0,
            ).ranked_candidates
        )


CandidateProposer = DeterministicHybridCandidateProposer
