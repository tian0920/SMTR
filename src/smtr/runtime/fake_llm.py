from typing import Any

# Interaction rules: (target_strategy, prefix_strategy) -> resolved_sequence_key
# These rules model how prefix memories modulate the target memory's effect.
# "valid" → follow the correct sequence (success)
# "wrong" → follow the wrong sequence (failure)
# "locked" → lock then wrong (failure, resource contention)
_PREFIX_TARGET_INTERACTIONS: dict[tuple[str, str], str] = {
    # recover + block → locked (positive → negative flip)
    ("recover", "block"): "locked",
    # recover + conflict → wrong (positive → negative flip)
    ("recover", "conflict"): "wrong",
    # destructive + override → valid (negative → positive flip)
    ("destructive", "override"): "valid",
    # destructive + amplify → locked (negative stays negative, different mechanism)
    ("destructive", "amplify"): "locked",
    # recover + reinforce → valid (positive stays positive, stronger)
    ("recover", "reinforce"): "valid",
    # irrelevant + enable → valid (neutral → positive flip)
    ("irrelevant", "enable"): "valid",
    # irrelevant + block → wrong (neutral → negative flip)
    ("irrelevant", "block"): "wrong",
}


class DeterministicFakeLLM:
    def plan(
        self,
        task: str,
        observation: dict[str, Any],
        visible_payloads: list[dict],
    ) -> dict[str, Any]:
        strategies = _payload_strategies(visible_payloads)
        target_strategy = _target_strategy(strategies)
        prefix_strategies = _prefix_strategies(strategies)

        # Check for prefix-target interaction effects
        sequence_key = _resolve_interaction(target_strategy, prefix_strategies)

        # Hidden perturbation: when offset is 0 AND there is a prefix-target
        # interaction, override the result to "default".  This creates label
        # noise invisible to the critic (the perturbation field is excluded
        # from the context fingerprint), making scenario-family splits
        # non-trivially separable.
        perturbation = observation.get("perturbation_offset", -1)
        if perturbation == 0 and prefix_strategies and target_strategy is not None:
            sequence_key = "default"

        if sequence_key == "valid":
            sequence = list(observation.get("valid_sequence", []))
        elif sequence_key == "locked":
            sequence = ["lock_target", *list(observation.get("valid_sequence", []))]
        elif sequence_key == "wrong":
            sequence = ["wrong_action"]
        elif "lock_target" in prefix_strategies:
            # Legacy: lock_target as a prefix strategy directly
            sequence = ["lock_target", *list(observation.get("valid_sequence", []))]
        elif target_strategy == "recover":
            sequence = list(observation.get("valid_sequence", []))
        elif target_strategy == "destructive":
            sequence = ["wrong_action"]
        else:
            sequence = list(
                observation.get("default_sequence", observation.get("valid_sequence", []))
            )
        return {
            "plan": sequence,
            "explanation": f"Use the observed valid sequence for task: {task}",
            "local_trace": {
                "source": "deterministic_fake_llm",
                "step_count": len(sequence),
                "target_strategy": target_strategy,
                "prefix_strategies": prefix_strategies,
                "resolved_sequence": sequence_key,
            },
        }

    def summarize_execution(self, results: list[dict[str, Any]]) -> str:
        ok_count = sum(1 for result in results if result.get("ok"))
        return f"Executed {ok_count}/{len(results)} actions successfully."


def _payload_strategies(visible_payloads: list[dict]) -> list[str]:
    strategies: list[str] = []
    for payload in visible_payloads:
        for step in payload.get("steps", []):
            text = str(step).strip().lower()
            if text.startswith("strategy:"):
                strategies.append(text.split(":", 1)[1].strip())
    return strategies


def _target_strategy(strategies: list[str]) -> str | None:
    """Extract the target memory's strategy (first non-prefix strategy)."""
    prefix_set = {"block", "conflict", "override", "amplify", "reinforce", "enable"}
    for s in strategies:
        if s not in prefix_set:
            return s
    return None


def _prefix_strategies(strategies: list[str]) -> list[str]:
    """Extract prefix memory strategies."""
    prefix_set = {"block", "conflict", "override", "amplify", "reinforce", "enable"}
    return [s for s in strategies if s in prefix_set]


def _resolve_interaction(
    target_strategy: str | None,
    prefix_strategies: list[str],
) -> str:
    """Resolve prefix-target interaction to a sequence key.

    Returns one of: "valid", "wrong", "locked", "default".
    
    Key behavior: prefix strategies alone (without a target strategy) do NOT
    trigger interactions. They fall through to "default" which uses the
    environment's default_sequence. This ensures that when the target is
    withheld but prefix memories are visible, the planner generates a plan
    based on the default sequence (not the prefix strategy alone).
    """
    if not prefix_strategies:
        # No prefix → use target strategy directly
        return target_strategy or "default"

    if target_strategy is None:
        # Only prefix visible, no target → prefix alone has no effect
        # Fall through to default behavior
        return "default"

    # Check interaction rules for each prefix strategy
    for prefix_s in prefix_strategies:
        key = (target_strategy, prefix_s)
        if key in _PREFIX_TARGET_INTERACTIONS:
            return _PREFIX_TARGET_INTERACTIONS[key]

    # No interaction rule → fall through to target strategy
    return target_strategy
