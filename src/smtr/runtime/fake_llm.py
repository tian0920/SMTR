from typing import Any


class DeterministicFakeLLM:
    def plan(
        self,
        task: str,
        observation: dict[str, Any],
        visible_payloads: list[dict],
    ) -> dict[str, Any]:
        strategies = _payload_strategies(visible_payloads)
        if "lock_target" in strategies:
            sequence = ["lock_target", *list(observation.get("valid_sequence", []))]
        elif "recover" in strategies:
            sequence = list(observation.get("valid_sequence", []))
        elif "destructive" in strategies:
            sequence = ["wrong_action"]
        else:
            sequence = list(
                observation.get("default_sequence", observation.get("valid_sequence", []))
            )
        return {
            "plan": sequence,
            "explanation": f"Use the observed valid sequence for task: {task}",
            "local_trace": {"source": "deterministic_fake_llm", "step_count": len(sequence)},
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
