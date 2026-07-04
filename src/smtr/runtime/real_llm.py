"""Real LLM adapter supporting local model inference or remote API.

Implements the same interface as DeterministicFakeLLM so it can be used
as a drop-in replacement in the SMTR pipeline.
"""

import json
import os
import re
from typing import Any


class RealLLM:
    """LLM adapter supporting local model or remote API."""

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3.5-2B",
        api_base: str | None = None,
        api_key: str | None = None,
        load_in_8bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> None:
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._client: Any = None
        self._model: Any = None
        self._tokenizer: Any = None

        if api_base:
            import httpx

            # Normalize: strip trailing /v1 so /v1/chat/completions works consistently
            normalized_base = api_base.rstrip("/")
            if normalized_base.endswith("/v1"):
                normalized_base = normalized_base[:-3]

            # Resolve API key: explicit param > env var > None
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
            headers = {}
            if resolved_key:
                headers["Authorization"] = f"Bearer {resolved_key}"

            self._client = httpx.Client(
                base_url=normalized_base, timeout=120, headers=headers,
            )
        else:
            self._load_local_model(model_name=model_name, load_in_8bit=load_in_8bit)

    def _load_local_model(self, *, model_name: str, load_in_8bit: bool) -> None:
        """Load model locally with transformers + optional 8-bit quantization."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs: dict[str, Any] = {"device_map": "auto", "torch_dtype": torch.float16}
        if load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = bnb_config
        self._model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    def _generate(self, prompt: str) -> str:
        """Generate text from prompt using local model or API."""
        if self._client is not None:
            return self._generate_via_api(prompt)
        return self._generate_locally(prompt)

    def _generate_locally(self, prompt: str) -> str:
        """Generate using local model."""
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                do_sample=self._temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _generate_via_api(self, prompt: str) -> str:
        """Generate via remote OpenAI-compatible API."""
        response = self._client.post(
            "/v1/chat/completions",
            json={
                "model": self._model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._temperature,
                "max_tokens": self._max_new_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def plan(
        self,
        task: str,
        observation: dict[str, Any],
        visible_payloads: list[dict],
    ) -> dict[str, Any]:
        """Generate a plan for the given task and observation."""
        prompt = self._build_plan_prompt(task, observation, visible_payloads)
        raw_output = self._generate(prompt)
        plan_data = self._parse_json_response(raw_output)

        if "plan" not in plan_data:
            plan_data["plan"] = list(
                observation.get("default_sequence", observation.get("valid_sequence", []))
            )
        if "explanation" not in plan_data:
            plan_data["explanation"] = f"Generated plan for task: {task}"

        plan_data.setdefault("local_trace", {})
        plan_data["local_trace"]["source"] = "real_llm"
        plan_data["local_trace"]["model"] = self._model_name
        plan_data["local_trace"]["step_count"] = len(plan_data["plan"])
        return plan_data

    def summarize_execution(self, results: list[dict[str, Any]]) -> str:
        """Summarize execution results."""
        ok_count = sum(1 for r in results if r.get("ok"))
        prompt = self._build_summarize_prompt(results)
        try:
            raw_output = self._generate(prompt)
            summary = raw_output.strip()
            if len(summary) > 500:
                summary = summary[:500] + "..."
            return summary if summary else f"Executed {ok_count}/{len(results)} actions."
        except Exception:
            return f"Executed {ok_count}/{len(results)} actions successfully."

    def _build_plan_prompt(
        self,
        task: str,
        observation: dict[str, Any],
        visible_payloads: list[dict],
    ) -> str:
        """Build prompt for plan generation."""
        payload_summaries = []
        for payload in visible_payloads[:3]:
            steps = payload.get("steps", [])
            goal = payload.get("goal_summary", "")
            payload_summaries.append({"goal": goal, "steps": steps[:5]})

        obs_summary = {
            k: v
            for k, v in observation.items()
            if k
            in {
                "valid_sequence",
                "target_artifact",
                "location",
                "inventory",
                "resource_locked",
                "tags",
            }
        }

        return f"""You are a planning agent. Generate a plan as JSON.

Task: {task}

Environment: {json.dumps(obs_summary)}

Relevant memories: {json.dumps(payload_summaries)}

Output ONLY this JSON format:
{{"plan": ["action1", "action2"], "explanation": "brief reason"}}

No other text. Only JSON."""

    def _build_summarize_prompt(self, results: list[dict[str, Any]]) -> str:
        """Build prompt for execution summarization."""
        result_summary = [
            {"action": r.get("action"), "ok": r.get("ok"), "error": r.get("error")}
            for r in results
        ]
        return f"""Summarize the following execution results concisely.

Results: {json.dumps(result_summary)}

Provide a brief summary of what happened."""

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks and think tags."""
        text = raw.strip()
        # Strip <think>...</think> tags (common in small models)
        text = re.sub(r"<think>[\s\S]*?</think>", "", text)
        # Strip markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            text = json_match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            json_obj_match = re.search(r"\{[\s\S]*\}", text)
            if json_obj_match:
                try:
                    return json.loads(json_obj_match.group())
                except json.JSONDecodeError:
                    pass
            return {"plan": [], "explanation": "Failed to parse LLM response"}
