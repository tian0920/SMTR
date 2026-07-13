"""Output writer for B0/B1/M0 comparison experiment."""

import json
from pathlib import Path
from typing import Any

from smtr.experiment.schemas import (
    BaseEpisodeManifestRecord,
    ComparisonRunRecord,
    ExperimentConfig,
    ExperimentSummary,
)


class ExperimentWriter:
    """Handles writing experiment outputs to disk."""

    def __init__(self, output_dir: str | Path, overwrite: bool = False) -> None:
        self.output_dir = Path(output_dir)
        if self.output_dir.exists() and not overwrite:
            raise FileExistsError(
                f"output directory already exists: {self.output_dir}; "
                "use --overwrite to replace"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._runs_path = self.output_dir / "runs.jsonl"
        self._errors_path = self.output_dir / "errors.jsonl"

    def initialize(self) -> None:
        """Truncate output files for a fresh experiment run."""
        self._runs_path.write_text("")
        self._errors_path.write_text("")

    @property
    def runs_path(self) -> Path:
        return self._runs_path

    @property
    def errors_path(self) -> Path:
        return self._errors_path

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "summary.json"

    @property
    def config_path(self) -> Path:
        return self.output_dir / "config.json"

    def write_config(self, config: ExperimentConfig) -> None:
        """Write experiment configuration."""
        data = config.model_dump()
        self.config_path.write_text(
            json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8"
        )

    def append_run(self, record: ComparisonRunRecord) -> None:
        """Append a single run record to runs.jsonl."""
        data = record.model_dump()
        with self._runs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def append_error(self, error_record: dict) -> None:
        """Append an error record to errors.jsonl."""
        with self._errors_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(error_record, default=str) + "\n")

    def write_summary(self, summary: ExperimentSummary) -> None:
        """Write experiment summary."""
        data = summary.model_dump()
        self.summary_path.write_text(
            json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8"
        )

    def load_runs(self) -> list[ComparisonRunRecord]:
        """Parse and load all runs from runs.jsonl."""
        records = []
        if not self._runs_path.exists():
            return records
        for line in self._runs_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                records.append(ComparisonRunRecord.model_validate_json(line))
        return records

    def load_errors(self) -> list[dict]:
        """Parse and load all errors from errors.jsonl."""
        errors = []
        if not self._errors_path.exists():
            return errors
        for line in self._errors_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                errors.append(json.loads(line))
        return errors

    def write_base_episode_manifest(
        self,
        records: list[BaseEpisodeManifestRecord],
    ) -> None:
        """Write frozen base episode manifest."""
        path = self.output_dir / "base_episode_manifest.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(record.model_dump_json() + "\n")

    def write_invocations(self, runs: list[ComparisonRunRecord]) -> None:
        """Write one record per routing invocation."""
        path = self.output_dir / "invocations.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for run in runs:
                for invocation in run.invocations:
                    payload = {
                        "base_episode_id": run.base_episode_id,
                        "method": run.method,
                        "traversal_seed": run.traversal_seed,
                        **invocation.model_dump(),
                    }
                    f.write(json.dumps(payload, default=str) + "\n")

    # --- Next-round ablation output writers ---

    def write_json(self, filename: str, data: Any) -> None:
        """Write arbitrary JSON data to a file in the output directory."""
        path = self.output_dir / filename
        path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")

    def write_jsonl(self, filename: str, records: list[dict]) -> None:
        """Write a list of records as JSONL."""
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")

    def write_decisions(self, runs: list[dict]) -> None:
        """Write decisions.jsonl: one record per router decision."""
        decisions = []
        for run in runs:
            for trace in run.get("router_trace", []):
                for dec in trace.get("decisions", []):
                    record = {
                        "episode_id": run.get("episode_id", ""),
                        "method": run.get("method", ""),
                        "task_seed": run.get("task_seed"),
                        "generation_seed": run.get("generation_seed"),
                        "traversal_seed": run.get("traversal_seed"),
                        "memory_id": dec.get("memory_id", ""),
                        "action": dec.get("action", ""),
                        "reason": dec.get("reason", ""),
                        "candidate_position": dec.get("candidate_position"),
                        "score": dec.get("score"),
                        "tau_mean": dec.get("tau_mean"),
                        "tau_lcb": dec.get("tau_lcb"),
                        "tau_ucb": dec.get("tau_ucb"),
                        "negative_risk_ucb": dec.get("negative_risk_ucb"),
                        "support_distance": dec.get("support_distance"),
                    }
                    decisions.append(record)
        self.write_jsonl("decisions.jsonl", decisions)

    def write_prefix_traces(self, traces: list[dict]) -> None:
        """Write prefix_traces.jsonl."""
        self.write_jsonl("prefix_traces.jsonl", traces)

    def write_scenario_slices(self, slices: dict) -> None:
        """Write scenario_slices.json."""
        self.write_json("scenario_slices.json", slices)

    def write_bottleneck_funnel(self, funnel: dict) -> None:
        """Write bottleneck_funnel.json."""
        self.write_json("bottleneck_funnel.json", funnel)

    def write_paired_comparisons(self, comparisons: dict) -> None:
        """Write paired_comparisons.json."""
        self.write_json("paired_comparisons.json", comparisons)
