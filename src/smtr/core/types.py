from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

AgentRole = Literal[
    "planner",
    "executor",
    "critic",
    "verifier",
    "coordinator",
    "unknown",
]


class AgentProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    role: AgentRole
    capabilities: tuple[str, ...] = ()
    model_name: str | None = None
    tool_names: tuple[str, ...] = ()


class ReceiverState(BaseModel):
    """Receiver context available before team execution begins.

    SMTR-v1 scope: this is *pre-execution* context only (task, receiver
    profile, environment signature). It is not an online/dynamic episode
    state. Routing decisions are made before the team episode starts.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    scenario: str
    task_instruction: str
    receiver: AgentProfile

    subtask: str | None = None
    local_context_summary: str = ""
    team_context_summary: str = ""
    environment_signature: tuple[str, ...] = ()
    # SMTR-v1 fixes S = ∅: no selected-memory prefix is routed.
    selected_memory_ids: tuple[str, ...] = ()


class MemoryProvenance(BaseModel):
    """Provenance of a memory (清单 Writer-Agnostic 第二章).

    Source-agent identity is provenance only: it may be used for auditing,
    debugging and reproducibility, but MUST NOT enter critic features,
    candidate scoring, cohorts, baseline ranking, router inputs,
    calibration, epsilon selection or result stratification.
    This object must never be placed inside ``CandidateExposureInput``.
    """

    model_config = ConfigDict(frozen=True)

    source_agent_id: str
    source_agent_role: AgentRole = "unknown"
    source_task_id: str
    source_trajectory_id: str
    source_split: str
    source_scenario: str


class ProcedurePayload(BaseModel):
    """Memory payload (schema v2, 清单 Writer-Agnostic 第三章).

    ``provenance`` lives in the memory-pool artifact only; the rendered
    injection text never includes provenance fields.
    """

    model_config = ConfigDict(frozen=True)

    memory_id: str
    procedure: str
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()

    provenance: MemoryProvenance
    version: str = "v2"


class MemoryRoutingCard(BaseModel):
    """Routing-card schema v3 (清单 Writer-Agnostic 第三章).

    Writer/source-agent identity is removed from the routing surface;
    implicit writer capabilities are replaced by explicit memory
    requirements (required tools / capabilities / execution roles /
    environment constraints / preconditions).
    """

    model_config = ConfigDict(frozen=True)

    memory_id: str
    goal_summary: str
    task_tags: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    execution_role_tags: tuple[AgentRole, ...] = ()
    environment_constraints: tuple[str, ...] = ()
    precondition_tags: tuple[str, ...] = ()
    procedure_type: str = "unknown"
    procedure_length_bucket: str = "unknown"
    read_write_scope: str = "unknown"

    evidence_count: int = 0


class CandidateExposureInput(BaseModel):
    """One pre-execution routing input: (t, x_r^pre, m, r).

    Writer-agnostic (清单 Writer-Agnostic 第一章): conditioning is on the
    task, the receiver's pre-execution state, the candidate memory and the
    receiver identity. No provenance enters the routing input.

    SMTR-v1: single receiver, single candidate memory exposure, S = ∅.
    """

    model_config = ConfigDict(frozen=True)

    receiver_state: ReceiverState
    candidate_card: MemoryRoutingCard


class PairedTransferOutcome(BaseModel):
    """Paired potential outcomes on the team-level success indicator.

    Y_1 = team outcome when receiver r is exposed to memory m;
    Y_0 = team outcome when memory m is withheld from receiver r.
    Labels follow the four-outcome taxonomy derived from (Y_1, Y_0).
    """

    model_config = ConfigDict(frozen=True)

    y_share_team: bool
    y_withhold_team: bool
    y_share_local: bool | None = None
    y_withhold_local: bool | None = None

    @property
    def four_outcome_label(self) -> str:
        if self.y_share_team and not self.y_withhold_team:
            return "positive_transfer"
        if not self.y_share_team and self.y_withhold_team:
            return "negative_transfer"
        if self.y_share_team and self.y_withhold_team:
            return "neutral_success"
        return "neutral_failure"


class TransferPrediction(BaseModel):
    """Predicted four-outcome distribution q(t, x_r^pre, m, r).

    Writer-agnostic conditioning (清单 Writer-Agnostic 第一章):
    q(Y^share, Y^withhold | t, o_r, m, r).

    Estimands:
      tau = P(Y_1=1 | t, x_r^pre, m, r) - P(Y_0=1 | t, x_r^pre, m, r)
          = q10 - q01   (tau_hat)
      eta = P(Y_1=0, Y_0=1 | t, x_r^pre, m, r)
          = q01         (eta_hat, negative-transfer risk)
    """

    model_config = ConfigDict(frozen=True)

    q00_neutral_failure: float
    q01_negative_transfer: float
    q10_positive_transfer: float
    q11_neutral_success: float
    eta_hat_calibrated: float | None = None

    @property
    def tau_hat(self) -> float:
        return self.q10_positive_transfer - self.q01_negative_transfer

    @property
    def eta_hat(self) -> float:
        return self.q01_negative_transfer

    @property
    def eta_hat_raw(self) -> float:
        return self.q01_negative_transfer

    @property
    def eta_raw(self) -> float:
        """Raw negative-transfer risk (= q01), before any calibration (R6 P0-6)."""
        return self.q01_negative_transfer

    @property
    def eta_calibrated(self) -> float | None:
        """Calibrated negative-transfer risk, or None when not calibrated (R6 P0-6)."""
        return self.eta_hat_calibrated


@dataclass(frozen=True)
class TransferPredictionDistribution:
    """Bootstrap-ensemble uncertainty over the transfer estimands (清单第九章).

    ``mean`` is the ensemble-mean prediction; ``tau_lower`` and
    ``eta_upper`` are the pessimistic/optimistic member quantiles used by
    SMTR-UCB: tau_lower = quantile(member_tau, 0.10),
    eta_upper = quantile(member_eta, 0.90).
    """

    mean: TransferPrediction
    tau_std: float
    eta_std: float
    tau_lower: float
    eta_upper: float


class RouterDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    action: Literal["share", "withhold"]
    tau_hat: float
    # R6 P0-7: raw and calibrated risk are stored separately; the gate only
    # ever compares eta_calibrated against risk_budget.
    eta_raw: float = 0.0
    eta_calibrated: float | None = None
    risk_budget: float | None = None
    # Deprecated (R6 P0-7): kept for legacy traces; equals eta_calibrated
    # (falls back to eta_raw). New code must not depend on this field.
    eta_hat: float | None = None
    reason: str

    @model_validator(mode="before")
    @classmethod
    def _normalize_eta_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if data.get("eta_raw") is None and data.get("eta_hat") is not None:
                # Legacy construction passed the risk value as eta_hat only.
                data["eta_raw"] = data["eta_hat"]
            if data.get("eta_hat") is None:
                calibrated = data.get("eta_calibrated")
                data["eta_hat"] = (
                    calibrated if calibrated is not None else data.get("eta_raw")
                )
        return data
