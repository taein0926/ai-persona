from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Tendency(StrEnum):
    FAST = "fast"
    NORMAL = "normal"
    SLOW = "slow"


class Familiarity(StrEnum):
    FULL = "full"
    DISCOVERY = "discovery"


class SocialMode(StrEnum):
    INDIVIDUAL = "individual"
    GROUP = "group"


class PersonaProfile(BaseModel):
    model_config = ConfigDict(extra="allow")
    uuid: str = Field(min_length=1)
    persona: str = ""
    sex: str = "unknown"
    age: int = Field(ge=0, le=120)
    province: str = "unknown"
    district: str = "unknown"
    occupation: str = "unknown"


class BehaviorReasoning(StrictModel):
    response_tendency: str = Field(min_length=1, max_length=240)
    speed_tendency: str = Field(min_length=1, max_length=240)


class EvidenceQuote(StrictModel):
    field: Literal['persona','professional_persona','sports_persona','hobbies_and_interests','skills_and_expertise']
    text: str = Field(min_length=1,max_length=120)


class TendencyEvidence(StrictModel):
    status: Literal['supported','insufficient']
    quotes: list[EvidenceQuote] = Field(max_length=2)


class BehaviorEvidence(StrictModel):
    response_tendency: TendencyEvidence
    speed_tendency: TendencyEvidence


class OvertakingDisposition(StrictModel):
    # unknown is accepted only for historical artifacts; new inference uses normal.
    disposition: Literal['urgent','normal','relaxed','unknown']
    reason: str = Field(min_length=1,max_length=240,description='한국어 문장으로 성격 근거를 짧게 설명한다.')
    quotes: list[EvidenceQuote] = Field(max_length=2)


class StationFamiliarity(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    level: Literal['first_visit', 'familiar']
    assumption: str = Field(min_length=1, max_length=240)


class BehaviorMapping(StrictModel):
    response_tendency: Tendency
    speed_tendency: Tendency
    reasoning_summary: BehaviorReasoning
    evidence: BehaviorEvidence | None = None
    overtaking: OvertakingDisposition | None = None
    familiarity: StationFamiliarity | None = None


class EvidenceReasoning(StrictModel):
    response_tendency: str = Field(min_length=8,max_length=60,description='위험 인지·대응 관련 근거와 해석을 한국어 문장으로 짧게 설명한다.')
    speed_tendency: str = Field(min_length=8,max_length=60,description='본인의 실제 운동·활동 서술과 이동 해석을 한국어 문장으로 짧게 설명한다.')


class EvidenceBehaviorMapping(BehaviorMapping):
    evidence: BehaviorEvidence
    reasoning_summary: EvidenceReasoning
    overtaking: OvertakingDisposition


class AssignmentRow(StrictModel):
    experiment_id: str
    replicate_seed: int
    assignment_seed: int
    agent_slot: int = Field(ge=0)
    persona_uuid: str
    persona_manifest_hash: str
    assignment_algorithm: Literal["fisher_yates_sha256_v1"] = "fisher_yates_sha256_v1"
    assignment_version: Literal["1"] = "1"


class RuntimeAgentRow(StrictModel):
    agent_slot: int
    jupedsim_agent_id: int
    persona_uuid: str
    variant: Literal["A", "B", "C"]
    replicate_seed: int
    assignment_manifest_hash: str


class EmpiricalRecord(StrictModel):
    record_id: str
    treatment: SocialMode
    response_time_s: float = Field(ge=0)
    observed_speed_proxy_mps: float = Field(gt=0)
    response_tendency: Tendency | None = None
    speed_tendency: Tendency | None = None


class ResolvedBehavior(StrictModel):
    response_record_id: str
    speed_record_id: str
    response_treatment: SocialMode
    response_time_s: float
    observed_speed_proxy_mps: float
    desired_speed_mps: float
    requested_response_tendency: Tendency
    requested_speed_tendency: Tendency
    resolved_response_tendency: Tendency
    resolved_speed_tendency: Tendency
    fallback_distance: float = 0.0
    speed_is_proxy: Literal[True] = True


class ScenarioContext(StrictModel):
    role: Literal["visitor"] = "visitor"
    social_mode: SocialMode
    familiarity: Literal["discovery"] = "discovery"


class SimulationParameters(StrictModel):
    premovement_s: float = Field(ge=0)
    desired_speed_mps: float = Field(gt=0)
    familiarity: Literal["discovery"] = "discovery"


class Provenance(StrictModel):
    source_kind: Literal[
        "nemotron", "llm", "empirical", "scenario_assumption", "pyfds_default", "derived"
    ]
    source_ref: str
    record_id: str | None = None
    method: str
    version: str
    artifact_hash: str


class TrajectoryPoint(StrictModel):
    t: float
    x: float
    y: float
    status: Literal[
        "pre_moving", "evacuating", "waiting_or_rerouting", "incapacitated", "evacuated"
    ]
    extinction: float = 0.0
    fed: float = 0.0


class AgentResult(StrictModel):
    variant: Literal["A", "B", "C"]
    agent_slot: int
    agent_id: int
    persona: PersonaProfile
    mapping: BehaviorMapping | None = None
    # 과거 저장 결과의 출처 값도 읽는다. 실제 모델 호출은 로컬 Ollama만 지원한다.
    mapping_source: Literal["none", "mock", "openai", "groq", "ollama", "fallback"] = "none"
    scenario_context: ScenarioContext
    empirical: ResolvedBehavior | None = None
    simulation_parameters: SimulationParameters
    exit_id: str
    pre_movement_s: float
    desired_speed_mps: float
    evacuation_time_s: float | None
    route_switches: int = 0
    final_fed: float = 0.0
    trajectory: list[TrajectoryPoint]
    provenance: dict[str, Provenance]


class VariantSummary(StrictModel):
    variant: Literal["A", "B", "C"]
    backend: str
    run_success: bool
    all_evacuated: bool
    total_evacuation_time_s: float | None
    mean_pre_movement_s: float
    mean_speed_mps: float
    exit_split: dict[str, int]
    total_route_switches: int
    max_fed: float


class ExperimentResults(StrictModel):
    schema_version: Literal["1"] = "1"
    experiment_id: str
    replicate_seed: int
    assignment_manifest_hash: str
    persona_manifest_hash: str
    backend: str
    backend_notice: str
    variants: dict[Literal["A", "B", "C"], list[AgentResult]]
    summaries: dict[Literal["A", "B", "C"], VariantSummary]
    provenance: dict[str, Any]
