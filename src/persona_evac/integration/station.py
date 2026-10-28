"""Station inputs and constraints. Targets are chosen only by private cognition."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class StairSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str
    travel_time_s: float = Field(gt=0, le=300)
    entry_rate_per_s: float = Field(gt=0, le=20)
    capacity: int = Field(ge=1, le=100, strict=True)


class StairGeometry(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str
    width_m: float = Field(ge=0.3, le=10)
    length_m: float = Field(ge=1, le=100)
    speed_mps: float = Field(ge=0.1, le=2)
    density_per_m2: float = Field(ge=0.1, le=3)


class StationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stairs: list[StairSettings | StairGeometry] = Field(default_factory=list, max_length=16)
    behavior_mode: Literal["off", "qwen"] = "off"
    agent_count: int = Field(default=10, ge=10, le=100, strict=True)
    fire_mode: Literal["off", "fds"] = "off"
    experiment_seed: int = Field(default=20261027, ge=0, le=2147483647, strict=True)
    profile_method: Literal['keyword_candidates', 'full_persona'] | None = None
    duration_s: float = Field(default=600, ge=5, le=600)


def resolve_stair(stair: dict) -> None:
    """Uniform stream approximation: Q=rho*v*w, T=L/v, N=floor(rho*w*L).

    Geometry here is a model assumption unless the attached evidence says otherwise.
    An entrance's published width is only a proxy for effective stair width.
    """
    if stair.get("mode") != "manual":
        geometry = StairGeometry.model_validate(
            {key: stair[key] for key in StairGeometry.model_fields}
        )
        occupancy = geometry.width_m * geometry.length_m * geometry.density_per_m2
        if occupancy < 1:
            raise ValueError(
                f"{stair['id']}: width × length × density must allow at least one person"
            )
        stair.update(
            mode="geometry",
            travel_time_s=geometry.length_m / geometry.speed_mps,
            entry_rate_per_s=geometry.density_per_m2 * geometry.speed_mps * geometry.width_m,
            capacity=math.floor(occupancy + 1e-9),
        )
    reference = stair.get("width_reference", {})
    stair["width_basis"] = (
        "official_entrance_proxy"
        if reference.get("reported_width_m") == stair.get("width_m")
        and reference.get("reported_width_m") is not None
        else "assumed"
    )
    if reference.get("method") == "drawing_ratio" and reference.get(
        "estimated_width_m"
    ) == stair.get("width_m"):
        stair["width_basis"] = "drawing_estimate"
    stair["parallel_lanes"] = max(1, math.floor(stair["width_m"] / stair.get("lane_width_m", 0.60)))


def initialize_station(source: dict, config: dict, overrides: StationRequest | None = None) -> dict:
    config = deepcopy(config)
    stairs = {item["id"]: item for item in config["stairs"]}
    seen = set()
    for override in (overrides or StationRequest()).stairs:
        if override.id not in stairs or override.id in seen:
            raise ValueError(f"알 수 없거나 중복된 계단입니다: {override.id}")
        seen.add(override.id)
        stairs[override.id].update(override.model_dump())
        stairs[override.id]["mode"] = (
            "geometry" if isinstance(override, StairGeometry) else "manual"
        )
    for settings in stairs.values():
        resolve_stair(settings)
        if settings["mode"] == "manual":
            StairSettings.model_validate({key: settings[key] for key in StairSettings.model_fields})
    originals = sorted(source["variants"]["C"], key=lambda agent: agent["agent_slot"])
    count = (overrides or StationRequest()).agent_count
    if len({a["persona"]["uuid"] for a in originals}) != len(originals):
        raise ValueError("페르소나 UUID가 중복되었습니다. 서로 다른 원본 인물이 필요합니다.")
    if len(originals) < count:
        raise ValueError(
            f"서로 다른 페르소나 {count}명이 필요하지만 {len(originals)}명만 준비되었습니다. prepare-station-personas로 준비하세요."
        )
    originals = originals[:count]
    if not config.get("areas") or len(config.get("spawn_points", [])) < count:
        raise ValueError("층별 보행 영역과 인원수만큼의 시작 위치가 필요합니다.")
    agents = []
    for index, original in enumerate(originals):
        speed = float(original["desired_speed_mps"])
        delay = float(original["pre_movement_s"])
        if not math.isfinite(speed) or speed <= 0 or not math.isfinite(delay) or delay < 0:
            raise ValueError("시작 지연과 기본 이동속도가 유효하지 않습니다.")
        agent = {
            key: deepcopy(original.get(key))
            for key in (
                "agent_slot",
                "persona",
                "mapping",
                "mapping_source",
                "empirical",
                "scenario_context",
                "provenance",
                "pre_movement_s",
                "desired_speed_mps",
            )
        }
        agent.update(
            agent_id=index + 1,
            source_agent_id=original["agent_id"],
            spawn=deepcopy(config["spawn_points"][index]),
            segments=[],
            stair_wait_s=0.0,
            evacuation_time_s=None,
        )
        agents.append(agent)
    return {
        "schema_version": "station-initialization",
        "scenario": config,
        "agents": agents,
        "source": source_manifest(source, originals),
    }


def source_manifest(source, originals):
    return {
        "experiment_id": source["experiment_id"],
        "backend": source["backend"],
        "pool_assignment_manifest_hash": source["assignment_manifest_hash"],
        "assignment_manifest_hash": hashlib.sha256(
            json.dumps(
                [(a["agent_slot"], a["persona"]["uuid"]) for a in originals],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "provenance": source["provenance"],
        "notice": "서로 다른 원본 페르소나와 개인별 Jülich 시작 지연·기본 속도를 보존",
    }


def simulate_station(
    source: dict, config: dict, overrides: StationRequest | None = None, decision_provider=None
) -> dict:
    from persona_evac.integration.station_dynamics import simulate

    return simulate(source, config, overrides or StationRequest(), decision_provider)


def run_station(
    source_path: Path,
    config_path: Path,
    output_path: Path,
    *,
    qwen: bool = False,
    agent_count: int = 10,
    fire: bool = False,
    experiment_seed=20261027,
    duration_s=600,
    profile_method=None,
) -> Path:
    from persona_evac.station_artifacts import station_signature

    signature = station_signature()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = simulate_station(
        source,
        config,
        StationRequest(
            behavior_mode="qwen" if qwen else "off",
            agent_count=agent_count,
            fire_mode="fds" if fire else "off",
            experiment_seed=experiment_seed,
            duration_s=duration_s,
            profile_method=profile_method,
        ),
    )
    if (
        signature != station_signature()
        or source_sha256 != hashlib.sha256(source_path.read_bytes()).hexdigest()
        or config_sha256 != hashlib.sha256(config_path.read_bytes()).hexdigest()
    ):
        raise RuntimeError("계산 중 코드 또는 입력 자료가 변경되어 결과를 저장하지 않았습니다.")
    result["source"]["artifact_sha256"] = source_sha256
    result["input_config_sha256"] = config_sha256
    result["implementation_sha256"] = signature
    result["computation"] = {
        "code_sha256": signature,
        "kind": "recomputed_physics_with_recorded_decisions",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(f".{uuid4().hex}.tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    temp.replace(output_path)
    return output_path
