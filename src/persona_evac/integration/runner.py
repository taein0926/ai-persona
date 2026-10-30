from __future__ import annotations

import json
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from persona_evac.assignment import (
    assignment_hash,
    bind_runtime_agents,
    build_assignment,
    write_assignment_bundle,
    write_rows,
)
from persona_evac.empirical import EmpiricalResolver, load_empirical_records
from persona_evac.hashing import sha256_file, sha256_value, stable_seed
from persona_evac.llm import MockProvider, OllamaProvider
from persona_evac.persona.sampling import load_personas, persona_manifest, stratified_sample
from persona_evac.schemas import (
    AgentResult,
    BehaviorMapping,
    ExperimentResults,
    Familiarity,
    PersonaProfile,
    Provenance,
    ScenarioContext,
    SimulationParameters,
    Tendency,
    TrajectoryPoint,
    VariantSummary,
)
from persona_evac.integration.pyfds_adapter import build_persona_scenario


BACKEND = "deterministic_mvp"
BACKEND_NOTICE = (
    "결정론적 MVP 궤적 엔진 결과입니다. FDS/JuPedSim 물리 결과가 아니며 데이터 결합, "
    "paired experiment, provenance 및 UI 검증 용도로만 사용합니다."
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _provenance(kind: str, ref: str, record: str | None, method: str, value: Any) -> Provenance:
    return Provenance(
        source_kind=kind,
        source_ref=ref,
        record_id=record,
        method=method,
        version="1",
        artifact_hash=sha256_value(value),
    )


def _spawn_position(slot: int, replicate_seed: int, polygon: list[list[float]]) -> tuple[float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    rng = random.Random(stable_seed("spawn", replicate_seed, slot))
    return rng.uniform(min(xs) + 0.2, max(xs) - 0.2), rng.uniform(min(ys) + 0.2, max(ys) - 0.2)


def _interpolate_path(points: list[tuple[float, float]], progress: float) -> tuple[float, float]:
    segments = [math.dist(points[index - 1], points[index]) for index in range(1, len(points))]
    remaining = sum(segments) * min(1.0, max(0.0, progress))
    for index, length in enumerate(segments):
        if remaining <= length:
            ratio = 0.0 if length == 0 else remaining / length
            start, end = points[index], points[index + 1]
            return start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio
        remaining -= length
    return points[-1]


def _simulate_agent(
    *,
    variant: str,
    slot: int,
    agent_id: int,
    profile: PersonaProfile,
    mapping: BehaviorMapping | None,
    mapping_source: str,
    empirical: Any,
    scenario: dict[str, Any],
    replicate_seed: int,
    persona_source: Path,
    empirical_source: Path,
    scenario_context: ScenarioContext,
) -> AgentResult:
    start = _spawn_position(slot, replicate_seed, scenario["spawn_polygon"])
    if variant == "A":
        pre_movement = float(scenario["baseline"]["pre_movement_s"])
        desired_speed = float(scenario["baseline"]["desired_speed_mps"])
        familiarity = Familiarity.DISCOVERY
    else:
        pre_movement = empirical.response_time_s
        desired_speed = empirical.desired_speed_mps
        familiarity = Familiarity.DISCOVERY

    exits = {key: tuple(value) for key, value in scenario["exits"].items()}
    nearest = min(exits, key=lambda key: math.dist(start, exits[key]))
    if familiarity == Familiarity.FULL:
        exit_id = nearest
    else:
        exit_id = sorted(exits)[stable_seed("exit", profile.uuid, variant) % len(exits)]
    target = exits[exit_id]
    route_switches = 0
    fire = (6.0, 3.6)
    midpoint = ((start[0] + target[0]) / 2, (start[1] + target[1]) / 2)
    if familiarity == Familiarity.DISCOVERY and math.dist(midpoint, fire) < 2.4:
        other = next(key for key in sorted(exits) if key != exit_id)
        exit_id, target, route_switches = other, exits[other], 1
    bend_y = 0.8 if target[1] < 4 else 7.2
    path = [start, (5.2, bend_y), target]
    distance = sum(math.dist(path[index - 1], path[index]) for index in range(1, len(path)))
    travel_time = distance / desired_speed
    evacuation_time = pre_movement + travel_time
    dt = float(scenario["time_step_s"])
    duration = float(scenario["duration_s"])
    fed = 0.0
    trajectory: list[TrajectoryPoint] = []
    step_count = int(duration / dt) + 1
    for step in range(step_count):
        t = round(step * dt, 6)
        if t < pre_movement:
            x, y, status = *start, "pre_moving"
        elif t >= evacuation_time:
            x, y, status = *target, "evacuated"
        else:
            x, y = _interpolate_path(path, (t - pre_movement) / travel_time)
            status = "evacuating"
        if status == "evacuated":
            extinction = 0.0
        else:
            ramp = min(1.0, t / 20.0)
            extinction = round(1.8 * math.exp(-math.dist((x, y), fire) / 1.8) * ramp, 6)
            fed += extinction * dt * 0.0008
        trajectory.append(
            TrajectoryPoint(t=t, x=round(x, 4), y=round(y, 4), status=status, extinction=extinction, fed=round(fed, 8))
        )
        if status == "evacuated":
            break

    provenance = {
        "persona": _provenance(
            "scenario_assumption" if getattr(profile, "dataset_notice", None) else "nemotron",
            str(persona_source),
            profile.uuid,
            "fixed synthetic fixture profile" if getattr(profile, "dataset_notice", None) else "fixed snapshot profile",
            profile.model_dump(mode="json"),
        ),
        "assignment": _provenance(
            "derived", "assignment_manifest.parquet", profile.uuid, "seeded Fisher-Yates permutation", {"slot": slot, "uuid": profile.uuid}
        ),
        "motion": _provenance(
            "scenario_assumption", "deterministic_mvp", None, "piecewise-linear validation trajectory", {"path": path, "speed": desired_speed}
        ),
    }
    if mapping:
        provenance["mapping"] = _provenance(
            "llm", mapping_source, profile.uuid, "structured categorical mapping", mapping.model_dump(mode="json")
        )
    if empirical:
        provenance["empirical"] = _provenance(
            "empirical",
            str(empirical_source),
            f"response:{empirical.response_record_id};speed:{empirical.speed_record_id}",
            "treatment-specific response quantile and pooled observed-speed quantile",
            empirical.model_dump(mode="json"),
        )
    return AgentResult(
        variant=variant,
        agent_slot=slot,
        agent_id=agent_id,
        persona=profile,
        mapping=mapping,
        mapping_source=mapping_source,
        scenario_context=scenario_context,
        empirical=empirical,
        simulation_parameters=SimulationParameters(
            premovement_s=pre_movement,
            desired_speed_mps=desired_speed,
            familiarity="discovery",
        ),
        exit_id=exit_id,
        pre_movement_s=pre_movement,
        desired_speed_mps=desired_speed,
        evacuation_time_s=round(evacuation_time, 4) if evacuation_time <= duration else None,
        route_switches=route_switches,
        final_fed=round(fed, 8),
        trajectory=trajectory,
        provenance=provenance,
    )


def _summary(variant: str, agents: list[AgentResult]) -> VariantSummary:
    times = [agent.evacuation_time_s for agent in agents if agent.evacuation_time_s is not None]
    return VariantSummary(
        variant=variant,
        backend=BACKEND,
        run_success=True,
        all_evacuated=len(times) == len(agents),
        total_evacuation_time_s=max(times) if len(times) == len(agents) else None,
        mean_pre_movement_s=round(mean(agent.pre_movement_s for agent in agents), 4),
        mean_speed_mps=round(mean(agent.desired_speed_mps for agent in agents), 4),
        exit_split=dict(sorted(Counter(agent.exit_id for agent in agents).items())),
        total_route_switches=sum(agent.route_switches for agent in agents),
        max_fed=max(agent.final_fed for agent in agents),
    )


def run_experiment(config_path: Path) -> Path:
    config = _load_json(config_path)
    scenario_path = Path(config["scenario"])
    scenario = _load_json(scenario_path)
    if scenario["backend"] != BACKEND:
        raise ValueError(f"This runner only executes {BACKEND}; requested {scenario['backend']}")
    persona_source = Path(config["persona_source"])
    empirical_source = Path(config["empirical_source"])
    all_profiles = load_personas(persona_source)
    selected = stratified_sample(all_profiles, int(config["agent_count"]))
    manifest = persona_manifest(selected, str(persona_source))
    persona_manifest_hash = sha256_value(manifest)
    profile_by_uuid = {profile.uuid: profile for profile in selected}
    assignments, assignment_manifest_hash = build_assignment(
        experiment_id=config["experiment_id"],
        master_seed=int(config["master_seed"]),
        replicate_seed=int(config["replicate_seed"]),
        profiles=selected,
        persona_manifest_hash=persona_manifest_hash,
        agent_count=int(config["agent_count"]),
    )
    output_root = Path(os.environ.get("PERSONA_EVAC_OUTPUT", config["output_dir"]))
    run_dir = output_root / config["experiment_id"] / f"seed-{config['replicate_seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "persona_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written_hash = write_assignment_bundle(run_dir, assignments)
    if written_hash != assignment_manifest_hash:
        raise RuntimeError("Assignment hash changed while writing")

    records = load_empirical_records(empirical_source)
    resolver = EmpiricalResolver(records, int(config["master_seed"]))
    provider_name = os.environ.get(
        "LLM_PROVIDER", os.environ.get("PERSONA_EVAC_PROVIDER", config["provider"])
    )
    if provider_name == "ollama":
        provider = OllamaProvider()
    elif provider_name == "mock":
        provider = MockProvider()
    else:
        raise ValueError(f"지원하지 않는 LLM_PROVIDER입니다: {provider_name}. 모델 추론은 로컬 Ollama만 지원합니다.")
    scenario_context_payload = {
        "scenario": scenario["name"],
        "multiple_exits": True,
        "incident": "fire",
        "mapping_policy": "categorical_only",
        "role": config["role"],
        "social_mode": config["social_mode"],
        "familiarity": config["familiarity"],
    }
    fixed_context = ScenarioContext(
        role=config["role"],
        social_mode=config["social_mode"],
        familiarity=config["familiarity"],
    )
    variants: dict[str, list[AgentResult]] = {}
    runtime_rows = []
    for variant_index, variant in enumerate(("A", "B", "C"), start=1):
        runtime_ids = [variant_index * 10_000 + slot for slot in range(len(assignments))]
        bound = bind_runtime_agents(assignments, runtime_ids, variant, assignment_manifest_hash)
        runtime_rows.extend(bound)
        agents = []
        for row in bound:
            profile = profile_by_uuid[row.persona_uuid]
            mapping = None
            mapping_source = "none"
            empirical = None
            if variant == "B":
                tendency_values = list(Tendency)
                seed = stable_seed("empirical", profile.uuid, config["master_seed"])
                response = tendency_values[seed % 3]
                speed = tendency_values[(seed >> 4) % 3]
                empirical = resolver.resolve(
                    persona_uuid=profile.uuid,
                    variant=variant,
                    treatment=fixed_context.social_mode,
                    response=response,
                    speed=speed,
                )
            elif variant == "C":
                mapping = provider.map(profile, scenario_context_payload)
                mapping_source = provider.source_name
                empirical = resolver.resolve(
                    persona_uuid=profile.uuid,
                    variant=variant,
                    treatment=fixed_context.social_mode,
                    response=mapping.response_tendency,
                    speed=mapping.speed_tendency,
                )
            agents.append(
                _simulate_agent(
                    variant=variant,
                    slot=row.agent_slot,
                    agent_id=row.jupedsim_agent_id,
                    profile=profile,
                    mapping=mapping,
                    mapping_source=mapping_source,
                    empirical=empirical,
                    scenario=scenario,
                    replicate_seed=int(config["replicate_seed"]),
                    persona_source=persona_source,
                    empirical_source=empirical_source,
                    scenario_context=fixed_context,
                )
            )
        variants[variant] = agents
    write_rows(run_dir / "agent_runtime_map.parquet", runtime_rows)
    if len({assignment_hash(assignments) for _ in variants}) != 1:
        raise RuntimeError("Paired variants do not share one assignment manifest")
    summaries = {variant: _summary(variant, agents) for variant, agents in variants.items()}
    result = ExperimentResults(
        experiment_id=config["experiment_id"],
        replicate_seed=int(config["replicate_seed"]),
        assignment_manifest_hash=assignment_manifest_hash,
        persona_manifest_hash=persona_manifest_hash,
        backend=BACKEND,
        backend_notice=BACKEND_NOTICE,
        variants=variants,
        summaries=summaries,
        provenance={
            "config": str(config_path),
            "config_hash": sha256_file(config_path),
            "scenario": str(scenario_path),
            "scenario_hash": sha256_file(scenario_path),
            "persona_source_hash": sha256_file(persona_source),
            "empirical_source_hash": sha256_file(empirical_source),
            "master_seed": config["master_seed"],
            "replicate_seed": config["replicate_seed"],
            "provider": provider_name,
            "model": getattr(provider, "model", None),
            "backend": BACKEND,
            "pyfds_evac_commit": "4e4b5f2b650b822e84409e68d2f27379b7a2aebf",
            "fds_version": "6.11.1",
        },
    )
    pyfds_scenario = build_persona_scenario(
        upstream_config=Path("vendor/pyFDS-Evac/assets/t_junction/config.json"),
        agents=variants["C"],
        output=run_dir / "pyfds_scenario.json",
        seed=int(config["replicate_seed"]),
    )
    result.provenance["pyfds_scenario"] = str(pyfds_scenario)
    result.provenance["pyfds_scenario_hash"] = sha256_file(pyfds_scenario)
    results_path = run_dir / "results.json"
    results_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    trajectory_rows = [
        {
            "variant": variant,
            "agent_id": agent.agent_id,
            "agent_slot": agent.agent_slot,
            "persona_uuid": agent.persona.uuid,
            **point.model_dump(mode="json"),
        }
        for variant, agents in variants.items()
        for agent in agents
        for point in agent.trajectory
    ]
    pq.write_table(pa.Table.from_pylist(trajectory_rows), run_dir / "trajectories.parquet")
    latest = output_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(results_path, latest / "results.json")
    shutil.copy2(run_dir / "assignment_manifest.parquet", latest / "assignment_manifest.parquet")
    return results_path
