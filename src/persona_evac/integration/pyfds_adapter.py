from __future__ import annotations

import importlib.util
import json
import shutil
from copy import deepcopy
from pathlib import Path

from persona_evac.hashing import stable_seed
from persona_evac.schemas import AgentResult


REQUIRED_FDS_SLICES = {
    "extinction": ("QUANTITY='EXTINCTION COEFFICIENT'",),
    "carbon monoxide": ("QUANTITY='VOLUME FRACTION'", "SPEC_ID='CARBON MONOXIDE'"),
    "carbon dioxide": ("QUANTITY='VOLUME FRACTION'", "SPEC_ID='CARBON DIOXIDE'"),
    "oxygen": ("QUANTITY='VOLUME FRACTION'", "SPEC_ID='OXYGEN'"),
}


def inspect_pyfds_checkout(path: Path = Path("vendor/pyFDS-Evac")) -> dict[str, object]:
    return {
        "checkout": str(path),
        "present": (path / "pyfds_evac").is_dir(),
        "scenario_module": (path / "pyfds_evac/core/scenario.py").exists(),
        "t_junction_geometry": (path / "assets/t_junction/geometry.wkt").exists(),
        "t_junction_fds": (path / "assets/t_junction/t_junction.fds").exists(),
        "importable": importlib.util.find_spec("pyfds_evac") is not None,
    }


def validate_fds_deck(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore").upper()
    slice_blocks = [line for line in text.splitlines() if line.strip().startswith("&SLCF")]
    missing = [
        name
        for name, tokens in REQUIRED_FDS_SLICES.items()
        if not any(all(token in block for token in tokens) for block in slice_blocks)
    ]
    if missing:
        raise ValueError(f"FDS deck is missing required slice quantities: {', '.join(missing)}")


class PyFDSEvacBackend:
    """Explicit boundary for the real backend; never silently substitutes the MVP engine."""

    name = "pyfds_evac"

    def __init__(self, checkout: Path = Path("vendor/pyFDS-Evac")):
        status = inspect_pyfds_checkout(checkout)
        if not status["present"]:
            raise RuntimeError("pyFDS-Evac checkout is missing")
        if not status["importable"]:
            raise RuntimeError(
                "pyFDS-Evac is checked out but not installed. Build its Python 3.12 environment first."
            )

    def run(self) -> None:
        raise RuntimeError(
            "Real pyFDS execution requires precomputed FDS slices. "
            "Use the fds compose profile, then provide the result directory."
        )


def build_persona_scenario(
    *, upstream_config: Path, agents: list[AgentResult], output: Path, seed: int
) -> Path:
    """Build a 10-distribution derived scenario without changing upstream."""
    if len(agents) != 10:
        raise ValueError(f"MVP pyFDS scenario requires exactly 10 agents, got {len(agents)}")
    slots = [agent.agent_slot for agent in agents]
    if len(set(slots)) != len(slots):
        raise ValueError("Agent slots must be unique")
    raw = json.loads(upstream_config.read_text(encoding="utf-8"))
    source_distributions = raw.get("distributions", {})
    if len(source_distributions) != 1:
        raise ValueError("Expected the upstream t_junction scenario to have one distribution")
    source_id, source = next(iter(source_distributions.items()))
    old_transitions = raw.get("transitions", [])
    downstream = [item for item in old_transitions if item.get("from") == source_id]
    retained = [item for item in old_transitions if item.get("from") != source_id]
    if not downstream:
        raise ValueError("Upstream distribution has no transition")
    new_distributions = {}
    new_transitions = retained[:]
    metadata = {}
    for agent in sorted(agents, key=lambda item: item.agent_slot):
        # pyFDS-Evac identifies non-JuPedSim spawn stages by this prefix when
        # it constructs journeys. Preserve that upstream data contract.
        dist_id = f"jps-distributions_persona_{agent.agent_slot:02d}"
        distribution = deepcopy(source)
        # Separate one-agent spawn cells prevent the upstream placement RNG
        # from producing the same first point for every cloned distribution.
        column = agent.agent_slot % 5
        row = agent.agent_slot // 5
        center_x = 18.4 + column * 0.8
        center_y = 3.0 + row * 2.0
        half_size = 0.2
        distribution["coordinates"] = [
            [center_x - half_size, center_y - half_size],
            [center_x + half_size, center_y - half_size],
            [center_x + half_size, center_y + half_size],
            [center_x - half_size, center_y + half_size],
            [center_x - half_size, center_y - half_size],
        ]
        distribution["parameters"] = {
            **source.get("parameters", {}),
            "number": 1,
            "radius": 0.15,
            "v0": agent.simulation_parameters.desired_speed_mps,
            "use_flow_spawning": False,
            "use_premovement": True,
            "premovement_distribution": "uniform",
            "premovement_param_a": agent.simulation_parameters.premovement_s,
            "premovement_param_b": agent.simulation_parameters.premovement_s,
            "premovement_seed": stable_seed(seed, agent.persona.uuid, "premovement") % (2**32),
            "radius_distribution": "constant",
            "v0_distribution": "constant",
            "familiarity": "discovery",
        }
        new_distributions[dist_id] = distribution
        new_transitions.extend({**item, "from": dist_id} for item in downstream)
        metadata[dist_id] = {
            "agent_slot": agent.agent_slot,
            "persona_uuid": agent.persona.uuid,
            "response_record_id": agent.empirical.response_record_id if agent.empirical else None,
            "speed_record_id": agent.empirical.speed_record_id if agent.empirical else None,
        }
    raw["distributions"] = new_distributions
    raw["transitions"] = new_transitions
    for journey in raw.get("journeys", []):
        stages = journey.get("stages", [])
        journey["stages"] = [
            *new_distributions.keys(),
            *(stage for stage in stages if stage != source_id),
        ]
        journey["transitions"] = [
            item for item in new_transitions if item.get("journey_id") == journey.get("id")
        ]
    routing = raw.get("waypoint_routing", {})
    if source_id in routing:
        source_routing = routing.pop(source_id)
        for dist_id in new_distributions:
            routing[dist_id] = deepcopy(source_routing)
    raw.setdefault("config", {}).setdefault("simulation_settings", {})["baseSeed"] = seed
    raw["persona_evac_metadata"] = {
        "schema_version": "1",
        "upstream_config": str(upstream_config),
        "role": "visitor",
        "social_mode": agents[0].scenario_context.social_mode,
        "familiarity": "discovery",
        "distributions": metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    geometry = upstream_config.with_name("geometry.wkt")
    if not geometry.exists():
        raise FileNotFoundError(f"Upstream scenario geometry is missing: {geometry}")
    shutil.copy2(geometry, output.parent / "geometry.wkt")
    return output
