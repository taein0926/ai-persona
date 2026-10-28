from __future__ import annotations

import bisect
import csv
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import pyarrow as pa
import pyarrow.parquet as pq

from persona_evac.assignment import write_rows
from persona_evac.hashing import sha256_file
from persona_evac.schemas import (
    ExperimentResults,
    Provenance,
    RuntimeAgentRow,
    TrajectoryPoint,
    VariantSummary,
)


def _read_history(path: Path, value_field: str) -> dict[int, tuple[list[float], list[float]]]:
    rows: dict[int, list[tuple[float, float]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["agent_id"])].append((float(row["time_s"]), float(row[value_field])))
    return {
        agent_id: ([item[0] for item in values], [item[1] for item in values])
        for agent_id, values in rows.items()
    }


def _at_or_before(history: tuple[list[float], list[float]] | None, time_s: float) -> float:
    if not history:
        return 0.0
    times, values = history
    index = max(0, bisect.bisect_right(times, time_s) - 1)
    return values[index]


def _summary(agents) -> VariantSummary:
    times = [agent.evacuation_time_s for agent in agents if agent.evacuation_time_s is not None]
    return VariantSummary(
        variant="C",
        backend="pyfds_evac",
        run_success=True,
        all_evacuated=len(times) == len(agents),
        total_evacuation_time_s=max(times) if len(times) == len(agents) else None,
        mean_pre_movement_s=round(mean(agent.pre_movement_s for agent in agents), 4),
        mean_speed_mps=round(mean(agent.desired_speed_mps for agent in agents), 4),
        exit_split=dict(sorted(Counter(agent.exit_id for agent in agents).items())),
        total_route_switches=sum(agent.route_switches for agent in agents),
        max_fed=max(agent.final_fed for agent in agents),
    )


def materialize_pyfds_results(run_dir: Path, latest_dir: Path | None = None) -> Path:
    """Join real JuPedSim/FDS artifacts back to the Persona result contract."""
    source_path = run_dir / "results.json"
    sqlite_path = run_dir / "pyfds_trajectories.sqlite"
    smoke_path = run_dir / "pyfds_smoke_history.csv"
    fed_path = run_dir / "pyfds_fed_history.csv"
    route_path = run_dir / "pyfds_route_history.csv"
    required = (source_path, sqlite_path, smoke_path, fed_path, route_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing pyFDS artifacts: {', '.join(missing)}")

    result = ExperimentResults.model_validate_json(source_path.read_text(encoding="utf-8"))
    agents = sorted(result.variants["C"], key=lambda item: item.agent_slot)
    smoke = _read_history(smoke_path, "extinction_per_m")
    fed = _read_history(fed_path, "fed_cumulative")
    routes: dict[int, list[dict[str, str]]] = defaultdict(list)
    with route_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            routes[int(row["agent_id"])].append(row)

    with sqlite3.connect(sqlite_path) as connection:
        fps = float(connection.execute("SELECT value FROM metadata WHERE key='fps'").fetchone()[0])
        runtime_ids = [row[0] for row in connection.execute(
            "SELECT DISTINCT id FROM trajectory_data ORDER BY id"
        )]
        if len(runtime_ids) != len(agents):
            raise ValueError(
                f"JuPedSim produced {len(runtime_ids)} agents, expected {len(agents)}"
            )
        scalar_rows = connection.execute(
            "SELECT id, MAX(fed), MAX(speed) FROM agent_scalars GROUP BY id"
        ).fetchall()
        max_fed = {int(row[0]): float(row[1] or 0.0) for row in scalar_rows}
        trajectory_rows = connection.execute(
            "SELECT frame,id,pos_x,pos_y FROM trajectory_data ORDER BY id,frame"
        ).fetchall()

    by_id: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for frame, agent_id, x, y in trajectory_rows:
        by_id[int(agent_id)].append((int(frame), float(x), float(y)))

    runtime_map = []
    parquet_rows = []
    for agent, runtime_id in zip(agents, runtime_ids, strict=True):
        points = by_id[runtime_id]
        sampled = [item for item in points if item[0] % 5 == 0]
        if sampled[-1] != points[-1]:
            sampled.append(points[-1])
        route_events = routes.get(runtime_id, [])
        route_times = [float(event["time_s"]) for event in route_events]
        trajectory = []
        for frame, x, y in sampled:
            time_s = frame / fps
            if frame == points[-1][0]:
                status = "evacuated"
            elif any(abs(time_s - event_time) <= 0.25 for event_time in route_times):
                status = "waiting_or_rerouting"
            elif time_s < agent.pre_movement_s:
                status = "pre_moving"
            else:
                status = "evacuating"
            point = TrajectoryPoint(
                t=round(time_s, 3),
                x=round(x, 6),
                y=round(y, 6),
                status=status,
                extinction=round(_at_or_before(smoke.get(runtime_id), time_s), 8),
                fed=round(_at_or_before(fed.get(runtime_id), time_s), 10),
            )
            trajectory.append(point)
            parquet_rows.append(
                {
                    "variant": "C",
                    "agent_id": runtime_id,
                    "agent_slot": agent.agent_slot,
                    "persona_uuid": agent.persona.uuid,
                    **point.model_dump(mode="json"),
                }
            )
        agent.agent_id = runtime_id
        agent.trajectory = trajectory
        agent.evacuation_time_s = round(points[-1][0] / fps, 3)
        agent.route_switches = len(route_events)
        agent.exit_id = route_events[-1]["new_exit"] if route_events else "exit_B_right"
        agent.final_fed = max_fed.get(runtime_id, trajectory[-1].fed)
        agent.provenance["motion"] = Provenance(
            source_kind="derived",
            source_ref=str(sqlite_path),
            record_id=str(runtime_id),
            method="JuPedSim trajectory with FDS extinction/FED fields",
            version="pyFDS-Evac@4e4b5f2",
            artifact_hash=sha256_file(sqlite_path),
        )
        runtime_map.append(
            RuntimeAgentRow(
                agent_slot=agent.agent_slot,
                jupedsim_agent_id=runtime_id,
                persona_uuid=agent.persona.uuid,
                variant="C",
                replicate_seed=result.replicate_seed,
                assignment_manifest_hash=result.assignment_manifest_hash,
            )
        )

    result.variants["C"] = agents
    result.summaries["C"] = _summary(agents)
    result.backend = "pyfds_evac"
    result.backend_notice = (
        "실제 FDS 6.11.1 연기장과 pyFDS-Evac/JuPedSim 궤적입니다. "
        f"행동 범주 provider: {result.provenance.get('provider', 'unknown')}."
    )
    result.provenance.update(
        {
            "backend": "pyfds_evac",
            "fds_result_dir": str(run_dir.parent.parent / "fds" / "t_junction"),
            "pyfds_sqlite_hash": sha256_file(sqlite_path),
            "pyfds_smoke_history_hash": sha256_file(smoke_path),
            "pyfds_fed_history_hash": sha256_file(fed_path),
        }
    )
    output = run_dir / "pyfds_results.json"
    output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    write_rows(run_dir / "pyfds_agent_runtime_map.parquet", runtime_map)
    pq.write_table(pa.Table.from_pylist(parquet_rows), run_dir / "pyfds_trajectories.parquet")
    if latest_dir is not None:
        latest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, latest_dir / "results.json")
    return output
