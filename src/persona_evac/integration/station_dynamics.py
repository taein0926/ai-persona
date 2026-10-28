"""JuPedSim floor dynamics coupled to capacity-limited, spillback-aware stairs."""

from __future__ import annotations
from copy import deepcopy
from collections import Counter, deque
import hashlib
import json
import math
import os
import time
from pathlib import Path
import jupedsim as jps
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
from persona_evac.integration.station import initialize_station
from persona_evac.integration.fds_environment import FireField
from persona_evac.integration.station_geometry import floor_geometry
from persona_evac.integration.stair_dynamics import stair_speed_factor

DT = 0.05
RADIUS = 0.22
SEED = 20260916
DECISION_INTERVAL = 10


def point(node):
    return (node["x"], node["y"])


def make_simulation(geometry):
    return jps.Simulation(
        model=jps.AnticipationVelocityModel(rng_seed=SEED), geometry=geometry, dt=DT
    )


def parameters(position, journey, stage, speed):
    return jps.AnticipationVelocityModelAgentParameters(
        position=position,
        journey_id=journey,
        stage_id=stage,
        desired_speed=speed,
        radius=RADIUS,
        time_gap=1.0,
        anticipation_time=1.0,
        reaction_time=0.3,
    )


def pedestrian_geometry(geometry):
    """Close digitization slivers narrower than a pedestrian, retaining wide halls.

    Mitre joins preserve the measured straight edges. This is pedestrian
    navigation clearance only; smoke can still use the original air geometry.
    """
    clearance = RADIUS + 0.02
    return (
        geometry.buffer(-clearance, join_style=2)
        .buffer(clearance, join_style=2)
        .intersection(geometry)
    )


def can_intentionally_overtake(agent, effective_speed, fast_threshold):
    traits = (agent.get("experimental_profile") or {}).get("traits")
    if traits:
        return (
            traits["waiting_patience"]["level"] == "low"
            and traits["exploration"]["level"] in {"normal", "high"}
            and (effective_speed > fast_threshold)
        )
    disposition = (agent.get("mapping") or {}).get("overtaking") or {}
    return (
        disposition.get("disposition") == "urgent"
        and bool(disposition.get("quotes"))
        and (effective_speed > fast_threshold)
    )


def plan_overtake(
    sim,
    geometry,
    agent_id,
    goal,
    goal_stage,
    inactive_ids=(),
    observed_speeds=None,
    apply=True,
    intent_speed=None,
):
    """Choose clear lateral waypoints; JuPedSim still computes every movement step.

    The waypoint planner breaks the symmetry of exact single-file following. It
    never moves positions directly and never relaxes body size or collision rules.
    """
    actor = sim.agent(agent_id)
    own_speed = actor.model.desired_speed if intent_speed is None else intent_speed
    pos = tuple(actor.position)
    distance = math.dist(pos, goal)
    if distance < 3:
        return None
    direction = ((goal[0] - pos[0]) / distance, (goal[1] - pos[1]) / distance)
    normal = (-direction[1], direction[0])
    others = [a for a in sim.agents() if a.id != agent_id and a.id not in inactive_ids]
    candidates = []
    for other in others:
        delta = (other.position[0] - pos[0], other.position[1] - pos[1])
        ahead = sum((a * b for a, b in zip(delta, direction)))
        side = abs(sum((a * b for a, b in zip(delta, normal))))
        front_speed = (observed_speeds or {}).get(other.id, other.model.desired_speed)
        if (
            0.8 < ahead < 3.5
            and side < 0.6
            and (own_speed - front_speed >= 0.3)
            and (front_speed <= 0.7 * own_speed)
        ):
            candidates.append((ahead, other))
    if not candidates:
        return None
    ahead, front = min(candidates, key=lambda item: item[0])
    for sign in (1, -1):
        first = tuple((pos[i] + 0.2 * direction[i] + sign * 0.9 * normal[i] for i in (0, 1)))
        second = tuple(
            (pos[i] + (ahead + 2) * direction[i] + sign * 0.9 * normal[i] for i in (0, 1))
        )
        path = LineString([pos, first, second])
        if not geometry.covers(path.buffer(RADIUS + 0.06)):
            continue
        if any((path.distance(Point(other.position)) < 2 * RADIUS + 0.08 for other in others)):
            continue
        result = {
            "passed_jps_id": front.id,
            "waypoints": [first, second],
            "front_speed_mps": (observed_speeds or {}).get(front.id, front.model.desired_speed),
            "desired_speed_mps": own_speed,
            "ahead_m": ahead,
        }
        if not apply:
            return result
        a = sim.add_waypoint_stage(first, 0.2)
        b = sim.add_waypoint_stage(second, 0.25)
        journey = jps.JourneyDescription([a, b, goal_stage])
        journey.set_transition_for_stage(a, jps.Transition.create_fixed_transition(b))
        journey.set_transition_for_stage(b, jps.Transition.create_fixed_transition(goal_stage))
        jid = sim.add_journey(journey)
        sim.switch_agent_journey(agent_id, jid, a)
        return result
    return None


def plan_yield(sim, geometry, agent_id, goal, goal_stage, inactive_ids=()):
    """Break counterflow standstills with clear walking waypoints, never teleport."""
    actor = sim.agent(agent_id)
    pos = tuple(actor.position)
    route = jps.RoutingEngine(geometry).compute_waypoints(pos, goal)
    target = next((p for p in route if math.dist(pos, p) > 0.2), goal)
    distance = math.dist(pos, target)
    if distance < 0.3:
        return None
    direction = [(target[i] - pos[i]) / distance for i in (0, 1)]
    normal = [-direction[1], direction[0]]
    others = [a for a in sim.agents() if a.id != agent_id and a.id not in inactive_ids]
    if not any((math.dist(pos, a.position) < 1.5 for a in others)):
        return None
    for side in (1, -1):
        for offset in (0.65, 0.45):
            first = tuple((pos[i] + side * offset * normal[i] for i in (0, 1)))
            second = tuple((first[i] + 1.2 * direction[i] for i in (0, 1)))
            path = LineString([pos, first, second])
            if not geometry.covers(path.buffer(RADIUS + 0.015)):
                continue
            if any((path.distance(Point(a.position)) < 2 * RADIUS - 0.004 for a in others)):
                continue
            if any((math.dist(second, a.position) < 2 * RADIUS + 0.05 for a in others)):
                continue
            a = sim.add_waypoint_stage(first, 0.12)
            b = sim.add_waypoint_stage(second, 0.2)
            journey = jps.JourneyDescription([a, b, goal_stage])
            journey.set_transition_for_stage(a, jps.Transition.create_fixed_transition(b))
            journey.set_transition_for_stage(b, jps.Transition.create_fixed_transition(goal_stage))
            sim.switch_agent_journey(agent_id, sim.add_journey(journey), a)
            return [first, second]
    return plan_cluster_detour(sim, geometry, agent_id, goal, goal_stage, inactive_ids)


def plan_cluster_detour(sim, geometry, agent_id, goal, goal_stage, inactive_ids=()):
    """Route around a stationary cluster, allowing an initial step backwards.

    Other bodies become temporary routing obstacles with comfort clearance.
    A short connector may start inside that comfort margin, but never crosses
    another body. All waypoints are still walked by JuPedSim at normal speed.
    """
    pos = tuple(sim.agent(agent_id).position)
    others = [
        a
        for a in sim.agents()
        if a.id != agent_id and a.id not in inactive_ids and (math.dist(pos, a.position) < 6)
    ]
    if not others:
        return None
    centers = geometry.buffer(-RADIUS - 0.12).difference(
        unary_union([Point(a.position).buffer(2 * RADIUS + 0.12, quad_segs=4) for a in others])
    )
    components = [
        p
        for p in getattr(centers, "geoms", [centers])
        if p.geom_type == "Polygon" and p.covers(Point(goal))
    ]
    if not components:
        return None
    free = components[0]
    router = jps.RoutingEngine(free)
    candidates = []
    for distance in (0.6, 1.2, 2.0):
        for i in range(16):
            angle = i * math.tau / 16
            first = (pos[0] + distance * math.cos(angle), pos[1] + distance * math.sin(angle))
            if not free.covers(Point(first)):
                continue
            connector = LineString([pos, first])
            if not geometry.covers(connector.buffer(RADIUS + 0.015)):
                continue
            if any(
                (
                    connector.distance(Point(a.position))
                    < min(2 * RADIUS + 0.02, math.dist(pos, a.position) - 0.002)
                    for a in others
                )
            ):
                continue
            route = [tuple(p) for p in router.compute_waypoints(first, goal)]
            points = [first]
            for p in route[1:]:
                if math.dist(points[-1], p) > 0.05:
                    points.append(p)
            if math.dist(points[-1], goal) < 0.05:
                points.pop()
            cost = sum((math.dist(a, b) for a, b in zip([pos] + points, points + [goal])))
            candidates.append((cost, points))
    if not candidates:
        return None
    _, points = min(candidates, key=lambda item: item[0])
    stages = [sim.add_waypoint_stage(p, 0.08) for p in points] + [goal_stage]
    journey = jps.JourneyDescription(stages)
    for a, b in zip(stages, stages[1:]):
        journey.set_transition_for_stage(a, jps.Transition.create_fixed_transition(b))
    sim.switch_agent_journey(agent_id, sim.add_journey(journey), stages[0])
    return points


def simulate(source, config, request, decision_provider=None):
    wall_started = time.perf_counter()
    base = initialize_station(source, config, request)
    fire = FireField(config) if request.fire_mode == "fds" else None
    c = base["scenario"]
    c["id"] = config["id"] + "-jps"
    c["movement"] = {
        "engine": "jupedsim",
        "version": jps.__version__,
        "model": "AnticipationVelocityModel",
        "overtaking": "urgent_fast_clear_lateral_v2",
        "dt_s": DT,
        "seed": SEED,
        "radius_m": RADIUS,
        "decision_interval_s": DECISION_INTERVAL,
        "behavior_mode": request.behavior_mode,
        "hurry_multiplier": 1.35,
        "max_speed_mps": 2.2,
        "wait_duration_s": 2,
    }
    c["movement"]["minimum_navigation_gap_m"] = 2 * (RADIUS + 0.02)
    fast_threshold = source.get("provenance", {}).get("speed_fast_threshold_mps", 1.3)
    c["movement"]["overtaking_policy"] = {
        "fast_threshold_mps": fast_threshold,
        "threshold_basis": source.get("provenance", {}).get(
            "speed_fast_threshold_basis", "legacy experimental default"
        ),
        "minimum_speed_difference_mps": 0.3,
        "maximum_front_speed_ratio": 0.7,
        "disposition_required": "urgent",
        "front_speed_window_s": 1.0,
    }
    c["assumptions"] = [
        s for s in c["assumptions"] if not ("중심선" in s or "최단 예상" in s or "FDS 연기장" in s)
    ] + [
        "층별 이동은 JuPedSim AnticipationVelocityModel로 회피·감속·추월을 계산합니다. 실제 개인 행동의 검증 결과는 아닙니다.",
        "의도적 추월은 급한 성격의 직접 근거가 있고 현재 목표속도가 빠름 기준보다 높을 때만 시도합니다. 앞사람의 최근 1초 속도가 본인 목표의 70% 이하이고 0.30m/s 이상 느리며 측면 공간이 있어야 합니다. 느긋함·성격 정보 부족은 시도하지 않습니다. 비율·차이는 실험용 정책값입니다.",
        "계단은 FIFO 시간·용량 연결입니다. 계단 내부 추월은 없고 도착 층이 막히면 계단에서 기다립니다.",
        "Qwen 모드에서는 10초마다 이동 중인 사람의 상황을 판단합니다. 서두르기는 목표속도 1.35배(최대 2.2m/s), 기다리기는 2초이며 모두 실험용 정책값입니다.",
        "우회는 현재 공간에서 접근 가능한 다른 계단으로 변경합니다. 분리된 승강장 사이로 건너뛰지 않습니다.",
        "주변 인원은 3m 안에서 집계합니다. 출구 대기열과 경로의 현재 화재 관측은 안내 시스템이 공유한다고 가정합니다.",
        "도면 근사 과정의 좁은 틈은 몸 폭 0.44m와 여유 0.04m를 기준으로 보행 영역에서 닫습니다. FDS의 공기 공간은 별도로 유지합니다.",
    ]
    c["assumptions"] += (
        ["개화산역 형상으로 재계산한 FDS 연기·온도를 연결합니다. 화원·층고·환기는 가정입니다."]
        if fire
        else ["화재 끄기 실행입니다. 연기·온도 영향은 적용하지 않습니다."]
    )
    nodes = c["nodes"]
    stairs = {s["id"]: s for s in c["stairs"]}
    floors = {f["id"]: f for f in c["floors"]}
    from persona_evac.integration.station_layout import view_geometry, display_stair_path

    c["display_stair_paths"] = {s["id"]: display_stair_path(c, s) for s in c["stairs"]}
    c["display3d"] = {}
    for fid, f in floors.items():
        geom = floor_geometry(f)
        c["display3d"][fid] = [
            {
                "outer": list(poly.exterior.coords),
                "holes": [list(hole.coords) for hole in poly.interiors],
            }
            for poly in getattr(geom, "geoms", [geom])
            if not poly.is_empty
        ]
    c["display3d_openings"] = {f"{i / 10:.1f}": view_geometry(c, i / 10) for i in range(10, 41)}
    areas = {}
    area_specs = c.get(
        "areas",
        [
            {"id": "A", "floor": "B2", "region": 0},
            {"id": "B", "floor": "B2", "region": 1},
            {"id": "hall", "floor": "B1"},
        ],
    )
    fire_settings = fire.meta["fire"] if fire else None
    for spec in area_specs:
        name = spec["id"]
        geom = floor_geometry(floors[spec["floor"]], spec.get("region"))
        if fire and fire_settings["floor"] == spec["floor"]:
            x, y, w, h = fire_settings["rect"]
            geom = geom.difference(box(x, y, x + w, y + h))
        geom = pedestrian_geometry(geom)
        sim = make_simulation(geom)
        targets = [sid for sid, s in stairs.items() if nodes[s["from_node"]].get("area") == name]
        stages, lanes = ({}, {})
        for sid in targets:
            center = point(nodes[stairs[sid]["from_node"]])
            s = stairs[sid]
            n = s["parallel_lanes"]
            tangent = s.get("portal_tangent", [0, 1])
            lanes[sid] = []
            s["entry_lane_numbers"] = []
            for i in range(n):
                offset = (i - (n - 1) / 2) * s["width_m"] / n
                pos = tuple((center[j] + offset * tangent[j] for j in (0, 1)))
                if not geom.buffer(-RADIUS - 0.02).covers(Point(pos)):
                    continue
                stage = sim.add_waypoint_stage(pos, 0.35)
                journey = sim.add_journey(jps.JourneyDescription([stage]))
                lanes[sid].append((journey, stage, pos))
                s["entry_lane_numbers"].append(i)
            if not lanes[sid]:
                s["active_entry_lanes"] = 0
                continue
                raise ValueError(f"{sid}: 계단 진입점이 보행 공간 안에 없습니다")
            stages[sid] = lanes[sid][0][:2]
            s["active_entry_lanes"] = len(lanes[sid])
        areas[name] = {"sim": sim, "geometry": geom, "stages": stages, "lanes": lanes}
    agents = deepcopy(base["agents"])
    c["population"] = {
        "count": len(agents),
        "source_profiles": len(agents),
        "available_profiles": len(source["variants"]["C"]),
        "method": "고정 시드로 선정한 서로 다른 Nemotron 페르소나를 인원수만큼 일대일 배치",
    }
    runtime = {}
    passages, decisions, overtakes, pass_plans, yield_plans = ([], [], [], [], [])
    lane_changes = []
    rank_changes, pending_passes, hazard_history = ([], [], [])
    last_entry = {sid: -1000000.0 for sid in stairs}
    lane_entry = {}
    stair_gap = 1.0

    def landing_for(s, lane):
        dest = dict(nodes[s["to_node"]])
        n = s["parallel_lanes"]
        tangent = s.get("portal_tangent", [0, 1])
        offset = (lane - (n - 1) / 2) * s["width_m"] / n
        dest["x"] += offset * tangent[0]
        dest["y"] += offset * tangent[1]
        return dest

    for sid, s in stairs.items():
        area = areas[nodes[s["from_node"]]["area"]]
        valid = []
        numbers = []
        for option, lane in zip(area["lanes"][sid], s["entry_lane_numbers"]):
            dest = landing_for(s, lane)
            if dest["floor"] != "OUTSIDE" and (
                not areas[dest.get("area", "hall")]["geometry"]
                .buffer(-RADIUS - 0.02)
                .covers(Point(point(dest)))
            ):
                continue
            valid.append(option)
            numbers.append(lane)
        s["physical_valid_lanes"] = numbers
        s["active_entry_lanes"] = len(valid)
    min_gap = math.inf
    overlap_steps = 0
    pair_history = {}
    provider_error = None
    from persona_evac.integration.private_agents import PrivateController
    from persona_evac.agent_policy import POLICY as PRIVATE_POLICY, VERSIONS, POLICY_HASH

    cache = Path(
        os.environ.get("PERSONA_EVAC_DECISION_CACHE")
        or Path(os.environ.get("PERSONA_EVAC_OUTPUT", "output"))
        / "gaehwasan"
        / "decision_cache"
        / "private_agents_final_v2"
    )
    private = PrivateController(
        agents,
        areas,
        stairs,
        nodes,
        runtime,
        fire,
        request.behavior_mode,
        cache,
        decision_provider,
        request.experiment_seed,
        c,
        profile_method=request.profile_method,
    )
    decision_provider = private.provider
    minds = private.minds
    c["movement"].update(
        decision_interval_s=None,
        decision_trigger="private_events",
        policy_versions=private.policy_versions,
        overtaking="experimental_traits_visible_front_v1",
    )
    c["movement"]["overtaking_policy"].update(
        disposition_required=None,
        experimental_intent_gate="waiting_patience=low and exploration=normal/high",
        original_policy="원문 급함 판정은 보존하며 새 실험 성향 정책의 의도 필수 조건에서는 제외",
    )
    c["assumptions"] = [
        s
        for s in c["assumptions"]
        if not any(
            (
                x in s
                for x in (
                    "10초마다",
                    "공유",
                    "주변 인원",
                    "급한 성격",
                    "계단별 예상 점유율",
                    "경로 안내는",
                )
            )
        )
    ]
    classification_assumptions=list(PRIVATE_POLICY['assumptions'])
    if private.profiles_provider.method=='full_persona':
        classification_assumptions[1]='페르소나의 전체 원문 필드를 LLM이 해석하며 직접 서술·문맥 추정·정보 부족 가설을 구분합니다. 정보 부족도 LLM이 등급을 추정하며 분포·시드로 대체하지 않습니다.'
    c["assumptions"] += classification_assumptions
    now = 0.0

    def add_person(r, area_name, pos, target):
        area = areas[area_name]
        if not area["geometry"].buffer(-RADIUS - 0.02).covers(Point(pos)):
            raise ValueError("시작점 또는 계단 도착점이 보행 가능한 공간 밖에 있습니다.")
        if any((math.dist(pos, a.position) < 2 * RADIUS + 0.08 for a in area["sim"].agents())):
            return False
        goal = pos
        stage = area["sim"].add_waypoint_stage(pos, 0.2)
        journey = area["sim"].add_journey(jps.JourneyDescription([stage]))
        jid = area["sim"].add_agent(
            parameters(pos, journey, stage, r["agent"]["desired_speed_mps"])
        )
        r.update(area=area_name, jid=jid, target=target, pos=pos, goal=goal, goal_stage=stage)
        r["motion_samples"] = deque([(now, pos)], maxlen=21)
        r["observed_speed_mps"] = 0
        r["entry_lane"] = None
        r["goal_is_portal"] = False
        return True

    for i, a in enumerate(agents):
        a.update(segments=[], stair_wait_s=0.0, evacuation_time_s=None, exit_id=None)
        a.update(distance_walked_m=0.0, decision_wait_s=0.0)
        a["experimental_profile"] = minds[a["agent_id"]].profile
        area = a["spawn"].get("area", "A" if i % 2 == 0 else "B")
        r = {
            "agent": a,
            "phase": "pre_moving",
            "pos": point(a["spawn"]),
            "floor": "B2",
            "action": "continue",
            "wait_until": 0,
            "wait_streak": 0,
            "reroute_after": 0,
            "decision_pos": point(a["spawn"]),
            "decision_time": 0,
            "passing": False,
        }
        r.update(
            stall_origin=r["pos"],
            stall_since=0,
            yield_after=0,
            lane_change_after=0,
            overtake_intent_until=-1,
            pace="continue",
        )
        initial = "explore:none"
        if not add_person(r, area, r["pos"], initial):
            raise ValueError("시작 위치에서 사람이 겹칩니다. 서로 다른 시작 위치가 필요합니다.")
        runtime[a["agent_id"]] = r

    def private_overtake_probe(r, now):
        intent_speed = min(
            2.2, r["agent"]["desired_speed_mps"] * (1.35 if r.get("pace") == "hurry" else 1)
        )
        if not can_intentionally_overtake(r["agent"], intent_speed, fast_threshold):
            return None
        sim, geom = private.visible_helper(r, now)
        mind = private.minds[r["agent"]["agent_id"]]
        speeds = {
            o["jid"]: mind.space.people_track.get(aid, {}).get("estimated_speed_mps")
            if mind.space.people_track.get(aid, {}).get("estimated_speed_mps") is not None
            else math.inf
            for aid, o in runtime.items()
            if o["area"] == r["area"]
        }
        return plan_overtake(
            sim,
            geom,
            r["jid"],
            r["goal"],
            r["goal_stage"],
            observed_speeds=speeds,
            apply=False,
            intent_speed=intent_speed,
        )

    private.overtake_probe = private_overtake_probe

    def location(r):
        if r["phase"] == "on_stairs":
            p = r["passage"]
            fraction = min(1, max(0, (now - p["entry"]) / stairs[r["target"]]["travel_time_s"]))
            end = nodes[stairs[r["target"]]["to_node"]]
            start = r["stair_start"]
            return {
                "floor": start["floor"],
                "x": start["x"] + (end["x"] - start["x"]) * fraction,
                "y": start["y"] + (end["y"] - start["y"]) * fraction,
            }
        return {"floor": r["floor"], "x": r["pos"][0], "y": r["pos"][1]}

    def log_segment(r, start, end, a, b, status, sid=None):
        a, b = ({key: p[key] for key in ("floor", "x", "y")} for p in (a, b))
        segments = r["agent"]["segments"]
        item = {"start": start, "end": end, "from": a, "to": b, "status": status, "stair_id": sid}
        if (
            segments
            and a == b
            and (segments[-1]["from"] == a)
            and (segments[-1]["to"] == b)
            and (segments[-1]["status"] == status)
        ):
            segments[-1]["end"] = end
        else:
            segments.append(item)

    batches = 0
    now = 0.0
    max_time = request.duration_s
    if fire:
        max_time = min(max_time, float(fire.times[-1]) - fire.offset)
    for tick in range(int(max_time / DT)):
        now = round(tick * DT, 8)
        for sid, s in stairs.items():
            occupants = sorted(
                [r for r in runtime.values() if r["phase"] == "on_stairs" and r["target"] == sid],
                key=lambda r: -r["stair_distance"],
            )
            stair_speed = s["speed_mps"] * stair_speed_factor(s, len(occupants))
            fronts = {}
            for r in occupants:
                lane = r["stair_lane"]
                limit = fronts.get(lane, s["length_m"] + stair_gap) - stair_gap
                r["stair_distance"] = min(limit, r["stair_distance"] + stair_speed * DT)
                fronts[lane] = r["stair_distance"]
                r["passage"]["motion"].append([now, r["stair_distance"] / s["length_m"]])
        if fire:
            fire.indices(now)
        for r in runtime.values():
            if (
                r["phase"] != "on_stairs"
                or r["stair_distance"] < stairs[r["target"]]["length_m"] - 1e-08
            ):
                continue
            s = stairs[r["target"]]
            dest = landing_for(s, r["stair_lane"])
            if dest["floor"] == "OUTSIDE":
                r.update(phase="evacuated", floor="OUTSIDE", pos=point(dest))
                r["agent"].update(evacuation_time_s=now, exit_id=s["to_node"])
            else:
                dest_area = dest.get("area", "hall")
                target = "explore:none"
                if not add_person(r, dest_area, point(dest), target):
                    continue
                r.update(
                    phase="walking",
                    floor=dest["floor"],
                    action="continue",
                    passing=False,
                    pace="continue",
                )
            r["passage"]["exit"] = now
            r["passage"]["landing_wait_s"] = max(0, now - r["ready_at"])
            log_segment(
                r, r["passage"]["entry"], now, r["stair_start"], dict(dest), "on_stairs", s["id"]
            )
            r["agent"]["segments"][-1]["stair_motion"] = r["passage"]["motion"]
            r["agent"]["segments"][-1]["stair_lane"] = r["stair_lane"]
            if r["phase"] != "evacuated":
                private.transferred(r, dest_area, dest["floor"], point(dest), now, s["id"])
            else:
                private.minds[r["agent"]["agent_id"]].space.traversed(
                    s["id"], r["area"], "OUTSIDE", point(dest), now
                )
        if all((r["phase"] == "evacuated" for r in runtime.values())):
            break
        for r in runtime.values():
            if r["phase"] == "pre_moving" and now >= r["agent"]["pre_movement_s"]:
                r["phase"] = "walking"
            if r["phase"] == "waiting" and now >= r["wait_until"]:
                r.update(phase="walking", action="continue")
        private.tick(now)
        if tick % round(20 / DT) == 0:
            print(
                json.dumps(
                    {
                        "time_s": now,
                        "private_phases": dict(Counter((r["phase"] for r in runtime.values()))),
                        "navigation_blocked": dict(
                            Counter(
                                (
                                    r.get("navigation_blocked")
                                    for r in runtime.values()
                                    if r.get("navigation_blocked")
                                )
                            )
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        for r in runtime.values():
            if (
                r["phase"] == "walking"
                and r.get("goal_is_portal", True)
                and (math.dist(r["pos"], r["goal"]) < 0.6)
            ):
                r.update(phase="queueing", arrived_at=now)
        for sid, s in stairs.items():
            waiting = sorted(
                [r for r in runtime.values() if r["phase"] == "queueing" and r["target"] == sid],
                key=lambda r: (r["arrived_at"], r["agent"]["agent_id"]),
            )
            occupied = sum(
                (r["phase"] == "on_stairs" and r["target"] == sid for r in runtime.values())
            )
            if (
                waiting
                and occupied < s["capacity"]
                and (now + 1e-08 >= last_entry[sid] + 1 / s["entry_rate_per_s"])
            ):
                free = []
                for lane in range(s["parallel_lanes"]):
                    dest = landing_for(s, lane)
                    if dest["floor"] != "OUTSIDE" and (
                        not areas[dest.get("area", "hall")]["geometry"]
                        .buffer(-RADIUS - 0.02)
                        .covers(Point(point(dest)))
                    ):
                        continue
                    if now - lane_entry.get((sid, lane), -1000000.0) < stair_gap / s["speed_mps"]:
                        continue
                    if any(
                        (
                            o["phase"] == "on_stairs"
                            and o["target"] == sid
                            and (o["stair_lane"] == lane)
                            and (o["stair_distance"] < stair_gap)
                            for o in runtime.values()
                        )
                    ):
                        continue
                    free.append(lane)
                if not free:
                    continue
                ready = next((r for r in waiting if r["entry_lane"] in free), None)
                if ready is None:
                    continue
                r = ready
                lane = r["entry_lane"]
                r.update(stair_lane=lane, stair_distance=0)
                r["stair_destination"] = landing_for(s, lane)
                lane_entry[sid, lane] = now
                r["stair_start"] = location(r)
                areas[r["area"]]["sim"].mark_agent_for_removal(r["jid"])
                r.update(phase="on_stairs", ready_at=now + s["travel_time_s"])
                p = {
                    "agent_id": r["agent"]["agent_id"],
                    "stair_id": sid,
                    "arrival": r["arrived_at"],
                    "entry": now,
                    "exit": None,
                    "lane": lane,
                    "minimum_spacing_m": stair_gap,
                    "motion": [[now, 0.0]],
                }
                passages.append(p)
                r["passage"] = p
                r["agent"]["stair_wait_s"] += now - r["arrived_at"]
                last_entry[sid] = now
        before = {}
        for aid, r in runtime.items():
            if r["phase"] in {"on_stairs", "evacuated"}:
                continue
            before[aid] = location(r)
            actor = areas[r["area"]]["sim"].agent(r["jid"])
            if r["phase"] == "waiting":
                r["agent"]["decision_wait_s"] += DT
            speed = r["agent"]["desired_speed_mps"]
            if r["action"] == "hurry" or r.get("pace") == "hurry":
                speed = min(2.2, speed * 1.35)
            if fire:
                hazard = fire.sample(r["floor"], r["pos"], now)
                speed *= hazard["speed_factor"]
                if tick % 20 == 0:
                    hazard_history.append(
                        {
                            "agent_id": aid,
                            "time_s": now,
                            "floor": r["floor"],
                            "x": r["pos"][0],
                            "y": r["pos"][1],
                            **hazard,
                        }
                    )
            if r["phase"] != "walking":
                speed = 0
            if r["phase"] == "walking" and (
                r.get("navigation_blocked")
                or (
                    not r.get("passing")
                    and (
                        not private.sensor.visible(r["area"], r["floor"], r["pos"], r["goal"], now)
                    )
                )
            ):
                speed = 0
            actor.model.desired_speed = speed
        if tick % 10 == 0:
            for aid in before:
                r = runtime[aid]
                if r["phase"] != "walking":
                    continue
                area = areas[r["area"]]
                goal_stage = r["goal_stage"]
                if r["passing"] and area["sim"].agent(r["jid"]).stage_id == goal_stage:
                    r["passing"] = False
                if math.dist(r["pos"], r["stall_origin"]) > 0.15:
                    r.update(stall_origin=r["pos"], stall_since=now)
                recovery_busy = False
                if not recovery_busy and now - r["stall_since"] > 2.5 and (now >= r["yield_after"]):
                    helper_sim, helper_geom = private.visible_helper(r, now)
                    points = None
                    if helper_geom.covers(Point(r["pos"])) and helper_geom.covers(Point(r["goal"])):
                        try:
                            points = plan_yield(
                                helper_sim,
                                helper_geom,
                                r["jid"],
                                r["goal"],
                                goal_stage,
                                {
                                    o["jid"]
                                    for o in runtime.values()
                                    if o["area"] == r["area"]
                                    and o["phase"] in {"on_stairs", "evacuated"}
                                },
                            )
                        except RuntimeError:
                            pass
                    if points:
                        r.update(
                            passing=True,
                            yield_after=now + 4,
                            stall_origin=r["pos"],
                            stall_since=now,
                        )
                        yield_plans.append(
                            {
                                "agent_id": aid,
                                "time_s": now,
                                "floor": r["floor"],
                                "waypoints": points,
                                "observation_id": r.get("local_motion_observation_id"),
                            }
                        )
                if (
                    not r["passing"]
                    and now <= r["overtake_intent_until"]
                    and can_intentionally_overtake(
                        r["agent"], area["sim"].agent(r["jid"]).model.desired_speed, fast_threshold
                    )
                ):
                    helper_sim, helper_geom = private.visible_helper(r, now)
                    observed = {
                        o["jid"]: o["observed_speed_mps"]
                        for o in runtime.values()
                        if o["area"] == r["area"]
                        and o["phase"] not in {"on_stairs", "evacuated"}
                        and (len(o["motion_samples"]) >= 11)
                    }
                    plan = plan_overtake(
                        helper_sim,
                        helper_geom,
                        r["jid"],
                        r["goal"],
                        goal_stage,
                        {
                            o["jid"]
                            for o in runtime.values()
                            if o["area"] == r["area"] and o["phase"] in {"on_stairs", "evacuated"}
                        },
                        observed,
                    )
                    if plan:
                        r["passing"] = True
                        pass_plans.append(
                            {"agent_id": aid, "time_s": now, "floor": r["floor"], **plan}
                        )
                        front_id = next(
                            (
                                oid
                                for oid, o in runtime.items()
                                if o["area"] == r["area"]
                                and o["phase"] not in {"on_stairs", "evacuated"}
                                and (o["jid"] == plan["passed_jps_id"])
                            )
                        )
                        goal = r["goal"]
                        distance = math.dist(r["pos"], goal)
                        pending_passes.append(
                            {
                                "passing_agent": aid,
                                "passed_agent": front_id,
                                "start_s": now,
                                "area": r["area"],
                                "floor": r["floor"],
                                "target_stair": r["target"],
                                "axis": [(goal[i] - r["pos"][i]) / distance for i in (0, 1)],
                                "min_surface_gap_m": 1000000.0,
                                "crossing_lateral_m": None,
                            }
                        )
        for area in areas.values():
            area["sim"].iterate()
        for aid, old in before.items():
            r = runtime[aid]
            r["pos"] = tuple(areas[r["area"]]["sim"].agent(r["jid"]).position)
            r["agent"]["distance_walked_m"] += math.dist((old["x"], old["y"]), r["pos"])
            samples = r["motion_samples"]
            samples.append((now + DT, r["pos"]))
            span = samples[-1][0] - samples[0][0]
            r["observed_speed_mps"] = (
                sum((math.dist(a[1], b[1]) for a, b in zip(samples, list(samples)[1:]))) / span
                if span > 0
                else 0
            )
            log_segment(
                r,
                now,
                round(now + DT, 8),
                old,
                location(r),
                r["phase"],
                r["target"] if r["phase"] == "queueing" else None,
            )
        for event in pending_passes[:]:
            a, b = (runtime[event["passing_agent"]], runtime[event["passed_agent"]])
            if (
                now - event["start_s"] > 25
                or a["area"] != event["area"]
                or b["area"] != event["area"]
                or (a["phase"] != "walking")
                or (b["phase"] != "walking")
                or (a["target"] != event["target_stair"])
                or (b["target"] != event["target_stair"])
            ):
                pending_passes.remove(event)
                continue
            delta = [a["pos"][i] - b["pos"][i] for i in (0, 1)]
            longitudinal = sum((delta[i] * event["axis"][i] for i in (0, 1)))
            gap = math.dist(a["pos"], b["pos"]) - 2 * RADIUS
            event["min_surface_gap_m"] = min(event["min_surface_gap_m"], gap)
            if event["crossing_lateral_m"] is None and longitudinal >= 0:
                event["crossing_lateral_m"] = abs(
                    delta[0] * event["axis"][1] - delta[1] * event["axis"][0]
                )
            if longitudinal >= 0.5:
                if (
                    event["min_surface_gap_m"] >= -0.01
                    and (event["crossing_lateral_m"] or 0) >= 2 * RADIUS
                ):
                    overtakes.append({**event, "time_s": round(now + DT, 8)})
                pending_passes.remove(event)
        active = [(aid, r) for aid, r in runtime.items() if aid in before]
        for index, (aid, a) in enumerate(active):
            if not areas[a["area"]]["geometry"].covers(Point(a["pos"])):
                raise RuntimeError("JuPedSim 에이전트가 보행 가능한 공간을 벗어났습니다.")
            for bid, b in active[index + 1 :]:
                if a["area"] != b["area"]:
                    continue
                gap = math.dist(a["pos"], b["pos"]) - 2 * RADIUS
                min_gap = min(min_gap, gap)
                overlap_steps += int(gap < -0.01)
                if a["phase"] != "walking" or b["phase"] != "walking" or a["target"] != b["target"]:
                    continue
                dest = a["goal"]
                diff = math.dist(a["pos"], dest) - math.dist(b["pos"], dest)
                key = (aid, bid, a["area"], a["target"])
                order = 1 if diff > 0.5 else -1 if diff < -0.5 else 0
                old_order = pair_history.get(key)
                if order and old_order and (order != old_order):
                    rank_changes.append(
                        {
                            "time_s": round(now + DT, 8),
                            "floor": a["floor"],
                            "passing_agent": aid if order < 0 else bid,
                            "passed_agent": bid if order < 0 else aid,
                            "target_stair": a["target"],
                        }
                    )
                if order:
                    pair_history[key] = order
    else:
        now = max_time
        for r in runtime.values():
            if r["phase"] == "on_stairs":
                sid = r["target"]
                log_segment(
                    r,
                    r["passage"]["entry"],
                    now,
                    r["stair_start"],
                    nodes[stairs[sid]["to_node"]],
                    "on_stairs",
                    sid,
                )
                r["agent"]["segments"][-1].update(
                    stair_motion=r["passage"]["motion"], stair_lane=r["stair_lane"]
                )
    unique_passes = {}
    for event in overtakes:
        key = (event["passing_agent"], event["passed_agent"], event["area"], event["target_stair"])
        unique_passes.setdefault(key, event)
    overtakes = list(unique_passes.values())
    private.finalize(now)
    decisions = private.decisions
    batches = private.decision_batches
    if fire:
        hazard_history.extend(fire.stair_observations(c, agents))
        hazard_history.sort(key=lambda h: (h["time_s"], h["agent_id"]))
    stats = []
    for sid in stairs:
        ps = [p for p in passages if p["stair_id"] == sid]

        def peak(events):
            total = highest = 0
            for _, change in sorted(events):
                total += change
                highest = max(highest, total)
            return highest

        stats.append(
            {
                "id": sid,
                "count": len(ps),
                "peak_occupancy": peak(
                    [(p["entry"], 1) for p in ps] + [(p["exit"] or now, -1) for p in ps]
                ),
                "peak_queue": peak(
                    [(p["arrival"], 1) for p in ps if p["entry"] > p["arrival"]]
                    + [(p["entry"], -1) for p in ps if p["entry"] > p["arrival"]]
                ),
                "total_wait_s": sum((p["entry"] - p["arrival"] for p in ps)),
                "landing_wait_s": sum((p.get("landing_wait_s", 0) for p in ps)),
            }
        )
    canonical = json.dumps(c, sort_keys=True, ensure_ascii=False)
    base.update(
        schema_version="station-1",
        backend="jupedsim_avm_layered",
        backend_notice=c["name"]
        + " · JuPedSim 회피·추월 + 계단 시간·용량 연결 · "
        + ("FDS 연기·온도 연결" if fire else "화재 끄기 실행"),
        scenario_hash=hashlib.sha256(canonical.encode()).hexdigest(),
        agents=agents,
        passages=passages,
        stairs_summary=stats,
        decisions=decisions,
        fire=fire.replay(
            config,
            max(now, fire_settings["ramp_full_s"] - fire.offset + 30) if c.get("train") else now,
        )
        if fire
        else {"enabled": False},
        hazard_history=hazard_history,
        behavior={
            "mode": request.behavior_mode,
            "model": getattr(decision_provider, "model", None),
            "policy": PRIVATE_POLICY if private.profiles_provider.method=='keyword_candidates' else {
                **PRIVATE_POLICY, 'versions':private.policy_versions, 'assumptions':classification_assumptions},
            "event_counts": dict(Counter((e for d in decisions for e in d.get("events", [])))),
            "profile_timings": private.profiles_provider.records,
            "provider": getattr(
                decision_provider,
                "provider_name",
                None,
            ),
            "timings": getattr(decision_provider, "records", []),
            "batch_count": batches,
            "last_error": provider_error,
            "fallback_count": sum((d["source"] == "fallback" for d in decisions)),
        },
        movement_diagnostics={
            "min_surface_gap_m": min_gap if math.isfinite(min_gap) else None,
            "stair_lane_changes": lane_changes,
            "counterflow_recoveries": yield_plans,
            "overlap_pair_steps": overlap_steps,
            "overtakes": overtakes,
            "overtake_plans": pass_plans,
            "distance_rank_changes": rank_changes,
            "overtake_definition": "추월 계획의 동일 앞사람을 고정 진행축에서 실제로 지나 0.5m 앞에 도달, 교차 시 측면 거리≥0.44m·몸 겹침 없음. 동일 층·목표·방향의 인물 쌍은 1회 집계",
        },
        summary={
            "count": len(agents),
            "all_evacuated": all((a["evacuation_time_s"] is not None for a in agents)),
            "completed_count": sum((a["evacuation_time_s"] is not None for a in agents)),
            "total_time_s": now,
            "mean_wait_s": sum((a["stair_wait_s"] for a in agents)) / len(agents),
            "exit_split": {key: sum((a["exit_id"] == key for a in agents)) for key in c["exits"]},
        },
    )
    base["performance"] = {
        "wall_time_s": time.perf_counter() - wall_started,
        "profile_preparation_wall_s": private.profiles_provider.wall_s,
        "decision_workers": private.decision_workers,
        "decision_prompt_batch_size": getattr(decision_provider,"prompt_batch_size",None),
        "decision_batch_wall_s": private.decision_wall_s,
        "profile_inference_calls": getattr(decision_provider, "profile_inference_calls", 0),
        "llm_call_count": len(getattr(getattr(decision_provider, "local", None), "calls", [])),
        "llm_wall_s": sum(
            (m["wall_s"] for m in getattr(getattr(decision_provider, "local", None), "calls", []))
        ),
    }
    inference_local = getattr(decision_provider, "local", None) or private.profiles_provider.local
    attempts = getattr(inference_local, "attempts", [])
    base["performance"].update(
        llm_call_count=len(attempts),
        llm_attempt_count=len(attempts),
        profile_inference_calls=private.profile_inference_calls,
        llm_failed_attempt_count=sum((a["status"] == "failed" for a in attempts)),
        llm_wall_s=sum((a["wall_s"] for a in attempts)),
        cache_request_count=sum(
            (t["source"] == "ollama_cache" for t in getattr(decision_provider, "records", []))
        ),
    )
    base["behavior"]["inference_attempts"] = attempts
    base["behavior"]["inference_settings"] = getattr(
        getattr(decision_provider, "local", None), "settings", None
    )
    base["replay_duration_s"] = base["fire"]["frames"][-1]["time_s"] if fire else now
    if not base['summary']['all_evacuated']:
        base['replay_duration_s']=min(base['replay_duration_s'],now)
    from persona_evac.integration.personal_space import PRIOR_HASH

    base["agent_policy"] = {
        "versions": {**VERSIONS, "profile": private.profiles_provider.version},
        "policy_hash": POLICY_HASH,
        "classification_method": private.profiles_provider.method,
        "profile_version": private.profiles_provider.version,
        "profile_policy_hash": next(iter(private.profiles.values()))['policy_hash'],
        "experiment_seed": request.experiment_seed,
        "prior_hash": PRIOR_HASH,
        "observation_policy": PRIVATE_POLICY["sensor"],
        "assumptions": classification_assumptions,
    }
    base["navigation_updates"] = private.navigation_updates
    base["termination"] = {
        "requested_duration_s": request.duration_s,
        "available_duration_s": max_time,
        "reason": "all_evacuated"
        if base["summary"]["all_evacuated"]
        else "fds_coverage_limit"
        if fire and max_time < request.duration_s
        else "time_limit",
    }
    base["noncompletion"] = [
        {"agent_id": a["agent_id"], "cause": a["noncompletion_cause"]}
        for a in agents
        if a["evacuation_time_s"] is None
    ]
    return base
