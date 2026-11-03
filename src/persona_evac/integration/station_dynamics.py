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


def connector_occupants(runtime, connectors):
    """Snapshot membership before movement; retain runtime order for distance ties."""
    occupants = {sid: [] for sid in connectors}
    for r in runtime.values():
        if r["phase"] == "on_stairs" and r["target"] in occupants:
            occupants[r["target"]].append(r)
    for group in occupants.values():
        group.sort(key=lambda r: -r["stair_distance"])
    return occupants


def connector_admissions(runtime, connectors):
    """Fresh counts after departures, decisions and new queue arrivals."""
    waiting = {sid: [] for sid in connectors}
    occupied = {sid: 0 for sid in connectors}
    for r in runtime.values():
        if r["phase"] == "queueing" and r["target"] in waiting:
            waiting[r["target"]].append(r)
        elif r["phase"] == "on_stairs" and r["target"] in occupied:
            occupied[r["target"]] += 1
    for group in waiting.values():
        group.sort(key=lambda r: (r["arrived_at"], r["agent"]["agent_id"]))
    return waiting, occupied


def diagnose_people(runtime,before,areas,now,min_gap,overlap_steps,pair_history,rank_changes):
    """원래 사람·쌍 순서를 유지하며 같은 영역의 진단 대상만 순회한다."""
    active=[(aid,r) for aid,r in runtime.items() if aid in before]
    groups={};offsets={}
    for aid,r in active:groups.setdefault(r['area'],[]).append((aid,r))
    for aid,a in active:
        area=a['area'];index=offsets.get(area,0);offsets[area]=index+1
        if not areas[area]['geometry'].covers(Point(a['pos'])):
            raise RuntimeError('JuPedSim 에이전트가 보행 가능한 공간을 벗어났습니다.')
        own_distance=None
        for bid,b in groups[area][index+1:]:
            gap=math.dist(a['pos'],b['pos'])-2*RADIUS
            min_gap=min(min_gap,gap)
            overlap_steps+=int(gap<-.01)
            if a['phase']!='walking' or b['phase']!='walking' or a['target']!=b['target']:
                continue
            dest=a['goal']
            if own_distance is None:own_distance=math.dist(a['pos'],dest)
            diff=own_distance-math.dist(b['pos'],dest)
            key=(aid,bid,area,a['target'])
            order=1 if diff>.5 else -1 if diff<-.5 else 0
            old_order=pair_history.get(key)
            if order and old_order and order!=old_order:
                rank_changes.append({'time_s':round(now+DT,8),'floor':a['floor'],
                                     'passing_agent':aid if order<0 else bid,
                                     'passed_agent':bid if order<0 else aid,'target_stair':a['target']})
            if order:pair_history[key]=order
    return min_gap,overlap_steps


def append_motion_sample(r,now,pos):
    """기존 합산 순서는 유지하고 이미 구한 구간 거리만 재사용한다."""
    samples=r['motion_samples']
    if 'motion_distances' not in r:
        r['motion_distances']=deque((math.dist(a[1],b[1]) for a,b in zip(samples,list(samples)[1:])),maxlen=samples.maxlen-1)
    distances=r['motion_distances']
    if samples:distances.append(math.dist(samples[-1][1],pos))
    samples.append((now,pos))
    recovery=r.setdefault('recovery_samples',deque(maxlen=61))
    recovery.append((now,tuple(pos)))
    span=samples[-1][0]-samples[0][0]
    return sum(distances)/span if span>0 else 0


def update_stall_state(r,now):
    """작은 공간의 왕복을 전진으로 취급하지 않는다. 자신의 이동만 읽는다."""
    samples=r.get('recovery_samples',())
    oscillating=False
    if len(samples)>1 and samples[-1][0]-samples[0][0]>=2.5:
        positions=[p for _,p in samples]
        walked=sum(math.dist(a,b) for a,b in zip(positions,positions[1:]))
        extent=max(math.dist(positions[0],p) for p in positions)
        oscillating=(walked>=.5 and extent<.35
                     and math.dist(positions[0],positions[-1])<walked*.2)
    if oscillating:
        r['stall_since']=min(r['stall_since'],samples[0][0])
    elif math.dist(r['pos'],r['stall_origin'])>.15:
        r.update(stall_origin=r['pos'],stall_since=now)
    return now-r['stall_since']


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




def plan_gate_approach_yield(sim,geometry,agent_id,goal,goal_stage,inactive_ids=()):
    """Resolve a visible, near-mouth merge by walking one body out of the way."""
    actor=sim.agent(agent_id);pos=tuple(actor.position)
    if math.dist(pos,goal)>.45:return None
    others=[a for a in sim.agents() if a.id!=agent_id and a.id not in inactive_ids]
    competitors=[a for a in others if math.dist(a.position,goal)<=.45 and math.dist(pos,a.position)<2*RADIUS+.14]
    if not competitors:return None
    visible=[actor,*others];participants=[actor,*competitors]
    # Prefer the farther person, but do not order a trapped body to back into
    # a wall or a third person. Every candidate uses visible positions only.
    for yielding in sorted(participants,key=lambda a:(-round(math.dist(a.position,goal),3),a.id)):
        origin=tuple(yielding.position);distance=math.dist(origin,goal)
        if distance<1e-7:continue
        blockers=[a for a in visible if a.id!=yielding.id]
        away=tuple((origin[i]-goal[i])/distance for i in (0,1));first=None
        for step in (.3,.45,.65,.9):
            for angle in (0,math.pi/12,-math.pi/12,math.pi/6,-math.pi/6,math.pi/4,-math.pi/4,math.pi/2,-math.pi/2):
                direction=(away[0]*math.cos(angle)-away[1]*math.sin(angle),away[0]*math.sin(angle)+away[1]*math.cos(angle))
                if any(math.dist(origin,a.position)<2*RADIUS+.08 and
                       sum(direction[i]*(origin[i]-a.position[i]) for i in (0,1))<1e-5 for a in blockers):continue
                candidate=tuple(origin[i]+step*direction[i] for i in (0,1));path=LineString([origin,candidate])
                if not geometry.covers(path.buffer(RADIUS+.005)):continue
                if any(path.distance(Point(a.position))<min(2*RADIUS-.004,math.dist(origin,a.position)-.002) for a in blockers):continue
                if any(math.dist(candidate,a.position)<2*RADIUS+.05 for a in blockers):continue
                first=candidate;break
            if first is not None:break
        if first is None:continue
        priority=min((a for a in participants if a.id!=yielding.id),key=lambda a:(math.dist(a.position,goal),a.id))
        decision={'priority_jps_id':priority.id,'yielding_jps_id':yielding.id,'waypoints':[]}
        if yielding.id!=agent_id:return decision
        stage=sim.add_waypoint_stage(first,.08);journey=jps.JourneyDescription([stage,goal_stage])
        journey.set_transition_for_stage(stage,jps.Transition.create_fixed_transition(goal_stage))
        sim.switch_agent_journey(agent_id,sim.add_journey(journey),stage)
        decision['waypoints']=[first]
        return decision
    return None


def plan_yield(sim, geometry, agent_id, goal, goal_stage, inactive_ids=(), blocker_ids=None):
    """Break counterflow standstills with clear walking waypoints, never teleport."""
    actor = sim.agent(agent_id)
    pos = tuple(actor.position)
    route = jps.RoutingEngine(geometry).compute_waypoints(pos, goal)
    target = goal if geometry.covers(LineString([pos,goal])) else next((p for p in route if math.dist(pos,p)>.2),goal)
    distance = math.dist(pos, target)
    if distance < 1e-7:
        return None
    direction = [(target[i] - pos[i]) / distance for i in (0, 1)]
    normal = [-direction[1], direction[0]]
    others = [a for a in sim.agents() if a.id != agent_id and a.id not in inactive_ids]
    # 주변 사람 중 실제로 진행 방향을 막는 몸체가 있을 때만 우회한다.
    blockers=[a for a in others if
        0 < sum((a.position[i]-pos[i])*direction[i] for i in (0,1)) < min(1.5,distance+RADIUS)
        and abs(sum((a.position[i]-pos[i])*normal[i] for i in (0,1))) < 2*RADIUS+.12
    ]
    if not blockers:
        return None
    if blocker_ids is not None:
        blocker_ids.extend(a.id for a in blockers)
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
    route=plan_cluster_detour(sim, geometry, agent_id, goal, goal_stage, inactive_ids)
    if route:return route
    route=plan_contact_detour(sim,geometry,agent_id,goal,goal_stage,others)
    if route:return route
    # A short smoke-limited view may not contain an entire overtaking path.
    # A visible lateral step releases contact; the next observation then
    # refreshes the local goal before continuing. Never step through a body.
    for offset in (.45,.3,.65):
        for side in (1,-1):
            first=tuple(pos[i]+side*offset*normal[i] for i in (0,1))
            path=LineString([pos,first])
            if not geometry.covers(path.buffer(RADIUS+.015)):continue
            if any(path.distance(Point(a.position))<2*RADIUS-.004 for a in others):continue
            stage=sim.add_waypoint_stage(first,.08)
            journey=jps.JourneyDescription([stage,goal_stage])
            journey.set_transition_for_stage(stage,jps.Transition.create_fixed_transition(goal_stage))
            sim.switch_agent_journey(agent_id,sim.add_journey(journey),stage)
            return [first]
    return None


def plan_wall_recovery(sim,geometry,agent_id,goal,goal_stage,inactive_ids=()):
    """보이는 벽 근처에서 한 걸음 물러난 뒤 원래 목표로 걷는다."""
    pos=tuple(sim.agent(agent_id).position)
    wall=geometry.boundary.distance(Point(pos))
    distance=math.dist(pos,goal)
    if wall>.65 or distance<.3:return None
    others=[a for a in sim.agents() if a.id!=agent_id and a.id not in inactive_ids]
    angle=math.atan2(goal[1]-pos[1],goal[0]-pos[0]);candidates=[]
    clearance=max(0,min(RADIUS+.015,wall-1e-5))
    for step in (.4,.65,.9,1.2):
        for degrees in (0,30,-30,60,-60,90,-90,120,-120,150,-150,180):
            bearing=angle+math.radians(degrees)
            first=(pos[0]+step*math.cos(bearing),pos[1]+step*math.sin(bearing))
            first_wall=geometry.boundary.distance(Point(first))
            if first_wall<max(RADIUS+.18,wall+.12):continue
            # The first movement must release wall contact, rather than merely
            # end farther away after passing through the wall-repulsion zone.
            previous_wall=wall;outward=True
            for offset in (.05,.1,.15,.2):
                fraction=min(step,offset)/step
                probe=tuple(pos[i]+fraction*(first[i]-pos[i]) for i in (0,1))
                probe_wall=geometry.boundary.distance(Point(probe))
                if probe_wall<previous_wall+.002:outward=False;break
                previous_wall=probe_wall
            if not outward:continue
            if not geometry.covers(LineString([pos,first]).buffer(clearance)):continue
            # The second leg needs room for wall avoidance, beyond body contact.
            if not geometry.covers(LineString([first,goal]).buffer(RADIUS+.08)):continue
            path=LineString([pos,first,goal])
            if any(path.distance(Point(a.position))<2*RADIUS+.015 for a in others):continue
            candidates.append((path.length+.1/max(.01,first_wall),first))
    if not candidates:return None
    _,first=min(candidates)
    stage=sim.add_waypoint_stage(first,.08)
    journey=jps.JourneyDescription([stage,goal_stage])
    journey.set_transition_for_stage(stage,jps.Transition.create_fixed_transition(goal_stage))
    sim.switch_agent_journey(agent_id,sim.add_journey(journey),stage)
    return [first]


def recover_wall_route(controller,r,now,inactive_ids=()):
    """벽 접촉으로 일반 경로가 거절된 경우에도 보이는 목표로 복구한다."""
    goal=r.get('blocked_motion_target') or r['goal']
    if not controller.sensor.visible(r['area'],r['floor'],r['pos'],goal,now):return None
    helper,view=controller.visible_helper(r,now)
    if not view.covers(Point(r['pos'])) or not view.covers(Point(goal)):return None
    stage=helper.add_waypoint_stage(goal,.08) if r.get('blocked_motion_target') else r['goal_stage']
    points=plan_wall_recovery(helper,view,r['jid'],goal,stage,inactive_ids)
    if points and not controller.sensor.visible(r['area'],r['floor'],r['pos'],points[0],now):
        journey=helper.add_journey(jps.JourneyDescription([r['goal_stage']]))
        helper.switch_agent_journey(r['jid'],journey,r['goal_stage'])
        return None
    if points:
        r.update(goal=tuple(goal),goal_stage=stage,goal_is_portal=False,navigation_blocked=None,
                 movement_block_reason=None,passing=True,manoeuvre_waypoints=helper.waypoint_positions)
        r.pop('blocked_motion_target',None)
        if 'intended_walk_speed' in r:
            controller.areas[r['area']]['sim'].agent(r['jid']).model.desired_speed=r['intended_walk_speed']
    return points


def plan_contact_detour(sim,geometry,agent_id,goal,goal_stage,others):
    """Use physical body clearance when a comfort margin closes a narrow gap."""
    from shapely.ops import nearest_points
    pos=tuple(sim.agent(agent_id).position)
    free=geometry.buffer(-RADIUS-1e-5).difference(unary_union([
        Point(a.position).buffer(2*RADIUS+.001,quad_segs=16) for a in others]))
    components=[p for p in getattr(free,'geoms',[free]) if p.geom_type=='Polygon' and p.covers(Point(goal))]
    for component in components:
        first=nearest_points(component,Point(pos))[0]
        first=tuple(first.coords[0])
        connector=LineString([pos,first])
        if not geometry.covers(connector.buffer(RADIUS)):continue
        if any(connector.distance(Point(a.position))<min(2*RADIUS-.004,math.dist(pos,a.position)-.002) for a in others):continue
        route=[tuple(p) for p in jps.RoutingEngine(component).compute_waypoints(first,goal)]
        points=[p for p in [first,*route[1:-1]] if math.dist(pos,p)>1e-6]
        if not points:continue
        remaining=[*points,tuple(goal)];points=[];origin=pos
        while remaining:
            chosen=None
            for index in range(len(remaining)-1,-1,-1):
                line=LineString([origin,remaining[index]])
                if (geometry.covers(line.buffer(RADIUS)) and
                        all(line.distance(Point(a.position))>=2*RADIUS-.004 for a in others)):
                    chosen=index;break
            if chosen is None:break
            origin=remaining[chosen];points.append(origin);remaining=remaining[chosen+1:]
        if remaining or not points:continue
        points=points[:-1]
        if not points:continue
        path=LineString([pos,*points,goal])
        if any(path.distance(Point(a.position))<2*RADIUS-.004 for a in others):continue
        stages=[sim.add_waypoint_stage(p,.12) for p in points]+[goal_stage]
        journey=jps.JourneyDescription(stages)
        for a,b in zip(stages,stages[1:]):journey.set_transition_for_stage(a,jps.Transition.create_fixed_transition(b))
        sim.switch_agent_journey(agent_id,sim.add_journey(journey),stages[0])
        return points
    return None


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


def visible_motion_target(sim, runtime, sensor, now):
    """국소 이동도 매 물리 시간 간격마다 실제 다음 경유점의 시야를 확인한다."""
    if runtime.get('passing'):
        actor=sim.agent(runtime['jid'])
        if actor.stage_id==runtime['goal_stage']:
            runtime['passing']=False
            runtime.pop('manoeuvre_waypoints',None)
            runtime.pop('manoeuvre_progress',None)
        else:
            target=runtime.get('manoeuvre_waypoints',{}).get(actor.stage_id)
            if target is not None and sensor.visible(runtime['area'],runtime['floor'],runtime['pos'],target,now):
                progress=runtime.get('manoeuvre_progress')
                remaining=math.dist(runtime['pos'],target)
                if progress is None or progress['stage_id']!=actor.stage_id:
                    progress={'stage_id':actor.stage_id,'best_distance':remaining,'last_progress_at':now}
                    runtime['manoeuvre_progress']=progress
                elif remaining<progress['best_distance']-.08:
                    progress.update(best_distance=remaining,last_progress_at=now)
                if now-progress['last_progress_at']>=2.5:
                    event=runtime.get('manoeuvre_event')
                    if event is not None:event['outcome']='stalled_replan'
                    finish_manoeuvre(runtime,now)
                    journey=sim.add_journey(jps.JourneyDescription([runtime['goal_stage']]))
                    sim.switch_agent_journey(runtime['jid'],journey,runtime['goal_stage'])
                    runtime.update(passing=False,navigation_after=now,yield_after=now,
                                   stall_since=min(runtime.get('stall_since',now),now-3))
                    runtime.pop('manoeuvre_waypoints',None);runtime.pop('manoeuvre_progress',None)
                    if runtime.get('agent') is not None:
                        runtime['agent'].setdefault('local_recovery_replans',[]).append(
                            {'time_s':now,'waypoint':list(target),'reason':'no_progress'})
                    return sensor.visible(runtime['area'],runtime['floor'],runtime['pos'],runtime['goal'],now)
                return True
            journey=sim.add_journey(jps.JourneyDescription([runtime['goal_stage']]))
            sim.switch_agent_journey(runtime['jid'],journey,runtime['goal_stage'])
            actor.model.desired_speed=0.
            runtime.update(passing=False,navigation_after=now,
                           navigation_blocked='local_manoeuvre_not_currently_visible',movement_block_reason='visibility')
            runtime.pop('manoeuvre_waypoints',None)
            runtime.pop('manoeuvre_progress',None)
            return False
    else:
        runtime.pop('manoeuvre_waypoints',None)
        runtime.pop('manoeuvre_progress',None)
    return sensor.visible(runtime['area'],runtime['floor'],runtime['pos'],runtime['goal'],now)


def record_visibility_block(agent,now,blocked):
    history=agent.setdefault('visibility_blocked_history',[])
    if not history or history[-1]['blocked']!=blocked:
        history.append({'time_s':now,'blocked':blocked})


def finish_manoeuvre(runtime,now):
    event=runtime.pop('manoeuvre_event',None)
    if event is not None:event['end_s']=now


def start_manoeuvre(runtime,event,action,now):
    """이동 계획과 별개로 양보·추월의 실제 실행 구간을 기록한다."""
    finish_manoeuvre(runtime,now)
    runtime.pop('manoeuvre_progress',None)
    event.update(action=action,end_s=None)
    runtime['manoeuvre_event']=event


def update_manoeuvre(runtime,now):
    if (not runtime.get('passing') or runtime['phase']!='walking'
            or runtime['agent'].get('incapacitated_at_s') is not None):
        finish_manoeuvre(runtime,now)


def pass_attempt(passer,front,now,source):
    if (passer['area']!=front['area'] or passer['phase']!='walking' or front['phase']!='walking'
            or passer['target']!=front['target']):return None
    distance=math.dist(passer['pos'],passer['goal'])
    if distance<=.01:return None
    axis=[(passer['goal'][i]-passer['pos'][i])/distance for i in (0,1)]
    delta=[front['pos'][i]-passer['pos'][i] for i in (0,1)]
    if sum(delta[i]*axis[i] for i in (0,1))<=.2:return None
    return {'passing_agent':passer['agent']['agent_id'],'passed_agent':front['agent']['agent_id'],
            'start_s':now,'area':passer['area'],'floor':passer['floor'],'target_stair':passer['target'],
            'axis':axis,'min_surface_gap_m':1000000.,'crossing_lateral_m':None,'source':source}


def advance_pass_attempt(event,passer,front,now):
    if (now-event['start_s']>25 or passer['area']!=event['area'] or front['area']!=event['area']
            or passer['phase']!='walking' or front['phase']!='walking'
            or passer['target']!=event['target_stair'] or front['target']!=event['target_stair']):return False
    delta=[passer['pos'][i]-front['pos'][i] for i in (0,1)]
    longitudinal=sum(delta[i]*event['axis'][i] for i in (0,1))
    event['min_surface_gap_m']=min(event['min_surface_gap_m'],math.dist(passer['pos'],front['pos'])-2*RADIUS)
    if event['crossing_lateral_m'] is None and longitudinal>=0:
        event['crossing_lateral_m']=abs(delta[0]*event['axis'][1]-delta[1]*event['axis'][0])
    if longitudinal>=.5:
        if event['min_surface_gap_m']>=-.01 and (event['crossing_lateral_m'] or 0)>=2*RADIUS:
            return {**event,'time_s':round(now+DT,8)}
        return False
    return None


def simulate(source, config, request, decision_provider=None):
    wall_started = time.perf_counter()
    base = initialize_station(source, config, request)
    fire = FireField(config) if request.fire_mode == "fds" else None
    from persona_evac.integration.gas_exposure import MODEL as GAS_MODEL, accumulate_exposure
    if fire:fire.require_gases()
    c = base["scenario"]
    c["id"] = config["id"] + "-jps"
    c["movement"] = {
        "engine": "jupedsim",
        "version": jps.__version__,
        "model": "AnticipationVelocityModel",
        "overtaking": "local_recovery_only_v1",
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
    c['movement']['gas_exposure']={'enabled':bool(fire),**GAS_MODEL}
    c["movement"]["overtaking_policy"] = {
        "intentional_overtaking_enabled": False,
        "local_recovery_enabled": True,
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
        ["개화산역 형상으로 재계산한 FDS 연기·온도·CO·CO₂·산소 농도를 연결합니다. 화원·층고·환기는 가정입니다.",
         "정지·반응 전 대기·계단 이동 중에도 가스 노출 FED를 누적하고 FED≥1.0이면 대피 불능으로 처리합니다. 활동량은 모든 승객에 light work를 적용하며 사망·열 상해·HCN은 계산하지 않습니다."]
        if fire
        else ["화재 끄기 실행입니다. 연기·온도 영향은 적용하지 않습니다."]
    )
    nodes = c["nodes"]
    stairs = {s["id"]: s for s in c["stairs"]}
    floors = {f["id"]: f for f in c["floors"]}
    native=bool(c.get('jupedsim_connections',{}).get('enabled'))
    network=None
    if native:
        from persona_evac.integration.station_connectors import compile_network,connector_landing,NativePassage
        network=compile_network(c);nodes=network['nodes'];stairs=network['connectors']
        c['nodes']=nodes
        c['movement'].update(stairs='jupedsim_unfolded_branches',fare_gates='exclusive_capacity_one',
                             stair_turns='engine_walked_landing_xy_with_slope_speed')
        c['connection_topology']=[{'id':sid,'kind':s['kind'],'from_node':s['from_node'],'to_node':s['to_node'],
                                  'branches':len(s['native_paths']),'capacity':s['capacity']} for sid,s in stairs.items()]
        c['assumptions']=[a for a in c['assumptions'] if 'FIFO' not in a]
        c['assumptions'].append('계단은 3D 경로 길이를 펼친 별도 2D JuPedSim 영역으로 보행하며 실제 XYZ 경로로 투영합니다. 개찰구는 통로별 한 명만 진입하며 출구 쪽이 막히면 점유를 유지합니다.')
        c['assumptions'].append('회전하는 계단참은 실제 수평 경로를 JuPedSim에서 걸으며 높이와 경사 속도를 별도로 복원합니다. 직선 펼침 좌표를 꺾인 경로에 압축해 배치하지 않습니다.')
    if native:
        from persona_evac.integration.station_layout import attach_display_geometry
        attach_display_geometry(c)
    else:
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
    if native:area_specs=network['area_specs']
    fire_settings = fire.meta["fire"] if fire else None
    for spec in area_specs:
        name = spec["id"]
        geom = network['geometry'][name] if native else floor_geometry(floors[spec["floor"]], spec.get("region"))
        if fire and fire_settings["floor"] == spec["floor"]:
            x, y, w, h = fire_settings["rect"]
            geom = geom.difference(box(x, y, x + w, y + h))
        if not native or fire:geom = pedestrian_geometry(geom)
        center_geometry = geom.buffer(-RADIUS - 0.02)
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
                pos = tuple(s['native_lane_specs'][i]['entry'][:2]) if native else tuple((center[j] + offset * tangent[j] for j in (0, 1)))
                if not center_geometry.covers(Point(pos)):
                    continue
                stage = sim.add_waypoint_stage(pos, 0.07 if native else 0.35)
                journey = sim.add_journey(jps.JourneyDescription([stage]))
                lanes[sid].append((journey, stage, pos))
                s["entry_lane_numbers"].append(i)
            if not lanes[sid]:
                s["active_entry_lanes"] = 0
                continue
                raise ValueError(f"{sid}: 계단 진입점이 보행 공간 안에 없습니다")
            stages[sid] = lanes[sid][0][:2]
            s["active_entry_lanes"] = len(lanes[sid])
        areas[name] = {"sim": sim, "geometry": geom, "center_geometry": center_geometry, "stages": stages, "lanes": lanes}
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
        if native:return connector_landing(s,nodes,lane)
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
                not areas[dest.get("area", "hall")]["center_geometry"]
                .covers(Point(point(dest)))
            ):
                continue
            valid.append(option)
            numbers.append(lane)
        s["physical_valid_lanes"] = numbers
        s["active_entry_lanes"] = len(valid)
        if native and not valid:
            raise ValueError(f'{sid}: 유효한 JuPedSim 연결 진입·도착 지점이 없습니다.')
    native_passages={sid:NativePassage(s,make_simulation,parameters,RADIUS) for sid,s in stairs.items()} if native else {}
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
        classification_assumptions[0]='6개 성향은 실험용 가정이며 실제 성격·위기 대응 능력의 측정값이 아닙니다.'
        classification_assumptions[1]='페르소나 전체를 LLM이 해석해 여섯 성향의 등급과 판단 근거만 생성합니다. 후속 상황 판단에는 성향 등급만 전달합니다. 타인 추종성은 본인이 관측한 이동 방향에만 적용합니다.'
    classification_assumptions[-1]='추월 적극성에 따른 의도적 추월은 사용하지 않습니다. 정체 해소용 국소 우회와 실제 앞지름 집계는 유지합니다.'
    from persona_evac.llm.baseline_profile import POLICY as BASELINE_POLICY
    if any(((a.get('provenance') or {}).get('mapping') or {}).get('policy') == BASELINE_POLICY for a in agents):
        classification_assumptions[5]='역사 친숙도는 LLM이 페르소나 전체 문맥에서 초행·구조에 익숙함으로 분류합니다. 익숙한 사람은 실제 역 모델의 모든 계단 위치·층 연결을 기억하지만 현재 위험·혼잡·통행 상태나 전체 보행 경로는 제공받지 않습니다.'
    c["assumptions"] += classification_assumptions
    now = 0.0

    def add_person(r, area_name, pos, target):
        area = areas[area_name]
        reserved=r.get('landing_guard') if native else None
        if not area["center_geometry"].covers(Point(pos)):
            raise ValueError("시작점 또는 계단 도착점이 보행 가능한 공간 밖에 있습니다.")
        if reserved and (reserved['area']!=area_name or math.dist(pos,reserved['position'])>1e-7):
            raise ValueError('예약한 계단 도착 몸체의 영역 또는 위치가 일치하지 않습니다.')
        # 예약 몸체는 수신 엔진 안에 이미 존재한다. 주변 사람이 신규 삽입
        # 여유(.52m) 안으로 다가와도 몸체 재사용에는 추가 공간이 필요 없다.
        # 이 검사를 반복하면 예약 몸체와 출구 앞 사람이 서로 막고 멈춘다.
        if not reserved and any(math.dist(pos, a.position) < 2 * RADIUS + 0.08
                                for a in area["sim"].agents()):
            return False
        goal = pos
        stage = area["sim"].add_waypoint_stage(pos, 0.2)
        journey = area["sim"].add_journey(jps.JourneyDescription([stage]))
        if reserved:
            jid=reserved['jid'];area['sim'].switch_agent_journey(jid,journey,stage)
            r.pop('landing_guard')
        else:
            jid = area["sim"].add_agent(parameters(pos, journey, stage, r["agent"]["desired_speed_mps"]))
        r.update(area=area_name, jid=jid, target=target, pos=pos, goal=goal, goal_stage=stage)
        r["motion_samples"] = deque([(now, pos)], maxlen=21)
        r['recovery_samples'] = deque([(now,tuple(pos))],maxlen=61)
        r["motion_distances"] = deque(maxlen=20)
        r["observed_speed_mps"] = 0
        r["entry_lane"] = None
        r["goal_is_portal"] = False
        return True

    def reserve_landing(r,dest):
        area=areas[dest['area']];pos=point(dest)
        if not area['center_geometry'].covers(Point(pos)):return False
        if any(math.dist(pos,a.position)<2*RADIUS+.08 for a in area['sim'].agents()):return False
        stage=area['sim'].add_waypoint_stage(pos,.07)
        journey=area['sim'].add_journey(jps.JourneyDescription([stage]))
        jid=area['sim'].add_agent(parameters(pos,journey,stage,0.))
        r['landing_guard']={'area':dest['area'],'jid':jid,'position':pos}
        return True

    for i, a in enumerate(agents):
        a.update(segments=[], stair_wait_s=0.0, evacuation_time_s=None, exit_id=None)
        a.update(fed=0.,gas_exposure_until_s=0.,incapacitated_at_s=None,
                 gas_exposure_history=[{'time_s':0.,'fed':0.}] if fire else [])
        a['visibility_blocked_history']=[{'time_s':0.,'blocked':False}]
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
            pace="continue",
        )
        initial = "explore:none"
        if not add_person(r, area, r["pos"], initial):
            raise ValueError("시작 위치에서 사람이 겹칩니다. 서로 다른 시작 위치가 필요합니다.")
        runtime[a["agent_id"]] = r


    def location(r):
        if r["phase"] == "on_stairs":
            if native:
                _,_,xyz=native_passages[r['target']].sample(r['agent']['agent_id'])
                return {'floor':r['floor'],'x':xyz[0],'y':xyz[1]}
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
        occupants_by_connector = connector_occupants(runtime, stairs)
        for sid, s in stairs.items():
            occupants = occupants_by_connector[sid]
            stair_speed = s["speed_mps"] * (1 if s.get('kind')=='fare_gate' else stair_speed_factor(s, len(occupants)))
            if native:
                passage=native_passages[sid]
                passage.advance(stair_speed/s['speed_mps'])
                for r in occupants:
                    progress,lateral,xyz=passage.sample(r['agent']['agent_id'])
                    spec=s['native_lane_specs'][r['stair_lane']]
                    length=passage.branches[spec['branch']]['length']
                    guard=r.get('source_guard')
                    if guard and progress*length>=2*RADIUS+.08:
                        areas[guard['area']]['sim'].mark_agent_for_removal(guard['jid']);r.pop('source_guard')
                    if not r.get('incapacitated') and (1-progress)*length<=1.15:
                        dest=landing_for(s,r['stair_lane'])
                        if dest['floor']=='OUTSIDE' or r.get('landing_guard') or reserve_landing(r,dest):
                            passage.release_landing(r['agent']['agent_id'])
                    r['stair_distance']=progress*s['length_m']
                    r['passage']['motion'].append([now,progress])
                    r['passage']['native_motion'].append([now,*xyz])
                    r['passage']['native_engine_motion'].append([now,*passage.engine_position(r['agent']['agent_id'])])
                continue
            fronts = {}
            for r in occupants:
                lane = r["stair_lane"]
                limit = fronts.get(lane, s["length_m"] + stair_gap) - stair_gap
                r["stair_distance"] = min(limit, r["stair_distance"] + (0 if r.get('incapacitated') else stair_speed * DT))
                fronts[lane] = r["stair_distance"]
                r["passage"]["motion"].append([now, r["stair_distance"] / s["length_m"]])
        if fire:
            fire.indices(now)
        for r in runtime.values():
            if (
                r.get('incapacitated') or r["phase"] != "on_stairs"
                or (not native_passages[r['target']].ready(r['agent']['agent_id']) if native
                    else r["stair_distance"] < stairs[r["target"]]["length_m"] - 1e-08)
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
            if native:native_passages[s['id']].release(r['agent']['agent_id'])
            r["passage"]["exit"] = now
            r["passage"]["landing_wait_s"] = max(0, now - r["ready_at"])
            if native:
                z=next((f['elevation_m'] for f in c['floors'] if f['id']==dest['floor']),8.)
                r['passage']['native_motion'].append([now,dest['x'],dest['y'],z])
            log_segment(
                r, r["passage"]["entry"], now, r["stair_start"], dict(dest),
                'on_gate' if s.get('kind')=='fare_gate' else "on_stairs", s["id"]
            )
            r["agent"]["segments"][-1]["stair_motion"] = r["passage"]["motion"]
            r["agent"]["segments"][-1]["stair_lane"] = r["stair_lane"]
            if native:r['agent']['segments'][-1]['native_motion']=r['passage']['native_motion']
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
                and (math.dist(r["pos"], r["goal"]) < (0.10 if native else 0.6))
            ):
                r.update(phase="queueing", arrived_at=now)
        waiting_by_connector, occupied_by_connector = connector_admissions(runtime, stairs)
        for sid, s in stairs.items():
            waiting = waiting_by_connector[sid]
            occupied = occupied_by_connector[sid]
            if (
                waiting
                and occupied < s["capacity"]
                and (now + 1e-08 >= last_entry[sid] + 1 / s["entry_rate_per_s"])
            ):
                free = []
                for lane in range(s["parallel_lanes"]):
                    dest = landing_for(s, lane)
                    if dest["floor"] != "OUTSIDE" and (
                        not areas[dest.get("area", "hall")]["center_geometry"]
                        .covers(Point(point(dest)))
                    ):
                        continue
                    if native:
                        if native_passages[sid].can_enter(lane):free.append(lane)
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
                if native and not native_passages[sid].enter(r['agent']['agent_id'],lane,s['speed_mps']):
                    continue
                r.update(stair_lane=lane, stair_distance=0)
                r["stair_destination"] = landing_for(s, lane)
                lane_entry[sid, lane] = now
                r["stair_start"] = location(r)
                if native:
                    areas[r['area']]['sim'].agent(r['jid']).model.desired_speed=0
                    r['source_guard']={'area':r['area'],'jid':r['jid']}
                else:areas[r["area"]]["sim"].mark_agent_for_removal(r["jid"])
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
                if native:
                    _,_,xyz=native_passages[sid].sample(r['agent']['agent_id'])
                    branch=native_passages[sid].branches[s['native_lane_specs'][lane]['branch']]
                    p.update(kind=s['kind'],native_motion=[[now,*xyz]],engine='jupedsim',
                             coordinate_model='bent_planar' if branch.get('bent_path') else 'unfolded_straight',
                             native_engine_motion=[[now,*native_passages[sid].engine_position(r['agent']['agent_id'])]])
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
            r['intended_walk_speed']=speed
            visibility_blocked=False
            if r["phase"] == "walking":
                target_visible=visible_motion_target(areas[r['area']]['sim'],r,private.sensor,now)
                visibility_blocked=not target_visible or (
                    bool(r.get('navigation_blocked')) and r.get('movement_block_reason')=='visibility')
                if r.get('navigation_blocked') or not target_visible:
                    speed=0
            record_visibility_block(r['agent'],now,visibility_blocked)
            actor.model.desired_speed = speed
        if tick % 10 == 0:
            for aid in before:
                r = runtime[aid]
                if r["phase"] != "walking":
                    continue
                area = areas[r["area"]]
                goal_stage = r["goal_stage"]
                update_stall_state(r,now)
                movement_allowed=area['sim'].agent(r['jid']).model.desired_speed>0 and not r.get('navigation_blocked')
                wall_retry=(r.get('movement_block_reason')=='route_unavailable' and bool(r.get('blocked_motion_target')))
                if not r['passing'] and (movement_allowed or wall_retry) and now-r['stall_since']>2.5 and now>=r['yield_after']:
                    helper_sim, helper_geom = private.visible_helper(r, now)
                    points = None
                    blocker_ids=[]
                    gate_merge=None
                    wall_recovery=False
                    if movement_allowed and helper_geom.covers(Point(r["pos"])) and helper_geom.covers(Point(r["goal"])):
                        try:
                            inactive={o['jid'] for o in runtime.values() if o['area']==r['area'] and o['phase'] in {'on_stairs','evacuated'}}
                            if r.get('goal_is_portal') and stairs.get(r['target'],{}).get('kind')=='fare_gate':
                                gate_merge=plan_gate_approach_yield(helper_sim,helper_geom,r['jid'],r['goal'],goal_stage,inactive)
                            if gate_merge is not None:
                                points=gate_merge['waypoints']
                            else:
                                points = plan_yield(helper_sim,helper_geom,r['jid'],r['goal'],goal_stage,
                                                    inactive,blocker_ids=blocker_ids)
                        except RuntimeError:
                            pass
                    if not points and (wall_retry or not r.get('goal_is_portal')):
                        inactive={o['jid'] for o in runtime.values() if o['area']==r['area'] and o['phase'] in {'on_stairs','evacuated'}}
                        points=recover_wall_route(private,r,now,inactive)
                        wall_recovery=bool(points)
                    if points:
                        r.update(
                            passing=True,
                            manoeuvre_waypoints=(r['manoeuvre_waypoints'] if wall_recovery else helper_sim.waypoint_positions),
                            yield_after=now + 4,
                            stall_origin=r["pos"],
                            stall_since=now,
                        )
                        event={
                            "agent_id": aid,
                            "time_s": now,
                            "floor": r["floor"],
                            "waypoints": points,
                            "observation_id": r.get("local_motion_observation_id"),
                            "cause": ("visible_gate_approach_yield" if gate_merge else
                                      "visible_wall_recovery" if wall_recovery else "visible_person_blocking_path"),
                            **({'priority_jps_id':gate_merge['priority_jps_id']} if gate_merge else {}),
                        }
                        yield_plans.append(event)
                        start_manoeuvre(r,event,'yielding' if gate_merge else
                                        'route_recovery' if wall_recovery else 'overtaking',now)
                        for other in runtime.values():
                            if other['area']!=r['area'] or other['jid'] not in blocker_ids:continue
                            attempt=pass_attempt(r,other,now,'local_recovery')
                            if attempt and not any(e['passing_agent']==attempt['passing_agent'] and e['passed_agent']==attempt['passed_agent'] for e in pending_passes):
                                pending_passes.append(attempt)
        for r in runtime.values():
            update_manoeuvre(r,now)
        for area in areas.values():
            area["sim"].iterate()
        for aid, old in before.items():
            r = runtime[aid]
            r["pos"] = tuple(areas[r["area"]]["sim"].agent(r["jid"]).position)
            r["agent"]["distance_walked_m"] += math.dist((old["x"], old["y"]), r["pos"])
            r["observed_speed_mps"] = append_motion_sample(r,now + DT,r["pos"])
            log_segment(
                r,
                now,
                round(now + DT, 8),
                old,
                location(r),
                r["phase"],
                r["target"] if r["phase"] == "queueing" else None,
            )
        # Physics-wide exposure, independent of visibility, decisions and speed.
        # A threshold crossing at this interval's end stops the next motion step.
        if fire:
            end_s=round(now+DT,8)
            for aid,r in runtime.items():
                if r['phase']=='evacuated':continue
                if r['phase']=='on_stairs':
                    if native:
                        _,_,xyz=native_passages[r['target']].sample(aid)
                    else:
                        raw=fire.sample_stair(c,stairs[r['target']],
                            {'from':r['stair_start'],'to':r['stair_destination']},
                            r['stair_distance']/stairs[r['target']]['length_m'],now+DT/2)
                        xyz=raw['physical_position_m']
                else:
                    xyz=[*r['pos'],floors[r['floor']]['elevation_m']]
                reading=fire.sample_gases_xyz(xyz,now+DT/2)
                r['last_gas_reading']=reading
                if accumulate_exposure(r,reading,now,end_s):
                    r['agent']['incapacitation']['physical_position_m']=list(xyz)
                    if r['phase']=='on_stairs':
                        r['agent']['incapacitation']['stair_id']=r['target']
                        if native:native_passages[r['target']].incapacitate(aid)
                    else:
                        areas[r['area']]['sim'].agent(r['jid']).model.desired_speed=0.
        for event in pending_passes[:]:
            a, b = (runtime[event["passing_agent"]], runtime[event["passed_agent"]])
            result=advance_pass_attempt(event,a,b,now)
            if result is None:continue
            if result:overtakes.append(result)
            pending_passes.remove(event)
        min_gap,overlap_steps=diagnose_people(runtime,before,areas,now,min_gap,overlap_steps,pair_history,rank_changes)
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
                    'on_gate' if stairs[sid].get('kind')=='fare_gate' else "on_stairs",
                    sid,
                )
                r["agent"]["segments"][-1].update(
                    stair_motion=r["passage"]["motion"], stair_lane=r["stair_lane"]
                )
                if native:r['agent']['segments'][-1]['native_motion']=r['passage']['native_motion']
    for r in runtime.values():
        finish_manoeuvre(r,now)
    unique_passes = {}
    for event in overtakes:
        key = (event["passing_agent"], event["passed_agent"], event["area"], event["target_stair"])
        unique_passes.setdefault(key, event)
    overtakes = list(unique_passes.values())
    if fire:
        for r in runtime.values():
            a=r['agent'];until=a['gas_exposure_until_s']
            if a['gas_exposure_history'][-1]['time_s']<until:
                a['gas_exposure_history'].append({'time_s':until,'fed':a['fed'],**r['last_gas_reading']})
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
        + (" · JuPedSim 층별·계단 보행 + 개찰구별 1인 통과 · " if native else " · JuPedSim 회피·추월 + 계단 시간·용량 연결 · ")
        + ("FDS 연기·온도·가스 노출 연결" if fire else "화재 끄기 실행"),
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
            "policy": {
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
            "overtake_definition": "국소 우회의 동일 앞사람을 고정 진행축에서 실제로 지나 0.5m 앞에 도달, 교차 시 측면 거리≥0.44m·몸 겹침 없음. 동일 층·목표·방향의 인물 쌍은 1회 집계",
        },
        summary={
            "count": len(agents),
            "all_evacuated": all((a["evacuation_time_s"] is not None for a in agents)),
            "completed_count": sum((a["evacuation_time_s"] is not None for a in agents)),
            "incapacitated_count": sum(a['incapacitated_at_s'] is not None for a in agents),
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
    from persona_evac.integration.personal_space import PRIOR_HASH,PRIOR

    base["agent_policy"] = {
        "versions": {**VERSIONS, "profile": private.profiles_provider.version},
        "policy_hash": POLICY_HASH,
        "classification_method": private.profiles_provider.method,
        "profile_version": private.profiles_provider.version,
        "profile_policy_hash": next(iter(private.profiles.values()))['policy_hash'],
        "experiment_seed": request.experiment_seed,
        "prior_hash": PRIOR_HASH,
        "navigation_policy": PRIOR['route_selection'],
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
