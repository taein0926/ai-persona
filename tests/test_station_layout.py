import json
import math
from pathlib import Path

import pytest
from shapely.geometry import Point,box

from persona_evac.integration.station_layout import stair_path,head_opening,HEAD_HEIGHT,SLAB_THICKNESS
from persona_evac.integration.station_geometry import floor_geometry
from persona_evac.integration.station_dynamics import make_simulation,parameters,plan_yield,pedestrian_geometry,RADIUS
import jupedsim as jps


def test_stair_opening_starts_when_head_reaches_underside():
    c={'floors':[{'id':'low','elevation_m':0},{'id':'high','elevation_m':4}],
       'nodes':{'a':{'floor':'low','x':0,'y':0},'b':{'floor':'high','x':10,'y':0}}}
    s={'id':'s','from_node':'a','to_node':'b','width_m':2}
    for spread in (1,2.5,4):
        opening=head_opening(c,s,spread)
        x=(4*spread-SLAB_THICKNESS-HEAD_HEIGHT)/(4*spread)*10
        assert not opening.covers(Point(x-.05,0))
        assert opening.covers(Point(x+.05,0))


def test_drawing_endpoints_and_u_stair_has_flat_landing():
    c=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf8'))
    s=next(s for s in c['stairs'] if s['id']=='H_E');p=stair_path(c,s)
    assert p[1][2]==p[2][2] and math.dist(p[1][:2],p[2][:2])>s['width_m']
    assert p[0][0]==c['nodes'][s['from_node']]['x']
    assert p[-1][1]==c['nodes'][s['to_node']]['y']
    assert len(c['train']['cars'])==8 and len(c['spawn_points'])==100


def test_equal_speed_counterflow_can_pass_without_body_overlap():
    geom=box(0,0,12,3);sim=make_simulation(geom)
    left=sim.add_waypoint_stage((1,1.5),.3);right=sim.add_waypoint_stage((11,1.5),.3)
    jl=sim.add_journey(jps.JourneyDescription([left]));jr=sim.add_journey(jps.JourneyDescription([right]))
    a=sim.add_agent(parameters((5,1.5),jr,right,1.0));b=sim.add_agent(parameters((5.5,1.5),jl,left,1.0))
    assert plan_yield(sim,geom,a,(11,1.5),right)
    for _ in range(250):
        sim.iterate()
        assert math.dist(sim.agent(a).position,sim.agent(b).position)>=2*RADIUS-.01
    assert sim.agent(a).position[0]>sim.agent(b).position[0]+2


def test_narrow_room_sliver_is_not_a_pedestrian_route():
    original=box(0,0,20,4).difference(box(8,.8,12,3.8))
    geom=pedestrian_geometry(original)
    assert not geom.covers(Point(10,3.9))
    assert geom.covers(Point(10,.4))
    sim=make_simulation(geom);goal=(7,3.4)
    stage=sim.add_waypoint_stage(goal,.3);journey=sim.add_journey(jps.JourneyDescription([stage]))
    agent=sim.add_agent(parameters((13,3.4),journey,stage,1))
    for _ in range(600):
        sim.iterate()
        if math.dist(sim.agent(agent).position,goal)<.6:break
    assert math.dist(sim.agent(agent).position,goal)<.6


def test_gaehwasan_east_to_west_stair_avoids_function_room_sliver():
    c=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf8'))
    floor=next(f for f in c['floors'] if f['id']=='B2')
    geom=pedestrian_geometry(floor_geometry(floor,0))
    goal=(c['nodes']['A_W_in']['x'],c['nodes']['A_W_in']['y']);start=(115,30)
    route=jps.RoutingEngine(geom).compute_waypoints(start,goal)
    assert min(p[1] for p in route)<27
    sim=make_simulation(geom);stage=sim.add_waypoint_stage(goal,.35)
    journey=sim.add_journey(jps.JourneyDescription([stage]))
    actor=sim.add_agent(parameters(start,journey,stage,1.25))
    for _ in range(1600):
        sim.iterate()
        if math.dist(sim.agent(actor).position,goal)<.6:break
    assert math.dist(sim.agent(actor).position,goal)<.6
