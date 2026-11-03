"""기록된 먼 경유점·벽 앞 왕복을 다른 ID·방향·속도에서도 국소 재생한다."""
from collections import deque
import json
import math
from pathlib import Path
from types import SimpleNamespace

import jupedsim as jps
import pytest
from shapely.affinity import affine_transform
from shapely.geometry import Point,box

from persona_evac.integration.personal_space import OccludedSensor
from persona_evac.integration.station_connectors import compile_network
from persona_evac.integration.station_dynamics import make_simulation,parameters,pedestrian_geometry,append_motion_sample
from test_near_waypoint_navigation import controller_for


@pytest.fixture(scope='module')
def network():
    return compile_network(json.loads(Path('configs/scenarios/gaehwasan.json').read_bytes()))


@pytest.mark.parametrize('aid,start',[(15,(135.14431122915468,10.030308181391897)),(100,(131.,9.))])
@pytest.mark.parametrize('matrix',[[1,0,0,1,0,0],[-1,0,0,1,300,0],[0,-1,1,0,50,0]])
def test_recorded_distant_stair_is_reached_by_real_walking(network,aid,start,matrix):
    geom=affine_transform(pedestrian_geometry(network['geometry']['B']),matrix)
    convert=lambda p:tuple(affine_transform(Point(p),matrix).coords[0])
    start=convert(start);goal=convert((117.3312,7.672))
    ctl,sim,r=controller_for(geom,start,[goal],aid)
    for tick in range(500):
        r['pos']=tuple(sim.agent(r['jid']).position)
        if tick%10==0 or math.dist(r['pos'],r['goal'])<.65:ctl.update_navigation(r,tick*.05)
        assert not r.get('navigation_blocked')
        sim.iterate();p=tuple(sim.agent(r['jid']).position)
        assert geom.covers(Point(p)) and math.dist(r['pos'],p)<=.061
        if math.dist(p,goal)<.15:break
    assert math.dist(p,goal)<.15


@pytest.mark.parametrize('distance',[4.50001,5.,18.,40.])
@pytest.mark.parametrize('view',[.4,1.5,18.])
def test_long_route_uses_only_the_visible_prefix(distance,view):
    geom=box(0,0,60,12);start=(2.,6.);goal=(2.+distance,6.)
    visible=lambda area,floor,p,q,t:math.dist(p,q)<=view
    ctl,sim,r=controller_for(geom,start,[goal],visible=visible)
    ctl.update_navigation(r,0.)
    assert not r.get('navigation_blocked')
    assert .1<math.dist(start,r['goal'])<=min(view,4.5)+1e-8
    assert visible(None,None,start,r['goal'],0.)
    for _ in range(4):sim.iterate()
    assert math.dist(start,sim.agent(r['jid']).position)>.05


def test_clipping_a_long_route_does_not_cut_through_a_wall():
    geom=box(0,0,12,10).difference(box(4,0,6,8))
    ctl,_,r=controller_for(geom,(2.,3.),[(9.,3.)])
    ctl.update_navigation(r,0.)
    assert r['navigation_blocked'] and r['goal']==(9.,3.)


def test_reached_nearby_goal_cannot_hide_the_next_small_visible_step():
    ctl,sim,r=controller_for(box(0,0,10,10),(3.,3.),[(3.2,3.)],old_goal=(3.08,3.))
    ctl.update_navigation(r,0.)
    assert r['goal']==(3.2,3.)
    for _ in range(3):sim.iterate()
    assert sim.agent(r['jid']).position[0]>3.1


def test_final_smoke_limited_endpoint_is_rechecked_after_shortening():
    start=(2.,3.)
    def visible(area,floor,p,q,now):
        distance=math.dist(p,q)
        return distance<=2. and not 1.9<distance<1.97
    ctl,_,r=controller_for(box(0,0,10,10),start,[(6.,3.)],visible=visible)
    ctl.update_navigation(r,0.)
    assert not r.get('navigation_blocked') and visible(None,None,start,r['goal'],0.)


@pytest.mark.parametrize('matrix',[[1,0,0,1,0,0],[-1,0,0,1,250,0],[0,-1,1,0,50,0]])
@pytest.mark.parametrize('speed',[.9,1.181,1.5])
def test_wall_oscillation_recovery_walks_out_with_clearance_and_current_visibility(network,matrix,speed):
    from persona_evac.integration.station_dynamics import update_stall_state,plan_wall_recovery,visible_motion_target
    # Keep the recorded full floor for the original orientation. The transformed
    # copies use the same local wall, excluding distant degenerate drawing holes.
    base=pedestrian_geometry(network['geometry']['ground'])
    if matrix!=[1,0,0,1,0,0]:base=base.intersection(box(114,36,130,48))
    geom=affine_transform(base,matrix)
    convert=lambda p:tuple(affine_transform(Point(p),matrix).coords[0])
    start=convert((122.61615327402355,40.653116882071025));goal=convert((122.,42.))
    ctl,sim,r=controller_for(geom,start,[goal],aid=987)
    sim.agent(r['jid']).model.desired_speed=speed
    r.update(passing=False,stall_origin=start,stall_since=0.,yield_after=0.,motion_samples=deque([(0.,start)],maxlen=21))
    ctl.runtime={987:r};ctl.local_geom_cache={}
    ctl.sensor=OccludedSensor(ctl.areas,{}, {},None)
    ctl.minds[987].space=SimpleNamespace(record=lambda *args:'own_view')
    routes=[]
    for tick in range(200):
        now=tick*.05;r['pos']=tuple(sim.agent(r['jid']).position)
        assert visible_motion_target(sim,r,ctl.sensor,now)
        if tick%10==0 and not r['passing'] and update_stall_state(r,now)>2.5:
            helper,view=ctl.visible_helper(r,now)
            route=plan_wall_recovery(helper,view,r['jid'],goal,r['goal_stage'])
            if route:
                routes.append((now,route));r.update(passing=True,manoeuvre_waypoints=helper.waypoint_positions)
        before=r['pos'];sim.iterate();p=tuple(sim.agent(r['jid']).position)
        assert geom.covers(Point(p)) and math.dist(before,p)<=speed*.05+1e-6
        # AVM has soft wall contact; the unchanged body cannot cross more than
        # one physical time step into its wall margin.
        assert geom.boundary.distance(Point(p))>=.22-speed*.05-1e-5
        append_motion_sample(r,now+.05,p)
        if math.dist(p,goal)<.1:break
    assert math.dist(p,goal)<.1,(routes,p)
    if matrix==[1,0,0,1,0,0] and speed==1.181:assert routes
    assert tick*.05<7.5,(routes,tick*.05)


def test_recorded_small_loop_is_detected_despite_normal_instantaneous_speed():
    from persona_evac.integration.station_dynamics import update_stall_state
    samples=deque((i*.05,(math.cos(i*.6)*.08,math.sin(i*.6)*.08)) for i in range(61))
    r={'pos':samples[-1][1],'stall_origin':(0.,0.),'stall_since':2.9,'recovery_samples':samples}
    assert update_stall_state(r,3.)>=2.5


def test_slow_steady_walking_is_not_treated_as_a_loop():
    from persona_evac.integration.station_dynamics import update_stall_state
    r={'pos':(.3,0.),'stall_origin':(0.,0.),'stall_since':0.,
       'recovery_samples':deque((i*.05,(i*.005,0.)) for i in range(61))}
    assert update_stall_state(r,3.)==0.


def test_open_space_does_not_trigger_wall_recovery():
    from persona_evac.integration.station_dynamics import plan_wall_recovery
    ctl,sim,r=controller_for(box(0,0,12,10),(3.,3.),[(8.,3.)])
    assert plan_wall_recovery(sim,ctl.areas['hall']['geometry'],r['jid'],r['goal'],r['goal_stage']) is None


def test_contact_recovery_cannot_use_an_unseen_goal():
    from persona_evac.integration.station_dynamics import recover_wall_route
    ctl,sim,r=controller_for(box(0,0,10,10),(.3,4.),[(.5,6.)])
    ctl.sensor=SimpleNamespace(visible=lambda *args:False)
    ctl.visible_helper=lambda *args:pytest.fail('보이지 않는 목표에 우회를 시도했습니다.')
    r.update(blocked_motion_target=(.5,6.),navigation_blocked='contact',passing=False)
    assert recover_wall_route(ctl,r,0.) is None and r['navigation_blocked']=='contact'


def test_original_corner_without_recovery_walks_in_place(network):
    geom=pedestrian_geometry(network['geometry']['ground']);start=(122.61615327402355,40.653116882071025)
    _,sim,r=controller_for(geom,start,[(122.,42.)])
    sim.agent(r['jid']).model.desired_speed=1.181
    walked=0.;before=start
    for _ in range(80):
        sim.iterate();after=tuple(sim.agent(r['jid']).position);walked+=math.dist(before,after);before=after
    assert walked>4. and math.dist(start,after)<.2


def test_recovery_rechecks_the_first_step_before_changing_motion(network):
    from persona_evac.integration.private_agents import VisibleSimulation
    from persona_evac.integration.station_dynamics import recover_wall_route
    geom=pedestrian_geometry(network['geometry']['ground']);goal=(122.,42.)
    ctl,sim,r=controller_for(geom,(122.61615327402355,40.653116882071025),[goal])
    helper=VisibleSimulation(sim,{r['jid']});ctl.visible_helper=lambda *args:(helper,geom)
    ctl.sensor=SimpleNamespace(visible=lambda area,floor,p,q,t:tuple(q)==goal)
    assert recover_wall_route(ctl,r,0.) is None
    assert sim.agent(r['jid']).stage_id==r['goal_stage'] and not r.get('passing')


def test_wall_recovery_cannot_walk_through_a_visible_bystander(network):
    from persona_evac.integration.station_dynamics import plan_wall_recovery
    geom=pedestrian_geometry(network['geometry']['ground']);start=(122.61615327402355,40.653116882071025)
    ctl,sim,r=controller_for(geom,start,[(122.,42.)])
    block=(122.9,41.3);stage=sim.add_waypoint_stage(block,.08)
    sim.add_agent(parameters(block,sim.add_journey(jps.JourneyDescription([stage])),stage,0.))
    route=plan_wall_recovery(sim,geom,r['jid'],r['goal'],r['goal_stage'])
    if route:
        from shapely.geometry import LineString
        assert LineString([start,*route,r['goal']]).distance(Point(block))>=.455


LOOPS=json.loads(Path('tests/fixtures/recorded_movement_loops.json').read_bytes())


@pytest.mark.parametrize('case',LOOPS,ids=lambda c:f"{c['agent_id']}-{c['start_s']}")
def test_other_recorded_loop_positions_reach_their_local_goal(network,case):
    """기록 좌표·국소 목표만 재생한다. 미기록 원래 경유점·타인 계획은 추정하지 않는다."""
    from persona_evac.integration.station_dynamics import update_stall_state,recover_wall_route,visible_motion_target
    geom=pedestrian_geometry(network['geometry'][case['area']]);start=tuple(case['position']);goal=tuple(case['goal'])
    if math.dist(start,goal)<.12:return  # 이미 도달한 탐색점: 재관측 후 다음 목표 선택 대상.
    ctl,sim,r=controller_for(geom,start,[goal],aid=case['agent_id'])
    speed=case['speed'];sim.agent(r['jid']).model.desired_speed=speed
    r.update(passing=False,stall_origin=start,stall_since=0.,motion_samples=deque([(0.,start)],maxlen=21))
    ctl.runtime={case['agent_id']:r};ctl.local_geom_cache={};ctl.sensor=OccludedSensor(ctl.areas,{}, {},None)
    ctl.minds[case['agent_id']].space=SimpleNamespace(record=lambda *args:'own_view')
    for tick in range(200):
        now=tick*.05;r['pos']=tuple(sim.agent(r['jid']).position)
        if not r['passing']:ctl.update_navigation(r,now)
        sim.agent(r['jid']).model.desired_speed=0. if r.get('navigation_blocked') else speed
        r['intended_walk_speed']=speed
        if tick%10==0 and not r['passing'] and update_stall_state(r,now)>2.5:
            recover_wall_route(ctl,r,now)
        if not r.get('navigation_blocked'):assert visible_motion_target(sim,r,ctl.sensor,now)
        sim.iterate();p=tuple(sim.agent(r['jid']).position)
        assert geom.covers(Point(p)) and math.dist(p,r['pos'])<=speed*.05+1e-6
        append_motion_sample(r,now+.05,p)
        if math.dist(p,goal)<.12:break
    assert math.dist(p,goal)<.12,(case['agent_id'],p,goal)
