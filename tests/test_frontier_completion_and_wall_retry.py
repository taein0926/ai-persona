"""24·59번 공통 탐색점과 71번 벽 복구의 실제 좌표를 보행 엔진으로 검사한다."""
from collections import deque
import json,math
from pathlib import Path
from types import SimpleNamespace
import jupedsim as jps
import pytest
from shapely.affinity import affine_transform
from shapely.geometry import Point,box
from persona_evac.integration.personal_space import PersonalSpace,OccludedSensor
from persona_evac.integration.private_agents import PrivateController,PrivateMind
from persona_evac.integration.station_connectors import compile_network
from persona_evac.integration.station_dynamics import (
    make_simulation,parameters,pedestrian_geometry,plan_wall_recovery,recover_wall_route,
    visible_motion_target,append_motion_sample,update_stall_state,start_manoeuvre,
)

PAIRS=json.loads(Path('tests/fixtures/recorded_frontier_contacts.json').read_bytes())
MATRICES=[[1,0,0,1,0,0],[-1,0,0,1,160,0]]

@pytest.fixture(scope='module')
def network():return compile_network(json.loads(Path('configs/scenarios/gaehwasan.json').read_bytes()))

def profile():
    return {'traits':{name:{'level':'normal'} for name in
        ('crowd_avoidance','waiting_patience','risk_avoidance','exploration','social_following','overtaking_tendency')}}

def controller(geometry,people,k=0.,target=None,orientation=None):
    sim=make_simulation(geometry);runtime={};minds={}
    ctl=PrivateController.__new__(PrivateController);ctl.areas={'local':{'geometry':geometry,'sim':sim}}
    ctl.stairs={};ctl.mode='off';ctl.navigation_updates=[];ctl.local_geom_cache={};ctl.minds=minds;ctl.runtime=runtime
    class Smoke:
        def sample(self,*args):return {'extinction_per_m':k,'temperature_c':20}
    ctl.sensor=OccludedSensor(ctl.areas,{}, {},Smoke() if k else None)
    for person in people:
        aid=person['id'];p=tuple(person['position']);goal=tuple(target or (p[0]+1,p[1]))
        stage=sim.add_waypoint_stage(goal,.08);journey=sim.add_journey(jps.JourneyDescription([stage]));params=parameters(p,journey,stage,person['speed'])
        if orientation is not None:params.orientation=orientation
        jid=sim.add_agent(params)
        a={'agent_id':aid,'persona':{'uuid':str(aid)},'desired_speed_mps':person['speed'],'distance_walked_m':0.,'decision_wait_s':0.,'stair_wait_s':0.}
        mapping={'familiarity':{'level':'first_visit','assumption':'기록 재현'}}
        mind=PrivateMind({**a,'mapping':mapping},profile());mind.space.familiarity['level']='first_visit'
        mind.space.map_for('local')['adaptive']=True;minds[aid]=mind
        runtime[aid]={'agent':a,'jid':jid,'pos':p,'area':'local','floor':'B2','phase':'walking','target':'explore:none',
                     'goal':goal,'goal_stage':stage,'goal_is_portal':False,'passing':False,'wait_streak':0,
                     'action':'continue','stall_origin':p,'stall_since':0.,'yield_after':0.,'motion_samples':deque([(0.,p)],maxlen=21)}
    return ctl,sim,runtime

@pytest.mark.parametrize('case',PAIRS,ids=lambda c:str(c['time_s']))
@pytest.mark.parametrize('matrix',MATRICES)
def test_recorded_frontier_pair_keeps_moving_without_competing_for_the_same_point(network,case,matrix,record_property):
    origin=case['target'];base=pedestrian_geometry(network['geometry'][case['area']])
    if matrix!=MATRICES[0]:base=base.intersection(box(origin[0]-6,origin[1]-6,origin[0]+6,origin[1]+6))
    geom=affine_transform(base,matrix);convert=lambda p:tuple(affine_transform(Point(p),matrix).coords[0])
    people=[{**p,'id':p['id']+1000 if matrix!=MATRICES[0] else p['id'],'position':convert(p['position'])} for p in case['people']]
    goal=convert(case['target']);ctl,sim,runtime=controller(geom,people,case['k'],goal)
    target='explore:'+','.join(str(v/2) for v in goal)
    for r in runtime.values():r['target']=target
    starts={aid:r['pos'] for aid,r in runtime.items()};minimum=100.;longest_stop={aid:0. for aid in runtime};stopped={aid:0. for aid in runtime}
    for tick in range(160):
        now=tick*.05
        for r in runtime.values():r['pos']=tuple(sim.agent(r['jid']).position)
        ctl.tick(now)
        previous={aid:r['pos'] for aid,r in runtime.items()}
        for r in runtime.values():
            seen=visible_motion_target(sim,r,ctl.sensor,now)
            sim.agent(r['jid']).model.desired_speed=r['agent']['desired_speed_mps']*max(.1,1-.057*case['k']/.706) if seen and not r.get('navigation_blocked') else 0.
        sim.iterate()
        positions=[]
        for aid,r in runtime.items():
            p=tuple(sim.agent(r['jid']).position);distance=math.dist(previous[aid],p)
            assert geom.covers(Point(p)) and distance<=r['agent']['desired_speed_mps']*.05+1e-6
            stopped[aid]=stopped[aid]+.05 if distance<.0005 else 0.;longest_stop[aid]=max(longest_stop[aid],stopped[aid])
            r['agent']['distance_walked_m']+=distance;positions.append(p)
        minimum=min(minimum,math.dist(*positions));assert minimum>=.43
    moved={aid:math.dist(starts[aid],sim.agent(r['jid']).position) for aid,r in runtime.items()}
    record_property('moved_m',json.dumps(moved));record_property('minimum_center_distance_m',minimum)
    assert all(d>1. for d in moved.values()),(moved,longest_stop)
    assert max(longest_stop.values())<2.5,longest_stop
    assert all(any(o['kind']=='frontier_reached' for o in mind.space.observations) for mind in ctl.minds.values())

def test_recently_completed_frontier_is_not_readded_as_the_current_target():
    ctl,sim,rt=controller(box(0,0,12,8),[{'id':1,'position':(4.8,4.),'speed':1.}],target=(5.,4.))
    r=rt[1];space=ctl.minds[1].space;ctl.sensor.sense(space,r,rt,0.)
    sid='explore:2.5,2.0';space.mark_frontier_reached('local',(5.,4.),0.)
    assert sid not in {p['id'] for p in space.options('local',r['pos'],1.,1.,sid)}
    assert sid in {p['id'] for p in space.options('local',r['pos'],1.,11.,sid)}

@pytest.mark.parametrize('smoke,wall',[(20.,False),(0.,True)])
def test_nearby_frontier_is_not_completed_through_smoke_or_a_wall(smoke,wall):
    geom=box(0,0,10,8)
    if wall:geom=geom.difference(box(3.9,0,3.95,7))
    ctl,sim,rt=controller(geom,[{'id':1,'position':(3.65,4.),'speed':1.}],smoke,(3.98,4.));r=rt[1]
    r['target']='explore:1.99,2';option={'id':r['target'],'kind':'frontier','position':(3.98,4.),'arrival_radius_m':.4,'path':[(3.98,4.)]}
    ctl.observation=lambda *args,**kwargs:{'options':[option]}
    ctl.update_navigation(r,0.)
    assert ctl.minds[1].space.reached_frontiers=={}

@pytest.mark.parametrize('speed',[.8,1.170245,1.5])
def test_recorded_71_wall_recovery_first_moves_away_and_clears_the_corner(network,speed,record_property):
    geom=pedestrian_geometry(network['geometry']['ground']);start=(122.63906739988496,40.59057852724661);goal=(122.,42.)
    ctl,sim,rt=controller(geom,[{'id':71,'position':start,'speed':speed}],target=goal,orientation=(.9778789845975264,.20917144040836644));r=rt[71]
    points=recover_wall_route(ctl,r,0.)
    assert points
    initial=geom.boundary.distance(Point(start));p=points[0];length=math.dist(start,p)
    probe=tuple(start[i]+.05/length*(p[i]-start[i]) for i in (0,1))
    assert geom.boundary.distance(Point(probe))>initial+.002
    for tick in range(140):
        now=tick*.05;r['pos']=tuple(sim.agent(r['jid']).position)
        assert visible_motion_target(sim,r,ctl.sensor,now)
        sim.iterate();p=tuple(sim.agent(r['jid']).position)
        assert geom.covers(Point(p)) and math.dist(r['pos'],p)<=speed*.05+1e-6
        if math.dist(p,goal)<.12:break
    record_property('completion_s',round((tick+1)*.05,2))
    assert math.dist(p,goal)<.12 and (tick+1)*.05<5.,(p,tick*.05,points)

def test_a_stalled_active_recovery_releases_the_passing_lock_and_closes_its_event(network):
    geom=pedestrian_geometry(network['geometry']['ground']);start=(122.63906739988496,40.59057852724661);goal=(122.,42.)
    ctl,sim,rt=controller(geom,[{'id':71,'position':start,'speed':1.170245}],target=goal,orientation=(.9778789845975264,.20917144040836644));r=rt[71]
    bad=(122.14351864260428,41.68347899350947);stage=sim.add_waypoint_stage(bad,.08)
    journey=jps.JourneyDescription([stage,r['goal_stage']]);journey.set_transition_for_stage(stage,jps.Transition.create_fixed_transition(r['goal_stage']))
    sim.switch_agent_journey(r['jid'],sim.add_journey(journey),stage)
    r.update(passing=True,manoeuvre_waypoints={stage:bad});event={'agent_id':71,'time_s':0.}
    start_manoeuvre(r,event,'route_recovery',0.)
    for tick in range(101):
        now=tick*.05;r['pos']=tuple(sim.agent(r['jid']).position)
        assert visible_motion_target(sim,r,ctl.sensor,now)
        if not r['passing']:break
        sim.iterate()
    assert not r['passing'] and tick*.05<=3.5
    assert event['outcome']=='stalled_replan' and event['end_s']==tick*.05
    assert r['agent']['local_recovery_replans'] and r['yield_after']<=tick*.05
    assert r.get('movement_block_reason')!='visibility'

def test_slow_but_steadily_progressing_recovery_is_not_cancelled():
    ctl,sim,rt=controller(box(0,0,12,8),[{'id':1,'position':(3.,3.),'speed':.05}],target=(8.,3.));r=rt[1]
    step=(4.5,3.);stage=sim.add_waypoint_stage(step,.08);journey=jps.JourneyDescription([stage,r['goal_stage']])
    journey.set_transition_for_stage(stage,jps.Transition.create_fixed_transition(r['goal_stage']));sim.switch_agent_journey(r['jid'],sim.add_journey(journey),stage)
    r.update(passing=True,manoeuvre_waypoints={stage:step})
    for tick in range(100):
        r['pos']=tuple(sim.agent(r['jid']).position);assert visible_motion_target(sim,r,ctl.sensor,tick*.05);sim.iterate()
    assert r['passing'] and not r['agent'].get('local_recovery_replans')
