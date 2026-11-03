"""기록된 G_13 입구 접촉을 실제 보행 엔진과 단독 통행 채널로 재현한다."""
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import jupedsim as jps
import pytest
from shapely.geometry import Point,box

from persona_evac.integration.personal_space import OccludedSensor
from persona_evac.integration.private_agents import PrivateController,VisibleSimulation
from persona_evac.integration.station_connectors import compile_network,NativePassage
from persona_evac.integration.station_dynamics import (
    make_simulation,parameters,plan_yield,plan_gate_approach_yield,visible_motion_target,
)

CASES=json.loads(Path('tests/fixtures/recorded_gate_merges.json').read_bytes())


def replay(case,speed=1.,reverse_ids=False,recovery=True):
    network=compile_network(json.loads(Path('configs/scenarios/gaehwasan.json').read_bytes()))
    connector=network['connectors'][case['gate']];goal=connector['native_lane_specs'][0]['entry'][:2]
    geometry=network['geometry']['hall_paid'];sim=make_simulation(geometry)
    passage=NativePassage(connector,make_simulation,parameters);runtime={};completed={};entered=[];routes=[]
    people=list(case['people'])
    for index,person in enumerate(people):
        aid=people[-1-index]['agent_id'] if reverse_ids else person['agent_id']
        stage=sim.add_waypoint_stage(goal,.07);journey=sim.add_journey(jps.JourneyDescription([stage]))
        jid=sim.add_agent(parameters(tuple(person['position']),journey,stage,speed))
        runtime[aid]={'agent':{'agent_id':aid},'jid':jid,'area':'hall_paid','floor':'B1','phase':'walking',
            'pos':tuple(person['position']),'goal':tuple(goal),'goal_stage':stage,'goal_is_portal':True,
            'passing':False,'target':case['gate'],'stall_origin':tuple(person['position']),'stall_since':0.,'yield_after':0.}
    for person in case.get('stationary_bystanders',[]):
        pos=tuple(person['position']);stage=sim.add_waypoint_stage(pos,.07)
        jid=sim.add_agent(parameters(pos,sim.add_journey(jps.JourneyDescription([stage])),stage,0.))
        runtime[person['agent_id']]={'agent':{'agent_id':person['agent_id']},'jid':jid,'area':'hall_paid','floor':'B1',
                                   'phase':'waiting','pos':pos}
    class Smoke:
        def sample(self,*args):return {'extinction_per_m':max(p['smoke'] for p in people),'temperature_c':20}
    controller=PrivateController.__new__(PrivateController)
    controller.runtime=runtime;controller.areas={'hall_paid':{'geometry':geometry,'sim':sim}}
    controller.local_geom_cache={};controller.sensor=OccludedSensor(controller.areas,{}, {},Smoke())
    controller.minds={aid:SimpleNamespace(space=SimpleNamespace(record=lambda *args:'local_view')) for aid in runtime}
    minimum=math.inf
    for tick in range(400):
        now=tick*.05
        for aid in list(passage.occupants):
            progress,_,_=passage.sample(aid)
            if (1-progress)*connector['length_m']<=1.15:passage.release_landing(aid)
        passage.advance()
        for aid in list(passage.occupants):
            progress,_,_=passage.sample(aid);r=runtime[aid]
            if not r.get('guard_removed') and progress*connector['length_m']>=.52:
                sim.mark_agent_for_removal(r['jid']);r['guard_removed']=True
            if passage.ready(aid):
                passage.release(aid);r['phase']='evacuated';completed[aid]=now
        active=[r for r in runtime.values() if r['phase']=='walking']
        previous={r['agent']['agent_id']:tuple(sim.agent(r['jid']).position) for r in active}
        for r in active:
            r['pos']=previous[r['agent']['agent_id']]
            assert visible_motion_target(sim,r,controller.sensor,now)
            sim.agent(r['jid']).model.desired_speed=speed
            if math.dist(r['pos'],r['stall_origin'])>.15:r.update(stall_origin=r['pos'],stall_since=now)
            if not recovery or tick%10 or r['passing'] or now-r['stall_since']<=2.5 or now<r['yield_after']:continue
            helper,view=controller.visible_helper(r,now)
            decision=plan_gate_approach_yield(helper,view,r['jid'],goal,r['goal_stage'])
            points=decision['waypoints'] if decision is not None else plan_yield(helper,view,r['jid'],goal,r['goal_stage'])
            if points:
                routes.append((r['agent']['agent_id'],now,points))
                r.update(passing=True,manoeuvre_waypoints=helper.waypoint_positions,yield_after=now+4,stall_origin=r['pos'],stall_since=now)
        sim.iterate()
        physical=[]
        for r in active:
            aid=r['agent']['agent_id'];current=tuple(sim.agent(r['jid']).position)
            assert math.dist(previous[aid],current)<=speed*.05+.001
            assert geometry.covers(Point(current))
            r['pos']=current;physical.append((aid,current))
        for aid in passage.occupants:physical.append((aid,passage.sample(aid)[2][:2]))
        for r in runtime.values():
            if r['phase']=='waiting':physical.append((r['agent']['agent_id'],tuple(sim.agent(r['jid']).position)))
        for (aid,a),(bid,b) in itertools.combinations(physical,2):
            gap=math.dist(a,b);minimum=min(minimum,gap)
            assert gap>=.43,(aid,bid,now,gap)
        for r in active:
            aid=r['agent']['agent_id']
            if math.dist(r['pos'],goal)<.1 and passage.can_enter(0):
                assert passage.enter(aid,0,1.)
                sim.agent(r['jid']).model.desired_speed=0.;r['phase']='on_stairs';entered.append((aid,now))
        assert len(passage.occupants)<=1
        if len(completed)==len(people):break
    return completed,entered,routes,minimum


@pytest.mark.parametrize('case',CASES,ids=lambda p:str(p['time_s']))
@pytest.mark.parametrize('speed,reverse_ids',[(1.,False),(.8,True)])
def test_recorded_gate_merges_clear_with_one_person_yielding(case,speed,reverse_ids,record_property):
    completed,entered,routes,minimum=replay(case,speed,reverse_ids)
    record_property('completion_times_s',json.dumps(completed,sort_keys=True))
    record_property('entry_times_s',json.dumps(entered))
    record_property('minimum_center_distance_m',minimum)
    assert len(completed)==2 and max(completed.values())<12,(completed,entered,routes)
    assert routes and minimum>=.43


def test_recorded_gate_pair_needs_walking_recovery_to_break_the_contact_equilibrium():
    completed,entered,_,_=replay(CASES[0],recovery=False)
    assert len(completed)<2,(completed,entered)


def test_unseen_person_does_not_receive_or_change_gate_priority():
    sim=make_simulation(box(0,0,8,8));goal=(4.,4.)
    stage=sim.add_waypoint_stage(goal,.07);journey=sim.add_journey(jps.JourneyDescription([stage]))
    own=sim.add_agent(parameters((3.8,4.2),journey,stage,1.))
    sim.add_agent(parameters((3.8,3.75),journey,stage,1.))
    visible=VisibleSimulation(sim,{own})
    assert plan_gate_approach_yield(visible,box(0,0,8,8),own,goal,stage) is None


def test_recorded_nearby_body_does_not_block_the_gate_yield_path(record_property):
    case={**CASES[0],'stationary_bystanders':[{'agent_id':41,'position':[116.2767183622008,11.524065339990369]}]}
    completed,entered,routes,minimum=replay(case)
    record_property('completion_times_s',json.dumps(completed,sort_keys=True))
    record_property('minimum_center_distance_m',minimum)
    assert len(completed)==2 and max(completed.values())<12,(completed,entered,routes)
    assert minimum>=.43
