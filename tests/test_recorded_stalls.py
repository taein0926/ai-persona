"""실제 계단 진입 시각과 플랫폼 접촉 좌표를 재사용하는 보행 회귀 검사."""
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import jupedsim as jps
import pytest
from shapely import wkt
from shapely.geometry import Point

from persona_evac.integration.station_connectors import compile_network,NativePassage
from persona_evac.integration.station_dynamics import make_simulation,parameters,plan_yield,visible_motion_target
from persona_evac.integration.stair_dynamics import stair_speed_factor
from persona_evac.integration.private_agents import PrivateController
from persona_evac.integration.personal_space import OccludedSensor


@pytest.mark.parametrize('time_scale,swap_branch',[(1.,False),(1.2,False),(1.,True),(1.2,True)])
def test_recorded_h_e_cohort_clears_without_corner_deadlock(time_scale,swap_branch):
    config=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf-8'))
    connector=compile_network(config)['connectors']['H_E']
    schedule=json.loads(Path('tests/fixtures/recorded_h_e_arrivals.json').read_text(encoding='utf-8'))
    passage=NativePassage(connector,make_simulation,parameters)
    half=connector['parallel_lanes']//2
    pending=sorted([(p['entry']*time_scale,p['agent_id'],(p['lane']+half)%connector['parallel_lanes'] if swap_branch else p['lane'])
                    for p in schedule]);completed=set();pair_checks=0
    for tick in range(6000):
        now=tick*.05
        for record in pending[:]:
            entry,aid,lane=record
            if entry<=now and passage.enter(aid,lane,connector['speed_mps']):pending.remove(record)
        for aid in list(passage.occupants):
            progress,_,_=passage.sample(aid)
            branch=passage.branches[connector['native_lane_specs'][passage.occupants[aid][0]]['branch']]
            if (1-progress)*branch['length']<=1.15:passage.release_landing(aid)
        passage.advance(stair_speed_factor(connector,len(passage.occupants)))
        samples=[passage.sample(aid)[2] for aid in passage.occupants]
        for a,b in itertools.combinations(samples,2):
            assert math.dist(a,b)>=.43
            pair_checks+=1
        for aid in list(passage.occupants):
            if passage.ready(aid):passage.release(aid);completed.add(aid)
        if not pending and not passage.occupants:break
    assert completed=={p['agent_id'] for p in schedule}, {
        'pending':pending,'remaining':{aid:{'sample':passage.sample(aid),
        'stage':passage.branches[connector['native_lane_specs'][lane]['branch']]['sim'].agent(jid).stage_id,
        'hold':passage.branches[connector['native_lane_specs'][lane]['branch']]['stages'][lane],
        'final':passage.branches[connector['native_lane_specs'][lane]['branch']]['final_stages'][lane]}
        for aid,(lane,jid) in passage.occupants.items()}}
    assert {6,25,47}<=completed and pair_checks>1000


CASES=json.loads(Path('tests/fixtures/recorded_near_goal_contacts.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('case',CASES,ids=lambda p:str(p['agent_id']))
def test_recorded_short_view_contacts_resume_without_walking_blind(case):
    geometry=wkt.loads(case['geometry_wkt']);sim=make_simulation(geometry)
    aid=case['agent_id'];start=tuple(case['position']);anchor=tuple(case['goal'])
    following={17:(72.,24.),57:(74.,26.),58:(72.,26.),89:(64.,26.)}[aid]
    class Smoke:
        def sample(self,*args):return {'extinction_per_m':case['extinction_per_m'],'temperature_c':20}
    runtime={}
    for person in [{'id':aid,'position':start}]+case['visible_people']:
        own=person['id']==aid;point=tuple(person['position'])
        stage=sim.add_waypoint_stage(anchor if own else point,.08)
        journey=sim.add_journey(jps.JourneyDescription([stage]))
        jid=sim.add_agent(parameters(point,journey,stage,1. if own else 0.))
        runtime[person['id']]={'jid':jid,'agent':{'agent_id':person['id']},'area':'A','floor':'B2','phase':'walking',
            'pos':point,'goal':anchor if own else point,'goal_stage':stage,'goal_is_portal':False,
            'passing':False,'target':'known','local_replan_after':1000.}
    own=runtime[aid];controller=PrivateController.__new__(PrivateController)
    controller.runtime=runtime;controller.areas={'A':{'geometry':geometry,'sim':sim}};controller.stairs={}
    controller.local_geom_cache={};controller.sensor=OccludedSensor(controller.areas,{}, {},Smoke())
    controller.minds={aid:SimpleNamespace(pending=set(),space=SimpleNamespace(record=lambda *args:'own_view'))}
    controller.observation=lambda *args,**kwargs:{'options':[{'id':'known','kind':'up_stair','position':following,'path':[anchor,following]}]}
    stalled=0;routes=0
    for tick in range(400):
        now=tick*.05;previous=tuple(sim.agent(own['jid']).position);own['pos']=previous
        if not own['passing'] and (tick%10==0 or math.dist(previous,own['goal'])<.65):controller.update_navigation(own,now)
        assert not own.get('navigation_blocked')
        assert visible_motion_target(sim,own,controller.sensor,now)
        if stalled>50 and not own['passing']:
            helper,view=controller.visible_helper(own,now)
            route=plan_yield(helper,view,own['jid'],own['goal'],own['goal_stage'])
            if route:
                own.update(passing=True,manoeuvre_waypoints=helper.waypoint_positions)
                routes+=1;stalled=0
        sim.iterate();current=tuple(sim.agent(own['jid']).position)
        assert math.dist(previous,current)<=.051
        assert geometry.covers(Point(current))
        for other in runtime.values():
            if other is not own:assert math.dist(current,sim.agent(other['jid']).position)>=.43
        stalled=stalled+1 if math.dist(previous,current)<.0025 else 0
        direction=(following[0]-start[0],following[1]-start[1]);length=math.hypot(*direction)
        progress=sum((current[i]-start[i])*direction[i]/length for i in (0,1))
        if progress>1.1:break
    assert progress>1.1,(aid,current,routes)
