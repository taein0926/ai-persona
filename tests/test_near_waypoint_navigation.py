"""가까운 모서리 경유점을 건너뛰어 생기던 정지를 실제 보행으로 검증한다."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from shapely.affinity import affine_transform
from shapely.geometry import Point, box

from persona_evac.integration.personal_space import observed_connector_checker
from persona_evac.integration.private_agents import PrivateController
from persona_evac.integration.station_connectors import compile_network
from persona_evac.integration.station_dynamics import make_simulation, parameters, pedestrian_geometry


@pytest.fixture(scope='module')
def recorded_geometry():
    config=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf-8'))
    return pedestrian_geometry(compile_network(config)['geometry']['hall_paid'])


def controller_for(geometry,position,path,aid=1,visible=None,old_goal=None):
    sim=make_simulation(geometry)
    stage=sim.add_waypoint_stage(old_goal or path[0],.08)
    import jupedsim as jps
    journey=sim.add_journey(jps.JourneyDescription([stage]))
    jid=sim.add_agent(parameters(position,journey,stage,1.2))
    controller=PrivateController.__new__(PrivateController)
    controller.areas={'hall':{'geometry':geometry,'sim':sim}}
    controller.stairs={}
    controller.sensor=SimpleNamespace(visible=visible or (lambda *args:True))
    controller.minds={aid:SimpleNamespace(pending=set())}
    option={'id':'known','kind':'exit_stair','position':path[-1],'path':path}
    controller.observation=lambda *args,**kwargs:{'options':[option]}
    runtime={'agent':{'agent_id':aid},'area':'hall','floor':'B1','pos':position,
             'goal':old_goal or path[0],'goal_stage':stage,'goal_is_portal':False,
             'phase':'walking','target':'known','jid':jid}
    return controller,sim,runtime


@pytest.mark.parametrize('aid,matrix',[
    (1,[1,0,0,1,0,0]), (42,[-1,0,0,1,220,0]),
    (80,[0,-1,1,0,30,-90]), (100,[1,0,0,-1,20,40]),
])
@pytest.mark.parametrize('remaining',[.04,.09,.14561615908013])
def test_near_corner_remains_reachable_and_real_pedestrian_continues(recorded_geometry,aid,matrix,remaining):
    def transform(point):
        return tuple(affine_transform(Point(point),matrix).coords[0])
    geometry=affine_transform(recorded_geometry,matrix)
    position=transform((105.99389532400282,10-remaining))
    path=[transform(p) for p in [(106,10),(108,10),(110,10)]]
    check=observed_connector_checker(geometry,position)
    assert math.dist(position,path[0])<.15 and check(path[0])
    needs_corner=not check(path[1])
    assert needs_corner==(remaining>.14)
    controller,sim,runtime=controller_for(geometry,position,path,aid)
    controller.update_navigation(runtime,0.)
    assert not runtime['navigation_blocked']
    if needs_corner:assert runtime['goal']==path[0]
    # Exercise repeated navigation refreshes and engine movement, rather than
    # merely checking a selected coordinate. No other pedestrian or LLM is used.
    longest_stop=stopped=0
    for tick in range(160):
        before=tuple(sim.agent(runtime['jid']).position)
        runtime['pos']=before
        if tick%10==0:controller.update_navigation(runtime,tick*.05)
        assert not runtime['navigation_blocked']
        sim.iterate()
        after=tuple(sim.agent(runtime['jid']).position)
        assert geometry.covers(Point(after)) and math.dist(before,after)<=.061
        stopped=stopped+1 if math.dist(before,after)<1e-7 else 0
        longest_stop=max(longest_stop,stopped)
        if math.dist(after,path[-1])<.15:break
    assert math.dist(after,path[-1])<.15
    assert longest_stop<10


def test_clear_next_segment_skips_near_and_duplicate_points():
    position=(2.,2.)
    path=[position,position,(2.1,2.),(4.,2.)]
    controller,_,runtime=controller_for(box(0,0,8,6),position,path)
    controller.update_navigation(runtime,0.)
    assert runtime['goal']==(4.,2.) and not runtime['navigation_blocked']


def test_close_anchor_does_not_allow_an_unseen_next_segment():
    controller,_,runtime=controller_for(box(0,0,8,6),(2.,2.),[(2.1,2.),(4.,2.)],
        visible=lambda area,floor,p,q,now:q[0]<3)
    controller.update_navigation(runtime,0.)
    assert 2.1<=runtime['goal'][0]<3 and runtime['goal'][1]==2.
    assert not runtime['navigation_blocked']


def test_near_existing_goal_cannot_retain_a_newly_invisible_target(recorded_geometry):
    position=(105.99389532400282,9.85438384091987)
    anchor=(106.,10.)
    old_goal=(106.14,10.)
    controller,sim,runtime=controller_for(recorded_geometry,position,[anchor,(108.,10.)],old_goal=old_goal,
        visible=lambda area,floor,p,q,now:tuple(q)!=old_goal)
    controller.update_navigation(runtime,0.)
    assert runtime['goal']==anchor and not runtime['navigation_blocked']
    assert sim.agent(runtime['jid']).stage_id==runtime['goal_stage']
