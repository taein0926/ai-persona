import math
from types import SimpleNamespace

import jupedsim as jps
import pytest
from shapely.geometry import box

from persona_evac.integration.private_agents import OccludedSensor, PrivateController, VisibleSimulation
from persona_evac.integration.station import StationRequest, simulate_station
from persona_evac.integration.station_dynamics import (
    make_simulation, parameters, plan_yield, visible_motion_target,
    record_visibility_block, pass_attempt, advance_pass_attempt,
    start_manoeuvre, update_manoeuvre, finish_manoeuvre,
)
from station_fixtures import config, source


def scenario(other_position=None):
    geometry=box(0,0,12,8)
    sim=make_simulation(geometry)
    goal=(9.,3.)
    stage=sim.add_waypoint_stage(goal,.1)
    journey=sim.add_journey(jps.JourneyDescription([stage]))
    own=sim.add_agent(parameters((3.,3.),journey,stage,1.2))
    visible={own}
    if other_position is not None:
        visible.add(sim.add_agent(parameters(other_position,journey,stage,0.)))
    helper=VisibleSimulation(sim,visible)
    runtime={'jid':own,'area':'hall','floor':'B2','pos':(3.,3.),'goal':goal,'goal_stage':stage,'passing':False}
    return sim,geometry,helper,runtime


@pytest.mark.parametrize('other',[None,(2.,3.),(3.,4.),(3.6,3.9),(5.5,3.)])
def test_nearby_people_outside_the_forward_path_do_not_trigger_recovery(other):
    sim,geometry,helper,runtime=scenario(other)
    assert plan_yield(helper,geometry,runtime['jid'],runtime['goal'],runtime['goal_stage']) is None
    assert sim.agent(runtime['jid']).stage_id==runtime['goal_stage']
    assert helper.waypoint_positions=={}


def test_visible_person_blocking_the_forward_path_allows_a_local_route():
    sim,geometry,helper,runtime=scenario((4.,3.))
    route=plan_yield(helper,geometry,runtime['jid'],runtime['goal'],runtime['goal_stage'])
    assert route
    assert set(helper.waypoint_positions.values())==set(route)
    assert helper.waypoint_positions[sim.agent(runtime['jid']).stage_id]==route[0]
    assert tuple(sim.agent(runtime['jid']).position)==runtime['pos']


def test_smoke_without_a_visible_route_does_not_attempt_crowd_recovery(monkeypatch):
    monkeypatch.setattr(OccludedSensor,'visible',lambda self,area,floor,p,q,now:math.dist(p,q)<.1)
    def forbidden(*args,**kwargs):
        raise AssertionError('시야 차단으로 멈춘 사람에게 국소 우회를 시도했습니다.')
    monkeypatch.setattr(PrivateController,'visible_helper',forbidden)
    result=simulate_station(source(),config(),StationRequest(agent_count=10,behavior_mode='off',duration_s=5))
    assert result['movement_diagnostics']['counterflow_recoveries']==[]
    assert all(s['from']==s['to'] for a in result['agents'] for s in a['segments'])


def test_each_active_waypoint_is_checked_and_lost_visibility_cancels_motion():
    sim,geometry,helper,runtime=scenario((4.,3.))
    route=plan_yield(helper,geometry,runtime['jid'],runtime['goal'],runtime['goal_stage'])
    runtime.update(passing=True,manoeuvre_waypoints=helper.waypoint_positions)
    checks=[]
    def visible(area,floor,p,q,now):
        checks.append((now,tuple(q)))
        return now<1.
    sensor=SimpleNamespace(visible=visible)
    assert visible_motion_target(sim,runtime,sensor,0.)
    assert checks[-1][1]==route[0]
    second=list(helper.waypoint_positions)[1]
    journey=sim.add_journey(jps.JourneyDescription([second,runtime['goal_stage']]))
    sim.switch_agent_journey(runtime['jid'],journey,second)
    assert visible_motion_target(sim,runtime,sensor,.05)
    assert checks[-1][1]==route[1]
    assert not visible_motion_target(sim,runtime,sensor,1.)
    assert not runtime['passing']
    assert 'manoeuvre_waypoints' not in runtime
    assert runtime['navigation_blocked']=='local_manoeuvre_not_currently_visible'
    assert sim.agent(runtime['jid']).stage_id==runtime['goal_stage']
    assert sim.agent(runtime['jid']).model.desired_speed==0.
    previous=tuple(sim.agent(runtime['jid']).position)
    for _ in range(10):
        sim.iterate()
    assert math.dist(sim.agent(runtime['jid']).position,previous)<1e-8


def test_return_to_the_original_goal_removes_the_local_route_state():
    sim,_,_,runtime=scenario()
    runtime.update(passing=True,manoeuvre_waypoints={})
    sensor=SimpleNamespace(visible=lambda area,floor,p,q,now:tuple(q)==runtime['goal'])
    assert visible_motion_target(sim,runtime,sensor,2.)
    assert not runtime['passing']
    assert 'manoeuvre_waypoints' not in runtime


def test_untracked_waypoint_cannot_bypass_visibility():
    sim,_,_,runtime=scenario()
    different=sim.add_waypoint_stage((3.,4.),.1)
    sim.switch_agent_journey(runtime['jid'],sim.add_journey(jps.JourneyDescription([different])),different)
    runtime['passing']=True
    sensor=SimpleNamespace(visible=lambda *args:True)
    assert not visible_motion_target(sim,runtime,sensor,2.)
    assert sim.agent(runtime['jid']).model.desired_speed==0.


def test_visibility_history_records_recovery_without_repeating_every_step():
    agent={}
    for time,blocked in [(0,False),(.05,False),(1,True),(1.05,True),(2,False)]:
        record_visibility_block(agent,time,blocked)
    assert agent['visibility_blocked_history']==[
        {'time_s':0,'blocked':False},{'time_s':1,'blocked':True},{'time_s':2,'blocked':False}]


def passers():
    common={'area':'hall','floor':'B2','phase':'walking','target':'exit','goal':(8.,3.)}
    return ({**common,'agent':{'agent_id':1},'pos':(3.,3.)},
            {**common,'agent':{'agent_id':2},'pos':(4.,3.)})


def test_local_recovery_counts_as_overtaking_only_after_a_clear_physical_pass():
    passer,front=passers()
    attempt=pass_attempt(passer,front,0.,'local_recovery')
    assert advance_pass_attempt(attempt,passer,front,.1) is None
    passer['pos']=(4.,3.65)
    assert advance_pass_attempt(attempt,passer,front,1.) is None
    passer['pos']=(4.6,3.65)
    completed=advance_pass_attempt(attempt,passer,front,2.)
    assert completed['source']=='local_recovery'
    assert completed['passed_agent']==2


def test_sideways_or_reverse_movement_is_not_an_overtake():
    passer,front=passers()
    attempt=pass_attempt(passer,front,0.,'local_recovery')
    for position in [(3.,3.65),(2.8,3.65),(3.,3.)]:
        passer['pos']=position
        assert advance_pass_attempt(attempt,passer,front,1.) is None
    assert advance_pass_attempt(attempt,passer,front,26.) is False


def test_crossing_opposite_routes_is_not_overtaking():
    passer,front=passers()
    front['target']='other_exit'
    assert pass_attempt(passer,front,0.,'local_recovery') is None


@pytest.mark.parametrize('action',['yielding','overtaking','route_recovery'])
def test_manoeuvre_records_its_execution_interval_without_a_speed_condition(action):
    runtime={'agent':{},'phase':'walking','passing':True}
    event={'time_s':1.}
    start_manoeuvre(runtime,event,action,1.)
    update_manoeuvre(runtime,1.5)
    assert event=={'time_s':1.,'action':action,'end_s':None}
    runtime['passing']=False
    update_manoeuvre(runtime,2.)
    assert event['end_s']==2.
    assert 'manoeuvre_event' not in runtime
    finish_manoeuvre(runtime,3.)
    assert event['end_s']==2.


@pytest.mark.parametrize('phase,incapacitated',[('on_stairs',None),('queueing',None),('evacuated',None),('walking',2.)])
def test_leaving_the_walking_route_ends_the_manoeuvre(phase,incapacitated):
    runtime={'agent':{'incapacitated_at_s':incapacitated},'phase':phase,'passing':True}
    event={'time_s':1.}
    start_manoeuvre(runtime,event,'yielding',1.)
    update_manoeuvre(runtime,2.)
    assert event['end_s']==2.


def test_replacing_a_manoeuvre_closes_the_previous_interval():
    runtime={'agent':{},'phase':'walking','passing':True}
    previous={'time_s':1.};current={'time_s':2.}
    start_manoeuvre(runtime,previous,'yielding',1.)
    start_manoeuvre(runtime,current,'overtaking',2.)
    assert previous['end_s']==2.
    assert current['end_s'] is None
    finish_manoeuvre(runtime,3.)
    assert current['end_s']==3.
