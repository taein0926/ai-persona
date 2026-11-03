import math

import jupedsim as jps
from shapely.geometry import Point,box

from persona_evac.integration.personal_space import PersonalSpace,OccludedSensor,cell_pos
from persona_evac.integration.private_agents import PrivateController,PrivateMind
from persona_evac.integration.station_dynamics import make_simulation,parameters,visible_motion_target


class Smoke:
    def sample(self,*args):return {'extinction_per_m':3.1,'temperature_c':20}


def test_first_visitor_learns_only_visible_subgrid_cells():
    space=PersonalSpace({'uuid':'observer'},{'familiarity':{'level':'first_visit','assumption':'초행'}})
    areas={'hall':{'geometry':box(0,0,12,6)}}
    sensor=OccludedSensor(areas,{}, {},Smoke())
    runtime={'area':'hall','floor':'B1','pos':(3,3),'phase':'walking'}
    sensor.sense(space,runtime,{1:runtime},0.)
    m=space.maps['hall']
    assert m['adaptive'] and any(any(v!=round(v) for v in k) for k in m['cells'])
    assert all(sensor.visible('hall','B1',(3,3),cell_pos(k),0.) for k in m['cells'])
    assert space.initial_knowledge=={} and space.prior_maps=={}


def test_short_view_corridor_without_any_two_metre_grid_row_reaches_gate():
    geometry=box(0,2.4,12,3.6);sim=make_simulation(geometry);start=(1.,3.)
    nodes={'in':{'area':'hall','floor':'B1','x':11.,'y':3.},
           'out':{'area':'other','floor':'B1','x':13.,'y':3.}}
    stairs={'G':{'from_node':'in','to_node':'out','kind':'fare_gate'}}
    stage=sim.add_waypoint_stage(start,.08);journey=sim.add_journey(jps.JourneyDescription([stage]))
    jid=sim.add_agent(parameters(start,journey,stage,1.))
    agent={'agent_id':1,'persona':{'uuid':'corridor'},'mapping':{'familiarity':{'level':'first_visit','assumption':'초행'}},
           'desired_speed_mps':1.,'distance_walked_m':0.,'stair_wait_s':0.,'decision_wait_s':0.}
    profile={'traits':{name:{'level':level} for name,level in (
        ('crowd_avoidance','normal'),('waiting_patience','high'),('risk_avoidance','high'),('exploration','low'))}}
    mind=PrivateMind(agent,profile)
    runtime={'agent':agent,'jid':jid,'area':'hall','floor':'B1','pos':start,'phase':'walking','goal':start,
        'goal_stage':stage,'goal_is_portal':False,'target':'explore:none','passing':False,'wait_streak':0,'stall_since':0.}
    controller=PrivateController.__new__(PrivateController)
    controller.areas={'hall':{'geometry':geometry,'sim':sim}};controller.runtime={1:runtime};controller.minds={1:mind}
    controller.stairs={};controller.mode='off';controller.navigation_updates=[];controller.local_geom_cache={}
    controller.sensor=OccludedSensor(controller.areas,nodes,stairs,Smoke())
    reached=False
    for tick in range(800):
        now=tick*.05;runtime['pos']=tuple(sim.agent(jid).position)
        controller.tick(now)
        visible=visible_motion_target(sim,runtime,controller.sensor,now)
        sim.agent(jid).model.desired_speed=1. if visible and not runtime.get('navigation_blocked') else 0.
        before=runtime['pos'];sim.iterate();after=tuple(sim.agent(jid).position)
        assert geometry.covers(Point(after)) and math.dist(before,after)<=.051
        agent['distance_walked_m']+=math.dist(before,after)
        if runtime['target']=='G' and math.dist(after,(11,3))<.2:reached=True;break
    assert reached,(after,runtime['target'])
    assert mind.space.landmarks['G']['source']=='vision'
    assert mind.space.prior_maps=={}


def test_two_visible_endpoints_do_not_mark_the_hidden_middle_as_observed_route():
    space=PersonalSpace({'uuid':'hidden-middle'},{'familiarity':{'level':'first_visit','assumption':'초행'}})
    sensor=OccludedSensor({'hall':{'geometry':box(0,0,12,6)}},{},{})
    original=sensor.visible
    sensor.visible=lambda area,floor,p,q,now:math.dist(q,(3,2))>.12 and original(area,floor,p,q,now)
    runtime={'area':'hall','floor':'B1','pos':(3,3),'phase':'walking'}
    sensor.sense(space,runtime,{1:runtime},0.)
    m=space.maps['hall']
    assert (1,1) in m['cells'] and (2,1) in m['cells']
    assert (2,1) not in m['edges'].get((1,1),{})
