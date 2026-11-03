from copy import deepcopy

import pytest

from shapely.geometry import box

from persona_evac.integration.personal_space import (
    PersonalSpace,OccludedSensor,station_stair_memories,station_route_memories,
)


def station():
    areas={'paid':{'geometry':box(0,0,8,8)},'hall':{'geometry':box(9,0,18,8)}}
    nodes={'in':{'area':'paid','floor':'B1','x':7.2,'y':4},
           'out':{'area':'hall','floor':'B1','x':9.8,'y':4},
           'stair':{'area':'hall','floor':'B1','x':16,'y':4},
           'surface':{'floor':'OUTSIDE','x':16,'y':6}}
    stairs={'G':{'kind':'fare_gate','from_node':'in','to_node':'out'},
            'S':{'kind':'stairs','from_node':'stair','to_node':'surface'}}
    return areas,nodes,stairs


def space(level,prior,maps):
    result=PersonalSpace({'uuid':level},{'familiarity':{'level':level,'assumption':'실험 분류'}},prior)
    if level=='familiar':result.prior_maps=maps
    return result


def test_familiar_person_remembers_gate_and_routes_without_live_conditions():
    areas,nodes,stairs=station();prior=station_stair_memories(nodes,stairs)
    maps=station_route_memories(areas,prior)
    familiar=space('familiar',prior,maps);new=space('first_visit',prior,maps)
    familiar.current_origin_links=[(2,2)]
    options=familiar.options('paid',(3,3),1.,0.)
    assert [o['id'] for o in options]==['G']
    assert options[0]['kind']=='fare_gate' and options[0]['known_to_floor']=='B1'
    assert options[0]['path'] and not options[0]['state_is_current']
    assert options[0]['hazard'] is None and options[0]['visible_people'] is None
    assert familiar.maps['paid']['initial_source']=='remembered_station_structure'
    assert new.initial_knowledge=={} and new.maps=={}
    assert new.options('paid',(3,3),1.,0.)==[]


def test_familiar_localization_does_not_require_seeing_a_two_metre_grid_point():
    areas,nodes,stairs=station();prior=station_stair_memories(nodes,stairs)
    familiar=space('familiar',prior,station_route_memories(areas,prior))
    class Smoke:
        def sample(self,*args):return {'extinction_per_m':3.1,'temperature_c':20}
    sensor=OccludedSensor(areas,nodes,stairs,Smoke())
    runtime={'area':'paid','floor':'B1','pos':(3,3),'phase':'walking'}
    sensor.sense(familiar,runtime,{1:runtime},0.)
    assert not sensor.visible('paid','B1',(3,3),(4,4),0.)
    assert familiar.current_origin_links
    options=familiar.options('paid',(3,3),1.,0.)
    assert [o['id'] for o in options]==['G']
    assert options[0]['hazard'] is None
    assert not familiar.familiarity['current_conditions_known']


def test_private_maps_do_not_mutate_shared_structure_or_another_person():
    areas,nodes,stairs=station();prior=station_stair_memories(nodes,stairs)
    maps=station_route_memories(areas,prior);original=deepcopy(maps)
    one=space('familiar',prior,maps);two=space('familiar',prior,maps)
    one.map_for('paid')['cells'].clear()
    assert maps==original and two.map_for('paid')['cells']==original['paid']['cells']


def test_first_visitor_keeps_only_previously_observed_route_when_smoke_hides_grid():
    areas,nodes,stairs=station();prior=station_stair_memories(nodes,stairs)
    visitor=space('first_visit',prior,{})
    class Smoke:
        def sample(self,floor,p,now):return {'extinction_per_m':0. if now==0 else 3.1,'temperature_c':20}
    sensor=OccludedSensor(areas,nodes,stairs,Smoke())
    runtime={'area':'paid','floor':'B1','pos':(3,3),'phase':'walking'}
    assert visitor.initial_knowledge=={} and visitor.prior_maps=={}
    sensor.sense(visitor,runtime,{1:runtime},0.)
    remembered=set(visitor.map_for('paid')['cells'])
    assert any(p['id']=='G' for p in visitor.options('paid',(3,3),1.,0.))
    sensor.sense(visitor,runtime,{1:runtime},1.)
    assert not sensor.visible('paid','B1',(3,3),(4,4),1.)
    assert visitor.current_origin_links
    assert set(visitor.current_origin_links)<=remembered
    assert remembered<=visitor.map_for('paid')['cells']
    assert all(sensor.visible('paid','B1',(3,3),(k[0]*2,k[1]*2),1.)
        for k in visitor.map_for('paid')['cells']-remembered)
    assert visitor.prior_maps=={}
    assert any(p['id']=='G' for p in visitor.options('paid',(3,3),1.,1.))


@pytest.mark.parametrize('kind',['stairs','fare_gate'])
def test_first_visitor_uses_discovered_reachable_connection_before_more_exploration(kind):
    areas,nodes,stairs=station();prior=station_stair_memories(nodes,stairs)
    areas['hall']['geometry']=box(9,0,100,8)
    stairs['S']['kind']=kind
    if kind=='fare_gate':nodes['surface']['floor']='B1'
    visitor=space('first_visit',prior,{})
    sensor=OccludedSensor(areas,nodes,stairs)
    runtime={'area':'hall','floor':'B1','pos':(12,4),'phase':'walking'}
    assert visitor.options('hall',(12,4),1.,0.)==[]
    sensor.sense(visitor,runtime,{1:runtime},0.)
    options=visitor.options('hall',(12,4),1.,0.)
    assert visitor.initial_knowledge=={} and visitor.prior_maps=={}
    assert visitor.landmarks['S']['source']=='vision'
    assert [p['id'] for p in options]==['S']
    assert options[0]['kind']==('fare_gate' if kind=='fare_gate' else 'up_stair') and options[0]['path']
    # A remembered entrance without any route in the person's map is not a usable ascent.
    visitor.map_for('hall')['landmark_links']['S']=[]
    visitor.direct_landmark_links={};visitor._shortest_cache=None
    options=visitor.options('hall',(12,4),1.,0.)
    assert options and all(p['kind']=='frontier' for p in options)
