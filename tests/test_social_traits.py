from copy import deepcopy
from types import SimpleNamespace

import pytest

from persona_evac.persona.full_profile import build_full_profile
from persona_evac.integration.private_strategy import resolve_private
from persona_evac.integration.private_agents import PrivateController
from persona_evac.llm.private_decisions import private_schema, project_private, validate_private
from test_full_profile import estimates
from test_private_agents import sensor_fixture, default_choice


def observation():
    _,_,_,o,_=sensor_fixture()
    o['profile']=build_full_profile({'uuid':'social'},estimates(),1,{'name':'fixture','digest':'fixture'})
    base=deepcopy(o['options'][0]);base.update(hazard={'smoke':'clear','heat':'normal'},past_state=None,visible_people=0)
    o['options']=[{**base,'id':'near','walking_s':3,'observed_flow':None},
                  {**base,'id':'flow','walking_s':9,'observed_flow':{'people':3,'strength':1,
                    'observed_at_s':o['time_s'],'source':'own_observed_movement','observation_id':'V2'}}]
    return o


def test_following_grade_changes_known_route_preference_and_keeps_hazard_costs():
    o=observation();choice=default_choice(o)
    o['profile']['traits']['social_following']['level']='low'
    assert resolve_private(o,choice)[0]=='near'
    o['profile']['traits']['social_following']['level']='high'
    target,rows=resolve_private(o,choice)
    assert target=='flow' and rows[1]['following_bonus']>rows[0]['following_bonus']
    o['options'][1]['hazard']={'smoke':'opaque','heat':'hot'}
    assert resolve_private(o,choice)[0]=='near'


@pytest.mark.parametrize('change',[{'observed_at_s':-10},{'source':'other_agent_plan'},{'strength':0}])
def test_stale_unobserved_or_zero_flow_does_not_attract(change):
    o=observation();o['profile']['traits']['social_following']['level']='high'
    o['options'][1]['observed_flow'].update(change)
    assert resolve_private(o,default_choice(o))[0]=='near'


def test_follow_strategy_and_level_enter_situation_input_without_profile_summary():
    o=observation();projected=project_private(o)
    assert 'follow' in private_schema([o])['properties']['1']['properties']['priority']['enum']
    assert projected['options'][1]['observed_flow']['people']==3
    assert set(projected['profile']['traits']['social_following'])=={'level'}
    result=validate_private({'1':{'priority':'follow','movement':'continue','memory':[],'observations':[]}},[o])[0]
    assert result['used_trait_refs']==['social_following']
    o['options'][1]['observed_flow']=None
    assert 'follow' not in private_schema([o])['properties']['1']['properties']['priority']['enum']
    with pytest.raises(ValueError,match='관측'):
        validate_private({'1':{'priority':'follow','movement':'continue','memory':[],'observations':[]}},[o])


@pytest.mark.parametrize('level',['low','normal','high'])
def test_no_overtaking_option_is_offered_for_any_trait_level(level):
    _,space,r,_,_=sensor_fixture()
    data=estimates();data['overtaking_tendency']['level']=level
    profile=build_full_profile({'uuid':'passing'},data,1,{'name':'x','digest':'x'})
    controller=PrivateController.__new__(PrivateController)
    controller.minds={1:SimpleNamespace(space=space,profile=profile,memory=lambda *_:[])}
    r.update(agent={'agent_id':1,'desired_speed_mps':2.,'distance_walked_m':2.},
             target='explore:none',wait_streak=0)
    o=controller.observation(r,1)
    assert 'overtake' not in o['allowed_movements']
    assert o['front_observation']=={'slower_walker_and_clear_side_observed':False}
    assert 'overtake' not in private_schema([o])['properties']['1']['properties']['movement']['enum']
    with pytest.raises(ValueError,match='허용되지 않은'):
        validate_private({'1':{'priority':'environment','movement':'overtake','memory':[],'observations':[]}},[o])


@pytest.mark.parametrize('level,expected',[('low',18),('normal',8),('high',0)])
def test_waiting_patience_still_penalizes_current_queue_and_honors_priority(level,expected):
    o=observation();o['current_target']='near';o['queue_wait_s']=12
    o['profile']['traits']['waiting_patience']['level']=level
    choice={'priority':'environment','movement':'continue','memory':[]}
    _,rows=resolve_private(o,choice)
    assert rows[0]['own_wait_cost']==expected
    assert rows[1]['own_wait_cost']==0
    _,rows=resolve_private(o,{**choice,'priority':'patience'})
    assert rows[0]['own_wait_cost']==expected*1.5


def test_flow_uses_only_current_two_frame_visual_motion():
    sensor,space,r,_,_=sensor_fixture()
    people={1:r,**{i:{'area':'test','floor':'B2','phase':'walking','pos':(4+i/10,2),
                     'target':'HIDDEN_DESTINATION','observed_speed_mps':99} for i in (2,3,4)}}
    option={'position':[12,2],'path':[[4,2],[12,2]]}
    sensor.sense(space,r,people,2)
    assert space.observed_flow('test',r['pos'],2,option) is None
    for person in list(people.values())[1:]:person['pos']=(person['pos'][0]+1,2)
    sensor.sense(space,r,people,3)
    flow=space.observed_flow('test',r['pos'],3,option)
    assert flow['people']==3 and flow['strength']==1
    for person in list(people.values())[1:]:person['target']='DIFFERENT_HIDDEN_PLAN'
    assert space.observed_flow('test',r['pos'],3,option)==flow
    assert space.observed_flow('test',r['pos'],7,option) is None
    assert space.observed_flow('test',r['pos'],3,{'position':[-12,2],'path':[[-12,2]]})['strength']==0
