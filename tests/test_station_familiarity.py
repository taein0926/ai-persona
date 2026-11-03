from copy import deepcopy

import pytest

from persona_evac.integration.personal_space import PersonalSpace, familiarity_basis, station_stair_memories
from persona_evac.integration.private_agents import PrivateMind
from persona_evac.persona.experimental_profile import build_profile
from persona_evac.llm.private_decisions import project_private
from test_private_agents import sensor_fixture


def mapping(level):
    return {'familiarity':{'level':level,'assumption':'생활 이동 문맥에 따라 분류했습니다.'}}


def structure():
    nodes={'a':{'area':'A','floor':'B2','x':2,'y':2},
           'hall':{'area':'hall','floor':'B1','x':8,'y':2},
           'out':{'floor':'OUTSIDE','x':12,'y':2}}
    stairs={'S':{'from_node':'a','to_node':'hall'},
            'E':{'from_node':'hall','to_node':'out'},
            'G':{'from_node':'hall','to_node':'hall','kind':'fare_gate'}}
    return nodes,stairs


@pytest.mark.parametrize('level',['first_visit','familiar'])
def test_llm_grade_controls_static_stair_memory_without_current_information(level):
    nodes,stairs=structure();prior=station_stair_memories(nodes,stairs)
    persona={'uuid':'person','province':'서울','district':'서울-강서구'}
    space=PersonalSpace(persona,mapping(level),prior)
    assert space.familiarity['level']==level
    assert space.familiarity['classification_source']=='llm'
    assert not space.familiarity['current_conditions_known']
    assert space.maps=={} and space.observations==[]
    if level=='first_visit':
        assert space.initial_knowledge=={}
    else:
        assert set(space.initial_knowledge)=={'S','E','G'}
        assert space.initial_knowledge['S']['position']==[2,2]
        assert space.initial_knowledge['S']['to_floor']=='B1'
        assert space.initial_knowledge['E']['to_floor']=='OUTSIDE'
        assert all(k['current_state'] is None and k['observed_at_s'] is None for k in space.initial_knowledge.values())
        # 구조 지도를 공급하지 않은 단위 구성에서는 경로를 만들지 않는다.
        assert space.options('A',(0,0),1,0)==[]


def test_personal_mind_uses_agent_mapping_and_supplied_station_geometry():
    nodes,stairs=structure()
    agent={'persona':{'uuid':'person'},'mapping':mapping('familiar')}
    mind=PrivateMind(agent,build_profile(agent['persona']),station_stair_memories(nodes,stairs))
    assert set(mind.space.initial_knowledge)=={'S','E','G'}


def test_familiarity_summary_stays_outside_situation_llm_but_grade_and_connections_enter():
    _,_,_,observation,_=sensor_fixture()
    observation['familiarity']=familiarity_basis({'uuid':'p'},mapping('familiar'))
    nodes,stairs=structure();space=PersonalSpace({'uuid':'p'},mapping('familiar'),station_stair_memories(nodes,stairs))
    observation['private_knowledge']=space.snapshot('A',0)
    projected=project_private(observation)
    assert projected['familiarity']=={'level':'familiar'}
    assert len(projected['known_connections'])==3
    changed=deepcopy(observation);changed['familiarity']['assumption']='새 요약'
    assert project_private(changed)==projected
