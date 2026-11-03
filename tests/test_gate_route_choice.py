from copy import deepcopy
import json
from pathlib import Path

import pytest

from persona_evac.integration.private_strategy import resolve_private


CASES=json.loads(Path('tests/fixtures/recorded_gate_decisions.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('case',CASES,ids=lambda p:str(p['agent_id']))
def test_recorded_smoky_gate_is_not_rejected_for_an_assumed_clear_unknown_route(case):
    observation=deepcopy(case['observation']);choice=case['choice']
    target,rows=resolve_private(observation,choice)
    option=next(p for p in observation['options'] if p['id']==target)
    assert option['kind']=='fare_gate'
    assert observation['profile']['traits']['risk_avoidance']['level']=='high'
    assert any(row['unknown_smoke_cost']>0 for row in rows)
    assert all(p['hazard'] is None for p in observation['options'] if p['kind']=='frontier')
    assert next(row for row in rows if row['target_id']==target)['hazard_cost']>0


def test_observed_heat_is_compared_between_known_gate_alternatives():
    case=deepcopy(CASES[0]);observation=case['observation']
    gate=next(option for option in observation['options'] if option['kind']=='fare_gate')
    cooler=deepcopy(gate);cooler['id']='observed_cooler_gate';cooler['hazard']['heat']='normal'
    gate['hazard']['heat']='hot';observation['options']=[gate,cooler]
    target,rows=resolve_private(observation,case['choice'])
    assert target=='observed_cooler_gate'
    assert next(row for row in rows if row['target_id']==target)['hazard_cost']<next(row for row in rows if row['target_id']==gate['id'])['hazard_cost']
    assert all(row['unknown_smoke_cost']>=0 for row in rows)


def test_recorded_agent58_does_not_treat_unexplored_space_as_cooler_than_own_location():
    case=json.loads(Path('tests/fixtures/recorded_agent58_stair_decision.json').read_text(encoding='utf-8'))
    observation=case['observation']
    target,rows=resolve_private(observation,case['choice'])
    assert observation['local_hazard']=={'smoke':'opaque','heat':'hot'}
    assert target=='A_W'
    stair=next(row for row in rows if row['target_id']=='A_W')
    assert stair['hazard_cost']==180 and stair['unknown_heat_cost']==0
    assert all(row['unknown_heat_cost']==90 for row in rows if row['target_id'].startswith('explore:'))
    assert all(p['hazard'] is None for p in observation['options'] if p['kind']=='frontier')
