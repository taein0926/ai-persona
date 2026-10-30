import json
from copy import deepcopy

import pytest

from persona_evac.llm.baseline_profile import SCHEMA, validate_mapping
from persona_evac.llm.ollama import OllamaProvider
from persona_evac.schemas import PersonaProfile
from test_ollama import client


def payload():
    return {'response_tendency':{'level':'fast','assumption':'전체 문맥에서 신속한 반응을 추정합니다.'},
            'speed_tendency':{'level':'slow','assumption':'전체 생활 서술에서 느린 보행을 추정합니다.'},
            'familiarity':{'level':'familiar','assumption':'생활과 이동 문맥에서 역 구조에 익숙한 것으로 판단했습니다.'}}


def test_no_keyword_guard_and_full_persona_reaches_llm(tmp_path,monkeypatch):
    monkeypatch.setenv('OLLAMA_MODEL','qwen3:4b')
    monkeypatch.setenv('OLLAMA_BASE_URL','http://localhost:11434')
    calls=[]
    def generate(body):
        calls.append(body)
        assert body['format']==SCHEMA
        for schema in body['format']['$defs'].values():
            assert set(schema['properties'])=={'level','assumption'}
            assert schema['properties']['level']['enum'] in (['fast','normal','slow'],['first_visit','familiar'])
            assert schema['additionalProperties'] is False
        data=json.loads(body['messages'][1]['content'])
        assert data['persona']['culinary_persona']=='요리와 식생활의 전체 원문'
        assert 'uuid' not in data['persona'] and 'original_evidence' not in data
        return payload()
    provider=OllamaProvider(tmp_path,client(generate))
    person=PersonaProfile(uuid='all-fields',age=70,persona='차를 마십니다.',culinary_persona='요리와 식생활의 전체 원문')
    mapping=provider.map(person,{'mapping_policy':'sports_age_sex_urgency_evidence_v2'})
    assert mapping.response_tendency=='fast' and mapping.speed_tendency=='slow'
    assert mapping.evidence is None and mapping.overtaking is None
    assert mapping.familiarity.level=='familiar'
    assert mapping.reasoning_summary.response_tendency==payload()['response_tendency']['assumption']
    assert provider.map(person,{'mapping_policy':'sports_age_sex_urgency_evidence_v2'})==mapping
    assert len(calls)==1


@pytest.mark.parametrize('level',['first_visit','familiar'])
def test_familiarity_is_a_required_two_field_llm_classification(level):
    data=payload();data['familiarity']['level']=level
    assert validate_mapping(data).familiarity.level==level
    data['familiarity']['level']='unknown'
    with pytest.raises(ValueError):validate_mapping(data)


@pytest.mark.parametrize('change',[{'level':'invalid'},{'level':1},{'assumption':''},{'assumption':'   '},
    {'quotes':[]},{'source_fields':[]},{'evidence_level':'direct'},{'information_insufficient':False}])
def test_only_valid_levels_and_nonempty_summary_are_accepted(change):
    data=payload();data['response_tendency'].update(change)
    with pytest.raises(ValueError):validate_mapping(data)


def test_old_flat_response_and_extra_output_are_rejected():
    with pytest.raises(ValueError):validate_mapping({'response_tendency':'fast','speed_tendency':'slow'})
    data=payload();data['overtaking']={'disposition':'urgent'}
    with pytest.raises(ValueError):validate_mapping(data)
    for key in payload():
        data=payload();del data[key]
        with pytest.raises(ValueError):validate_mapping(data)


def test_failed_classification_is_retried_without_normal_fallback(tmp_path,monkeypatch):
    monkeypatch.setenv('OLLAMA_MODEL','qwen3:4b')
    monkeypatch.setenv('OLLAMA_BASE_URL','http://localhost:11434')
    calls=[]
    def invalid(body):
        calls.append(body);data=deepcopy(payload());data['response_tendency']['assumption']=' '
        return data
    provider=OllamaProvider(tmp_path,client(invalid))
    with pytest.raises(ValueError):provider.map(PersonaProfile(uuid='failure',age=40),{})
    assert len(calls)==3
    assert not [p for p in tmp_path.glob('*.json') if not p.name.endswith('.invalid.json')]
