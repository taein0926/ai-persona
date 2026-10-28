from copy import deepcopy
import pytest
from persona_evac.schemas import PersonaProfile
from persona_evac.llm.mapping_evidence import validate_mapping_evidence,evidence_schema
from persona_evac.integration.station_dynamics import can_intentionally_overtake


def payload():
    return {'response_tendency':'normal','speed_tendency':'fast',
            'reasoning_summary':{'response_tendency':'비상 대응 정보가 없어 기본값 보통','speed_tendency':'본인이 매일 달리는 운동 습관을 반영'},
            'evidence':{'response_tendency':{'status':'insufficient','quotes':[]},
                        'speed_tendency':{'status':'supported','quotes':[{'field':'sports_persona','text':'매일 달리기를 합니다'}]}},
            'overtaking':{'disposition':'urgent','reason':'기다림을 싫어하는 성격을 반영',
                          'quotes':[{'field':'persona','text':'기다리는 것을 싫어합니다'}]}}


def profile():
    return PersonaProfile(uuid='test',age=70,sex='female',persona='기다리는 것을 싫어합니다.',sports_persona='매일 달리기를 합니다.')


def test_literal_evidence_survives_and_age_sex_do_not_override_activity():
    m=validate_mapping_evidence(payload(),profile())
    assert m.speed_tendency=='fast' and m.overtaking.disposition=='urgent'
    p=payload();p['evidence']['speed_tendency']['quotes'][0]['text']='매일  달리기를 합니다'
    assert validate_mapping_evidence(p,profile()).evidence.speed_tendency.quotes[0].text=='매일 달리기를 합니다'


def test_invented_quote_and_unsubstantiated_default_rejected():
    p=payload();p['evidence']['speed_tendency']['quotes'][0]['text']='마라톤 선수입니다'
    with pytest.raises(ValueError,match='원문'):validate_mapping_evidence(p,profile())
    p=payload();p['evidence']['speed_tendency']={'status':'insufficient','quotes':[]}
    with pytest.raises(ValueError,match='기본값'):validate_mapping_evidence(p,profile())


def test_family_conflict_does_not_substantiate_emergency_response():
    p=payload();p['evidence']['response_tendency']={'status':'supported','quotes':[{'field':'persona','text':'가족 갈등을 조율합니다'}]}
    person=profile();person.persona+=' 가족 갈등을 조율합니다.'
    with pytest.raises(ValueError,match='일상 갈등'):validate_mapping_evidence(p,person)


@pytest.mark.parametrize('disposition,speed,allowed',[
    ('relaxed',1.8,False),('unknown',1.8,False),('urgent',1.0,False),('urgent',1.8,True)])
def test_intentional_overtake_requires_evidenced_urgency_and_fast_speed(disposition,speed,allowed):
    a={'mapping':{'overtaking':{'disposition':disposition,'quotes':[{'field':'persona','text':'급한 성격'}]}}}
    assert can_intentionally_overtake(a,speed,1.2782644) is allowed
    a['mapping']['overtaking']['quotes']=[]
    assert not can_intentionally_overtake(a,1.8,1.2782644)


def test_ordinary_fast_sport_is_not_evidence_of_impatience():
    p=payload();p['overtaking']['quotes']=deepcopy(p['evidence']['speed_tendency']['quotes'])
    with pytest.raises(ValueError,match='급함'):validate_mapping_evidence(p,profile())


def test_schema_restricts_absent_emergency_evidence_to_declared_default():
    s=evidence_schema(profile())
    assert s['properties']['response_tendency']['enum']==['normal']
    e=s['$defs']['BehaviorEvidence']['properties']['response_tendency']['properties']
    assert e['status']['enum']==['insufficient'] and e['quotes']['maxItems']==0
