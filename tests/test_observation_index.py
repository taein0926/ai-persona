"""관측 ID 직접 조회의 기록 순서·갱신·개인 분리·판단 입력 일치를 검사한다."""
import json
from types import SimpleNamespace

from persona_evac.integration.personal_space import PersonalSpace
from persona_evac.integration.private_agents import PrivateController


def make_space(uuid='index-test'):
    return PersonalSpace({'uuid':uuid,'province':'부산','district':'부산-강서구'})


def test_lookup_preserves_insertion_order_references_and_missing_id_behavior():
    space=make_space()
    for i in range(12):
        space.record(12-i,'local_hazard',{'value':i})
    ids=['O10','missing','O2','O10','O12','O1']
    before=json.dumps(space.observations,ensure_ascii=False)
    expected=[o for o in space.observations if o['id'] in set(ids)]
    actual=space.observations_for_ids(ids)
    assert actual==expected
    assert [o['id'] for o in actual]==['O1','O2','O10','O12']
    assert all(a is b for a,b in zip(actual,expected))
    assert json.dumps(space.observations,ensure_ascii=False)==before
    assert space.observations_for_ids([])==[]


def test_duplicate_ids_keep_all_records_and_personal_indices_are_separate():
    first=make_space('one');second=make_space('two')
    first.record(1,'local_hazard',{'id':'shared','value':'first'})
    first.record(2,'local_hazard',{'value':'one'})
    first.record(3,'local_hazard',{'id':'shared','value':'last'})
    second.record(1,'local_hazard',{'id':'shared','value':'other-person'})
    assert [o['value'] for o in first.observations_for_ids(['shared'])]==['first','last']
    assert [o['value'] for o in second.observations_for_ids(['shared'])]==['other-person']


def test_repeated_lookup_and_new_record_do_not_rescan_the_history():
    class CountedHistory(list):
        iterations=0
        def __iter__(self):
            self.iterations+=1
            return super().__iter__()
    space=make_space()
    for i in range(12):space.record(i,'local_hazard',{'value':i})
    space.observations=CountedHistory(space.observations)
    assert space.observations_for_ids(['O2'])[0]['value']==1
    space.observations.iterations=0
    for _ in range(20):
        assert [o['id'] for o in space.observations_for_ids(['O10','O2'])]==['O2','O10']
    space.record(13,'local_hazard',{'value':'new'})
    assert space.observations_for_ids(['O13'])[0]['value']=='new'
    assert space.observations.iterations==0


def test_restored_replaced_and_cleared_histories_rebuild_the_index():
    space=make_space()
    space.record(1,'local_hazard',{'value':'old'})
    space.observations=[{'id':'restored','value':'restored'}]
    assert space.observations_for_ids(['O1','restored'])==space.observations
    space.observations=[{'id':'replacement','value':'replacement'}]
    assert space.observations_for_ids(['restored','replacement'])==space.observations
    space.observations.append({'id':'appended','value':'appended'})
    assert space.observations_for_ids(['appended'])[0]['value']=='appended'
    space.observations.clear()
    assert space.observations_for_ids(['replacement','appended'])==[]
    assert space.record(4,'local_hazard',{'value':'fresh'})=='O1'
    assert space.observations_for_ids(['O1'])[0]['value']=='fresh'


def test_controller_context_is_identical_to_history_scanning(monkeypatch):
    space=make_space()
    for i in range(100):
        space.record(i/5,'local_hazard' if i%3==0 else 'landmark_seen',{'value':i})
    space.last_observation_ids=['O98','O2','O96','O97','O99','O95','O100','O98']
    options=[{'observation_ids':['O1','O20','O97']},{'observation_ids':['O7','O50','O99']}]
    monkeypatch.setattr(space,'options',lambda *_:options)
    controller=PrivateController.__new__(PrivateController)
    controller.minds={1:SimpleNamespace(space=space,profile={},memory=lambda *_:[])}
    r={'agent':{'agent_id':1,'desired_speed_mps':1,'distance_walked_m':0},
       'area':'room','floor':'B2','pos':(2,2),'target':'explore:none','phase':'walking','wait_streak':0}
    def scan(ids):
        wanted=set(ids)
        return [o for o in space.observations if o['id'] in wanted]
    with monkeypatch.context() as patch:
        patch.setattr(space,'observations_for_ids',scan)
        expected=controller.observation(r,20)
    actual=controller.observation(r,20)
    assert actual==expected
    assert json.dumps(actual,ensure_ascii=False)==json.dumps(expected,ensure_ascii=False)
