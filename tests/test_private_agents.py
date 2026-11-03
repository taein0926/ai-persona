from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
from shapely.geometry import box
from shapely.ops import unary_union

from persona_evac.agent_policy import VERSIONS
from persona_evac.persona.experimental_profile import build_profile,validate_profile
from persona_evac.integration.personal_space import PersonalSpace,OccludedSensor,residence_basis
from persona_evac.integration.private_agents import PrivateMind,PrivateController,default_choice
from persona_evac.integration.private_strategy import resolve_private
from persona_evac.llm.private_decisions import project_private,validate_private


def persona(uuid='test'):
    return {'uuid':uuid,'province':'부산','district':'부산-강서구'}


def test_all_traits_are_declared_reproducible_draws_without_fake_citations():
    p=persona();before=deepcopy(p);first=build_profile(p,seed=123)
    assert first==build_profile(p,seed=123) and p==before
    assert validate_profile(first,p,123)==first
    assert len(first['traits'])==4
    assert all(t['origin']=='seeded_assignment' and not t['evidence_ids'] for t in first['traits'].values())
    population=[build_profile(persona(str(i)),seed=123) for i in range(100)]
    assert all(len({a['traits'][t]['level'] for a in population})==3 for t in first['traits'])
    wrong=deepcopy(first);wrong['traits']['risk_avoidance']['evidence_ids']=['FAKE']
    with pytest.raises(AssertionError):validate_profile(wrong,p,123)


def test_original_quote_and_interpretation_are_separate_from_assignment():
    p={**persona(),'travel_persona':'인파를 피하고 한적한 길을 좋아합니다.'}
    first=build_profile(p,{'crowd_avoidance':{'evidence':'E1','level':'high'}},seed=2)
    assert first['traits']['crowd_avoidance']['origin']=='qwen_interpretation'
    assert first['excerpts'][0]['text'] in p['travel_persona']
    with pytest.raises(ValueError):build_profile(p,{'crowd_avoidance':{'evidence':'E99','level':'high'}})


def test_seoul_residence_is_not_busan_hometown_or_workplace():
    assert residence_basis(persona())['level']=='first_visit'
    assert residence_basis({**persona(),'province':'서울','district':'서울-강서구'})['level']=='limited'
    assert residence_basis({**persona(),'family_persona':'현재 서울 강서구의 아파트에서 거주합니다.'})['level']=='limited'
    for text in ('서울 강서구가 고향입니다.','서울 강서구의 직장에서 근무합니다.','서울 강서구를 방문하며 여행합니다.'):
        assert residence_basis({**persona(),'family_persona':text})['level']=='first_visit'
    assert residence_basis(persona())['distance_to_station_m'] is None
    assert residence_basis({'uuid':'conflicting','province':'서울','district':'부산-강서구'})['level']=='first_visit'
    assert residence_basis({'uuid':'moved','province':'서울','district':'서울-강서구',
                            'family_persona':'현재 부산 서구의 아파트에서 살고 있습니다.'})['level']=='first_visit'


def sensor_fixture(hidden_temperature=20,hidden_stair_x=8,hidden_plan=None):
    geom=unary_union([box(0,0,20,4),box(0,6,20,12)])
    areas={'test':{'geometry':geom}}
    nodes={'seen':{'area':'test','floor':'B2','x':8,'y':2},
           'hidden':{'area':'test','floor':'B2','x':hidden_stair_x,'y':8},
           'out':{'area':'upper','floor':'B1','x':8,'y':2}}
    stairs={'VISIBLE':{'from_node':'seen','to_node':'out'},'HIDDEN_EXIT_99':{'from_node':'hidden','to_node':'out'}}
    class Fire:
        def sample(self,floor,pos,time):
            return {'extinction_per_m':.8 if pos[1]>6 and hidden_temperature>60 else 0,
                    'temperature_c':hidden_temperature if pos[1]>6 else 20}
    r={'area':'test','floor':'B2','pos':(2,2),'phase':'walking'}
    hidden={'area':'test','floor':'B2','pos':(5,8),'phase':'walking','target':hidden_plan,'personal_memory':['secret']}
    space=PersonalSpace(persona());sensor=OccludedSensor(areas,nodes,stairs,Fire())
    change=sensor.sense(space,r,{1:r,2:hidden},1)
    options=space.options('test',r['pos'],1,1)
    o={'agent_id':1,'time_s':1,'events':['movement_started'],'floor':'B2','phase':'walking',
       'profile':build_profile(persona()),'familiarity':space.familiarity,'current_target':'explore:none',
       'memory':[],'observations':space.observations[-6:],'options':options,'local_hazard':space.last_risk,
       'queue_wait_s':0,'allowed_movements':['continue','hurry'],'private_knowledge':space.snapshot('test',1)}
    return sensor,space,r,o,change


def test_unobserved_geometry_hazard_people_plans_do_not_change_inputs_or_costs():
    _,a,_,one,_=sensor_fixture()
    _,b,_,two,_=sensor_fixture(300,15,'all_people_use_hidden_exit')
    assert a.snapshot('test',1)==b.snapshot('test',1)
    assert project_private(one)==project_private(two)
    assert resolve_private(one,default_choice(one))==resolve_private(two,default_choice(two))
    text=json.dumps(project_private(one))
    assert 'HIDDEN_EXIT' not in text and 'VISIBLE' not in text
    assert 'temperature_c' not in text and 'extinction_per_m' not in text
    assert one['private_knowledge']['visible_people']==[]


def test_local_collision_detour_view_does_not_include_the_hidden_room():
    views=[]
    for upper in (box(0,6,20,12),box(0,8,10,13)):
        geom=unary_union([box(0,0,20,4),upper])
        areas={'test':{'geometry':geom,'sim':None}}
        sensor=OccludedSensor(areas,{},{});space=PersonalSpace(persona())
        r={'agent':{'agent_id':1},'jid':1,'area':'test','floor':'B2','pos':(2,2),'phase':'walking'}
        controller=PrivateController.__new__(PrivateController)
        controller.areas=areas;controller.sensor=sensor;controller.runtime={1:r};controller.local_geom_cache={}
        controller.minds={1:SimpleNamespace(space=space)}
        _,view=controller.visible_helper(r,1)
        views.append(view)
        assert view.bounds[3]<=4
    assert views[0].equals(views[1])


def test_entrance_uses_observed_free_side_without_reading_hidden_landing_lanes():
    chosen=[]
    for hidden_valid_lanes in ([0],[1]):
        geom=box(0,0,10,5)
        class Sim:
            def switch_agent_journey(self,*args):pass
        people=[{'id':i,'area':'hall','position':[5,1]} for i in (2,3,4)]
        space=SimpleNamespace(last_visible_people=people,landmarks={'S':{}},revision=1)
        mind=SimpleNamespace(space=space,pending=set(),selected=None,failed_lanes={})
        controller=PrivateController.__new__(PrivateController)
        controller.minds={1:mind};controller.sensor=SimpleNamespace(visible=lambda *args:True)
        controller.areas={'hall':{'sim':Sim(),'geometry':geom,'lanes':{'S':[(10,20,(5,1)),(11,21,(5,4))]}}}
        controller.stairs={'S':{'entry_lane_numbers':[0,1],'physical_valid_lanes':hidden_valid_lanes}}
        controller.observation=lambda *args,**kwargs:{'options':[{'id':'S','kind':'up_stair','position':[5,2],'path':[]}]}
        r={'agent':{'agent_id':1,'desired_speed_mps':1},'area':'hall','floor':'B1','target':'S','pos':(2,2),
           'phase':'walking','stall_since':0,'goal':(5,1),'goal_is_portal':True,'entry_lane':0,'jid':1}
        controller.update_navigation(r,1)
        chosen.append(r['entry_lane'])
    assert chosen==[1,1]


def test_new_observation_updates_private_knowledge_and_own_traversal_learns_link():
    sensor,space,r,o,_=sensor_fixture()
    assert space.landmarks['VISIBLE'].get('to_floor') is None
    space.traversed('VISIBLE','upper','B1',(8,2),5)
    assert space.landmarks['VISIBLE']['to_floor']=='B1'
    assert space.landmarks['VISIBLE']['connection_source']=='own_traversal'
    r['pos']=(8,8)
    change=sensor.sense(space,r,{1:r},10)
    assert 'HIDDEN_EXIT_99' in change['discovered']
    old=next(p for p in space.snapshot('test',10)['known_landmarks'] if p['id']=='VISIBLE')
    assert not old['state_is_current'] and old['observed_at_s']==1


def test_smoke_and_floor_occlusion_and_only_implemented_signs():
    sensor,space,r,o,_=sensor_fixture()
    sensor.signs=[{'id':'real_fixture_sign','area':'test','position':[4,2],'content':'위쪽 연결 통로'}]
    sensor.sense(space,r,{1:r},2)
    assert [x for x in space.observations if x['kind']=='implemented_sign_seen']
    class Dense:
        def sample(self,*args):return {'extinction_per_m':2,'temperature_c':20}
    sensor.fire=Dense();sensor.hazard_cache={};sensor.frame=None
    assert not sensor.visible('test','B2',(2,2),(8,2),3)
    assert 'HIDDEN_EXIT_99' not in space.landmarks


def test_visible_stair_risk_changes_trigger_judgment_even_when_own_air_is_clear():
    sensor,space,r,_,_=sensor_fixture()
    class Changing:
        def sample(self,floor,pos,time):
            return {'extinction_per_m':.4 if pos[0]>=6 and time==2 else 0,'temperature_c':20}
    sensor.fire=Changing();sensor.hazard_cache={};sensor.frame=None
    change=sensor.sense(space,r,{1:r},2)
    assert space.last_risk['smoke']=='clear'
    assert space.landmarks['VISIBLE']['current_state']['hazard']['smoke']=='dense'
    assert change['risk_changed'] and change['critical']
    change=sensor.sense(space,r,{1:r},3)
    assert change['risk_changed'] and not change['critical']


def test_critical_risk_bypasses_routine_limit_and_cooldown():
    agent={'persona':persona(),'mapping':{},'stair_wait_s':0,'distance_walked_m':0,'decision_wait_s':0}
    mind=PrivateMind(agent,build_profile(persona()));mind.routine_count=6
    mind.last={'time':100};mind.last_floor='B2';mind.pending={'critical_risk'}
    r={'phase':'walking','pos':(0,0),'floor':'B2','agent':agent}
    assert 'critical_risk' in mind.events_private(r,101)


def test_floor_change_bypasses_budget_but_ordinary_stall_does_not():
    agent={'persona':persona(),'mapping':{},'stair_wait_s':0,'distance_walked_m':0,'decision_wait_s':0}
    mind=PrivateMind(agent,build_profile(persona()))
    mind.count=6;mind.routine_count=6;mind.last_decision_time=100;mind.last_floor='B2'
    mind.anchor=(95,(0,0));mind.pending={'stalled'}
    r={'phase':'walking','pos':(0,0),'floor':'B2','agent':agent}
    assert mind.events_private(r,101)==[]
    r['floor']='B1'
    assert 'floor_changed' in mind.events_private(r,101)


def test_seeded_profile_persistence_is_bound_to_uuid_seed_and_version(tmp_path):
    from persona_evac.persona.experimental_profile import ExperimentalProfiles
    agents=[{'agent_id':i,'persona':persona(str(i))} for i in (1,2)]
    first=ExperimentalProfiles(tmp_path,seed=7).prepare(agents)
    second=ExperimentalProfiles(tmp_path,seed=7).prepare(list(reversed(agents)))
    assert first==second
    assert {p['experiment_seed'] for p in first.values()}=={7}
    assert {p['uuid'] for p in first.values()}=={'1','2'}
    assert len(list(tmp_path.rglob('*.json')))==2


def test_crowd_trait_and_blocked_memory_affect_declared_selection_costs():
    _,_,_,o,_=sensor_fixture();base={'kind':'up_stair','position':[1,2],'known_to_floor':None,
        'visible_people':0,'hazard':{'smoke':'clear','heat':'normal'},'past_state':None,'state_is_current':True,
        'age_s':0,'source':'vision','observation_ids':['O1'],'path':[],'distance_m':3}
    o['options']=[{**base,'id':'near','walking_s':3,'visible_people':12},{**base,'id':'far','walking_s':8}]
    choice=default_choice(o);o['current_target']='near'
    o['profile']['traits']['crowd_avoidance']['level']='low'
    assert resolve_private(o,choice)[0]=='near'
    o['profile']['traits']['crowd_avoidance']['level']='high'
    assert resolve_private(o,choice)[0]=='far'
    o['profile']['traits']['crowd_avoidance']['level']='low'
    o['memory']=[{'id':'M1','stair_id':'near','result':'still_blocked','waited_s':20}]
    assert resolve_private(o,{**choice,'memory':['M1']})[0]=='far'


def test_historical_structure_does_not_imply_current_conditions():
    p={**persona(),'province':'서울','district':'서울-강서구'}
    space=PersonalSpace(p)
    assert len(space.landmarks)==4 and all(l['current_state'] is None for l in space.landmarks.values())
    assert space.options('A',(0,0),1,0)==[]


def test_open_nonwalkable_gap_is_not_a_wall_and_people_are_time_stamped():
    from shapely.geometry import Polygon
    a=box(0,0,12,3);b=box(0,5,12,8)
    config={'floors':[{'id':'B2','polygons':[list(a.exterior.coords),list(b.exterior.coords)],'obstacles':[]}],
            'train':{'track_polygon':list(box(0,3,12,5).exterior.coords),'cars':[]}}
    areas={'A':{'geometry':a},'B':{'geometry':b}}
    nodes={'in':{'area':'B','floor':'B2','x':8,'y':6},'out':{'area':'hall','floor':'B1','x':8,'y':6}}
    stairs={'REMOTE':{'from_node':'in','to_node':'out'}}
    class Fire:
        def sample(self,floor,pos,time):return {'extinction_per_m':0,'temperature_c':80 if pos[0]>6 else 20}
    sensor=OccludedSensor(areas,nodes,stairs,Fire(),physical_config=config)
    space=PersonalSpace(persona());r={'area':'A','floor':'B2','pos':(2,2),'phase':'walking'}
    other={'area':'B','floor':'B2','pos':(3,6),'phase':'walking','observed_speed_mps':99,'target':'secret'}
    sensor.sense(space,r,{1:r,2:other},1)
    assert space.last_visible_people[0]['observed_speed_mps'] is None
    assert space.last_visible_people[0]['observed_at_s']==1
    assert space.landmarks['REMOTE']['current_state']['hazard']['heat']=='unknown'
    assert not any(o['id']=='REMOTE' for o in space.options('A',r['pos'],1,1))
    other['pos']=(4,6);sensor.sense(space,r,{1:r,2:other},2)
    assert space.last_visible_people[0]['observed_speed_mps']==1
    assert space.people_observations[0]['people'][0]['position']==[3,6]
    other['floor']='B1';sensor.sense(space,r,{1:r,2:other},3)
    assert space.last_visible_people==[]
    # The train's opaque wall is a visual obstruction, unlike the open track.
    config['train']['cars']=[{'a':[1,4],'b':[10,4],'width_m':2.8,'height_m':2.4}]
    opaque=OccludedSensor(areas,nodes,stairs,physical_config=config)
    assert not opaque.visible('A','B2',(2,2),(3,6),1)


def test_own_navigation_changes_close_actual_experience_not_just_model_intent():
    a={'persona':persona(),'mapping':{},'stair_wait_s':0,'distance_walked_m':0,'decision_wait_s':0}
    mind=PrivateMind(a,build_profile(persona()))
    r={'agent':a,'floor':'B2','phase':'walking','pos':(0,0),'stall_since':0}
    mind.navigation_experience(r,'first',1,'personal_rule')
    a['distance_walked_m']=2;mind.navigation_experience(r,'second',4,'personal_rule')
    assert mind.history[0]['stair_id']=='first' and mind.history[0]['walked_m']==2
    r['stall_since']=4
    current=mind.memory(r,12)[-1]
    assert current['stair_id']=='second' and current['result']=='still_blocked' and current['stalled_s']==8


def test_observed_wall_contact_can_be_escaped_without_crossing_a_wall():
    from persona_evac.integration.personal_space import observed_connector
    geom=box(0,0,10,10)
    assert observed_connector(geom,(.18,3),(2,3))
    assert not observed_connector(geom,(.18,3),(-2,3))


def test_private_provider_rejects_shared_multi_person_context(tmp_path):
    from persona_evac.llm.private_decisions import PrivateDecisions
    class Local:
        model='fixture';digest='fixture';settings={}
    provider=PrivateDecisions(tmp_path,Local())
    with pytest.raises(ValueError,match='한 사람'):provider.decide([{},{}])


def test_input_batch_size_keeps_payloads_equal_and_separates_response_caches(tmp_path):
    from persona_evac.llm.private_decisions import PrivateDecisions
    _,_,_,observation,_=sensor_fixture()
    class Local:
        model='fixture';digest='fixture'
        def __init__(self):self.settings={};self.requests=[]
        def generate(self,prompt,body,schema,max_tokens,prompt_batch_size):
            self.requests.append((deepcopy(body),deepcopy(schema),prompt_batch_size))
            return {'1':default_choice(observation)},{'wall_s':0}
    local=Local()
    legacy=PrivateDecisions(tmp_path,local,prompt_batch_size=128)
    optimized=PrivateDecisions(tmp_path,local,prompt_batch_size=512)
    first,old_meta=legacy.decide([observation])
    second,new_meta=optimized.decide([observation])
    assert first==second and old_meta['cache_key']!=new_meta['cache_key']
    assert local.requests[0][:2]==local.requests[1][:2]
    assert [r[2] for r in local.requests]==[128,512]
    assert legacy.decide([observation])[1]['source']=='ollama_cache'
    assert optimized.decide([observation])[1]['source']=='ollama_cache'
    assert len(local.requests)==2


@pytest.mark.parametrize('value',['잘못된 값','0','4096'])
def test_invalid_input_batch_size_is_rejected_before_connecting(monkeypatch,tmp_path,value):
    from persona_evac.llm.private_decisions import PrivateDecisions
    monkeypatch.setenv('PERSONA_EVAC_PROMPT_BATCH_SIZE',value)
    with pytest.raises(ValueError,match='PERSONA_EVAC_PROMPT_BATCH_SIZE'):
        PrivateDecisions(tmp_path)


@pytest.mark.parametrize('failed_model',[False,True])
def test_initial_motion_replanning_and_fallback_have_no_hidden_route_oracle(monkeypatch,failed_model):
    from pathlib import Path
    from station_fixtures import source
    import persona_evac.integration.station as station
    c=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf-8'))
    modified=deepcopy(c)
    # 미관측 지상 출구를 바꾼다. 현재 승강장의 실제 물리 경계를 바꾸면
    # 판단 입력이 같아도 JuPedSim 보행 좌표는 달라질 수 있다.
    modified['nodes']['E_2_out']['x']+=5
    next(s for s in modified['stairs'] if s['id']=='E_2')['length_m']*=1.4
    def forbidden(*args,**kwargs):raise AssertionError('전체 지도 경로 탐색을 판단에 사용했습니다.')
    monkeypatch.setattr(station,'_route',forbidden,raising=False)
    class Scripted:
        model='fixture';provider_name='test'
        def __init__(self):self.inputs=[]
        def decide(self,observations):
            self.inputs.extend(project_private(o) for o in observations)
            if failed_model:raise RuntimeError('시험용 모델 실패')
            return validate_private({str(o['agent_id']):default_choice(o) for o in observations},observations),{'source':'test'}
    first=Scripted();second=Scripted();request=station.StationRequest(agent_count=10,behavior_mode='qwen',duration_s=5)
    one=station.simulate_station(source(),c,request,first)
    two=station.simulate_station(source(),modified,request,second)
    assert sorted(first.inputs,key=json.dumps)==sorted(second.inputs,key=json.dumps)
    assert one['navigation_updates']==two['navigation_updates']
    assert [a['segments'] for a in one['agents']]==[a['segments'] for a in two['agents']]
    for a in one['agents']:
        assert a['pre_movement_s']==.5 and a['desired_speed_mps']==1.2
        assert not any(s['status']=='waiting' for s in a['segments'])
        assert all(s['from']==s['to'] for s in a['segments'] if s['end']<=.5)
    assert bool(one['behavior']['fallback_count'])==failed_model


def test_parallel_http_requests_isolate_people_and_reuse_individual_caches(tmp_path,monkeypatch):
    from threading import Barrier,Event,Lock
    import httpx
    from persona_evac.llm.ollama import LocalOllama,DEFAULT_MODEL
    from persona_evac.llm.private_decisions import PrivateDecisions

    monkeypatch.setenv('OLLAMA_MODEL',DEFAULT_MODEL)
    barrier=Barrier(2);second_done=Event();lock=Lock();requests=[];finished=[]
    def respond(request):
        if request.url.path=='/api/tags':
            return httpx.Response(200,json={'models':[{'name':DEFAULT_MODEL,'digest':'parallel-fixture'}]})
        body=json.loads(request.content)
        people=json.loads(body['messages'][1]['content'])
        assert len(people)==1 and len(body['messages'])==2
        person=people[0];aid=person['agent_id']
        assert set(body['format']['properties'])=={str(aid)}
        assert [m['id'] for m in person['memory']]==[f'M{aid}']
        assert [o['id'] for o in person['observations']]==[f'O{aid}']
        with lock:requests.append(aid)
        barrier.wait(timeout=5)
        if aid==1:assert second_done.wait(timeout=5)
        with lock:finished.append(aid)
        if aid==2:second_done.set()
        payload={str(aid):{'priority':'environment','memory':[f'M{aid}'],
                           'observations':[f'O{aid}'],'movement':'continue'}}
        return httpx.Response(200,json={'done':True,'done_reason':'stop',
            'message':{'content':json.dumps(payload)},'eval_count':10,'eval_duration':1000000000})

    client=httpx.Client(base_url='http://localhost:11434',transport=httpx.MockTransport(respond))
    local=LocalOllama(client)
    controller=PrivateController.__new__(PrivateController)
    controller.provider=PrivateDecisions(tmp_path,local)
    controller.decision_workers=2;controller.decision_wall_s=0.0
    observations=[]
    for aid in (1,2):
        _,_,_,o,_=sensor_fixture()
        o['agent_id']=aid;o['profile']=build_profile(persona(str(aid)))
        o['memory']=[{'id':f'M{aid}','stair_id':None,'result':'still_blocked','waited_s':5}]
        o['observations']=[{'id':f'O{aid}','kind':'local_hazard'}];o['options']=[]
        observations.append(o)
    before=deepcopy(observations)
    first=controller.decide_batch(observations)
    assert observations==before and sorted(requests)==[1,2] and finished==[2,1]
    assert [choices[0]['agent_id'] for choices,_ in first]==[1,2]
    assert all(meta['source']=='ollama_live' and meta['context_people_count']==1 for _,meta in first)
    assert len({meta['cache_key'] for _,meta in first})==2
    cached=controller.decide_batch(observations)
    assert len(requests)==2 and len(local.calls)==2
    assert all(meta['source']=='ollama_cache' for _,meta in cached)
    assert [choices for choices,_ in first]==[choices for choices,_ in cached]
    assert controller.decision_wall_s>0


@pytest.mark.parametrize('failure',['none','error','wrong_person'])
def test_parallel_simulation_keeps_apply_order_and_per_person_fallback(monkeypatch,failure):
    from threading import Barrier,Event,Lock
    from station_fixtures import source,config
    from persona_evac.integration.station import simulate_station,StationRequest

    class Provider:
        model='fixture';provider_name='test'
        def __init__(self,parallel):
            self.parallel=parallel;self.barrier=Barrier(2);self.second_done=Event()
            self.lock=Lock();self.inputs=[];self.started=0;self.finished=[]
        def decide(self,observations):
            assert len(observations)==1
            o=observations[0];aid=o['agent_id']
            with self.lock:
                self.inputs.append(deepcopy(project_private(o)))
                index=self.started;self.started+=1
            if self.parallel and index<2:
                self.barrier.wait(timeout=5)
                if index==0:assert self.second_done.wait(timeout=5)
                with self.lock:self.finished.append(index)
                if index==1:self.second_done.set()
            if failure=='error' and aid==2:raise RuntimeError('해당 사람의 요청만 실패')
            choices=validate_private({str(aid):default_choice(o)},observations)
            if failure=='wrong_person' and aid==2:choices[0]['agent_id']=999
            return choices,{'source':'test'}

    request=StationRequest(agent_count=10,behavior_mode='qwen',duration_s=5)
    monkeypatch.setenv('PERSONA_EVAC_DECISION_WORKERS','1')
    sequential_provider=Provider(False)
    sequential=simulate_station(source(),config(),request,sequential_provider)
    monkeypatch.setenv('PERSONA_EVAC_DECISION_WORKERS','2')
    parallel_provider=Provider(True)
    parallel=simulate_station(source(),config(),request,parallel_provider)
    assert parallel_provider.finished==[1,0]
    assert sorted(sequential_provider.inputs,key=json.dumps)==sorted(parallel_provider.inputs,key=json.dumps)
    assert parallel['decisions']==sequential['decisions']
    assert parallel['navigation_updates']==sequential['navigation_updates']
    assert [a['segments'] for a in parallel['agents']]==[a['segments'] for a in sequential['agents']]
    assert parallel['summary']==sequential['summary']
    assert parallel['performance']['decision_workers']==2
    assert sequential['performance']['decision_workers']==1
    failed={d['agent_id'] for d in parallel['decisions'] if d['source']=='fallback'}
    assert failed==({2} if failure!='none' else set())
