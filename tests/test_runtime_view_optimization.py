"""계산을 진행하지 않고 위치 검사·속도·화면 캐시의 경계를 확인한다."""
from collections import deque
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

import jupedsim as jps
import pytest
from fastapi import HTTPException
from shapely.geometry import LineString,Point,box

from persona_evac.integration.personal_space import observed_connector_checker
from persona_evac.integration.station_dynamics import append_motion_sample
from persona_evac.llm.ollama import LocalOllama


@pytest.fixture(autouse=True)
def no_simulation(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('계산 보류: 엔진·추론 실행 금지')
    monkeypatch.setattr(jps.Simulation,'iterate',forbidden)
    monkeypatch.setattr(LocalOllama,'generate',forbidden)


def old_connector(geom,start,end):
    line=LineString([start,end])
    if not geom.covers(line):return False
    clearance=max(0,min(.22,geom.boundary.distance(Point(start))-.00001))
    return geom.covers(line.buffer(clearance)) if clearance else True


@pytest.mark.parametrize('start',[(0,2),(.01,2),(.22001,2),(2,2),(5.5,2),(-.01,2)])
def test_connector_preserves_wall_contact_and_candidate_order(start):
    geom=box(0,0,12,8).difference(box(5,1,6,7))
    ends=[(x/2,y/2) for x in range(25) for y in range(17)]
    for points in (ends,list(reversed(ends))):
        check=observed_connector_checker(geom,start,geom.boundary)
        assert [check(p) for p in points]==[old_connector(geom,start,p) for p in points]


def test_speed_is_bitwise_equal_after_stops_window_rollover_and_transfer():
    rng=random.Random(20261029);samples=deque(maxlen=21);runtime={}
    for i in range(12000):
        if i%199==0:
            pos=(rng.random()*100,rng.random()*50)
            samples=deque([(i*.05,pos)],maxlen=21)
            runtime={'motion_samples':deepcopy(samples),'motion_distances':deque(maxlen=20)}
        if i%13:pos=(pos[0]+rng.uniform(-.08,.08),pos[1]+rng.uniform(-.08,.08))
        now=(i+1)*.05;samples.append((now,pos));span=samples[-1][0]-samples[0][0]
        old=sum(math.dist(a[1],b[1]) for a,b in zip(samples,list(samples)[1:]))/span if span>0 else 0
        new=append_motion_sample(runtime,now,pos)
        assert old==new and runtime['motion_samples']==samples


@pytest.fixture
def api_files(tmp_path,monkeypatch):
    from persona_evac.visualization import app as module
    config=json.loads(Path('configs/scenarios/gaehwasan.json').read_bytes())
    source=json.loads(Path('output/gaehwasan/persona_source.json').read_bytes())
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'OUTPUT',tmp_path/'output')
    monkeypatch.setattr(module,'_station_signature',lambda:'fixed-test-code')
    monkeypatch.setattr(module,'LOADED_SIGNATURE','fixed-test-code')
    monkeypatch.setattr(module,'policy_matches',lambda result:True)
    monkeypatch.setenv('LLM_PROVIDER','ollama');module._JSON_FILES.clear()
    def write(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');return path
    write('configs/scenarios/gaehwasan.json',config)
    write('output/gaehwasan/persona_source.json',source)
    write('output/gaehwasan/results.json',{'agents':[{}]*10,'input_config_sha256':'previous-layout'})
    return module,write,config,source


def test_setup_uses_current_inputs_without_results_or_engine_steps(api_files):
    module,write,config,source=api_files
    first=module.gaehwasan_setup();second=module.gaehwasan_setup()
    assert first==second and first is not second
    second['scenario']['name']='caller-change'
    assert module.gaehwasan_setup()==first
    assert 'summary' not in first and 'fire' not in first
    assert first['calculation_available'] and len(first['agents'])==100
    assert all(not a['segments'] and a['evacuation_time_s'] is None for a in first['agents'])
    config['name']='갱신한 지도';write('configs/scenarios/gaehwasan.json',config)
    assert module.gaehwasan_setup()['scenario']['name']=='갱신한 지도'
    source['variants']['C'][0]['persona']['occupation']='갱신한 직업'
    write('output/gaehwasan/persona_source.json',source)
    assert module.gaehwasan_setup()['agents'][0]['persona']['occupation']=='갱신한 직업'
    write('output/gaehwasan/results.json',{'agents':[{}]*11,'input_config_sha256':'previous-layout'})
    assert len(module.gaehwasan_setup()['agents'])==100
    assert len(module.gaehwasan_setup(agent_count=10)['agents'])==10


def test_fire_metadata_creation_replacement_and_deletion_refresh_setup(api_files):
    from persona_evac.integration.fds_environment import geometry_hash,sha
    module,write,config,_=api_files
    assert 'fire_precomputation' not in module.gaehwasan_setup()
    deck=module.ROOT/'configs/fds/gaehwasan.fds';deck.parent.mkdir(parents=True);deck.write_text('deck')
    fields=module.OUTPUT/'fds/gaehwasan/derived/fields.npz';fields.parent.mkdir(parents=True);fields.write_bytes(b'test fields')
    meta={'geometry_sha256':geometry_hash(config),'deck_sha256':sha(deck),'time_end_s':600,'mesh_cell_m':1}
    path=write('output/fds/gaehwasan/derived/metadata.json',meta)
    assert module.gaehwasan_setup()['fire_precomputation']['duration_s']==600
    meta['time_end_s']=900;write('output/fds/gaehwasan/derived/metadata.json',meta)
    assert module.gaehwasan_setup()['fire_precomputation']['duration_s']==900
    deck.write_text('updated deck')
    assert 'fire_precomputation' not in module.gaehwasan_setup()
    path.unlink()
    assert 'fire_precomputation' not in module.gaehwasan_setup()


def test_saved_result_status_does_not_mutate_cache_and_file_replacement_refreshes(api_files):
    module,write,config,_=api_files
    digest=hashlib.sha256((module.ROOT/'configs/scenarios/gaehwasan.json').read_bytes()).hexdigest()
    path=write('output/gaehwasan/results.json',{'agents':[],'input_config_sha256':digest,'implementation_sha256':'old'})
    with pytest.raises(HTTPException) as error:module.gaehwasan_results()
    assert error.value.status_code==409
    assert 'artifact_status' not in module._read_json(path)
    other=path.with_suffix('.tmp');other.write_text(json.dumps({'agents':[],'input_config_sha256':digest,'implementation_sha256':'fixed-test-code'}))
    other.replace(path)
    assert module.gaehwasan_results()['artifact_status']['implementation_matches'] is True


def test_old_map_result_is_rejected_and_preserved(api_files,monkeypatch):
    module,_,_,_=api_files
    path=module.OUTPUT/'gaehwasan/results.json';before=path.read_bytes()
    monkeypatch.setattr(module,'simulate_station',lambda *a,**kw:pytest.fail('조회 중 계산 금지'))
    with pytest.raises(HTTPException) as error:module.gaehwasan_results()
    assert error.value.status_code==409 and error.value.detail['code']=='station_result_outdated'
    assert path.read_bytes()==before
    path.unlink()
    with pytest.raises(HTTPException) as error:module.gaehwasan_results()
    assert error.value.status_code==404 and error.value.detail['code']=='station_result_missing'
    assert len(module.gaehwasan_setup()['agents'])==100


def test_explicit_calculation_route_uses_current_config_with_stub_only(api_files,monkeypatch):
    module,_,config,_=api_files;calls=[]
    def stub(source,settings,request):
        calls.append((settings,request));return {'source':{},'agents':[]}
    monkeypatch.setattr(module,'simulate_station',stub)
    result=module.gaehwasan_run(module.StationRequest(agent_count=100,fire_mode='fds'))
    assert len(calls)==1 and calls[0][0]==config
    assert calls[0][1].fire_mode=='fds' and calls[0][1].agent_count==100
    assert result['computation']['kind']=='recomputed_by_api'
    assert module.gaehwasan_results()['input_config_sha256']==result['input_config_sha256']


def test_setup_query_validates_population_and_pause_guard_remains(api_files,monkeypatch):
    from fastapi.testclient import TestClient
    module,write,config,_=api_files
    with TestClient(module.app) as client:
        assert client.get('/api/gaehwasan/setup?agent_count=9').status_code==422
        assert client.get('/api/gaehwasan/setup?agent_count=101').status_code==422
    config['simulation_enabled']=False;write('configs/scenarios/gaehwasan.json',config)
    monkeypatch.setattr(module,'simulate_station',lambda *a,**kw:pytest.fail('보류 상태 계산 금지'))
    assert not module.gaehwasan_setup()['calculation_available']
    with pytest.raises(HTTPException) as error:module.gaehwasan_run(module.StationRequest())
    assert error.value.status_code==409


@pytest.fixture
def prepared_traits(api_files,monkeypatch):
    from persona_evac.agent_policy import POLICY
    from persona_evac.persona.experimental_profile import FULL_TRAITS
    from persona_evac.persona.full_profile import CONFIG,build_full_profile
    module,write,_,source=api_files
    monkeypatch.setenv('OLLAMA_MODEL','prepared-test')
    monkeypatch.setattr(LocalOllama,'__init__',lambda *a,**kw:pytest.fail('조회 중 모델 연결 금지'))
    model={'name':'prepared-test','digest':'prepared-test-digest'}
    source['provenance'].update(model=model['name'],mapping_settings={'model_digest':model['digest']})
    for agent in source['variants']['C']:
        agent.pop('classification_pending',None)
    source_path=write('output/gaehwasan/persona_source.json',source)
    estimates={t:{'level':'high' if i%2 else 'low','assumption':f'{label}의 준비된 근거입니다.'}
               for i,(t,label) in enumerate(FULL_TRAITS.items())}
    payload={'classification_method':'full_persona','version':CONFIG['version'],
        'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'profiles':[{'agent_id':a['agent_id'],'profile':build_full_profile(
            a['persona'],estimates,POLICY['experiment_seed'],model)} for a in source['variants']['C']]}
    name=f'output/gaehwasan/profile_preparations/full_persona/{CONFIG["version"]}/profiles.json'
    write(name,payload)
    return module,write,source,payload,name


def test_setup_attaches_prepared_traits_without_inference_and_keeps_cache_immutable(prepared_traits):
    module,_,_,payload,_=prepared_traits
    for count in (10,30,60,100):
        result=module.gaehwasan_setup(agent_count=count)
        assert [a['experimental_profile'] for a in result['agents']]==[
            row['profile'] for row in payload['profiles'][:count]]
        assert all(len(a['experimental_profile']['traits'])==6 for a in result['agents'])
        assert all(not a['segments'] for a in result['agents'])
    result['agents'][0]['experimental_profile']['traits']['exploration']['level']='normal'
    assert module.gaehwasan_setup()['agents'][0]['experimental_profile']==payload['profiles'][0]['profile']


@pytest.mark.parametrize('damage',['source_hash','version','model','policy','persona','missing','duplicate','invalid_json','pending'])
def test_setup_does_not_display_stale_or_invalid_prepared_traits(prepared_traits,damage):
    module,write,source,payload,name=prepared_traits
    if damage=='source_hash':payload['source_sha256']='outdated-source'
    elif damage=='version':payload['version']='outdated-version'
    elif damage=='model':payload['profiles'][0]['profile']['model']['digest']='other-model'
    elif damage=='policy':payload['profiles'][0]['profile']['policy_hash']='outdated-policy'
    elif damage=='persona':payload['profiles'][0]['profile']['uuid']='other-person'
    elif damage=='missing':payload['profiles']=payload['profiles'][1:]
    elif damage=='duplicate':payload['profiles'].append(payload['profiles'][0])
    elif damage=='pending':
        source['variants']['C'][0]['classification_pending']=True
        path=write('output/gaehwasan/persona_source.json',source)
        payload['source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    path=write(name,payload)
    if damage=='invalid_json':path.write_text('{',encoding='utf-8')
    assert all('experimental_profile' not in a for a in module.gaehwasan_setup()['agents'])


def test_setup_refreshes_prepared_traits_when_file_is_replaced(prepared_traits):
    module,write,_,payload,name=prepared_traits
    first=module.gaehwasan_setup()
    payload['profiles'][0]['profile']['traits']['exploration']['level']='normal'
    write(name,payload)
    updated=module.gaehwasan_setup()
    assert updated['agents'][0]['experimental_profile']!=first['agents'][0]['experimental_profile']
    assert updated['agents'][0]['experimental_profile']==payload['profiles'][0]['profile']
