from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest

from persona_evac.llm import MockProvider
from persona_evac.llm.ollama import DEFAULT_MODEL,LocalOllama
from persona_evac.persona.experimental_profile import ExperimentalProfiles,TRAITS
from persona_evac.persona.station_population import refresh_population
from station_fixtures import source
from station_fixtures import config
from persona_evac.integration.station import initialize_station,StationRequest


def test_default_model_is_gemma_when_environment_is_unset(monkeypatch):
    monkeypatch.delenv('OLLAMA_MODEL',raising=False)
    monkeypatch.setenv('OLLAMA_BASE_URL','http://localhost:11434')
    client=httpx.Client(base_url='http://localhost:11434',transport=httpx.MockTransport(
        lambda request:httpx.Response(200,json={'models':[{'name':'gemma4:e2b-it-qat','digest':'gemma'}]})))
    local=LocalOllama(client)
    assert local.model==DEFAULT_MODEL=='gemma4:e2b-it-qat' and local.digest=='gemma'


def test_same_policy_baselines_refresh_on_model_or_digest_change_only_for_requested_people():
    class Provider(MockProvider):
        def __init__(self,model,digest):self.model=model;self.local=SimpleNamespace(digest=digest);self.seen=[]
        def map(self,person,context):self.seen.append(person.uuid);return super().map(person,context)
    original=source(3);original['provenance']={'snapshot':{'selection_seed':123}}
    for a in original['variants']['C']:
        a['persona']['age']=30;a['provenance']={'empirical':{'source_ref':'data/fixtures/julich_features.csv'}}
    qwen=Provider('qwen3:4b','old');prepared=refresh_population(original,3,qwen)
    before=deepcopy(prepared)
    gemma=Provider(DEFAULT_MODEL,'new');updated=refresh_population(prepared,2,gemma)
    assert len(gemma.seen)==2 and prepared==before
    assert updated['variants']['C'][2]==before['variants']['C'][2]
    assert all(a['provenance']['mapping']['model_digest']=='new' for a in updated['variants']['C'][:2])
    assert refresh_population(updated,2,gemma) is updated and len(gemma.seen)==2
    revision=Provider(DEFAULT_MODEL,'revision');again=refresh_population(updated,2,revision)
    assert len(revision.seen)==2 and again['provenance']['baseline_refresh']['model_digest']=='revision'


def test_keyword_model_switch_preserves_old_cache_and_reuses_new_model_cache(tmp_path):
    class Local:
        def __init__(self,model,digest):self.model=model;self.digest=digest;self.settings={'num_ctx':4096};self.calls=0
        def generate(self,prompt,body,schema,*args,**kwargs):
            self.calls+=1
            return {aid:{t:{'evidence':'NONE'} for t in TRAITS} for aid in body},{'source':'ollama_live'}
    agents=[{'agent_id':1,'persona':{'uuid':'same-person','travel_persona':'인파를 피하고 한적한 길을 좋아합니다.'}}]
    qwen=Local('qwen3:4b','old');first=ExperimentalProfiles(tmp_path,qwen,method='keyword_candidates').prepare(agents)
    path=next(tmp_path.rglob('*.json'));old=path.read_bytes()
    gemma=Local(DEFAULT_MODEL,'new');second=ExperimentalProfiles(tmp_path,gemma,method='keyword_candidates').prepare(agents)
    third=ExperimentalProfiles(tmp_path,gemma,method='keyword_candidates').prepare(agents)
    assert first[1]['model']['name']=='qwen3:4b' and second[1]['model']['name']==DEFAULT_MODEL
    assert second==third and qwen.calls==gemma.calls==1 and path.read_bytes()==old


def test_deleted_classifications_preview_as_pending_and_cannot_supply_motion_values():
    raw=source(10)
    for a in raw['variants']['C']:
        a.pop('pre_movement_s');a.pop('desired_speed_mps');a['classification_pending']=True
    preview=initialize_station(raw,config(),StationRequest(agent_count=10),preview_only=True)
    assert all(a['classification_pending'] and a['pre_movement_s'] is None and a['desired_speed_mps'] is None for a in preview['agents'])
    with pytest.raises(ValueError,match='사전 분류'):
        initialize_station(raw,config(),StationRequest(agent_count=10))
