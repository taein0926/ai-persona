import json

import httpx
import pytest

from persona_evac.llm.ollama import LocalOllama, OllamaProvider
from persona_evac.schemas import PersonaProfile


def client(handler, digest='local-digest'):
    def respond(request):
        if request.url.path=='/api/tags':
            return httpx.Response(200,json={'models':[{'name':'qwen3:4b','digest':digest}]})
        body=json.loads(request.content)
        assert body['think'] is False and body['stream'] is False
        assert body['options']['num_ctx']==8192
        assert body['format']['type']=='object'
        return httpx.Response(200,json={'done':True,'done_reason':'stop',
            'message':{'content':json.dumps(handler(body))},'eval_count':10,'eval_duration':1000000000})
    return httpx.Client(base_url='http://localhost:11434',transport=httpx.MockTransport(respond))


@pytest.fixture(autouse=True)
def local_env(monkeypatch):
    monkeypatch.setenv('OLLAMA_MODEL','qwen3:4b')
    monkeypatch.setenv('OLLAMA_BASE_URL','http://localhost:11434')


def test_mapping_local_schema_cache_and_digest_isolation(tmp_path):
    calls=[]
    def mapping(body):
        calls.append(body)
        return {'response_tendency':'normal','speed_tendency':'normal',
                'reasoning_summary':{'response_tendency':'근거 부족으로 보통 가정','speed_tendency':'근거 부족으로 보통 가정'}}
    p=OllamaProvider(tmp_path,client(mapping))
    profile=PersonaProfile(uuid='a',age=32)
    assert p.map(profile,{})==p.map(profile,{})
    assert len(calls)==1 and p.records[-1]['source']=='ollama_cache'
    p2=OllamaProvider(tmp_path,client(mapping,'changed-digest'));p2.map(profile,{})
    assert len(calls)==2


@pytest.mark.parametrize('url,model',[('https://example.com','qwen3:4b'),('http://localhost:11434','qwen3-cloud')])
def test_local_mode_rejects_external_inference(monkeypatch,url,model):
    monkeypatch.setenv('OLLAMA_BASE_URL',url);monkeypatch.setenv('OLLAMA_MODEL',model)
    with pytest.raises(ValueError,match='로컬'):
        LocalOllama()


def test_launcher_backend_profile_separates_caches_but_restart_metadata_does_not(monkeypatch,tmp_path):
    runtime=tmp_path/'ollama/runtime_config.json';runtime.parent.mkdir()
    monkeypatch.setenv('PERSONA_EVAC_OUTPUT',str(tmp_path))
    calls=[]
    def mapping(body):
        calls.append(body)
        return {'response_tendency':'normal','speed_tendency':'normal',
                'reasoning_summary':{'response_tendency':'근거 부족으로 보통 가정','speed_tendency':'근거 부족으로 보통 가정'}}
    profile=PersonaProfile(uuid='a',age=32)
    def provider(backend,pid):
        runtime.write_text(json.dumps({'version':1,'server_pid':pid,'started_at':str(pid),
            'profile':{'backend':backend,'parallel':1,'kv_cache_type':'f16','flash_attention':True}}),encoding='utf-8')
        return OllamaProvider(tmp_path/'cache',client(mapping))
    cuda=provider('cuda_v13',1);cuda.map(profile,{})
    vulkan=provider('vulkan',2);vulkan.map(profile,{})
    restarted=provider('vulkan',3);restarted.map(profile,{})
    assert len(calls)==2 and restarted.records[-1]['source']=='ollama_cache'
    assert cuda.local.settings['runtime_profile']!=vulkan.local.settings['runtime_profile']
    assert vulkan.local.settings==restarted.local.settings
