"""Local Qwen structured inference with separate caches and measured runtimes."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx

from persona_evac.hashing import canonical_json
from persona_evac.llm.providers import GROQ_MAPPING_PROMPT, JsonCache
from persona_evac.schemas import BehaviorMapping


def atomic_json(path, payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(f'.{uuid4().hex}.tmp')
    temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


class LocalOllama:
    def __init__(self,client=None):
        self.model=os.environ.get('OLLAMA_MODEL','qwen3:4b')
        self.base_url=os.environ.get('OLLAMA_BASE_URL','http://localhost:11434').rstrip('/')
        if urlparse(self.base_url).hostname not in {'localhost','127.0.0.1','::1','host.docker.internal'} or 'cloud' in self.model.lower():
            raise ValueError('로컬 Ollama 주소와 로컬 모델만 사용할 수 있습니다.')
        self.client=client or httpx.Client(base_url=self.base_url,timeout=httpx.Timeout(180,connect=10),trust_env=False)
        try:
            response=self.client.get('/api/tags');response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError('로컬 Ollama 연결에 실패했습니다. Ollama 실행 상태를 확인하세요.') from exc
        model=next((m for m in response.json()['models'] if m['name']==self.model),None)
        if model is None: raise ValueError(f'Ollama에 {self.model} 모델이 설치되어 있지 않습니다.')
        self.digest=model['digest']
        self.settings={'provider':'ollama','model':self.model,'model_digest':self.digest,'think':False,
                       'num_ctx':8192,'temperature':0,'seed':20260916}
        runtime_path=Path(os.environ.get('PERSONA_EVAC_OUTPUT','output'))/'ollama/runtime_config.json'
        if urlparse(self.base_url).port==11434 and runtime_path.exists():
            runtime=json.loads(runtime_path.read_text(encoding='utf-8-sig'))
            if runtime.get('version')!=1 or not isinstance(runtime.get('profile'),dict):
                raise ValueError('로컬 Ollama 실행 프로필 파일의 형식이 올바르지 않습니다.')
            # Launcher metadata records the configured backend, not a GPU API probe.
            # PID and timestamps are deliberately excluded from response cache keys.
            self.settings['runtime_profile']=runtime['profile']
        self.calls=[];self.attempts=[]

    def generate(self,system,payload,schema,max_tokens,prompt_batch_size=512,think=None):
        started=time.perf_counter();attempt={'status':'running','model':self.model,'prompt_batch_size':prompt_batch_size}
        self.attempts.append(attempt)
        try:
            result=self._generate(system,payload,schema,max_tokens,prompt_batch_size,think)
            attempt['status']='response_received';return result
        except Exception as exc:
            attempt.update(status='failed',error=type(exc).__name__);raise
        finally:attempt['wall_s']=time.perf_counter()-started

    def _generate(self,system,payload,schema,max_tokens,prompt_batch_size=512,think=None):
        started=time.perf_counter()
        effective_think=self.settings['think'] if think is None else think
        response=self.client.post('/api/chat',json={'model':self.model,'stream':False,'think':effective_think,
            'keep_alive':'30m','format':schema,
            'options':{'num_ctx':self.settings['num_ctx'],'num_predict':max_tokens,'num_batch':prompt_batch_size,'temperature':0,'seed':20260916},
            'messages':[{'role':'system','content':system},{'role':'user','content':canonical_json(payload)}]})
        response.raise_for_status();data=response.json()
        metric={'provider':'ollama','model':self.model,'model_digest':self.digest,'source':'ollama_live',
                'prompt_batch_size':prompt_batch_size,
                'think':effective_think,
                'wall_s':time.perf_counter()-started,
                **{key:data.get(key,0) for key in ('total_duration','load_duration','prompt_eval_count','prompt_eval_duration','eval_count','eval_duration')}}
        metric['output_tokens_per_s']=metric['eval_count']/(metric['eval_duration']/1e9) if metric['eval_duration'] else 0
        if 'runtime_profile' in self.settings:metric['runtime_profile']=self.settings['runtime_profile']
        self.calls.append(metric)
        if not data.get('done') or data.get('done_reason')!='stop':
            raise ValueError('Ollama 출력이 완료되지 않았습니다. 토큰 제한을 확인하세요.')
        if data.get('prompt_eval_count',0)>=self.settings['num_ctx']-1:
            raise ValueError('Ollama 입력이 문맥 길이 제한에 도달했습니다.')
        return json.loads(data['message']['content']),metric


class OllamaProvider:
    source_name='ollama'
    def __init__(self,cache_dir=Path('data/cache/llm/ollama'),client=None):
        self.local=LocalOllama(client)
        self.model=self.local.model
        self.cache=JsonCache(cache_dir)
        self.settings={**self.local.settings,'system_prompt':GROQ_MAPPING_PROMPT,'max_completion_tokens':512}
        self.records=[]

    def map(self,profile,scenario_context):
        started=time.perf_counter()
        evidence_mode=scenario_context.get('mapping_policy')=='sports_age_sex_urgency_evidence_v2'
        if evidence_mode:
            from persona_evac.llm.mapping_evidence import SPORTS_MAPPING_PROMPT
            prompt=SPORTS_MAPPING_PROMPT
        else:prompt=GROQ_MAPPING_PROMPT
        self.settings={**self.local.settings,'system_prompt':prompt,'max_completion_tokens':1536 if evidence_mode else 512}
        if evidence_mode:self.settings['evidence_guard_version']='direct_cues_literal_ids_v2'
        key=self.cache.key(profile=profile,scenario_context=scenario_context,provider=self.source_name,
                           model=self.model,settings=self.settings)
        cached=self.cache.get(key)
        if cached and evidence_mode:
            from persona_evac.llm.mapping_evidence import validate_mapping_evidence
            cached=validate_mapping_evidence(cached.model_dump(mode='json'),profile)
        if cached:
            self.records.append({'source':'ollama_cache','wall_s':time.perf_counter()-started,'persona_uuid':profile.uuid})
            return cached
        payload={'persona':profile.model_dump(mode='json'),'scenario':scenario_context}
        if evidence_mode:
            from persona_evac.llm.mapping_evidence import validate_mapping_evidence,evidence_schema,evidence_candidates,expand_evidence_ids
            candidates=evidence_candidates(profile)
            payload={'persona':{k:v for k,v in profile.model_dump(mode='json').items() if k in {'uuid','age','sex','occupation'}},
                     'original_evidence':candidates,'scenario':scenario_context}
            for attempt in range(3):
                data,metric=self.local.generate(prompt,payload,evidence_schema(profile,candidates),1536)
                try:mapping=validate_mapping_evidence(expand_evidence_ids(data,candidates),profile);break
                except ValueError as exc:
                    if attempt==2:
                        atomic_json(self.cache.directory/f'{key}.invalid.json',{'persona_uuid':profile.uuid,'response':data,'error':str(exc)})
                        raise
                    payload={**payload,'validation_error':str(exc),'previous_response':data,
                             'repair_instruction':'원문과 일치하는 인용과 구체적인 이유로 다시 판단하세요.'}
            metric['evidence_attempts']=attempt+1
        else:
            data,metric=self.local.generate(prompt,payload,BehaviorMapping.model_json_schema(),512)
            mapping=BehaviorMapping.model_validate(data)
        self.cache.put(key,mapping)
        self.records.append({**metric,'persona_uuid':profile.uuid})
        atomic_json(self.cache.directory/f'{key}.timing.json',self.records[-1])
        return mapping
