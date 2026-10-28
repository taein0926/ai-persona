"""Run an isolated, uncached 100-person simulation and measure real inference waits."""
import argparse
from collections import Counter
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
from threading import Event,Lock,Thread
import time
from uuid import uuid4

import jupedsim
import numpy
import shapely

from persona_evac.integration.fds_environment import FireField
from persona_evac.integration.private_agents import PrivateController
from persona_evac.integration.station import initialize_station,simulate_station,StationRequest
from persona_evac.llm.ollama import LocalOllama,atomic_json
from persona_evac.llm.private_decisions import PrivateDecisions
from persona_evac.persona.experimental_profile import ExperimentalProfiles
from persona_evac.station_artifacts import station_signature


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def timestamp():return datetime.now(timezone(timedelta(hours=9))).isoformat()


class FreshPrivateDecisions:
    """A unique response directory per request prevents all file-cache hits."""
    provider_name='ollama'

    def __init__(self,directory,local,batch_size):
        self.directory=directory;self.local=local;self.model=local.model
        self.prompt_batch_size=batch_size;self.records=[];self.lock=Lock();self.count=0

    def decide(self,observations):
        if len(observations)!=1:raise ValueError('판단 요청에는 한 사람만 전달해야 합니다.')
        with self.lock:self.count+=1;number=self.count
        provider=PrivateDecisions(self.directory/f'{number:05d}_{uuid4().hex}',self.local,self.prompt_batch_size)
        choices,meta=provider.decide(observations)
        if meta['source']!='ollama_live':raise RuntimeError('측정 중 응답 파일 캐시가 사용되었습니다.')
        with self.lock:self.records.append(meta)
        return choices,meta


def main():
    parser=argparse.ArgumentParser(description='100명 전체 시뮬레이션의 실제 추론 대기 시간을 측정합니다.')
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--variant',choices=['initial','optimized'],required=True)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    root=args.run_root;workspace=Path('/benchmark')
    case=root/args.variant
    attempt=case/('attempt_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    attempt.mkdir(parents=True,exist_ok=False)
    expected={'backend':'cuda_v13' if args.variant=='initial' else 'vulkan',
              'parallel':1 if args.variant=='initial' else 2,'kv_cache_type':'q8_0','flash_attention':True}
    workers=1 if args.variant=='initial' else 2
    batch_size=128 if args.variant=='initial' else 512
    os.environ['PERSONA_EVAC_DECISION_WORKERS']=str(workers)
    os.environ['PERSONA_EVAC_PROMPT_BATCH_SIZE']=str(batch_size)
    runtime_path=Path('/host_ollama/runtime_config.json')
    runtime=json.loads(runtime_path.read_text(encoding='utf-8-sig'))
    if runtime['profile']!=expected:raise RuntimeError('측정 대상과 Ollama 서버 설정이 다릅니다.')
    frozen_runtime=Path(os.environ['PERSONA_EVAC_OUTPUT'])/'ollama/runtime_config.json'
    atomic_json(frozen_runtime,runtime)
    source_path=workspace/'inputs/persona_source.json'
    config_path=workspace/'configs/scenarios/gaehwasan.json'
    source=json.loads(source_path.read_text(encoding='utf-8'))
    config=json.loads(config_path.read_text(encoding='utf-8'))
    signature=station_signature();source_hash=sha(source_path);config_hash=sha(config_path)
    request=StationRequest(agent_count=100,behavior_mode='qwen',fire_mode='fds',duration_s=600,experiment_seed=20261027)
    local=LocalOllama();local.settings['num_ctx']=4096
    initialized=initialize_station(source,config,request)
    profiles=ExperimentalProfiles(Path(os.environ['PERSONA_EVAC_OUTPUT'])/'gaehwasan/private_profile_cache',local,request.experiment_seed)
    prepared=profiles.prepare(initialized['agents'])
    if len(prepared)!=100 or any(r['source']!='profile_cache' for r in profiles.records):
        raise RuntimeError('100명의 사전 준비 성향 캐시를 그대로 재사용할 수 없습니다.')
    FireField(config)
    if args.preflight:
        print(json.dumps({'phase':'사전 검증 통과','people':100,'profiles':len(prepared),
            'signature':signature,'runtime_profile':runtime['profile'],'model_digest':local.digest},ensure_ascii=False),flush=True)
        return
    started=None;finished=False;guard_error=None;stop=Event()
    state={'phase':'모델 준비','simulation_time_s':0.0,'evacuated':0}
    provider=FreshPrivateDecisions(attempt/'decision_cache',local,batch_size)

    def snapshot():
        return {**state,'variant':args.variant,'attempt':str(attempt),'updated_at_kst':timestamp(),
            'wall_s':time.perf_counter()-started if started is not None else 0,
            'inference_attempts':len(local.attempts) if started is not None else 0,
            'inference_responses':len(provider.records),'decision_workers':workers,
            'prompt_batch_size':batch_size,'runtime_profile':expected,'guard_error':guard_error}

    def progress():
        nonlocal guard_error
        last_print=0
        while not stop.is_set():
            try:
                current=json.loads(runtime_path.read_text(encoding='utf-8-sig'))
                if current['profile']!=expected or current['server_pid']!=runtime['server_pid']:
                    guard_error='측정 도중 Ollama 서버 또는 실행 설정이 변경되었습니다.'
            except Exception as exc:guard_error='Ollama 실행 설정 확인 실패: '+str(exc)
            status=snapshot();atomic_json(case/'status.json',status)
            if time.perf_counter()-last_print>=30:
                print(json.dumps(status,ensure_ascii=False),flush=True);last_print=time.perf_counter()
            stop.wait(5)

    original_tick=PrivateController.tick
    last_stats=0
    def measured_tick(controller,now):
        nonlocal last_stats
        state['simulation_time_s']=now
        if guard_error:raise RuntimeError(guard_error)
        result=original_tick(controller,now)
        if time.perf_counter()-last_stats>=1:
            phases=Counter(r['phase'] for r in controller.runtime.values())
            state.update(evacuated=phases.get('evacuated',0),phase_counts=dict(phases),
                         inference_elapsed_s=controller.decision_wall_s)
            last_stats=time.perf_counter()
        if guard_error:raise RuntimeError(guard_error)
        return result
    thread=Thread(target=progress,daemon=True)
    thread.start()
    try:
        local.generate('준비 상태만 확인하세요.',{'ready':True},
            {'type':'object','properties':{'ready':{'type':'boolean','const':True}},'required':['ready'],'additionalProperties':False},
            32,prompt_batch_size=batch_size)
        local.calls.clear();local.attempts.clear()
        state['phase']='100명 실제 시뮬레이션 진행'
        PrivateController.tick=measured_tick
        started=time.perf_counter()
        result=simulate_station(source,config,request,provider)
        simulation_wall=time.perf_counter()-started
        stop.set();thread.join(timeout=10)
        if guard_error:raise RuntimeError(guard_error)
        if station_signature()!=signature or sha(source_path)!=source_hash or sha(config_path)!=config_hash:
            raise RuntimeError('고정한 계산 코드 또는 입력 자료가 변경되었습니다.')
        performance=result['performance']
        if performance['cache_request_count'] or performance['profile_inference_calls']:
            raise RuntimeError('측정 중 응답 캐시 또는 추가 성향 추론이 사용되었습니다.')
        if result['behavior']['fallback_count'] or performance['llm_failed_attempt_count']:
            raise RuntimeError('추론 실패가 있어 순수 성능 비교 결과로 사용할 수 없습니다.')
        result['source']['artifact_sha256']=source_hash
        result['implementation_sha256']=signature;result['input_config_sha256']=config_hash
        result['computation']={'kind':'full_100_person_timing','variant':args.variant,'code_sha256':signature}
        output=attempt/'results.json';temp=output.with_suffix('.tmp')
        temp.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')),encoding='utf-8');temp.replace(output)
        report={'status':'완료','variant':args.variant,'finished_at_kst':timestamp(),'people':100,
            'requested_simulation_s':600,'actual_simulation_s':result['summary']['total_time_s'],
            'simulation_wall_s':simulation_wall,'total_processing_wall_s':time.perf_counter()-started,
            'inference_elapsed_s':performance['decision_batch_wall_s'],
            'aggregate_request_wall_s':performance['llm_wall_s'],
            'inference_calls':performance['llm_call_count'],'decision_count':len(result['decisions']),
            'response_cache_hits':performance['cache_request_count'],'profile_inference_calls':0,
            'fallback_count':result['behavior']['fallback_count'],'runtime_profile':expected,
            'workers':workers,'prompt_batch_size':batch_size,'model':local.model,'model_digest':local.digest,
            'inference_settings':local.settings,'source_sha256':source_hash,'config_sha256':config_hash,
            'code_sha256':signature,'result':str(output),'result_sha256':sha(output),
            'summary':result['summary'],'python_runtime':{'jupedsim':jupedsim.__version__,'numpy':numpy.__version__,'shapely':shapely.__version__},
            'method':'같은 100명·같은 화재·600초. 응답 파일 캐시 없음. 사전 성향과 모델 준비 시간은 추론 비교에서 제외.',
            'inference_time_definition':'같은 모의 시점의 판단 묶음에서 응답을 기다린 실제 경과 시간의 합계. 병렬 요청 시간을 중복 합산하지 않음.'}
        atomic_json(attempt/'report.json',report);atomic_json(case/'report.json',report)
        state.update(phase='완료',simulation_time_s=report['actual_simulation_s'],inference_elapsed_s=report['inference_elapsed_s'])
        atomic_json(case/'status.json',{**snapshot(),'report':str(case/'report.json')})
        print(json.dumps(report,ensure_ascii=False),flush=True);finished=True
    except Exception as exc:
        stop.set();thread.join(timeout=10)
        state.update(phase='측정 오류',error=str(exc))
        atomic_json(attempt/'error.json',snapshot());atomic_json(case/'status.json',snapshot())
        raise
    finally:
        stop.set();PrivateController.tick=original_tick
        if not finished:thread.join(timeout=10)


if __name__=='__main__':main()
