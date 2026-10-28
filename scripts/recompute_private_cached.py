"""Actually rerun JuPedSim/FDS with validated cached choices; never retag a run."""
import hashlib,json,time
from pathlib import Path
from persona_evac.integration.station import run_station
from persona_evac.llm.ollama import LocalOllama,atomic_json

class UnexpectedCacheMiss(BaseException):pass
def no_new_inference(*args,**kwargs):raise UnexpectedCacheMiss('검증용 재계산에서 캐시 미스가 발생했습니다. 새 추론이나 기본 행동으로 대체하지 않습니다.')
LocalOllama.generate=no_new_inference
root=Path('output/gaehwasan/private_verification');target=root/'recomputed.json'
started=time.perf_counter()
run_station(Path('output/gaehwasan/persona_source.json'),Path('configs/scenarios/gaehwasan.json'),target,
            qwen=True,agent_count=100,fire=True,duration_s=600)
new=json.loads(target.read_text(encoding='utf-8'));old=json.loads((root/'live.json').read_text(encoding='utf-8'))
assert new['performance']['llm_call_count']==0 and new['behavior']['fallback_count']==0
assert all(d['source']=='ollama_cache' for d in new['decisions'])
for key in ('scenario','source','passages','summary','agent_policy','navigation_updates','movement_diagnostics','noncompletion'):
    assert old[key]==new[key],key
for a,b in zip(old['agents'],new['agents']):
    normalized={k:v for k,v in a.items() if k!='decision_profile'}
    assert normalized==b,('agent',a['agent_id'])
for a,b in zip(old['decisions'],new['decisions']):
    for key in ('raw_model_choice','model_choice','final_target','strategy_costs','events','observation','executed_target','executed_goal','execution_source'):
        assert a[key]==b[key],('decision',a['agent_id'],a['time_s'],key)
report={'status':'통과','wall_s':time.perf_counter()-started,'performance':new['performance'],
    'count':100,'summary':new['summary'],'raw_choices_and_private_inputs_identical':True,
    'all_physical_trajectories_observations_memory_identical':True,'all_stair_and_collision_diagnostics_identical':True,
    'removed_duplicate_field':'decision_profile','new_inference_calls':0,'cache_decisions':len(new['decisions']),
    'live_sha256':hashlib.sha256((root/'live.json').read_bytes()).hexdigest(),
    'recomputed_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
    'implementation_sha256':new['implementation_sha256'],'recomputed_path':str(target)}
atomic_json(root/'recompute_comparison.json',report)
print(json.dumps(report,ensure_ascii=False),flush=True)
