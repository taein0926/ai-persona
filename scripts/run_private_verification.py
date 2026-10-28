"""A finite, frozen-code 100-person live run. No activation or forced completion."""
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime,timezone
from persona_evac.agent_policy import POLICY,VERSIONS,POLICY_HASH
from persona_evac.integration.station import run_station
from persona_evac.llm.ollama import atomic_json

root=Path('output/gaehwasan/private_verification');root.mkdir(parents=True,exist_ok=True)
for name in ('status.json','live_code_manifest.json','protected_inputs.json'):
    p=root/name
    if p.exists():
        archive=root/'prototype_archive';archive.mkdir(exist_ok=True)
        backup=archive/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+name)
        backup.write_bytes(p.read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
protected=[
 'output/gaehwasan/persona_source.json','data/raw/nemotron_personas_100.jsonl',
 'data/processed/julich_features.csv','configs/scenarios/gaehwasan.json','configs/fds/gaehwasan.fds',
 'output/fds/gaehwasan/derived/fields.npz','output/fds/gaehwasan/derived/metadata.json',
 'output/latest/results.json']
protected_hashes={p:sha(p) for p in protected if Path(p).exists()}
code_files=sorted(Path('src/persona_evac').rglob('*.py'))+sorted(Path('configs/agents').glob('*.json'))
code={str(p):sha(p) for p in code_files}
atomic_json(root/'protected_inputs.json',protected_hashes)
atomic_json(root/'live_code_manifest.json',code)
status=root/'status.json';target=root/'live.json'
if target.exists():
    archive=root/'previous_verified_runs';archive.mkdir(exist_ok=True)
    (archive/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_live.json')).write_bytes(target.read_bytes())
atomic_json(status,{'phase':'100명 실제 로컬 추론 진행 중','started_at_utc':datetime.now(timezone.utc).isoformat(),'policy_versions':VERSIONS,'policy_hash':POLICY_HASH,'result':str(target)})
started=time.perf_counter()
try:
    run_station(Path('output/gaehwasan/persona_source.json'),Path('configs/scenarios/gaehwasan.json'),target,
                qwen=True,agent_count=100,fire=True,experiment_seed=POLICY['experiment_seed'],duration_s=600)
    assert all(sha(p)==h for p,h in protected_hashes.items()),'보존 대상 자료가 변경되었습니다.'
    assert all(sha(p)==h for p,h in code.items()),'실행 중 코드가 변경되었습니다.'
    result=json.loads(target.read_text(encoding='utf-8'))
    report={'phase':'100명 실행 완료·검증 대기','wall_s':time.perf_counter()-started,
            'summary':result['summary'],'performance':result['performance'],
            'fallback_count':result['behavior']['fallback_count'],'result':str(target),
            'policy_versions':VERSIONS,'policy_hash':POLICY_HASH,'protected_inputs_unchanged':True,
            'code_frozen_during_run':True}
    atomic_json(status,report);print(json.dumps(report,ensure_ascii=False),flush=True)
except Exception as exc:
    atomic_json(status,{'phase':'실행 오류','error':str(exc),'result':str(target)});raise
