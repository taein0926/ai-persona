"""Activate an audited computation; never attach a new signature to old data."""
import argparse,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4
from persona_evac.agent_policy import ROOT
from persona_evac.station_artifacts import station_signature,policy_matches
from persona_evac.llm.ollama import atomic_json

parser=argparse.ArgumentParser();parser.add_argument('result',type=Path);args=parser.parse_args()
data=args.result.read_bytes();r=json.loads(data);digest=hashlib.sha256(data).hexdigest()
audit=json.loads(args.result.with_suffix('.audit.json').read_text(encoding='utf-8'))
assert audit['artifact_sha256']==digest and audit['checks_passed']
signature=station_signature()
assert r['implementation_sha256']==signature and r['computation']['code_sha256']==signature
assert policy_matches(r)
output=Path(os.environ.get('PERSONA_EVAC_OUTPUT',ROOT/'output'))
source=output/'gaehwasan/persona_source.json';config=ROOT/'configs/scenarios/gaehwasan.json'
assert r['source']['artifact_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
assert r['input_config_sha256']==hashlib.sha256(config.read_bytes()).hexdigest()
people=json.loads(source.read_text(encoding='utf-8'))['variants']['C'][:r['summary']['count']]
assert [a['persona'] for a in people]==[a['persona'] for a in r['agents']]
target=output/'gaehwasan/results.json';archive=output/'gaehwasan/private_verification/previous_active'
if target.exists() and target.read_bytes()!=data:
    archive.mkdir(parents=True,exist_ok=True)
    (archive/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_results.json')).write_bytes(target.read_bytes())
temp=target.with_suffix(f'.{uuid4().hex}.tmp');temp.write_bytes(data);temp.replace(target)
assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
report={'activated':str(target),'artifact_sha256':digest,'implementation_sha256':signature,
        'policy_versions':r['agent_policy']['versions'],'summary':r['summary'],'inference_provenance':r['performance']}
atomic_json(output/'gaehwasan/private_verification/activation.json',report)
print(json.dumps(report,ensure_ascii=False))
