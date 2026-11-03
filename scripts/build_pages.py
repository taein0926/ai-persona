"""검증된 저장 결과만 GitHub Pages의 읽기 전용 정적 사이트로 내보낸다."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
AGENT_FIELDS = (
    'agent_id','persona','mapping','pre_movement_s','desired_speed_mps','spawn','segments',
    'evacuation_time_s','exit_id','fed','incapacitated_at_s','gas_exposure_history',
    'visibility_blocked_history','experimental_profile','familiarity','initial_knowledge',
    'personal_knowledge','observations','people_observations',
)
DECISION_FIELDS = ('agent_id','time_s','priority','requested_movement','raw_model_choice','executed_target')
OBSERVATION_FIELDS = ('id','time_s','kind','landmark_id','to_floor','floor','perceived')
ASSETS = ('gaehwasan.js','gaehwasan3d.js','station_viewer.js','private_inspector.js',
    'gaehwasan.css','station_base.css','pages_replay.js','vendor/three/three.module.js',
    'vendor/three/three.core.js','vendor/three/OrbitControls.js','vendor/three/LICENSE')
PUBLISHED_FILES = {*ASSETS,'index.html','data/replay.json.gz','data/manifest.json','.nojekyll',
    'build-info.json'}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def pick(value, names):
    return {name:value[name] for name in names if name in value}


def project_replay(source):
    """시각·좌표·농도 값을 반올림하거나 시간 표본을 줄이지 않는다."""
    data=pick(source,('schema_version','scenario','backend','backend_notice','scenario_hash','summary',
        'replay_duration_s','fire','agent_policy','movement_diagnostics','navigation_updates',
        'input_config_sha256','implementation_sha256'))
    data['behavior']=pick(source.get('behavior',{}),('mode','model'))
    data['agents']=[]
    for original in source['agents']:
        agent=pick(original,AGENT_FIELDS)
        agent['personal_knowledge']=pick(original.get('personal_knowledge',{}),('local_ids',))
        agent['initial_knowledge']={key:pick(value,('to_floor',))
                                    for key,value in original.get('initial_knowledge',{}).items()}
        agent['observations']=[pick(row,OBSERVATION_FIELDS) for row in original.get('observations',[])
            if row['kind'] in {'landmark_seen','connection_learned','local_hazard'}]
        agent['people_observations']=[{**pick(row,('time_s','floor')),'people_count':len(row['people'])}
                                      for row in original.get('people_observations',[])]
        data['agents'].append(agent)
    data['decisions']=[pick(row,DECISION_FIELDS) for row in source.get('decisions',[])]
    data['hazard_history']=[pick(row,('agent_id','time_s','floor','stair_id','extinction_per_m'))
                            for row in source.get('hazard_history',[])]
    return data


def csp(importmap=None):
    scripts="'self'"
    if importmap:
        scripts+=" 'sha256-"+base64.b64encode(hashlib.sha256(importmap.encode()).digest()).decode()+"'"
    return ("default-src 'self'; script-src "+scripts+"; style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; "
        "form-action 'none'; frame-src 'none'")


def write(root,name,content):
    if name not in PUBLISHED_FILES:raise ValueError('허용하지 않은 공개 파일입니다.')
    target=root/name
    target.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(content,bytes):target.write_bytes(content)
    else:target.write_text(content,encoding='utf-8',newline='\n')


def build(root,output,source_path=None,activation_path=None):
    if output.resolve()==root.resolve():raise ValueError('소스 저장소에 직접 배포 파일을 덮어쓸 수 없습니다.')
    if output.exists():
        for path in output.rglob('*'):
            name=path.relative_to(output)
            if path.is_file() and '.git' not in name.parts and name.as_posix() not in PUBLISHED_FILES:
                raise ValueError(f'배포 폴더에 허용하지 않은 파일이 있습니다: {name}')
    source_path=source_path or root/'output/gaehwasan/results.json'
    activation_path=activation_path or root/'output/gaehwasan/private_verification/activation.json'
    raw=source_path.read_bytes();activation=json.loads(activation_path.read_bytes())
    source=json.loads(raw)
    if (activation.get('artifact_sha256')!=sha(raw) or
            activation.get('implementation_sha256')!=source.get('implementation_sha256') or
            not activation.get('activated')):
        raise ValueError('저장 결과와 활성화 검증 기록이 일치하지 않습니다.')
    if source['summary']['count']!=len(source['agents']):
        raise ValueError('저장 결과의 인원 정보가 다릅니다.')
    output.mkdir(parents=True,exist_ok=True)
    projection=project_replay(source)
    payload=json.dumps(projection,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    if len(payload)>128*1024*1024:raise ValueError('정적 재생 자료가 브라우저 로딩 한도를 넘었습니다.')
    compressed=gzip.compress(payload,compresslevel=9,mtime=0)
    manifest={'version':1,'file':'replay.json.gz','summary':source['summary'],
        'source_artifact_sha256':sha(raw),'source_implementation_sha256':source['implementation_sha256'],
        'payload_bytes':len(payload),'payload_sha256':sha(payload),
        'compressed_bytes':len(compressed),'compressed_sha256':sha(compressed),
        'notice':'저장된 계산의 시간·좌표·연기·가스·판단 시점을 유지한 재생용 자료입니다.'}
    write(output,'data/replay.json.gz',compressed)
    write(output,'data/manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    html=(root/'gaehwasan.html').read_text(encoding='utf-8').replace('href="/','href="./').replace('src="/','src="./')
    html=html.replace('"three":"/vendor/','"three":"./vendor/')
    html=html.replace('data-station="gaehwasan"','data-station="gaehwasan" data-static-replay="true"')
    html=html.replace('<title>개화산역 · 페르소나 대피 시뮬레이션</title>','<title>개화산역 · 저장된 대피 시뮬레이션</title>')
    html=html.replace('<h2 class="panel-title">시뮬레이션 설정</h2>',
                      '<h2 class="panel-title">저장된 시뮬레이션</h2>')
    html=re.sub(r'(<(?:select|button)\b[^>]*\bid="(?:agentCount|fireMode|behaviorMode|behaviorRun)"[^>]*)>',
                lambda m:m[0] if re.search(r'\bdisabled\b',m[1]) else m[1]+' disabled>',html)
    html=html.replace('<script src="./gaehwasan.js','<script src="./pages_replay.js"></script><script src="./gaehwasan.js')
    html=html.replace('로컬 Gemma 4 E2B','저장된 개인별 판단')
    importmap=re.search(r'<script type="importmap">(.*?)</script>',html).group(1)
    security='<meta name="referrer" content="no-referrer"><meta http-equiv="Content-Security-Policy" content="'+csp(importmap)+'">'
    html=html.replace('<head>','<head>\n  '+security)
    write(output,'index.html',html)
    for name in ASSETS:
        text=(root/name).read_text(encoding='utf-8')
        text=text.replace("'/station_base.css'","'./station_base.css'")
        text=text.replace("from '/vendor/three/","from './vendor/three/")
        if name=='gaehwasan.css':
            text+='\nbody[data-static-replay] #behaviorRun{display:none}\n'
        write(output,name,text)
    write(output,'.nojekyll','')
    write(output,'build-info.json',json.dumps({'source_artifact_sha256':sha(raw),
          'summary':source['summary'],'read_only':True,'external_services':False},ensure_ascii=False,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'public')
    args=parser.parse_args()
    info=build(ROOT,args.output.resolve())
    print(json.dumps({'status':'정적 재생 사이트 생성 완료','summary':info['summary'],
                      'download_mb':round(info['compressed_bytes']/1e6,2)},ensure_ascii=False))
