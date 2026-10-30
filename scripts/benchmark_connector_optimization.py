"""계산 보류를 유지하며 연결 도형·좌표 변환만 전후 비교한다."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import time
from types import SimpleNamespace

import jupedsim as jps
from shapely.geometry import Point
from persona_evac.integration import station_connectors as current
from persona_evac.integration.station_dynamics import RADIUS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path,default=Path('output/qa/connector_optimization_20261029'))
    parser.add_argument('--rounds',type=int,default=7)
    args=parser.parse_args();suite=args.suite
    def forbidden(*a,**kw):raise AssertionError('계산 보류: 엔진 시간 진행 금지')
    jps.Simulation.iterate=forbidden
    from persona_evac.llm.ollama import LocalOllama
    LocalOllama.generate=forbidden
    protected=[Path(p) for p in ('configs/scenarios/gaehwasan.json',
        'src/persona_evac/integration/station_connectors.py',
        'src/persona_evac/integration/station_dynamics.py',
        'src/persona_evac/integration/private_agents.py','output/gaehwasan/results.json')]
    hashes={str(p):digest(p) for p in protected}
    config=json.loads(protected[0].read_bytes())
    assert config['simulation_enabled'] is False and config['layout_status']=='integrated_paused'
    spec=importlib.util.spec_from_file_location('before_connectors',suite/'before/station_connectors.py')
    before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before)
    old=before.compile_network(config);new=current.compile_network(config)
    for key in old:
        if key=='geometry':assert {k:v.wkb for k,v in old[key].items()}=={k:v.wkb for k,v in new[key].items()}
        else:assert old[key]==new[key]
    queries=[];landings=[];samples=[]
    for sid,s in new['connectors'].items():
        for lane,entry in enumerate(s['native_lane_specs']):
            landings.append((sid,lane))
            dest=current.connector_landing(s,new['nodes'],lane)
            points=[(new['nodes'][s['from_node']]['area'],entry['entry'][:2])]
            if dest['floor']!='OUTSIDE':points.append((dest['area'],[dest['x'],dest['y']]))
            for area,xy in points:
                for dx,dy in ((0,0),(.01,0),(-.01,0),(0,.01),(0,-.01)):
                    queries.append((area,(xy[0]+dx,xy[1]+dy)))
            length=current.path_length(s['native_paths'][entry['branch']])
            # 시작점부터 끝점까지, 도착 판정 경계의 전후를 포함한 고정 좌표를 준다.
            distances=[-0.01,0,length*.1,length*.5,length*.9,length-.091,
                       length-.09,length-.089,length,length+.01]
            distances += [length*i/20 for i in range(21)]
            for d in distances:
                for lateral in (entry['offset']-.1,entry['offset'],entry['offset']+.1):
                    samples.append((sid,lane,d,lateral))

    def clearance(optimized,repeats):
        geometries=new['geometry'] if optimized else old['geometry']
        # 매 측정에 최초 검사 도형 생성 비용까지 포함한다.
        centers={area:g.buffer(-RADIUS-.02) for area,g in geometries.items()} if optimized else None
        result=[]
        for _ in range(repeats):
            for area,xy in queries:
                geom=centers[area] if optimized else geometries[area].buffer(-RADIUS-.02)
                result.append(geom.covers(Point(xy)))
        return result

    def landing(optimized,repeats):
        module,net=(current,new) if optimized else (before,old)
        # 기본 도착 좌표는 기존 코드도 compile_network에서 만들던 값이다.
        return [module.connector_landing(net['connectors'][sid],net['nodes'],lane)
                for _ in range(repeats) for sid,lane in landings]

    def sample_ready(optimized,repeats):
        module,net=(current,new) if optimized else (before,old)
        passages={};actors={}
        for sid,s in net['connectors'].items():
            passage=module.NativePassage.__new__(module.NativePassage)
            passage.connector=s;passage.occupants={};passage.branches=[]
            for path in s['native_paths']:
                actor=SimpleNamespace(position=(0.,0.))
                actors[sid,len(passage.branches)]=actor
                branch={'sim':SimpleNamespace(agent=lambda jid,a=actor:a),
                        'path':path,'length':module.path_length(path)}
                if optimized:branch['prepared_path']=current.PreparedPath(path)
                passage.branches.append(branch)
            for lane in range(len(s['native_lane_specs'])):passage.occupants[lane]=(lane,lane)
            passages[sid]=passage
        result=[]
        for _ in range(repeats):
            for sid,lane,d,lateral in samples:
                passage=passages[sid]
                branch=passage.connector['native_lane_specs'][lane]['branch']
                actors[sid,branch].position=(d,lateral)
                result.append((passage.sample(lane),passage.ready(lane)))
        return result

    jobs=[('진입·도착 검사 도형 재사용',clearance,len(queries)),
          ('고정 도착 좌표 재사용',landing,len(landings)),
          ('계단 좌표 변환·도착 판정',sample_ready,len(samples))]
    reports=[]
    for name,run,count in jobs:
        assert run(False,1)==run(True,1),name
        start=time.perf_counter();reference=run(False,1);probe=time.perf_counter()-start
        repeats=max(1,min(1000,math.ceil(.6/max(probe,1e-6))))
        timings={'before':[],'after':[]}
        # 양쪽을 예열한 뒤 실행 순서를 번갈아 적용한다.
        assert run(False,2)==run(True,2)
        for round_no in range(args.rounds):
            outputs={}
            for optimized in ((False,True) if round_no%2==0 else (True,False)):
                start=time.perf_counter();output=run(optimized,repeats);elapsed=time.perf_counter()-start
                key='after' if optimized else 'before'
                timings[key].append(elapsed);outputs[key]=output
            assert outputs['before']==outputs['after'],name
        b,a=[statistics.median(timings[k]) for k in ('before','after')]
        report={'name':name,'cases_per_pass':count,'repeats':repeats,
                'checks_per_round':count*repeats,'rounds':args.rounds,
                'before_s':b,'after_s':a,'reduction_percent':100*(1-a/b),
                'before_us_per_check':b/(count*repeats)*1e6,
                'after_us_per_check':a/(count*repeats)*1e6,
                'timings_s':timings,'results_identical':True}
        reports.append(report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    assert all(digest(p)==hashes[str(p)] for p in protected)
    report={'method':'동일 입력·교차 순서 7회 반복 중앙값. 도형 및 경로 사전 계산 비용 포함. 전체 대피 시간으로 환산하지 않음.',
        'simulation_enabled':False,'layout_status':'integrated_paused','engine_steps':0,'new_inference_calls':0,
        'scope':'실제 연결망의 고정 도형 검사와 좌표 함수만 실행. sample/ready의 엔진 위치 조회는 양쪽 모두 같은 고정 좌표 객체로 대체.',
        'case_note':'1번은 도형 검사와 기본 도착 좌표 조회로 분리. 2번은 입력 좌표마다 sample과 ready를 각 1회 호출.',
        'python':platform.python_version(),'jupedsim':jps.__version__,'protected_hashes':hashes,
        'protected_files_unchanged':True,'results':reports}
    (suite/'isolated_benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
