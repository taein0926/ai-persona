"""현재 위치 연결 검사·이동속도를 계산 진행 없이 분리 측정한다."""
from collections import deque
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

import jupedsim as jps
from persona_evac.integration import personal_space as current
from persona_evac.integration.station_dynamics import append_motion_sample
from persona_evac.integration.station_connectors import compile_network
from persona_evac.llm.ollama import LocalOllama

ROOT=Path('output/qa/runtime_view_optimization_20261029')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    def forbidden(*a,**kw):raise AssertionError('계산 보류: 엔진·추론 실행 금지')
    jps.Simulation.iterate=forbidden;LocalOllama.generate=forbidden
    protected=json.loads((ROOT/'preserved_hashes.json').read_bytes())
    def check():
        for item in protected:assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'].lower(),item['path']
    check()
    old=load('old_space',ROOT/'before/personal_space.py')
    config=json.loads(Path('configs/scenarios/gaehwasan.json').read_bytes())
    assert config['simulation_enabled'] is True and config['layout_status']=='ready'
    network=compile_network(config);fixtures=[]
    for key,geom in network['geometry'].items():
        p=geom.representative_point();start=(p.x,p.y)
        ends=[(start[0]+dx,start[1]+dy) for dx in (-2,-1,0,1,2) for dy in (-2,-1,0,1,2)]
        fixtures.append((geom,start,ends))
    def connector_run(optimized):
        out=[]
        for _ in range(100):
            for geom,p,ends in fixtures:
                if optimized:
                    test=current.observed_connector_checker(geom,p);out.extend(test(q) for q in ends)
                else:out.extend(old.observed_connector(geom,p,q) for q in ends)
        return out
    points=[(i*.05,(math.sin(i*.003)*10,math.cos(i*.004)*10)) for i in range(40000)]
    def speed_run(optimized):
        r={'motion_samples':deque([points[0]],maxlen=21),'motion_distances':deque(maxlen=20)};out=[]
        for t,p in points[1:]:
            if optimized:out.append(append_motion_sample(r,t,p))
            else:
                samples=r['motion_samples'];samples.append((t,p));span=samples[-1][0]-samples[0][0]
                out.append(sum(math.dist(a[1],b[1]) for a,b in zip(samples,list(samples)[1:]))/span if span>0 else 0)
        return out
    # 동일 센서 입력에서 개인 지도·관측 기록·경로 후보까지 비교한다.
    areas={k:{'geometry':g} for k,g in network['geometry'].items()}
    sensors=[m.OccludedSensor(areas,network['nodes'],network['connectors']) for m in (old,current)]
    spaces=[m.PersonalSpace({'uuid':'isolated-audit','province':'부산','district':'부산-강서구'}) for m in (old,current)]
    area=next(iter(areas));p=areas[area]['geometry'].representative_point();p=(p.x,p.y)
    for now in (0,1,2):
        r={'area':area,'floor':next(s['floor'] for s in network['area_specs'] if s['id']==area),'pos':p,'phase':'walking'}
        assert sensors[0].sense(spaces[0],r,{1:r},now)==sensors[1].sense(spaces[1],r,{1:r},now)
        assert vars(spaces[0])==vars(spaces[1])
        assert spaces[0].options(area,p,1.2,now)==spaces[1].options(area,p,1.2,now)
    results={}
    def measure(name,before,after,units):
        assert before()==after()
        timings={'before':[],'after':[]}
        for i in range(7):
            values={}
            for key,fn in ((('before',before),('after',after)) if i%2==0 else (('after',after),('before',before))):
                start=time.perf_counter();values[key]=fn();timings[key].append(time.perf_counter()-start)
            assert values['before']==values['after']
        b,a=(statistics.median(timings[k]) for k in ('before','after'))
        results[name]={'before_s':b,'after_s':a,'reduction_percent':100*(1-a/b),'units':units,'timings_s':timings,'equal':True}
    measure('current_position_connections',lambda:connector_run(False),lambda:connector_run(True),len(fixtures)*25*100)
    measure('observed_speed',lambda:speed_run(False),lambda:speed_run(True),len(points)-1)
    check()
    report={'measurements':results,'sensor_maps_observations_options_identical':True,
            'rounds':7,'simulation_enabled':config['simulation_enabled'],'engine_steps':0,'new_inference_calls':0,'protected_files_unchanged':True,
            'method':'동일 고정 입력으로 각 부분만 7회 교차 측정한 중앙값. API는 Python GET 함수의 반복 조회로 HTTP 전송·브라우저 렌더링을 제외함. 전체 대피 단축률로 합산하지 않음.'}
    (ROOT/'isolated_benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
