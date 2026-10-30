"""기존 FDS 파일과 고정 인원 상태로 진단·조회·출력만 분리 비교한다."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import time

import jupedsim as jps
import numpy as np
from shapely.geometry import box,Point
from persona_evac.integration import fds_environment as current
from persona_evac.integration import station_dynamics as dynamics
from persona_evac.integration.station_connectors import compile_network
from persona_evac.llm.ollama import LocalOllama

ROOT=Path('output/qa/diagnostics_fds_20261029')


def load_previous():
    spec=importlib.util.spec_from_file_location('before_fds',ROOT/'before/fds_environment.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def diagnostic_reference():
    old=ast.parse((ROOT/'before/station_dynamics.py').read_text(encoding='utf-8'))
    new=ast.parse(Path(dynamics.__file__).read_text(encoding='utf-8'))
    sim=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='simulate')
    loop=next(n for n in sim.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='tick')
    block=deepcopy(loop.body[-2:]);assert isinstance(block[0],ast.Assign) and block[0].targets[0].id=='active'
    del loop.body[-2:]
    sim2=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='simulate')
    loop2=next(n for n in sim2.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='tick')
    assert loop2.body[-1].value.func.id=='diagnose_people';del loop2.body[-1]
    new.body=[n for n in new.body if not (isinstance(n,ast.FunctionDef) and n.name=='diagnose_people')]
    assert ast.dump(old)==ast.dump(new),'진단 외의 시뮬레이션 동작 변경'
    wrapper=ast.parse('def original(runtime,before,areas,now,min_gap,overlap_steps,pair_history,rank_changes):\n    return min_gap,overlap_steps')
    wrapper.body[0].body[0:0]=block
    scope={'math':math,'Point':Point,'RADIUS':dynamics.RADIUS,'DT':dynamics.DT}
    exec(compile(ast.fix_missing_locations(wrapper),'<original-diagnostics>','exec'),scope)
    return scope['original']


def volume_builder(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FireField')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='replay')
    block=next(n for n in method.body if isinstance(n,ast.If))
    wrapper=ast.parse("def volume(self,until):\n    result={}\n    return result['volume']")
    wrapper.body[0].body[1:1]=[deepcopy(block)]
    scope={'np':np};exec(compile(ast.fix_missing_locations(wrapper),str(path),'exec'),scope)
    return scope['volume']


def main():
    def forbidden(*a,**kw):raise AssertionError('계산 보류: 엔진·새 추론 실행 금지')
    jps.Simulation.iterate=forbidden;LocalOllama.generate=forbidden
    protected=json.loads((ROOT/'preserved_hashes.json').read_bytes())
    for path in ('output/fds/gaehwasan/derived/fields.npz','output/fds/gaehwasan/derived/metadata.json'):
        protected.append({'path':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    def check():
        for item in protected:assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'].lower(),item['path']
    check()
    config=json.loads(Path('configs/scenarios/gaehwasan.json').read_bytes())
    assert config['simulation_enabled'] is False and config['layout_status']=='integrated_paused'
    old=load_previous();old_diagnostic=diagnostic_reference()
    results={}
    def measure(name,before,after,units,setup=None):
        if setup:setup()
        assert before()==after()
        elapsed={'before':[],'after':[]}
        for i in range(7):
            outputs={}
            for key,fn in ((('before',before),('after',after)) if i%2==0 else (('after',after),('before',before))):
                if setup:setup()
                start=time.perf_counter();outputs[key]=fn();elapsed[key].append(time.perf_counter()-start)
            assert outputs['before']==outputs['after']
        b,a=(statistics.median(elapsed[k]) for k in ('before','after'))
        results[name]={'before_s':b,'after_s':a,'reduction_percent':100*(1-a/b),'units':units,'timings_s':elapsed,'exact_match':True}
        print(json.dumps({'완료':name,'전':b,'후':a},ensure_ascii=False),flush=True)
    for area_count in (5,1):
        rng=random.Random(20261029);cases=[]
        areas={i:{'geometry':box(-100,-100,100,100)} for i in range(area_count)}
        ids=list(range(100));rng.shuffle(ids)
        for frame in range(10):
            runtime={aid:{'area':aid%area_count,'floor':str(aid%area_count),
                          'pos':(rng.random()*20,rng.random()*20),'goal':(rng.random()*40,rng.random()*40),
                          'phase':'walking' if aid%4 else 'waiting','target':str(aid%3)} for aid in ids}
            cases.append(runtime)
        def run(fn):
            history={};changes=[];minimum=math.inf;overlaps=0
            for i in range(200):
                runtime=cases[i%len(cases)]
                minimum,overlaps=fn(runtime,runtime,areas,i*.05,minimum,overlaps,history,changes)
            return minimum,overlaps,history,changes
        measure(f'diagnostics_{area_count}_areas',lambda:run(old_diagnostic),lambda:run(dynamics.diagnose_people),200)
    field=current.FireField(config)
    prior=old.FireField.__new__(old.FireField);prior.__dict__.update(field.__dict__)
    net=compile_network(config);cases=[]
    until=min(600.,float(field.times[-1])-field.offset)
    for t in (0.,min(5.,until),until):
        for s in net['connectors'].values():
            segment={'from':net['nodes'][s['from_node']],'to':net['nodes'][s['to_node']]}
            for p in (0.,.1,.25,.5,.75,.9,1.):cases.append((s,segment,p,t))
    def stairs(f):return [f.sample_stair(config,s,segment,p,t) for _ in range(4) for s,segment,p,t in cases]
    def clear():
        current._stair_path_metrics.cache_clear()
        for f in (field,prior):f._sample_frame=None;f._sample_grid=None
    measure('stair_fds_sampling',lambda:stairs(prior),lambda:stairs(field),len(cases)*4,clear)
    # 경로를 제자리 수정해도 새로운 경로 내용으로 재계산되는지 확인한다.
    s,segment,_,t=cases[0];s=deepcopy(s)
    from persona_evac.integration.station_layout import stair_path
    s['path_xyz']=deepcopy(stair_path(config,s));s['path_xyz'][0][0]+=.001
    assert prior.sample_stair(config,s,segment,.5,t)==field.sample_stair(config,s,segment,.5,t)
    old_volume=volume_builder(ROOT/'before/fds_environment.py');new_volume=volume_builder(Path(current.__file__))
    measure('volume_replay',lambda:old_volume(prior,until),lambda:new_volume(field,until),{'until_s':until,'frames':len(set([float(until)]+list(np.arange(0,until,4))))})
    # 전체 출력 구조도 짧은 재생 구간에서 비교한다. FDS를 새로 계산하지 않는다.
    assert prior.replay(config,min(8.,until))==field.replay(config,min(8.,until))
    check()
    report={'measurements':results,'fds_volume_shape':list(field.data['volume_k'].shape),
            'selected_voxels':int(np.count_nonzero(field.data['volume_k'].max(axis=0)>.003)),
            'other_simulation_logic_ast_identical':True,'full_replay_8s_exact_match':True,
            'protected_files_unchanged':True,'simulation_enabled':False,'engine_steps':0,'new_inference_calls':0,
            'protected_files':protected,'rounds':7,
            'method':'진단은 고정된 100명 입력을 200회 처리. FDS는 현재 저장된 실제 결과만 조회. 각 항목 7회 교차 측정 중앙값. 계단 경로 캐시는 매 회 비워 최초 준비 비용 포함. 전체 대피 시간으로 합산하지 않음.'}
    (ROOT/'isolated_benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
