"""저장한 변경 전 코드와 현재 표시 도형 블록만 분리 실행해 비교한다."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import jupedsim as jps
from persona_evac.integration import station_layout
from persona_evac.integration.station import resolve_stair
from persona_evac.integration.station_connectors import compile_network
from persona_evac.integration.station_geometry import floor_geometry
from persona_evac.llm.ollama import LocalOllama


ROOT=Path('output/qa/display_geometry_20261029')
SOURCE=Path('src/persona_evac/integration/station_dynamics.py')


def extract_builder(path,before):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    simulate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='simulate')
    end=next(i for i,n in enumerate(simulate.body) if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='areas' for t in n.targets))
    if before:
        start=next(i for i,n in enumerate(simulate.body) if isinstance(n,ast.ImportFrom)
                   and n.module=='persona_evac.integration.station_layout')
    else:
        start=end-1
        assert isinstance(simulate.body[start],ast.If) and ast.dump(simulate.body[start].test)==ast.dump(ast.Name(id='native',ctx=ast.Load()))
    block=deepcopy(simulate.body[start:end])
    del simulate.body[start:end]
    wrapper=ast.parse('def build(c,native):\n    floors={f["id"]:f for f in c["floors"]}\n    return c')
    wrapper.body[0].body[1:1]=block
    scope={'floor_geometry':floor_geometry}
    exec(compile(ast.fix_missing_locations(wrapper),str(path),'exec'),scope)
    return scope['build'],ast.dump(tree)


def main():
    def forbidden(*a,**kw):raise AssertionError('계산 보류: 엔진 진행·새 추론 금지')
    jps.Simulation.iterate=forbidden
    LocalOllama.generate=forbidden
    protected=json.loads((ROOT/'preserved_hashes.json').read_bytes())
    def check_files():
        for item in protected:
            assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'].lower(),item['path']
    check_files()
    source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    old,old_rest=extract_builder(ROOT/'before/station_dynamics.py',True)
    new,new_rest=extract_builder(SOURCE,False)
    assert old_rest==new_rest,'표시 도형 외의 시뮬레이션 로직 변경'
    config=json.loads(Path('configs/scenarios/gaehwasan.json').read_bytes())
    assert config['simulation_enabled'] is False and config['layout_status']=='integrated_paused'
    assert config['jupedsim_connections']['enabled'] is True
    for s in config['stairs']:resolve_stair(s)
    native_config=deepcopy(config)
    native_config['nodes']=compile_network(config)['nodes']
    verification=[]
    for native,fixture in ((True,native_config),(False,config)):
        outputs=[];counts=[]
        for build in (old,new):
            with patch.object(station_layout,'view_geometry',wraps=station_layout.view_geometry) as view:
                outputs.append(build(deepcopy(fixture),native));counts.append(view.call_count)
        assert outputs[0]==outputs[1]
        encoded=[json.dumps(o,ensure_ascii=False,separators=(',',':')).encode() for o in outputs]
        assert encoded[0]==encoded[1],'JSON 항목·좌표 순서 변경'
        assert len(outputs[1]['display3d_openings'])==31
        assert counts==([62,31] if native else [31,31])
        verification.append({'native':native,'complete_scenario_equal':True,'serialized_json_equal':True,
                             'json_sha256':hashlib.sha256(encoded[0]).hexdigest(),
                             'view_calls_before':counts[0],'view_calls_after':counts[1]})

    # 원본 준비와 입력 복사·동등성 검사는 시간에서 제외한다. 캐시는 사용하지 않는다.
    rounds=7;repeats=3
    elapsed={'before':[],'after':[]};cpu_times={'before':[],'after':[]}
    for r in range(rounds):
        outputs={}
        for key,build in ((('before',old),('after',new)) if r%2==0 else (('after',new),('before',old))):
            inputs=[deepcopy(native_config) for _ in range(repeats)]
            wall=time.perf_counter();cpu=time.thread_time()
            outputs[key]=[build(c,True) for c in inputs]
            cpu_times[key].append((time.thread_time()-cpu)/repeats)
            elapsed[key].append((time.perf_counter()-wall)/repeats)
        assert outputs['before']==outputs['after']
    check_files();assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash
    b,a=statistics.median(elapsed['before']),statistics.median(elapsed['after'])
    result={'before_s_per_initialization':b,'after_s_per_initialization':a,
            'saved_s_per_initialization':b-a,'reduction_percent':100*(1-a/b),
            'rounds':rounds,'initializations_per_round':repeats,'heights':31,
            'timings_s_per_initialization':elapsed,'cpu_times_s_per_initialization':cpu_times,
            'verification':verification,'other_simulation_logic_ast_identical':True,
            'protected_files_unchanged':True,'simulation_enabled':False,'engine_steps':0,'new_inference_calls':0,
            'method':'현재 지도·연결 좌표를 사용해 표시 도형 블록만 측정. 3회 생성 평균을 전후 7회 교차 측정한 중앙값. 입력 준비·복사·검증 제외. 전체 대피·전체 화면 로딩 시간은 측정하지 않음.'}
    (ROOT/'isolated_benchmark.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
