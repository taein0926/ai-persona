"""현재 지도 격자 연결 검사만 측정한다. 엔진 진행·새 추론은 금지한다."""
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import jupedsim as jps
from shapely.geometry import Point
from persona_evac.integration import personal_space as current
from persona_evac.integration.station_connectors import compile_network


def reference(before_source,after_source):
    old=ast.parse(before_source);new=ast.parse(after_source)
    def sensor(tree):return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OccludedSensor')
    old_class=sensor(old);new_class=sensor(new)
    old_sense=next(n for n in old_class.body if isinstance(n,ast.FunctionDef) and n.name=='sense')
    links=next(n for n in ast.walk(old_sense) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='links' for t in n.targets))
    class Normalize(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if node.name in {'landmark_links','_landmark_link_covered'}:return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Attribute) and t.attr=='_static_landmark_link' for t in node.targets):return None
            if any(isinstance(t,ast.Name) and t.id=='links' for t in node.targets):return deepcopy(links)
            return node
    assert ast.dump(Normalize().visit(deepcopy(new_class)))==ast.dump(old_class)
    # 변경한 센서 클래스 외의 지도·경로 처리도 동일해야 한다.
    assert ast.dump(ast.Module(body=[n for n in old.body if n is not old_class],type_ignores=[]))==ast.dump(ast.Module(body=[n for n in new.body if n is not new_class],type_ignores=[]))
    wrapper=ast.parse('def original(sensor,area,seen,position):\n    geom=sensor.areas[area]["geometry"]\n    return []')
    wrapper.body[0].body[-1]=ast.Return(value=deepcopy(links.value))
    scope=dict(vars(current));exec(compile(ast.fix_missing_locations(wrapper),'<original_links>','exec'),scope)
    return scope['original']


def main():
    suite=Path('output/qa/landmark_links_20261029')
    def forbidden(*a,**kw):raise AssertionError('계산 보류: 엔진·추론 실행 금지')
    jps.Simulation.iterate=forbidden
    from persona_evac.llm.ollama import LocalOllama
    LocalOllama.generate=forbidden
    protected=json.loads((suite/'preserved_hashes.json').read_bytes())
    def check_files():
        for item in protected:
            assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'].lower(),item['path']
    check_files()
    source=Path(current.__file__).read_bytes();source_hash=hashlib.sha256(source).hexdigest()
    before_source=(suite/'before/personal_space.py').read_text(encoding='utf-8')
    original=reference(before_source,source.decode())
    config=json.loads(Path('configs/scenarios/gaehwasan.json').read_bytes())
    assert config['simulation_enabled'] is False and config['layout_status']=='integrated_paused'
    net=compile_network(config)
    areas={key:{'geometry':g} for key,g in net['geometry'].items()}
    old_sensor=current.OccludedSensor(areas,net['nodes'],net['connectors'])
    new_sensor=current.OccludedSensor(areas,net['nodes'],net['connectors'])
    cases=[]
    for sid,s in net['connectors'].items():
        node=net['nodes'][s['from_node']];area=node['area'];position=[node['x'],node['y']]
        center=current.cell_key(position);radius=math.ceil(current.SENSOR['range_m']/current.GRID)
        cells=[(center[0]+dx,center[1]+dy) for dx in range(-radius,radius+1) for dy in range(-radius,radius+1)]
        covered=[k for k in cells if math.dist(current.cell_pos(k),position)<=current.SENSOR['range_m']
                 and areas[area]['geometry'].covers(Point(current.cell_pos(k)))]
        # 실제 지도 위에 준비한 고정 관측 격자 입력으로만 비교한다.
        for seen in (set(covered),set(covered[::2]),set(covered[1::2]),
                     {k for k in covered if math.dist(current.cell_pos(k),position)<8}):
            cases.append((area,seen,position))
    for area,seen,p in cases:
        assert original(old_sensor,area,seen,p)==new_sensor.landmark_links(area,seen,p)

    def run(optimized,repeats):
        sensor=new_sensor if optimized else old_sensor
        outputs=[]
        for _ in range(repeats):
            for area,seen,p in cases:
                outputs.append(sensor.landmark_links(area,seen,p) if optimized else original(sensor,area,seen,p))
        return outputs
    assert run(False,1)==run(True,1)
    repeats=50;timings={'before':[],'after':[]};cpu_times={'before':[],'after':[]};stats=[]
    for round_no in range(7):
        outputs={}
        for optimized in ((False,True) if round_no%2==0 else (True,False)):
            key='after' if optimized else 'before'
            if optimized:new_sensor._static_landmark_link.cache_clear()
            start=time.perf_counter();cpu=time.thread_time()
            outputs[key]=run(optimized,repeats)
            cpu_times[key].append(time.thread_time()-cpu);timings[key].append(time.perf_counter()-start)
            if optimized:stats.append(new_sensor._static_landmark_link.cache_info()._asdict())
        assert outputs['before']==outputs['after']
    check_files();assert hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest()==source_hash
    b,a=statistics.median(timings['before']),statistics.median(timings['after'])
    result={'connectors':len(net['connectors']),'input_sets':len(cases),'repeats':repeats,
            'link_list_builds_per_round':len(cases)*repeats,'rounds':7,
            'before_s':b,'after_s':a,'reduction_percent':100*(1-a/b),
            'before_us_per_list':b/(len(cases)*repeats)*1e6,'after_us_per_list':a/(len(cases)*repeats)*1e6,
            'timings_s':timings,'cpu_timings_s':cpu_times,'cache_statistics':stats,
            'cache_starts_empty_each_round':True,'results_identical':True,
            'other_sensor_and_map_logic_ast_identical':True,'protected_files_unchanged':True,
            'simulation_enabled':False,'engine_steps':0,'new_inference_calls':0,
            'method':'실제 지도에서 구성한 고정 관측 격자 입력 80종으로 연결 목록 생성만 7회 교차 측정한 중앙값. 매 회 빈 캐시로 시작해 최초 검사 비용 포함. 실제 대피 시간으로 환산하지 않음.'}
    (suite/'isolated_benchmark.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
