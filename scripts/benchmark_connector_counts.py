"""보류 상태에서 실제 변경 전 집계식과 현재 집계 함수만 비교한다."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import jupedsim as jps
from persona_evac.integration import station_dynamics as current


def reference_functions(before_source,after_source):
    def simulation(source):
        return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='simulate')
    def tick(sim):
        return next(n for n in sim.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='tick')
    def assignment(node,name):
        return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
    old_sim=simulation(before_source);old_tick=tick(old_sim)
    movement=next(n for n in old_tick.body if isinstance(n,ast.For) and assignment(n.body[0],'occupants'))
    admission=next(n for n in old_tick.body if isinstance(n,ast.For) and assignment(n.body[0],'waiting'))
    replacements={'occupants':movement.body[0],'waiting':admission.body[0],'occupied':admission.body[1]}
    new_sim=simulation(after_source);new_tick=tick(new_sim)
    assert assignment(new_tick.body[1],'occupants_by_connector')
    assert isinstance(new_tick.body[2],ast.For) and assignment(new_tick.body[2].body[0],'occupants')
    queue_loop=next(i for i,n in enumerate(new_tick.body) if isinstance(n,ast.For) and assignment(n.body[0],'waiting'))
    assert ast.unparse(new_tick.body[queue_loop-1]).startswith('waiting_by_connector, occupied_by_connector = connector_admissions(runtime, stairs)')
    assert isinstance(new_tick.body[queue_loop-2],ast.For)
    assert 'phase=\'queueing\'' in ast.unparse(new_tick.body[queue_loop-2])
    class Normalize(ast.NodeTransformer):
        def visit_Assign(self,node):
            names=[n.id for target in node.targets for n in ast.walk(target) if isinstance(n,ast.Name)]
            if 'occupants_by_connector' in names or 'waiting_by_connector' in names:return None
            for name,old in replacements.items():
                if name in names and isinstance(node.value,ast.Subscript):return deepcopy(old)
            return node
    # 집계 이외의 이동·판단·입퇴장 처리 코드가 그대로인지 확인한다.
    assert ast.dump(Normalize().visit(deepcopy(new_sim)))==ast.dump(old_sim)
    template=ast.parse('''
def old_movement(runtime, stairs):
    groups={}
    for sid,s in stairs.items():
        pass
    return groups
def old_admission(runtime, stairs):
    groups={};counts={}
    for sid,s in stairs.items():
        pass
    return groups,counts
''')
    template.body[0].body[1].body=[deepcopy(movement.body[0]),*ast.parse('groups[sid]=occupants').body]
    template.body[1].body[2].body=[deepcopy(admission.body[0]),deepcopy(admission.body[1]),
                                  *ast.parse('groups[sid]=waiting\ncounts[sid]=occupied').body]
    namespace={};exec(compile(ast.fix_missing_locations(template),'<original_counting>','exec'),namespace)
    return namespace['old_movement'],namespace['old_admission']


def main():
    suite=Path('output/qa/connector_counts_20261029')
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
    old_move,old_admit=reference_functions((suite/'before/station_dynamics.py').read_text(encoding='utf-8'),source.decode())
    config=json.loads(Path('configs/scenarios/gaehwasan.json').read_bytes())
    assert config['simulation_enabled'] is False and config['layout_status']=='integrated_paused'
    ids=[s['id'] for s in config['stairs']]+[g['id'] for g in config.get('fare_gates',[])]
    stairs={sid:{} for sid in ids};rng=random.Random(20261029);cases=[]
    phases=['walking','queueing','on_stairs','waiting','evacuated']
    for state in range(40):
        agent_ids=list(range(1,101));rng.shuffle(agent_ids)
        runtime={i:{'agent':{'agent_id':i},'phase':phases[state%5] if state<5 else rng.choice(phases),
                    'target':rng.choice(ids),'arrived_at':rng.randrange(5),'stair_distance':rng.randrange(5)}
                 for i in agent_ids}
        cases.append(runtime)
    cases_before=deepcopy(cases)
    for runtime in cases:
        assert old_move(runtime,stairs)==current.connector_occupants(runtime,stairs)
        assert old_admit(runtime,stairs)==current.connector_admissions(runtime,stairs)
    reports=[]
    jobs=[('이동 중 인원 분류·정렬',old_move,current.connector_occupants),
          ('대기자 정렬·점유 인원 집계',old_admit,current.connector_admissions)]
    for name,before,after in jobs:
        def run(fn,repeats):
            for _ in range(repeats):
                for runtime in cases:fn(runtime,stairs)
        repeats=100  # 두 항목 모두 동일한 40개 상태를 100번 처리한다.
        run(before,2);run(after,2);times={'before':[],'after':[]};cpu_times={'before':[],'after':[]}
        for round_no in range(7):
            for key,fn in ([('before',before),('after',after)] if round_no%2==0 else [('after',after),('before',before)]):
                start=time.perf_counter();cpu_start=time.thread_time()
                run(fn,repeats)
                cpu_times[key].append(time.thread_time()-cpu_start);times[key].append(time.perf_counter()-start)
        b,a=statistics.median(times['before']),statistics.median(times['after'])
        row={'name':name,'people':100,'connectors':len(stairs),'states':len(cases),
             'batches_per_round':repeats*len(cases),'rounds':7,'before_s':b,'after_s':a,
             'before_us_per_batch':b/(repeats*len(cases))*1e6,'after_us_per_batch':a/(repeats*len(cases))*1e6,
             'reduction_percent':100*(1-a/b),'timings_s':times,
             'cpu_timings_s':cpu_times,
             'before_cpu_s':statistics.median(cpu_times['before']),
             'after_cpu_s':statistics.median(cpu_times['after'])}
        reports.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    assert cases==cases_before
    check_files();assert hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest()==source_hash
    result={'results':reports,'results_identical':True,'input_states_unchanged':True,
            'non_counting_simulation_logic_ast_identical':True,'protected_files_unchanged':True,
            'simulation_enabled':False,'engine_steps':0,'new_inference_calls':0,
            'seed':20261029,'method':'고정 인원 상태 40종·100명·현재 연결 통로로 집계 함수만 7회 교차 측정한 중앙값. 실제 대피 시간으로 환산하지 않음.'}
    (suite/'isolated_benchmark.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
