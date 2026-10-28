"""센서 관측 안의 벽 교차·부수 기하 연산·FDS 조회 시간을 분리 계측한다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import functools
import hashlib
import inspect
import json
from pathlib import Path
import time


class SensorProbe:
    def __init__(self,clock=time.perf_counter):
        self.clock=clock
        self.active=False
        self.geometry_frame=None
        self.physical_depth=0
        self.no_fire_calls=0
        self.rows={name:[0,0.,0.] for name in (
            'intersection','covers','distance','buffer','physical','sample_miss','sample_direct')}
        self.senses=[]

    def geometry_wrapper(self,original,name):
        row=self.rows[name]
        @functools.wraps(original)
        def measured(*args,**kwargs):
            if not self.active:return original(*args,**kwargs)
            parent=self.geometry_frame;frame=[0.]
            self.geometry_frame=frame
            started=self.clock()
            try:
                return original(*args,**kwargs)
            finally:
                elapsed=self.clock()-started
                self.geometry_frame=parent
                if parent is not None:parent[0]+=elapsed
                row[0]+=1;row[1]+=elapsed;row[2]+=elapsed-frame[0]
        return measured

    def physical_wrapper(self,original):
        row=self.rows['physical']
        @functools.wraps(original)
        def measured(sensor,*args,**kwargs):
            if not self.active:return original(sensor,*args,**kwargs)
            if sensor.fire is None:self.no_fire_calls+=1
            self.physical_depth+=1
            started=self.clock()
            try:
                return original(sensor,*args,**kwargs)
            finally:
                elapsed=self.clock()-started
                self.physical_depth-=1
                row[0]+=1;row[1]+=elapsed;row[2]+=elapsed
        return measured

    def sample_wrapper(self,original):
        @functools.wraps(original)
        def measured(*args,**kwargs):
            if not self.active:return original(*args,**kwargs)
            row=self.rows['sample_miss' if self.physical_depth else 'sample_direct']
            started=self.clock()
            try:
                return original(*args,**kwargs)
            finally:
                elapsed=self.clock()-started
                row[0]+=1;row[1]+=elapsed;row[2]+=elapsed
        return measured

    def sense_wrapper(self,original):
        @functools.wraps(original)
        def measured(sensor,space,r,runtime,now):
            assert not self.active, '센서 관측이 중첩되어 계측 범위를 확인해야 합니다.'
            before={k:v[:] for k,v in self.rows.items()}
            self.active=True
            started=self.clock()
            try:
                return original(sensor,space,r,runtime,now)
            finally:
                elapsed=self.clock()-started
                self.active=False
                operations={k:{'calls':v[0]-before[k][0],'inclusive_s':v[1]-before[k][1],
                               'exclusive_s':v[2]-before[k][2]} for k,v in self.rows.items()}
                operations['physical']['exclusive_s']=operations['physical']['inclusive_s']-operations['sample_miss']['inclusive_s']
                self.senses.append({'agent_id':r.get('agent',{}).get('agent_id'),
                    'time_s':now,'sensor_s':elapsed,
                    'operations':operations})
        return measured

    def report(self):
        total=sum(r['sensor_s'] for r in self.senses)
        wall=self.rows['intersection'][2]
        auxiliary=sum(self.rows[k][2] for k in ('covers','distance','buffer'))
        fds=self.rows['physical'][1]+self.rows['sample_direct'][1]
        other=total-wall-auxiliary-fds
        assert other>=-1e-6, '구간 시간이 중복 합산되었습니다.'
        components={'wall_intersection':wall,'other_geometry_operators':auxiliary,
                    'fds_lookup_including_cache':fds,'remaining_sensor_work':max(0.,other)}
        physical_calls=self.rows['physical'][0];misses=self.rows['sample_miss'][0]
        operators={k:{'calls':v[0],'inclusive_s':v[1],'exclusive_s':v[2]} for k,v in self.rows.items()}
        operators['physical']['exclusive_s']=self.rows['physical'][1]-self.rows['sample_miss'][1]
        return {'sensor_calls':len(self.senses),'sensor_total_s':total,
            'components':{k:{'seconds':v,'percent_of_sensor':100*v/total if total else 0} for k,v in components.items()},
            'operators':operators,
            'fds':{'physical_lookup_calls':physical_calls,'cache_miss_sample_calls':misses,
                'direct_sample_calls':self.rows['sample_direct'][0],
                'cache_hits':physical_calls-misses if not self.no_fire_calls else None,
                'cache_hit_percent':100*(1-misses/physical_calls) if physical_calls and not self.no_fire_calls else None,
                'actual_array_sampling_s':self.rows['sample_miss'][1]+self.rows['sample_direct'][1],
                'cache_key_lookup_and_dispatch_s':self.rows['physical'][1]-self.rows['sample_miss'][1],
                'no_fire_calls':self.no_fire_calls},
            'definitions':{'scope':'OccludedSensor.sense 호출 내부에서만 집계. 다른 호출 경로의 시야·경로·3번 도형 처리는 제외한다.',
                'wall_intersection':'실제로 실행된 Shapely intersection 메서드의 자체 시간',
                'other_geometry_operators':'covers·distance·buffer 자체 시간. 중첩 연산 시간을 제외한다.',
                'fds_lookup_including_cache':'physical의 캐시 확인부터 필요한 실제 FDS 배열 조회·보간까지의 시간. 직접 sample 호출이 있으면 추가한다.',
                'fds_actual_sampling':'실제 FDS 배열 검색·시간 보간 시간으로 FDS 전체 조회 시간의 하위 항목이다. 별도 합산하지 않는다.',
                'remaining_sensor_work':'선·점 등 객체 생성, 격자·개인 지도 갱신, 연기 광선 누적, 관측 기록 등. 계측 부가 비용 일부도 포함된다.',
                'measurement':'전수 호출 수와 perf_counter 경과 시간. 계측을 포함한 단일 실행이며 최적화 전후 속도 비교가 아니다.'}}


def install_probe(sensor_module,field_class,probe):
    from shapely.geometry.base import BaseGeometry
    installed=[]
    def replace(owner,name,wrapper):
        original=getattr(owner,name)
        installed.append((owner,name,original))
        setattr(owner,name,wrapper(original))
    for name in ('intersection','covers','distance','buffer'):
        replace(BaseGeometry,name,lambda original,name=name:probe.geometry_wrapper(original,name))
    replace(sensor_module.OccludedSensor,'physical',probe.physical_wrapper)
    replace(field_class,'sample',probe.sample_wrapper)
    replace(sensor_module.OccludedSensor,'sense',probe.sense_wrapper)
    def restore():
        for owner,name,original in reversed(installed):setattr(owner,name,original)
    return restore


def refresh_report(suite):
    """저장된 중첩 FDS 구간을 자체 시간과 포함 시간으로 구분해 정리한다."""
    report=json.loads((suite/'sensor_costs.json').read_bytes())
    ops=report['operators']
    ops['physical']['exclusive_s']=ops['physical']['inclusive_s']-ops['sample_miss']['inclusive_s']
    rows=[json.loads(line) for line in (suite/'sensor_calls.jsonl').read_text(encoding='utf-8').splitlines()]
    for row in rows:
        ops=row['operations']
        ops['physical']['exclusive_s']=ops['physical']['inclusive_s']-ops['sample_miss']['inclusive_s']
    report['validation']['measurement_method']='고정한 현재 결과에 계측만 추가한 100명·600초 실행. 기존 센서·시야·도형 캐시 정책과 계산 순서는 유지함.'
    from compare_observation_optimization import save
    save(suite/'sensor_costs.json',report)
    with (suite/'sensor_calls.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({k:report[k] for k in ('sensor_calls','sensor_total_s','components','fds')},ensure_ascii=False),flush=True)


def self_test():
    class Clock:
        now=0.
        def __call__(self):return self.now
    clock=Clock();probe=SensorProbe(clock)
    def intersection(*_):clock.now+=2;return 'intersection'
    intersection=probe.geometry_wrapper(intersection,'intersection')
    def buffer(*_):intersection();clock.now+=3;return 'buffer'
    buffer=probe.geometry_wrapper(buffer,'buffer')
    class Field:
        def sample(self,*_):clock.now+=4;return {'value':1}
    Field.sample=probe.sample_wrapper(Field.sample)
    class Sensor:
        def __init__(self):self.fire=Field();self.cache=None
        def physical(self,*_):
            clock.now+=1
            if self.cache is None:self.cache=self.fire.sample()
            clock.now+=1
            return self.cache
        def sense(self,*_):
            intersection();self.physical();self.physical();buffer();clock.now+=1
            return {'unchanged':True}
    Sensor.physical=probe.physical_wrapper(Sensor.physical)
    Sensor.sense=probe.sense_wrapper(Sensor.sense)
    sensor=Sensor()
    assert sensor.sense(None,{},None,0)=={'unchanged':True}
    report=probe.report()
    assert report['sensor_total_s']==16
    assert {k:v['seconds'] for k,v in report['components'].items()}=={
        'wall_intersection':4.,'other_geometry_operators':3.,'fds_lookup_including_cache':8.,'remaining_sensor_work':1.}
    assert report['fds']['cache_hits']==1 and report['fds']['actual_array_sampling_s']==4.
    assert report['operators']['physical']['exclusive_s']==4.
    # Outside the requested scope operations must not enter the sensor totals.
    old=deepcopy(report)
    intersection();sensor.physical()
    assert probe.report()==old
    from persona_evac.integration import personal_space as module
    from shapely.geometry import box
    class Fire:
        def __init__(self):self.calls=[]
        def sample(self,floor,pos,now):
            self.calls.append((floor,tuple(pos),now))
            return {'extinction_per_m':now*.1,'temperature_c':20+10*now}
    def setup():
        geom=box(0,0,14,6).difference(box(5,1,7,5))
        fire=Fire();sensor=module.OccludedSensor({'room':{'geometry':geom}},{},{},fire)
        space=module.PersonalSpace({'uuid':'sensor-measurement','province':'부산','district':'부산-강서구'})
        r={'agent':{'agent_id':1},'area':'room','floor':'B2','pos':(2.,2.),'phase':'walking'}
        runtime={1:r,2:{'area':'room','floor':'B2','pos':(10.,2.),'phase':'walking'}}
        return sensor,space,r,runtime,fire
    expected=setup();actual=setup()
    states=[]
    for now in (0.,1.,2.):
        sensor,space,r,runtime,_=expected;r['pos']=(2.+now*.1,2.)
        result=sensor.sense(space,r,runtime,now)
        states.append(deepcopy((result,space.snapshot('room',now),space.observations,space.maps)))
    actual_probe=SensorProbe();restore=install_probe(module,Fire,actual_probe)
    try:
        for index,now in enumerate((0.,1.,2.)):
            sensor,space,r,runtime,_=actual;r['pos']=(2.+now*.1,2.)
            result=sensor.sense(space,r,runtime,now)
            assert (result,space.snapshot('room',now),space.observations,space.maps)==states[index]
        assert actual[-1].calls==expected[-1].calls
        actual_report=actual_probe.report()
        assert actual_report['sensor_calls']==3
        assert actual_report['operators']['intersection']['calls']>0
        assert actual_report['fds']['cache_miss_sample_calls']>0
    finally:
        restore()
    print('중첩 시간 중복 제외·FDS 캐시 적중 구분·센서 외부 제외·원본 결과와 표본 호출 순서 보존 검사 통과',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--refresh-report',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    if args.suite is None:parser.error('--suite가 필요합니다.')
    if args.refresh_report:refresh_report(args.suite);return
    from compare_observation_optimization import run,verify_reference,save,sha
    from persona_evac.integration import personal_space as module
    from persona_evac.integration.fds_environment import FireField
    probe=SensorProbe();restore=install_probe(module,FireField,probe)
    source=inspect.getsource(module.OccludedSensor.visible)
    try:
        run(args.suite,'after',600)
        verify_reference(args.suite)
    finally:
        restore()
        report=probe.report()
        save(args.suite/'sensor_costs.json',report)
        with (args.suite/'sensor_calls.jsonl').open('w',encoding='utf-8') as stream:
            for row in probe.senses:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    measured=json.loads((args.suite/'after/measurement/report.json').read_bytes())
    validation=json.loads((args.suite/'comparison.json').read_bytes())
    root=Path('/app')
    code=json.loads((args.suite/'after/measurement/code_hashes.json').read_bytes())
    report.update(simulation_wall_s=measured['phases']['total_s'],
        production_code_unchanged=all((root/p).exists() and sha(root/p)==h for p,h in code.items()),
        active_result_unchanged=sha(root/'output/gaehwasan/results.json')==sha(args.suite/'inputs/gaehwasan/results.json'),
        validation=validation,
        instrumentation={'geometry_methods': ['intersection','covers','distance','buffer'],
            'scope':'기존 sense 내부에서만 시간·횟수 집계. 메서드의 원래 인수와 반환값을 그대로 전달함.',
            'visible_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'cache_policy_changed':False,'sampling_order_changed':False})
    assert report['production_code_unchanged'] and report['active_result_unchanged']
    save(args.suite/'sensor_costs.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
