"""기존 시야 도형 캐시를 그대로 실행하며 정확히 같은 시점의 중복을 계측한다."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time


class ViewProbe:
    def __init__(self, clock=time.perf_counter):
        self.clock=clock
        self.events=[]
        self.groups={}
        self.person_times=Counter()
        self.current=None

    def begin(self,controller,r,now,site):
        assert self.current is None, '시야 계산이 중첩되어 호출 구분을 확인해야 합니다.'
        aid=r['agent']['agent_id'];pos=tuple(r['pos'])
        key=(aid,now,r['area'],r['floor'],pos)
        group=self.groups.setdefault(key,{'calls':0,'builds':0,'geometry_hashes':set()})
        legacy_key=(aid,round(now,1),pos)
        cached=list(controller.local_geom_cache)
        event={'call':len(self.events)+1,'agent_id':aid,'time_s':now,'area':r['area'],'floor':r['floor'],
            'position':list(pos),'site':site,'repeated_exact_call':group['calls']>0,
            'cache_hit_before':legacy_key in controller.local_geom_cache,
            'previous_cached_agent_id':cached[-1][0] if cached else None,
            'geometry_built':False,'repeated_exact_build':False,'geometry_s':0.,'helper_s':0.}
        group['calls']+=1
        self.person_times[(aid,now)]+=1
        self.current=(key,group,event)
        return event

    @contextmanager
    def geometry_span(self):
        _,group,event=self.current
        assert not event['geometry_built'], '한 호출에서 도형 생성 구간이 여러 번 실행됐습니다.'
        event['geometry_built']=True
        event['repeated_exact_build']=group['builds']>0
        group['builds']+=1
        started=self.clock()
        try:
            yield
        finally:
            event['geometry_s']+=self.clock()-started

    def finish(self,result,started,error=None):
        elapsed=self.clock()-started
        _,group,event=self.current
        event['helper_s']=elapsed
        if error is not None:
            event['error']=repr(error)
        else:
            geometry_hash=hashlib.sha256(result[1].wkb).hexdigest()
            event['geometry_sha256']=geometry_hash
            group['geometry_hashes'].add(geometry_hash)
            assert event['cache_hit_before'] is not event['geometry_built']
        self.events.append(event)
        self.current=None

    def report(self):
        duplicates=[]
        for key,group in self.groups.items():
            if group['calls']<2:continue
            aid,now,area,floor,pos=key
            events=[e for e in self.events if e['agent_id']==aid and e['time_s']==now
                    and e['area']==area and e['floor']==floor and tuple(e['position'])==pos]
            duplicates.append({'agent_id':aid,'time_s':now,'area':area,'floor':floor,'position':list(pos),
                'calls':group['calls'],'builds':group['builds'],
                'same_geometry':len(group['geometry_hashes'])==1,
                'extra_geometry_build_s':sum(e['geometry_s'] for e in events if e['repeated_exact_build']),
                'sites':[e['site'] for e in events]})
        repeated=[e for e in self.events if e['repeated_exact_call']]
        extra=[e for e in self.events if e['repeated_exact_build']]
        differing_keys={key for key,g in self.groups.items() if len(g['geometry_hashes'])>1}
        safe_extra=[e for e in extra if (e['agent_id'],e['time_s'],e['area'],e['floor'],tuple(e['position'])) not in differing_keys]
        site_counts=Counter(e['site'] for e in self.events)
        return {'calls':len(self.events),'unique_person_time_keys':len(self.person_times),
            'same_person_time_repeat_calls':sum(v-1 for v in self.person_times.values()),
            'unique_exact_view_keys':len(self.groups),'same_exact_view_repeat_calls':len(repeated),
            'cache_hits':sum(not e['geometry_built'] for e in self.events),
            'geometry_builds':sum(e['geometry_built'] for e in self.events),
            'repeated_calls_served_by_cache':sum(not e['geometry_built'] for e in repeated),
            'geometry_builds_on_repeated_calls':sum(e['geometry_built'] for e in repeated),
            'extra_same_exact_view_builds':len(extra),
            'helper_total_s':sum(e['helper_s'] for e in self.events),
            'geometry_build_total_s':sum(e['geometry_s'] for e in self.events),
            'repeated_call_helper_s':sum(e['helper_s'] for e in repeated),
            'extra_geometry_build_s':sum(e['geometry_s'] for e in extra),
            'same_geometry_extra_builds':len(safe_extra),
            'same_geometry_extra_build_s':sum(e['geometry_s'] for e in safe_extra),
            'repeated_groups_with_different_geometry':len(differing_keys),
            'by_call_site':dict(site_counts),
            'duplicate_groups':sorted(duplicates,key=lambda d:-d['extra_geometry_build_s']),
            'definition':{'person_time_key':'인물 ID와 반올림하지 않은 정확한 모의 시각',
                'exact_view_key':'인물 ID·정확한 모의 시각·영역·층·정확한 좌표',
                'extra_build':'같은 exact_view_key에서 실제 도형 생성 블록이 두 번째 이후 실행된 횟수',
                'timing':'도형 생성 블록의 실제 경과 시간. 전체 함수 시간에는 주변 사람 검사와 새 관측 기록 생성도 포함됨',
                'limit':'중복 도형 시간은 재사용 가능성을 보는 측정값이며 실제 최적화 후 전체 단축률을 뜻하지 않음'}}


def install_probe(controller_module,probe):
    original=controller_module.PrivateController.visible_helper
    lines,start_line=inspect.getsourcelines(original)
    source=textwrap.dedent(''.join(lines))
    tree=ast.parse(source)
    target=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.If) and ast.unparse(node.test)=='key not in self.local_geom_cache':
            target.append(node)
    assert len(target)==1, '계측할 기존 캐시 미스 블록이 정확히 하나여야 합니다.'
    node=target[0]
    before_dump=ast.dump(tree,include_attributes=False)
    wrapped=ast.With(items=[ast.withitem(context_expr=ast.Call(
        func=ast.Attribute(value=ast.Name(id='_view_duplicate_probe',ctx=ast.Load()),
                           attr='geometry_span',ctx=ast.Load()),args=[],keywords=[]))],body=node.body)
    ast.copy_location(wrapped,node.body[0])
    node.body=[wrapped]
    # Removing the added timing context must exactly recover the original algorithm.
    node.body=wrapped.body
    assert ast.dump(tree,include_attributes=False)==before_dump
    node.body=[wrapped]
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree,start_line-1)
    controller_module.__dict__['_view_duplicate_probe']=probe
    namespace={}
    exec(compile(tree,original.__code__.co_filename,'exec'),controller_module.__dict__,namespace)
    instrumented=namespace['visible_helper']

    @functools.wraps(original)
    def measured(self,r,now):
        frame=sys._getframe(1)
        # The common phase timer may wrap this method once more.
        while frame is not None and frame.f_code.co_name=='measured' and Path(frame.f_code.co_filename).name=='profile_evacuation.py':
            frame=frame.f_back
        site=f'{Path(frame.f_code.co_filename).name}:{frame.f_lineno}:{frame.f_code.co_name}' if frame else 'unknown'
        probe.begin(self,r,now,site)
        started=probe.clock()
        try:
            result=instrumented(self,r,now)
        except BaseException as exc:
            probe.finish(None,started,exc)
            raise
        else:
            probe.finish(result,started)
            return result

    controller_module.PrivateController.visible_helper=measured
    return {'original_file':original.__code__.co_filename,'original_start_line':start_line,
        'original_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'algorithm_ast_preserved_when_timing_context_removed':True,
        'cache_policy_changed':False}


def self_test():
    class Clock:
        now=0.
        def __call__(self):return self.now
    class Geometry:
        def __init__(self,value):self.wkb=value
    class Controller:
        local_geom_cache={}
    clock=Clock();probe=ViewProbe(clock);controller=Controller()
    def call(aid,now,pos):
        r={'agent':{'agent_id':aid},'pos':pos,'area':'room','floor':'B2'}
        probe.begin(controller,r,now,'test')
        started=clock()
        key=(aid,round(now,1),pos)
        if key not in controller.local_geom_cache:
            with probe.geometry_span():
                clock.now+=2
                controller.local_geom_cache={key:Geometry(repr((aid,pos)).encode())}
        clock.now+=1
        probe.finish((None,controller.local_geom_cache[key]),started)
    call(1,10.,(1,2));call(2,10.,(3,4));call(1,10.,(1,2));call(1,10.,(1,2))
    call(1,10.,(1.1,2));call(1,10.04,(1.1,2))
    report=probe.report()
    assert report['calls']==6 and report['geometry_builds']==4 and report['cache_hits']==2
    assert report['same_person_time_repeat_calls']==3 and report['same_exact_view_repeat_calls']==2
    assert report['extra_same_exact_view_builds']==1 and report['extra_geometry_build_s']==2.
    assert report['repeated_call_helper_s']==4. and report['same_geometry_extra_build_s']==2.
    assert report['repeated_groups_with_different_geometry']==0
    # Different results for the same key must be exposed rather than called safe reuse.
    probe.groups[(1,10.,'room','B2',(1,2))]['geometry_hashes'].add('different')
    report=probe.report()
    assert report['repeated_groups_with_different_geometry']==1 and report['same_geometry_extra_build_s']==0
    # Exercise the actual method with instrumentation, including the original
    # single-entry cache and every emitted personal observation.
    from types import SimpleNamespace
    from shapely.geometry import box
    from persona_evac.integration import private_agents as module
    original=module.PrivateController.visible_helper
    class Memory:
        def __init__(self):self.rows=[]
        def record(self,now,kind,data,source='vision'):
            self.rows.append({'time_s':now,'kind':kind,'source':source,**data})
            return f'O{len(self.rows)}'
    def make_controller():
        c=module.PrivateController.__new__(module.PrivateController)
        c.runtime={aid:{'agent':{'agent_id':aid},'jid':aid,'area':'room','floor':'B2',
            'pos':(float(aid),0.),'phase':'walking'} for aid in (1,2)}
        c.minds={aid:SimpleNamespace(space=Memory()) for aid in (1,2)}
        c.areas={'room':{'sim':object(),'geometry':box(-10,-10,10,10)}}
        c.sensor=SimpleNamespace(visible=lambda *_:True)
        c.local_geom_cache={}
        return c
    plain=make_controller();measured=make_controller();actual_probe=ViewProbe()
    try:
        installation=install_probe(module,actual_probe)
        for aid in (1,2,1,1):
            expected=original(plain,plain.runtime[aid],10.)
            actual=measured.visible_helper(measured.runtime[aid],10.)
            assert expected[1].wkb==actual[1].wkb
            assert list(plain.local_geom_cache)==list(measured.local_geom_cache)
        assert all(plain.minds[aid].space.rows==measured.minds[aid].space.rows for aid in (1,2))
        assert actual_probe.report()['extra_same_exact_view_builds']==1
        assert installation['algorithm_ast_preserved_when_timing_context_removed']
    finally:
        module.PrivateController.visible_helper=original
        module.__dict__.pop('_view_duplicate_probe',None)
    print('정확한 시각·좌표 구분, 캐시 적중·재생성 시간 분리, 도형 불일치 탐지·원본 메서드 동작 보존 검사 통과',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if args.suite is None:parser.error('--suite가 필요합니다.')
    from compare_observation_optimization import run,verify_reference,save,sha
    from persona_evac.integration import private_agents as controller
    probe=ViewProbe()
    instrumentation=install_probe(controller,probe)
    suite=args.suite
    try:
        run(suite,'after',600)
        verify_reference(suite)
    finally:
        report=probe.report()
        report['instrumentation']=instrumentation
        save(suite/'view_duplicates.json',report)
        with (suite/'view_calls.jsonl').open('w',encoding='utf-8') as stream:
            for event in probe.events:
                stream.write(json.dumps(event,ensure_ascii=False)+'\n')
    validation=json.loads((suite/'comparison.json').read_bytes())
    measured=json.loads((suite/'after/measurement/report.json').read_bytes())
    report['validation']={'all_semantic_result_fields_identical':validation['all_semantic_result_fields_identical'],
        'all_reference_checks_passed':validation['all_reference_checks_passed'],
        'code_unchanged_during_run':measured['code_unchanged_during_run'],
        'inputs_and_caches_unchanged':measured['inputs_and_caches_unchanged']}
    report['simulation_wall_s']=measured['phases']['total_s']
    report['extra_build_percent_of_simulation']=100*report['extra_geometry_build_s']/report['simulation_wall_s']
    report['active_result_unchanged']=sha(Path('/app/output/gaehwasan/results.json'))==sha(suite/'inputs/gaehwasan/results.json')
    save(suite/'view_duplicates.json',report)
    compact={k:v for k,v in report.items() if k not in {'duplicate_groups','definition'}}
    print(json.dumps(compact,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
