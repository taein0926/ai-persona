"""시간 진행 없이 고정 도형·경로 재사용과 실시간 점유 검사를 검증한다."""
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import jupedsim as jps
import pytest
from shapely.geometry import Point, box

from persona_evac.integration import station_connectors as connectors
from persona_evac.integration import station_dynamics as dynamics


@pytest.fixture(autouse=True)
def prohibit_calculation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('계산 보류: iterate 호출 금지')
    monkeypatch.setattr(jps.Simulation, 'iterate', forbidden)


def original_position(path, distance, lateral=0):
    # 최적화 이전 수식: 허용오차 없이 좌표와 부동소수점 연산 순서를 비교한다.
    distance=max(0,min(distance,sum(math.dist(a,b) for a,b in zip(path,path[1:]))))
    normals=[]
    for a,b in zip(path,path[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1];flat=math.hypot(dx,dy)
        normals.append((-dy/flat,dx/flat) if flat else (0,1))
    vertex_normals=[normals[0]]
    for previous,following in zip(normals,normals[1:]):
        denominator=1+sum(a*b for a,b in zip(previous,following))
        vertex_normals.append(tuple((a+b)/denominator for a,b in zip(previous,following)) if denominator>.05 else following)
    vertex_normals.append(normals[-1])
    for index,(a,b) in enumerate(zip(path,path[1:])):
        length=math.dist(a,b)
        if distance<=length+1e-9:
            t=distance/length if length else 0
            dx,dy=b[0]-a[0],b[1]-a[1]
            nx,ny=[(1-t)*vertex_normals[index][k]+t*vertex_normals[index+1][k] for k in (0,1)]
            return [a[0]+t*dx+lateral*nx,a[1]+t*dy+lateral*ny,a[2]+t*(b[2]-a[2])]
        distance-=length
    return list(path[-1])


@pytest.fixture(scope='module')
def network():
    config=json.loads(Path('configs/scenarios/gaehwasan.json').read_bytes())
    assert config['simulation_enabled'] is True
    return connectors.compile_network(config)


def test_every_branch_matches_original_projection_exactly(network):
    paths=[p for s in network['connectors'].values() for p in s['native_paths']]
    paths += [[[0,0,0],[0,0,0],[0,0,3],[2,0,3]],
              [[0,0,0],[2,0,1],[0,0,2]], [[0,0,0],[0,0,3]]]
    for path in paths:
        prepared=connectors.PreparedPath(path)
        distances=[-1,0,prepared.length,prepared.length+1]
        distances += [prepared.length*i/37 for i in range(38)]
        boundary=0
        for a,b in zip(path,path[1:]):
            boundary+=math.dist(a,b)
            distances += [boundary,math.nextafter(boundary,-math.inf),
                          math.nextafter(boundary,math.inf),boundary-1e-9,boundary+1e-9]
        for distance in distances:
            for lateral in (-.71,0,.13,.71):
                expected=original_position(path,distance,lateral)
                assert prepared.position(distance,lateral)==expected
                assert connectors.path_position(path,distance,lateral)==expected


def test_fixed_landings_match_original_and_return_fresh_metadata(network,monkeypatch):
    for s in network['connectors'].values():
        for lane,spec in enumerate(s['native_lane_specs']):
            path=s['native_paths'][spec['branch']]
            for lateral in (None,-.17,.32):
                xyz=original_position(path,connectors.path_length(path),spec['offset'] if lateral is None else lateral)
                expected={**network['nodes'][s['to_node']],'x':xyz[0],'y':xyz[1]}
                assert connectors.connector_landing(s,network['nodes'],lane,lateral)==expected
    s=next(iter(network['connectors'].values()))
    original=connectors.connector_landing(s,network['nodes'],0)
    original['x']=-999
    assert connectors.connector_landing(s,network['nodes'],0)['x']!=-999
    def forbidden(*args,**kwargs):raise AssertionError('고정 도착 좌표 재계산')
    monkeypatch.setattr(connectors,'path_position',forbidden)
    connectors.connector_landing(s,network['nodes'],0)


def test_ready_uses_fresh_position_without_projection(network,monkeypatch):
    s=next(iter(network['connectors'].values()));spec=s['native_lane_specs'][0]
    path=s['native_paths'][spec['branch']];length=connectors.path_length(path)
    actor=SimpleNamespace(position=(0,spec['offset']))
    branch={'sim':SimpleNamespace(agent=lambda jid:actor),'length':length,
            'prepared_path':connectors.PreparedPath(path)}
    passage=connectors.NativePassage.__new__(connectors.NativePassage)
    passage.connector=s;passage.branches=[branch]*len(s['native_paths']);passage.occupants={1:(0,11)}
    passage.incapacitated=set()
    for x in (-1,0,length-.091,length-.09,length-.089,length,length+1):
        for lateral in (spec['offset'],spec['offset']+.09,spec['offset']+.1):
            actor.position=(x,lateral)
            progress=max(0,min(1,x/length))
            assert passage.sample(1)==(progress,lateral,original_position(path,progress*length,lateral))
            assert passage.ready(1)==(math.hypot((1-progress)*length,lateral-spec['offset'])<=.09)
    monkeypatch.setattr(branch['prepared_path'],'position',lambda *a:pytest.fail('도착 판정의 불필요한 좌표 변환'))
    assert passage.ready(1) is False
    actor.position=(length,spec['offset'])
    assert passage.ready(1) is True


def test_arrival_helpers_reuse_geometry_but_refresh_live_occupancy():
    # 실제 내부 함수를 분리해 호출한다. simulate()와 엔진 iterate()는 실행하지 않는다.
    tree=ast.parse(Path(dynamics.__file__).read_text(encoding='utf-8'))
    simulate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='simulate')
    helpers=[n for n in simulate.body if isinstance(n,ast.FunctionDef) and n.name in {'add_person','reserve_landing'}]
    actors=[];calls=[]
    sim=SimpleNamespace(agents=lambda:iter(actors),add_waypoint_stage=lambda *a:1,
        add_journey=lambda *a:1,add_agent=lambda *a:9,
        switch_agent_journey=lambda *a:calls.append(a))
    geom=box(0,0,10,10);center=geom.buffer(-dynamics.RADIUS-.02)
    area={'sim':sim,'center_geometry':center}
    env={**vars(dynamics),'areas':{'hall':area},'native':True,'now':0}
    exec(compile(ast.Module(body=helpers,type_ignores=[]),'<arrival_helpers>','exec'),env)
    dest={'area':'hall','x':2,'y':2}
    r={'agent':{'desired_speed_mps':1}}
    actors.append(SimpleNamespace(id=3,position=(2,2)))
    assert env['reserve_landing'](r,dest) is False
    assert env['add_person'](r,'hall',(2,2),'target') is False
    actors.clear()
    assert env['reserve_landing'](r,dest) is True
    actors.append(SimpleNamespace(id=9,position=(2,2)))
    assert env['add_person'](r,'hall',(2,2),'target') is True
    assert calls==[(9,1,1)] and 'landing_guard' not in r
    assert env['reserve_landing']({}, {'area':'hall','x':0,'y':0}) is False
    with pytest.raises(ValueError,match='공간 밖'):
        env['add_person'](r,'hall',(0,0),'target')


def test_reserved_landing_handoff_does_not_require_new_insertion_clearance():
    # 기존 몸체 사이 0.48m는 몸 지름 0.44m보다 넓다. 신규 삽입 여유
    # 0.52m를 이미 예약한 몸체의 소유권 전환에 다시 적용하면 교착된다.
    tree=ast.parse(Path(dynamics.__file__).read_text(encoding='utf-8'))
    simulate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='simulate')
    helper=next(n for n in simulate.body if isinstance(n,ast.FunctionDef) and n.name=='add_person')
    actors=[SimpleNamespace(id=9,position=(2,2)),SimpleNamespace(id=3,position=(2.48,2))]
    switches=[];added=[]
    sim=SimpleNamespace(agents=lambda:iter(actors),add_waypoint_stage=lambda *a:1,
        add_journey=lambda *a:1,add_agent=lambda *a:added.append(a),
        switch_agent_journey=lambda *a:switches.append(a))
    env={**vars(dynamics),'areas':{'hall':{'sim':sim,'center_geometry':box(.24,.24,9.76,9.76)}},
         'native':True,'now':10}
    exec(compile(ast.Module(body=[helper],type_ignores=[]),'<arrival_helper>','exec'),env)
    reserved={'agent':{'desired_speed_mps':1},'landing_guard':{'area':'hall','jid':9,'position':(2,2)}}
    assert env['add_person'](reserved,'hall',(2,2),'target') is True
    assert reserved['jid']==9 and 'landing_guard' not in reserved
    assert switches==[(9,1,1)] and added==[]
    assert [a.position for a in actors]==[(2,2),(2.48,2)]
    for invalid in ({'area':'other','jid':9,'position':(2,2)},
                    {'area':'hall','jid':9,'position':(3,2)}):
        with pytest.raises(ValueError,match='예약한'):
            env['add_person']({'agent':{'desired_speed_mps':1},'landing_guard':invalid},'hall',(2,2),'target')
    # 예약 몸체가 없는 신규 진입은 기존의 엄격한 점유 검사를 유지한다.
    assert env['add_person']({'agent':{'desired_speed_mps':1}},'hall',(2,2),'target') is False


def test_fixed_path_preparation_is_not_repeated_during_sampling(monkeypatch):
    path=[[0,0,0],[1,2,3],[4,5,3]]
    prepared=connectors.PreparedPath(path)
    expected=original_position(path,2,.1)
    def forbidden(*args):raise AssertionError('고정 구간 길이 재계산')
    monkeypatch.setattr(connectors.math,'dist',forbidden)
    assert prepared.position(2,.1)==expected
