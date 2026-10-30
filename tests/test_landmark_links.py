"""정적 연결 검사 재사용이 현재 관측과 판단 입력을 보존하는지 검증한다."""
import math
from types import SimpleNamespace

import jupedsim as jps
import pytest
from shapely.geometry import LineString, box

from persona_evac.integration.personal_space import OccludedSensor, PersonalSpace, cell_pos
from persona_evac.integration.private_agents import PrivateController, default_choice
from persona_evac.integration.private_strategy import resolve_private
from persona_evac.llm.private_decisions import project_private
from persona_evac.persona.experimental_profile import build_profile


@pytest.fixture(autouse=True)
def no_engine_or_inference(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('계산 보류: 엔진·새 추론 금지')
    monkeypatch.setattr(jps.Simulation,'iterate',forbidden)
    from persona_evac.llm.ollama import LocalOllama
    monkeypatch.setattr(LocalOllama,'generate',forbidden)


def original(sensor,area,seen,position):
    geom=sensor.areas[area]['geometry']
    return [k for k in seen if math.dist(cell_pos(k),position)<3.5
            and geom.covers(LineString([cell_pos(k),position]).buffer(.24))]


def test_current_seen_order_endpoint_and_area_are_not_cached():
    sensor=OccludedSensor({'a':{'geometry':box(0,0,12,8)},
                           'b':{'geometry':box(0,0,12,8).difference(box(5,0,6,8))}}, {}, {})
    seen=[(1,1),(2,1),(3,1),(4,1),(2,2)]
    for area in ('a','b'):
        for position in ([5.5,2],[math.nextafter(5.5,0),2],[4,2],[6,2]):
            for cells in (seen,list(reversed(seen)),seen[:2],[],seen[2:]):
                assert sensor.landmark_links(area,cells,position)==original(sensor,area,cells,position)
    assert (1,1) not in sensor.landmark_links('a',seen,[5.5,2])
    assert (1,1) in sensor.landmark_links('a',seen,[math.nextafter(5.5,0),2])
    result=sensor.landmark_links('a',seen,[4,2]);result.clear()
    assert sensor.landmark_links('a',seen,[4,2])==original(sensor,'a',seen,[4,2])
    sensor._static_landmark_link.cache_clear()
    assert sensor.landmark_links('b',[(2,1)],[7,2])==[]
    assert sensor.landmark_links('b',[(2,1)],[7,2])==[]
    assert sensor._static_landmark_link.cache_info().hits==1


def test_private_observations_candidates_and_judgments_match_with_changing_smoke():
    class Fire:
        def __init__(self):self.calls=[]
        def sample(self,floor,pos,now):
            self.calls.append((floor,tuple(pos),now))
            return {'extinction_per_m':2 if now==1 else 0,'temperature_c':20}
    geom=box(0,0,16,8).difference(box(7,4,8,8))
    nodes={'stair':{'area':'hall','floor':'B1','x':10,'y':2},
           'gate':{'area':'hall','floor':'B1','x':6,'y':2},
           'hidden':{'area':'hall','floor':'B1','x':10,'y':6},
           'out':{'area':'upper','floor':'F1','x':6,'y':2}}
    stairs={'S':{'from_node':'stair','to_node':'out'},
            'G':{'kind':'fare_gate','from_node':'gate','to_node':'out'},
            'H':{'from_node':'hidden','to_node':'out'}}
    fires=[Fire(),Fire()]
    sensors=[OccludedSensor({'hall':{'geometry':geom}},nodes,stairs,f) for f in fires]
    sensors[1].landmark_links=lambda area,seen,position:original(sensors[1],area,seen,position)
    spaces=[PersonalSpace({'uuid':'links-test','province':'부산','district':'부산-강서구'}) for _ in fires]
    profile=build_profile({'uuid':'links-test','province':'부산','district':'부산-강서구'},seed=20261029)
    for now,pos in ((0,(2,2)),(0,(2,2)),(1,(2,2)),(2,(4,2)),(3,(9,2)),(4,(2,2))):
        changes=[];observations=[]
        for sensor,space in zip(sensors,spaces):
            r={'area':'hall','floor':'B1','pos':pos,'phase':'walking','target':'explore:none',
               'agent':{'agent_id':1,'desired_speed_mps':1,'distance_walked_m':2},'wait_streak':0}
            other={'area':'hall','floor':'B1','pos':(5,2),'phase':'walking'}
            changes.append(sensor.sense(space,r,{1:r,2:other},now))
            controller=PrivateController.__new__(PrivateController)
            controller.minds={1:SimpleNamespace(space=space,profile=profile,memory=lambda *_:[])}
            observations.append(controller.observation(r,now))
        assert changes[0]==changes[1]
        assert vars(spaces[0])==vars(spaces[1])
        assert fires[0].calls==fires[1].calls
        assert observations[0]==observations[1]
        assert project_private(observations[0])==project_private(observations[1])
        assert resolve_private(observations[0],default_choice(observations[0]))==resolve_private(observations[1],default_choice(observations[1]))
    assert sensors[0]._static_landmark_link.cache_info().hits>0
