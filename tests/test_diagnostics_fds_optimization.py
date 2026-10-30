"""진단 순서와 FDS 보간 수치 보존을 엔진 진행 없이 확인한다."""
from copy import deepcopy
import math
import random

import jupedsim as jps
import numpy as np
import pytest
from shapely.geometry import box,Point

from persona_evac.integration.station_dynamics import diagnose_people,RADIUS,DT
from persona_evac.integration.fds_environment import _stair_path_metrics
from persona_evac.llm.ollama import LocalOllama


@pytest.fixture(autouse=True)
def no_simulation(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('계산 보류: 엔진·추론 실행 금지')
    monkeypatch.setattr(jps.Simulation,'iterate',forbidden)
    monkeypatch.setattr(LocalOllama,'generate',forbidden)


def original(runtime,before,areas,now,min_gap,overlap_steps,pair_history,rank_changes):
    active=[(aid,r) for aid,r in runtime.items() if aid in before]
    for index,(aid,a) in enumerate(active):
        if not areas[a['area']]['geometry'].covers(Point(a['pos'])):
            raise RuntimeError('JuPedSim 에이전트가 보행 가능한 공간을 벗어났습니다.')
        for bid,b in active[index+1:]:
            if a['area']!=b['area']:continue
            gap=math.dist(a['pos'],b['pos'])-2*RADIUS
            min_gap=min(min_gap,gap);overlap_steps+=int(gap<-.01)
            if a['phase']!='walking' or b['phase']!='walking' or a['target']!=b['target']:continue
            dest=a['goal'];diff=math.dist(a['pos'],dest)-math.dist(b['pos'],dest)
            key=(aid,bid,a['area'],a['target']);order=1 if diff>.5 else -1 if diff<-.5 else 0
            old_order=pair_history.get(key)
            if order and old_order and order!=old_order:
                rank_changes.append({'time_s':round(now+DT,8),'floor':a['floor'],
                                     'passing_agent':aid if order<0 else bid,
                                     'passed_agent':bid if order<0 else aid,'target_stair':a['target']})
            if order:pair_history[key]=order
    return min_gap,overlap_steps


@pytest.mark.parametrize('count',[0,1,10,100])
@pytest.mark.parametrize('area_count',[1,5])
def test_diagnostics_interleaved_areas_order_changes_goals_and_inactive(count,area_count):
    rng=random.Random(20261029);ids=list(range(count));rng.shuffle(ids)
    areas={i:{'geometry':box(-100,-100,100,100)} for i in range(area_count)}
    records={aid:{'area':aid%area_count,'floor':str(aid%area_count),'pos':(rng.random()*10,rng.random()*10),
                  'phase':'walking','target':str(aid%2),'goal':(rng.random()*20,rng.random()*20)} for aid in ids}
    histories=[{},{}];changes=[[],[]];stats=[(math.inf,0),(math.inf,0)]
    for now in range(8):
        if now:
            for aid,r in records.items():
                r['pos']=(rng.random()*10,rng.random()*10)
                r['phase']='queueing' if (aid+now)%9==0 else 'walking'
                r['area']=(aid+now)%area_count
        before={aid:None for aid in ids if (aid+now)%7!=0}
        for i,fn in enumerate((original,diagnose_people)):
            stats[i]=fn(records,before,areas,now,*stats[i],histories[i],changes[i])
        assert stats[0]==stats[1] and histories[0]==histories[1] and changes[0]==changes[1]
    if count==100:assert changes[0]


def test_diagnostic_thresholds_and_outside_error():
    areas={0:{'geometry':box(-10,-10,10,10)}}
    for distance in (.5,math.nextafter(.5,0),math.nextafter(.5,1),2*RADIUS-.01,0):
        records={aid:{'area':0,'floor':'B1','phase':'walking','target':'S','goal':(0,0),'pos':p}
                 for aid,p in [(9,(0,0)),(2,(distance,0)),(5,(2,0))]}
        h1={(9,2,0,'S'):1};h2=deepcopy(h1);a=[];b=[]
        assert original(records,records,areas,1,math.inf,0,h1,a)==diagnose_people(records,records,areas,1,math.inf,0,h2,b)
        assert a==b and h1==h2
    records[2]['pos']=(11,0)
    for fn in (original,diagnose_people):
        with pytest.raises(RuntimeError,match='보행 가능한 공간'):fn(records,records,areas,0,math.inf,0,{},[])


def test_stair_cache_content_change_zero_segments_and_original_sum_order():
    _stair_path_metrics.cache_clear()
    paths=[[(1.,2.,0.),(1.,2.,0.),(4.,6.,2.),(8.,6.,4.)],[(1.,2.,0.),(4.,6.,2.00000001),(8.,6.,4.)]]
    for path in paths+paths:
        points,lengths,total=_stair_path_metrics(tuple(map(tuple,path)))
        expected=np.asarray(path,dtype=float);distances=np.linalg.norm(np.diff(expected,axis=0),axis=1)
        assert np.array_equal(points,expected) and np.array_equal(lengths,distances) and total==sum(distances)
        assert not points.flags.writeable and not lengths.flags.writeable
    assert _stair_path_metrics.cache_info().misses==2 and _stair_path_metrics.cache_info().hits==2


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('density',[0,.1,1])
def test_selected_volume_preserves_values_threshold_and_order(dtype,density):
    rng=np.random.default_rng(20261029);raw=rng.random((4,20,15,8)).astype(dtype)
    raw[:,rng.random(raw.shape[1:])>=density]=0
    raw[:,0,0,0]=.003
    indices=np.argwhere(raw.max(axis=0)>.003);selection=tuple(indices.T)
    for w in (0,.1,np.float64(.5),math.nextafter(1,0),1):
        full=(1-w)*raw[0]+w*raw[1]
        expected=np.array([full[ix,iy,iz] for ix,iy,iz in indices],dtype=full.dtype)
        selected=(1-w)*raw[0][selection]+w*raw[1][selection]
        assert selected.dtype==full.dtype and np.array_equal(selected,expected)
        assert [round(float(x),4) for x in selected]==[round(float(full[ix,iy,iz]),4) for ix,iy,iz in indices]
