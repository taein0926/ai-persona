"""정적 계산 재사용이 새 연기·지도·판단 입력을 가리지 않는지 검사한다."""
from types import SimpleNamespace

from shapely.geometry import box

from persona_evac.integration.personal_space import OccludedSensor, PersonalSpace
from persona_evac.integration.private_agents import PrivateController


def test_static_ray_reuse_preserves_smoke_sample_order_and_time_changes():
    class Fire:
        def __init__(self):
            self.calls=[]
        def sample(self,floor,pos,now):
            self.calls.append((floor,tuple(pos),now))
            return {'extinction_per_m':2 if now==1 else (.08 if pos[0]>3 else 0),
                    'temperature_c':20}
    fires=[Fire(),Fire()]
    sensors=[OccludedSensor({'room':{'geometry':box(0,0,12,5)}},{},{},fire) for fire in fires]
    # Compare cached wall predicates with the same exact predicates without memoization.
    sensors[1]._static_ray_covered=sensors[1]._ray_covered
    requests=[((1,2),(9,2),0),((1,2),(9,2),0),((1,2),(9,2),1),
              ((1,2),(9,2),2),((1.49,2),(5.51,2),2),((1.51,2),(5.49,2),2),
              ((1,2),(13,2),2),((1,2),(9,2),0)]
    results=[[sensor.visible('room','B2',p,q,now) for p,q,now in requests] for sensor in sensors]
    assert results[0]==results[1]
    assert results[0][0] and not results[0][2] and results[0][3]
    assert fires[0].calls==fires[1].calls
    assert sensors[0]._static_ray_covered.cache_info().hits>0


def test_private_search_cache_refreshes_without_new_cells_and_keeps_time_dependent_options(monkeypatch):
    sensor=OccludedSensor({'room':{'geometry':box(0,0,12,5)}},{},{})
    space=PersonalSpace({'uuid':'private-cache-test','province':'부산','district':'부산-강서구'})
    r={'area':'room','floor':'B2','pos':(2,2),'phase':'walking'}
    sensor.sense(space,r,{1:r},0)
    m=space.map_for('room')
    anchor=min(m['cells'],key=lambda k:abs(k[0]-4)+abs(k[1]-1))
    space.landmarks['S']={'id':'S','area':'room','floor':'B2','position':[8,2],
        'observed_at_s':0,'current_state':{'people':1,'hazard':{'smoke':'clear','heat':'normal'}},
        'source':'vision','observation_id':space.last_observation_ids[0],'kind':'up_stair'}
    space.visible_landmark_ids.add('S');m['landmark_links']['S']=[anchor]
    original=space.shortest;calls=[]
    def shortest(area,pos):
        calls.append((area,tuple(pos)))
        return original(area,pos)
    monkeypatch.setattr(space,'shortest',shortest)
    first=space.options('room',r['pos'],1,1)
    second=space.options('room',r['pos'],2,4)
    assert len(calls)==1
    a=next(o for o in first if o['id']=='S');b=next(o for o in second if o['id']=='S')
    assert a['state_is_current'] and not b['state_is_current']
    assert a['distance_m']==b['distance_m'] and a['walking_s']>b['walking_s']
    saved_cells=set(m['cells']);saved_revision=space.revision
    sensor.sense(space,r,{1:r},5)
    assert m['cells']==saved_cells and space.revision==saved_revision
    space.options('room',r['pos'],1,5)
    assert len(calls)==2
    space.options('room',(2.1,2),1,5)
    assert len(calls)==3


def test_navigation_omits_only_unused_snapshot_and_model_context_remains_complete(monkeypatch):
    sensor=OccludedSensor({'room':{'geometry':box(0,0,12,5)}},{},{})
    space=PersonalSpace({'uuid':'snapshot-test','province':'부산','district':'부산-강서구'})
    r={'area':'room','floor':'B2','pos':(2,2),'phase':'walking','target':'explore:none',
       'agent':{'agent_id':1,'desired_speed_mps':1,'distance_walked_m':0},'wait_streak':0}
    sensor.sense(space,r,{1:r},0)
    controller=PrivateController.__new__(PrivateController)
    controller.minds={1:SimpleNamespace(space=space,profile={},memory=lambda *_:[])}
    calls=[];original=space.snapshot
    def snapshot(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(space,'snapshot',snapshot)
    full=controller.observation(r,1)
    navigation=controller.observation(r,1,(),False,False)
    assert len(calls)==1
    assert full['private_knowledge'] is not None and navigation['private_knowledge'] is None
    assert {k:v for k,v in full.items() if k!='private_knowledge'}=={
        k:v for k,v in navigation.items() if k!='private_knowledge'}
    assert controller.observation(r,1)==full
    assert len(calls)==2
