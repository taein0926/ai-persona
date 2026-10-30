"""엔진 진행 없이 연결 통로별 순서·집계 갱신을 확인한다."""
import random

from persona_evac.integration.station_dynamics import connector_occupants, connector_admissions


def original_occupants(runtime, connectors):
    return {sid:sorted([r for r in runtime.values() if r['phase']=='on_stairs' and r['target']==sid],
                       key=lambda r:-r['stair_distance']) for sid in connectors}


def original_admissions(runtime, connectors):
    return ({sid:sorted([r for r in runtime.values() if r['phase']=='queueing' and r['target']==sid],
                        key=lambda r:(r['arrived_at'],r['agent']['agent_id'])) for sid in connectors},
            {sid:sum(r['phase']=='on_stairs' and r['target']==sid for r in runtime.values()) for sid in connectors})


def test_groups_match_original_for_shuffled_people_and_ties():
    rng=random.Random(20261029)
    connectors={f'S{i}':{} for i in range(20)}
    for count in (0,1,20,100,500):
        for _ in range(10):
            ids=list(range(count));rng.shuffle(ids)
            runtime={i:{'agent':{'agent_id':i},'phase':rng.choice(['on_stairs','queueing','walking','waiting','evacuated']),
                        'target':rng.choice([*connectors,'unknown']),'arrived_at':rng.randrange(3),
                        'stair_distance':rng.randrange(3)} for i in ids}
            assert connector_occupants(runtime,connectors)==original_occupants(runtime,connectors)
            assert connector_admissions(runtime,connectors)==original_admissions(runtime,connectors)
            for group in connector_occupants(runtime,connectors).values():
                assert all(r is runtime[r['agent']['agent_id']] for r in group)


def test_departures_blocked_receivers_reroutes_and_new_arrivals_refresh_counts():
    connectors={'stairs':{},'gate':{}}
    def person(i,phase,target,arrival=0):
        return {'agent':{'agent_id':i},'phase':phase,'target':target,'arrived_at':arrival,'stair_distance':2}
    runtime={8:person(8,'on_stairs','stairs'),2:person(2,'on_stairs','stairs'),
             4:person(4,'on_stairs','gate'),9:person(9,'queueing','stairs',1),
             3:person(3,'queueing','stairs',1),7:person(7,'walking','gate')}
    moving=connector_occupants(runtime,connectors)
    assert [r['agent']['agent_id'] for r in moving['stairs']]==[8,2]
    # 첫 인물만 퇴장하고, 도착층이 막힌 두 번째 인물은 통로 점유를 유지한다.
    runtime[8].update(phase='walking',target='explore:none')
    runtime[4].update(phase='evacuated',target='gate')
    runtime[9].update(target='gate')  # 판단 이후의 목표 변경
    runtime[7].update(phase='queueing',arrived_at=1)
    waiting,counts=connector_admissions(runtime,connectors)
    assert counts=={'stairs':1,'gate':0}
    assert [r['agent']['agent_id'] for r in waiting['gate']]==[7,9]
    assert [r['agent']['agent_id'] for r in waiting['stairs']]==[3]
    assert (waiting,counts)==original_admissions(runtime,connectors)
    runtime[7].update(phase='on_stairs',stair_distance=0)
    next_waiting,next_counts=connector_admissions(runtime,connectors)
    assert next_counts['gate']==1 and [r['agent']['agent_id'] for r in next_waiting['gate']]==[9]
    assert connector_occupants(runtime,connectors)==original_occupants(runtime,connectors)


def test_unrelated_phases_do_not_require_connector_fields():
    runtime={1:{'phase':'walking'},2:{'phase':'evacuated'}}
    assert connector_occupants(runtime,{'gate':{}})=={'gate':[]}
    assert connector_admissions(runtime,{'gate':{}})==({'gate':[]},{'gate':0})
