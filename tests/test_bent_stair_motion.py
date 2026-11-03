"""H_E 계단참을 실제로 걸으며 엔진·3D 좌표의 간격을 함께 확인한다."""
import itertools
import ast
import json
import math
from pathlib import Path

from shapely.geometry import LineString,box

from persona_evac.integration.station_connectors import compile_network,NativePassage
from persona_evac.integration.station_dynamics import make_simulation,parameters


def test_reserved_landing_body_handoff_clears_with_nearby_real_pedestrian():
    import jupedsim as jps
    from persona_evac.integration import station_dynamics as dynamics
    sim=make_simulation(box(0,0,12,6))

    def standing(pos):
        stage=sim.add_waypoint_stage(pos,.07)
        journey=sim.add_journey(jps.JourneyDescription([stage]))
        return sim.add_agent(parameters(pos,journey,stage,0.))

    guard=standing((2,2));neighbor=standing((2.48,2))
    tree=ast.parse(Path(dynamics.__file__).read_text(encoding='utf-8'))
    body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='simulate')
    helper=next(n for n in body.body if isinstance(n,ast.FunctionDef) and n.name=='add_person')
    env={**vars(dynamics),'areas':{'hall':{'sim':sim,'center_geometry':box(.24,.24,11.76,5.76)}},
         'native':True,'now':0}
    exec(compile(ast.Module(body=[helper],type_ignores=[]),'<arrival_helper>','exec'),env)
    runtime={'agent':{'desired_speed_mps':1},
             'landing_guard':{'area':'hall','jid':guard,'position':(2,2)}}
    before=tuple(sim.agent(guard).position)
    assert env['add_person'](runtime,'hall',(2,2),'next-observed-point')
    assert runtime['jid']==guard and sim.agent_count()==2
    assert tuple(sim.agent(guard).position)==before
    stage=sim.add_waypoint_stage((2,4.5),.07)
    journey=sim.add_journey(jps.JourneyDescription([stage]))
    sim.switch_agent_journey(guard,journey,stage)
    sim.agent(guard).model.desired_speed=1.
    previous=before
    for _ in range(120):
        sim.iterate()
        pos=tuple(sim.agent(guard).position)
        assert math.dist(previous,pos)<=.051
        assert math.dist(pos,sim.agent(neighbor).position)>=.43
        assert sim.agent_count()==2
        previous=pos
    assert previous[1]>4 and tuple(sim.agent(neighbor).position)==(2.48,2)


def test_bent_stair_engine_follows_landing_without_projection_overlap():
    config=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf-8'))
    connector=compile_network(config)['connectors']['H_E']
    passage=NativePassage(connector,make_simulation,parameters)
    assert all(b['bent_path'] for b in passage.branches)
    pending=list(range(1,37));completed=[];minimum=math.inf;pair_checks=0
    for tick in range(5000):
        if tick%6==0:
            for lane in range(connector['parallel_lanes']):
                if pending and passage.enter(pending[0],lane,connector['speed_mps']):
                    pending.pop(0)
        for aid in list(passage.occupants):passage.release_landing(aid)
        passage.advance()
        samples={aid:passage.sample(aid)[2] for aid in passage.occupants}
        for a,b in itertools.combinations(samples.values(),2):
            minimum=min(minimum,math.dist(a,b))
            assert math.dist(a,b)>=.44-.01
            pair_checks+=1
        for aid in list(passage.occupants):
            if passage.ready(aid):passage.release(aid);completed.append(aid)
        if not pending and not passage.occupants:break
    assert len(completed)==36 and pair_checks>1000
    assert minimum>=.43


def test_bent_stair_routing_cannot_cross_between_overlapping_floors():
    import jupedsim as jps
    config=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf-8'))
    connector=compile_network(config)['connectors']['H_E']
    passage=NativePassage(connector,make_simulation,parameters)
    for branch in passage.branches:
        bent=branch['bent_path'];geometry=bent.geometry(connector['native_branch_width_m'])
        source=bent.path[0][:2];dest=bent.path[-1][:2]
        assert not geometry.covers(LineString([source,dest]))
        route=jps.RoutingEngine(geometry).compute_waypoints(source,dest)
        assert max(p[0] for p in route)>141
        assert LineString(route).length>2*math.dist(source,dest)
