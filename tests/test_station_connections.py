"""Connection preparation/ownership checks only: never advance JuPedSim."""
import json
import math
from collections import Counter
from pathlib import Path

import jupedsim as jps
import pytest
from shapely.geometry import Point,Polygon,LineString

from persona_evac.integration.station_connectors import compile_network,NativePassage,connector_landing,path_length,path_position
from persona_evac.integration.station_dynamics import make_simulation,parameters,pedestrian_geometry,RADIUS
from persona_evac.integration.station_geometry import floor_geometry


@pytest.fixture
def config():
    return json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf8'))


@pytest.fixture(autouse=True)
def prohibit_calculation(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('계산 보류: iterate 호출 금지')
    monkeypatch.setattr(jps.Simulation,'iterate',forbidden)


def test_native_domains_and_every_connection_have_valid_portals(config):
    network=compile_network(config)
    geometries=network['geometry']
    sims={name:make_simulation(g) for name,g in geometries.items()}
    counts={}
    for sid,s in network['connectors'].items():
        source=network['nodes'][s['from_node']];dest=network['nodes'][s['to_node']]
        valid=[]
        for lane,spec in enumerate(s['native_lane_specs']):
            end=connector_landing(s,network['nodes'],lane)
            start_ok=geometries[source['area']].buffer(-RADIUS-.02).covers(Point(spec['entry'][:2]))
            end_ok=dest['floor']=='OUTSIDE' or geometries[dest['area']].buffer(-RADIUS-.02).covers(Point(end['x'],end['y']))
            if start_ok and end_ok:valid.append(lane)
        counts[sid]=len(valid)
        assert valid,f'{sid}: 유효한 연결 지점 없음'
        passage=NativePassage(s,make_simulation,parameters)
        assert all(b['sim'].elapsed_time()==0 for b in passage.branches)
    assert len([sid for sid in counts if sid.startswith('G_')])==13
    assert len(sims)==5
    # Receivers of all four platform stairs are on the paid side. H_E is on
    # the other side: the only cross-area connectors are the 13 gate channels.
    for sid in ('A_W','A_E','B_W','B_E'):
        assert network['nodes'][network['connectors'][sid]['to_node']]['area']=='hall_paid'
    assert network['nodes'][network['connectors']['H_E']['from_node']]['area']=='hall'
    assert geometries['hall_paid'].intersection(geometries['hall']).area==0


def test_gate_ownership_is_exclusive_and_blocked_receiver_keeps_lock(config):
    network=compile_network(config);s=network['connectors']['G_01']
    passage=NativePassage(s,make_simulation,parameters)
    assert passage.enter(1,0,1)
    assert not passage.enter(2,0,1)
    assert not passage.enter(1,0,1)
    # Merely reaching/waiting for a receiver never releases ownership.
    assert 1 in passage.occupants and not passage.can_enter(0)
    assert passage.sample(1)[0]==0
    assert passage.branches[0]['sim'].elapsed_time()==0
    lane,jid=passage.occupants[1]
    sim=passage.branches[0]['sim']
    assert sim.agent(jid).stage_id==passage.branches[0]['stages'][lane][1]
    passage.release_landing(1)
    assert sim.agent(jid).stage_id==passage.branches[0]['final_stages'][lane][1]
    assert not passage.can_enter(0)  # reservation permission does not release gate
    passage.release(1)
    assert not passage.occupants


def test_split_stair_paths_and_destination_mapping(config):
    network=compile_network(config);s=network['connectors']['H_E']
    assert len(s['native_paths'])==2 and s['parallel_lanes']==12
    for lane,spec in enumerate(s['native_lane_specs']):
        path=s['native_paths'][spec['branch']]
        assert path_position(path,0,spec['offset'])==pytest.approx(spec['entry'])
        assert path_position(path,path_length(path),spec['offset'])==pytest.approx(spec['destination'])
        end=connector_landing(s,network['nodes'],lane)
        assert end['floor']=='F1' and end['area']=='ground'
        center=config['nodes'][s['from_node']]['y']
        assert abs(end['y']-center)>s['width_m']/2
        distance=0
        for a,b in zip(path,path[1:-1]):
            distance+=math.dist(a,b)
            before=path_position(path,distance-1e-6,spec['offset'])
            after=path_position(path,distance+1e-6,spec['offset'])
            assert math.dist(before,after)<1e-4  # flat turns have no lateral jumps


def test_exit_boundaries_are_updated_full_width_and_population_is_80_20(config):
    one=next(s for s in config['stairs'] if s['id']=='E_1')
    assert one['exit_width_m']==pytest.approx(151/4*4/15.12,abs=.001)
    floor=next(f for f in config['floors'] if f['id']=='F1')
    boundary=Polygon(floor['polygons'][0]).boundary
    for s in [s for s in config['stairs'] if s['kind']=='exit']:
        assert boundary.buffer(.0002).covers(LineString(s['exit_boundary_xy']))
    b2=next(f for f in config['floors'] if f['id']=='B2')
    for count in (10,30,60,100):
        points=config['spawn_points'][:count]
        assert Counter(p['area'] for p in points)=={'A':int(count*.8),'B':int(count*.2)}
        for p in points:
            geom=floor_geometry(b2,0 if p['area']=='A' else 1)
            assert geom.buffer(-.59).covers(Point(p['x'],p['y']))
        assert min(math.dist((p['x'],p['y']),(q['x'],q['y'])) for i,p in enumerate(points) for q in points[i+1:])>=1.2
    assert config['simulation_enabled'] is True and config['layout_status']=='ready'


def test_calculation_is_blocked_before_initialization(config):
    from persona_evac.integration.station import simulate_station
    config={**config,'simulation_enabled':False}
    with pytest.raises(ValueError,match='보류'):
        simulate_station({},config)
