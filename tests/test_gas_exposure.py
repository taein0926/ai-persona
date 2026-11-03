"""가스 농도 단위·누적 시간·정지 상태 및 엔진의 재이동 방지를 검증한다."""
from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from persona_evac.integration.gas_exposure import accumulate_exposure, fed_rate_per_s
from persona_evac.integration.station_connectors import NativePassage,compile_network
from persona_evac.integration.station_dynamics import make_simulation,parameters
from persona_evac.integration.private_agents import PrivateController
from persona_evac.integration.fds_environment import cache_obstruction_masks
from test_fds_environment import field_fixture
from station_fixtures import config

AIR={'co_ppm':0.,'co2_percent':.04,'o2_percent':20.9}
CO={'co_ppm':5000.,'co2_percent':.04,'o2_percent':20.9}


def test_obstruction_cache_preserves_time_axis_and_cell_centered_mode():
    class Mesh:
        def __init__(self):self.calls=0
        def get_obstruction_mask(self,times,cell_centered=False):
            self.calls+=1
            result=np.ones((len(times),3,3,3),dtype=bool)
            for i,t in enumerate(times):result[i,0,0,0]=t<2 if cell_centered else t<1
            return result
    mesh=Mesh();reference=Mesh();times=np.array([0.,1.,2.])
    cache_obstruction_masks(SimpleNamespace(meshes=[mesh]))
    for _ in range(4):
        for centered in (False,True):
            result=mesh.get_obstruction_mask(times,cell_centered=centered)
            assert np.array_equal(result,reference.get_obstruction_mask(times,cell_centered=centered))
            assert not result.flags.writeable
    assert mesh.calls==2
    assert np.array_equal(mesh.get_obstruction_mask([2.,0.]),reference.get_obstruction_mask([2.,0.]))
    assert mesh.calls==3


def person(phase='walking'):
    return {'agent':{'agent_id':1,'fed':0.,'incapacitated_at_s':None},'phase':phase,
            'floor':'B2','pos':(60.,10.),'navigation_blocked':'known_path_not_currently_visible'}


def test_fds_6111_light_work_reference_and_clean_air():
    assert fed_rate_per_s(AIR)==0
    # 4000 ppm CO, 0.04% CO2, normal oxygen: independently evaluated reference.
    rate=fed_rate_per_s({**CO,'co_ppm':4000.})
    assert 1/rate==pytest.approx(383.74496138116535,rel=1e-10)
    assert fed_rate_per_s({**CO,'co2_percent':4})>2*fed_rate_per_s(CO)
    assert fed_rate_per_s({**AIR,'o2_percent':10})>0


@pytest.mark.parametrize('phase',['walking','waiting','queueing','pre_moving','on_stairs'])
def test_every_unevacuated_phase_accumulates_and_crosses_once(phase):
    r=person(phase);dt=.05;rate=fed_rate_per_s(CO)
    count=math.ceil(1/(rate*dt));events=[]
    for tick in range(count+3):
        if accumulate_exposure(r,CO,round(tick*dt,8),round((tick+1)*dt,8)):
            events.append(deepcopy(r['agent']['incapacitation']))
    assert len(events)==1
    assert events[0]['time_s']==pytest.approx(count*dt)
    assert events[0]['fed']>=1
    assert events[0]['fed']-rate*dt<1
    assert events[0]['previous_phase']==phase
    assert events[0]['navigation_blocked']=='known_path_not_currently_visible'
    assert r['incapacitated']
    assert r['phase']==('on_stairs' if phase=='on_stairs' else 'incapacitated')
    assert r['agent']['fed']>events[0]['fed']


def test_threshold_includes_exactly_one_and_is_permanent(monkeypatch):
    monkeypatch.setattr('persona_evac.integration.gas_exposure.fed_rate_per_s',lambda _: .25)
    r=person()
    for t in range(3):assert not accumulate_exposure(r,CO,t,t+1)
    assert accumulate_exposure(r,CO,3,4)
    assert r['agent']['fed']==1 and r['agent']['incapacitated_at_s']==4
    # A later concentration decrease must not clear the health state or dose.
    monkeypatch.setattr('persona_evac.integration.gas_exposure.fed_rate_per_s',lambda _: 0.)
    assert not accumulate_exposure(r,AIR,4,5)
    assert r['incapacitated'] and r['agent']['fed']==1


def test_route_and_phase_change_keep_dose_and_evacuation_stops_exposure():
    r=person('waiting');accumulate_exposure(r,CO,0,1);first=r['agent']['fed']
    r.update(phase='walking',floor='B1',pos=(40.,20.))
    accumulate_exposure(r,CO,1,2)
    assert r['agent']['fed']==pytest.approx(2*first)
    r['phase']='evacuated';before=deepcopy(r)
    assert not accumulate_exposure(r,CO,2,3) and r==before


@pytest.mark.parametrize('reading',[{}, {**CO,'co_ppm':float('nan')},{**CO,'o2_percent':-1},{**CO,'co2_percent':101}])
def test_missing_or_invalid_gases_never_mean_clean_air(reading):
    with pytest.raises(ValueError,match='FED'):fed_rate_per_s(reading)


def test_duplicate_and_missing_exposure_intervals_are_rejected():
    r=person();accumulate_exposure(r,CO,0,1)
    for start,end in [(0,1),(2,3),(1,1)]:
        with pytest.raises(ValueError,match='구간'):accumulate_exposure(r,CO,start,end)


def test_xyz_sampling_converts_volume_fraction_and_uses_breathing_height(tmp_path):
    f=field_fixture(tmp_path)
    with pytest.raises(ValueError,match='CO'):f.require_gases()
    f.data['gas_z']=np.array([1.,3.,6.])
    f.data['gas_valid']=np.ones((3,3,3),dtype=bool)
    for key,value in [('co',.001),('co2',.02),('o2',.06)]:
        grid=np.full((2,3,3,3),value)
        grid[1,...]*=2
        grid[...,1]*=3
        f.data['gas_'+key]=grid
        for floor in ['B1','B2']:f.data[floor+'_'+key]=grid[...,0]
    f.require_gases()
    # Feet z=.5 -> breathing z=2: halfway between 1m and 3m slices.
    r=f.sample_gases_xyz((60,0,.5),5)
    assert r==pytest.approx({'co_ppm':3000,'co2_percent':6,'o2_percent':18})
    assert f.sample('B1',(60,0),5)['co_ppm']==pytest.approx(1500)
    with pytest.raises(ValueError,match='범위'):f.sample_gases_xyz((60,0,10),5)


def test_solid_node_zero_oxygen_is_never_treated_as_air(tmp_path):
    f=field_fixture(tmp_path)
    f.data.update(x=np.array([59.,60.,61.]),y=np.array([0.,1.,2.]),gas_z=np.array([1.,3.]),
                  gas_valid=np.ones((3,3,2),dtype=bool))
    f.data['gas_valid'][1,1,:]=False
    for key,value in [('co',.001),('co2',.02),('o2',.18)]:
        grid=np.full((2,3,3,2),value);grid[:,1,1,:]=0
        f.data['gas_'+key]=grid
        for floor in ['B1','B2']:f.data[floor+'_'+key]=grid[...,0]
    f.require_gases()
    assert f.sample_gases_xyz((60,1,.5),5)==pytest.approx({'co_ppm':1000,'co2_percent':2,'o2_percent':18})
    # Below a ceiling, the upper interpolation level can be entirely solid.
    # Use the air side rather than mixing in a sample from beyond the slab.
    f.data['gas_valid'][:,:,1]=False;f.data['gas_valid'][1,1,0]=True
    for key,value in [('co',.001),('co2',.02),('o2',.18)]:f.data['gas_'+key][:,1,1,0]=value
    f._gas_air_nodes.clear()
    assert f.sample_gases_xyz((60,1,.55),5)==pytest.approx({'co_ppm':1000,'co2_percent':2,'o2_percent':18})
    f.data['gas_valid'][:]=False;f._gas_air_nodes.clear()
    with pytest.raises(ValueError,match='공기 표본'):f.sample_gases_xyz((60,1,.5),5)


def test_incapacitated_passage_body_keeps_capacity_and_cannot_resume():
    c=config();s=compile_network(c)['connectors']['G_01']
    p=NativePassage(s,make_simulation,parameters)
    assert p.enter(1,0,1)
    for _ in range(4):p.advance()
    p.incapacitate(1);position=p.engine_position(1)
    assert not p.ready(1) and not p.can_enter(0)
    for _ in range(20):
        p.release_landing(1);p.advance(1.5)
        assert p.engine_position(1)==pytest.approx(position)
    assert 1 in p.occupants


def test_private_controller_never_restarts_incapacitated_agents():
    controller=PrivateController.__new__(PrivateController)
    controller.runtime={1:{**person('incapacitated'),'incapacitated':True},
                        2:{**person('on_stairs'),'incapacitated':True}}
    controller.minds={}
    controller.mode='off'
    # Access to minds or sensors here would fail; both stopped people must skip it.
    controller.tick(10)


def test_simulation_incapacitation_while_smoke_blocks_navigation(monkeypatch,tmp_path):
    from persona_evac.integration.station import StationRequest,simulate_station
    from station_fixtures import source
    c=config()
    class ToxicField:
        def __init__(self,*args):
            self.meta={'fire':c['fire']};self.times=np.array([0.,600.]);self.offset=0
        def require_gases(self):pass
        def indices(self,time):return (0,1,time/600)
        def sample(self,*args):
            return {'source':'fds','extinction_per_m':100.,'temperature_c':20.,'speed_factor':.1}
        def sample_gases_xyz(self,xyz,time):return {**AIR,'co_ppm':1e6}
        def stair_observations(self,*args):return []
        def replay(self,*args):return {'enabled':True,'frames':[{'time_s':0},{'time_s':5}]}
    monkeypatch.setattr('persona_evac.integration.station_dynamics.FireField',ToxicField)
    monkeypatch.setenv('PERSONA_EVAC_OUTPUT',str(tmp_path))
    result=simulate_station(source(),c,StationRequest(fire_mode='fds',duration_s=5,profile_method='keyword_candidates'))
    assert result['summary']['incapacitated_count']==10
    assert result['summary']['completed_count']==0
    for a in result['agents']:
        at=a['incapacitated_at_s']
        assert 0<at<5 and a['incapacitation']['fed']>=1
        # Includes the initial 0.5s reaction delay: CO + ambient CO2 reaches
        # FED=1 at 1.26s, rounded up to the first 0.05s threshold boundary.
        assert at==pytest.approx(1.3)
        assert a['noncompletion_cause']=='toxic_gas_exposure'
        assert a['ending_state']['phase']=='incapacitated'
        assert a['gas_exposure_until_s']==5 and a['fed']>a['incapacitation']['fed']
        assert a['incapacitation']['navigation_blocked']
        stopped=[s for s in a['segments'] if s['start']>=at]
        assert stopped and all(s['status']=='incapacitated' and s['from']==s['to'] for s in stopped)
        assert not any(d['agent_id']==a['agent_id'] and d['time_s']>=at for d in result['decisions'])
