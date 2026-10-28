"""센서 캐시가 수치·벽 판정·시간별 연기 조회 순서를 보존하는지 검사한다."""
import math

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, box

from persona_evac.integration.personal_space import GRID, OccludedSensor, PersonalSpace, cell_pos
from test_fds_environment import field_fixture


def original_sample(field, floor, pos, now):
    low, high, w = field.indices(now)
    x, y = pos
    ix = int(np.argmin(abs(field.data['x']-x)))
    iy = int(np.argmin(abs(field.data['y']-y)))
    values = {key: float((1-w)*field.data[f'{floor}_{key}'][low, ix, iy]
                        + w*field.data[f'{floor}_{key}'][high, ix, iy]) for key in ('k', 'temperature')}
    k = values['k']
    return {'source':'fds', 'fire_time_s':round(now+field.offset,3),
            'extinction_per_m':round(k,5), 'temperature_c':round(values['temperature'],2),
            'visibility_m_capped_100':round(min(100,3/max(k,1e-9)),2),
            'speed_factor':max(.1,min(1.,1-.057*k/.706))}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_exact_node_cache_preserves_midpoint_ties_dtypes_and_time(tmp_path, dtype):
    field = field_fixture(tmp_path)
    field.data['x'] = np.array([0, 7.3, 120], dtype=dtype)
    field.data['y'] = np.array([0, 2.7, 20], dtype=dtype)
    def probes(axis):
        points = list(axis)
        for a,b in zip(axis,axis[1:]):
            midpoint = (float(a)+float(b))/2
            points += [midpoint, math.nextafter(midpoint,-math.inf), math.nextafter(midpoint,math.inf),
                       np.float32(midpoint), np.float64(midpoint)]
        return points
    positions = [(x,y) for x in probes(field.data['x']) for y in probes(field.data['y'])]
    for now in (0,5,5,10,2.3,np.float32(.1),float(np.float32(.1)),0):
        for floor in ('B1','B2'):
            for pos in positions:
                assert field.sample(floor,pos,now) == original_sample(field,floor,pos,now)
    assert field._nearest_x.cache_info().hits > 0
    assert field._nearest_x.cache_info().currsize <= 8192
    assert sum(len(v[2]) for v in field._sample_floors.values()) <= 18


def test_sample_cache_returns_independent_results_and_refreshes_replaced_fields(tmp_path):
    field = field_fixture(tmp_path)
    first = field.sample('B1',(60,0),5)
    first['extinction_per_m'] = -123
    assert field.sample('B1',(60,0),5) == original_sample(field,'B1',(60,0),5)
    field.data['B1_k'] = field.data['B1_k']+1
    assert field.sample('B1',(60,0),5) == original_sample(field,'B1',(60,0),5)
    field.data['x'] = np.array([0,100,120])
    assert field.sample('B1',(60,0),5) == original_sample(field,'B1',(60,0),5)
    field.offset += 1
    assert field.sample('B1',(60,0),5) == original_sample(field,'B1',(60,0),5)
    for now in (-2,11):
        with pytest.raises(ValueError,match='FDS'):
            field.sample('B1',(60,0),now)
    with pytest.raises(ValueError,match='outside'):
        field.sample('B1',(121,0),5)


def original_wall_face(sensor, area, p, k):
    boundary = sensor.areas[area]['geometry'].boundary
    q = cell_pos(k); qpoint = Point(q); line = LineString([p,q])
    intersections = line.intersection(boundary)
    if not intersections.is_empty and qpoint.distance(intersections)<GRID*.7:
        first = boundary.intersection(line)
        points = [g for g in getattr(first,'geoms',[first]) if g.geom_type=='Point']
        if points:
            edge = min(points,key=Point(p).distance)
            if qpoint.distance(edge)<GRID*.7:
                return edge.x,edge.y
    return None


def test_wall_cache_preserves_maps_observations_and_smoke_query_order():
    class Fire:
        def __init__(self):self.calls=[]
        def sample(self,floor,pos,now):
            self.calls.append((floor,tuple(pos),now))
            return {'extinction_per_m':2 if now==1 else .03 if pos[0]>6 else 0,
                    'temperature_c':20}
    geometry = Polygon(box(0,0,14,10).exterior.coords,[box(5,3,7,7).exterior.coords])
    fires = [Fire(),Fire()]
    sensors = [OccludedSensor({'room':{'geometry':geometry}},{},{},f) for f in fires]
    sensors[1]._static_wall_face = lambda area,p,k: original_wall_face(sensors[1],area,p,k)
    spaces = [PersonalSpace({'uuid':'sensor-exact','province':'부산','district':'부산-강서구'}) for _ in fires]
    for now,pos in [(0,(2,4)),(0,(2,4)),(1,(2,4)),(2,(2.01,4)),(3,(2,4))]:
        outcomes=[]
        for sensor,space in zip(sensors,spaces):
            r={'area':'room','floor':'B2','pos':pos,'phase':'walking'}
            outcomes.append(sensor.sense(space,r,{1:r},now))
        assert outcomes[0] == outcomes[1]
        assert vars(spaces[0]) == vars(spaces[1])
        assert fires[0].calls == fires[1].calls
    assert sensors[0]._static_wall_face.cache_info().hits > 0
    assert sensors[0]._static_wall_face.cache_info().currsize <= 65536
