import json
from collections import deque
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from persona_evac.integration.fds_station_geometry import rasterize,solid_boxes,write_station_fds


def config():return json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf8'))


def test_mesh_covers_current_outline_and_boxes_reconstruct_exact_solids():
    c=config();g=rasterize(c);solid=g['solid']
    for floor in c['floors']:
        for ring in floor['polygons']:
            x0,y0,x1,y1=Polygon(ring).bounds
            assert g['x0']<=x0<=x1<=g['x1'] and g['y0']<=y0<=y1<=g['y1']
    rebuilt=np.zeros_like(solid)
    for x0,x1,y0,y1,z0,z1 in solid_boxes(solid):rebuilt[z0:z1,x0:x1,y0:y1]=True
    assert np.array_equal(rebuilt,solid)


def test_fire_air_reaches_both_upper_levels_and_outdoors():
    c=config();g=rasterize(c);solid=g['solid']
    x,y,_,_=c['fire']['rect'];start=(0,x,y-g['y0'])
    assert not solid[start]
    seen={start};queue=deque([start])
    while queue:
        p=queue.popleft()
        for axis in range(3):
            for direction in (-1,1):
                q=list(p);q[axis]+=direction;q=tuple(q)
                if any(q[k]<0 or q[k]>=solid.shape[k] for k in range(3)) or q in seen or solid[q]:continue
                seen.add(q);queue.append(q)
    for fid,height in [('B1',5),('F1',9)]:
        assert any(p[0]==height for p in seen),fid
    assert any(p[0]==11 for p in seen)  # outside air at the open top boundary


def test_deck_has_new_full_width_exits_gates_and_only_actual_floor_openings(tmp_path):
    c=config();deck=tmp_path/'gaehwasan.fds';report=write_station_fds(c,deck)
    text=deck.read_text(encoding='utf8')
    assert report['gate_bodies']==14 and report['cells']==129600
    assert text.count("&OBST ID='GATE_BODY_")==14
    assert "MB='ZMAX', SURF_ID='OPEN'" in text
    assert report['upper_opening_areas_m2']['F1']>100
    assert all(len(points)==2 for points in report['exit_boundaries'].values())
    # Full-height rooms and gate bodies have distinct height representations.
    assert ',4,5, PERMIT_HOLE=.FALSE.' in text
