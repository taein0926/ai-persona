"""Shared illustrative stair paths; heights and detailed stair sections are assumed."""
import math

from shapely.geometry import LineString

from persona_evac.integration.station_geometry import floor_geometry

HEAD_HEIGHT = 1.76
SLAB_THICKNESS = .30


def stair_path(config, stair):
    if stair.get('path_xyz'):
        return stair['path_xyz']
    heights={f['id']:f['elevation_m'] for f in config['floors']}
    a,b=[config['nodes'][stair[k]] for k in ('from_node','to_node')]
    p=[a['x'],a['y'],heights[a['floor']]]
    q=[b['x'],b['y'],heights.get(b['floor'],heights[a['floor']])]
    if stair['id']=='H_W':
        return [[p[0]+(q[0]-p[0])*t-1.1*math.sin(math.pi*t),
                 p[1]+(q[1]-p[1])*t,p[2]+(q[2]-p[2])*t] for t in [i/20 for i in range(21)]]
    if stair['id']=='H_E':
        mid=(p[2]+q[2])/2
        return [p,[p[0]+4.5,p[1],mid],[q[0]+4.5,q[1],mid],q]
    return [p,q]


def polygons(geom):
    return [{'outer':list(p.exterior.coords),'holes':[list(h.coords) for h in p.interiors]}
            for p in getattr(geom,'geoms',[geom]) if p.geom_type=='Polygon' and not p.is_empty]


def display_stair_path(config, stair):
    return stair_path(config,stair)


def head_opening(config,stair,spread=1):
    target=config['nodes'][stair['to_node']]['floor']
    if target=='OUTSIDE':return None
    top=next(f['elevation_m'] for f in config['floors'] if f['id']==target)*spread
    threshold=top-SLAB_THICKNESS-HEAD_HEIGHT
    path=display_stair_path(config,stair);pieces=[]
    for a,b in zip(path,path[1:]):
        za,zb=a[2]*spread,b[2]*spread
        if max(za,zb)<threshold:continue
        t=0 if za>=threshold else (threshold-za)/(zb-za)
        start=[a[i]+t*(b[i]-a[i]) for i in (0,1)]
        pieces.append(LineString([start,b[:2]]).buffer(stair['width_m']/2+.10,cap_style=2,join_style=2))
    from shapely.ops import unary_union
    return unary_union(pieces) if pieces else None


def view_geometry(config,spread):
    result={}
    for floor in config['floors']:
        geom=floor_geometry(floor)
        for s in config['stairs']:
            if config['nodes'][s['to_node']]['floor']==floor['id']:
                opening=head_opening(config,s,spread)
                if opening is not None:geom=geom.difference(opening)
        result[floor['id']]=polygons(geom)
    return result
