"""Shared illustrative stair paths; heights and detailed stair sections are assumed."""
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

HEAD_HEIGHT = 1.76
SLAB_THICKNESS = .30


def stair_path(config, stair):
    if stair.get('path_xyz'):
        return stair['path_xyz']
    heights={f['id']:f['elevation_m'] for f in config['floors']}
    a,b=[config['nodes'][stair[k]] for k in ('from_node','to_node')]
    p=[a['x'],a['y'],heights[a['floor']]]
    q=[b['x'],b['y'],heights.get(b['floor'],heights[a['floor']])]
    return [p,q]


def polygons(geom):
    return [{'outer':list(p.exterior.coords),'holes':[list(h.coords) for h in p.interiors]}
            for p in getattr(geom,'geoms',[geom]) if p.geom_type=='Polygon' and not p.is_empty]


def display_stair_path(config, stair):
    return stair_path(config,stair)


def room_foundations(floor):
    return unary_union([Polygon(o['polygon']) for o in floor.get('obstacles',[])
                        if o.get('label')!='기둥' and o.get('kind')!='fare_gate' and o.get('display',True)])


def wall_foundations(floor):
    return unary_union([Polygon(w['polygon']) for w in floor.get('walls',[])])


def display_floor_geometry(floor):
    """A structural slab includes room foundations; navigation excludes rooms."""
    return unary_union([*[Polygon(p) for p in floor['polygons']],room_foundations(floor),wall_foundations(floor)])


def head_opening(config,stair,spread=1):
    target=config['nodes'][stair['to_node']]['floor']
    if target=='OUTSIDE':return None
    if stair.get('floor_opening_polygon'):
        return Polygon(stair['floor_opening_polygon'])
    top=next(f['elevation_m'] for f in config['floors'] if f['id']==target)*spread
    threshold=top-SLAB_THICKNESS-HEAD_HEIGHT
    pieces=[]
    for flight in stair.get('flights',[{'path_xyz':display_stair_path(config,stair),'width_m':stair['width_m']}]):
        path=flight['path_xyz']
        for a,b in zip(path,path[1:]):
            za,zb=a[2]*spread,b[2]*spread
            if max(za,zb)<threshold:continue
            t=0 if za>=threshold else (threshold-za)/(zb-za)
            start=[a[i]+t*(b[i]-a[i]) for i in (0,1)]
            pieces.append(LineString([start,b[:2]]).buffer(flight['width_m']/2+.10,cap_style=2,join_style=2))
    from shapely.ops import unary_union
    return unary_union(pieces) if pieces else None


def view_geometry(config,spread):
    result={}
    for floor in config['floors']:
        geom=display_floor_geometry(floor)
        for s in config['stairs']:
            if config['nodes'][s['to_node']]['floor']==floor['id']:
                opening=head_opening(config,s,spread)
                if opening is not None:geom=geom.difference(opening)
        geom=geom.union(room_foundations(floor)).union(wall_foundations(floor))
        result[floor['id']]=polygons(geom)
    return result


def attach_display_geometry(config):
    config['display_stair_paths'] = {s['id']: display_stair_path(config,s) for s in config['stairs']}
    config['display3d'] = {f['id']:polygons(display_floor_geometry(f)) for f in config['floors']}
    config['display3d_openings'] = {f'{i/10:.1f}':view_geometry(config,i/10) for i in range(10,41)}
    return config
