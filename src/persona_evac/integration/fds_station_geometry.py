"""Rasterize current station solids and stair headspace on the FDS mesh."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from shapely import contains_xy
from shapely.geometry import LineString,Polygon,Point
from shapely.ops import unary_union

from persona_evac.integration.station_layout import stair_path,view_geometry


def rasterize(config):
    floors={f['id']:f for f in config['floors']}
    footprints={fid:unary_union([Polygon(p) for p in f['polygons']]) for fid,f in floors.items()}
    bounds=unary_union(list(footprints.values())).bounds
    # One-metre cells; dimensions keep efficient 2/3/5 FFT factors.
    x0,x1=0,180;y0=6*math.floor(min(0,bounds[1]-2)/6)
    ny=6*math.ceil((max(52,bounds[3]+2)-y0)/6);y1=y0+ny
    xx,yy=np.meshgrid(np.arange(x0+.5,x1),np.arange(y0+.5,y1),indexing='ij')
    mask=lambda g:contains_xy(g,xx,yy)
    solid=np.zeros((12,x1-x0,ny),dtype=bool)
    thermal_rooms={};openings={}
    views=view_geometry(config,1)
    for fid,f in floors.items():
        z=int(f['elevation_m']);footprint=footprints[fid]
        obstacles=[Polygon(o['polygon']) for o in f['obstacles'] if o.get('kind') not in {'fare_gate','fare_barrier'}]
        rooms=unary_union(obstacles);thermal_rooms[fid]=rooms
        if fid=='B2':footprint=footprint.union(Polygon(config['train']['track_polygon']))
        if fid!='F1':solid[z:z+3]|=~mask(footprint)
        solid[z:z+3]|=mask(rooms)
        for wall in f.get('walls',[]):
            wall_mask=mask(Polygon(wall['polygon']).buffer(.45))
            solid[z:z+max(1,round(wall['height_m']))]|=wall_mask
        if fid=='F1':
            # Exterior air surrounds F1. The blue perimeter is a solid wall
            # except at the two actual full-width exit boundary segments.
            exits=unary_union([LineString(s['exit_boundary_xy']).buffer(.8,cap_style=3)
                              for s in config['stairs'] if s.get('kind')=='exit'])
            exterior_wall=footprint.boundary.buffer(.52).difference(exits)
            solid[z:z+3]|=mask(exterior_wall)
            solid[11]|=mask(footprint)  # ceiling; outdoor top remains open
        incoming=[Polygon(s['floor_opening_polygon']) for s in config['stairs']
                  if config['nodes'][s['to_node']]['floor']==fid and s.get('floor_opening_polygon')]
        slab=unary_union([Polygon(p['outer'],p['holes']) for p in views[fid]])
        openings[fid]=unary_union(incoming).difference(slab)
    # Open only the space above a stair flight, rather than punching a full
    # vertical shaft through every room under its plan projection.
    treads=np.zeros_like(solid)
    for s in config['stairs']:
        if s.get('kind')=='exit':continue
        flights=s.get('flights') or [{'path_xyz':stair_path(config,s),'width_m':s['width_m']}]
        for flight in flights:
            for a,b in zip(flight['path_xyz'],flight['path_xyz'][1:]):
                dx,dy=b[0]-a[0],b[1]-a[1];length2=dx*dx+dy*dy
                if length2<1e-9:continue
                t=np.clip(((xx-a[0])*dx+(yy-a[1])*dy)/length2,0,1)
                height=a[2]+t*(b[2]-a[2])
                footprint=LineString([a[:2],b[:2]]).buffer(flight['width_m']/2,cap_style=2)
                covered=mask(footprint)
                top=np.ceil(height-1e-7)
                for k in range(12):
                    solid[k]&=~(covered&(k+.5>=height)&(k+.5<height+2.5))
                    treads[k]|=covered&(k+.5>=top-1)&(k+.5<top)
        for landing in s.get('landings',[]):
            covered=mask(Polygon(landing['polygon']));z=landing['elevation_m']
            for k in range(12):
                solid[k]&=~(covered&(k+.5>=z)&(k+.5<z+2.5))
                treads[k]|=covered&(k+.5>=z-1)&(k+.5<z)
    # Structural slabs keep their actual rectangular upper openings. Room
    # foundations remain solid even when near a stair cutout.
    for fid in ('B1','F1'):
        z=int(floors[fid]['elevation_m'])
        solid[z-1]=~mask(openings[fid])
    solid|=treads
    return {'solid':solid,'x0':x0,'x1':x1,'y0':y0,'y1':y1,
            'footprints':footprints,'openings':openings}


def solid_boxes(solid):
    """Merge equal horizontal runs, adjacent rows and consecutive heights."""
    rectangles=[]
    for layer in solid:
        active={};done=[]
        for y in range(layer.shape[1]+1):
            row=layer[:,y] if y<layer.shape[1] else np.zeros(layer.shape[0],dtype=bool)
            edges=np.flatnonzero(np.diff(np.r_[False,row,False].astype(int)))
            runs=set(zip(edges[::2],edges[1::2]))
            for key in list(active):
                if key not in runs:done.append((*key,active.pop(key),y))
            for key in runs:active.setdefault(key,y)
        rectangles.append(done)
    active={};boxes=[]
    for z in range(len(rectangles)+1):
        current=set(rectangles[z]) if z<len(rectangles) else set()
        for rect in list(active):
            if rect not in current:boxes.append((*rect,active.pop(rect),z))
        for rect in current:active.setdefault(rect,z)
    return boxes


def write_station_fds(config,path,audit_path=None):
    path=Path(path);grid=rasterize(config);ny=grid['y1']-grid['y0']
    lines=["&HEAD CHID='gaehwasan', TITLE='Current annotated station geometry, fire-only recomputation' /"]
    for a,b in ((0,45),(45,90),(90,135),(135,180)):
        lines.append(f"&MESH IJK=45,{ny},12, XB={a},{b},{grid['y0']},{grid['y1']},0,12 /")
    lines.extend([f"&TIME T_END={config['fire']['time_end_s']}.0 /",
        "&DUMP DT_SLCF=2.0, DT_HRR=2.0 /","&MISC TMPA=20.0 /",
        "&REAC FUEL='PROPANE', SOOT_YIELD=0.06, CO_YIELD=0.01 /"])
    for i,stage in enumerate(config['fire']['stages']):
        lines.append(f"&SURF ID='CAR_{i}', HRRPUA=1000.0, RAMP_Q='GROWTH_{i}', COLOR='RED' /")
        for t,value in sorted({0:0,stage['start_s']:0,stage['full_s']:1,config['fire']['time_end_s']:1}.items()):
            lines.append(f"&RAMP ID='GROWTH_{i}', T={t}.0, F={value}.0 /")
    boxes=solid_boxes(grid['solid'])
    for xmin,xmax,ymin,ymax,zmin,zmax in boxes:
        lines.append(f"&OBST XB={xmin},{xmax},{ymin+grid['y0']},{ymax+grid['y0']},{zmin},{zmax}, PERMIT_HOLE=.FALSE. /")
    # Waist-high gate bodies stay distinct from full-height rooms. Sub-cell
    # widths are snapped by FDS; this 1m run does not resolve a 0.65m channel.
    gates=[o for f in config['floors'] for o in f['obstacles'] if o.get('kind')=='fare_gate']
    for gate in gates:
        xmin,ymin,xmax,ymax=Polygon(gate['polygon']).bounds
        lines.append(f"&OBST ID='{gate['id']}', XB={xmin:.4f},{xmax:.4f},{ymin:.4f},{ymax:.4f},4,5, PERMIT_HOLE=.FALSE. /")
    for car in config['train']['cars']:
        a,b=car['a'],car['b'];axis=LineString([a,b]);outer=axis.buffer(car['width_m']/2,cap_style=2)
        walls=outer.difference(axis.buffer(.75,cap_style=2));xmin,ymin,xmax,ymax=outer.bounds
        for y in range(math.floor(ymin),math.ceil(ymax)):
            for x in range(math.floor(xmin),math.ceil(xmax)):
                if outer.contains(Point(x+.5,y+.5)):
                    lines.append(f'&OBST XB={x},{x+1},{y},{y+1},2,3 /')
                if walls.contains(Point(x+.5,y+.5)):
                    lines.append(f'&OBST XB={x},{x+1},{y},{y+1},0,2 /')
        dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy);nx,ny=-dy/length,dx/length
        for fraction in (.2,.5,.8):
            x=a[0]+fraction*dx+nx*car['width_m']/2;y=a[1]+fraction*dy+ny*car['width_m']/2
            lines.append(f'&HOLE XB={x-1.1:.3f},{x+1.1:.3f},{y-.9:.3f},{y+.9:.3f},0,2 /')
    for i,stage in enumerate(config['fire']['stages']):
        x,y,w,h=stage['rect'];lines.append(f"&VENT XB={x},{x+w},{y},{y+h},0,0, SURF_ID='CAR_{i}' /")
    lines.append("&VENT MB='ZMAX', SURF_ID='OPEN' /")
    for x in (0,180):lines.append(f"&VENT XB={x},{x},{grid['y0']},{grid['y1']},8,12, SURF_ID='OPEN' /")
    for y in (grid['y0'],grid['y1']):lines.append(f"&VENT XB=0,180,{y},{y},8,12, SURF_ID='OPEN' /")
    for height in config['fire']['slice_height_m'].values():
        for quantity in ('EXTINCTION COEFFICIENT','TEMPERATURE'):
            lines.append(f"&SLCF PBZ={height}, QUANTITY='{quantity}' /")
    for height in config['fire']['volume_slice_heights_m']:
        if height not in config['fire']['slice_height_m'].values():
            lines.append(f"&SLCF PBZ={height}, QUANTITY='EXTINCTION COEFFICIENT' /")
    # Volume fractions (mol/mol), including every breathing height on stairs.
    for height in sorted(set(config['fire']['volume_slice_heights_m']) | set(config['fire']['slice_height_m'].values())):
        for species in ('CARBON MONOXIDE','CARBON DIOXIDE','OXYGEN'):
            lines.append(f"&SLCF PBZ={height}, QUANTITY='VOLUME FRACTION', SPEC_ID='{species}' /")
    lines.append('&TAIL /');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(lines)+'\n',encoding='utf8')
    report={'mesh_cell_m':1,'cells':int(grid['solid'].size),'domain_x':[0,180],
        'domain_y':[grid['y0'],grid['y1']],'domain_z':[0,12],'solid_boxes':len(boxes),
        'gate_bodies':len(gates),'exit_boundaries':{s['id']:s['exit_boundary_xy'] for s in config['stairs'] if s['kind']=='exit'},
        'upper_opening_areas_m2':{fid:g.area for fid,g in grid['openings'].items()},
        'deck_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'limits':['1m 격자의 계단·곡선·벽 근사이며 0.65m 개찰 통로와 실제 단 높이는 해상하지 못함.',
            '방은 열전달 특성 미확인의 불연성 폐쇄 공간으로 근사. 실외 경계·층고·환기는 가정.'],
        'solid_cells_per_z':[int(layer.sum()) for layer in grid['solid']]}
    if audit_path:Path(audit_path).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return report
