"""Read-only metric audit of the current layout; no movement calculations."""
import hashlib
import json
import math
import statistics
from pathlib import Path

from shapely.geometry import Polygon,LineString,Point
from shapely.ops import unary_union

ROOT=Path(__file__).resolve().parents[1]


def measure():
    config_path=ROOT/'configs/scenarios/gaehwasan.json'
    c=json.loads(config_path.read_text(encoding='utf8'))
    nodes=c['nodes'];floors={f['id']:f for f in c['floors']};stairs={s['id']:s for s in c['stairs']}
    platforms=[]
    for area,ring in zip(('A','B'),floors['B2']['polygons']):
        polygon=Polygon(ring);xmin,ymin,xmax,ymax=polygon.bounds
        widths=[]
        for i in range(1,1000):
            x=xmin+(xmax-xmin)*i/1000
            widths.append(polygon.intersection(LineString([(x,ymin-1),(x,ymax+1)])).length)
        platforms.append({'area':area,'length_x_m':xmax-xmin,'extent_y_m':ymax-ymin,
            'cross_section_y_width_m':{'min':min(widths),'median':statistics.median(widths),'max':max(widths)},
            'floor_area_m2':polygon.area})
    overall=unary_union([Polygon(p) for p in floors['B2']['polygons']]).bounds
    main=stairs['H_E'];lower=[f['path_xyz'][0][:2] for f in main['flights'] if f['role']=='lower']
    b1=[]
    for sid in ('A_W','A_E','B_W','B_E'):
        a=stairs[sid]['path_xyz'][-1][:2]
        distances=[math.dist(a,p) for p in lower]
        b1.append({'arrival_stair':sid,'arrival_xy':a,'to_lower_branch_centres_m':distances,
            'to_combined_stair_centre_m':math.dist(a,[nodes[main['from_node']][k] for k in ('x','y')])})
    f1=[]
    for flight in [f for f in main['flights'] if f['role']=='upper']:
        a=flight['path_xyz'][-1][:2];out={}
        for sid in ('E_1','E_2'):
            s=stairs[sid];edge=LineString(s['exit_boundary_xy']);mid=edge.interpolate(.5,normalized=True)
            out[sid]={'to_boundary_centre_m':Point(a).distance(mid),'nearest_boundary_m':Point(a).distance(edge)}
        f1.append({'side':flight['side'],'arrival_xy':a,'exits':out})
    source=json.loads((ROOT/'output/gaehwasan/persona_source.json').read_text(encoding='utf8'))
    agents=sorted(source['variants']['C'],key=lambda a:a['agent_slot'])[:100]
    speeds=[a['desired_speed_mps'] for a in agents]
    report={'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'basis':'현재 도면 근사 모델의 미터 좌표. 계단의 도착·출발 단면 중심 사이 수평 직선거리이며 우회·대기·층고는 제외.',
        'platforms':platforms,'b2_combined_extent_xy_m':[overall[2]-overall[0],overall[3]-overall[1]],
        'b1_straight_distances':b1,'f1_straight_distances':f1,
        'body_and_gate':{'collision_diameter_m':.44,'rendered_max_body_width_m':.50,
            'selected_outline_width_m':.50*1.16,'selected_ring_diameter_m':1.16,
            'gate_width_m':c['fare_gates'][0]['width_m'],
            'physical_clearance_each_side_m':(c['fare_gates'][0]['width_m']-.44)/2,
            'visual_clearance_each_side_m':(c['fare_gates'][0]['width_m']-.50)/2},
        'desired_speed_mps':{'count':len(speeds),'min':min(speeds),'mean':statistics.mean(speeds),
            'median':statistics.median(speeds),'max':max(speeds),
            'hurry_multiplier':1.35,'hurry_mean':statistics.mean(min(2.2,v*1.35) for v in speeds),
            'hurry_min':min(min(2.2,v*1.35) for v in speeds),'hurry_max':max(min(2.2,v*1.35) for v in speeds),
            'stairs_configured_speed':main['speed_mps'],'gate_configured_speed':1.},
        'viewer':{'default_playback_multiplier':2,'horizontal_metric_scale':1,'floor_separation_multiplier':2.5},
        'movement_simulation_run':False}
    out=ROOT/'output/qa/station_dimensions/measurements.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':measure()
