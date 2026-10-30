import json
import math
from pathlib import Path

import pytest
from shapely.geometry import Point,box

from persona_evac.integration.station_layout import stair_path,head_opening,HEAD_HEIGHT,SLAB_THICKNESS
from persona_evac.integration.station_geometry import floor_geometry
from persona_evac.integration.station_dynamics import make_simulation,parameters,plan_yield,pedestrian_geometry,RADIUS
import jupedsim as jps


def test_stair_opening_starts_when_head_reaches_underside():
    c={'floors':[{'id':'low','elevation_m':0},{'id':'high','elevation_m':4}],
       'nodes':{'a':{'floor':'low','x':0,'y':0},'b':{'floor':'high','x':10,'y':0}}}
    s={'id':'s','from_node':'a','to_node':'b','width_m':2}
    for spread in (1,2.5,4):
        opening=head_opening(c,s,spread)
        x=(4*spread-SLAB_THICKNESS-HEAD_HEIGHT)/(4*spread)*10
        assert not opening.covers(Point(x-.05,0))
        assert opening.covers(Point(x+.05,0))


def test_drawing_split_stairs_return_after_two_flat_landing_turns():
    c=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf8'))
    s=next(s for s in c['stairs'] if s['id']=='H_E');p=stair_path(c,s)
    assert p[1][2]==p[2][2]==p[3][2]==p[4][2]==6
    assert len(s['branch_paths_xyz'])==2
    assert len([s for s in c['stairs'] if c['nodes'][s['from_node']]['floor']=='B1'])==1
    assert p[0][0]==c['nodes'][s['from_node']]['x']
    assert p[-1][1]==c['nodes'][s['to_node']]['y']
    assert len(c['train']['cars'])==8 and len(c['spawn_points'])==100
    from shapely.geometry import Polygon,LineString
    lower=[f for f in s['flights'] if f['role']=='lower']
    upper=[f for f in s['flights'] if f['role']=='upper']
    assert len(lower)==len(upper)==2
    assert all(f['width_m']==s['width_m']/2 for f in s['flights'])
    landing=Polygon(s['landings'][0]['polygon'])
    assert landing.bounds[3]-landing.bounds[1]==pytest.approx(2*s['width_m'])
    for low,high in zip(lower,upper):
        a,b=low['path_xyz'];c,d=high['path_xyz']
        assert b[0]==pytest.approx(landing.bounds[0])
        assert a[1]==b[1] and c[1]==d[1]
        assert (b[0]-a[0])*(d[0]-c[0])<0
        assert (c[1]-b[1])*low['side']>0
        for point in (b,c):
            assert landing.boundary.distance(Point(point[:2]))<1e-9
        mid_x=(landing.bounds[0]+landing.bounds[2])/2
        assert landing.covers(LineString([b[:2],[mid_x,b[1]],[mid_x,c[1]],c[:2]]))
    assert s['center_handrail_xyz'][0][1]==pytest.approx((lower[0]['path_xyz'][0][1]+lower[1]['path_xyz'][0][1])/2)
    assert all(a[2]!=b[2] for a,b in zip(s['center_handrail_xyz'],s['center_handrail_xyz'][1:]))
    perimeter=[]
    for rail in s['rail_paths_xyz']:
        for a,b in zip(rail,rail[1:]):
            if a[2]==b[2]:
                segment=LineString([a[:2],b[:2]])
                assert landing.boundary.buffer(1e-7).covers(segment)
                perimeter.append(segment)
    assert perimeter


def test_drawing_platform_stair_lengths_rooms_and_gate_height():
    c=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf8'))
    stairs=[s for s in c['stairs'] if c['nodes'][s['from_node']]['floor']=='B2']
    lengths=[sum(math.dist(a,b) for a,b in zip(s['path_xyz'],s['path_xyz'][1:])) for s in stairs]
    assert max(lengths)-min(lengths)<.001
    platform=c['floors'][0]
    # Centers of both between-stair rooms must remain excluded from walking.
    for x,y in [(100,29),(100,5)]:
        assert not floor_geometry(platform).covers(Point(x,y))
    hall=next(f for f in c['floors'] if f['id']=='B1')
    bodies=[o for o in hall['obstacles'] if o['label']=='개찰구 본체']
    assert bodies and all(o['height_m']==1 for o in bodies)
    assert c['fare_gates'] and all(g['passage_model']=='jupedsim_exclusive_channel' for g in c['fare_gates'])
    assert c['layout_status']=='ready' and c['simulation_enabled']
    assert len(hall['polygons'])==1


def test_drawing_arrival_pads_and_room_foundations_remain_solid():
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from persona_evac.integration.station_layout import view_geometry
    c=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf8'))
    views=view_geometry(c,2.5)
    slabs={fid:unary_union([Polygon(p['outer'],p['holes']) for p in parts]) for fid,parts in views.items()}
    for floor in c['floors']:
        for obstacle in floor['obstacles']:
            if obstacle.get('label')=='기둥' or obstacle.get('kind')=='fare_gate' or not obstacle.get('display',True):
                continue
            assert slabs[floor['id']].buffer(1e-7).covers(Polygon(obstacle['polygon']))
    for s in c['stairs'][:4]:
        a,b=[c['nodes'][s[key]] for key in ('from_node','to_node')]
        assert a['y']==b['y'] and s['portal_tangent']==[0,1]
        assert slabs['B1'].buffer(-.24).covers(Point(b['x'],b['y']))
        rect=s['floor_opening_polygon']
        assert len({p[0] for p in rect})==len({p[1] for p in rect})==2
        assert not Polygon(rect).covers(Point(b['x'],b['y']))
    from shapely.geometry import LineString
    for first,second in [('A_W','A_E'),('B_W','B_E')]:
        heads=[]
        for sid in (first,second):
            stair=next(s for s in c['stairs'] if s['id']==sid)
            n=c['nodes'][stair['to_node']]
            heads.append((n['x'],n['y']))
        assert slabs['B1'].buffer(-.24).covers(LineString(heads))
    assert slabs['B1'].covers(box(99,7.2,105,29.5))


def test_exit_plan_has_bent_first_stair_and_two_vestibules():
    c=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf8'))
    one,two=[next(s for s in c['stairs'] if s['id']==sid) for sid in ('E_1','E_2')]
    assert one['exit_form']=='bent_stair' and len(one['exit_boundary_xy'])==2
    assert one['path_xyz'][-1][0]<one['path_xyz'][0][0]
    assert two['exit_form']=='vestibule_passage'
    assert one['render_mode']==two['render_mode']=='floor_only'
    assert {v['exit_id'] for v in c['vestibules']}=={'E_1','E_2'}
    floor=next(f for f in c['floors'] if f['id']=='F1')
    for stair in (one,two):
        start=c['nodes'][stair['from_node']]
        assert floor_geometry(floor).buffer(-.24).covers(Point(start['x'],start['y']))


def test_drawing_walls_follow_public_boundaries_and_have_supported_bases():
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from persona_evac.integration.station_layout import view_geometry
    c=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf8'))
    views=view_geometry(c,2.5)
    for floor in c['floors']:
        slab=unary_union([Polygon(p['outer'],p['holes']) for p in views[floor['id']]])
        for wall in floor.get('walls',[]):
            assert wall['passenger_facing'] and wall['source_pdf_points']
            assert wall['height_m']==2.1 and wall['thickness_m']==.18
            assert slab.buffer(1e-7).covers(Polygon(wall['polygon']))
            assert wall['physical_integration']=='jupedsim_solid_boundary'
    hall=next(f for f in c['floors'] if f['id']=='B1')
    assert next(o for o in hall['obstacles'] if o['label']=='역무·통신실')['render_as']=='building'
    assert not any(o['label']=='상부 설비층 곡선 계단' for o in hall['obstacles'])
    assert {w['id'] for w in hall['walls']}=={'B1_N_EDGE','B1_S_EDGE'}
    assert len([o for o in hall['obstacles'] if o.get('kind')=='fare_gate' and o['label']=='개찰구 본체'])==len(c['fare_gates'])+1


def test_drawing_b1_annotations_define_floor_and_only_two_buildings():
    from shapely.geometry import LineString,Polygon
    c=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf8'))
    floor=next(f for f in c['floors'] if f['id']=='B1')
    scale=c['drawing']['calibration']['m_per_pdf_point'];ox,oy=c['drawing']['floor_origins_pdf']['B1']
    def p(x,y):return [round((x-ox)*scale,4),round((oy-y)*scale,4)]
    assert floor_geometry(floor).buffer(-.24).covers(LineString([p(784,255),p(784,299)]))
    boundary=Polygon(floor['polygons'][0])
    # Right storage strip and lower restroom/waiting-room extension are floor;
    # the southeast substation annex outside the blue line is excluded.
    for point in [(950,260),(870,350),(950,339)]:
        assert boundary.covers(Point(p(*point)))
    assert not boundary.covers(Point(p(980,380)))
    rooms=[o for o in floor['obstacles'] if o.get('render_as')=='building']
    assert {o['id'] for o in rooms}=={'B1_LEFT_BUILDING','B1_MERGED_BUILDING'}
    merged=Polygon(next(o for o in rooms if o['id']=='B1_MERGED_BUILDING')['polygon'])
    # Both green joins lie inside a single polygon, with no internal boundary.
    for point in [(900,229),(932,295)]:
        assert merged.buffer(-.15).covers(Point(p(*point)))
    assert not merged.covers(Point(p(905,260)))  # stair notch remains open
    assert not merged.covers(Point(p(860,270)))  # public concourse remains open
    assert all(boundary.buffer(1e-7).covers(Polygon(o['polygon'])) for o in rooms)
    for i,first in enumerate(rooms):
        for second in rooms[i+1:]:
            assert Polygon(first['polygon']).intersection(Polygon(second['polygon'])).area<1e-5


def test_drawing_b1_perimeter_stair_clearance_and_gate_ends():
    from shapely.geometry import Polygon,LineString
    from shapely.ops import unary_union
    from persona_evac.integration.station_layout import display_floor_geometry
    c=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf8'))
    floor=next(f for f in c['floors'] if f['id']=='B1')
    boundary=Polygon(floor['polygons'][0])
    rooms={o['id']:Polygon(o['polygon']) for o in floor['obstacles'] if o.get('render_as')=='building'}
    merged=rooms['B1_MERGED_BUILDING'];left=rooms['B1_LEFT_BUILDING']
    # Every outside-facing edge uses the slab perimeter itself, with no rim.
    outer=floor['polygons'][0]
    outside=LineString(outer[1:-1])
    assert merged.boundary.buffer(1e-7).covers(outside)
    assert left.bounds[0]==pytest.approx(boundary.bounds[0])
    # Room foundations and wall bases must not recreate an apron outside it.
    assert display_floor_geometry(floor).difference(boundary).area<1e-7
    for stair in c['stairs'][:4]:
        shaft=Polygon(stair['floor_opening_polygon'])
        assert left.distance(shaft)>.30
    buildings=unary_union(list(rooms.values()))
    for wall in floor['walls']:
        assert Polygon(wall['polygon']).intersection(buildings).area<1e-8
        original_length=119*c['drawing']['calibration']['m_per_pdf_point']
        assert LineString(wall['path_xy']).length<original_length-.5
    assert not any(o.get('kind')=='fare_barrier' for o in floor['obstacles'])
    assert len([o for o in floor['obstacles'] if o.get('kind')=='fare_gate'])==14


def test_drawing_f1_has_three_buildings_and_only_stair_upper_opening():
    from shapely.geometry import Polygon,LineString
    from shapely.ops import unary_union
    from persona_evac.integration.station_layout import display_floor_geometry,view_geometry
    c=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf8'))
    floor=next(f for f in c['floors'] if f['id']=='F1')
    assert len(floor['polygons'])==1
    boundary=Polygon(floor['polygons'][0])
    assert boundary.is_valid and len(floor['polygons'][0])>100
    rooms=[o for o in floor['obstacles'] if o.get('render_as')=='building']
    assert len(rooms)==3
    assert {o['id'] for o in rooms}=={'F1_WEST_BUILDING','F1_NORTH_BUILDING','F1_SOUTH_BUILDING'}
    stair=next(s for s in c['stairs'] if s['id']=='H_E')
    opening=Polygon(stair['floor_opening_polygon'])
    assert boundary.buffer(1e-7).covers(opening)
    assert all(Polygon(o['polygon']).intersection(opening).area<1e-8 for o in rooms)
    assert display_floor_geometry(floor).difference(boundary).area<1e-7
    views=view_geometry(c,2.5)['F1']
    assert len(views)==1 and len(views[0]['holes'])==1
    slab=unary_union([Polygon(p['outer'],p['holes']) for p in views])
    assert boundary.difference(slab).symmetric_difference(opening).area<1e-6
    for key in stair['destination_nodes']:
        node=c['nodes'][key]
        # The head connects directly into the surrounding unbroken slab.
        assert slab.covers(LineString([[node['x']-1,node['y']],[node['x'],node['y']]]))
    assert not floor['walls']  # obsolete U-shaped wall was inside new buildings


def test_equal_speed_counterflow_can_pass_without_body_overlap():
    geom=box(0,0,12,3);sim=make_simulation(geom)
    left=sim.add_waypoint_stage((1,1.5),.3);right=sim.add_waypoint_stage((11,1.5),.3)
    jl=sim.add_journey(jps.JourneyDescription([left]));jr=sim.add_journey(jps.JourneyDescription([right]))
    a=sim.add_agent(parameters((5,1.5),jr,right,1.0));b=sim.add_agent(parameters((5.5,1.5),jl,left,1.0))
    assert plan_yield(sim,geom,a,(11,1.5),right)
    for _ in range(250):
        sim.iterate()
        assert math.dist(sim.agent(a).position,sim.agent(b).position)>=2*RADIUS-.01
    assert sim.agent(a).position[0]>sim.agent(b).position[0]+2


def test_narrow_room_sliver_is_not_a_pedestrian_route():
    original=box(0,0,20,4).difference(box(8,.8,12,3.8))
    geom=pedestrian_geometry(original)
    assert not geom.covers(Point(10,3.9))
    assert geom.covers(Point(10,.4))
    sim=make_simulation(geom);goal=(7,3.4)
    stage=sim.add_waypoint_stage(goal,.3);journey=sim.add_journey(jps.JourneyDescription([stage]))
    agent=sim.add_agent(parameters((13,3.4),journey,stage,1))
    for _ in range(600):
        sim.iterate()
        if math.dist(sim.agent(agent).position,goal)<.6:break
    assert math.dist(sim.agent(agent).position,goal)<.6


def test_gaehwasan_east_to_west_stair_avoids_function_room_sliver():
    c=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf8'))
    floor=next(f for f in c['floors'] if f['id']=='B2')
    geom=pedestrian_geometry(floor_geometry(floor,0))
    goal=(c['nodes']['A_W_in']['x'],c['nodes']['A_W_in']['y']);start=(120,30)
    route=jps.RoutingEngine(geom).compute_waypoints(start,goal)
    assert min(p[1] for p in route)<27
    sim=make_simulation(geom);stage=sim.add_waypoint_stage(goal,.35)
    journey=sim.add_journey(jps.JourneyDescription([stage]))
    actor=sim.add_agent(parameters(start,journey,stage,1.25))
    for _ in range(1600):
        sim.iterate()
        if math.dist(sim.agent(actor).position,goal)<.6:break
    assert math.dist(sim.agent(actor).position,goal)<.6
