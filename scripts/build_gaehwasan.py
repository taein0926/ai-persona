"""Digitize the public station plan in PDF points, calibrated to its 4000mm dimension.

Re-running reproduces metric geometry, measurement evidence and the FDS input.
The 5028mm grid is a cross-check; a PDF's printed scale alone is not used.
"""
import hashlib
import json
import math
from pathlib import Path

from shapely.geometry import Point, Polygon, LineString, box
from shapely.ops import unary_union

from persona_evac.integration.station_geometry import floor_geometry
from persona_evac.integration.station_layout import stair_path

ROOT = Path(__file__).resolve().parents[1]
SCALE = 4.0 / 15.12
# Register the repeated platform stair mouths, rather than the sheet margins.
ORIGINS = {"B2": (320, 730), "B1": (382, 347), "F1": (54, 350)}
PDF = ROOT / "data/reference/station_candidates/gaehwasan_mechanical_2026.pdf"
F1_ANNOTATION_PATH=ROOT/'data/reference/station_candidates/gaehwasan_f1_annotation_20261029.json'
F1_ANNOTATION=json.loads(F1_ANNOTATION_PATH.read_text(encoding='utf8'))
URL = "https://www.g2b.go.kr/pn/pnp/pnpe/UntyAtchFile/downloadFile.do?bidPbancNo=R26BK01488834&bidPbancOrd=000&fileSeq=6&fileType=&prcmBsneSeCd=07"


def xy(floor, p):
    if floor=='F1':
        registration=F1_ANNOTATION['registration']
        anchor=registration['source_anchor_pdf'];target=registration['target_anchor_b1_pdf']
        angle=math.radians(registration['rotation_degrees'])
        dx,dy=(p[0]-anchor[0])*SCALE,(anchor[1]-p[1])*SCALE
        tx=(target[0]-ORIGINS['B1'][0])*SCALE;ty=(ORIGINS['B1'][1]-target[1])*SCALE
        return [round(tx+dx*math.cos(angle)-dy*math.sin(angle),4),
                round(ty+dx*math.sin(angle)+dy*math.cos(angle),4)]
    ox, oy = ORIGINS[floor]
    return [round((p[0]-ox)*SCALE, 4), round((oy-p[1])*SCALE, 4)]


def ring(floor, points):
    return [xy(floor, p) for p in points]


def obstacle(floor, label, points):
    return {"label": label, "polygon": ring(floor, points)}


def wall(floor, wid, label, points):
    path=ring(floor,points)
    footprint=LineString(path).buffer(.09,cap_style=2,join_style=2)
    return {'id':wid,'label':label,'path_xy':path,'polygon':list(footprint.exterior.coords),
            'height_m':2.1,'thickness_m':.18,'passenger_facing':True,
            'source_pdf_points':points,'drawing_page':3,
            'basis':'평면도의 굵은 경계선 중 승객 통로와 접하는 구간. 높이·두께는 표시용 가정.',
            'physical_integration':'jupedsim_solid_boundary'}


def build(write_fds=False):
    # Boundary vertices follow the drawing, not a stretched schematic rectangle.
    north = [(325,592),(380,604),(440,615),(500,623),(550,628),(592,630),
             (593,611),(645,612),(700,611),(750,609),(800,604),(850,597),
             (900,588),(945,578),(970,571),(974,596),(940,606),(900,615),
             (850,624),(800,633),(750,639),(700,644),(650,646),(600,646),
             (550,644),(500,640),(440,633),(380,624),(325,612)]
    south = [(320,644),(380,657),(440,667),(500,675),(550,680),(600,682),
             (650,682),(700,681),(750,677),(800,672),(850,665),(900,657),
             (956,644),(958,668),(910,680),(860,691),(810,701),(760,710),
             (710,715),(650,718),(592,718),(593,705),(550,703),(500,699),
             (440,692),(380,682),(320,668)]
    # User annotations share the original drawing pixels. Register them to the
    # calibrated PDF before converting to station coordinates.
    annotation_path=ROOT/'data/reference/station_candidates/gaehwasan_b1_annotation_20261029.json'
    annotation=json.loads(annotation_path.read_text(encoding='utf8'))
    registration=annotation['registration']
    def annotation_pdf(points):
        scale=registration['pixels_per_pdf_point'];offset=registration['pixel_offset_xy']
        return [[round((p[k]-offset[k])/scale,3) for k in (0,1)] for p in points]
    traced_hall=annotation_pdf(annotation['floor_outline_px'])
    # Trim the western apron to the stair walls' outer faces. The curved and
    # eastern boundary keeps the user's blue trace.
    north_edge=229-.09/SCALE;south_edge=326+.09/SCALE
    hall=[(705,north_edge),(traced_hall[1][0],north_edge),
          *traced_hall[1:-2],(traced_hall[-2][0],south_edge),(705,south_edge)]
    hall_boundary=Polygon(ring('B1',hall))
    hall_buildings=[]
    for item in annotation['buildings']:
        if item['id']=='B1_LEFT_BUILDING':
            # Leave a clear strip at both stair openings; the western face
            # shares the floor perimeter instead of leaving a narrow apron.
            outline=[(705,246.5),(744.8,246.5),(744.8,308.5),(705,308.5)]
        else:
            # Use the SAME perimeter vertices as the floor, from the north
            # junction around the east side to the south junction. Only the
            # concourse-facing red boundary is independent of the floor edge.
            inner=annotation_pdf(item['outline_px'][-8:])
            outline=[*hall[1:-1],*inner]
        footprint=Polygon(ring('B1',outline)).intersection(hall_boundary)
        assert footprint.geom_type=='Polygon' and footprint.is_valid
        hall_buildings.append({'id':item['id'],'label':item['label'],
                              'polygon':list(footprint.exterior.coords),
                              'annotation_reference':annotation_path.name})
    def f1_pdf(points):
        return [[F1_ANNOTATION['pdf_origin'][k]+p[k]/F1_ANNOTATION['pixels_per_pdf_point'] for k in (0,1)] for p in points]
    def curve(points):
        return [[sum(math.comb(3,j)*(1-t)**(3-j)*t**j*points[j][k] for j in range(4)) for k in (0,1)]
                for t in [i/32 for i in range(33)]]
    north_curve=curve(F1_ANNOTATION['north_curve_px'])
    south_curve=curve(F1_ANNOTATION['south_curve_px'])
    west_curve=curve(F1_ANNOTATION['west_curves_px'][0])+curve(F1_ANNOTATION['west_curves_px'][1])[1:]
    ground=f1_pdf([[144,33],[225,66],[392,86],*north_curve,[644,213],[682,488],
                   *south_curve,[460,689],[292,707],*west_curve,[144,184]])
    ground_boundary=Polygon(ring('F1',ground))
    ground_buildings=[]
    for bid,label,points in [
        ('F1_WEST_BUILDING','서측 곡선 시설 구역',[[245,144],[290,658],*west_curve[1:]]),
        ('F1_NORTH_BUILDING','북측 계단 옆 시설 구역',[*north_curve,[410,255]]),
        ('F1_SOUTH_BUILDING','남측 계단 옆 시설 구역',[*south_curve,[443,518]])]:
        footprint=Polygon(ring('F1',f1_pdf(points))).intersection(ground_boundary)
        assert footprint.is_valid and footprint.geom_type=='Polygon'
        ground_buildings.append({'id':bid,'label':label,'polygon':list(footprint.exterior.coords),
                                 'annotation_reference':F1_ANNOTATION_PATH.name})
    floors = [
        {"id":"B2","label":"B2 · 곡선 상대식 승강장","elevation_m":0,
         "polygons":[ring("B2",north),ring("B2",south)],
         "obstacles":[obstacle("B2","전원공급실·청원기계실·통신창고",[(660,612),(756,611),(756,623),(660,628)]),
                      obstacle("B2","창고·전원공급실",[(660,699),(756,695),(756,714),(660,718)])]},
        {"id":"B1","label":"B1 · 대합실","elevation_m":4,
         "polygons":[ring("B1",hall)],
         "boundary_reference":annotation_path.name,
         "obstacles":hall_buildings},
        {"id":"F1","label":"1F · 전기실 평면도 / 지상 대합실","elevation_m":8,
         "polygons":[ring("F1",ground)],
         "boundary_reference":F1_ANNOTATION_PATH.name,
         "drawing_registration":F1_ANNOTATION['registration'],
         "obstacles":ground_buildings}
    ]
    floors[1]['walls']=[
        wall('B1','B1_N_EDGE','북측 계단 옆 외벽',[(705,229),(824,229)]),
        wall('B1','B1_S_EDGE','남측 계단 옆 외벽',[(705,326),(824,326)])]
    # End the stair walls before the adjoining building, including the wall
    # thickness, rather than drawing the former long walls through its roof.
    rooms=unary_union([Polygon(o['polygon']) for o in hall_buildings])
    for item in floors[1]['walls']:
        path=LineString(item['path_xy']).difference(rooms.buffer(.10))
        assert path.geom_type=='LineString' and not path.is_empty
        item['path_xy']=list(path.coords)
        item['polygon']=list(path.buffer(.09,cap_style=2,join_style=2).intersection(hall_boundary).exterior.coords)
        item['source_pdf_points']=[[(p[0]/SCALE)+ORIGINS['B1'][0],ORIGINS['B1'][1]-(p[1]/SCALE)] for p in path.coords]
    floors[2]['walls']=[]
    for f in floors:
        for item in f['obstacles']:
            item['render_as']='building'
    # Circular column footprints also prevent artificial straight paths through columns.
    for floor, points in [("B2",[(612,638),(631,638),(784,620),(803,618),(612,702),(631,702),(790,695),(809,692)]),
                          ("B1",[(763,243),(765,276),(768,309),(803,274),(842,270),(860,268),(879,266)]),
                          ("F1",[(553,232),(560,278),(560,301)])]:
        f = next(f for f in floors if f["id"] == floor)
        for p in points:
            c = xy(floor,p)
            f["obstacles"].append({"label":"기둥","polygon":[[round(x,4),round(y,4)] for x,y in Point(c).buffer(.27,quad_segs=4).exterior.coords]})
    nodes, stairs, measurements = {}, [], []
    def node(key,floor,p,area):
        x,y=xy(floor if floor != "OUTSIDE" else "F1",p)
        nodes[key]={"floor":floor,"x":x,"y":y,"area":area}
    def stair(sid,label,src,srcp,area,dst,dstp,darea,ends,length,speed=.5,kind="stairs"):
        node(sid+"_in",src,srcp,area)
        node(sid+"_out",dst,dstp,darea)
        measured = math.dist(*ends)*SCALE
        usable = round(measured-.20, 2)
        reference={"method":"drawing_ratio","drawing_page":3,"pdf_endpoints":ends,
                   "measured_width_m":round(measured,3),"estimated_width_m":usable,
                   "handrail_allowance_m":.20,"uncertainty_m":.30,
                   "note":"도면 폭을 비례 산출하고 양쪽 여유 합계 0.20m를 차감한 모델 유효폭. 실측값이 아님."}
        # A passage spans the transverse direction of the measured stair mouth.
        dx,dy=ends[1][0]-ends[0][0],-(ends[1][1]-ends[0][1])
        norm=math.hypot(dx,dy)
        stairs.append({"id":sid,"label":label,"from_node":sid+"_in","to_node":sid+"_out",
                       "mode":"geometry","width_m":usable,"length_m":length,"speed_mps":speed,
                       "density_per_m2":1.,"width_reference":reference,"kind":kind,
                       "portal_tangent":[round(dx/norm,5),round(dy/norm,5)],
                       "lane_width_m":.60})
        measurements.append({"id":sid,"label":label,**reference})
    stair("A_W","북측 승강장 서쪽 → B1","B2",(645,620),"A","B1",(747,237),"hall",[(647,613),(647,628)],12)
    stair("A_E","북측 승강장 동쪽 → B1","B2",(762,620),"A","B1",(784,237),"hall",[(758,609),(759,625)],12)
    stair("B_W","남측 승강장 서쪽 → B1","B2",(645,701),"B","B1",(747,318),"hall",[(648,701),(648,716)],12)
    stair("B_E","남측 승강장 동쪽 → B1","B2",(762,701),"B","B1",(784,318),"hall",[(758,696),(759,712)],12)
    for item in stairs:
        a,b=[nodes[item[k]] for k in ('from_node','to_node')]
        b['y']=a['y']
        item['portal_tangent']=[0.,1.]
        # Axis-aligned shafts stop before the arrival pad; never cut the pad.
        xmin,xmax=sorted([a['x'],b['x']]);y=a['y'];half_width=item['width_m']/2
        item['floor_opening_polygon']=[[xmin+.35,y-half_width],[xmax-.35,y-half_width],
                                       [xmax-.35,y+half_width],[xmin+.35,y+half_width]]
    stair("H_E","B1 중앙 두 계단·양옆 되돌림 계단 → 1F","B1",(900,263),"hall","F1",(572,242.5),"ground",[(900,247),(903,279)],16)
    main = stairs[-1]
    main['portal_tangent'] = [0., 1.]
    width = main['width_m']
    px, py = xy('B1', (900,263))
    half, run = width/2, 4.5
    end_x, landing_x = px+run, px+run+half/2
    # The enlarged landing accommodates two flat 90-degree turns per branch.
    # Both outer upper flights ascend opposite to the adjacent lower flights.
    branches, lower_flights, upper_flights = [], [], []
    for side, key in [(-1,'H_E_left_out'),(1,'H_E_out')]:
        lower_y = py+side*half/2
        outer_y = py+side*half*1.5
        lower = [[px,lower_y,4.],[end_x,lower_y,6.]]
        upper = [[end_x,outer_y,6.],[px,outer_y,8.]]
        nodes[key] = {'floor':'F1','x':px,'y':outer_y,'area':'ground'}
        branches.append([*lower,[landing_x,lower_y,6.],[landing_x,outer_y,6.],*upper])
        lower_flights.append({'role':'lower','side':side,'path_xyz':lower,'width_m':half})
        upper_flights.append({'role':'upper','side':side,'path_xyz':upper,'width_m':half})
    landing_polygon=[[end_x,py-width],[end_x+half,py-width],
                     [end_x+half,py+width],[end_x,py+width]]
    outer_rails=[]
    for side in (-1,1):
        outer_rails.append([[px,py+side*half,4.],[end_x,py+side*half,6.],
                            [px,py+side*half,8.]])
        outer_rails.append([[px,py+side*width,8.],[end_x,py+side*width,6.],
                            [end_x+half,py+side*width,6.],[end_x+half,py,6.]])
    main.update(shape='split_return_with_extended_landing', destination_nodes=['H_E_left_out','H_E_out'],
                branch_paths_xyz=branches, path_xyz=branches[1],
                floor_opening_polygon=[[px+.15,py-width],[end_x+half,py-width],
                                       [end_x+half,py+width],[px+.15,py+width]],
                flights=[*lower_flights,*upper_flights],
                landings=[{'polygon':landing_polygon,'elevation_m':6.,'thickness_m':.22}],
                center_handrail_xyz=[[px,py,4.],[end_x,py,6.]],
                rail_paths_xyz=outer_rails)
    # Keep approximated room edges out of the stair opening and head aprons.
    stair_clearance=Polygon(main['floor_opening_polygon']).buffer(.02)
    corrected=[]
    for item in floors[2]['obstacles']:
        geom=Polygon(item['polygon']).difference(stair_clearance)
        for part in getattr(geom,'geoms',[geom]):
            if part.geom_type=='Polygon' and not part.is_empty:
                corrected.append({**item,'polygon':list(part.exterior.coords)})
    floors[2]['obstacles']=corrected
    stair("E_1","방풍실·꺾인 외부 계단 → 1번 출구","F1",(533,203),"ground","OUTSIDE",(510,194),None,[(514,188),(513,203)],4,.8,"exit")
    stairs[-1].update(exit_form='bent_stair',
                     render_mode='floor_only',
                     path_xyz=[[*xy('F1',p),8.] for p in [(533,203),(533,196),(526,196),(518,198),(510,194)]],
                     vertical_profile='외부 계단의 평면 투영 형상. 지면 높이·단면 미확인으로 높이차를 확정하지 않음.')
    stair("E_2","방풍실 통과 → 2번 출구","F1",(557,319),"ground","OUTSIDE",(558,343),None,[(528,330),(551,328)],2,1.,"exit")
    stairs[-1].update(exit_form='vestibule_passage',
                     render_mode='floor_only',
                     path_xyz=[[*xy('F1',p),8.] for p in [(557,319),(558,329),(558,343)]],
                     vertical_profile='방풍실과 외부 접속부의 평면 연결. DOWN B1FL(4M) 표기는 단면 확인 전 높이에 적용하지 않음.')
    # Updated exits are the outside edges of the blue floor. E1 uses the full
    # wide mouth of the funnel, not its narrow neck or the former inner point.
    for sid,pixels,inside_pdf in [
        ('E_1',[[144,33],[144,184]],(509,197)),
        ('E_2',[[292,707],[460,689]],(558,339))]:
        s=next(s for s in stairs if s['id']==sid)
        edge=ring('F1',f1_pdf(pixels));a,b=edge
        midpoint=[(a[k]+b[k])/2 for k in (0,1)]
        tangent=[(b[k]-a[k])/math.dist(a,b) for k in (0,1)]
        inward=xy('F1',inside_pdf);normal=[-tangent[1],tangent[0]]
        if sum((inward[k]-midpoint[k])*normal[k] for k in (0,1))<0:normal=[-v for v in normal]
        start=[midpoint[k]+.85*normal[k] for k in (0,1)]
        # Stop at the actual outside boundary; completion means leaving F1.
        nodes[s['from_node']].update(x=start[0],y=start[1])
        nodes[s['to_node']].update(x=midpoint[0],y=midpoint[1])
        s.update(exit_boundary_xy=edge,exit_width_m=math.dist(a,b),
                 width_m=min(9.99,math.dist(a,b)-.50),portal_tangent=tangent,
                 path_xyz=[[*start,8.],[*midpoint,8.]],length_m=1.,
                 exit_basis='파란 바닥의 바깥쪽 변 전체. 1번 출구는 깔때기의 넓은 입구를 출구로 가정.')
    vestibules=[
        {'id':'V_1','exit_id':'E_1','floor':'F1','label':'1번 출구 방풍실',
         'polygon':ring('F1',[(519,205),(550,202),(551,216),(520,219)]),
         'door_lines':[ring('F1',[(520,205),(549,202)]),ring('F1',[(521,218),(550,215)])]},
        {'id':'V_2','exit_id':'E_2','floor':'F1','label':'2번 출구 방풍실',
         'polygon':ring('F1',[(520,318),(557,315),(559,330),(521,334)]),
         'door_lines':[ring('F1',[(522,318),(556,315)]),ring('F1',[(523,333),(557,329)])]}
    ]
    for vestibule in vestibules:
        vestibule['render_mode']='floor_only'
        poly=vestibule['polygon']
        vestibule['reference_side_walls']=[]
        for a,b in [(poly[0],poly[3]),(poly[1],poly[2])]:
            vestibule['reference_side_walls'].append({'label':vestibule['label']+' 측벽','kind':'vestibule_wall','display':False,
                'height_m':2.1,'polygon':list(LineString([a,b]).buffer(.08,cap_style=2).exterior.coords)})
    # Gate cabinet row follows the diagonal paid/unpaid boundary in the B1 plan.
    # Count and clear opening width are digitized estimates, not a site survey.
    gate_start, gate_end = xy('B1',(821,243)), xy('B1',(827,307))
    row_length = math.dist(gate_start,gate_end)
    tangent = [(gate_end[i]-gate_start[i])/row_length for i in (0,1)]
    normal = [-tangent[1],tangent[0]]
    gates = []
    count, opening, depth = 13, .65, 1.20
    pitch = row_length/count
    hall_floor = floors[1]
    def gate_rect(along, half_along, half_normal):
        return [[gate_start[k]+(along+a*half_along)*tangent[k]+b*half_normal*normal[k]
                 for k in (0,1)] for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]]
    for i in range(count+1):
        hall_floor['obstacles'].append({'id':f'GATE_BODY_{i:02d}','label':'개찰구 본체',
            'kind':'fare_gate','height_m':1.0,'polygon':gate_rect(i*pitch,(pitch-opening)/2,depth/2)})
    for i in range(count):
        center=[gate_start[k]+(i+.5)*pitch*tangent[k] for k in (0,1)]
        gates.append({'id':f'G_{i+1:02d}','floor':'B1','area':'hall','center':center,
                      'tangent':tangent,'normal':normal,'width_m':opening,'depth_m':depth,
                      'capacity':1,'height_m':1.0,'passage_model':'jupedsim_exclusive_channel','polygon':gate_rect((i+.5)*pitch,opening/2,depth/2),
                      'basis':'B1 도면 개찰구 열에서 근사한 13개 통로. 개수·폭은 현장 실측 미확인.'})
    # Gate-end diagonal barriers are omitted from the station geometry.
    # Physical gate connections will be defined during JuPedSim integration.
    # Network routes serve validation/provenance; JuPedSim computes all planar trajectories.
    node("platform_a","B2",(635,637),"A"); node("platform_b","B2",(635,692),"B")
    edges=[]
    for area in ("A","B","hall","ground"):
        keys=[key for key,n in nodes.items() if n["area"]==area]
        edges.extend([a,b] for i,a in enumerate(keys) for b in keys[i+1:])
    # Distributed, body-clear spawns on both curved platforms (not on tracks or rooms).
    spawn={}
    stair_footprints=unary_union([LineString([p[:2] for p in stair_path({'nodes':nodes,'floors':floors},s)]).buffer(s['width_m']/2+.35,cap_style=2)
                                 for s in stairs if nodes[s['from_node']]['floor']=='B2'])
    for area,idx in [("A",0),("B",1)]:
        geom=floor_geometry(floors[0],idx).buffer(-.6).difference(stair_footprints)
        choices=[]
        for x in range(8,165,3):
            for y in range(3,40):
                if geom.covers(Point(x,y)) and all(math.dist((x,y),q)>1.2 for q in choices):
                    choices.append((x,y))
        # Farthest-first order keeps even small requested populations spread
        # along the platform, with the same clearance at every population size.
        chosen=[min(choices,key=lambda p:abs(p[0]-85))]
        for _ in range((80 if area=='A' else 20)-1):
            chosen.append(max(choices,key=lambda p:min(math.dist(p,q) for q in chosen)))
        spawn[area]=[{"floor":"B2","x":x,"y":y,"area":area} for x,y in chosen]
    spawn_order=[]
    for i in range(20):spawn_order.extend([*spawn['A'][4*i:4*i+4],spawn['B'][i]])
    source={"pdf":"data/reference/station_candidates/gaehwasan_mechanical_2026.pdf",
            "pdf_sha256":hashlib.sha256(PDF.read_bytes()).hexdigest(),"page_1_based":3,
            "drawing_date":"2025-12","url":URL,"coordinate_system":"PDF points, top-left origin",
            "calibration":{"dimension_mm":4000,"endpoints_pdf":[[689.76,230.0],[689.76,245.12]],
                           "m_per_pdf_point":SCALE,"isotropic":True,
                           "grid_check_mm":5028,"grid_interval_pdf_points":18.96,
                           "grid_reconstructed_m":round(18.96*SCALE,4)},
            "floor_origins_pdf":ORIGINS,"floor_registrations":{"F1":F1_ANNOTATION['registration']},
            "note":"도면의 명시 치수로 등비 보정한 윤곽 근사. 하단 도면명 석계 표기는 본문 개화산 제목·위치와 불일치."}
    config={"id":"gaehwasan-drawing-v2","station_slug":"gaehwasan","name":"개화산역 · 도면 비율 기반 대피",
            "layout_status":"ready","simulation_enabled":True,
            "jupedsim_connections":{"enabled":True,"stairs":"unfolded_native_domains",
                "fare_gates":"exclusive_capacity_one","calculation_status":"not_run"},
            "source":"서울교통공사 67공구 기계설비 도면 3쪽 (2025.12) · 공식 개화산역 비상대피 안내도 대조",
            "source_url":URL,"drawing":source,"floors":floors,"nodes":nodes,"stairs":stairs,"walk_edges":edges,"fare_gates":gates,"vestibules":vestibules,
            "exits":["E_1_out","E_2_out"],"exit_labels":{"E_1_out":"1번 출구","E_2_out":"2번 출구"},
            "areas":[{"id":"A","floor":"B2","region":0},{"id":"B","floor":"B2","region":1},
                     {"id":"hall","floor":"B1"},{"id":"ground","floor":"F1"}],
            "spawn_points":spawn_order,"spawn_x":[],
            "spawn_distribution":{"arrival_platform_area":"A","opposite_platform_area":"B","arrival_fraction":.8,"minimum_center_gap_m":1.2,"structure_clearance_m":.6},
            "assumptions":["B2 승강장 → B1 대합실 → 1F 대합실 → 지상 출구. 공식 안내도의 세 층과 연결 관계를 반영했습니다.",
                "모든 층의 x·y에 같은 축척을 적용했습니다. 곡선 윤곽·기능실·기둥은 도면을 수작업으로 근사한 값이며 최신 준공 실측도가 아닙니다.",
                "계단 폭은 도면에서 비례 추정했습니다. 양쪽 여유 합계 0.20m를 차감하며 폭 불확실성은 약 ±0.30m로 기록합니다.",
                "계단 경로 길이는 도면 접속점과 가정한 층간 높이 4m로 계산합니다. 세부 단면과 계단참 치수는 추정입니다.",
                "계단은 분기별 경로 길이를 펼친 2D JuPedSim 보행 영역과 연결하고 native 위치를 실제 XYZ 경로에 투영합니다. 엔진 자체는 3차원 보행 엔진이 아닙니다.",
                "1인당 가로 점유폭 0.60m는 나란히 이동하는 열 수를 표시하기 위한 모델 가정입니다.",
                "개찰구는 높이 1m, 개별 폭 0.65m의 13개 통로이며 각 통로는 JuPedSim에서 한 명만 점유합니다. 수신 측이 막히면 점유를 유지합니다. 개수와 폭은 도면 근사입니다.",
                "Qwen 분류·Jülich 반응 시간 및 속도와 원본 페르소나를 재사용합니다. 사람 간 회피·추월은 JuPedSim으로 계산합니다.",
                "열차 화재와 차량별 확대 시각·발열률은 가정한 시나리오입니다. 대구 지하철 화재의 실측 재현이 아닙니다.",
                "B1→1F 대피 연결은 동쪽 한 곳입니다. 중앙 손잡이로 나눈 두 하부 계단의 끝에서 늘린 계단참으로 들어가 각각 바깥쪽으로 이동한 뒤 한 번 더 꺾어, 하부와 반대 방향의 양옆 상부 계단으로 올라갑니다. 서쪽 곡선 설비층 계단은 대피 연결에서 제외합니다.",
                "승강장 계단 네 곳은 같은 수평 길이와 층고로 정합했습니다. 같은 승강장 두 계단 사이의 전원·기계·창고 공간은 통행할 수 없습니다."]}
    config['assumptions'].append('1번 출구는 깔때기 모양 바닥의 넓은 바깥쪽 변 전체를 출구로 가정합니다. 2번 출구도 수정한 바닥의 바깥쪽 변과 일치시킵니다. 외부 지면 높이는 미확인입니다.')
    config['assumptions'].append('승객 통로와 맞닿는 벽·건물은 JuPedSim 보행 영역에서 제외합니다. 벽 높이 2.1m·두께 0.18m는 가정입니다. 계산은 명시적인 실행 요청으로 시작합니다.')
    for s in stairs:
        s['path_xyz']=stair_path(config,s)
        s['path_basis']='도면 끝점·방향 기반, 층고와 세부 경사는 가정'
        if s['kind']!='exit':s['length_m']=round(sum(math.dist(a,b) for a,b in zip(s['path_xyz'],s['path_xyz'][1:])),1)
    # The track space admits smoke but remains nonwalkable in JuPedSim.
    inner=north[15:]
    track=ring('B2',inner+south[:13])
    track_geom=Polygon(track).buffer(0)
    edge=sorted(ring('B2',inner))
    def train_y(x):
        for a,b in zip(edge,edge[1:]):
            if a[0]<=x<=b[0]:return a[1]+(b[1]-a[1])*(x-a[0])/(b[0]-a[0])-1.55
        raise ValueError('열차가 승강장 범위를 벗어났습니다.')
    cars=[]
    for i in range(8):
        x0=8+i*19.1;x1=x0+18.4
        cars.append({'id':i+1,'a':[round(x0,3),round(train_y(x0),3)],'b':[round(x1,3),round(train_y(x1),3)],'width_m':2.8,'height_m':2.4})
    config['train']={'line':'5호선','color':'#9965d9','cars':cars,'track_polygon':list(track_geom.exterior.coords),'arrival_platform_area':'A',
                     'assumption':'차량 수·외형·치수는 표현과 화재 시나리오용 근사. 100명은 승강장에서 출발하며 차내 하차는 미구현.'}
    stages=[]
    for index,start in [(3,0),(4,90),(2,180)]:
        car=cars[index];cx=(car['a'][0]+car['b'][0])/2;cy=train_y(cx)
        stages.append({'car_id':index+1,'start_s':start,'full_s':start+60,'peak_hrr_kw':2000,
                       'rect':[math.floor(cx),math.floor(cy),2,1]})
    config["fire"]={"floor":"B2","rect":stages[0]['rect'],"stages":stages,'spread_model':'prescribed_vehicle_sequence',
                    "peak_hrr_kw":6000,"ramp_full_s":240,
                    "evacuation_start_s":0,"slice_height_m":{"B2":1.5,"B1":5.5,"F1":9.5},
                    "mesh_cell_m":1.,"time_end_s":600,"volume_slice_heights_m":[i+.5 for i in range(12)],
                    "label":"B2 정차한 5호선 열차 · 4호차에서 시작",
                    'notice':'4호차 0초·5호차 90초·3호차 180초에 시작해 각 60초간 2MW로 증가시키는 가정. 재료 열분해·자연 착화 전파를 계산한 결과가 아닙니다.'}
    (ROOT/"configs/scenarios/gaehwasan.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    evidence={"source":source,"measurements":measurements,"applied_to_simulation":True}
    (ROOT/"data/reference/station_candidates/gaehwasan_measurements.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding="utf-8")
    if write_fds:
        make_fds(config)
    print(json.dumps({"scale":SCALE,"widths":{s["id"]:s["width_m"] for s in stairs},
                      "floor_areas_m2":{f["id"]:round(floor_geometry(f).area,1) for f in floors}}))


def make_fds(c):
    from persona_evac.integration.fds_station_geometry import write_station_fds
    return write_station_fds(c,ROOT/'configs/fds/gaehwasan.fds',
                             ROOT/'data/reference/station_candidates/gaehwasan_fds_geometry_20261029.json')

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--write-fds',action='store_true',help='수정한 구조의 FDS 입력도 생성합니다. 화재 계산은 실행하지 않습니다.')
    build(parser.parse_args().write_fds)
