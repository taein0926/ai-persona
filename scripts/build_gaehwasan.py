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
ORIGINS = {"B2": (320, 730), "B1": (376, 350), "F1": (74, 350)}
PDF = ROOT / "data/reference/station_candidates/gaehwasan_mechanical_2026.pdf"
URL = "https://www.g2b.go.kr/pn/pnp/pnpe/UntyAtchFile/downloadFile.do?bidPbancNo=R26BK01488834&bidPbancOrd=000&fileSeq=6&fileType=&prcmBsneSeCd=07"


def xy(floor, p):
    ox, oy = ORIGINS[floor]
    return [round((p[0]-ox)*SCALE, 4), round((oy-p[1])*SCALE, 4)]


def ring(floor, points):
    return [xy(floor, p) for p in points]


def obstacle(floor, label, points):
    return {"label": label, "polygon": ring(floor, points)}


def build():
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
    hall = [(705,230),(745,229),(785,227),(820,225),(840,212),(860,205),
            (879,209),(897,219),(913,235),(926,255),(931,276),(929,294),
            (917,311),(897,324),(875,331),(852,332),(831,325),(815,314),
            (780,320),(744,325),(706,325)]
    ground = [(479,243),(484,227),(497,215),(512,207),(513,188),(550,192),
              (550,207),(571,211),(590,225),(604,243),(610,265),(608,288),
              (598,308),(581,325),(561,334),(537,336),(515,330),(496,317),
              (483,298),(477,277),(477,260)]
    floors = [
        {"id":"B2","label":"B2 · 곡선 상대식 승강장","elevation_m":0,
         "polygons":[ring("B2",north),ring("B2",south)],
         "obstacles":[obstacle("B2","승강장 북측 기능실",[(660,612),(747,610),(747,626),(660,629)]),
                      obstacle("B2","승강장 남측 기능실",[(665,699),(746,693),(746,713),(665,717)])]},
        {"id":"B1","label":"B1 · 대합실","elevation_m":4,
         "polygons":[ring("B1",hall)],
         "obstacles":[obstacle("B1","역무·통신실",[(705,246),(743,245),(744,308),(706,310)]),
                      obstacle("B1","통신·소방 창고",[(823,224),(842,213),(860,208),(879,212),(893,220),(893,230),(824,240)]),
                      obstacle("B1","갱의실",[(822,311),(835,310),(836,327),(826,324)]),
                      obstacle("B1","설비 창고",[(930,228),(937,265),(933,287),(924,289),(924,279),(918,246)]),
                      obstacle("B1","화장실·용역창고",[(875,301),(897,299),(906,313),(893,321),(876,324)])]},
        {"id":"F1","label":"1F · 대합실 / 지상 출구","elevation_m":8,
         "polygons":[ring("F1",ground)],
         "obstacles":[obstacle("F1","상가·시설물",[(501,238),(526,236),(535,241),(542,303),(513,307),(508,289)]),
                      obstacle("F1","설비 공간",[(589,298),(607,296),(601,312),(591,317)])]}
    ]
    # Circular column footprints also prevent artificial straight paths through columns.
    for floor, points in [("B2",[(612,638),(631,638),(784,620),(803,618),(612,702),(631,702),(790,695),(809,692)]),
                          ("B1",[(763,243),(765,276),(768,309),(803,274),(842,270),(860,268),(879,266)]),
                          ("F1",[(553,232),(560,278),(560,301),(496,274)])]:
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
    stair("A_E","북측 승강장 동쪽 → B1","B2",(757,617),"A","B1",(788,237),"hall",[(758,609),(759,625)],12)
    stair("B_W","남측 승강장 서쪽 → B1","B2",(645,708),"B","B1",(748,315),"hall",[(648,701),(648,716)],12)
    stair("B_E","남측 승강장 동쪽 → B1","B2",(757,702),"B","B1",(790,315),"hall",[(758,696),(759,712)],12)
    stair("H_W","B1 서쪽 곡선 계단 → 1F","B1",(785,300),"hall","F1",(487,262),"ground",[(776,274),(793,274)],14)
    stair("H_E","B1 동쪽 왕복 계단 → 1F","B1",(897,275),"hall","F1",(591,246),"ground",[(912,247),(914,262)],14)
    stair("E_1","1F → 1번 지상 출구","F1",(529,200),"ground","OUTSIDE",(530,181),None,[(514,195),(529,199)],4,.8,"exit")
    stair("E_2","1F → 2번 지상 출구","F1",(549,327),"ground","OUTSIDE",(550,344),None,[(538,330),(559,329)],2,1.,"exit")
    # Network routes serve validation/provenance; JuPedSim computes all planar trajectories.
    node("platform_a","B2",(635,637),"A"); node("platform_b","B2",(635,692),"B")
    edges=[]
    for area in ("A","B","hall","ground"):
        keys=[key for key,n in nodes.items() if n["area"]==area]
        edges.extend([a,b] for i,a in enumerate(keys) for b in keys[i+1:])
    # Distributed, body-clear spawns on both curved platforms (not on tracks or rooms).
    spawn=[]
    for area,idx in [("A",0),("B",1)]:
        geom=floor_geometry(floors[0],idx).buffer(-.6)
        choices=[]
        for x in range(8,165,3):
            for y in range(3,40):
                if geom.covers(Point(x,y)) and all(math.dist((x,y),q)>1.2 for q in choices):
                    choices.append((x,y))
        # Spread in longitudinal order to include the platform ends and both stair routes.
        chosen=[choices[round(i*(len(choices)-1)/49)] for i in range(50)]
        spawn.append([{"floor":"B2","x":x,"y":y,"area":area} for x,y in chosen])
    source={"pdf":"data/reference/station_candidates/gaehwasan_mechanical_2026.pdf",
            "pdf_sha256":hashlib.sha256(PDF.read_bytes()).hexdigest(),"page_1_based":3,
            "drawing_date":"2025-12","url":URL,"coordinate_system":"PDF points, top-left origin",
            "calibration":{"dimension_mm":4000,"endpoints_pdf":[[689.76,230.0],[689.76,245.12]],
                           "m_per_pdf_point":SCALE,"isotropic":True,
                           "grid_check_mm":5028,"grid_interval_pdf_points":18.96,
                           "grid_reconstructed_m":round(18.96*SCALE,4)},
            "floor_origins_pdf":ORIGINS,
            "note":"도면의 명시 치수로 등비 보정한 윤곽 근사. 하단 도면명 석계 표기는 본문 개화산 제목·위치와 불일치."}
    config={"id":"gaehwasan-drawing-v1","station_slug":"gaehwasan","name":"개화산역 · 도면 비율 기반 대피",
            "source":"서울교통공사 67공구 기계설비 도면 3쪽 (2025.12) · 공식 개화산역 비상대피 안내도 대조",
            "source_url":URL,"drawing":source,"floors":floors,"nodes":nodes,"stairs":stairs,"walk_edges":edges,
            "exits":["E_1_out","E_2_out"],"exit_labels":{"E_1_out":"1번 출구","E_2_out":"2번 출구"},
            "areas":[{"id":"A","floor":"B2","region":0},{"id":"B","floor":"B2","region":1},
                     {"id":"hall","floor":"B1"},{"id":"ground","floor":"F1"}],
            "spawn_points":[p for pair in zip(*spawn) for p in pair],"spawn_x":[],
            "assumptions":["B2 승강장 → B1 대합실 → 1F 대합실 → 지상 출구. 공식 안내도의 세 층과 연결 관계를 반영했습니다.",
                "모든 층의 x·y에 같은 축척을 적용했습니다. 곡선 윤곽·기능실·기둥은 도면을 수작업으로 근사한 값이며 최신 준공 실측도가 아닙니다.",
                "계단 폭은 도면에서 비례 추정했습니다. 양쪽 여유 합계 0.20m를 차감하며 폭 불확실성은 약 ±0.30m로 기록합니다.",
                "계단 경로 길이는 도면 접속점과 가정한 층간 높이 4m로 계산합니다. 세부 단면과 계단참 치수는 추정입니다.",
                "계단 진입 지점을 폭에 따라 분산합니다. 계단 내부는 시간·통과량 연결이며 3D 보행과 내부 추월은 계산하지 않습니다.",
                "1인당 가로 점유폭 0.60m는 나란히 이동하는 열 수를 표시하기 위한 모델 가정입니다.",
                "개찰구는 비상 개방, 승강기와 에스컬레이터는 이용하지 않는 대피 시나리오입니다. 도면 이후 설비 변경은 자동 반영하지 않습니다.",
                "Qwen 분류·Jülich 반응 시간 및 속도와 원본 페르소나를 재사용합니다. 사람 간 회피·추월은 JuPedSim으로 계산합니다.",
                "열차 화재와 차량별 확대 시각·발열률은 가정한 시나리오입니다. 대구 지하철 화재의 실측 재현이 아닙니다.",
                "B1→1F 접속점은 계단 중앙 대신 도면에서 읽은 끝부분으로 보정했습니다. 서쪽은 곡선, 동쪽은 평평한 중간 계단참을 가진 왕복 형상이며 세부 높이·단수는 가정입니다."]}
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
    config['train']={'line':'5호선','color':'#9965d9','cars':cars,'track_polygon':list(track_geom.exterior.coords),
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
    make_fds(config)
    print(json.dumps({"scale":SCALE,"widths":{s["id"]:s["width_m"] for s in stairs},
                      "floor_areas_m2":{f["id"]:round(floor_geometry(f).area,1) for f in floors}}))


def make_fds(c):
    # Shared metric geometry, coarsened explicitly to 1m cells for the fire solver.
    width,height=180,52
    lines=["&HEAD CHID='gaehwasan', TITLE='Drawing-scaled Gaehwasan three-level station' /"]
    for a,b in [(0,45),(45,90),(90,135),(135,180)]:
        lines.append(f"&MESH IJK=45,52,12, XB={a},{b},0,52,0,12 /")
    lines += [f"&TIME T_END={c['fire']['time_end_s']}.0 /","&DUMP DT_SLCF=2.0, DT_HRR=2.0 /","&MISC TMPA=20.0 /",
              "&REAC FUEL='PROPANE', SOOT_YIELD=0.06, CO_YIELD=0.01 /"]
    for i,s in enumerate(c['fire']['stages']):
        lines.append(f"&SURF ID='CAR_{i}', HRRPUA=1000.0, RAMP_Q='GROWTH_{i}', COLOR='RED' /")
        for t,f in sorted({0:0,s['start_s']:0,s['full_s']:1,c['fire']['time_end_s']:1}.items()):
            lines.append(f"&RAMP ID='GROWTH_{i}', T={t}.0, F={f}.0 /")
    floor_geoms={f["id"]:floor_geometry(f) for f in c["floors"]}
    floor_geoms['B2']=floor_geoms['B2'].union(Polygon(c['train']['track_polygon']))
    # Horizontal extents of timed stairs are simple shaft approximations, recorded in metadata.
    shafts=[]
    for s in c["stairs"]:
        a,b=(c["nodes"][s[k]] for k in ("from_node","to_node"))
        if b["floor"]=="OUTSIDE": continue
        shaft=LineString([p[:2] for p in stair_path(c,s)]).buffer(s["width_m"]/2,cap_style=2,join_style=2)
        floor_geoms[a["floor"]]=floor_geoms[a["floor"]].union(shaft)
        floor_geoms[b["floor"]]=floor_geoms[b["floor"]].union(shaft)
        shafts.append((shaft,a["floor"],b["floor"]))
    for f in c["floors"]:
        z=f["elevation_m"]; geom=floor_geoms[f["id"]]
        # Merge solid runs in each row to keep the deck small and deterministic.
        for y in range(height):
            start=None
            for x in range(width+1):
                solid=x<width and not geom.covers(Point(x+.5,y+.5))
                if solid and start is None: start=x
                if not solid and start is not None:
                    lines.append(f"&OBST XB={start},{x},{y},{y+1},{z},{z+3} /"); start=None
        if z>0: lines.append(f"&OBST XB=0,{width},0,{height},{z-1},{z} /")
    elev={f["id"]:f["elevation_m"] for f in c["floors"]}
    for geom,a,b in shafts:
        for y in range(height):
            start=None
            for x in range(width+1):
                opening=x<width and geom.covers(Point(x+.5,y+.5))
                if opening and start is None: start=x
                if not opening and start is not None:
                    z=elev[b]; lines.append(f"&HOLE XB={start},{x},{y},{y+1},{z-1},{z} /"); start=None
    # Coarse inert car shells with platform-side doors open. Prescribed burners
    # supply the fire: these shells do not predict material pyrolysis/ignition.
    for car in c['train']['cars']:
        a,b=car['a'],car['b'];axis=LineString([a,b]);outer=axis.buffer(car['width_m']/2,cap_style=2)
        inner=axis.buffer(.75,cap_style=2);walls=outer.difference(inner)
        xmin,ymin,xmax,ymax=outer.bounds
        for y in range(math.floor(ymin),math.ceil(ymax)):
            for x in range(math.floor(xmin),math.ceil(xmax)):
                p=Point(x+.5,y+.5)
                if outer.covers(p):lines.append(f'&OBST XB={x},{x+1},{y},{y+1},2,3 /')
                if walls.covers(p):lines.append(f'&OBST XB={x},{x+1},{y},{y+1},0,2 /')
        dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy);nx,ny=-dy/length,dx/length
        for f in (.2,.5,.8):
            x=a[0]+f*dx+nx*car['width_m']/2;y=a[1]+f*dy+ny*car['width_m']/2
            lines.append(f'&HOLE XB={x-1.1:.3f},{x+1.1:.3f},{y-.9:.3f},{y+.9:.3f},0,2 /')
    for i,s in enumerate(c['fire']['stages']):
        x,y,w,h=s['rect'];lines.append(f"&VENT XB={x},{x+w},{y},{y+h},0,0, SURF_ID='CAR_{i}' /")
    lines.append("&OBST XB=0,180,0,52,11,12 /")
    for sid in ("E_1","E_2"):
        n=c["nodes"][sid+"_in"]; x,y=round(n["x"]),round(n["y"])
        lines.append(f"&HOLE XB={x-1},{x+1},{y-1},{y+1},11,12 /")
        lines.append(f"&VENT XB={x-1},{x+1},{y-1},{y+1},12,12, SURF_ID='OPEN' /")
    # All walls are solid unless a door/shaft is explicitly opened.
    for z in c["fire"]["slice_height_m"].values():
        for q in ("EXTINCTION COEFFICIENT","TEMPERATURE"):
            lines.append(f"&SLCF PBZ={z}, QUANTITY='{q}' /")
    for z in c['fire'].get('volume_slice_heights_m',[]):
        if z not in c['fire']['slice_height_m'].values():
            lines.append(f"&SLCF PBZ={z}, QUANTITY='EXTINCTION COEFFICIENT' /")
    lines.append("&TAIL /")
    (ROOT/"configs/fds/gaehwasan.fds").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__": build()
