"""Prepare and inspect native JuPedSim objects without any iterate calls."""
import hashlib
import json
from pathlib import Path

import jupedsim as jps
from shapely.geometry import Point

from persona_evac.integration.station_connectors import compile_network,NativePassage,connector_landing
from persona_evac.integration.station_dynamics import make_simulation,parameters,pedestrian_geometry,RADIUS

ROOT=Path(__file__).resolve().parents[1]


def prepare():
    def forbidden(*args,**kwargs):raise RuntimeError('계산 보류: JuPedSim 시간 진행을 호출할 수 없습니다.')
    jps.Simulation.iterate=forbidden
    config_path=ROOT/'configs/scenarios/gaehwasan.json'
    config=json.loads(config_path.read_text(encoding='utf8'))
    result=ROOT/'output/gaehwasan/results.json'
    before=hashlib.sha256(result.read_bytes()).hexdigest() if result.exists() else None
    net=compile_network(config)
    geoms=net['geometry']
    sims={}
    for area,geom in geoms.items():
        try:sims[area]=make_simulation(geom)
        except RuntimeError as exc:
            raise ValueError(f'{area}: 보행 영역 생성 실패, 부분 면적 {[round(p.area,3) for p in getattr(geom,"geoms",[geom])]}') from exc
    routers={a:jps.RoutingEngine(g) for a,g in geoms.items()}
    connections=[];entry_by_area={a:[] for a in geoms};arrivals=[];native_domains=[]
    for sid,s in net['connectors'].items():
        src,dst=(net['nodes'][s[k]] for k in ('from_node','to_node'))
        valid=[]
        for lane,spec in enumerate(s['native_lane_specs']):
            p=tuple(spec['entry'][:2]);end=connector_landing(s,net['nodes'],lane)
            q=(end['x'],end['y'])
            if not geoms[src['area']].buffer(-RADIUS-.02).covers(Point(p)):continue
            if dst['floor']!='OUTSIDE' and not geoms[dst['area']].buffer(-RADIUS-.02).covers(Point(q)):continue
            sims[src['area']].add_waypoint_stage(p,.35)
            valid.append(lane);entry_by_area[src['area']].append(p)
            if dst['floor']!='OUTSIDE':arrivals.append((sid,dst['area'],q))
        if not valid:raise ValueError(f'{sid}: 유효한 출입 지점이 없습니다.')
        passage=NativePassage(s,make_simulation,parameters)
        native_domains.extend(b['sim'] for b in passage.branches)
        connections.append({'id':sid,'kind':s['kind'],'source':src['area'],
            'destination':dst.get('area','OUTSIDE'),'valid_lanes':valid,
            'branches':len(passage.branches),'capacity':s['capacity']})
    def reachable(area,p):
        for goal in entry_by_area[area]:
            try:
                if routers[area].compute_waypoints(p,goal):return True
            except RuntimeError:pass
        return False
    for sid,area,p in arrivals:
        if not reachable(area,p):raise ValueError(f'{sid}: 도착 후 다음 연결에 도달할 수 없습니다.')
    for p in config['spawn_points']:
        if not reachable(p['area'],(p['x'],p['y'])):raise ValueError('시작 위치에서 계단에 도달할 수 없습니다.')
    all_sims=[*sims.values(),*native_domains]
    assert all(s.elapsed_time()==0 and s.agent_count()==0 for s in all_sims)
    after=hashlib.sha256(result.read_bytes()).hexdigest() if result.exists() else None
    assert before==after
    report={'status':'prepared_not_run','jupedsim_version':jps.__version__,
        'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'floor_domains':len(sims),'connector_domains':len(native_domains),
        'connections':connections,'excluded_island_areas_m2':net['excluded_island_areas_m2'],'native_elapsed_time_s':0,'agents_added':0,
        'iterate_calls':0,'spawn_counts':{a:sum(p['area']==a for p in config['spawn_points']) for a in ('A','B')},
        'all_spawn_and_arrival_routes_reachable':True,'stored_result_unchanged':before==after,
        'stored_result_sha256':before,'simulation_enabled':config['simulation_enabled']}
    out=ROOT/'output/qa/station_connections/preparation.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('connections','stored_result_sha256')},ensure_ascii=False))


if __name__=='__main__':prepare()
