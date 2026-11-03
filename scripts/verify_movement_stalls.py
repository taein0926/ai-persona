"""저장된 전원 기록에서 비시야성 정지·제자리 왕복과 먼 경유점을 검사한다.

물리 재계산과 LLM 호출 없이 원본 궤적을 스캔하고, 기록된 경로만 새 제어기로 대조한다.
"""
import argparse
from bisect import bisect_right
from collections import deque
import json
import math
from pathlib import Path
from types import SimpleNamespace

from persona_evac.agent_policy import ROOT
from persona_evac.integration.fds_environment import FireField
from persona_evac.integration.personal_space import OccludedSensor,observed_connector_checker
from persona_evac.integration.private_agents import PrivateController
from persona_evac.integration.station_connectors import compile_network
from persona_evac.integration.station_dynamics import pedestrian_geometry


def stalled_episodes(agent):
    samples=deque(maxlen=61);lengths=deque(maxlen=60);episodes=[];current=None
    visibility=agent.get('visibility_blocked_history',[]);times=[x['time_s'] for x in visibility]
    for segment in agent['segments']:
        if segment['status']!='walking':samples.clear();lengths.clear();current=None;continue
        start=segment['start'];end=segment['end'];a=segment['from'];b=segment['to']
        count=max(1,round((end-start)/.05))
        for i in range(count):
            now=round(start+(end-start)*i/count,8);f=i/count
            pos=tuple(a[k]+f*(b[k]-a[k]) for k in ('x','y'))
            if samples:lengths.append(math.dist(samples[-1][1],pos))
            samples.append((now,pos))
            if abs(now*2-round(now*2))>1e-5 or now-samples[0][0]<2.5:continue
            index=bisect_right(times,now)-1
            if index>=0 and visibility[index]['blocked']:current=None;continue
            walked=sum(lengths);positions=[p for _,p in samples]
            extent=max(math.dist(positions[0],p) for p in positions)
            net=math.dist(positions[0],pos)
            kind='stationary' if extent<.03 else 'oscillation' if walked>=.5 and extent<.35 and net<walked*.2 else None
            if kind is None:current=None;continue
            if current and current['kind']==kind and now-current['end_s']<=.75:current['end_s']=now
            else:
                current={'agent_id':agent['agent_id'],'kind':kind,'start_s':samples[0][0],'end_s':now,
                         'floor':a['floor'],'position':list(positions[0])};episodes.append(current)
    return [e for e in episodes if e['end_s']-e['start_s']>=3.]


def inspect(result_path,fds_dir=None):
    result=json.loads(Path(result_path).read_bytes());config=json.loads((ROOT/'configs/scenarios/gaehwasan.json').read_bytes())
    net=compile_network(config)
    sim=SimpleNamespace(add_waypoint_stage=lambda *args:1,add_journey=lambda *args:1,switch_agent_journey=lambda *args:None)
    areas={k:{'geometry':pedestrian_geometry(v),'sim':sim} for k,v in net['geometry'].items()}
    sensor=OccludedSensor(areas,net['nodes'],net['connectors'],FireField(config,directory=fds_dir),physical_config=config)
    agents={a['agent_id']:a for a in result['agents']}
    starts={aid:[s['start'] for s in a['segments']] for aid,a in agents.items()}
    checks=[];failures=[];walking=0;all_paths=0;safe_goals=0;blocked_paths=0;unsafe=[]
    for decision in result['decisions']:
        aid=decision['agent_id'];now=decision['time_s'];o=decision['observation'];area=(o.get('private_knowledge') or {}).get('area')
        index=bisect_right(starts[aid],now)-1
        if index<0 or area not in areas:continue
        s=agents[aid]['segments'][index]
        if s['status']!='walking' or not s['start']<=now<s['end']:continue
        walking+=1;f=(now-s['start'])/(s['end']-s['start'])
        pos=tuple(s['from'][k]+f*(s['to'][k]-s['from'][k]) for k in ('x','y'))
        body=observed_connector_checker(areas[area]['geometry'],pos)
        for option in o['options']:
            first=next((p for p in option['path'] if math.dist(pos,p)>1e-7),None)
            path={**option,'id':'recorded','kind':'up_stair'}
            ctl=PrivateController.__new__(PrivateController);ctl.areas=areas;ctl.sensor=sensor;ctl.stairs={}
            ctl.minds={aid:SimpleNamespace(pending=set())};ctl.observation=lambda *args,path=path:{'options':[path]}
            runtime={'agent':{'agent_id':aid},'area':area,'floor':o['floor'],'pos':pos,'phase':'walking',
                     'goal':pos,'goal_is_portal':False,'target':'recorded','jid':1}
            ctl.update_navigation(runtime,now)
            all_paths+=1
            if runtime.get('navigation_blocked'):blocked_paths+=1
            else:
                safe=(math.dist(pos,runtime['goal'])<=4.5+1e-7 and body(runtime['goal'])
                      and sensor.visible(area,o['floor'],pos,runtime['goal'],now))
                safe_goals+=bool(safe)
                if not safe:unsafe.append({'agent_id':aid,'time_s':now,'target':option['id'],'goal':runtime['goal']})
            if first is None or math.dist(pos,first)<=4.5:continue
            if not body(first) or not sensor.visible(area,o['floor'],pos,first,now):continue
            ok=(not runtime.get('navigation_blocked') and .1<math.dist(pos,runtime['goal'])<=4.5+1e-7
                and body(runtime['goal']) and sensor.visible(area,o['floor'],pos,runtime['goal'],now))
            row={'agent_id':aid,'time_s':now,'target':option['id'],'original_distance_m':math.dist(pos,first),
                 'new_goal':runtime['goal'],'passed':ok}
            checks.append(row)
            if not ok:failures.append(row)
    episodes=[e for a in agents.values() for e in stalled_episodes(a)]
    return {'people_checked':len(agents),'walking_decisions_checked':walking,
            'all_recorded_paths_checked':all_paths,'safe_visible_goals':safe_goals,
            'paths_requiring_new_observation_or_contact_recovery':blocked_paths,'unsafe_goal_count':len(unsafe),'unsafe_goals':unsafe,
            'historical_nonvisibility_stalls':episodes,'distant_path_cases_checked':len(checks),
            'distant_path_people':sorted({x['agent_id'] for x in checks}),
            'distant_path_replays':checks,'distant_path_failures':failures,
            'scope':'원본 100명 궤적 전수 스캔 및 기록된 먼 경로의 새 제어기 대조. 수정 후 전체 시뮬레이션 결과는 아직 없음.',
            'new_llm_calls':0,'full_simulation_recalculated':False,'saved_result_modified':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('result',type=Path);parser.add_argument('--fds-dir',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.resolve()==args.result.resolve():raise ValueError('검증 보고서로 원본 결과를 덮어쓸 수 없습니다.')
    report=inspect(args.result,args.fds_dir);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='distant_path_replays'},ensure_ascii=True,indent=2))
    assert not report['distant_path_failures'] and not report['unsafe_goals']
