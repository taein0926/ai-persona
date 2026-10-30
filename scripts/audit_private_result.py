"""Audit actual private-agent artifacts without requiring evacuation or differences."""
import argparse
import bisect
from collections import Counter,defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path

from persona_evac.agent_policy import POLICY,VERSIONS,POLICY_HASH
from persona_evac.hashing import canonical_json
from persona_evac.persona.experimental_profile import validate_profile,FULL_TRAITS as TRAITS
from persona_evac.integration.personal_space import familiarity_basis,cell_key
from persona_evac.integration.private_strategy import resolve_private
from persona_evac.llm.private_decisions import project_private,validate_private
from persona_evac.llm.ollama import atomic_json
from persona_evac.station_artifacts import policy_matches


def audit_native_passages(result, config):
    """펼친 JuPedSim 계단·개찰구의 실제 XYZ 기록과 점유를 검사한다."""
    from persona_evac.integration.station_connectors import compile_network
    network = compile_network(config)
    connectors = network['connectors']
    end = result['summary']['total_time_s']
    by_connector = defaultdict(list)
    by_person = defaultdict(list)
    frames = defaultdict(dict)
    engine_frames = defaultdict(dict)
    samples = 0
    for p in result['passages']:
        sid = p['stair_id']
        assert sid in connectors and p['engine']=='jupedsim'
        assert p['kind']==connectors[sid]['kind']
        assert 0 <= p['lane'] < connectors[sid]['parallel_lanes']
        assert p['arrival'] <= p['entry'] <= (p['exit'] if p['exit'] is not None else end)
        by_connector[sid].append(p)
        by_person[p['agent_id']].append(p)
        times = [row[0] for row in p['native_motion']]
        assert times and times==sorted(times)
        assert abs(times[0]-p['entry'])<1e-6
        assert p['exit'] is None or abs(times[-1]-p['exit'])<1e-6
        spec = connectors[sid]['native_lane_specs'][p['lane']]
        assert math.dist(p['native_motion'][0][1:], spec['entry'])<1e-6
        if p['exit'] is not None:
            assert math.dist(p['native_motion'][-1][1:], spec['destination'])<1e-6
        for t, x, y, z in p['native_motion']:
            assert p['entry']-1e-6 <= t <= (p['exit'] if p['exit'] is not None else end)+1e-6
            assert all(math.isfinite(v) for v in (x,y,z))
            if p['exit'] is None or t < p['exit']:
                frames[sid, round(t, 8)][p['agent_id']] = (x,y,z)
            samples += 1
        for t,x,y in p.get('native_engine_motion',[]):
            assert all(math.isfinite(v) for v in (x,y))
            if p['exit'] is None or t<p['exit']:
                branch=connectors[sid]['native_lane_specs'][p['lane']]['branch']
                engine_frames[sid,branch,round(t,8)][p['agent_id']]=(x,y)
    peaks = {}
    for sid, passages in by_connector.items():
        events = [(p['entry'], 1) for p in passages]
        events += [(p['exit'], -1) for p in passages if p['exit'] is not None]
        occupied = peak = 0
        for _, change in sorted(events):
            occupied += change
            assert 0 <= occupied <= connectors[sid]['capacity'], (sid, occupied)
            peak = max(peak, occupied)
        peaks[sid] = peak
        if connectors[sid]['kind']=='fare_gate':
            assert peak <= 1, sid
    for aid, passages in by_person.items():
        for first, second in itertools.pairwise(sorted(passages,key=lambda p:p['entry'])):
            assert first['exit'] is not None and first['exit']<=second['entry'], aid
    minimum = math.inf
    pairs = 0
    for (sid, t), people in frames.items():
        for (aid, a), (bid, b) in itertools.combinations(people.items(), 2):
            gap = math.dist(a,b)-.44
            minimum = min(minimum,gap)
            pairs += 1
            assert gap >= -.01, (sid,t,aid,bid,gap)
    engine_minimum=math.inf;engine_pairs=0
    for (sid,branch,t),people in engine_frames.items():
        for (aid,a),(bid,b) in itertools.combinations(people.items(),2):
            gap=math.dist(a,b)-.44
            engine_minimum=min(engine_minimum,gap);engine_pairs+=1
            assert gap>=-.01,(sid,branch,t,aid,bid,gap)
    return {'connectors':len(connectors), 'used_connectors':len(peaks), 'peak_occupancy':peaks,
            'native_motion_samples':samples, 'native_body_pair_checks':pairs,
            'native_min_surface_gap_m':minimum if pairs else None,
            'engine_body_pair_checks':engine_pairs,
            'engine_min_surface_gap_m':engine_minimum if engine_pairs else None,
            'gate_capacity_one_verified':True, 'person_passages_do_not_overlap':True}


def audit(path,source_path=Path('output/gaehwasan/persona_source.json')):
    r=json.loads(path.read_text(encoding='utf-8'))
    source=json.loads(source_path.read_text(encoding='utf-8'))
    originals={a['persona']['uuid']:a for a in source['variants']['C']}
    raw={p['uuid']:p for p in map(json.loads,Path('data/raw/nemotron_personas_100.jsonl').read_text(encoding='utf-8').splitlines())}
    agents={a['agent_id']:a for a in r['agents']}
    assert len({a['persona']['uuid'] for a in agents.values()})==len(agents)
    assert policy_matches(r)
    classification_method=r['agent_policy'].get('classification_method','keyword_candidates')
    from persona_evac.llm.baseline_profile import POLICY as BASELINE_POLICY, validate_mapping
    from persona_evac.empirical import EmpiricalResolver, load_empirical_records
    origins=Counter();levels={t:Counter() for t in TRAITS};familiar=Counter()
    for a in agents.values():
        old=originals[a['persona']['uuid']]
        assert all(a['persona'].get(k)==v for k,v in raw[a['persona']['uuid']].items())
        assert a['persona']==old['persona']
        if a['provenance']['mapping'].get('policy')==BASELINE_POLICY:
            payload={key:{'level':a['mapping'][key],
                'assumption':a['mapping']['reasoning_summary'][key]} for key in ('response_tendency','speed_tendency')}
            payload['familiarity']=a['mapping']['familiarity']
            mapping=validate_mapping(payload)
            assert a['mapping']==mapping.model_dump(mode='json')
            resolver=EmpiricalResolver(load_empirical_records(Path(a['provenance']['empirical']['source_ref'])),
                                       source['provenance']['snapshot']['selection_seed'])
            expected=resolver.resolve(persona_uuid=a['persona']['uuid'],variant='C',treatment='individual',
                response=mapping.response_tendency,speed=mapping.speed_tendency)
            assert a['empirical']==expected.model_dump(mode='json')
            assert a['pre_movement_s']==expected.response_time_s and a['desired_speed_mps']==expected.desired_speed_mps
        else:
            for key in ('mapping','empirical','pre_movement_s','desired_speed_mps'):
                assert a[key]==old[key],(a['agent_id'],key)
        p=validate_profile(a['experimental_profile'],a['persona'],r['agent_policy']['experiment_seed'])
        assert canonical_json(a['familiarity'])==canonical_json(familiarity_basis(a['persona'],a.get('mapping')))
        assert a['familiarity']['distance_to_station_m'] is None
        assert not a['familiarity']['current_conditions_known']
        assert all(l['current_state'] is None for l in a['initial_knowledge'].values())
        familiar[a['familiarity']['level']]+=1
        for t,v in p['traits'].items():origins[v['origin']]+=1;levels[t][v['level']]+=1
        for key in ('observations','people_observations'):
            times=[x['time_s'] for x in a[key]]
            assert times==sorted(times)
            assert len({x['id'] for x in a[key]})==len(times)
        for s in a['segments']:
            if s['status']=='waiting':assert s['start']>=a['pre_movement_s']
        assert bool(a['noncompletion_cause'])==(a['evacuation_time_s'] is None)

    forbidden={'temperature_c','extinction_per_m','available_routes','planned_exit','expected_exit_time_s','queue_count','stair_capacity','destination','route_plan'}
    def no_forbidden(value):
        if isinstance(value,dict):
            assert not forbidden.intersection(value)
            for v in value.values():no_forbidden(v)
        elif isinstance(value,list):
            for v in value:no_forbidden(v)
    counts=Counter();choices_by_trait={t:defaultdict(Counter) for t in TRAITS}
    last_routine={};routine=Counter();critical_after_limits=0;known_path_points=0;actual_cache=0;model_goal_revisions=0
    for d in r['decisions']:
        aid=d['agent_id'];a=agents[aid];o=d['observation'];now=d['time_s']
        assert o['agent_id']==aid and o['profile']==a['experimental_profile']
        assert now>=a['pre_movement_s']
        assert d['policy_versions']==r['agent_policy']['versions'] and d['policy_hash']==POLICY_HASH
        if d['source'] in {'ollama_live','ollama_cache'}:
            assert d['subject_agent_id']==aid and d['subject_uuid']==a['persona']['uuid']
            assert d['context_people_count']==1
            assert d['input_sha256']==hashlib.sha256(canonical_json([project_private(o)]).encode()).hexdigest()
            actual_cache+=d['source']=='ollama_cache'
        projected=project_private(o);no_forbidden(projected)
        assert 'policy_hash' not in projected and 'prior_hash' not in projected['familiarity']
        own_obs={p['id']:p for p in a['observations']}
        for p in o['observations']:
            assert p==own_obs[p['id']] and p['time_s']<=now
        own_mem={m['id']:m for m in a['personal_memory']}
        for m in o['memory']:
            final=own_mem[m['id']]
            for key in ('id','time_s','floor','action','stair_id','selection_source'):assert m[key]==final[key]
            assert m['time_s']+m['elapsed_s']<=now+.11
            assert m['elapsed_s']<=final['elapsed_s']+.11
        for p in o['options']:
            assert set(p['observation_ids'])<=set(own_obs)
            known=o['private_knowledge']['known_landmarks']
            assert p['kind']=='frontier' or p['id'] in {l['id'] for l in known}
            if not p['state_is_current']:assert p['visible_people'] is None and p['hazard'] is None
            cells={tuple(v[:2]):v[2] for v in a['known_map'][o['private_knowledge']['area']]['cell_first_seen']}
            edges={tuple(v[:4]):v[4] for v in a['known_map'][o['private_knowledge']['area']]['edge_first_seen']}
            previous=None
            for xy in p['path']:
                k=cell_key(xy)
                # The terminal stair point can be off-grid; its observed anchor precedes it.
                if abs(xy[0]-k[0]*POLICY['sensor']['grid_m'])<1e-6 and abs(xy[1]-k[1]*POLICY['sensor']['grid_m'])<1e-6:
                    assert k in cells and cells[k]<=now
                    if previous is not None and k!=previous:
                        assert (*previous,*k) in edges and edges[*previous,*k]<=now
                    previous=k
                    known_path_points+=1
                else:previous=None
        verified=validate_private({str(aid):d['raw_model_choice']},[o])[0]
        assert verified['strategy_costs']==d['strategy_costs']
        if o['phase']!='on_stairs':assert verified['final_target']==d['final_target']
        assert set(d['used_memory_ids'])<={m['id'] for m in o['memory']}
        assert set(d['used_observation_ids'])<={p['id'] for p in o['observations']}
        if d['requested_movement']=='wait':assert o['own_flat_walked_m']>=.5 and o['wait_context_observed']
        essential=bool(set(d['events'])&{'critical_risk','floor_changed'})
        if not essential:
            assert now-last_routine.get(aid,-1e9)>=POLICY['decision']['routine_minimum_interval_s']-1e-5
            routine[aid]+=1;assert routine[aid]<=POLICY['decision']['routine_maximum_calls_per_person']
        elif routine[aid]>=POLICY['decision']['routine_maximum_calls_per_person']:critical_after_limits+=1
        last_routine[aid]=now
        counts[(d['source'],d['priority'],d['requested_movement'])]+=1
        model_goal_revisions+=d['final_target']!=o['current_target']
        for t,v in a['experimental_profile']['traits'].items():choices_by_trait[t][v['level']][d['priority']]+=1
    assert actual_cache==r['performance']['cache_request_count']
    attempts=r['behavior']['inference_attempts']
    assert len(attempts)==r['performance']['llm_call_count']
    assert sum(x['status']=='failed' for x in attempts)==r['performance']['llm_failed_attempt_count']

    native=bool(r['scenario'].get('jupedsim_connections',{}).get('enabled'))
    native_report=None
    if native:
        config=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf-8'))
        native_report=audit_native_passages(r,config)
        from persona_evac.integration.station_connectors import compile_network,path_length
        stairs=compile_network(config)['connectors']
    else:
        stairs={s['id']:s for s in r['scenario']['stairs']}
    groups=defaultdict(list)
    for p in r['passages']:groups[p['stair_id'],p['lane']].append(p)
    minimum=float('inf');pairs=0
    for (sid,lane),ps in groups.items():
        for front,rear in itertools.pairwise(sorted(ps,key=lambda p:p['entry'])):
            ft=[m[0] for m in front['motion']]
            for t,distance in rear['motion']:
                if t>=(front['exit'] or r['summary']['total_time_s']):break
                i=bisect.bisect_right(ft,t)-1
                if i<0:continue
                length=path_length(stairs[sid]['native_paths'][stairs[sid]['native_lane_specs'][lane]['branch']]) if native else stairs[sid]['length_m']
                gap=(front['motion'][i][1]-distance)*length
                minimum=min(minimum,gap);pairs+=1
                if not native:assert gap>=.999,(sid,lane,t,gap)
    diagnostics=r['movement_diagnostics']
    from persona_evac.integration.station_dynamics import can_intentionally_overtake
    threshold=r['scenario']['movement']['overtaking_policy']['fast_threshold_mps']
    for plan in diagnostics['overtake_plans']:
        assert can_intentionally_overtake(agents[plan['agent_id']],plan['desired_speed_mps'],threshold)
        assert plan['desired_speed_mps']-plan['front_speed_mps']>=.30-1e-9
        assert plan['front_speed_mps']<=.70*plan['desired_speed_mps']+1e-9
    for passed in diagnostics['overtakes']:
        assert passed['min_surface_gap_m']>=-.01 and passed['crossing_lateral_m']>=.44
    fds_checked=0
    if r['fire']['enabled']:
        from persona_evac.integration.fds_environment import FireField
        config=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf-8'))
        field=FireField(config)
        assert field.meta['fields_sha256']==r['fire']['metadata']['fields_sha256']
        assert r['summary']['total_time_s']<=float(field.times[-1])-field.offset+1e-5
        by_person=defaultdict(list)
        for h in r['hazard_history']:
            if not h.get('stair_id'):by_person[h['agent_id']].append(h)
        for history in by_person.values():
            for h in [history[0],history[len(history)//2],history[-1]]:
                expected=field.sample(h['floor'],(h['x'],h['y']),h['time_s'])
                assert h['extinction_per_m']==expected['extinction_per_m'] and h['temperature_c']==expected['temperature_c']
                fds_checked+=1
    report={'artifact_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'count':len(agents),
        'checks_passed':['원본·UUID·Jülich 값 보존','성향 인용·배정·정책 버전 검증','거주지 기준·과거 지식 구분',
                         '한 사람만 독립 추론','본인 관측·기억 참조 및 시각 검증','개인 지도 관측 시각과 경로 대조',
                         '모델 입력에 정확한 FDS·공유 경로 없음','전략 비용·최종 목표 재계산 일치','일반 판단 한도·중요 사건 예외',
                         'JuPedSim 연결 몸 간격·점유·개찰구 단독 통행 검사' if native else '계단 같은 줄 간격 검사',
                         '추론 시도·실패·캐시 집계 일치'],
        'classification_method':classification_method,'profile_version':r['agent_policy']['versions']['profile'],
        'evidence_levels':dict(Counter(v.get('evidence_level',v['origin']) for a in agents.values() for v in a['experimental_profile']['traits'].values())),
        'profile_origins':dict(origins),'trait_levels':levels,'familiarity':dict(familiar),
        'summary':r['summary'],'noncompletion_causes':dict(Counter(a['noncompletion_cause'] for a in agents.values() if a['noncompletion_cause'])),
        'performance':r['performance'],'decision_count':len(r['decisions']),
        'fds_provenance_and_sampling_checks':fds_checked,
        'fallback_count':r['behavior']['fallback_count'],'event_counts':r['behavior']['event_counts'],
        'fallback_errors':dict(Counter(d.get('error','분류되지 않은 응답 오류') for d in r['decisions'] if d['source']=='fallback')),
        'termination':r.get('termination'),
        'critical_calls_after_routine_limit':critical_after_limits,'checked_known_path_points':known_path_points,
        'decisions_using_own_memory':sum(bool(d['used_memory_ids']) for d in r['decisions']),
        'decisions_with_strategy_linked_literal_evidence':sum(bool(d['used_evidence_ids']) for d in r['decisions']),
        'decision_histogram':[{'source':k[0],'priority':k[1],'movement':k[2],'count':v} for k,v in counts.items()],
        'priority_counts_by_trait':{t:{level:dict(c) for level,c in rows.items()} for t,rows in choices_by_trait.items()},
        'interpretation_limit':'개인별 위치·관측·경험도 다르므로 집단 통계만으로 성격의 인과 효과를 단정하지 않습니다.',
        'code_resolved_target_revisions':model_goal_revisions,'navigation_update_sources':dict(Counter(n['source'] for n in r['navigation_updates'])),
        'body_overlap_pair_steps':diagnostics['overlap_pair_steps'],'min_surface_gap_m':diagnostics['min_surface_gap_m'],
        'stair_following_steps':pairs,'minimum_same_lane_gap_m':minimum if pairs else None,
        'stair_used_lanes':{sid:len({p['lane'] for p in r['passages'] if p['stair_id']==sid}) for sid in stairs},
        'overtake_intents':sum(d['requested_movement']=='overtake' for d in r['decisions']),
        'overtake_plans':len(diagnostics['overtake_plans']),'completed_overtakes':len(diagnostics['overtakes'])}
    if native:report['native_connection_checks']=native_report
    report['interpretation_limit']='동일 상황의 통제 시험 이외의 개인차에는 위치·관측·경험 차이가 포함되므로 성격의 인과 효과를 단정하지 않습니다.'
    atomic_json(path.with_suffix('.audit.json'),report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('result',type=Path);args=parser.parse_args()
    print(json.dumps(audit(args.result),ensure_ascii=False,indent=2))
