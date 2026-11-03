"""Cognitive controller beside the existing physics, never an omniscient route oracle."""
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jupedsim as jps
from shapely.geometry import Polygon

from persona_evac.agent_policy import POLICY,VERSIONS,POLICY_HASH
from persona_evac.persona.experimental_profile import ExperimentalProfiles
from persona_evac.integration.agent_memory import AgentMemory
from persona_evac.integration.personal_space import PersonalSpace,OccludedSensor,SENSOR,smoke_label,heat_label,observed_connector_checker,station_stair_memories,station_route_memories
from persona_evac.integration.private_strategy import resolve_private
from persona_evac.llm.private_decisions import PrivateDecisions,validate_private

EVENT_LABELS={'movement_started':'이동 시작','floor_changed':'층 이동','new_route':'새 계단·경로 발견',
              'queue_delayed':'대기 증가','stalled':'이동 정체','risk_changed':'위험 변화','critical_risk':'중요 위험 상승','slow_front':'느린 앞사람 관측'}


def default_choice(o):
    blocked=[m['id'] for m in o['memory'] if m['result']=='still_blocked'][-1:]
    return {'priority':'experience' if blocked else 'environment','trait_refs':[],'memory':blocked,
            'observations':[p['id'] for p in o['observations'][-1:]],'movement':'continue'}


class PrivateMind(AgentMemory):
    def __init__(self,agent,profile,prior_landmarks=()):
        super().__init__(agent);self.profile=profile;self.space=PersonalSpace(agent['persona'],agent.get('mapping'),prior_landmarks)
        self.pending=set();self.routine_count=0;self.emergency_count=0;self.selected=None
        self.last_floor=agent.get('spawn',{}).get('floor','B2');self.last_wait=0;self.anchor=None
        self.navigation_updates=[];self.failed_lanes={};self.decision_refs=[]
        self.last_decision_time=-1e9

    def outcome(self,r,now):
        outcome=super().outcome(r,now)
        if outcome:
            outcome['id']=f'M{len(self.history)+1}'
            outcome['selection_source']=self.last.get('selection_source','model_strategy')
            outcome['stalled_s']=round(min(outcome['elapsed_s'],max(0,now-r.get('stall_since',now))),1) if r['phase'] in {'walking','queueing'} else 0
            if r['floor']==self.last['floor'] and r['phase'] not in {'on_stairs','evacuated'} and outcome['walked_m']<.5 and outcome['stalled_s']>=6:
                outcome['result']='still_blocked'
        return outcome

    def navigation_experience(self,r,target,now,source):
        if self.last and self.last['target']==target:return
        current=self.outcome(r,now)
        if current:self.history.append(current)
        self.last={'time':now,'floor':r['floor'],'target':target,'action':'navigate',
                   'walked':r['agent']['distance_walked_m'],'waited':self.waited(r,now),'selection_source':source}

    def commit_private(self,r,now,choice,o):
        current=self.outcome(r,now)
        if current:self.history.append(current)
        self.count+=1
        if 'critical_risk' in o['events']:self.emergency_count+=1
        elif 'floor_changed' not in o['events']:self.routine_count+=1
        self.last={'time':now,'floor':r['floor'],'target':choice['final_target'] or r['target'],
                   'action':choice['applied_action'],'walked':r['agent']['distance_walked_m'],
                   'waited':self.waited(r,now),'selection_source':'model_strategy' if choice.get('decision_source')!='fallback' else 'private_fallback'}
        self.selected=choice['model_choice'];self.last_wait=o['queue_wait_s'];self.pending.clear()
        self.anchor=(now,tuple(r['pos']));self.last_floor=r['floor']
        self.last_decision_time=now

    def events_private(self,r,now):
        if self.anchor is None or math.dist(self.anchor[1],r['pos'])>.3:self.anchor=(now,tuple(r['pos']))
        if self.count==0:self.pending.add('movement_started')
        if r['floor']!=self.last_floor:self.pending.add('floor_changed')
        wait=max(0,now-r.get('arrived_at',now)) if r['phase']=='queueing' else 0
        if wait>=self.last_wait+POLICY['decision']['queue_increase_s']:self.pending.add('queue_delayed')
        if r['phase']=='walking' and now-self.anchor[0]>=POLICY['decision']['stalled_s']:self.pending.add('stalled')
        essential=bool(self.pending & {'critical_risk','floor_changed'})
        if not essential and (self.routine_count>=POLICY['decision']['routine_maximum_calls_per_person'] or
            self.count and now-self.last_decision_time<POLICY['decision']['routine_minimum_interval_s']):return []
        return sorted(self.pending)


class VisibleSimulation:
    """Motion helpers can inspect visible bodies only; no intentions are exposed."""
    def __init__(self,sim,visible_ids):
        self.sim=sim;self.ids=visible_ids;self.waypoint_positions={}
    def add_waypoint_stage(self,position,distance):
        stage=self.sim.add_waypoint_stage(position,distance)
        self.waypoint_positions[stage]=tuple(position)
        return stage
    def agents(self):return [a for a in self.sim.agents() if a.id in self.ids]
    def agent(self,aid):
        if aid not in self.ids:raise ValueError('관측하지 않은 사람을 검사할 수 없습니다.')
        return self.sim.agent(aid)
    def __getattr__(self,key):
        if key not in {'add_journey','switch_agent_journey'}:
            raise AttributeError('국소 이동 검사에는 전체 지도나 다른 사람의 계획을 제공하지 않습니다.')
        return getattr(self.sim,key)


class PrivateController:
    policy_versions=VERSIONS
    def __init__(self,agents,areas,stairs,nodes,runtime,fire,mode,cache_dir,provider=None,seed=None,physical_config=None,profile_method=None):
        self.areas,self.stairs,self.nodes,self.runtime=areas,stairs,nodes,runtime
        self.mode=mode;self.provider=provider
        try:self.decision_workers=int(os.environ.get('PERSONA_EVAC_DECISION_WORKERS','2')) if mode=='qwen' else 1
        except ValueError:raise ValueError('PERSONA_EVAC_DECISION_WORKERS는 1~16 사이의 정수여야 합니다.') from None
        if not 1<=self.decision_workers<=16:
            raise ValueError('PERSONA_EVAC_DECISION_WORKERS는 1~16 사이의 정수여야 합니다.')
        self.decision_wall_s=0.0
        self.fire=fire;self.physical_config=physical_config
        if mode=='qwen' and provider is None:self.provider=PrivateDecisions(cache_dir)
        local=getattr(self.provider,'local',None)
        self.profiles_provider=ExperimentalProfiles(Path(os.environ.get('PERSONA_EVAC_OUTPUT','output'))/'gaehwasan'/'private_profile_cache',local,seed,method=profile_method)
        self.profiles=self.profiles_provider.prepare(agents)
        self.policy_versions={**VERSIONS,'profile':self.profiles_provider.version}
        self.profile_inference_calls=sum(record.get('source')=='ollama_live' for record in self.profiles_provider.records)
        prior_landmarks=station_stair_memories(nodes,stairs)
        self.minds={a['agent_id']:PrivateMind(a,self.profiles[a['agent_id']],prior_landmarks) for a in agents}
        if any(mind.space.familiarity['level']=='familiar' for mind in self.minds.values()):
            prior_maps=station_route_memories(areas,prior_landmarks)
            for mind in self.minds.values():
                if mind.space.familiarity['level']=='familiar':mind.space.prior_maps=prior_maps
        self.sensor=OccludedSensor(areas,nodes,stairs,fire,(physical_config or {}).get('wayfinding_signs',[]),physical_config)
        self.decisions=[];self.navigation_updates=[];self.safety_rejections=[]
        self.local_geom_cache={};self.decision_batches=0

    def observation(self,r,now,events=(),with_knowledge=True):
        aid=r['agent']['agent_id'];mind=self.minds[aid];space=mind.space
        options=space.options(r['area'],r['pos'],r['agent']['desired_speed_mps'],now,r['target'])
        if r['phase']=='on_stairs':options=[]
        observations=space.observations_for_ids(space.last_observation_ids)[-6:]
        # A reference to an old observation remains explicitly dated and stale.
        ids={oid for p in options for oid in p['observation_ids']}
        observations+= [p for p in space.observations_for_ids(ids) if p not in observations][-4:]
        referenced=[p for p in observations if p['id'] in ids]
        local=[p for p in observations if p['kind']=='local_hazard'][-1:]
        observations=list({p['id']:p for p in referenced[-3:]+local+observations[-1:]}.values())[-4:]
        allowed=['continue','hurry']
        queue_wait=max(0,now-r.get('arrived_at',now)) if r['phase']=='queueing' else 0
        wait_context=(queue_wait>0 or sum(p.get('area',r['area'])==r['area'] for p in space.last_visible_people)>=3 and r.get('observed_speed_mps',0)<.25)
        wait_allowed=r['agent']['distance_walked_m']>=.5 and wait_context
        if wait_allowed and r['wait_streak']<2 and 'critical_risk' not in events:allowed.append('wait')
        if r['phase']=='on_stairs':allowed=['continue']
        return {'agent_id':aid,'time_s':now,'events':list(events),'floor':r['floor'],'phase':r['phase'],
                'profile':mind.profile,'familiarity':space.familiarity,'current_target':r['target'],
                'memory':mind.memory(r,now),'observations':observations,'options':options,
                'local_hazard':space.last_risk,
                'queue_wait_s':round(max(0,now-r.get('arrived_at',now)),2) if r['phase']=='queueing' else 0,
                'allowed_movements':allowed,'private_knowledge':space.snapshot(r['area'],now) if with_knowledge else None,
                # Preserve the saved no-overtaking input shape and response cache.
                'front_observation':{'slower_walker_and_clear_side_observed':False},
                'own_flat_walked_m':r['agent']['distance_walked_m'],'wait_context_observed':wait_context,
                'policy_versions':self.policy_versions,'policy_hash':POLICY_HASH}

    def set_target(self,r,target,now,source):
        target=target or 'explore:none'
        if target==r.get('target'):return
        aid=r['agent']['agent_id'];mind=self.minds[aid]
        mind.navigation_experience(r,target,now,source)
        if r['phase']=='queueing':r['agent']['stair_wait_s']+=now-r['arrived_at']
        old=r.get('target');r.update(target=target or 'explore:none',phase='walking',passing=False,goal_is_portal=False)
        item={'agent_id':aid,'time_s':now,'from_target':old,'target':r['target'],'source':source,
              'based_on_model_decision':mind.count if mind.selected else None,
              'knowledge_revision':mind.space.revision,'policy_versions':self.policy_versions}
        mind.navigation_updates.append(item);self.navigation_updates.append(item)

    def update_navigation(self,r,now,force=False):
        r.pop('blocked_motion_target',None)
        # Navigation costs read explicit options, memory and observations; only
        # actual model requests need the additional full knowledge snapshot.
        aid=r['agent']['agent_id'];mind=self.minds[aid];o=self.observation(r,now,with_knowledge=False)
        options={p['id']:p for p in o['options']};current=options.get(r['target'])
        reached=(current and current['kind']=='frontier'
                 and math.dist(r['pos'],current['position'])<current.get('arrival_radius_m',1.)
                 and self.sensor.visible(r['area'],r['floor'],r['pos'],current['position'],now)
                 and observed_connector_checker(self.areas[r['area']]['geometry'],r['pos'])(current['position']))
        if reached:
            mind.space.mark_frontier_reached(r['area'],current['position'],now)
            mind.space.record(now,'frontier_reached',{'area':r['area'],'floor':r['floor'],
                'position':list(current['position']),'observer_position':list(r['pos']),
                'arrival_radius_m':current.get('arrival_radius_m',1.)},'own_navigation')
            o=self.observation(r,now,with_knowledge=False)
            options={p['id']:p for p in o['options']};current=options.get(r['target'])
        new_stair=('new_route' in mind.pending and r['target'].startswith('explore:') and
                   len(mind.space.landmarks)>r.get('navigation_known_landmarks',0))
        blocked=now-r.get('stall_since',now)>=6 and now-r.get('local_replan_after',-100)>=10
        if not current or reached or new_stair or force or blocked:
            choice=default_choice(o) if blocked else mind.selected or default_choice(o)
            choice={**choice,'memory':[m for m in choice['memory'] if m in {m['id'] for m in o['memory']}]}
            target,_=resolve_private(o,choice)
            source='private_rule_after_blocked_experience' if blocked else 'personal_rule_after_observation' if source_is_rule(mind) else 'navigation_for_last_model_strategy'
            self.set_target(r,target,now,source)
            r['navigation_known_landmarks']=len(mind.space.landmarks)
            if blocked:r['local_replan_after']=now
            current=options.get(r['target'])
        if current is None:
            r['navigation_blocked']='no_observed_reachable_frontier_or_portal'
            r['movement_block_reason']='route_unavailable';return
        goal=None;portal=False;lane=None;stage=None;journey=None
        connector=observed_connector_checker(self.areas[r['area']]['geometry'],r['pos'])
        if r['target'] in self.stairs and math.dist(r['pos'],current['position'])<4.5:
            # Entrance-local points only. Landing admissibility stays inside physics.
            visible={p['id']:p for p in mind.space.last_visible_people if p.get('area',r['area'])==r['area']};candidates=[]
            for option,lane_number in zip(self.areas[r['area']]['lanes'][r['target']],self.stairs[r['target']]['entry_lane_numbers']):
                j,s,p=option
                if not self.sensor.visible(r['area'],r['floor'],r['pos'],p,now):continue
                if not connector(p):continue
                ahead=sum(math.dist(person['position'],p)<3 for person in visible.values())
                own_failure=mind.failed_lanes.get((r['target'],lane_number),0)
                score=math.dist(r['pos'],p)/r['agent']['desired_speed_mps']+ahead*1.2+own_failure*5
                candidates.append((score,lane_number,j,s,p))
            if candidates:
                _,lane,journey,stage,goal=min(candidates);portal=True
        if goal is None:
            blocked_by_visibility=False
            for p in current['path']:
                d=math.dist(r['pos'],p)
                # A nearby corner is still needed when the following segment
                # cannot be reached with body clearance from the current position.
                # Only discard a point coincident with the pedestrian center.
                if d<=1e-7:continue
                # A direct observed landmark can be many metres away, without
                # grid anchors. Limit the next step, not the existence of a route.
                fraction=min(1.,4.5/d)
                candidate=tuple(r['pos'][i]+fraction*(p[i]-r['pos'][i]) for i in (0,1))
                visible=self.sensor.visible(r['area'],r['floor'],r['pos'],candidate,now)
                if visible and connector(candidate):
                    goal=candidate
                else:
                    blocked_by_visibility=not visible
                    if visible and goal is None:r['blocked_motion_target']=candidate
                    # Remembered grid points can be farther apart than the
                    # current smoke visibility. Walk only a currently visible
                    # prefix of the known, structurally clear segment.
                    if not visible and connector(candidate):
                        low,high=0.,fraction
                        for _ in range(10):
                            fraction=(low+high)/2
                            q=tuple(r['pos'][i]+fraction*(p[i]-r['pos'][i]) for i in (0,1))
                            if self.sensor.visible(r['area'],r['floor'],r['pos'],q,now):low=fraction
                            else:high=fraction
                        # FDS node/sample counts can change at the shortened
                        # endpoint. Verify the actual final point as well.
                        for retreat in (.05,0.):
                            distance=low*d-retreat
                            if distance<=.1:continue
                            q=tuple(r['pos'][i]+distance/d*(p[i]-r['pos'][i]) for i in (0,1))
                            if connector(q) and self.sensor.visible(r['area'],r['floor'],r['pos'],q,now):
                                goal=q;break
                    break
                if d>4.5:break
            if goal is None:
                r['navigation_blocked']='known_path_not_currently_visible'
                r['movement_block_reason']='visibility' if blocked_by_visibility else 'route_unavailable';return
        r['navigation_blocked']=None
        r['movement_block_reason']=None
        if r['phase']=='queueing' and r.get('entry_lane')==lane and portal:return
        if (r.get('goal') and math.dist(goal,r['goal'])<.15 and r.get('goal_is_portal')==portal
                and math.dist(r['pos'],r['goal'])<=4.5
                and (math.dist(goal,r['goal'])<1e-7 or math.dist(r['pos'],r['goal'])>.15)
                and connector(r['goal'])
                and self.sensor.visible(r['area'],r['floor'],r['pos'],r['goal'],now)):return
        sim=self.areas[r['area']]['sim']
        if stage is None:
            stage=sim.add_waypoint_stage(goal,.08);journey=sim.add_journey(jps.JourneyDescription([stage]))
        sim.switch_agent_journey(r['jid'],journey,stage)
        if r['phase']=='queueing':r['agent']['stair_wait_s']+=now-r['arrived_at']
        r.update(goal=tuple(goal),goal_stage=stage,goal_is_portal=portal,entry_lane=lane,passing=False,phase='walking')

    def _decide_one(self,o):
        # Every HTTP request and validation context contains exactly one person.
        chunk=[o]
        try:
            selected,meta=self.provider.decide(chunk)
            if any('model_choice' not in d for d in selected):raise ValueError('새 개인 지식 정책의 응답 형식이 아닙니다.')
            selected=validate_private({str(d['agent_id']):d.get('raw_model_choice',d['model_choice']) for d in selected},chunk)
        except Exception as exc:
            selected=validate_private({str(o['agent_id']):default_choice(o)},chunk)
            meta={'source':'fallback','model':getattr(self.provider,'model',None),'error':str(exc)}
        return selected,meta

    def decide_batch(self,batch):
        started=time.perf_counter()
        try:
            if self.decision_workers==1 or len(batch)<=1:
                return [self._decide_one(o) for o in batch]
            with ThreadPoolExecutor(max_workers=min(self.decision_workers,len(batch)),thread_name_prefix='private-qwen') as executor:
                # Collect every response before changing physics, knowledge or memory.
                # map retains input order even when responses finish out of order.
                return list(executor.map(self._decide_one,batch))
        finally:
            self.decision_wall_s+=time.perf_counter()-started

    def tick(self,now):
        batch=[]
        for aid,r in self.runtime.items():
            if r['phase']=='evacuated' or r.get('incapacitated'):continue
            mind=self.minds[aid];space=mind.space
            if r['phase']=='on_stairs':
                if self.fire and now-space.last_sense>=SENSOR['refresh_s']-1e-6:
                    stair=self.stairs[r['target']]
                    raw=self.fire.sample_stair(self.physical_config,stair,
                         {'from':r['stair_start'],'to':r['stair_destination']},r['stair_distance']/stair['length_m'],now)
                    h={'smoke':smoke_label(raw['extinction_per_m']),'heat':heat_label(raw['temperature_c'])}
                    old=space.last_risk;space.last_risk=h;space.last_sense=now;space.last_visible_people=[]
                    space.people_view_active=False;space.visible_landmark_ids=set()
                    oid=space.record(now,'local_hazard',{'floor':r['floor'],'perceived':h,'location_context':'ascending_stair'},'stair_sensation')
                    space.last_observation_ids=[oid]
                    smoke_order={'clear':0,'light':1,'dense':2,'opaque':3};heat_order={'normal':0,'warm':1,'hot':3}
                    critical=(smoke_order[h['smoke']]>smoke_order[old['smoke']] and smoke_order[h['smoke']]>=2 or
                              heat_order[h['heat']]>heat_order[old['heat']] and heat_order[h['heat']]>=3)
                    if critical:mind.pending.add('critical_risk')
                if self.mode=='qwen' and 'critical_risk' in mind.pending and abs(now-round(now))<1e-6:
                    batch.append(self.observation(r,now,sorted(mind.pending)))
                continue
            if now-space.last_sense>=SENSOR['refresh_s']-1e-6 or r['floor']!=getattr(space,'sensed_floor',None) or space.last_sense<0:
                change=self.sensor.sense(space,r,self.runtime,now)
                space.sensed_floor=r['floor']
                if change['discovered']:mind.pending.add('new_route')
                if change['risk_changed']:mind.pending.add('risk_changed')
                if change['critical']:mind.pending.add('critical_risk')
            if r['phase']=='pre_moving':continue
            if r['phase']=='queueing' and now-r['arrived_at']>=5 and now-r.get('lane_failure_at',-100)>=5:
                key=(r['target'],r.get('entry_lane'));mind.failed_lanes[key]=mind.failed_lanes.get(key,0)+1;r['lane_failure_at']=now
            if not r.get('passing') and r['phase']!='waiting' and (now>=r.get('navigation_after',0) or math.dist(r['pos'],r['goal'])<.65):
                self.update_navigation(r,now);r['navigation_after']=now+.5
            if self.mode=='qwen' and abs(now-round(now))<1e-6:
                events=mind.events_private(r,now)
                if events:batch.append(self.observation(r,now,events))
        if not batch:return
        self.decision_batches+=1
        outcomes=self.decide_batch(batch)
        for o,(selected,meta) in zip(batch,outcomes):
            chunk=[o]
            if meta['source']=='fallback':
                print(json.dumps({'time_s':now,'private_decision_error':meta['error'],'agent_ids':[o['agent_id']]},ensure_ascii=False),flush=True)
            for choice,o in zip(selected,chunk):
                r=self.runtime[choice['agent_id']];mind=self.minds[choice['agent_id']]
                movement=choice['requested_movement'];target=choice['final_target']
                choice['decision_source']=meta['source']
                if r['phase']=='on_stairs':
                    choice['final_target']=r['target'];choice.update(action='continue',applied_action='continue',target_stair=None,
                        execution_constraint='계단 내부의 방향 전환·속도 제어는 지원하지 않아 통행을 유지하고 다음 층에서 재판단',
                        reason='중요 위험을 관측했으며 계단 통행 제약을 유지합니다.')
                    mind.commit_private(r,now,choice,o)
                    self.decisions.append({**choice,**meta,'time_s':now,'events':o['events'],
                        'event_labels':[EVENT_LABELS[e] for e in o['events']],'observation':o,
                        'policy_versions':self.policy_versions,'policy_hash':POLICY_HASH,'executed_target':r['target'],
                        'executed_goal':None,'execution_source':'stair_physical_constraint'})
                    continue
                action='reroute' if target and target!=r['target'] else movement
                choice.update(action=action,applied_action=action,target_stair=target if action=='reroute' else None,
                              reason=f"{choice['priority']} · {target or '관측 공간 없음'} · {movement}")
                mind.commit_private(r,now,choice,o)
                self.set_target(r,target,now,'model_strategy_resolution' if meta['source']!='fallback' else 'private_fallback')
                if r['phase']=='queueing' and movement=='wait':r['agent']['stair_wait_s']+=now-r['arrived_at']
                r['pace']=movement;r['action']=action;r['wait_streak']=r['wait_streak']+1 if movement=='wait' else 0
                if movement=='wait':r.update(phase='waiting',wait_until=now+2)
                elif r['phase']=='waiting':r['phase']='walking'
                if r['phase']=='walking':self.update_navigation(r,now)
                self.decisions.append({**choice,**meta,'time_s':now,'events':o['events'],
                    'event_labels':[EVENT_LABELS[e] for e in o['events']],
                    'observation':o,'policy_versions':self.policy_versions,'policy_hash':POLICY_HASH,
                    'executed_target':r['target'],'executed_goal':list(r['goal']),
                    'execution_source':'personal_knowledge_strategy' if meta['source']!='fallback' else 'private_fallback'})
        print(json.dumps({'time_s':now,'private_decisions':len(batch),'fallback':sum(d['source']=='fallback' for d in self.decisions if d['time_s']==now)},ensure_ascii=False),flush=True)

    def transferred(self,r,area,floor,pos,now,stair_id):
        mind=self.minds[r['agent']['agent_id']];mind.space.traversed(stair_id,area,floor,pos,now)
        mind.pending.add('new_route' if self.stairs[stair_id].get('kind')=='fare_gate' else 'floor_changed');r['target']='explore:none';r['goal_is_portal']=False
        mind.space.last_sense=-1e9

    def visible_helper(self,r,now):
        mind=self.minds[r['agent']['agent_id']]
        visible={aid for aid,o in self.runtime.items() if o is not r and o['area']==r['area'] and o['floor']==r['floor']
                 and o['phase'] not in {'on_stairs','evacuated'} and math.dist(o['pos'],r['pos'])<=6
                 and self.sensor.visible(r['area'],r['floor'],r['pos'],o['pos'],now)}
        ids={o['jid'] for aid,o in self.runtime.items() if o['area']==r['area'] and (aid in visible or o is r)}
        r['local_motion_observation_id']=mind.space.record(now,'local_manoeuvre_view',
            {'floor':r['floor'],'visible_person_ids':sorted(visible),'range_m':6,
             'people':[{'id':aid,'position':[round(x,2) for x in self.runtime[aid]['pos']]} for aid in sorted(visible)]},'vision')
        sim=VisibleSimulation(self.areas[r['area']]['sim'],ids)
        # A conservative star-shaped current view; the helper cannot route behind walls.
        key=(r['agent']['agent_id'],round(now,1),tuple(r['pos']))
        if key not in self.local_geom_cache:
            pts=[]
            angles=[i*math.tau/48 for i in range(48)]
            if r.get('goal'):
                angles.append(math.atan2(r['goal'][1]-r['pos'][1],r['goal'][0]-r['pos'][0])%math.tau)
            for angle in sorted(set(angles)):
                end=r['pos']
                for j in range(1,13):
                    q=(r['pos'][0]+.5*j*math.cos(angle),r['pos'][1]+.5*j*math.sin(angle))
                    if not self.sensor.visible(r['area'],r['floor'],r['pos'],q,now):
                        # Resolve the last visible half-metre interval instead
                        # of throwing it away in short smoke-limited views.
                        low,high=.5*(j-1),.5*j
                        for _ in range(6):
                            distance=(low+high)/2
                            candidate=(r['pos'][0]+distance*math.cos(angle),r['pos'][1]+distance*math.sin(angle))
                            if self.sensor.visible(r['area'],r['floor'],r['pos'],candidate,now):low=distance
                            else:high=distance
                        end=(r['pos'][0]+low*math.cos(angle),r['pos'][1]+low*math.sin(angle))
                        break
                    end=q
                pts.append(end)
            self.local_geom_cache={key:Polygon(pts).buffer(0).intersection(self.areas[r['area']]['geometry'])}
        return sim,self.local_geom_cache[key]

    def finalize(self,now):
        for aid,mind in self.minds.items():
            r=self.runtime[aid];outcome=mind.outcome(r,now)
            blocked=r.get('navigation_blocked')
            cause=None
            if r['phase']!='evacuated':
                if r.get('incapacitated'):cause='toxic_gas_exposure'
                elif r['phase']=='on_stairs':cause='stair_or_landing_blocked'
                elif blocked and mind.space.last_risk['smoke'] in {'dense','opaque'}:cause='smoke_limits_observed_navigation'
                elif blocked and not mind.space.current_origin_links:cause='local_map_discretization_or_contact'
                elif blocked:cause=blocked
                elif now-r.get('stall_since',now)>=6:cause='observed_congestion_or_local_stall'
                else:cause='still_moving_at_time_limit'
            r['agent'].update(experimental_profile=mind.profile,
                familiarity=mind.space.familiarity,initial_knowledge=mind.space.initial_knowledge,
                personal_knowledge=mind.space.snapshot(r['area'],now),observations=mind.space.observations,
                people_observations=mind.space.people_observations,
                known_map_summary={a:{'cells':len(m['cells']),'edges':sum(map(len,m['edges'].values()))} for a,m in mind.space.maps.items()},
                known_map={a:{'initial_source':m.get('initial_source','observation'),'adaptive_grid_m':.5 if m.get('adaptive') else None,
                              'cell_first_seen':[[*k,t] for k,t in m['cell_first_seen'].items()],
                              'edge_first_seen':[[*k,*n,t] for (k,n),t in m['edge_first_seen'].items()]}
                           for a,m in mind.space.maps.items()},
                personal_memory=mind.history+([outcome] if outcome else []),decision_count=mind.count,
                ending_state={'phase':'incapacitated' if r.get('incapacitated') else r['phase'],
                              'physical_phase':r['phase'],'area':r['area'],'floor':r['floor'],'position':list(r['pos']),
                              'target':r['target'],'goal':list(r['goal']),'goal_is_portal':r.get('goal_is_portal'),
                              'entry_lane':r.get('entry_lane'),'navigation_blocked':blocked,
                              'perceived_hazard':mind.space.last_risk},
                decision_limits={'routine_used':mind.routine_count,'critical_used':mind.emergency_count},
                noncompletion_cause=cause)


def source_is_rule(mind):return mind.selected is None
