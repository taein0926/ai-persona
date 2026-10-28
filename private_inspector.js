'use strict';
const traitNames={crowd_avoidance:'혼잡 회피',waiting_patience:'대기 인내',risk_avoidance:'위험 회피',exploration:'탐색 적극성'};
const traitLevels={low:'낮음',normal:'보통',high:'높음'};
const priorityNames={crowding:'혼잡 회피',patience:'대기 인내',risk:'위험 회피',explore:'탐색',time:'아는 경로의 이동 부담',experience:'본인 경험',environment:'현재 관측'};
const smokeNames={clear:'뚜렷한 연기 없음',light:'옅은 연기',dense:'짙은 연기',opaque:'시야를 가리는 연기'};
const heatNames={normal:'평상시 열감',warm:'따뜻한 열감',hot:'강한 열감',unknown:'열감 미관측'};
const causeNames={smoke_limits_observed_navigation:'연기와 관측 범위 안에서 다음 경로를 확보하지 못함',observed_congestion_or_local_stall:'국소 혼잡·이동 정체',still_moving_at_time_limit:'계산 종료 시점에도 이동 중',stair_or_landing_blocked:'계단 또는 도착부에서 정체',local_map_discretization_or_contact:'관측 격자 또는 벽 접촉으로 다음 경로 미확보',no_observed_reachable_frontier_or_portal:'아는 공간에서 접근 가능한 다음 목표 없음',known_path_not_currently_visible:'기억한 경로가 현재 보이지 않음'};
const navSources={personal_rule_after_observation:'새 관측에 따른 기본 탐색',navigation_for_last_model_strategy:'직전 Qwen 전략을 개인 지도에 적용',private_rule_after_blocked_experience:'본인의 막힌 경험에 따른 기본 재탐색',model_strategy_resolution:'Qwen 전략의 목표 변환',private_fallback:'응답 실패 후 개인 지식으로 선택'};
function perceivedText(h){return h?`${smokeNames[h.smoke]||'연기 미관측'} · ${heatNames[h.heat]||'열감 미관측'}`:'현재 상태 미관측';}
function personalAt(agent,time,p){
  const ttl=state.data.agent_policy?.observation_policy?.current_observation_ttl_s??3;
  const known=new Map(Object.entries(agent.initial_knowledge||{}).map(([id,k])=>[id,{...k,knowledge_source:'residential_prior'}]));
  const observations=(agent.observations||[]).filter(o=>o.time_s<=time);
  for(const o of observations){
    if(o.kind==='landmark_seen')known.set(o.landmark_id,{...(known.get(o.landmark_id)||{}),id:o.landmark_id,floor:o.floor,observed_at_s:o.time_s,current_state:o.perceived_state,observation_id:o.id,knowledge_source:known.get(o.landmark_id)?.knowledge_source||'vision'});
    if(o.kind==='connection_learned')known.set(o.landmark_id,{...(known.get(o.landmark_id)||{}),id:o.landmark_id,to_floor:o.to_floor,connection_source:'own_traversal',connection_time_s:o.time_s});
  }
  const local=observations.filter(o=>o.kind==='local_hazard').at(-1);
  for(const k of known.values())k.current=p.status!=='on_stairs'&&p.status!=='evacuated'&&k.floor===p.floor&&k.observed_at_s===local?.time_s&&time-k.observed_at_s<=ttl;
  const view=(agent.people_observations||[]).filter(o=>o.time_s<=time).at(-1);
  return {known,observations,local,view,peopleCurrent:view&&p.status!=='on_stairs'&&p.status!=='evacuated'&&view.floor===p.floor&&time-view.time_s<=ttl};
}
function privateTarget(id,agent){
  if(!id||id==='explore:none')return '다음 경로 미확보';
  if(id.startsWith('explore:'))return '관측한 공간의 탐색 경계';
  return `${agent.personal_knowledge?.local_ids?.[id]||'발견한 계단'}`;
}
function drawPrivateInspector(){
  const a=state.data.agents.find(x=>x.agent_id===state.selected);if(!a)return;
  const key=`${state.selected}-${Math.floor(state.time*4)}`;if(state.inspectorKey===key)return;state.inspectorKey=key;
  const visual=agentVisual(a),p=visual.p,box=$('personDetails');
  const opened=new Set([...box.querySelectorAll('details[open][data-section]')].map(n=>n.dataset.section));
  box.replaceChildren();
  const badge=$('selectedMotionStatus');if(badge){badge.textContent=visual.label;badge.style.color=visual.color;badge.style.background=visual.color+'18';badge.style.borderColor=visual.color+'55';}
  const facts=el('div',{class:'facts'});
  fact(facts,'현재 상태',visual.label);fact(facts,'위치',p.status==='on_stairs'?'층간 계단':p.floor==='OUTSIDE'?'지상':p.floor);
  fact(facts,'대피 시작 지연',`${a.pre_movement_s.toFixed(2)}초`);fact(facts,'기본 목표 보행 속도',`${a.desired_speed_mps.toFixed(2)}m/s`);
  const waited=state.data.passages.filter(r=>r.agent_id===a.agent_id).reduce((s,r)=>s+Math.max(0,Math.min(state.time,r.entry)-r.arrival),0);
  fact(facts,'지금까지 계단 입구 대기',`${waited.toFixed(1)}초`);
  fact(facts,'계산 종료 시 결과',a.evacuation_time_s==null?`${state.data.summary.total_time_s.toFixed(0)}초까지 미완료`:`${a.evacuation_time_s.toFixed(1)}초 · ${state.data.scenario.exit_labels?.[a.exit_id]||a.exit_id}`);
  box.append(facts);
  const reason=el('div',{class:'reason'}),profile=a.experimental_profile;
  reason.append(el('strong',{},'실험 성향 · 측정된 성격이 아닌 가정'));
  if(profile)reason.append(el('small',{},`분류 방식 · ${(profile.classification_method||state.data.agent_policy?.classification_method)==='full_persona'?'페르소나 전체 해석':'키워드 후보 해석·시드 배정'} · ${profile.version}`));
  if(profile)for(const [name,value] of Object.entries(profile.traits)){
    const evidence={direct:'직접 관련 서술',contextual:'전체 문맥의 간접 추정',hypothesis:'정보 부족 가설'}[value.evidence_level];
    reason.append(el('p',{},`${traitNames[name]} · ${traitLevels[value.level]} · ${evidence||(value.origin==='qwen_interpretation'?'Qwen 원문 해석':'고정 시드 배정')}`));
    reason.append(el('small',{},value.assumption));
    if(value.source_fields?.length)reason.append(el('small',{},`사용한 원문 필드 · ${value.source_fields.join(', ')}`));
    for(const id of value.evidence_ids){const q=profile.excerpts.find(q=>q.id===id);if(q)reason.append(el('p',{},`원문 ${id} · ${q.field} · “${q.text}”`));}
    if(value.information_insufficient)reason.append(el('small',{},'관련 정보 부족 · 원문 인용 없음'));
    if(value.origin==='seeded_assignment')reason.append(el('small',{},'낮음 25% · 보통 50% · 높음 25% · 원문 인용 없음'));
  }
  reason.append(el('small',{},'추월 의도: 낮은 대기 인내·보통 이상 탐색과 상황 판단. 실행에는 실제 속도 차이·빈 공간·충돌 조건이 필요합니다.'));
  const info=personalAt(a,state.time,p);
  reason.append(el('strong',{},'초기 친숙도'));
  reason.append(el('p',{},a.familiarity?.level==='limited'?'제한적인 과거 구조 지식':'초행 · 사전 구조 지식 없음'));
  reason.append(el('small',{},a.familiarity?.assumption||''));
  for(const e of a.familiarity?.evidence||[])reason.append(el('p',{},`거주 근거 · ${e.text}`));
  if(Object.keys(a.initial_knowledge||{}).length)reason.append(el('small',{},'사전 지식은 과거의 층 연결이며 현재 혼잡·화재·통행 가능성을 알려주지 않습니다.'));
  reason.append(el('strong',{},'개인이 아는 정보 · 재생 시점 기준'));
  reason.append(el('p',{},`발견·기억한 계단 ${info.known.size}개 · 연결층을 아는 계단 ${[...info.known.values()].filter(k=>k.to_floor).length}개`));
  for(const [id,k] of info.known){
    const source=k.connection_source==='own_traversal'?'직접 이동해 연결 확인':k.knowledge_source==='residential_prior'?'과거 거주 기반 구조 지식':'직접 발견';
    const when=k.observed_at_s==null?'현재 상태 미관측':`${k.observed_at_s.toFixed(1)}초 관측 · ${k.current?'현재 시야':'과거 관측'}`;
    reason.append(el('p',{},`${privateTarget(id,a)} · ${k.floor||''} → ${k.to_floor||'연결층 미확인'} · ${source} · ${when}`));
    if(k.current_state)reason.append(el('small',{},`${k.current?'진입부 관측':'지난 진입부 관측'} · ${perceivedText(k.current_state.hazard)} · 주변 ${k.current_state.people}명`));
  }
  if(info.local)reason.append(el('p',{},`${info.local.time_s.toFixed(1)}초 ${info.local.source==='stair_sensation'?'계단에서 체감':'주변에서 지각'} · ${perceivedText(info.local.perceived)}`));
  reason.append(el('small',{},info.peopleCurrent?`${info.view.time_s.toFixed(1)}초 본 사람 ${info.view.people.length}명 · 목적지·기억·계획은 제공받지 않음`:'현재 시야의 사람 수는 미관측'));
  const latest=(state.data.decisions||[]).filter(d=>d.agent_id===a.agent_id&&d.time_s<=state.time).at(-1);
  if(latest){
    const raw=latest.raw_model_choice||{},priority=(raw.priority||latest.priority||'').split(':')[0];
    reason.append(el('strong',{},`${latest.time_s.toFixed(1)}초 ${latest.source==='fallback'?'기본 행동':'Qwen 선택'}`));
    reason.append(el('small',{},(latest.event_labels||[]).join(' · ')));
    reason.append(el('p',{},`전략 · ${priorityNames[priority]||priority} / 이동 의도 · ${actionLabels[raw.movement||latest.requested_movement]||'계속 이동'}`));
    reason.append(el('p',{},`개인 지도 계산의 실행 목표 · ${privateTarget(latest.executed_target,a)}`));
    if(latest.execution_constraint)reason.append(el('small',{},latest.execution_constraint));
    reason.append(el('small',{},`선택한 관측 ${(latest.used_observation_ids||[]).join(', ')||'없음'} · 선택한 기억 ${(latest.used_memory_ids||[]).join(', ')||'없음'}`));
    for(const id of latest.used_evidence_ids||[]){const q=latest.observation?.profile?.excerpts.find(q=>q.id===id);if(q)reason.append(el('p',{},`선택 전략에 연결된 원문 ${id} · “${q.text}”`));}
    reason.append(el('strong',{},'판단에 제공한 본인 관측'));
    for(const o of latest.observation?.observations||[]){
      const kind={local_hazard:'본인 주변의 연기·열감',landmark_seen:'본인이 본 계단 진입부',space_seen:'새 공간 발견',connection_learned:'직접 이동한 층 연결',implemented_sign_seen:'실제로 배치된 안내 표지'}[o.kind]||'본인 관측';
      reason.append(el('p',{},`${o.id} · ${o.time_s.toFixed(1)}초 · ${kind}${latest.used_observation_ids?.includes(o.id)?' · 이번 판단에 사용':''}`));
    }
    const memories=latest.observation?.memory||[];
    if(memories.length){reason.append(el('strong',{},'본인이 겪은 최근 경험 · 판단 당시'));
      for(const m of memories){const outcome={reached_next_floor:'다음 층 도착',evacuated:'대피 완료',still_blocked:'정체 지속',advanced:'이동 진행',little_progress:'이동량 적음'}[m.result]||m.result;
        reason.append(el('p',{},`${m.id} · ${m.time_s.toFixed(1)}초 이후 ${m.elapsed_s.toFixed(1)}초 · 평면 ${m.walked_m.toFixed(1)}m · 대기 ${m.waited_s.toFixed(1)}초 · ${outcome}${latest.used_memory_ids?.includes(m.id)?' · 이번 판단에 사용':''}`));}
    }
    reason.append(el('small',{},latest.source==='ollama_live'?'실제 로컬 Ollama 추론':latest.source==='ollama_cache'?'저장된 Qwen 응답 사용':'응답 실패 후 개인 지식으로 기본 행동'));
  }
  const nav=(state.data.navigation_updates||[]).filter(n=>n.agent_id===a.agent_id&&n.time_s<=state.time).at(-1);
  if(nav&&p.status!=='evacuated')reason.append(el('p',{},`현재 목표 · ${privateTarget(nav.target,a)} · ${navSources[nav.source]||'개인 지식에 따른 이동'}`));
  if(state.time>=state.data.summary.total_time_s&&a.noncompletion_cause)reason.append(el('p',{},`미완료 원인 · ${causeNames[a.noncompletion_cause]||a.noncompletion_cause}`));
  box.append(reason);
  const hazard=(state.hazardByAgent?.get(a.agent_id)||[]).filter(h=>h.time_s<=state.time&&h.floor===p.floor&&(p.status!=='on_stairs'||h.stair_id===p.segment?.stair_id)).at(-1);
  if(hazard&&p.status!=='evacuated'){
    const detail=el('details',{'data-section':'fds',class:'persona-background'});detail.open=opened.has('fds');detail.append(el('summary',{},'연구자용 FDS 수치'));
    detail.append(el('small',{},'정확한 수치는 물리 계산·표시용입니다. 에이전트에게는 연기·열감 단계만 전달합니다.'));
    const values=el('div',{class:'facts'});fact(values,'FDS 연기',`K ${hazard.extinction_per_m.toFixed(3)} /m`);fact(values,'가시거리 · 상한 100m',`${hazard.visibility_m_capped_100.toFixed(1)}m`);
    fact(values,hazard.temperature_sampling==='connected_floor_interpolation'?'온도 · 연결층 관측값 보간':'FDS 관측 온도',`${hazard.temperature_c.toFixed(1)}°C`);
    fact(values,hazard.speed_factor_applied===false?'연기 계수 · 참고':'연기 목표속도 계수',`${hazard.speed_factor.toFixed(3)}배`);detail.append(values);box.append(detail);
  }
  if(a.mapping){
    const detail=el('details',{'data-section':'baseline',class:'persona-background'});detail.open=opened.has('baseline');detail.append(el('summary',{},'시작 지연·기본 속도 근거'));
    for(const [key,title] of [['response_tendency','반응'],['speed_tendency','이동']]){detail.append(el('p',{},`${title} · ${a.mapping.reasoning_summary?.[key]||''}`));for(const q of a.mapping.evidence?.[key]?.quotes||[])detail.append(el('p',{},`원문 · “${q.text}”`));}
    if(a.mapping.overtaking)detail.append(el('p',{},`보존된 원문상 급함 기록 · ${a.mapping.overtaking.reason}`));box.append(detail);
  }
  const background=el('details',{'data-section':'background',class:'persona-background'});background.open=opened.has('background');background.append(el('summary',{},'페르소나 배경 · 원본 정보'));
  const fields={sex:'성별',province:'시·도',district:'시·군·구',education_level:'학력',bachelors_field:'전공',marital_status:'혼인 상태',family_type:'가족 유형',housing_type:'주거 유형',military_status:'병역',professional_persona:'직업·경력',career_goals_and_ambitions:'진로 목표',family_persona:'가족·생활',hobbies_and_interests:'취미·관심사',skills_and_expertise:'기술·전문성',cultural_background:'문화적 배경',arts_persona:'예술',culinary_persona:'요리',sports_persona:'스포츠',travel_persona:'여행',uuid:'원본 식별자'};
  for(const [key,title] of Object.entries(fields)){const value=a.persona[key];if(value!=null&&value!==''){const field=el('div');field.append(el('small',{},title),el('p',{},Array.isArray(value)?value.join(', '):String(value)));background.append(field);}}
  box.append(background);
  $('decisionLog').replaceChildren(...(state.data.decisions||[]).filter(d=>d.agent_id===a.agent_id&&d.time_s<=state.time).map(d=>{const row=el('div',{class:'decision-row'});row.append(el('b',{},`${d.time_s.toFixed(1)}초 · ${priorityNames[d.priority]||d.priority}`),el('span',{},`${actionLabels[d.requested_movement]} → ${privateTarget(d.executed_target,a)}`));return row;}));
}
