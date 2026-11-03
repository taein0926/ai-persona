'use strict';
const traitNames={crowd_avoidance:'혼잡 회피',waiting_patience:'대기 인내',risk_avoidance:'위험 회피',exploration:'탐색 적극성',social_following:'타인 추종성',overtaking_tendency:'추월 적극성'};
const traitLevels={low:'낮음',normal:'보통',high:'높음'};
const priorityNames={crowding:'혼잡 회피',patience:'대기 인내',risk:'위험 회피',explore:'탐색',follow:'주변 이동 따르기',time:'아는 경로의 이동 부담',experience:'본인 경험',environment:'현재 관측'};
const smokeNames={clear:'뚜렷한 연기 없음',light:'옅은 연기',dense:'짙은 연기',opaque:'시야를 가리는 연기'};
const heatNames={normal:'평상시 열감',warm:'따뜻한 열감',hot:'강한 열감',unknown:'열감 미관측'};
function perceivedText(h){return h?`${smokeNames[h.smoke]||'연기 미관측'} · ${heatNames[h.heat]||'열감 미관측'}`:'현재 상태 미관측';}
let privatePlayback=null,timeIndexes=new WeakMap();
function resetPrivatePlayback(data){
  privatePlayback={data,people:new WeakMap(),groups:{}};timeIndexes=new WeakMap();
  for(const name of ['decisions','navigation_updates']){
    const groups=new Map();
    for(const row of data[name]||[]){if(!groups.has(row.agent_id))groups.set(row.agent_id,[]);groups.get(row.agent_id).push(row);}
    privatePlayback.groups[name]=groups;
  }
}
function ensurePrivatePlayback(){if(privatePlayback?.data!==state.data)resetPrivatePlayback(state.data);return privatePlayback;}
function timeIndex(rows){
  let index=timeIndexes.get(rows);
  if(!index||index.length!==rows.length){
    index={length:rows.length,ordered:rows.every((r,i)=>Number.isFinite(r.time_s)&&(!i||rows[i-1].time_s<=r.time_s))};
    timeIndexes.set(rows,index);
  }
  return index;
}
function throughTime(rows,time){
  if(!timeIndex(rows).ordered)return rows.filter(r=>r.time_s<=time);
  let lo=0,hi=rows.length;
  while(lo<hi){const mid=(lo+hi)>>1;if(rows[mid].time_s<=time)lo=mid+1;else hi=mid;}
  return rows.slice(0,lo);
}
function privateRows(name,id,time){
  const rows=ensurePrivatePlayback().groups[name].get(id)||[];
  return time===undefined?rows:throughTime(rows,time);
}
function reconcileChildren(parent,children){
  children.forEach((node,i)=>{const old=parent.childNodes[i];if(!old)parent.append(node);else if(!old.isEqualNode(node))parent.replaceChild(node,old);});
  while(parent.childNodes.length>children.length)parent.lastChild.remove();
}
function fitDecisionLog(log){
  if(!log?.closest('details')?.open)return;
  const height=[...log.children].slice(0,5).reduce((sum,row)=>sum+row.getBoundingClientRect().height,0);
  log.style.maxHeight=`${Math.ceil(height)}px`;
}
function personalAt(agent,time,p){
  const ttl=state.data.agent_policy?.observation_policy?.current_observation_ttl_s??3;
  const playback=ensurePrivatePlayback(),rows=agent.observations||[];
  const observations=throughTime(rows,time),ordered=timeIndex(rows).ordered;
  let cached=playback.people.get(agent);
  if(!cached||cached.rows!==rows||cached.length!==rows.length||time<cached.time||!ordered){
    cached={rows,length:rows.length,time:-Infinity,count:0,known:new Map(Object.entries(agent.initial_knowledge||{}).map(([id,k])=>[id,{to_floor:k.to_floor}])),local:undefined};
    playback.people.set(agent,cached);
  }
  const known=cached.known;
  for(let i=cached.count;i<observations.length;i++){const o=observations[i];
    if(o.kind==='landmark_seen'&&!known.has(o.landmark_id))known.set(o.landmark_id,{});
    if(o.kind==='connection_learned')known.set(o.landmark_id,{to_floor:o.to_floor});
    if(o.kind==='local_hazard')cached.local=o;
  }
  cached.count=observations.length;cached.time=time;
  const local=cached.local;
  const view=throughTime(agent.people_observations||[],time).at(-1);
  return {known:new Map(known),local,view,peopleCurrent:view&&p.status!=='on_stairs'&&p.status!=='evacuated'&&view.floor===p.floor&&time-view.time_s<=ttl};
}
function privateTarget(id,agent){
  if(!id||id==='explore:none')return '다음 경로 미확보';
  if(id.startsWith('explore:'))return '관측한 공간의 탐색 경계';
  const local=agent.personal_knowledge?.local_ids?.[id];
  const scenario=state.data.scenario;
  const kind=(scenario.fare_gates||[]).some(gate=>gate.id===id)||scenario.stairs.some(stair=>stair.id===id&&stair.kind==='fare_gate')?'개찰구':'계단';
  const number=/^L(\d+)$/.exec(local||'');
  return number?`${kind} ${number[1]}`:local||`발견한 ${kind}`;
}
function agentTraitOverview(agent){
  const section=el('section',{class:'agent-traits','aria-label':'성향'}),profile=agent.experimental_profile;
  const grid=el('dl',{class:'trait-grid'});
  for(const [name,label] of Object.entries(traitNames)){
    const item=el('div'),level=profile?.traits?.[name]?.level;
    item.append(el('dt',{},label),el('dd',{'data-level':traitLevels[level]?level:'unknown'},traitLevels[level]||'미설정'));grid.append(item);
  }
  section.append(grid);
  return section;
}
function agentClassificationEvidence(agent,opened){
  const evidence=el('details',{'data-section':'classification',class:'classification-evidence'});
  const body=el('div',{class:'classification-evidence-body'});
  evidence.open=opened.has('classification');evidence.append(el('summary',{},'분류 근거'));
  for(const [name,label] of Object.entries(traitNames)){
    body.append(el('strong',{},label),el('p',{},agent.experimental_profile?.traits?.[name]?.assumption||'근거 요약이 없습니다.'));
  }
  for(const [key,label] of [['response_tendency','대피 반응 시간'],['speed_tendency','평균 보행 속도']]){
    body.append(el('strong',{},label),el('p',{},agent.mapping?.reasoning_summary?.[key]||'근거 요약이 없습니다.'));
  }
  const familiarity=agent.mapping?.familiarity||agent.familiarity;
  body.append(el('strong',{},'역사 친숙도'),el('p',{},familiarity?.assumption||'근거 요약이 없습니다.'));
  evidence.append(body);return evidence;
}
function appendSummaryRow(body,label,value,className=''){
  const row=el('tr',{class:className}),headerId=`agent-summary-label-${body.children.length}`;
  row.append(el('th',{id:headerId,colspan:2},label),el('td',{headers:headerId,colspan:2,title:value},value));
  body.append(row);
}
function appendKnowledgeRows(body,agent,p){
  const info=personalAt(agent,state.time,p);
  const stairIds=new Set(state.data.scenario.stairs.filter(stair=>stair.kind!=='fare_gate').map(stair=>stair.id));
  const knownStairs=[...info.known].filter(([id])=>stairIds.has(id));
  const connections=knownStairs.filter(([,k])=>k.to_floor).length;
  appendSummaryRow(body,'아는 계단',`${knownStairs.length}개 · 층 연결 ${connections}개`,'agent-knowledge-row');
  appendSummaryRow(body,'연기·열감',info.local?perceivedText(info.local.perceived):'미관측','agent-knowledge-row');
  appendSummaryRow(body,'관측 시각',info.local?`${info.local.time_s.toFixed(1)}초`:'미관측','agent-knowledge-row');
  appendSummaryRow(body,'주변 사람',info.peopleCurrent?`시야 안 ${info.view.people.length}명`:'미관측','agent-knowledge-row');
}
function drawPrivateInspector(){
  const a=state.data.agents.find(x=>x.agent_id===state.selected);if(!a)return;
  const visual=agentVisual(a),p=visual.p;
  const key=`${state.selected}-${Math.floor(state.time*4)}-${p.status}`;if(state.inspectorKey===key)return;state.inspectorKey=key;
  const currentBox=$('personDetails'),box=document.createElement('div');
  const opened=new Set([...currentBox.querySelectorAll('details[open][data-section]')].map(n=>n.dataset.section));
  const badge=$('selectedMotionStatus');if(badge){badge.textContent=state.hasResults?visual.label:'계산 대기';badge.style.color=visual.color;badge.style.background=visual.color+'18';badge.style.borderColor=visual.color+'55';}
  box.append(agentTraitOverview(a));
  const movement=el('table',{class:'agent-movement-facts','aria-label':'대피 반응 시간과 평균 보행 속도, 역사 친숙도'}),rows=el('tbody');
  const familiarity=a.mapping?.familiarity||a.familiarity;
  const familiarityLabel=a.classification_pending?'계산 대기':{first_visit:'초행',familiar:'구조에 익숙함',limited:'제한적 구조 지식'}[familiarity?.level]||'—';
  const reaction=Number.isFinite(a.pre_movement_s)?`${a.pre_movement_s.toFixed(2)}초`:'계산 대기';
  const speed=Number.isFinite(a.desired_speed_mps)?`${a.desired_speed_mps.toFixed(2)}m/s`:'계산 대기';
  for(const [label,value] of [['대피 반응 시간',reaction],['평균 보행 속도',speed],['역사 친숙도',familiarityLabel]]){
    const row=el('tr');row.append(el('th',{scope:'row'},label),el('td',{},value));rows.append(row);
  }
  movement.append(rows);box.append(movement);
  box.append(agentClassificationEvidence(a,opened));
  const latest=privateRows('decisions',a.agent_id,state.time).at(-1);
  const nav=privateRows('navigation_updates',a.agent_id,state.time).at(-1);
  const summary=el('div',{class:'agent-summary'});
  const location=p.status==='on_stairs'?'층간 계단':p.floor==='OUTSIDE'?'지상':p.floor;
  const target=p.status==='on_stairs'?p.segment?.stair_id:nav&&(!latest||nav.time_s>=latest.time_s)?nav.target:latest?.executed_target;
  const targetText=state.hasResults&&p.status==='incapacitated'?'대피 불능':state.hasResults&&p.status==='evacuated'?'대피 완료':!state.hasResults?'계산 대기':p.status==='pre_moving'?'대피 시작 대기':target?.startsWith('explore:')&&target!=='explore:none'?'관측한 공간 탐색 중':privateTarget(target,a);
  const positionTable=el('table',{class:'agent-position-facts','aria-label':'현재 위치와 목표, 개인이 아는 정보, 최근 판단'}),positionRows=el('tbody'),positionRow=el('tr');
  const columns=el('colgroup');
  for(const width of ['36px','68px','36px',null])columns.append(el('col',width?{style:`width:${width}`} : {}));
  positionTable.append(columns);
  for(const [label,value] of [['위치',location],['목표',targetText]]){
    const headerId=label==='위치'?'agent-location-label':'agent-target-label';
    positionRow.append(el('th',{id:headerId},label),el('td',{headers:headerId,title:value},value));
  }
  positionRows.append(positionRow);
  appendKnowledgeRows(positionRows,a,p);
  const exposure=gasExposureAt(a,state.time);
  const exposureText=!state.hasResults?'계산 후 표시됩니다.':exposure?`FED ${exposure.fed.toFixed(3)} / 대피 불능 기준 1.0`:a.gas_exposure_history?.length?'노출 계산 전':'기록 없음';
  appendSummaryRow(positionRows,'가스 노출',exposureText,'agent-gas-row');
  let decisionText=state.hasResults?'아직 판단 기록 없음':'계산 후 표시됩니다.';
  if(latest){
    const raw=latest.raw_model_choice||{},priority=(raw.priority||latest.priority||'').split(':')[0];
    decisionText=`${latest.time_s.toFixed(1)}초 · ${priorityNames[priority]||'주변 상황 확인'} · ${actionLabels[raw.movement||latest.requested_movement]||'계속 이동'}`;
  }
  appendSummaryRow(positionRows,'최근 판단',decisionText,'agent-decision-row');
  positionTable.append(positionRows);summary.append(positionTable);
  if(state.hasResults&&p.status==='evacuated'){
    const completed=el('section',{class:'agent-completion'});
    completed.append(el('strong',{},'대피 완료'),el('p',{},`${a.evacuation_time_s.toFixed(1)}초 · ${state.data.scenario.exit_labels?.[a.exit_id]||a.exit_id||'지상'}`));
    summary.append(completed);
  }
  if(state.hasResults&&p.status==='incapacitated'){
    const stopped=el('section',{class:'agent-completion'});
    stopped.append(el('strong',{},'가스 노출로 대피 불능'),el('p',{},`${a.incapacitated_at_s.toFixed(1)}초 · FED 1.0 이상`));
    summary.append(stopped);
  }
  box.append(summary);
  const background=el('details',{'data-section':'background',class:'persona-background'});background.open=opened.has('background');background.append(el('summary',{},'페르소나 정보'));
  if(a.persona.persona)background.append(el('p',{},a.persona.persona));
  const fields={sex:'성별',province:'시·도',district:'시·군·구',education_level:'학력',bachelors_field:'전공',marital_status:'혼인 상태',family_type:'가족 유형',housing_type:'주거 유형',military_status:'병역',professional_persona:'직업·경력',career_goals_and_ambitions:'진로 목표',family_persona:'가족·생활',hobbies_and_interests:'취미·관심사',skills_and_expertise:'기술·전문성',cultural_background:'문화적 배경',arts_persona:'예술',culinary_persona:'요리',sports_persona:'스포츠',travel_persona:'여행',uuid:'원본 식별자'};
  for(const [key,title] of Object.entries(fields)){const value=a.persona[key];if(value!=null&&value!==''){const field=el('div');field.append(el('small',{},title),el('p',{},Array.isArray(value)?value.join(', '):String(value)));background.append(field);}}
  const previousLog=$('decisionLog'),decisions=privateRows('decisions',a.agent_id,state.time);
  const sameAgent=previousLog?.dataset.agentId===String(a.agent_id);
  const scrollTop=sameAgent?previousLog.scrollTop:0;
  const followLatest=state.playing&&sameAgent&&decisions.length>previousLog.childElementCount;
  const history=el('details',{'data-section':'history',class:'assumptions'}),log=el('div',{id:'decisionLog','data-agent-id':a.agent_id});
  log.style.maxHeight=previousLog?.style.maxHeight||'';
  history.open=opened.has('history');history.append(el('summary',{},'판단 기록'),log);
  history.addEventListener('toggle',()=>{if(history.isConnected)fitDecisionLog(log);});
  log.append(...decisions.map(d=>{
    const row=el('button',{type:'button',class:'decision-row','data-decision-time':d.time_s,title:`${d.time_s.toFixed(1)}초로 이동`});
    row.append(el('b',{},`${d.time_s.toFixed(1)}초 · ${priorityNames[d.priority]||d.priority}`),el('span',{},`${actionLabels[d.requested_movement]} → ${privateTarget(d.executed_target,a)}`));
    row.addEventListener('click',()=>{
      seekToTime(d.time_s);
      $('decisionLog').querySelector(`[data-decision-time="${d.time_s}"]`)?.focus({preventScroll:true});
    });
    return row;
  }));
  box.append(history,background);
  reconcileChildren(currentBox,[...box.childNodes]);
  const currentLog=$('decisionLog');
  fitDecisionLog(currentLog);
  currentLog.scrollTop=followLatest?currentLog.scrollHeight:scrollTop;
}
