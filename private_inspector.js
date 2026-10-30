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
function personalAt(agent,time,p){
  const ttl=state.data.agent_policy?.observation_policy?.current_observation_ttl_s??3;
  const playback=ensurePrivatePlayback(),rows=agent.observations||[];
  const observations=throughTime(rows,time),ordered=timeIndex(rows).ordered;
  let cached=playback.people.get(agent);
  if(!cached||cached.rows!==rows||cached.length!==rows.length||time<cached.time||!ordered){
    cached={rows,length:rows.length,time:-Infinity,count:0,known:new Map(Object.entries(agent.initial_knowledge||{}).map(([id,k])=>[id,{...k,knowledge_source:k.source||'residential_prior'}])),local:undefined};
    playback.people.set(agent,cached);
  }
  const known=cached.known;
  for(let i=cached.count;i<observations.length;i++){const o=observations[i];
    if(o.kind==='landmark_seen')known.set(o.landmark_id,{...(known.get(o.landmark_id)||{}),id:o.landmark_id,floor:o.floor,observed_at_s:o.time_s,current_state:o.perceived_state,observation_id:o.id,knowledge_source:known.get(o.landmark_id)?.knowledge_source||'vision'});
    if(o.kind==='connection_learned')known.set(o.landmark_id,{...(known.get(o.landmark_id)||{}),id:o.landmark_id,to_floor:o.to_floor,connection_source:'own_traversal',connection_time_s:o.time_s});
    if(o.kind==='local_hazard')cached.local=o;
  }
  cached.count=observations.length;cached.time=time;
  const local=cached.local;
  for(const k of known.values())k.current=p.status!=='on_stairs'&&p.status!=='evacuated'&&k.floor===p.floor&&k.observed_at_s===local?.time_s&&time-k.observed_at_s<=ttl;
  const view=throughTime(agent.people_observations||[],time).at(-1);
  return {known:new Map([...known].map(([id,k])=>[id,{...k}])),observations,local,view,peopleCurrent:view&&p.status!=='on_stairs'&&p.status!=='evacuated'&&view.floor===p.floor&&time-view.time_s<=ttl};
}
function privateTarget(id,agent){
  if(!id||id==='explore:none')return '다음 경로 미확보';
  if(id.startsWith('explore:'))return '관측한 공간의 탐색 경계';
  return `${agent.personal_knowledge?.local_ids?.[id]||'발견한 계단'}`;
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
  for(const [key,label] of [['response_tendency','대피 반응 시간'],['speed_tendency','보행 속도']]){
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
function appendKnowledgeRows(body,agent,p,opened){
  const info=personalAt(agent,state.time,p);
  const connections=[...info.known.values()].filter(k=>k.to_floor).length;
  appendSummaryRow(body,'아는 계단',`${info.known.size}개 · 층 연결 ${connections}개`,'agent-knowledge-row');
  appendSummaryRow(body,'연기·열감',info.local?perceivedText(info.local.perceived):'미관측','agent-knowledge-row');
  appendSummaryRow(body,'관측 시각',info.local?`${info.local.time_s.toFixed(1)}초`:'미관측','agent-knowledge-row');
  appendSummaryRow(body,'주변 사람',info.peopleCurrent?`시야 안 ${info.view.people.length}명`:'미관측','agent-knowledge-row');
  if(info.known.size){
    const detail=el('details',{'data-section':'knowledge',class:'knowledge-landmarks'});
    detail.open=opened.has('knowledge');detail.append(el('summary',{},'계단 기억 보기'));
    const list=el('div');
    for(const [id,k] of info.known){
      const source=k.connection_source==='own_traversal'?'직접 이동해 연결 확인':k.knowledge_source==='familiarity_prior'?'사전 구조 기억':k.knowledge_source==='residential_prior'?'과거 거주 기반 기억':'직접 발견';
      const when=k.observed_at_s==null?'현재 상태 미관측':`${k.observed_at_s.toFixed(1)}초 관측 · ${k.current?'현재 시야':'과거 관측'}`;
      list.append(el('p',{},`${privateTarget(id,agent)} · ${k.floor||''} → ${k.to_floor||'연결층 미확인'} · ${source} · ${when}`));
      if(k.current_state)list.append(el('p',{},`${k.current?'진입부 관측':'지난 진입부 관측'} · ${perceivedText(k.current_state.hazard)} · 주변 ${k.current_state.people}명`));
    }
    detail.append(list);
    const row=el('tr',{class:'agent-knowledge-detail'}),cell=el('td',{colspan:4});
    cell.append(detail);row.append(cell);body.append(row);
  }
}
function drawPrivateInspector(){
  const a=state.data.agents.find(x=>x.agent_id===state.selected);if(!a)return;
  const key=`${state.selected}-${Math.floor(state.time*4)}`;if(state.inspectorKey===key)return;state.inspectorKey=key;
  const visual=agentVisual(a),p=visual.p,currentBox=$('personDetails'),box=document.createElement('div');
  const opened=new Set([...currentBox.querySelectorAll('details[open][data-section]')].map(n=>n.dataset.section));
  const badge=$('selectedMotionStatus');if(badge){badge.textContent=state.hasResults?visual.label:'계산 대기';badge.style.color=visual.color;badge.style.background=visual.color+'18';badge.style.borderColor=visual.color+'55';}
  box.append(agentTraitOverview(a));
  const movement=el('table',{class:'agent-movement-facts','aria-label':'대피 반응 시간과 보행 속도, 역사 친숙도'}),rows=el('tbody');
  const familiarity=a.mapping?.familiarity||a.familiarity;
  const familiarityLabel=a.classification_pending?'계산 대기':{first_visit:'초행',familiar:'구조에 익숙함',limited:'제한적 구조 지식'}[familiarity?.level]||'—';
  const reaction=Number.isFinite(a.pre_movement_s)?`${a.pre_movement_s.toFixed(2)}초`:'계산 대기';
  const speed=Number.isFinite(a.desired_speed_mps)?`${a.desired_speed_mps.toFixed(2)}m/s`:'계산 대기';
  for(const [label,value] of [['대피 반응 시간',reaction],['보행 속도',speed],['역사 친숙도',familiarityLabel]]){
    const row=el('tr');row.append(el('th',{scope:'row'},label),el('td',{},value));rows.append(row);
  }
  movement.append(rows);box.append(movement);
  box.append(agentClassificationEvidence(a,opened));
  const latest=privateRows('decisions',a.agent_id,state.time).at(-1);
  const nav=privateRows('navigation_updates',a.agent_id,state.time).at(-1);
  const summary=el('div',{class:'agent-summary'});
  const location=p.status==='on_stairs'?'층간 계단':p.floor==='OUTSIDE'?'지상':p.floor;
  const target=p.status==='on_stairs'?p.segment?.stair_id:nav&&(!latest||nav.time_s>=latest.time_s)?nav.target:latest?.executed_target;
  const targetText=state.hasResults&&p.status==='evacuated'?'대피 완료':!state.hasResults?'계산 대기':p.status==='pre_moving'?'대피 시작 대기':target?.startsWith('explore:')&&target!=='explore:none'?'관측한 공간 탐색 중':privateTarget(target,a);
  const positionTable=el('table',{class:'agent-position-facts','aria-label':'현재 위치와 목표, 개인이 아는 정보, 최근 판단'}),positionRows=el('tbody'),positionRow=el('tr');
  const columns=el('colgroup');
  for(const width of ['36px','36px','36px',null])columns.append(el('col',width?{style:`width:${width}`} : {}));
  positionTable.append(columns);
  for(const [label,value] of [['위치',location],['목표',targetText]]){
    const headerId=label==='위치'?'agent-location-label':'agent-target-label';
    positionRow.append(el('th',{id:headerId},label),el('td',{headers:headerId,title:value},value));
  }
  positionRows.append(positionRow);
  appendKnowledgeRows(positionRows,a,p,opened);
  if(!(state.hasResults&&p.status==='evacuated')){
    let decisionText=state.hasResults?'아직 판단 기록 없음':'계산 후 표시됩니다.';
    if(latest){
      const raw=latest.raw_model_choice||{},priority=(raw.priority||latest.priority||'').split(':')[0];
      decisionText=`${latest.time_s.toFixed(1)}초 · ${priorityNames[priority]||'주변 상황 확인'} · ${actionLabels[raw.movement||latest.requested_movement]||'계속 이동'}`;
    }
    appendSummaryRow(positionRows,'최근 판단',decisionText,'agent-decision-row');
  }
  positionTable.append(positionRows);summary.append(positionTable);
  if(state.hasResults&&p.status==='evacuated'){
    const completed=el('section',{class:'agent-completion'});
    completed.append(el('strong',{},'대피 완료'),el('p',{},`${a.evacuation_time_s.toFixed(1)}초 · ${state.data.scenario.exit_labels?.[a.exit_id]||a.exit_id||'지상'}`));
    summary.append(completed);
  }
  box.append(summary);
  const background=el('details',{'data-section':'background',class:'persona-background'});background.open=opened.has('background');background.append(el('summary',{},'페르소나 정보'));
  if(a.persona.persona)background.append(el('p',{},a.persona.persona));
  const fields={sex:'성별',province:'시·도',district:'시·군·구',education_level:'학력',bachelors_field:'전공',marital_status:'혼인 상태',family_type:'가족 유형',housing_type:'주거 유형',military_status:'병역',professional_persona:'직업·경력',career_goals_and_ambitions:'진로 목표',family_persona:'가족·생활',hobbies_and_interests:'취미·관심사',skills_and_expertise:'기술·전문성',cultural_background:'문화적 배경',arts_persona:'예술',culinary_persona:'요리',sports_persona:'스포츠',travel_persona:'여행',uuid:'원본 식별자'};
  for(const [key,title] of Object.entries(fields)){const value=a.persona[key];if(value!=null&&value!==''){const field=el('div');field.append(el('small',{},title),el('p',{},Array.isArray(value)?value.join(', '):String(value)));background.append(field);}}
  box.append(background);
  const history=el('details',{'data-section':'history',class:'assumptions'}),log=el('div',{id:'decisionLog'});
  history.open=opened.has('history');history.append(el('summary',{},'판단 기록'),log);
  box.append(history);
  reconcileChildren(currentBox,[...box.childNodes]);
  reconcileChildren($('decisionLog'),privateRows('decisions',a.agent_id,state.time).map(d=>{const row=el('div',{class:'decision-row'});row.append(el('b',{},`${d.time_s.toFixed(1)}초 · ${priorityNames[d.priority]||d.priority}`),el('span',{},`${actionLabels[d.requested_movement]} → ${privateTarget(d.executed_target,a)}`));return row;}));
}
