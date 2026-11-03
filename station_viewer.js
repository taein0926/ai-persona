'use strict';
const $ = id => document.getElementById(id);
const ns = 'http://www.w3.org/2000/svg';
const state = {data: null, hasResults: false, time: 0, playing: false, selected: null, frame: null, last: null, dirty: false};
const labels = {pre_moving: '대피 전 대기', walking: '대피중', waiting: '판단 후 대기', queueing: '통로 진입 대기', on_stairs: '계단 이동', on_gate:'개찰구 통과', evacuated: '지상 도착', incapacitated:'가스 노출로 대피 불능',visibility_blocked:'시야 확보 불가'};
const liveStatsLabels = {pre_moving:'대피 전',evacuating:'대피중',evacuated:'대피 완료',incapacitated:'대피 불능'};
const colors = {pre_moving: '#e6d68d', walking: '#74e0bb', waiting: '#efb35d', queueing: '#efb35d', on_stairs: '#91baff', on_gate: '#74e0bb', evacuated: '#6c9691', incapacitated:'#b91c1c',visibility_blocked:'#ff4e88'};
const actionLabels = {continue: '계속 이동', hurry: '서두르기', wait: '2초 기다리기', reroute: '다른 계단으로 우회', overtaking:'추월 시도',overtake:'추월 시도'};
function el(tag, attrs = {}, text = '') {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  node.textContent = text;
  return node;
}
function svg(tag, attrs = {}, text = '', parent = $('map')) {
  const node = document.createElementNS(ns, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  node.textContent = text;
  parent.append(node);
  return node;
}
// Screen height follows physical floor order. B1 is mirrored so street stairs face upward.
function xy(p){return stationXY(p);}
let playbackFrame=null;
function frameEntry(agent,time) {
  if(time!==state.time)return null;
  if(!playbackFrame||playbackFrame.data!==state.data||playbackFrame.time!==time)
    playbackFrame={data:state.data,time,agents:new WeakMap()};
  let value=playbackFrame.agents.get(agent);
  if(!value){value={};playbackFrame.agents.set(agent,value);}
  return value;
}
function position(agent,time) {
  const entry=frameEntry(agent,time);
  if(!entry)return calculatePosition(agent,time);
  return entry.position||(entry.position=calculatePosition(agent,time));
}
function calculatePosition(agent, time) {
  const p=calculatePhysicalPosition(agent,time);
  if(agent.incapacitated_at_s!=null&&time>=agent.incapacitated_at_s)
    return {...p,physical_status:p.status,status:'incapacitated'};
  if(p.status==='walking'&&visibilityBlockedAt(agent,time))
    return {...p,physical_status:p.status,status:'visibility_blocked'};
  return p;
}
function visibilityBlockedAt(agent,time) {
  const history=agent.visibility_blocked_history||[];
  let lo=0,hi=history.length;
  while(lo<hi){const mid=(lo+hi)>>1;if(history[mid].time_s<=time)lo=mid+1;else hi=mid;}
  return lo>0&&history[lo-1].blocked;
}
function smokeObscuredAt(agent,time,p) {
  const history=state.hazardByAgent?.get(agent.agent_id)||[];
  let lo=0,hi=history.length;
  while(lo<hi){const mid=(lo+hi)>>1;if(history[mid].time_s<=time)lo=mid+1;else hi=mid;}
  const sample=history[lo-1];
  if(!sample||time-sample.time_s>1.05)return false;
  const physicalStatus=p.physical_status||p.status;
  const sameLocation=physicalStatus==='on_stairs'
    ?sample.stair_id===p.segment?.stair_id
    :sample.floor===p.floor&&!sample.stair_id;
  // 기존 연기 관측의 opaque 기준을 사용하며, 이동 속도와는 무관하다.
  return sameLocation&&sample.extinction_per_m>=1;
}
function gasExposureAt(agent,time) {
  const history=agent.gas_exposure_history||[];
  let lo=0,hi=history.length;
  while(lo<hi){const mid=(lo+hi)>>1;if(history[mid].time_s<=time)lo=mid+1;else hi=mid;}
  return lo?history[lo-1]:null;
}
function calculatePhysicalPosition(agent, time) {
  const final = agent.segments.at(-1);
  if (agent.evacuation_time_s!=null&&time >= agent.evacuation_time_s) return {status: 'evacuated', floor: 'OUTSIDE', pos: xy(final.to), segment: null};
  let lo = 0, hi = agent.segments.length - 1, s;
  while (lo <= hi) {const mid = (lo + hi) >> 1, candidate = agent.segments[mid];
    if (time < candidate.start) hi = mid - 1;
    else if (time >= candidate.end) lo = mid + 1;
    else {s = candidate; break;}}
  if(!s&&final&&time>=final.end)s=final;
  if (!s) return {status: 'pre_moving', floor: 'B2', pos: xy(agent.spawn), segment: null};
  const duration = s.status === 'on_stairs' ? state.data.scenario.stairs.find(stair => stair.id === s.stair_id).travel_time_s : s.end - s.start;
  const a = xy(s.from), b = xy(s.to); let p = Math.min(1, (time - s.start) / duration);
  if(s.stair_motion){let l=0,h=s.stair_motion.length-1;while(l<h){const m=Math.ceil((l+h)/2);if(s.stair_motion[m][0]<=time)l=m;else h=m-1;}const u=s.stair_motion[l],v=s.stair_motion[Math.min(l+1,s.stair_motion.length-1)];p=u[1]+(v[1]-u[1])*(v[0]>u[0]?(time-u[0])/(v[0]-u[0]):0);}
  return {status: s.status, floor: s.status === 'on_stairs' ? '층간 연결' : s.from.floor,
    pos: {x: a.x + (b.x - a.x) * p, y: a.y + (b.y - a.y) * p}, segment: s, progress: p};
}
function agentVisual(agent,time=state.time) {
  const entry=frameEntry(agent,time);
  if(!entry)return calculateAgentVisual(agent,time);
  return entry.visual||(entry.visual=calculateAgentVisual(agent,time));
}
function prepareManoeuvres(data) {
  const diagnostics=data.movement_diagnostics||{},events=[
    ...(diagnostics.counterflow_recoveries||[]),
    ...(diagnostics.overtake_plans||[]).map(e=>({...e,action:'overtaking'}))
  ];
  const navigation=new Map(),byAgent=new Map();
  for(const row of data.navigation_updates||[]){
    if(!navigation.has(row.agent_id))navigation.set(row.agent_id,[]);
    navigation.get(row.agent_id).push(row);
  }
  const agents=new Map(data.agents.map(a=>[a.agent_id,a]));
  for(const event of events){
    const action=event.action||(event.cause==='visible_gate_approach_yield'?'yielding'
      :event.cause==='visible_person_blocking_path'?'overtaking':null);
    if(!['yielding','overtaking'].includes(action))continue;
    const agent=agents.get(event.agent_id);if(!agent)continue;
    let end=event.end_s;
    if(!Number.isFinite(end)){
      // 종료 시각이 없는 이전 결과는 경유점 도달·경로 변경·이동 중단으로 구간을 복원한다.
      end=Math.min(data.replay_duration_s??data.summary?.total_time_s??agent.segments.at(-1)?.end??event.time_s,
        agent.evacuation_time_s??Infinity,agent.incapacitated_at_s??Infinity);
      for(const row of navigation.get(event.agent_id)||[])
        if(row.time_s>event.time_s)end=Math.min(end,row.time_s);
      for(const next of events)
        if(next.agent_id===event.agent_id&&next.time_s>event.time_s)end=Math.min(end,next.time_s);
      for(const row of agent.visibility_blocked_history||[])
        if(row.blocked&&row.time_s>=event.time_s)end=Math.min(end,row.time_s);
      const last=event.waypoints?.at(-1),radius=event.waypoints?.length===1 ? .08 : .2;
      for(const segment of agent.segments){
        if(segment.end<=event.time_s)continue;
        if(segment.start>=end)break;
        if(segment.status!=='walking'||segment.from.floor!==event.floor){end=Math.min(end,segment.start);break;}
        if(last&&Math.hypot(segment.to.x-last[0],segment.to.y-last[1])<=radius){end=Math.min(end,segment.end);break;}
      }
    }
    if(end<=event.time_s)continue;
    if(!byAgent.has(event.agent_id))byAgent.set(event.agent_id,[]);
    byAgent.get(event.agent_id).push({start_s:event.time_s,end_s:end,action});
  }
  for(const rows of byAgent.values())rows.sort((a,b)=>a.start_s-b.start_s);
  return byAgent;
}
function manoeuvreAt(agent,time) {
  const rows=state.manoeuvresByAgent?.get(agent.agent_id)||[];
  let lo=0,hi=rows.length;
  while(lo<hi){const mid=(lo+hi)>>1;if(rows[mid].start_s<=time)lo=mid+1;else hi=mid;}
  const event=rows[lo-1];
  return event&&time<event.end_s?event.action:null;
}
function calculateAgentVisual(agent,time) {
  const p=position(agent,time),s=p.segment;
  let speed=0;
  if(s&&p.status==='walking')speed=Math.hypot(s.to.x-s.from.x,s.to.y-s.from.y)/Math.max(.001,s.end-s.start);
  if(s&&p.status==='on_stairs'){
    const dt=Math.min(.35,Math.max(.001,time-s.start)),past=position(agent,Math.max(s.start,time-dt));
    speed=Math.max(0,p.progress-past.progress)*state.data.scenario.stairs.find(x=>x.id===s.stair_id).length_m/dt;
  }
  const slow=speed<.25||['pre_moving','waiting','queueing'].includes(p.status);
  const manoeuvre=p.status==='walking'?manoeuvreAt(agent,time):null;
  const overtaking=p.status==='walking'&&(manoeuvre==='overtaking'||(state.overtakesByAgent?.get(agent.agent_id)||[]).some(e=>e.start_s<=time&&time<e.time_s));
  const obscured=p.status==='visibility_blocked'||smokeObscuredAt(agent,time,p);
  const kind=p.status==='incapacitated'?'incapacitated':p.status==='evacuated'?'evacuated':obscured?'visibility_blocked':manoeuvre==='yielding'?'yielding':overtaking?'overtaking':slow?'slow':'moving';
  return {p,speed,kind,color:{slow:'#efb35d',yielding:'#efb35d',visibility_blocked:'#ff4e88',overtaking:'#8050e4',moving:'#138eff',evacuated:'#7c8797',incapacitated:'#b91c1c'}[kind],
    label:kind==='visibility_blocked'?'시야 차단':kind==='yielding'?'양보 중':kind==='overtaking'?'추월 중':kind==='slow'&&p.status==='walking'?'정지·서행':kind==='slow'&&p.status==='on_stairs'?'계단 위 정지·서행':labels[p.status]};
}
function buildMap(){buildStationMap();}
function select(id) {
  state.selected = id;
  const agent = state.data.agents.find(a => a.agent_id === id);
  if (state.data.scenario.drawing) stationSelectAgent(agent);
  $('personTitle').textContent = agent.persona.occupation;
  $('personDescription').textContent = `${agent.persona.age}세`;
  document.querySelectorAll('.persona-list button').forEach(b => b.classList.toggle('selected', +b.dataset.id === id));
  document.querySelectorAll('.persona-list button').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.id===id)));
  if($('selectedAgentBadge'))$('selectedAgentBadge').textContent=`#${id}`;
  draw();
}
let selectedRouteFrame=null;
function drawSelectedRoute() {
  if(window.station3D?.ready)return;
  const agent=state.data.agents.find(a=>a.agent_id===state.selected);if(!agent)return;
  if(selectedRouteFrame?.agent===agent&&selectedRouteFrame.time===state.time)return;
  selectedRouteFrame={agent,time:state.time};
  const layer = $('selectedRoute'); layer.replaceChildren();
  let path = '', last = null;
  const floorPaths = {};
  for (const s of agent.segments) {
    if(s.start>=state.time)break;
    if (s.status !== 'walking') {last = null; continue;}
    const fraction=Math.min(1,(state.time-s.start)/(s.end-s.start));
    const end={...s.to,x:s.from.x+fraction*(s.to.x-s.from.x),y:s.from.y+fraction*(s.to.y-s.from.y)};
    const a = xy(s.from), b = xy(end);
    if (state.data.scenario.drawing) {
      floorPaths[s.from.floor] = (floorPaths[s.from.floor] || '') + `M${a.x},${a.y}L${b.x},${b.y}`;
      continue;
    }
    if (!last || last.x !== a.x || last.y !== a.y) path += `M${a.x},${a.y}`;
    path += `L${b.x},${b.y}`; last = b;
  }
  if (state.data.scenario.drawing) {
    for (const [floor,d] of Object.entries(floorPaths)) svg('path', {'data-map-floor':floor,d,stroke:'#ffffff44','stroke-width':1.5,fill:'none'},'',layer);
  } else svg('path', {d: path, stroke: '#ffffff44', 'stroke-width': 1.5, fill: 'none'}, '', layer);
}
function fact(parent, key, value) {const item = el('div'); item.append(el('small', {}, key), el('b', {}, value)); parent.append(item);}
function drawInspector(){drawPrivateInspector();}
function drawFire() {
  const fire = state.data.fire;
  if (!fire?.enabled) return;
  const mode = $('fieldMode').value, key = `${mode}-${Math.floor(state.time*5)}`;
  if (state.lastFireDraw === key) return;
  state.lastFireDraw = key;
  const frames = fire.frames;
  let hi = frames.findIndex(f => f.time_s >= state.time);
  if (hi < 0) hi = frames.length-1;
  const a = frames[Math.max(0,hi-1)], b = frames[hi];
  const w = a === b ? 0 : (state.time-a.time_s)/(b.time_s-a.time_s);
  let maxK = 0, maxT = 20;
  state.fireCells.forEach((node,i) => {
    const k = a.k[i]+w*(b.k[i]-a.k[i]), t = a.temperature_c[i]+w*(b.temperature_c[i]-a.temperature_c[i]);
    maxK = Math.max(maxK,k); maxT = Math.max(maxT,t);
    node.setAttribute('fill', mode === 'temperature' ? `hsl(${Math.max(0,50-(t-20)/180*50)} 95% 55%)` : '#9ba9b3');
    node.setAttribute('opacity', mode === 'none' ? 0 : mode === 'temperature' ? Math.max(0,Math.min(.85,(t-20)/180)) : Math.min(.85,1-Math.exp(-2*k)));
  });
  const ft = state.time+fire.metadata.fire.evacuation_start_s;
  if($('fireClock'))$('fireClock').textContent = `발화 후 ${ft.toFixed(1)}초 · 대피 ${state.time.toFixed(1)}초 | ${mode==='volume'?'보행 높이 단면':'표시 영역'} 최대 K ${maxK.toFixed(2)} /m · 최고 ${maxT.toFixed(1)}°C`;
  if($('fireSpreadStatus'))$('fireSpreadStatus').textContent=(fire.metadata.fire.stages||[]).map(s=>`${s.car_id}호차 · ${ft<s.start_s?'발화 전':ft<s.full_s?'발화·성장 중':'입력한 연소 단계'}`).join(' → ');
}
function draw() {
  if (!state.data) return;
  drawSelectedRoute();
  drawFire();
  const counts = {pre_moving:0,evacuating:0,evacuated:0,incapacitated:0}, stairCounts = {}, grouped = {};
  for (const stair of state.data.scenario.stairs) stairCounts[stair.id] = {queueing: 0, on_stairs: 0};
  for (const agent of state.data.agents) {
    const p = position(agent, state.time), g = $(`agent-${agent.agent_id}`);
    const category=p.status==='visibility_blocked'?'incapacitated':p.status==='pre_moving'||p.status==='evacuated'||p.status==='incapacitated'?p.status:'evacuating';
    counts[category]++;
    if (stairCounts[p.segment?.stair_id] && ['queueing', 'on_stairs'].includes(p.status)) stairCounts[p.segment.stair_id][p.status]++;
    // Spread only completed agents; live queues retain their actual JuPedSim coordinates.
    let dx = 0, dy = 0;
    if (p.status === 'evacuated') {
      const key = `${p.status}-${p.segment?.stair_id || agent.exit_id}`;
      const i = grouped[key] || 0; grouped[key] = i + 1;
      dx = (i % 5 - 2) * 21; dy = -Math.floor(i / 5) * 23;
    }
    g.setAttribute('transform', `translate(${p.pos.x + dx},${p.pos.y + dy})`);
    const circle = g.querySelector('circle'); circle.setAttribute('fill', agentVisual(agent).color);
    if(state.data.scenario.drawing){const visual=agentVisual(agent);circle.setAttribute('fill',visual.color);const dot=document.querySelector(`.persona-list button[data-id="${agent.agent_id}"] .agent-status-dot`);if(dot)dot.style.background=visual.color;}
    circle.setAttribute('stroke', state.selected === agent.agent_id ? '#ffffff' : '#102127');
    circle.setAttribute('stroke-width', state.selected === agent.agent_id ? 3 : 2);
    g.setAttribute('aria-label', `페르소나 ${agent.agent_id}, ${labels[p.status]}`);
  }
  for (const [id, count] of Object.entries(stairCounts)) if($(`live-${id}`)) $(`live-${id}`).textContent = `대기 ${count.queueing} · 이동 ${count.on_stairs}`;
  $('clock').textContent = `${state.time.toFixed(1)}초`; $('time').value = state.time;
  const statsKey=JSON.stringify(counts);
  if(state.liveStatsKey!==statsKey){state.liveStatsKey=statsKey;$('liveStats').replaceChildren(...Object.entries(liveStatsLabels).map(([key, label]) => {
    const span = el('span', {}, `${label} `); span.append(el('b', {}, `${counts[key]}명`)); return span;
  }));}
  drawInspector();
  if (state.data.scenario.drawing) drawStationView();
}
function stop() {state.playing = false; state.last = null; if (state.frame !== null) cancelAnimationFrame(state.frame); state.frame = null; $('play').textContent = '▶'; $('play').setAttribute('aria-label', '재생');}
function seekToTime(time) {
  if(!state.hasResults||!Number.isFinite(time))return;
  stop();
  const duration=state.data.replay_duration_s||state.data.summary.total_time_s;
  state.time=Math.max(0,Math.min(time,duration));
  state.inspectorKey=null;
  draw();
}
function animate(now) {
  if (!state.playing) return;
  if (state.last !== null) state.time += (now - state.last) / 1000 * Number($('speed').value);
  state.last = now;
  const duration=state.data.replay_duration_s||state.data.summary.total_time_s;
  if (state.time >= duration) {state.time = duration; stop();}
  draw();
  if (state.playing) state.frame = requestAnimationFrame(animate);
}
const agentTraitFilters=new Map();
function matchesAgentTraits(agent){
  for(const [name,levels] of agentTraitFilters){
    if(levels.size&&!levels.has(agent.experimental_profile?.traits?.[name]?.level))return false;
  }
  return true;
}
function updateTraitFilters(){
  const tags=$('activeTraitFilters');tags.replaceChildren();
  for(const [name,label] of Object.entries(traitNames)){
    const selected=agentTraitFilters.get(name)||new Set();
    for(const button of document.querySelectorAll(`#traitFilterOptions [data-trait="${name}"] button`))
      button.setAttribute('aria-pressed',String(selected.has(button.dataset.level)));
    if(!selected.size)continue;
    const levels=Object.entries(traitLevels).filter(([level])=>selected.has(level)).map(([,text])=>text).join('·');
    const tag=el('button',{type:'button',class:'active-trait-filter','aria-label':`${label} ${levels} 조건 해제`});
    tag.append(el('span',{},`${label} · ${levels}`),el('span',{'aria-hidden':'true'},'×'));
    tag.addEventListener('click',()=>{agentTraitFilters.delete(name);updateTraitFilters();$('toggleTraitFilters').focus();});
    tags.append(tag);
  }
  tags.hidden=!tags.childElementCount;
  sortAgentList();
  if(state.data){
    const visible=state.data.agents.filter(matchesAgentTraits);
    if(visible.length&&!visible.some(a=>a.agent_id===state.selected)){
      const first=$('personaList').querySelector('button:not([hidden])');
      if(first)select(Number(first.dataset.id));
    }
  }
}
function positionTraitFilters(){
  const popover=$('agentTraitFilters');if(!popover.matches(':popover-open'))return;
  const anchor=$('toggleTraitFilters').getBoundingClientRect(),box=popover.getBoundingClientRect();
  popover.style.left=`${Math.max(12,Math.min(anchor.left,innerWidth-box.width-12))}px`;
  const top=anchor.top-box.height-8;
  popover.style.top=`${Math.max(12,Math.min(top,innerHeight-box.height-12))}px`;
}
function initializeTraitFilters(){
  for(const [name,label] of Object.entries(traitNames)){
    const id=`trait-filter-${name}`,row=el('div',{class:'trait-filter-row','data-trait':name,role:'group','aria-labelledby':id});
    const options=el('div',{class:'trait-filter-levels'});
    for(const [level,text] of Object.entries(traitLevels)){
      const button=el('button',{type:'button','data-level':level,'aria-label':`${label} ${text}`,'aria-pressed':'false'},text);
      button.addEventListener('click',()=>{
        const selected=agentTraitFilters.get(name)||new Set();
        if(selected.has(level))selected.delete(level);else selected.add(level);
        if(selected.size)agentTraitFilters.set(name,selected);else agentTraitFilters.delete(name);
        updateTraitFilters();
      });
      options.append(button);
    }
    row.append(el('span',{id},label),options);$('traitFilterOptions').append(row);
  }
  $('agentTraitFilters').addEventListener('toggle',positionTraitFilters);
  window.addEventListener('resize',positionTraitFilters);
}
function sortAgentList(){
  if(!state.data)return;
  const mode=$('agentSort')?.value||'number',agents=[...state.data.agents];
  agents.sort((a,b)=>{
    if(mode==='speed_desc'||mode==='speed_asc'){
      const aKnown=Number.isFinite(a.desired_speed_mps),bKnown=Number.isFinite(b.desired_speed_mps);
      if(aKnown!==bKnown)return aKnown?-1:1;
      if(aKnown&&a.desired_speed_mps!==b.desired_speed_mps)
        return (mode==='speed_desc'?-1:1)*(a.desired_speed_mps-b.desired_speed_mps);
    }
    return a.agent_id-b.agent_id;
  });
  const list=$('personaList'),buttons=new Map([...list.children].map(b=>[b.dataset.id,b]));
  list.append(...agents.map(a=>buttons.get(String(a.agent_id))));
  let matchingCount=0;
  for(const agent of agents){
    const matches=matchesAgentTraits(agent);
    buttons.get(String(agent.agent_id)).hidden=!matches;
    if(matches)matchingCount++;
  }
  $('agentListCount').textContent=`${matchingCount}명`;
  $('agentFilterEmpty').hidden=matchingCount>0;
  list.scrollTop=0;
}
function showResults(data) {
  playbackFrame=null;state.liveStatsKey=null;resetPrivatePlayback(data);
  stop(); state.data = data; state.hasResults=!!data.summary; state.time = 0;state.inspectorKey=null;
  state.inspectorKey=null;state.overtakesByAgent=new Map();
  state.manoeuvresByAgent=prepareManoeuvres(data);
  for(const e of data.movement_diagnostics?.overtakes||[]){if(!state.overtakesByAgent.has(e.passing_agent))state.overtakesByAgent.set(e.passing_agent,[]);state.overtakesByAgent.get(e.passing_agent).push(e);}
  state.hazardByAgent = new Map();
  for (const h of data.hazard_history || []) {
    if (!state.hazardByAgent.has(h.agent_id)) state.hazardByAgent.set(h.agent_id, []);
    state.hazardByAgent.get(h.agent_id).push(h);
  }
  const summary = data.summary;
  $('metrics').replaceChildren();
  {
    const completed=summary?(summary.completed_count??data.agents.filter(a=>a.evacuation_time_s!=null).length):null;
    const counts=[['대피 완료',completed],['미완료',summary?(summary.count??data.agents.length)-completed:null]];
    const exits=new Set([...Object.keys(data.scenario.exit_labels||{}),...Object.keys(summary?.exit_split||{})]);
    for(const id of exits)counts.push([data.scenario.exit_labels?.[id]||id,summary?(summary.exit_split?.[id]??0):null]);
    const grid=el('div',{class:'result-grid'});
    for(const [label,value] of counts){
      const card=el('div',{class:'result-card'});
      card.append(el('small',{},label),el('strong',{},value??'—'),el('em',{},'명'));grid.append(card);
    }
    $('metrics').append(el('h3',{},'최종 결과'),grid);
  }
  // Round the control endpoint up; browsers may truncate arbitrary float maxima.
  const duration=data.replay_duration_s||summary?.total_time_s||0;
  $('time').max = Math.ceil(duration * 10) / 10; $('endTime').textContent = state.hasResults?`${duration.toFixed(1)}초`:'—';
  $('personaList').replaceChildren(...data.agents.map(a => {
    const speed=Number.isFinite(a.desired_speed_mps)?`${a.desired_speed_mps.toFixed(2)}m/s`:'계산 대기';
    const b=el('button',{'data-id':a.agent_id,'aria-label':`페르소나 ${a.agent_id} 선택`,title:`#${a.agent_id} · ${a.persona.occupation} · ${a.persona.age}세 · 평균 보행 속도 ${speed}`});
    if(data.scenario.drawing)b.append(el('span',{class:'agent-status-dot'}),el('span',{class:'agent-list-id'},`#${a.agent_id}`),el('span',{class:'agent-list-job'},a.persona.occupation),el('span',{class:'agent-list-age'},`${a.persona.age}세`));else b.textContent=a.agent_id;
    b.addEventListener('click',()=>select(a.agent_id));return b;
  }));
  sortAgentList();
  if(state.hasResults)$('behaviorMode').value=data.behavior?.mode||'off';
  $('agentCount').value=String(data.agents.length);
  if(state.hasResults)$('fireMode').value=data.fire?.enabled?'fds':'off';
  $('decisionLog')?.replaceChildren();
  for (const id of ['play', 'reset', 'time', 'download']) if($(id)) $(id).disabled = !state.hasResults;
  for(const id of ['run','behaviorRun'])if($(id))$(id).disabled=data.calculation_available===false||data.scenario.simulation_enabled===false;
  $('behaviorRun').textContent='시뮬레이션 계산';
  $('message').className = ''; $('message').textContent = data.artifact_status?.mode==='stored_replay'?'저장 결과 재생 · 현재 계산 설정과 다름':state.hasResults?'계산 완료':$('behaviorRun').disabled?'계산 사용 불가':'계산 준비';
  $('message').title=data.artifact_status?.mode==='stored_replay'?'저장 당시의 지도·이동·연기·개인별 기록을 그대로 재생합니다. 현재 설정이나 모델로 다시 계산하지 않습니다.':data.backend_notice||'';
  const visible=data.agents.filter(matchesAgentTraits);
  const selected=visible.find(a=>a.agent_id===state.selected)||visible[0]||data.agents.find(a=>a.agent_id===state.selected)||data.agents[0];
  state.dirty = false; buildMap(); select(selected.agent_id);
}
async function load(payload) {
  $('behaviorRun').disabled = true;
  stop(); ($('run')||{}).disabled = true; $('message').className = ''; $('message').title=''; $('message').textContent = payload?'계산 중':'불러오는 중';
  try {
    const station=document.body.dataset.station||'gaehwasan';
    const requestedRun=new URL(location.href).searchParams.get('result');
    const historyRun=!payload&&station==='gaehwasan'&&/^[0-9]{8}T[0-9]{6}$/.test(requestedRun||'')?requestedRun:null;
    const endpoint=historyRun?`/api/gaehwasan/history/${historyRun}`:`/api/${station}`;
    let response = await fetch(endpoint, payload ? {cache:'no-store',method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)} : {cache:'no-store'});
    if(!payload&&!historyRun&&station==='gaehwasan'&&[404,409].includes(response.status)){
      const problem=await response.clone().json();
      if(problem.detail?.code==='station_result_outdated')
        response=await fetch('/api/gaehwasan/replay',{cache:'no-store'});
      else if(problem.detail?.code==='station_result_missing')
        response=await fetch('/api/gaehwasan/setup',{cache:'no-store'});
    }
    if (!response.ok) {const problem = await response.json(); throw new Error(`계산 실패 (${response.status}): ${typeof problem.detail === 'string' ? problem.detail : problem.detail?.message||'입력값을 확인해 주세요.'}`);}
    showResults(await response.json());
  } catch (error) {$('message').className = 'error'; $('message').textContent = error.message; $('behaviorRun').disabled = false; if (state.data) ($('run')||{}).disabled = false;}
}
$('stairsForm').addEventListener('submit', event => {
  event.preventDefault();
  const stairs = [...document.querySelectorAll('[data-stair-id]')].map(card => ({id: card.dataset.stairId, ...Object.fromEntries([...card.querySelectorAll('input')].map(input => [input.name, Number(input.value)]))}));
  load({stairs, behavior_mode: $('behaviorMode').value, profile_method: $('behaviorMode').value==='qwen'?'full_persona':'keyword_candidates', agent_count: Number($('agentCount').value), fire_mode: $('fireMode').value});
});
$('stairsForm').addEventListener('input', () => {state.dirty = true; $('message').className=''; $('message').textContent = '설정 변경됨 · 다시 계산해 주세요';$('message').title='';});
$('agentSort')?.addEventListener('change',sortAgentList);
window.addEventListener('resize',()=>fitDecisionLog($('decisionLog')));
$('play').addEventListener('click', () => {if (state.playing) return stop(); if (!state.hasResults) return; if (state.time >= (state.data.replay_duration_s||state.data.summary.total_time_s)) state.time = 0; if(state.data.scenario.drawing) stationPlaybackStarted(); state.playing = true; state.last = null; $('play').textContent = '❚❚'; $('play').setAttribute('aria-label', '일시정지'); state.frame = requestAnimationFrame(animate);});
$('reset').addEventListener('click', () => seekToTime(0));
$('time').addEventListener('input', e => seekToTime(Number(e.target.value)));
$('download')?.addEventListener('click', () => {if (!state.data) return; const url = URL.createObjectURL(new Blob([JSON.stringify(state.data, null, 2)], {type: 'application/json'})); const a = el('a', {href: url, download: `${document.body.dataset.station || 'gaehwasan'}-${state.data.scenario_hash.slice(0, 12)}.json`}); document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);});
function updateOpacityControls(){
  $('smokeOpacityControls').hidden=$('fieldMode').value!=='volume';
}
$('fieldMode').addEventListener('change', () => {updateOpacityControls();state.lastFireDraw = null;if(state.data)draw();});
updateOpacityControls();
initializeTraitFilters();
async function followPendingResult() {
  const runId=new URL(location.href).searchParams.get('pending');
  if(!runId||!/^\d{8}T\d{6}$/.test(runId))return;
  try{
    const response=await fetch('/api/gaehwasan/activation',{cache:'no-store'});
    if(response.ok){
      const info=await response.json(),run=info.run;
      if(run?.run_id===runId){
        if(run.status==='실패'){
          $('message').className='error';$('message').textContent='새 계산 실패 · 기존 결과 유지';return;
        }
        if(run.artifact_sha256&&run.artifact_sha256===info.active_artifact_sha256){
          const saved=await fetch('/api/gaehwasan/replay',{cache:'no-store'});
          if(!saved.ok)throw new Error('새 결과 조회 대기');
          const result=await saved.json();
          if(result.artifact_status?.artifact_sha256!==run.artifact_sha256)throw new Error('새 결과 반영 대기');
          showResults(result);
          const url=new URL(location.href);url.searchParams.set('result',runId);url.searchParams.delete('pending');history.replaceState(null,'',url);
          $('message').textContent=run.status.includes('검증 중')?'새 계산 결과 반영 · 검증 중':'새 계산 결과 반영';
          return;
        }
      }
    }
  }catch(error){ /* 일시적인 파일·연결 전환 중에는 현재 화면을 보존한다. */ }
  setTimeout(followPendingResult,5000);
}
load().then(followPendingResult);
