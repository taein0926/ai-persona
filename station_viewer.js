'use strict';
const $ = id => document.getElementById(id);
const ns = 'http://www.w3.org/2000/svg';
const state = {data: null, hasResults: false, time: 0, playing: false, selected: null, frame: null, last: null, dirty: false};
const labels = {pre_moving: '대피 전 대기', walking: '평면 이동', waiting: '판단 후 대기', queueing: '통로 진입 대기', on_stairs: '계단 이동', on_gate:'개찰구 통과', evacuated: '지상 도착'};
const colors = {pre_moving: '#e6d68d', walking: '#74e0bb', waiting: '#ef94cb', queueing: '#f0b55d', on_stairs: '#91baff', on_gate: '#74e0bb', evacuated: '#6c9691'};
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
function calculateAgentVisual(agent,time) {
  const p=position(agent,time),s=p.segment;
  let speed=0;
  if(s&&p.status==='walking')speed=Math.hypot(s.to.x-s.from.x,s.to.y-s.from.y)/Math.max(.001,s.end-s.start);
  if(s&&p.status==='on_stairs'){
    const dt=Math.min(.35,Math.max(.001,time-s.start)),past=position(agent,Math.max(s.start,time-dt));
    speed=Math.max(0,p.progress-past.progress)*state.data.scenario.stairs.find(x=>x.id===s.stair_id).length_m/dt;
  }
  const slow=speed<.25||['pre_moving','waiting','queueing'].includes(p.status);
  const overtaking=p.status==='walking'&&(state.overtakesByAgent?.get(agent.agent_id)||[]).some(e=>e.start_s<=time&&time<e.time_s);
  const kind=p.status==='evacuated'?'evacuated':slow?'slow':overtaking?'overtaking':'moving';
  return {p,speed,kind,color:{slow:'#ff4e88',overtaking:'#8050e4',moving:'#138eff',evacuated:'#7c8797'}[kind],
    label:kind==='overtaking'?'추월 중':kind==='slow'&&p.status==='walking'?'정지·서행':kind==='slow'&&p.status==='on_stairs'?'계단 위 정지·서행':labels[p.status]};
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
  const layer = $('selectedRoute'); layer.replaceChildren();
  let path = '', last = null;
  const floorPaths = {};
  for (const s of agent.segments) {
    if (s.status !== 'walking') {last = null; continue;}
    const a = xy(s.from), b = xy(s.to);
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
  draw();
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
  drawFire();
  const counts = {}, stairCounts = {}, grouped = {};
  for (const stair of state.data.scenario.stairs) stairCounts[stair.id] = {queueing: 0, on_stairs: 0};
  for (const agent of state.data.agents) {
    const p = position(agent, state.time), g = $(`agent-${agent.agent_id}`);
    counts[p.status] = (counts[p.status] || 0) + 1;
    if (stairCounts[p.segment?.stair_id] && ['queueing', 'on_stairs'].includes(p.status)) stairCounts[p.segment.stair_id][p.status]++;
    // Spread only completed agents; live queues retain their actual JuPedSim coordinates.
    let dx = 0, dy = 0;
    if (p.status === 'evacuated') {
      const key = `${p.status}-${p.segment?.stair_id || agent.exit_id}`;
      const i = grouped[key] || 0; grouped[key] = i + 1;
      dx = (i % 5 - 2) * 21; dy = -Math.floor(i / 5) * 23;
    }
    g.setAttribute('transform', `translate(${p.pos.x + dx},${p.pos.y + dy})`);
    const circle = g.querySelector('circle'); circle.setAttribute('fill', colors[p.status]);
    if(state.data.scenario.drawing){const visual=agentVisual(agent);circle.setAttribute('fill',visual.color);const dot=document.querySelector(`.persona-list button[data-id="${agent.agent_id}"] .agent-status-dot`);if(dot)dot.style.background=visual.color;}
    circle.setAttribute('stroke', state.selected === agent.agent_id ? '#ffffff' : '#102127');
    circle.setAttribute('stroke-width', state.selected === agent.agent_id ? 3 : 2);
    g.setAttribute('aria-label', `페르소나 ${agent.agent_id}, ${labels[p.status]}`);
  }
  for (const [id, count] of Object.entries(stairCounts)) if($(`live-${id}`)) $(`live-${id}`).textContent = `대기 ${count.queueing} · 이동 ${count.on_stairs}`;
  $('clock').textContent = `${state.time.toFixed(1)}초`; $('time').value = state.time;
  const statsKey=JSON.stringify(counts);
  if(state.liveStatsKey!==statsKey){state.liveStatsKey=statsKey;$('liveStats').replaceChildren(...Object.entries(labels).map(([key, label]) => {
    const span = el('span', {}, `${label} `); span.append(el('b', {}, `${counts[key] || 0}명`)); return span;
  }));}
  drawInspector();
  if (state.data.scenario.drawing) drawStationView();
}
function stop() {state.playing = false; state.last = null; if (state.frame !== null) cancelAnimationFrame(state.frame); state.frame = null; $('play').textContent = '▶'; $('play').setAttribute('aria-label', '재생');}
function animate(now) {
  if (!state.playing) return;
  if (state.last !== null) state.time += (now - state.last) / 1000 * Number($('speed').value);
  state.last = now;
  const duration=state.data.replay_duration_s||state.data.summary.total_time_s;
  if (state.time >= duration) {state.time = duration; stop();}
  draw();
  if (state.playing) state.frame = requestAnimationFrame(animate);
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
  list.scrollTop=0;
}
function showResults(data) {
  playbackFrame=null;state.liveStatsKey=null;resetPrivatePlayback(data);
  stop(); state.data = data; state.hasResults=!!data.summary; state.time = 0;state.inspectorKey=null;
  state.inspectorKey=null;state.overtakesByAgent=new Map();
  for(const e of data.movement_diagnostics?.overtakes||[]){if(!state.overtakesByAgent.has(e.passing_agent))state.overtakesByAgent.set(e.passing_agent,[]);state.overtakesByAgent.get(e.passing_agent).push(e);}
  state.hazardByAgent = new Map();
  for (const h of data.hazard_history || []) {
    if (!state.hazardByAgent.has(h.agent_id)) state.hazardByAgent.set(h.agent_id, []);
    state.hazardByAgent.get(h.agent_id).push(h);
  }
  const summary = data.summary;
  $('metrics').hidden=!state.hasResults;
  $('liveStats').hidden=!state.hasResults;
  $('metrics').replaceChildren();
  if(summary){
    const completed=summary.completed_count??data.agents.filter(a=>a.evacuation_time_s!=null).length;
    const counts=[['대피 완료',completed],['미완료',(summary.count??data.agents.length)-completed]];
    const exits=new Set([...Object.keys(data.scenario.exit_labels||{}),...Object.keys(summary.exit_split||{})]);
    for(const id of exits)counts.push([data.scenario.exit_labels?.[id]||id,summary.exit_split?.[id]??0]);
    const grid=el('div',{class:'result-grid'});
    for(const [label,value] of counts){
      const card=el('div',{class:'result-card'});
      card.append(el('small',{},label),el('strong',{},value),el('em',{},'명'));grid.append(card);
    }
    $('metrics').append(el('h3',{},'최종 결과'),grid);
  }
  // Round the control endpoint up; browsers may truncate arbitrary float maxima.
  const duration=data.replay_duration_s||summary?.total_time_s||0;
  $('time').max = Math.ceil(duration * 10) / 10; $('endTime').textContent = state.hasResults?`${duration.toFixed(1)}초`:'—';
  $('personaList').replaceChildren(...data.agents.map(a => {
    const speed=Number.isFinite(a.desired_speed_mps)?`${a.desired_speed_mps.toFixed(2)}m/s`:'계산 대기';
    const b=el('button',{'data-id':a.agent_id,'aria-label':`페르소나 ${a.agent_id} 선택`,title:`#${a.agent_id} · ${a.persona.occupation} · ${a.persona.age}세 · 보행 속도 ${speed}`});
    if(data.scenario.drawing)b.append(el('span',{class:'agent-status-dot'}),el('span',{class:'agent-list-id'},`#${a.agent_id}`),el('span',{class:'agent-list-job'},a.persona.occupation),el('span',{class:'agent-list-age'},`${a.persona.age}세`));else b.textContent=a.agent_id;
    b.addEventListener('click',()=>select(a.agent_id));return b;
  }));
  if($('agentListCount'))$('agentListCount').textContent=`${data.agents.length}명`;
  sortAgentList();
  if(state.hasResults)$('behaviorMode').value=data.behavior?.mode||'off';
  $('agentCount').value=String(data.agents.length);
  if(state.hasResults)$('fireMode').value=data.fire?.enabled?'fds':'off';
  $('decisionLog')?.replaceChildren();
  for (const id of ['play', 'reset', 'time', 'download']) if($(id)) $(id).disabled = !state.hasResults;
  for(const id of ['run','behaviorRun'])if($(id))$(id).disabled=data.calculation_available===false||data.scenario.simulation_enabled===false;
  $('behaviorRun').textContent='설정으로 계산';
  $('message').className = ''; $('message').textContent = state.hasResults?'계산 완료':$('behaviorRun').disabled?'계산 사용 불가':'계산 준비';
  $('message').title=data.backend_notice||'';
  state.dirty = false; buildMap(); select(data.agents.some(a => a.agent_id === state.selected) ? state.selected : data.agents[0].agent_id);
}
async function load(payload) {
  $('behaviorRun').disabled = true;
  stop(); ($('run')||{}).disabled = true; $('message').className = ''; $('message').title=''; $('message').textContent = payload?'계산 중':'불러오는 중';
  try {
    const station=document.body.dataset.station||'gaehwasan';
    let response = await fetch(`/api/${station}`, payload ? {cache:'no-store',method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)} : {cache:'no-store'});
    if(!payload&&station==='gaehwasan'&&[404,409].includes(response.status)){
      const problem=await response.clone().json();
      if(['station_result_missing','station_result_outdated'].includes(problem.detail?.code))
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
$('play').addEventListener('click', () => {if (state.playing) return stop(); if (!state.hasResults) return; if (state.time >= (state.data.replay_duration_s||state.data.summary.total_time_s)) state.time = 0; if(state.data.scenario.drawing) stationPlaybackStarted(); state.playing = true; state.last = null; $('play').textContent = '❚❚'; $('play').setAttribute('aria-label', '일시정지'); state.frame = requestAnimationFrame(animate);});
$('reset').addEventListener('click', () => {stop(); state.time = 0; draw();});
$('time').addEventListener('input', e => {if(!state.hasResults)return;stop(); state.time = Math.min(Number(e.target.value), state.data.replay_duration_s||state.data.summary.total_time_s); draw();});
$('download')?.addEventListener('click', () => {if (!state.data) return; const url = URL.createObjectURL(new Blob([JSON.stringify(state.data, null, 2)], {type: 'application/json'})); const a = el('a', {href: url, download: `${document.body.dataset.station || 'gaehwasan'}-${state.data.scenario_hash.slice(0, 12)}.json`}); document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);});
function updateOpacityControls(){
  $('smokeOpacityControls').hidden=$('fieldMode').value!=='volume';
}
$('fieldMode').addEventListener('change', () => {updateOpacityControls();state.lastFireDraw = null;if(state.data)draw();});
updateOpacityControls();
load();
