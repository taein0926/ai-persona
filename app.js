const state = { data: null, variant: 'C', time: 0, playing: false, selected: null, lastFrame: 0 };
const $ = id => document.getElementById(id);
const canvas = $('replay');
const ctx = canvas.getContext('2d');
const colors = { pre_moving: '#ffd261', evacuating: '#2be0c4', waiting_or_rerouting: '#ffad55', incapacitated: '#ff6868', evacuated: '#66858a' };

function pointAt(agent, time) {
  const points = agent.trajectory;
  if (time <= points[0].t) return points[0];
  if (time >= points.at(-1).t) return points.at(-1);
  const hi = points.findIndex(point => point.t >= time);
  const a = points[hi - 1], b = points[hi], ratio = (time - a.t) / (b.t - a.t || 1);
  return { ...a, x: a.x + (b.x - a.x) * ratio, y: a.y + (b.y - a.y) * ratio, fed: a.fed + (b.fed - a.fed) * ratio };
}
function screen(point) { return { x: 45 + point.x / 30 * 810, y: 510 - point.y / 13 * 460 }; }
function drawMap() {
  ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.fillStyle = '#19353d'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#31535b'; ctx.lineWidth = 1;
  for (let x = 55; x < 850; x += 36) { ctx.beginPath(); ctx.moveTo(x, 55); ctx.lineTo(x, 500); ctx.stroke(); }
  for (let y = 55; y < 510; y += 36) { ctx.beginPath(); ctx.moveTo(45, y); ctx.lineTo(850, y); ctx.stroke(); }
  ctx.strokeStyle = '#89aeb4'; ctx.lineWidth = 3; ctx.strokeRect(45, 50, 810, 460);
  ctx.fillStyle = '#214750'; ctx.fillRect(45, 50, 810, 106); ctx.fillRect(504, 156, 162, 354);
  ctx.fillStyle = '#d6ebec'; ctx.font = '14px DM Mono'; ctx.fillText('T-JUNCTION', 520, 178); ctx.fillText('SPAWN', 565, 475);
  drawExit(45, 103, 27, 'EXIT A'); drawExit(828, 103, 27, 'EXIT B');
  const fire = screen({ x: 25, y: 11 }); const smoke = ctx.createRadialGradient(fire.x, fire.y, 5, fire.x, fire.y, 105);
  smoke.addColorStop(0, '#ff5c5c99'); smoke.addColorStop(1, '#ff5c5c00'); ctx.fillStyle = smoke; ctx.beginPath(); ctx.arc(fire.x, fire.y, 105, 0, Math.PI * 2); ctx.fill();
  ctx.font = '25px sans-serif'; ctx.fillText('🔥', fire.x - 12, fire.y + 8);
}
function drawExit(x, y, width, label) {
  ctx.strokeStyle = '#2be0c4'; ctx.lineWidth = 3; ctx.strokeRect(x, y - 22, width, 44);
  ctx.fillStyle = '#bffbf2'; ctx.font = '13px DM Mono'; ctx.textAlign = 'center';
  ctx.fillText(label, x + width / 2, y - 31); ctx.textAlign = 'left';
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.arc(x + width / 2, y, 7, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.arc(x + width / 2, y, 2.5, 0, Math.PI * 2); ctx.fill();
}
function draw() {
  if (!state.data) return; drawMap();
  state.data.variants[state.variant].forEach(agent => {
    const point = pointAt(agent, state.time), pos = screen(point), selected = state.selected?.agent_id === agent.agent_id;
    ctx.beginPath(); ctx.arc(pos.x, pos.y, selected ? 8 : 5, 0, Math.PI * 2); ctx.fillStyle = colors[point.status]; ctx.shadowColor = colors[point.status]; ctx.shadowBlur = selected ? 18 : 8; ctx.fill(); ctx.shadowBlur = 0;
    if (selected) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke(); }
  });
  $('clock').textContent = `SIM 00:${state.time.toFixed(1).padStart(4, '0')}`; $('timeLabel').textContent = `${state.time.toFixed(1)}초`;
}
function renderSummary() {
  const s = state.data.summaries[state.variant];
  $('summary').innerHTML = `<div><span>전원 대피</span><b>${s.all_evacuated ? 'YES' : 'NO'}</b></div><div><span>완료 시간</span><b>${s.total_evacuation_time_s?.toFixed(1) ?? '—'}s</b></div><div><span>평균 사전지연</span><b>${s.mean_pre_movement_s.toFixed(2)}s</b></div><div><span>평균 속도</span><b>${s.mean_speed_mps.toFixed(2)}m/s</b></div><div><span>경로 변경</span><b>${s.total_route_switches}</b></div><div><span>최대 FED</span><b>${s.max_fed.toFixed(4)}</b></div>`;
  $('comparison').innerHTML = `<p><span>backend</span><b>${state.data.backend}</b></p><p><span>provider</span><b>${state.data.provenance.provider}</b></p>`;
}
function renderInspector() {
  const agent = state.selected; $('emptyInspector').hidden = Boolean(agent); $('agentInspector').hidden = !agent; if (!agent) return;
  const point = pointAt(agent, state.time); $('agentState').textContent = point.status.replaceAll('_', ' ').toUpperCase(); $('agentId').textContent = `#${agent.agent_id} · SLOT ${agent.agent_slot}`;
  $('occupation').textContent = `${agent.persona.occupation} · ${agent.persona.age}세`; $('personaText').textContent = agent.persona.persona;
  const mapping = agent.mapping;
  $('behaviorFacts').innerHTML = [['response_tendency', mapping?.response_tendency ?? '사용 안 함'], ['speed_tendency', mapping?.speed_tendency ?? '사용 안 함']].map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join('');
  $('empiricalFacts').innerHTML = [['Jülich response_time', `${agent.empirical?.response_time_s?.toFixed(2) ?? agent.pre_movement_s.toFixed(2)}s`], ['Jülich speed proxy', `${agent.empirical?.observed_speed_proxy_mps?.toFixed(2) ?? agent.desired_speed_mps.toFixed(2)}m/s`], ['premovement', `${agent.pre_movement_s.toFixed(2)}s`], ['desired speed proxy', `${agent.desired_speed_mps.toFixed(2)}m/s`]].map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join('');
  $('scenarioFacts').innerHTML = [['age / sex', `${agent.persona.age}세 · ${agent.persona.sex}`], ['occupation', agent.persona.occupation], ['Persona UUID', agent.persona.uuid], ['지역', `${agent.persona.province} ${agent.persona.district}`], ['social mode', agent.scenario_context.social_mode], ['familiarity', agent.scenario_context.familiarity], ['출구', agent.exit_id], ['현재 FED', point.fed.toFixed(6)]].map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join('');
  $('reasonText').innerHTML = mapping ? `<strong>Response</strong> ${mapping.reasoning_summary.response_tendency}<br><strong>Speed</strong> ${mapping.reasoning_summary.speed_tendency}` : '이 실험군은 LLM mapping을 사용하지 않습니다.'; $('provenance').textContent = JSON.stringify(agent.provenance, null, 2);
}
function selectVariant(variant) { state.variant = variant; state.selected = null; document.querySelectorAll('[data-variant]').forEach(button => button.classList.toggle('active', button.dataset.variant === variant)); renderSummary(); renderInspector(); draw(); }
function animate(timestamp) { if (!state.playing) return; if (!state.lastFrame) state.lastFrame = timestamp; state.time += (timestamp - state.lastFrame) / 180; state.lastFrame = timestamp; const max = +$('time').max; if (state.time >= max) { state.time = max; state.playing = false; $('play').textContent = '▶'; } $('time').value = state.time; draw(); renderInspector(); if (state.playing) requestAnimationFrame(animate); }
canvas.addEventListener('click', event => { const rect = canvas.getBoundingClientRect(), sx = canvas.width / rect.width, sy = canvas.height / rect.height, click = { x: (event.clientX - rect.left) * sx, y: (event.clientY - rect.top) * sy }; state.selected = state.data.variants[state.variant].map(agent => ({ agent, pos: screen(pointAt(agent, state.time)) })).sort((a, b) => Math.hypot(a.pos.x - click.x, a.pos.y - click.y) - Math.hypot(b.pos.x - click.x, b.pos.y - click.y))[0]?.agent ?? null; renderInspector(); draw(); });
$('variantTabs').addEventListener('click', event => { const button = event.target.closest('button'); if (button) selectVariant(button.dataset.variant); });
$('time').addEventListener('input', event => { state.time = +event.target.value; state.playing = false; $('play').textContent = '▶'; draw(); renderInspector(); });
$('play').addEventListener('click', () => { state.playing = !state.playing; $('play').textContent = state.playing ? '❚❚' : '▶'; state.lastFrame = 0; if (state.playing) { if (state.time >= +$('time').max) state.time = 0; requestAnimationFrame(animate); } });
async function loadResults() {
  let response = await fetch('/api/results');
  if (!response.ok) response = await fetch('results.json');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}
loadResults().then(data => { state.data = data; $('experiment').textContent = `${data.experiment_id} · SEED ${data.replicate_seed}`; $('assignmentHash').textContent = data.assignment_manifest_hash; $('backendNotice').textContent = data.backend_notice; $('footerMeta').textContent = `${data.backend.toUpperCase()} · ${data.variants.C.length} AGENTS`; const max = Math.max(...data.variants.C.map(agent => agent.trajectory.at(-1).t)); $('time').max = max; $('maxTime').textContent = `${max.toFixed(1)}s`; renderSummary(); draw(); }).catch(error => { $('backendNotice').textContent = `결과를 불러오지 못했습니다: ${error.message}`; });
