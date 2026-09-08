const controls = ['normal', 'follower', 'anxious', 'delayed'];
const defaults = [45, 30, 15, 10];
const SIM_MS_PER_SECOND = 170;
let running = false, animationId, startedAt = 0, simSeconds = 0, completionTime = 56;
let firePosition = { x: 64, y: 36 };
const $ = id => document.getElementById(id);
const defaultRiskText = $('riskText').textContent;

function values() { return controls.map(id => +$(id).value); }
function refresh() {
  const v = values(), total = v.reduce((a, b) => a + b, 0);
  controls.forEach((id, i) => $(id + 'Out').value = v[i]);
  $('total').textContent = total + '%'; $('total').style.color = total === 100 ? 'var(--teal)' : 'var(--red)'; updateMetrics(v);
}
function updateMetrics(v) {
  const [normal, follower, anxious, delayed] = v, crowd = Math.round(1230 + follower * 7 + delayed * 4 - anxious * 3), bottleneck = Math.min(98, Math.round(50 + follower * .95 + delayed * .25 - anxious * .30));
  $('currentPeople').textContent = crowd.toLocaleString('ko-KR'); $('futurePeople').textContent = Math.round(crowd * (1.1 + bottleneck / 600)).toLocaleString('ko-KR');
  $('denA').textContent = Math.round(33 + normal * .28) + '%'; $('denB').textContent = bottleneck + '%'; $('denC').textContent = Math.round(27 + anxious * .4 + delayed * .2) + '%';
  $('barA').style.width = $('denA').textContent; $('barB').style.width = bottleneck + '%'; $('barC').style.width = $('denC').textContent;
  const high = bottleneck >= 80; $('riskLevel').textContent = high ? '위험' : '주의'; $('riskLevel').style.color = high ? 'var(--red)' : 'var(--yellow)';
  $('riskText').textContent = high ? '3번 출구 앞 병목 발생 예상' : defaultRiskText; $('recommendation').innerHTML = high ? '3번 출구 병목 예상<br><b>1번 출구로 승객 35% 분산</b>' : '환승 통로 흐름 관찰 필요<br><b>안내 방송 준비 권고</b>';
  document.querySelector('.h2').style.opacity = Math.min(.85, .2 + bottleneck / 130);
}
function randomStart() { let x, y; do { x = 12 + Math.random() * 57; y = 18 + Math.random() * 57; } while (Math.abs(x - firePosition.x) < 12 && Math.abs(y - firePosition.y) < 15); return { x, y }; }
function routeFor(type, index, start) {
  const exitOne = type === 'anxious' || (type === 'normal' && index % 3 === 0) || (type === 'delayed' && index % 2 === 0);
  const safeLeft = Math.max(8, firePosition.x - 18), safeTop = Math.max(9, firePosition.y - 18), safeBottom = Math.min(82, firePosition.y + 20);
  return exitOne ? [start, { x: safeLeft, y: safeTop }, { x: 76, y: safeTop }, { x: 91, y: 18 }] : [start, { x: safeLeft, y: safeBottom }, { x: 77, y: safeBottom }, { x: 91, y: 66 }];
}
function generatePeople() {
  const box = $('people'); box.innerHTML = ''; const types = ['normal', 'follower', 'anxious', 'delayed']; let globalIndex = 0;
  types.forEach((type, k) => { const count = Math.max(3, Math.round(values()[k] / 4)); for (let i = 0; i < count; i++, globalIndex++) {
    const start = randomStart(), p = document.createElement('i'); p.className = 'person ' + type; p.textContent = type === 'anxious' ? '◆' : type === 'delayed' ? '▲' : '●';
    p.dataset.delay = type === 'delayed' ? '8' : '0'; p.dataset.travel = type === 'follower' ? '50' : type === 'delayed' ? '48' : '45'; p.dataset.route = JSON.stringify(routeFor(type, globalIndex, start)); p.style.left = start.x + '%'; p.style.top = start.y + '%'; box.appendChild(p);
  }});
}
function pointAlong(route, progress) { const lengths = route.slice(1).map((p, i) => Math.hypot(p.x - route[i].x, p.y - route[i].y)); let d = lengths.reduce((a, b) => a + b, 0) * Math.min(1, progress); for (let i = 0; i < lengths.length; i++) { if (d <= lengths[i]) { const t = d / lengths[i]; return { x: route[i].x + (route[i + 1].x - route[i].x) * t, y: route[i].y + (route[i + 1].y - route[i].y) * t }; } d -= lengths[i]; } return route.at(-1); }
function movePeople(sec) { document.querySelectorAll('.person').forEach(p => { const delay = +p.dataset.delay, travel = +p.dataset.travel; if (sec < delay) { p.style.opacity = '.62'; return; } const point = pointAlong(JSON.parse(p.dataset.route), (sec - delay) / travel); p.style.opacity = sec >= delay + travel ? '.12' : '1'; p.style.left = point.x + '%'; p.style.top = point.y + '%'; }); }
function setTime(sec) { simSeconds = Math.min(completionTime, Math.max(0, sec)); const display = Math.floor(simSeconds); $('time').value = display; $('clock').textContent = 'SIM TIME 00:' + String(display).padStart(2, '0'); $('timeLabel').textContent = display === 0 ? '현재 상태' : display >= completionTime ? '전원 대피 완료' : display + '초 경과'; movePeople(simSeconds); }
function stop(completed = false) { running = false; cancelAnimationFrame(animationId); $('play').textContent = '▶'; if (completed) { $('timeLabel').textContent = '전원 대피 완료'; $('evacTime').textContent = Math.ceil(completionTime) + '초'; } }
function animate(now) { setTime((now - startedAt) / SIM_MS_PER_SECOND); if (simSeconds >= completionTime) { stop(true); return; } animationId = requestAnimationFrame(animate); }
function start() { if (values().reduce((a, b) => a + b, 0) !== 100) { $('total').textContent = '합계를 100%로 맞춰주세요'; return; } cancelAnimationFrame(animationId); generatePeople(); completionTime = Math.max(...[...document.querySelectorAll('.person')].map(p => +p.dataset.delay + +p.dataset.travel)); $('time').max = completionTime; document.querySelector('.timeline span:last-of-type').textContent = completionTime + 's'; $('evacTime').textContent = '—'; setTime(0); running = true; startedAt = performance.now(); $('play').textContent = '❚❚'; animationId = requestAnimationFrame(animate); }
controls.forEach(id => $(id).addEventListener('input', refresh));
controls.forEach(id => $(id + 'Out').addEventListener('input', e => { $(id).value = Math.min(100, Math.max(0, +e.target.value || 0)); refresh(); }));
$('simulate').addEventListener('click', start);
$('reset').addEventListener('click', () => { controls.forEach((id, i) => $(id).value = defaults[i]); refresh(); generatePeople(); setTime(0); $('evacTime').textContent = '—'; stop(); });
$('time').addEventListener('input', e => { stop(); setTime(+e.target.value); }); $('play').addEventListener('click', () => { if (running) stop(); else start(); });
$('station').addEventListener('click', e => { const rect = $('station').getBoundingClientRect(); firePosition = { x: Math.max(8, Math.min(86, (e.clientX - rect.left) / rect.width * 100)), y: Math.max(8, Math.min(82, (e.clientY - rect.top) / rect.height * 100)) }; $('fire').style.left = firePosition.x + '%'; $('fire').style.top = firePosition.y + '%'; generatePeople(); setTime(0); $('evacTime').textContent = '—'; stop(); });
refresh(); generatePeople();
