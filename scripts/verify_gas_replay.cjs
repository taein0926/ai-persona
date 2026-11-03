// 실제 재생 함수를 사용해 기준 도달 전후와 되감기 시 상태를 확인한다.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const code=fs.readFileSync('station_viewer.js','utf8');
const context=vm.createContext({console,document:{getElementById:()=>null},window:{},URLSearchParams});
// 초기 네트워크 로딩 이전까지의 순수 재생 함수를 검사한다.
const end=code.indexOf('function buildMap()');
vm.runInContext(code.slice(0,end),context);
vm.runInContext(`
function stationXY(p){return {x:p.x,y:p.y};}
state.data={scenario:{stairs:[{id:'S',travel_time_s:4,length_m:4}]}};
globalThis.agent={spawn:{x:1,y:2},evacuation_time_s:null,incapacitated_at_s:2,
  gas_exposure_history:[{time_s:0,fed:0},{time_s:1,fed:.5},{time_s:2,fed:1},{time_s:3,fed:1.5}],
  segments:[{start:0,end:2,from:{floor:'B2',x:1,y:2},to:{floor:'B2',x:2,y:2},status:'walking'},
    {start:2,end:4,from:{floor:'B2',x:2,y:2},to:{floor:'B2',x:2,y:2},status:'incapacitated'}]};
`,context);
function read(expression){return vm.runInContext(expression,context);}
assert.equal(read('position(agent,1.99).status'),'walking');
assert.equal(read('position(agent,2).status'),'incapacitated');
assert.equal(read('agentVisual(agent,3).label'),'가스 노출로 대피 불능');
assert.equal(read('agentVisual(agent,3).color'),'#b91c1c');
assert.equal(read('position(agent,3).pos.x'),2);
assert.equal(read('gasExposureAt(agent,1.5).fed'),.5);
assert.equal(read('gasExposureAt(agent,2).fed'),1);
assert.equal(read('gasExposureAt(agent,0).fed'),0);
assert.equal(read('position(agent,1).status'),'walking');
read(`agent.segments=[{start:0,end:4,from:{floor:'B2',x:1,y:2},to:{floor:'B1',x:1,y:4},status:'on_stairs',stair_id:'S',stair_motion:[[0,0],[2,.5],[4,.5]]}];`);
assert.equal(read('position(agent,3).physical_status'),'on_stairs');
assert.equal(read('position(agent,3).progress'),.5);
assert.equal(read('position(agent,3).status'),'incapacitated');
console.log('가스 대피 불능 재생·색상·노출 기록·되감기 검증 통과');
read(`agent.incapacitated_at_s=null;
agent.visibility_blocked_history=[{time_s:0,blocked:false},{time_s:1,blocked:true},{time_s:2,blocked:false}];
agent.segments=[{start:0,end:4,from:{floor:'B2',x:1,y:2},to:{floor:'B2',x:1,y:2},status:'walking'}];`);
assert.equal(read('agentVisual(agent,.5).color'),'#efb35d');
assert.equal(read('position(agent,1).status'),'visibility_blocked');
assert.equal(read('agentVisual(agent,1).label'),'시야 차단');
assert.equal(read('agentVisual(agent,1).color'),'#ff4e88');
assert.equal(read('position(agent,2).status'),'walking');
assert.equal(read('agentVisual(agent,2).color'),'#efb35d');
assert.equal(read('position(agent,.5).status'),'walking');
read('agent.incapacitated_at_s=1.25;');
assert.equal(read('position(agent,1.5).status'),'incapacitated');
read('agent.incapacitated_at_s=null;delete agent.visibility_blocked_history;');
assert.equal(read('position(agent,1.5).status'),'walking');
// 연기 색상은 실제 정지 여부와 분리한다. 현재 위치의 초당 FDS 관측만 사용한다.
read(`state.hazardByAgent=new Map([[agent.agent_id,[
  {time_s:0,floor:'B2',extinction_per_m:.99},
  {time_s:1,floor:'B2',extinction_per_m:1},
  {time_s:2,floor:'B2',extinction_per_m:1.5},
  {time_s:3,floor:'B2',extinction_per_m:.1}
]]]);
agent.segments=[{start:0,end:4,from:{floor:'B2',x:1,y:2},to:{floor:'B2',x:5,y:2},status:'walking'}];`);
assert.equal(read('agentVisual(agent,.5).color'),'#138eff');
assert.equal(read('agentVisual(agent,1).speed'),1);
assert.equal(read('agentVisual(agent,1).color'),'#ff4e88');
assert.equal(read('agentVisual(agent,1).label'),'시야 차단');
assert.equal(read('position(agent,1).status'),'walking');
assert.equal(read('agentVisual(agent,3).color'),'#138eff');
assert.equal(read('agentVisual(agent,.5).color'),'#138eff');
for(const status of ['pre_moving','waiting','queueing']){
  read(`agent.segments[0].status='${status}';`);
  assert.equal(read('agentVisual(agent,1).color'),'#ff4e88');
  assert.equal(read('position(agent,1).status'),status);
}
read(`agent.segments=[{start:0,end:4,from:{floor:'B2',x:1,y:2},to:{floor:'B1',x:1,y:4},status:'on_stairs',stair_id:'S'}];`);
assert.notEqual(read('agentVisual(agent,1).color'),'#ff4e88');
read(`state.hazardByAgent=new Map([[agent.agent_id,[{time_s:1,floor:'층간 연결',stair_id:'S',extinction_per_m:1.5}]]]);`);
assert.equal(read('agentVisual(agent,1).color'),'#ff4e88');
assert.notEqual(read('agentVisual(agent,2.1).color'),'#ff4e88');
read('agent.incapacitated_at_s=1;');
assert.equal(read('agentVisual(agent,1).color'),'#b91c1c');
read('agent.incapacitated_at_s=null;agent.evacuation_time_s=1;');
assert.equal(read('agentVisual(agent,1).color'),'#7c8797');
console.log('이동·대기·계단의 연기 시야 차단 색상과 물리 상태 분리 검증 통과');
read(`agent.agent_id=1;agent.evacuation_time_s=null;
agent.segments=[{start:0,end:10,from:{floor:'B2',x:1,y:2},to:{floor:'B2',x:11,y:2},status:'walking'}];
state.hazardByAgent=new Map();
state.data.agents=[agent];
state.data.movement_diagnostics={counterflow_recoveries:[
  {agent_id:1,time_s:1,end_s:2,action:'yielding',floor:'B2'},
  {agent_id:1,time_s:3,end_s:5,action:'overtaking',floor:'B2'}
]};
state.manoeuvresByAgent=prepareManoeuvres(state.data);`);
assert.equal(read('agentVisual(agent,1).color'),'#efb35d');
assert.equal(read('agentVisual(agent,1).label'),'양보 중');
assert.equal(read('agentVisual(agent,2).color'),'#138eff');
assert.equal(read('agentVisual(agent,3).color'),'#8050e4');
assert.equal(read('agentVisual(agent,3).label'),'추월 중');
assert.equal(read('agentVisual(agent,5).color'),'#138eff');
assert.equal(read('agentVisual(agent,1.5).label'),'양보 중');
read('agent.segments[0].to.x=1;');
assert.equal(read('agentVisual(agent,3).color'),'#8050e4');
assert.equal(read('agentVisual(agent,1).color'),'#efb35d');
read(`state.hazardByAgent=new Map([[1,[{time_s:3,floor:'B2',extinction_per_m:2}]]]);`);
assert.equal(read('agentVisual(agent,3).color'),'#ff4e88');
// 이전 결과는 앞사람 우회와 개찰구 양보를 구분하고 경로 변경 시 색상을 해제한다.
read(`state.hazardByAgent=new Map();agent.segments[0].to.x=11;
state.data.summary={total_time_s:10};
state.data.navigation_updates=[{agent_id:1,time_s:4}];
state.data.movement_diagnostics={counterflow_recoveries:[
  {agent_id:1,time_s:1,floor:'B2',cause:'visible_gate_approach_yield',waypoints:[[3,2]]},
  {agent_id:1,time_s:3,floor:'B2',cause:'visible_person_blocking_path',waypoints:[[20,2]]},
  {agent_id:1,time_s:5,floor:'B2',cause:'visible_wall_recovery',waypoints:[[20,2]]}
]};state.manoeuvresByAgent=prepareManoeuvres(state.data);`);
assert.equal(read('agentVisual(agent,1).label'),'양보 중');
assert.equal(read('agentVisual(agent,3).label'),'추월 중');
assert.equal(read('agentVisual(agent,4).color'),'#138eff');
assert.equal(read('agentVisual(agent,5).color'),'#138eff');
read(`agent.visibility_blocked_history=[{time_s:3.5,blocked:true}];
state.manoeuvresByAgent=prepareManoeuvres(state.data);`);
assert.equal(read('manoeuvreAt(agent,3.5)'),null);
read('delete agent.visibility_blocked_history;state.manoeuvresByAgent=new Map();');
console.log('양보·국소 우회 추월 색상과 실행 구간·되감기·종료 검증 통과');
vm.runInContext(fs.readFileSync('private_inspector.js','utf8'),context);
read(`state.data.scenario.stairs=[{id:'S',kind:'stairs'},{id:'G',kind:'fare_gate'}];
agent.personal_knowledge={local_ids:{S:'L1',G:'L2'}};`);
assert.equal(read('privateTarget("S",agent)'),'계단 1');
assert.equal(read('privateTarget("G",agent)'),'개찰구 2');
console.log('서행·시야 차단 색상 분리·시야 회복·개인 목표 이름 검증 통과');
