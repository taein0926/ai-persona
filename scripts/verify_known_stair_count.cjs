const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('private_inspector.js','utf8');
const start=source.indexOf('function appendKnowledgeRows('),end=source.indexOf('\nfunction drawPrivateInspector(',start);
assert(start>=0&&end>start);
const rows=[],context={state:{time:0,data:{scenario:{stairs:[
  {id:'S',kind:'stairs'},{id:'E',kind:'stairs'},{id:'G',kind:'fare_gate'},{id:'SECOND_GATE',kind:'fare_gate'}]}}},
  personalAt:()=>({known:new Map([['S',{to_floor:'B1'}],['E',{to_floor:'OUTSIDE'}],['G',{to_floor:'B1'}],['SECOND_GATE',{}]]),peopleCurrent:false}),
  appendSummaryRow:(_body,label,value)=>rows.push({label,value})};
vm.createContext(context);vm.runInContext(source.slice(start,end)+'\nappendKnowledgeRows({}, {}, {});',context);
assert.deepStrictEqual(rows[0],{label:'아는 계단',value:'2개 · 층 연결 2개'});
rows.length=0;
context.state.data.scenario={stairs:[{id:'S',kind:'stairs'},{id:'E',kind:'stairs'}],fare_gates:[{id:'G'},{id:'SECOND_GATE'}]};
vm.runInContext('appendKnowledgeRows({}, {}, {});',context);
assert.deepStrictEqual(rows[0],{label:'아는 계단',value:'2개 · 층 연결 2개'});
const labelStart=source.indexOf('function privateTarget('),labelEnd=source.indexOf('\nfunction agentTraitOverview(',labelStart);
vm.runInContext(source.slice(labelStart,labelEnd),context);
const agent={personal_knowledge:{local_ids:{G:'L3',S:'L1'}}};
assert.strictEqual(context.privateTarget('G',agent),'개찰구 3');
assert.strictEqual(context.privateTarget('S',agent),'계단 1');
console.log('아는 계단 집계 검사 통과: 개찰구 제외, 출구 계단 포함');
