// 저장된 결과만 사용한다. API의 대피 실행 요청은 전송하지 않는다.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),{performance}=require('perf_hooks');
const {chromium}=require('playwright');
const root='output/qa/runtime_view_optimization_20261029';
const data=JSON.parse(fs.readFileSync('output/gaehwasan/results.json','utf8'));
const normalize=value=>JSON.parse(JSON.stringify(value,(_,v)=>v instanceof Map?[...v]:v));
function context(before){
  const ctx=vm.createContext({state:{data,time:0},Map,WeakMap});
  vm.runInContext(fs.readFileSync(before?`${root}/before/private_inspector.js`:'private_inspector.js','utf8'),ctx);
  return (a,t,p)=>{ctx.state.time=t;ctx.a=a;ctx.t=t;ctx.p=p;return vm.runInContext('personalAt(a,t,p)',ctx);};
}
async function main(){
  const old=context(true),now=context(false);
  let checks=0;
  for(const a of data.agents){
    for(const t of [0,1,20,60,120,300,600,100,99.99,0,30,30]){
      for(const p of [{status:'walking',floor:'B1'},{status:'on_stairs',floor:'층간 연결'}]){
        assert.deepStrictEqual(normalize(now(a,t,p)),normalize(old(a,t,p)));checks++;
      }
    }
  }
  // 정렬되지 않은 기록과 같은 시각의 기록도 기존 배열 순서대로 처리한다.
  const unordered={...data.agents[0],observations:[...(data.agents[0].observations||[])].reverse(),people_observations:[...(data.agents[0].people_observations||[])].reverse()};
  for(const t of [100,600,30,0])assert.deepStrictEqual(normalize(now(unordered,t,{status:'walking',floor:'B2'})),normalize(old(unordered,t,{status:'walking',floor:'B2'})));
  const a=data.agents.reduce((a,b)=>a.observations.length>b.observations.length?a:b);
  const timing={before:[],after:[]};const times=Array.from({length:600},(_,i)=>i);
  for(let r=0;r<7;r++)for(const [name,fn] of r%2?[['after',now],['before',old]]:[['before',old],['after',now]]){
    fn(a,0,{status:'walking',floor:'B2'});
    const start=performance.now();for(const t of times)fn(a,t,{status:'walking',floor:'B2'});
    timing[name].push(performance.now()-start);
  }
  const picked=[data.agents[0],data.agents.find(a=>a.segments.some(s=>s.status==='on_stairs')),data.agents.find(a=>a.evacuation_time_s==null)].filter(Boolean);
  const agents=[...new Map(picked.map(a=>[a.agent_id,a])).values()],ids=new Set(agents.map(a=>a.agent_id));
  const fixture={...data,agents,fire:{enabled:false},summary:{...data.summary,count:agents.length,completed_count:agents.filter(a=>a.evacuation_time_s!=null).length}};
  for(const name of ['passages','decisions','navigation_updates','hazard_history'])fixture[name]=(data[name]||[]).filter(r=>ids.has(r.agent_id));
  const body=JSON.stringify(fixture),browser=await chromium.launch({headless:true,channel:'chrome',args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  const errors=[],snapshots=[];
  try{
    for(const before of [true,false]){
      const page=await browser.newPage({viewport:{width:1600,height:1000}});
      page.on('pageerror',e=>errors.push(e.message));
      await page.route('**/api/gaehwasan',route=>{
        assert.strictEqual(route.request().method(),'GET');return route.fulfill({contentType:'application/json',body});
      });
      if(before)for(const name of ['station_viewer.js','private_inspector.js','gaehwasan3d.js'])
        await page.route(`**/${name}*`,route=>route.fulfill({contentType:'text/javascript',body:fs.readFileSync(`${root}/before/${name}`,'utf8')}));
      await page.goto('http://localhost:8010/',{waitUntil:'domcontentloaded'});
      await page.waitForFunction(()=>!!state.data&&window.station3D?.ready,{timeout:90000});
      const captures=[];
      for(const agent of agents){
        const stair=agent.segments.find(s=>s.status==='on_stairs');
        for(const time of [0,40,150,stair?stair.start+.5:200,600,20,0]){
          captures.push(await page.evaluate(({id,time})=>{
            stop();state.time=time;select(id);draw();
            return {details:document.getElementById('personDetails').innerHTML,log:document.getElementById('decisionLog').innerHTML,
              stats:document.getElementById('liveStats').innerHTML,clock:document.getElementById('clock').textContent,
              positions:state.data.agents.map(a=>({id:a.agent_id,position:position(a,time),visual:agentVisual(a,time)}))};
          },{id:agent.agent_id,time}));
        }
      }
      await page.screenshot({path:`${root}/${before?'before':'after'}_view.png`});
      snapshots.push(captures);await page.close();
    }
    assert.deepStrictEqual(snapshots[0],snapshots[1]);assert.deepStrictEqual(errors,[]);
  }finally{await browser.close();}
  const median=xs=>[...xs].sort((a,b)=>a-b)[Math.floor(xs.length/2)];
  const b=median(timing.before),n=median(timing.after);
  const report={personal_state_exact_checks:checks+4,browser_cases:snapshots[0].length,dom_and_position_equal:true,page_errors:errors,
    historical_agent_id:a.agent_id,observations:a.observations.length,personal_at_calls_per_round:times.length,
    before_ms:b,after_ms:n,reduction_percent:100*(1-n/b),timings_ms:timing,
    method:'저장된 이전 결과 중 관측 기록이 가장 많은 한 사람의 상태 조회 600회·7회 교차 중앙값. 전체 화면 프레임 속도 측정은 아님.',engine_steps:0,new_inference_calls:0};
  fs.writeFileSync(`${root}/ui_verification.json`,JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
