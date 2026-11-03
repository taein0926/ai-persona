const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs');

(async()=>{
  const setup=await(await fetch('http://localhost:8010/api/gaehwasan/setup')).json();
  const fixture={...setup,fire:{enabled:false},behavior:{mode:'off'},replay_duration_s:20,
    summary:{count:2,completed_count:0,total_time_s:20,exit_split:{}},
    navigation_updates:[],hazard_history:[],movement_diagnostics:{}};
  fixture.agents=setup.agents.slice(0,2).map((a,i)=>({...a,agent_id:i+1,
    evacuation_time_s:null,incapacitated_at_s:null,observations:[],people_observations:[],
    visibility_blocked_history:[],gas_exposure_history:[],
    segments:[{start:0,end:20,status:'walking',from:a.spawn,to:{...a.spawn,x:a.spawn.x+10}}]}));
  fixture.decisions=[
    ...[2.05,7.15,12.3].map(time_s=>({agent_id:1,time_s,priority:'risk',requested_movement:'continue',executed_target:'explore:none'})),
    {agent_id:2,time_s:5.35,priority:'time',requested_movement:'wait',executed_target:'explore:none'}
  ];
  const browser=await chromium.launch({headless:true,channel:'chrome',args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[],posts=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.route('**/api/gaehwasan',r=>{
      if(r.request().method()!=='GET')posts.push(r.request().method());
      return r.fulfill({json:fixture});
    });
    await page.goto('http://localhost:8010/',{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>typeof state!=='undefined'&&state.hasResults);
    await page.evaluate(()=>seekToTime(15));
    await page.locator('[data-section="history"] > summary').click();
    assert.equal(await page.locator('#decisionLog button').count(),3);
    await page.locator('#play').click();
    await page.locator('#decisionLog [data-decision-time="7.15"]').click();
    assert.deepEqual(await page.evaluate(()=>({time:state.time,playing:state.playing,selected:state.selected,frame:state.frame})),
      {time:7.15,playing:false,selected:1,frame:null});
    assert.equal(Number(await page.locator('#time').inputValue()),7.15);
    assert.equal(await page.locator('#clock').innerText(),'7.2초');
    assert.match(await page.locator('.agent-decision-row').innerText(),/7\.2초/);
    assert.equal(await page.locator('#decisionLog button').count(),2);
    assert(await page.locator('[data-section="history"]').evaluate(n=>n.open));
    assert.equal(await page.locator('#decisionLog [data-decision-time="7.15"]').evaluate(n=>n===document.activeElement),true);
    const expected=await page.evaluate(()=>xy({...state.data.agents[0].spawn,x:state.data.agents[0].spawn.x+7.15/2}));
    const actual=await page.evaluate(()=>position(state.data.agents[0],state.time).pos);
    assert(Math.hypot(actual.x-expected.x,actual.y-expected.y)<1e-8);
    const first=page.locator('#decisionLog [data-decision-time="2.05"]');
    await first.focus();await page.keyboard.press('Enter');
    assert.equal(await page.evaluate(()=>state.time),2.05);
    assert.equal(await page.locator('#decisionLog button').count(),1);
    await page.evaluate(()=>seekToTime(15));
    await page.locator('#personaList button[data-id="2"]').click();
    await page.locator('#decisionLog [data-decision-time="5.35"]').click();
    assert.equal(await page.evaluate(()=>state.time),5.35);
    assert.equal(await page.evaluate(()=>state.selected),2);
    await page.locator('#reset').click();
    assert.equal(await page.evaluate(()=>state.time),0);
    assert.equal(await page.locator('#decisionLog button').count(),0);
    await page.locator('#time').evaluate(n=>{n.value=10;n.dispatchEvent(new Event('input',{bubbles:true}));});
    assert.equal(await page.evaluate(()=>state.time),10);
    assert.equal(await page.evaluate(()=>state.playing),false);
    // 긴 기록 목록에서 새 판단만 따라가며 일반 갱신은 사용자가 읽던 위치를 보존한다.
    await page.evaluate(()=>{
      state.data.decisions=Array.from({length:35},(_,i)=>({agent_id:1,time_s:(i+1)/2,
        priority:'risk',requested_movement:'continue',executed_target:'explore:none'}));
      resetPrivatePlayback(state.data);select(1);seekToTime(2.5);
    });
    assert(await page.locator('[data-section="history"]').evaluate(n=>n.open));
    assert.equal(await page.locator('#decisionLog button').count(),5);
    assert(await page.locator('#decisionLog').evaluate(n=>n.scrollHeight===n.clientHeight));
    await page.evaluate(()=>seekToTime(3));
    assert.equal(await page.locator('#decisionLog button').count(),6);
    assert(await page.locator('#decisionLog').evaluate(n=>{
      const five=[...n.children].slice(0,5).reduce((sum,row)=>sum+row.getBoundingClientRect().height,0);
      return Math.abs(n.clientHeight-five)<1&&n.scrollHeight>n.clientHeight
        &&n.children[5].getBoundingClientRect().top>=n.getBoundingClientRect().bottom-1;
    }));
    for(const width of [1100,390,1440]){
      await page.setViewportSize({width,height:900});
      assert(await page.locator('#decisionLog').evaluate(n=>{
        const five=[...n.children].slice(0,5).reduce((sum,row)=>sum+row.getBoundingClientRect().height,0);
        return Math.abs(n.clientHeight-five)<1;
      }));
    }
    await page.evaluate(()=>seekToTime(10));
    assert.equal(await page.locator('#decisionLog button').count(),20);
    assert(await page.locator('#decisionLog').evaluate(n=>n.scrollHeight>n.clientHeight));
    await page.evaluate(()=>{state.playing=true;state.time=10.6;draw();});
    assert.equal(await page.locator('#decisionLog button').count(),21);
    assert(await page.locator('#decisionLog').evaluate(n=>Math.abs(n.scrollHeight-n.clientHeight-n.scrollTop)<2));
    await page.locator('#decisionLog').evaluate(n=>n.scrollTop=0);
    await page.evaluate(()=>{state.time=10.8;draw();});
    assert.equal(await page.locator('#decisionLog').evaluate(n=>n.scrollTop),0);
    await page.evaluate(()=>{state.time=11.1;draw();});
    assert.equal(await page.locator('#decisionLog button').count(),22);
    assert(await page.locator('#decisionLog').evaluate(n=>Math.abs(n.scrollHeight-n.clientHeight-n.scrollTop)<2));
    await page.evaluate(()=>stop());
    await page.locator('#decisionLog').evaluate(n=>n.scrollTop=60);
    await page.evaluate(()=>{state.time=11.3;draw();});
    assert.equal(await page.locator('#decisionLog').evaluate(n=>n.scrollTop),60);
    await page.locator('#decisionLog [data-decision-time="1"]').click();
    assert.equal(await page.evaluate(()=>state.time),1);
    assert.equal(await page.evaluate(()=>state.playing),false);
    assert.deepEqual(errors,[]);assert.deepEqual(posts,[]);
    const dir='output/qa/decision_seek_20261103';fs.mkdirSync(dir,{recursive:true});
    await page.screenshot({path:dir+'/screen.png'});
    fs.writeFileSync(dir+'/report.json',JSON.stringify({status:'통과',checks:['정확한 판단 시각 이동','재생 일시정지','키보드 선택','선택 승객 유지','위치·최근 판단·기록 갱신','기록 펼침 유지','되감기·초기화·시간바 호환','재생 중 새 판단 자동 스크롤','새 기록이 없을 때 스크롤 위치 유지','5개까지 표시·6개부터 내부 스크롤','화면 폭 변경 시 5개 높이 유지'],errors,posts},null,2));
    console.log('판단 기록 5개 표시·6개부터 스크롤·최신 기록 자동 이동·시각 선택 검증 통과');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
