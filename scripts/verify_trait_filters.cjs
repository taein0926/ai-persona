const fs=require('node:fs'),assert=require('node:assert/strict');
const {chromium}=require('playwright');
(async()=>{
  fs.mkdirSync('output/qa/trait_filters',{recursive:true});
  const setup=await (await fetch('http://localhost:8010/api/gaehwasan/setup')).json();
  const names=['crowd_avoidance','waiting_patience','risk_avoidance','exploration','social_following','overtaking_tendency'];
  const special=[
    {crowd_avoidance:'high',risk_avoidance:'low',social_following:'low'},
    {crowd_avoidance:'high',risk_avoidance:'high',social_following:'low'},
    {crowd_avoidance:'normal',risk_avoidance:'high',social_following:'low',overtaking_tendency:'high'},
    {crowd_avoidance:'low',risk_avoidance:'high',social_following:'high'},
    {crowd_avoidance:'high',risk_avoidance:'normal',social_following:'high'},
    {crowd_avoidance:'normal',risk_avoidance:'low',social_following:'normal',overtaking_tendency:'low'}];
  const speeds=[1.1,1.5,.9,1.3,1.,null];
  const fixture={...setup,fire:{enabled:false},behavior:{mode:'off'},decisions:[],navigation_updates:[],
    summary:{count:100,completed_count:0,total_time_s:300,exit_split:{}}};
  fixture.agents=setup.agents.map((a,i)=>({...a,agent_id:i+1,classification_pending:false,pre_movement_s:1,
    desired_speed_mps:i<6?speeds[i]:1.2,evacuation_time_s:null,incapacitated_at_s:null,
    initial_knowledge:{},observations:[],people_observations:[],gas_exposure_history:[],visibility_blocked_history:[],
    experimental_profile:i===99?null:{traits:Object.fromEntries(names.map(n=>[n,{level:special[i]?.[n]||(n==='crowd_avoidance'?'low':'normal'),assumption:'필터 검증용 등급입니다.'}]))},
    segments:[{start:0,end:300,status:'walking',from:a.spawn,to:a.spawn}]}));
  const browser=await chromium.launch({headless:true,channel:'chrome',args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[],posts=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.route('**/api/gaehwasan',r=>{if(r.request().method()!=='GET')posts.push(r.request().method());return r.fulfill({json:fixture});});
    await page.goto('http://localhost:8010/',{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>typeof state!=='undefined'&&state.hasResults);
    const ids=()=>page.locator('#personaList button:not([hidden])').evaluateAll(ns=>ns.map(n=>+n.dataset.id));
    const popup=page.locator('#agentTraitFilters'),filter=async(name,level)=>{
      if(!await popup.evaluate(n=>n.matches(':popover-open')))await page.locator('#toggleTraitFilters').click();
      await page.getByRole('button',{name:`${name} ${level}`,exact:true}).click();
    };
    assert.equal((await ids()).length,100);
    assert.equal(await page.locator('#activeTraitFilters').isVisible(),false);
    await page.locator('#toggleTraitFilters').click();
    assert.equal(await page.locator('#traitFilterOptions [aria-pressed="false"]').count(),18);
    assert.equal(await page.locator('#traitFilterOptions button').filter({hasText:'전체'}).count(),0);
    await filter('혼잡 회피','높음');assert.deepEqual(await ids(),[1,2,5]);
    assert.equal(await page.locator('#agentListCount').innerText(),'3명');
    assert.equal(await page.locator('#activeTraitFilters button').count(),1);
    await filter('혼잡 회피','높음');assert.equal((await ids()).length,100);
    assert.equal(await page.locator('#activeTraitFilters').isVisible(),false);
    await filter('혼잡 회피','높음');await filter('혼잡 회피','보통');
    assert.deepEqual(await ids(),[1,2,3,5,6]);
    await filter('위험 회피','높음');assert.deepEqual(await ids(),[2,3]);
    assert.equal(await page.locator('#activeTraitFilters button').count(),2);
    assert((await page.locator('#activeTraitFilters').innerText()).includes('보통·높음'));
    await page.keyboard.press('Escape');assert.equal(await popup.evaluate(n=>n.matches(':popover-open')),false);
    await page.selectOption('#agentSort','speed_asc');assert.deepEqual(await ids(),[3,2]);
    await page.selectOption('#agentSort','speed_desc');assert.deepEqual(await ids(),[2,3]);
    assert.equal(await page.evaluate(()=>state.dirty),false);
    await page.locator('#activeTraitFilters button').filter({hasText:'위험 회피'}).click();
    assert.deepEqual(await ids(),[2,1,5,3,6]);
    await page.locator('#activeTraitFilters button').filter({hasText:'혼잡 회피'}).click();
    assert.equal((await ids()).length,100);
    await filter('추월 적극성','높음');assert.deepEqual(await ids(),[3]);
    await page.evaluate(()=>{state.time=37;draw();});
    await page.keyboard.press('Escape');await page.locator('#play').click();
    await filter('혼잡 회피','보통');
    assert.equal(await page.evaluate(()=>state.playing),true);assert(await page.evaluate(()=>state.time>=37));
    await page.keyboard.press('Escape');await page.locator('#play').click();
    await page.evaluate(data=>showResults(data),fixture);
    assert.equal(await page.locator('#agentSort').inputValue(),'speed_desc');assert.deepEqual(await ids(),[3]);
    await filter('타인 추종성','높음');assert.deepEqual(await ids(),[]);
    assert(await page.locator('#agentFilterEmpty').isVisible());
    assert.equal(await page.locator('#agentListCount').innerText(),'0명');
    assert(!/\d+\s*명/.test(await popup.innerText()));
    for(const name of ['혼잡 회피','대기 인내','위험 회피','탐색 적극성','타인 추종성','추월 적극성'])
      assert.equal(await page.locator('#traitFilterOptions [role="group"]').filter({hasText:name}).count(),1);
    for(const width of [1440,1100,600,390]){
      await page.keyboard.press('Escape');await page.setViewportSize({width,height:800});
      await page.locator('#toggleTraitFilters').click();
      await page.waitForFunction(()=>{const n=document.getElementById('agentTraitFilters'),r=n.getBoundingClientRect();return r.left>=0&&r.right<=innerWidth&&r.top>=0&&r.bottom<=innerHeight;});
      assert(await popup.isVisible());
      assert.equal(await popup.locator('h3,p').count(),0);
      assert(await page.evaluate(()=>{
        const icon=document.getElementById('toggleTraitFilters').getBoundingClientRect();
        const sort=document.getElementById('agentSort').getBoundingClientRect();
        const title=document.querySelector('.agent-heading-label').getBoundingClientRect();
        const pop=document.getElementById('agentTraitFilters').getBoundingClientRect();
        return title.right<=icon.left&&icon.right<=sort.left&&Math.abs(icon.top-sort.top)<=1
          &&icon.width===24&&icon.height===24&&pop.width<=224&&pop.height<=190&&pop.bottom<=icon.top-4;
      }));
      assert(await page.locator('.scene-panel').evaluate(n=>n.scrollWidth<=n.clientWidth+1));
      if(width===1100)await page.screenshot({path:'output/qa/trait_filters/open.png',fullPage:true});
    }
    await page.keyboard.press('Escape');
    while(await page.locator('#activeTraitFilters button').count())await page.locator('#activeTraitFilters button').first().click();
    assert.equal((await ids()).length,100);
    assert.equal(await page.locator('#activeTraitFilters').isVisible(),false);
    assert.equal(await page.locator('#agentListCount').innerText(),'100명');
    assert.deepEqual(await page.evaluate(()=>state.data.agents.map(a=>a.agent_id)),Array.from({length:100},(_,i)=>i+1));
    assert.deepEqual(errors,[]);assert.deepEqual(posts,[]);
    const report={기본_무선택_전체:true,전체_버튼_없음:true,재클릭_해제:true,같은성향_OR_다른성향_AND:true,
      적용조건_태그와_해제:true,일치인원_숫자_갱신과_전체복원:true,보행속도_정렬_유지:true,재생과_재계산_필터_유지:true,
      빈결과_안내:true,팝업_화면내_표시:true,실제_계산요청:posts,화면_오류:errors};
    fs.writeFileSync('output/qa/trait_filters/report.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
