const {chromium}=require('playwright');
const fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto');

(async()=>{
  const output='output/qa/station_ready';fs.mkdirSync(output,{recursive:true});
  const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
  const preserved=JSON.parse(fs.readFileSync(`${output}/preserved_hashes.json`,'utf8').replace(/^\uFEFF/,''));
  const before=JSON.parse(fs.readFileSync(`${output}/config_before.json`,'utf8'));
  const current=JSON.parse(fs.readFileSync('configs/scenarios/gaehwasan.json','utf8'));
  for(const key of ['floors','nodes','stairs','walk_edges','fare_gates','vestibules','spawn_points','spawn_distribution','fire','train'])
    assert.deepEqual(current[key],before[key],key);
  const browser=await chromium.launch({headless:true,channel:'chrome'});
  try{
    const page=await browser.newPage({viewport:{width:1860,height:1050}}),errors=[],posts=[];
    page.on('pageerror',e=>errors.push(e.message));
    let mode='actual',fixture;
    await page.route('**/api/**',async route=>{
      const request=route.request();
      if(request.method()!=='GET'){
        posts.push(request.postDataJSON());
        return route.fulfill({status:422,contentType:'application/json',body:JSON.stringify({detail:'검증용 요청 차단: 실제 계산은 실행하지 않았습니다.'})});
      }
      if(new URL(request.url()).pathname==='/api/gaehwasan'){
        if(mode==='fixture')return route.fulfill({json:fixture});
        if(mode==='missing')return route.fulfill({status:404,json:{detail:{code:'station_result_missing',message:'계산 결과 없음'}}});
      }
      return route.continue();
    });
    const ready=()=>page.waitForFunction(()=>window.station3D?.ready&&state.data?.scenario&&document.getElementById('behaviorRun').disabled===false);
    const snapshot=()=>page.evaluate(()=>({has_results:state.hasResults,summary_present:!!state.data.summary,
      people:state.data.agents.length,segments:state.data.agents.reduce((n,a)=>n+a.segments.length,0),
      status:state.data.scenario.layout_status,enabled:state.data.scenario.simulation_enabled,
      play_disabled:document.getElementById('play').disabled,calculate_disabled:document.getElementById('behaviorRun').disabled,
      results_hidden:document.getElementById('metrics').hidden&&document.getElementById('liveStats').hidden,
      fire_mode:document.getElementById('fireMode').value,behavior_mode:document.getElementById('behaviorMode').value,
      fds_status:state.data.fire_precomputation?.status,clock:document.getElementById('clock').textContent,
      message:document.getElementById('message').textContent}));
    const rejected=await page.request.get('http://localhost:8010/api/gaehwasan');
    assert.equal(rejected.status(),409);assert.equal((await rejected.json()).detail.code,'station_result_outdated');
    await page.goto('http://localhost:8010/',{waitUntil:'domcontentloaded'});await ready();
    const initial=await snapshot();
    assert(!initial.has_results&&!initial.summary_present&&initial.results_hidden&&initial.play_disabled&&!initial.calculate_disabled);
    assert.equal(initial.segments,0);assert.equal(initial.people,100);assert.equal(initial.status,'ready');assert(initial.enabled);
    assert.equal(initial.fire_mode,'fds');assert.equal(initial.behavior_mode,'qwen');assert.equal(initial.fds_status,'ready');
    assert(!/미리보기|이전 구조|보류 중/.test(initial.message));assert.equal(posts.length,0);
    await page.getByRole('button',{name:'페르소나 2 선택',exact:true}).click();
    await page.getByRole('button',{name:'설정으로 계산',exact:true}).click();
    await page.getByRole('status').filter({hasText:'검증용 요청 차단'}).waitFor();
    assert.equal(posts.length,1);assert.equal(posts[0].agent_count,100);assert.equal(posts[0].fire_mode,'fds');assert.equal(posts[0].behavior_mode,'qwen');
    mode='missing';await page.reload();await ready();assert(!(await snapshot()).has_results);
    // A stationary, test-only replay checks the result UI without a solver call.
    fixture=await page.evaluate(()=>JSON.parse(JSON.stringify(state.data)));
    fixture.schema_version='station-test-fixture';fixture.replay_duration_s=1;
    fixture.summary={count:fixture.agents.length,completed_count:0,total_time_s:1,mean_wait_s:0,exit_split:{}};
    fixture.behavior={mode:'off'};
    for(const a of fixture.agents)a.segments=[{start:0,end:1,status:'pre_moving',from:a.spawn,to:a.spawn}];
    mode='fixture';await page.reload();await ready();
    assert((await snapshot()).has_results);assert(!(await snapshot()).results_hidden);
    await page.getByRole('button',{name:'재생',exact:true}).click();
    await page.waitForFunction(()=>state.time>0);
    mode='actual';await page.reload();await ready();
    const final=await snapshot();assert(!final.has_results&&final.segments===0&&final.results_hidden&&final.clock==='0.0초');
    assert.equal(posts.length,1);assert.deepEqual(errors,[]);
    assert(Object.entries(preserved).every(([p,digest])=>hash(p)===digest));
    await page.screenshot({path:`output/qa/station_ready/ready.png`,fullPage:true});
    const report={initial,final,old_result_status:rejected.status(),missing_result_supported:true,
      replay_fixture_supported:true,intercepted_test_posts:posts.length,forwarded_posts:0,
      physical_geometry_unchanged:true,preserved_files_unchanged:true,page_errors:errors};
    fs.writeFileSync(`${output}/verification.json`,JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
