const {chromium}=require('C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert=require('assert'),fs=require('fs');
(async()=>{
  const browser=await chromium.launch({headless:true,channel:'chrome',args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  try{
    const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto('http://localhost:8010/',{waitUntil:'domcontentloaded'});
    await page.locator('#play:not([disabled])').waitFor({timeout:180000});
    await page.waitForFunction(()=>window.station3D?.ready,{timeout:90000});
    const sample=await page.evaluate(()=>{
      const a=state.data.agents.find(a=>Object.values(a.experimental_profile.traits).some(t=>t.evidence_ids.length));
      const eid=Object.values(a.experimental_profile.traits).find(t=>t.evidence_ids.length).evidence_ids[0];
      const d=state.data.decisions.find(d=>d.used_memory_ids?.length);
      const stair=state.data.agents.find(a=>a.segments.some(s=>s.status==='on_stairs'));
      return {summary:state.data.summary,policy:state.data.agent_policy,performance:state.data.performance,
        cited:{id:a.agent_id,text:a.experimental_profile.excerpts.find(q=>q.id===eid).text},
        assigned:state.data.agents.find(a=>Object.values(a.experimental_profile.traits).some(t=>t.origin==='seeded_assignment')).agent_id,
        memory:{id:d.agent_id,time:d.time_s},stair:{id:stair.agent_id,time:stair.segments.find(s=>s.status==='on_stairs').start+.5},
        unfinished:state.data.agents.find(a=>a.evacuation_time_s==null).agent_id};
    });
    assert(sample.summary.count===100&&sample.summary.completed_count===85);
    assert(sample.performance.llm_call_count===0&&sample.performance.cache_request_count===696);
    assert(await page.locator('header,footer,#floorSpread,#stairCards,#overtakeLog,#sourceMeta,#fireStatus').count()===0);
    assert(await page.locator('#personaList button').count()===100);
    const left=await page.locator('.scene-panel').boundingBox(),middle=await page.locator('.map-panel').boundingBox(),right=await page.locator('#personaInspector').boundingBox();
    assert(left.x+left.width<=middle.x+1&&middle.x+middle.width<=right.x+1);
    const select=async id=>page.locator(`#personaList button[data-id="${id}"]`).click();
    const time=async value=>page.locator('#time').evaluate((n,t)=>{n.value=t;n.dispatchEvent(new Event('input',{bubbles:true}));},value);
    await select(sample.cited.id);await time(0);
    const text=await page.locator('#personDetails').innerText();
    for(const label of ['실험 성향','초기 친숙도','개인이 아는 정보'])assert(text.includes(label),label);
    assert(text.includes(sample.cited.text));
    await select(sample.assigned);assert((await page.locator('#personDetails').innerText()).includes('고정 시드 배정'));await select(sample.cited.id);
    assert(await page.locator('#decisionLog .decision-row').count()===0);
    assert(await page.evaluate(()=>[...personalAt(state.data.agents.find(a=>a.agent_id===state.selected),0,position(state.data.agents.find(a=>a.agent_id===state.selected),0)).known.values()].every(k=>!k.to_floor)));
    await select(sample.memory.id);await time(sample.memory.time+.01);
    const memory=await page.locator('#personDetails').innerText();
    for(const label of ['Qwen 선택','개인 지도 계산의 실행 목표','본인이 겪은 최근 경험','이번 판단에 사용','저장된 Qwen 응답'])assert(memory.includes(label),label);
    await select(sample.stair.id);await time(sample.stair.time);
    await page.locator('details[data-section="fds"] summary').click();
    const stair=await page.locator('#personDetails').innerText();
    for(const label of ['FDS 연기','가시거리 · 상한 100m','온도 · 연결층 관측값 보간','연기 계수 · 참고'])assert(stair.includes(label),label);
    await select(sample.unfinished);await time(600);
    const end=await page.evaluate(()=>{const a=state.data.agents.find(a=>a.agent_id===state.selected);return {status:position(a,600).status,pos:position(a,600).pos,expected:xy(a.segments.at(-1).to)};});
    assert(end.status!=='evacuated'&&Math.hypot(end.pos.x-end.expected.x,end.pos.y-end.expected.y)<1e-6);
    assert((await page.locator('#personDetails').innerText()).includes('미완료 원인'));
    assert((await page.locator('#metrics').innerText()).includes('미완료 15명'));
    await time(0);await page.locator('details[data-section="background"] summary').click();
    await page.locator('#play').click();await page.waitForFunction(()=>state.time>1,{timeout:30000});await page.locator('#play').click();
    assert(await page.locator('details[data-section="background"]').evaluate(n=>n.open));
    await page.locator('#zoomFit').click();assert.deepEqual(errors,[]);
    const report={count:100,layout_preserved:true,original_quotes_and_assignment_methods_visible:true,
      no_future_knowledge_at_start:true,personal_memory_and_model_execution_distinction_visible:true,
      stair_fds_measurements_separated_from_perception:true,unfinished_agent_remains_visible:true,
      playback_and_selection:true,expanded_background_survives_playback:true,removed_controls_stay_removed:true,
      page_errors:errors,screenshots_created:false};
    fs.mkdirSync('output/qa',{recursive:true});fs.writeFileSync('output/qa/private_editor_verification.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
