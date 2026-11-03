const {chromium}=require('playwright');
const assert=require('assert'),fs=require('fs');
(async()=>{
  const browser=await chromium.launch({headless:true,channel:'chrome',args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  try{
    const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto('http://localhost:8010/',{waitUntil:'domcontentloaded'});
    await page.locator('#play:not([disabled])').waitFor({timeout:180000});
    await page.waitForFunction(()=>window.station3D?.ready,{timeout:90000});
    const sample=await page.evaluate(()=>{
      const a=state.data.agents.find(a=>a.experimental_profile);
      const d=state.data.decisions.find(d=>d.used_memory_ids?.length);
      const stair=state.data.agents.find(a=>a.segments.some(s=>s.status==='on_stairs'));
      return {summary:state.data.summary,policy:state.data.agent_policy,performance:state.data.performance,
        profile:{id:a.agent_id,summaries:Object.values(a.experimental_profile.traits).map(t=>t.assumption)},
        memory:{id:d.agent_id,time:d.time_s},stair:{id:stair.agent_id,time:stair.segments.find(s=>s.status==='on_stairs').start+.5},
        unfinished:state.data.agents.find(a=>a.evacuation_time_s==null).agent_id,
        completed:(()=>{const a=state.data.agents.find(a=>a.evacuation_time_s!=null);return {id:a.agent_id,time:a.evacuation_time_s};})()};
    });
    assert(sample.summary.count===100&&sample.summary.completed_count>0);
    assert(sample.performance.llm_call_count>=0&&sample.performance.cache_request_count>=0);
    assert(await page.locator('header,footer,#floorSpread,#stairCards,#overtakeLog,#sourceMeta,#fireStatus').count()===0);
    assert(await page.locator('#personaList button').count()===100);
    const left=await page.locator('.scene-panel').boundingBox(),middle=await page.locator('.map-panel').boundingBox(),right=await page.locator('#personaInspector').boundingBox();
    assert(left.x+left.width<=middle.x+1&&middle.x+middle.width<=right.x+1);
    const select=async id=>page.locator(`#personaList button[data-id="${id}"]`).click();
    const time=async value=>page.locator('#time').evaluate((n,t)=>{n.value=t;n.dispatchEvent(new Event('input',{bubbles:true}));},value);
    await select(sample.profile.id);await time(0);
    assert(await page.locator('details[data-section="details"]').count()===0);
    assert(await page.locator('.agent-summary').isVisible());
    assert.deepEqual(await page.locator('.agent-position-facts th').allTextContents(),['위치','목표','아는 계단','연기·열감','관측 시각','주변 사람','가스 노출','최근 판단']);
    assert(await page.locator('.agent-position-facts tbody > tr:first-child > *').count()===4);
    assert(await page.locator('.agent-position-facts .agent-knowledge-row').count()===4);
    assert(await page.locator('[data-section="knowledge"],.knowledge-landmarks,.agent-knowledge-detail').count()===0);
    assert.deepEqual(await page.locator('#liveStats > span').evaluateAll(nodes=>nodes.map(n=>n.firstChild.textContent.trim())),['대피 전','대피중','대피 완료','대피 불능']);
    assert(await page.locator('#liveStats b').evaluateAll(nodes=>nodes.reduce((sum,n)=>sum+parseInt(n.textContent),0))===100);
    assert(!(await page.locator('.reason').isVisible()));
    assert(!(await page.locator('.agent-completion').count()));
    assert(!(await page.locator('#personDetails').innerText()).includes('계산 종료 시 결과'));
    assert(await page.locator('.trait-grid > div').count()===6);
    const evidence=page.locator('#personDetails > details[data-section="classification"]');
    assert(await evidence.count()===1);
    assert(await evidence.locator(':scope > summary').innerText()==='분류 근거');
    assert(await page.locator('[data-section="traits"],[data-section="baseline"],.trait-heading,.agent-traits h4').count()===0);
    assert(!(await evidence.evaluate(n=>n.open)));
    const movement=await page.locator('.agent-movement-facts').innerText();
    for(const label of ['대피 반응 시간','평균 보행 속도'])assert(movement.includes(label),label);
    for(const label of ['대피 시작 지연','기본 목표 보행 속도'])assert(!movement.includes(label),label);
    const heading=await page.locator('.selection-heading').evaluate(n=>{
      const badge=n.querySelector('.motion-badge').getBoundingClientRect(),title=n.querySelector('h3').getBoundingClientRect();
      return {badgeBottom:badge.bottom,titleTop:title.top,badgeLeft:badge.left,titleLeft:title.left};
    });
    assert(heading.badgeBottom<=heading.titleTop&&Math.abs(heading.badgeLeft-heading.titleLeft)<1);
    assert(await page.locator('#personDetails').evaluate(n=>{
      const traits=n.querySelector('.agent-traits').getBoundingClientRect(),movement=n.querySelector('.agent-movement-facts').getBoundingClientRect(),evidence=n.querySelector('.classification-evidence'),summary=n.querySelector('.agent-summary');
      return traits.bottom<=movement.top&&movement.bottom<=evidence.getBoundingClientRect().top&&evidence.nextElementSibling===summary;
    }));
    await evidence.locator(':scope > summary').click();
    for(const summary of sample.profile.summaries)assert((await evidence.innerText()).includes(summary));
    assert.deepEqual(await evidence.locator('.classification-evidence-body > strong').allTextContents(),['혼잡 회피','대기 인내','위험 회피','탐색 적극성','타인 추종성','추월 적극성','대피 반응 시간','평균 보행 속도','역사 친숙도']);
    assert(await evidence.locator('.classification-evidence-body > p').count()===9);
    assert(await evidence.locator('.classification-evidence-body > :not(strong):not(p)').count()===0);
    assert(!(await page.locator('#personDetails').innerText()).includes('시뮬레이션 가정'));
    await time(.3);assert(await evidence.evaluate(n=>n.open));
    await evidence.locator(':scope > summary').click();
    await time(0);
    const text=await page.locator('#personDetails').innerText();
    for(const label of ['역사 친숙도','아는 계단','연기·열감','관측 시각','주변 사람'])assert(text.includes(label),label);
    assert(!text.includes('실험 성향'));
    assert(await page.locator('#decisionLog .decision-row').count()===0);
    assert(await page.evaluate(()=>{
      const a=state.data.agents.find(a=>a.agent_id===state.selected),known=personalAt(a,0,position(a,0)).known;
      return [...known].every(([id,k])=>!k.to_floor||k.to_floor===a.initial_knowledge?.[id]?.to_floor);
    }));
    await select(sample.memory.id);await time(sample.memory.time+.01);
    assert((await page.locator('.agent-decision-row').innerText()).includes(`${sample.memory.time.toFixed(1)}초`));
    await page.locator('details[data-section="history"] > summary').click();
    assert(await page.locator('#decisionLog .decision-row').count()>0);
    await page.locator('details[data-section="history"] > summary').click();
    await select(sample.stair.id);await time(sample.stair.time);
    assert(await page.locator('details[data-section="fds"]').count()===0);
    await select(sample.unfinished);await time(600);
    const end=await page.evaluate(()=>{const a=state.data.agents.find(a=>a.agent_id===state.selected);return {status:position(a,600).status,pos:position(a,600).pos,expected:xy(a.segments.at(-1).to)};});
    assert(end.status!=='evacuated'&&Math.hypot(end.pos.x-end.expected.x,end.pos.y-end.expected.y)<1e-6);
    assert(await page.locator('#metrics .result-card').filter({hasText:'미완료'}).locator('strong').innerText()===String(sample.summary.count-sample.summary.completed_count));
    await time(0);await page.locator('details[data-section="background"] summary').click();
    await page.locator('#play').click();await page.waitForFunction(()=>state.time>1,{timeout:30000});await page.locator('#play').click();
    assert(await page.locator('details[data-section="background"]').evaluate(n=>n.open));
    await select(sample.completed.id);await time(sample.completed.time+.3);
    assert((await page.locator('.agent-completion').innerText()).includes(`${sample.completed.time.toFixed(1)}초`));
    assert(await page.locator('.agent-decision-row').count()===1);
    assert(await page.locator('.agent-gas-row').count()===1);
    await time(0);
    assert(await page.locator('.agent-completion').count()===0);
    assert(await page.locator('.agent-decision-row').isVisible());
    await page.locator('details[data-section="background"] > summary').click();
    await select(sample.memory.id);await time(sample.memory.time+.01);
    assert(!(await page.locator('#decisionLog').isVisible()));
    const fits=await page.locator('.inspector-content').evaluate(n=>n.scrollHeight<=n.clientHeight);
    assert(fits);
    await page.locator('#personaInspector').screenshot({path:'output/qa/compact_inspector.png'});
    await page.setViewportSize({width:1100,height:720});
    assert(await page.locator('.inspector-content').evaluate(n=>n.scrollHeight<=n.clientHeight));
    const titleTop=await page.locator('#personTitle').evaluate(n=>n.getBoundingClientRect().top);
    await page.locator('#selectedMotionStatus').evaluate(n=>n.textContent='계단 위 정지·서행');
    assert(await page.locator('#personTitle').evaluate(n=>n.getBoundingClientRect().top)===titleTop);
    await page.setViewportSize({width:600,height:800});
    assert(await page.locator('.inspector-content').evaluate(n=>getComputedStyle(n).columnCount==='auto'));
    for(const width of [1600,1100,600]){
      await page.setViewportSize({width,height:800});
      const overlap=await page.evaluate(()=>{
        const failures=[],titleNode=$('personTitle'),ageNode=$('personDescription');
        const previousTitle=titleNode.textContent,previousAge=ageNode.textContent;
        // 이름 길이에 따른 배치만 검사하며, 지도 전체의 재계산은 생략한다.
        for(const a of state.data.agents){
          titleNode.textContent=a.persona.occupation;ageNode.textContent=`${a.persona.age}세`;
          const title=titleNode.getBoundingClientRect(),age=ageNode.getBoundingClientRect(),row=document.querySelector('.agent-identity').getBoundingClientRect();
          const textRange=document.createRange();textRange.selectNodeContents(titleNode);
          if(title.right>age.left||Math.abs(age.right-row.right)>1||[...textRange.getClientRects()].some(r=>r.right>age.left))failures.push(a.agent_id);
        }
        titleNode.textContent=previousTitle;ageNode.textContent=previousAge;
        return failures;
      });
      assert.deepEqual(overlap,[],`${width}px 직업·나이 겹침`);
    }
    await page.locator('#zoomFit').click();assert.deepEqual(errors,[]);
    const report={count:100,layout_preserved:true,unified_classification_evidence_nine_summaries:true,
      no_future_knowledge_at_start:true,decision_history_visible_on_demand:true,
      researcher_fds_details_removed:true,unfinished_agent_remains_visible:true,
      playback_and_selection:true,expanded_background_survives_playback:true,removed_controls_stay_removed:true,
      compact_summary_visible:true,evidence_collapsed_by_default:true,completion_follows_playback:true,
      status_above_occupation:true,trait_heading_removed:true,evidence_above_location:true,movement_terms_updated:true,
      occupation_age_no_overlap_for_all_agents_at_three_widths:true,
      compact_summary_fits_without_scrolling:true,page_errors:errors,screenshots_created:true};
    fs.mkdirSync('output/qa',{recursive:true});fs.writeFileSync('output/qa/private_editor_verification.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
