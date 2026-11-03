// 실제 저장 결과만 조회하며 POST·추론·대피 계산을 실행하지 않는다.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const assert=require('node:assert/strict'),{chromium}=require('playwright');
const output=process.env.REPLAY_QA_DIR||'output/qa/saved_replay_20261102';
const artifact='output/gaehwasan/results.json';
function digest(file){return new Promise((resolve,reject)=>{
  const hash=crypto.createHash('sha256'),stream=fs.createReadStream(file);
  stream.on('data',chunk=>hash.update(chunk));stream.on('end',()=>resolve(hash.digest('hex')));stream.on('error',reject);
});}
(async()=>{
  fs.mkdirSync(output,{recursive:true});
  const before=await digest(artifact);
  const activation=JSON.parse(fs.readFileSync('output/gaehwasan/private_verification/activation.json','utf8').replace(/^\uFEFF/,''));
  const endTime=activation.summary.total_time_s;
  assert.equal(before,activation.artifact_sha256);
  const browser=await chromium.launch({headless:true,channel:'chrome',args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  try{
    const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[],posts=[],responses=[];
    page.on('pageerror',error=>errors.push(error.message));
    page.on('response',response=>{if(new URL(response.url()).pathname.startsWith('/api/'))responses.push({path:new URL(response.url()).pathname,status:response.status()});});
    await page.route('**/api/**',route=>{
      if(route.request().method()!=='GET'){posts.push(route.request().url());return route.abort();}
      return route.continue();
    });
    await page.goto('http://localhost:8010/',{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>typeof state!=='undefined'&&state.hasResults&&window.station3D?.ready,{timeout:120000});
    const snapshot=await page.evaluate(()=>({
      summary:state.data.summary,artifact:state.data.artifact_status,
      implementation_sha256:state.data.implementation_sha256,
      count:state.data.agents.length,segments:state.data.agents.reduce((n,a)=>n+a.segments.length,0),
      message:document.getElementById('message').textContent,
      max_time:Number(document.getElementById('time').max),
      play_disabled:document.getElementById('play').disabled,
      fire_enabled:state.data.fire?.enabled,
    }));
    assert.deepEqual(snapshot.summary,activation.summary);
    assert.equal(snapshot.implementation_sha256,activation.implementation_sha256);
    assert.equal(snapshot.artifact.mode,'stored_replay');assert(snapshot.artifact.snapshot_verified);
    assert.equal(snapshot.artifact.artifact_sha256,before);
    assert.equal(snapshot.artifact.automatic_recomputation,false);
    assert.equal(snapshot.count,100);assert(snapshot.segments>0);assert(snapshot.fire_enabled);
    assert.equal(snapshot.max_time,endTime);assert(!snapshot.play_disabled);
    assert(snapshot.message.includes('저장 결과 재생'));
    assert(responses.some(r=>r.path==='/api/gaehwasan'&&r.status===409));
    assert(responses.some(r=>r.path==='/api/gaehwasan/replay'&&r.status===200));
    assert(!responses.some(r=>r.path==='/api/gaehwasan/setup'));
    const readAt=async time=>{
      await page.locator('#time').evaluate((node,value)=>{node.value=value;node.dispatchEvent(new Event('input',{bubbles:true}));},time);
      return page.evaluate(()=>({time:state.time,clock:document.getElementById('clock').textContent,
        positions:state.data.agents.map(a=>({id:a.agent_id,status:position(a,state.time).status,pos:position(a,state.time).pos}))}));
    };
    const first=await readAt(0),middle=await readAt(Math.min(120,endTime/2)),end=await readAt(endTime),rewound=await readAt(0);
    assert.notDeepEqual(first.positions,middle.positions);
    assert.equal(end.positions.filter(p=>p.status==='evacuated').length,activation.summary.completed_count);
    assert.deepEqual(first,rewound);
    await page.getByRole('button',{name:'페르소나 2 선택',exact:true}).click();
    assert.equal(await page.evaluate(()=>state.selected),2);
    await page.getByRole('button',{name:'재생',exact:true}).click();
    await page.waitForFunction(()=>state.time>.2);
    await page.getByRole('button',{name:'일시정지',exact:true}).click();
    await readAt(120);
    await page.screenshot({path:path.join(output,'replay.png'),fullPage:true});
    assert.deepEqual(errors,[]);assert.deepEqual(posts,[]);
    assert.equal(await digest(artifact),before);
    const report={snapshot,responses,timeline_checked:[0,Math.min(120,endTime/2),endTime],rewind_matches:true,
      selection_and_playback:true,post_requests:posts,page_errors:errors,saved_result_unchanged:true};
    fs.writeFileSync(path.join(output,'browser-verification.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify(report,null,2));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
