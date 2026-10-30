const {chromium}=require('playwright');
const assert=require('assert'),fs=require('fs');
(async()=>{
  const browser=await chromium.launch({headless:true,channel:'chrome'});
  try{
    const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    for(const url of ['/mangwon','/api/mangwon'])assert((await page.request.get('http://localhost:8010'+url)).status()===404);
    assert((await page.request.get('http://localhost:8010/')).status()===200);
    await page.goto('http://localhost:8010/t-junction',{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>typeof state!=='undefined'&&!!state.data,{timeout:30000});
    const model=await page.evaluate(()=>({backend:state.data.backend,count:state.data.variants.C.length,summary:state.data.summaries.C}));
    assert(model.count===10&&model.summary.all_evacuated);
    const p=await page.evaluate(()=>screen(pointAt(state.data.variants.C[0],0))),bounds=await page.locator('#replay').boundingBox();
    await page.mouse.click(bounds.x+p.x*bounds.width/900,bounds.y+p.y*bounds.height/560);
    assert(await page.locator('#agentInspector').isVisible());
    await page.locator('#play').click();await page.waitForFunction(()=>state.time>.5);await page.locator('#play').click();
    assert.deepEqual(errors,[]);
    const report={gaehwasan_available:true,t_junction_available:true,mangwon_routes_removed:true,t_junction_model:model,t_junction_persona_selection:true,t_junction_playback:true,page_errors:errors};
    fs.writeFileSync('output/qa/simulation_views.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
