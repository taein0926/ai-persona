const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs');

(async()=>{
  const setup=await(await fetch('http://localhost:8010/api/gaehwasan/setup')).json();
  const dir='output/qa/home_view_20261103';fs.mkdirSync(dir,{recursive:true});
  const browser=await chromium.launch({headless:true,channel:'chrome',args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  try{
    const page=await browser.newPage(),errors=[],posts=[],checks=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.route('**/api/gaehwasan',r=>{
      if(r.request().method()!=='GET')posts.push(r.request().method());
      return r.fulfill({json:setup});
    });
    for(const viewport of [{width:2425,height:1252},{width:1440,height:900},{width:1100,height:720}]){
      await page.setViewportSize(viewport);
      await page.goto('http://localhost:8010/',{waitUntil:'domcontentloaded'});
      await page.waitForFunction(()=>window.station3D?.ready&&state.data);
      const canvas=page.locator('#map3d canvas');
      const initial=await canvas.screenshot();
      assert.equal(await page.evaluate(()=>stationView.follow),false);
      await page.evaluate(()=>station3D.zoom(1.6));
      const rect=await canvas.boundingBox();
      await page.mouse.move(rect.x+rect.width*.6,rect.y+rect.height*.2);
      await page.mouse.down();
      await page.mouse.move(rect.x+rect.width*.75,rect.y+rect.height*.3,{steps:8});
      await page.mouse.up();
      assert(!initial.equals(await canvas.screenshot()));
      await page.locator('#zoomFit').click();
      assert(initial.equals(await canvas.screenshot()),'처음 시점 복귀 화면이 첫 로딩 화면과 다릅니다.');
      assert.equal(await page.evaluate(()=>stationView.follow),false);
      await page.evaluate(()=>station3D.follow());
      assert.equal(await page.evaluate(()=>stationView.follow),true);
      await page.locator('#zoomFit').click();
      assert(initial.equals(await canvas.screenshot()),'승객 추적 후 기준 시점 복귀 화면이 다릅니다.');
      await page.locator('.map-viewport').screenshot({path:`${dir}/home-${viewport.width}.png`});
      checks.push({...viewport,initialEqualsReset:true,followCleared:true});
    }
    assert.deepEqual(errors,[]);assert.deepEqual(posts,[]);
    fs.writeFileSync(dir+'/report.json',JSON.stringify({status:'통과',checks,errors,posts},null,2));
    console.log('화면 폭 3종의 첫 로딩·회전/확대 후 복귀·승객 추적 후 복귀 화면 일치 검증 통과');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
