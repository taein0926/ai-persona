const {chromium}=require('playwright');
const fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto'),sharp=require('sharp');

(async()=>{
  const output='output/qa/floor_occlusion/after';
  fs.mkdirSync(output,{recursive:true});
  const hash=path=>crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
  const preserved=['configs/scenarios/gaehwasan.json','output/gaehwasan/results.json','output/fds/gaehwasan/derived/fields.npz'];
  const hashes=preserved.map(hash),browser=await chromium.launch({headless:true,channel:'chrome'});
  try{
    const page=await browser.newPage({viewport:{width:1600,height:1050}}),errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    await page.route('**/api/**',route=>{assert.equal(route.request().method(),'GET');return route.continue();});
    await page.route('**/gaehwasan3d.js*',route=>route.fulfill({contentType:'application/javascript',body:fs.readFileSync('gaehwasan3d.js','utf8')+`
      api.__floorOcclusion=(fid,hideLower=false,angle=0,follow=false)=>{
        stationView.follow=follow;api.update(data,0,state.selected);
        labelRoot.visible=false;selectionScene.visible=false;
        for(const n of root.children)n.visible=true;
        structureRoot.visible=true;for(const n of structureRoot.children)n.visible=true;
        floorDepthRoot.visible=true;for(const n of floorDepthRoot.children)n.visible=true;
        const level=elevation(fid),target=fid==='F1'?new THREE.Vector3(128,level,-25):new THREE.Vector3(120,level,-20);
        const offsets=[new THREE.Vector3(55,75,-70),new THREE.Vector3(-60,65,-70),new THREE.Vector3(0,100,.01)];
        controls.target.copy(target);camera.position.copy(target).add(offsets[angle]);camera.zoom=fid==='F1'?2.7:2.2;
        const minimum=controls.minPolarAngle;controls.minPolarAngle=0;camera.updateProjectionMatrix();controls.update();controls.minPolarAngle=minimum;
        root.updateMatrixWorld(true);structureRoot.updateMatrixWorld(true);
        const bodies=root.children.filter(n=>n.userData.role==='floor'&&n.userData.floorId===fid);
        const bounds=new THREE.Box3();for(const n of bodies)bounds.union(new THREE.Box3().setFromObject(n));
        const checker=new THREE.Raycaster();checker.params.Line.threshold=.015;
        const objects=[...root.children,...structureRoot.children],samples=[];
        for(let x=bounds.min.x+1;x<bounds.max.x-1;x+=2)for(let z=bounds.min.z+1;z<bounds.max.z-1;z+=2){
          const p=new THREE.Vector3(x,level,z).project(camera);
          if(Math.abs(p.x)>.97||Math.abs(p.y)>.97)continue;
          const px=Math.round((p.x+1)*host.clientWidth/2),py=Math.round((1-p.y)*host.clientHeight/2);
          // Sample pixel centers inside a clear floor patch, away from rail,
          // stair-opening and building silhouettes and their antialiasing.
          const clear=[[-1,-1],[1,-1],[-1,1],[1,1],[0,0]].every(([dx,dy])=>{
            checker.setFromCamera(new THREE.Vector2((px+dx+.5)*2/host.clientWidth-1,1-(py+dy+.5)*2/host.clientHeight),camera);
            const first=checker.intersectObjects(objects,true)[0];
            return first?.object.userData.role==='floor'&&first.object.userData.floorId===fid&&Math.abs(first.point.y-level)<.02;
          });
          if(clear)samples.push([px,py]);
        }
        if(hideLower){
          // Keep stair endpoints flush with the arrival floor and rails above
          // it: those are visible geometry, not objects below the slab.
          for(const group of [root,structureRoot,floorDepthRoot])for(const n of group.children)
            n.visible=new THREE.Box3().setFromObject(n).max.y>=level-.04;
        }
        render();
        return {samples,materials:bodies.map(n=>({opacity:n.material.opacity,transparent:n.material.transparent,depthWrite:n.material.depthWrite}))};
      };
      api.__occlusionProof=()=>{labelRoot.visible=true;selectionScene.visible=true;render();};
    `}));
    await page.goto('http://localhost:8010/',{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>window.station3D?.ready&&state.data?.scenario,{timeout:30000});
    const checks=[];
    for(const fid of ['F1','B1'])for(const angle of [0,1,2])for(const follow of [false,true]){
      const name=`${fid}_${angle}_${follow?'follow':'orbit'}`;
      const result=await page.evaluate(({fid,angle,follow})=>station3D.__floorOcclusion(fid,false,angle,follow),{fid,angle,follow});
      assert(result.samples.length>25);
      const full=await page.locator('#map3d').screenshot({path:`${output}/${name}_full.png`});
      await page.evaluate(({fid,angle,follow})=>station3D.__floorOcclusion(fid,true,angle,follow),{fid,angle,follow});
      const hidden=await page.locator('#map3d').screenshot({path:`${output}/${name}_lower_hidden.png`});
      const a=await sharp(full).removeAlpha().raw().toBuffer({resolveWithObject:true});
      const b=await sharp(hidden).removeAlpha().raw().toBuffer({resolveWithObject:true});
      const deltas=result.samples.map(([x,y])=>{
        const i=(Math.round(y)*a.info.width+Math.round(x))*3;
        return Math.max(...Array.from(a.data.subarray(i,i+3),(v,k)=>Math.abs(v-b.data[i+k])));
      });
      checks.push({fid,angle,follow,samples:deltas.length,max_delta:Math.max(...deltas),changed_samples:deltas.filter(d=>d>3).length,
        changed_pixels:result.samples.filter((p,i)=>deltas[i]>3),materials:result.materials});
    }
    await page.evaluate(()=>{station3D.__floorOcclusion('B1',false,1,false);station3D.__occlusionProof();});
    await page.locator('#map3d').screenshot({path:`${output}/attachment_angle.png`});
    const report={checks,page_errors:errors,preserved_data_unchanged:preserved.every((p,i)=>hash(p)===hashes[i])};
    fs.writeFileSync(`${output}/verification.json`,JSON.stringify(report,null,2));
    console.log(JSON.stringify({checks:checks.map(({materials,...rest})=>rest),page_errors:errors,preserved_data_unchanged:report.preserved_data_unchanged},null,2));
    assert.deepEqual(errors,[]);assert(report.preserved_data_unchanged);
    assert(checks.every(c=>c.materials.every(m=>m.opacity===1&&!m.transparent&&m.depthWrite)),'추적 시에도 바닥이 불투명해야 합니다.');
    assert(checks.every(c=>c.changed_samples===0),'아래층 선·형상이 상층 바닥에 비칩니다.');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
