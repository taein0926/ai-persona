const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto'),sharp=require('sharp');

(async()=>{
  const output='output/qa/station_layout';fs.mkdirSync(output,{recursive:true});
  const resultPath='output/gaehwasan/results.json';
  const hash=()=>crypto.createHash('sha256').update(fs.readFileSync(resultPath)).digest('hex');
  const before=hash(),browser=await chromium.launch({headless:true,channel:'chrome'});
  try{
    const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.route('**/api/**',route=>{assert.equal(route.request().method(),'GET');return route.continue();});
    // Test-only camera hook; the production geometry and renderer stay intact.
    await page.route('**/gaehwasan3d.js*',route=>route.fulfill({contentType:'application/javascript',
      body:fs.readFileSync('gaehwasan3d.js','utf8')+`\napi.__inspectStair=(isolated=false)=>{
        const s=data.scenario.stairs.find(s=>s.id==='H_E'),p=s.landings[0].polygon;
        const x=p.reduce((n,q)=>n+q[0]/p.length,0),y=p.reduce((n,q)=>n+q[1]/p.length,0);
        controls.target.set(x,15,-y);camera.position.copy(controls.target).add(isolated?new THREE.Vector3(0,90,40):new THREE.Vector3(26,60,28));
        for(const child of root.children)if(child.userData.role==='floor'&&child.userData.floorId==='F1')child.visible=false;
        for(const child of floorDepthRoot.children)if(child.userData.floorId==='F1')child.visible=false;
        structureRoot.visible=!isolated;
        floorDepthRoot.visible=!isolated;
        if(isolated)for(const child of root.children)child.visible=(child.material===stairMaterial||child.userData.role==='tread')&&(!child.userData.stairId||child.userData.stairId==='H_E');
        camera.zoom=6;camera.updateProjectionMatrix();controls.update();render();
        const counts={};root.traverse(n=>{if(n.userData.stairId){const key=n.userData.stairId+':'+n.userData.role;counts[key]=(counts[key]||0)+1;}});
        api.__stairBounds=root.children.filter(n=>n.userData.stairId==='H_E').map(n=>({role:n.userData.role,position:n.position.toArray(),bounds:new THREE.Box3().setFromObject(n).getSize(new THREE.Vector3()).toArray()}));
        api.__inspectB1=()=>{
          const keep=floors.find(f=>f.id==='B1').mats;
          for(const child of root.children)child.visible=keep.includes(child.material)||!!child.userData.stairId;
          for(const mat of keep)mat.opacity=1;
          structureRoot.visible=true;for(const child of structureRoot.children)child.visible=child.userData.floorId==='B1';
          floorDepthRoot.visible=true;for(const child of floorDepthRoot.children)child.visible=child.userData.floorId==='B1';
          controls.target.set(120,10,-20);camera.position.set(120,110,-19.99);camera.zoom=2.3;
          const angle=controls.minPolarAngle;controls.minPolarAngle=0;
          camera.updateProjectionMatrix();controls.update();controls.minPolarAngle=angle;render();
        };
        api.__inspectF1=(angled=false)=>{
          const keep=floors.find(f=>f.id==='F1').mats;
          for(const child of root.children)child.visible=keep.includes(child.material)||child.userData.stairId==='H_E';
          for(const mat of keep)mat.opacity=1;
          structureRoot.visible=true;for(const child of structureRoot.children)child.visible=child.userData.floorId==='F1';
          floorDepthRoot.visible=true;for(const child of floorDepthRoot.children)child.visible=child.userData.floorId==='F1';
          controls.target.set(126,20,-25);camera.position.copy(controls.target).add(angled?new THREE.Vector3(24,65,36):new THREE.Vector3(0,100,.01));
          camera.zoom=angled?2.8:3.4;
          const angle=controls.minPolarAngle;controls.minPolarAngle=0;
          camera.updateProjectionMatrix();controls.update();controls.minPolarAngle=angle;render();
          return {buildings:structureRoot.children.filter(n=>n.visible&&n.userData.kind==='building').length,
            holes:data.scenario.display3d_openings[separation.toFixed(1)].F1.reduce((n,p)=>n+p.holes.length,0)};
        };
        api.__overlapCheck=(hideLower=false)=>{
          for(const child of root.children)child.visible=!(hideLower&&child.userData.floorId==='B2'&&child.userData.role==='floor');
          structureRoot.visible=true;for(const child of structureRoot.children)child.visible=true;
          floorDepthRoot.visible=true;for(const child of floorDepthRoot.children)child.visible=!(hideLower&&child.userData.floorId==='B2');
          for(const floor of floors)for(const mat of floor.mats)mat.opacity=1;
          controls.target.set(117,10,-22);camera.position.copy(controls.target).add(new THREE.Vector3(-80,34,16));
          camera.zoom=3.2;camera.updateProjectionMatrix();controls.update();render();
          const gates=structureRoot.children.filter(n=>n.userData.floorId==='B1'&&n.userData.kind==='gate');
          const samples=gates.map(n=>{const b=new THREE.Box3().setFromObject(n),p=b.getCenter(new THREE.Vector3());p.y=b.max.y+.01;p.project(camera);return [(p.x+1)*host.clientWidth/2,(1-p.y)*host.clientHeight/2];});
          return {samples,gates:gates.length,walls:structureRoot.children.filter(n=>n.userData.kind==='wall').length,
            opaque:gates.every(n=>!n.material.transparent&&n.material.depthWrite&&n.material.depthTest)};
        };
        api.__edgeMaskCheck=(showLower=true)=>{
          for(const child of root.children)child.visible=child.userData.role==='floor'&&child.userData.floorId!=='F1';
          for(const floor of floors)for(const mat of floor.mats)mat.opacity=1;
          structureRoot.visible=true;
          for(const child of structureRoot.children)child.visible=child.userData.kind==='floor_edge'&&child.userData.floorId!=='F1'&&(showLower||child.userData.floorId!=='B2');
          floorDepthRoot.visible=true;for(const child of floorDepthRoot.children)child.visible=child.userData.floorId!=='F1';
          controls.target.set(120,10,-20);camera.position.set(120,110,-19.99);camera.zoom=2.3;
          const angle=controls.minPolarAngle;controls.minPolarAngle=0;camera.updateProjectionMatrix();controls.update();controls.minPolarAngle=angle;render();
          const inside=(p,ring)=>{let yes=false;for(let i=0,j=ring.length-1;i<ring.length;j=i++){
            const a=ring[i],b=ring[j];if((a[1]>p[1])!==(b[1]>p[1])&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])yes=!yes;}return yes;};
          const slab=data.scenario.display3d_openings[separation.toFixed(1)].B1,samples=[];
          for(const edge of structureRoot.children.filter(n=>n.userData.kind==='floor_edge'&&n.userData.floorId==='B2')){
            const a=edge.geometry.attributes.position;
            for(let i=1;i<a.count;i++)for(const t of [.25,.5,.75]){
              const p=new THREE.Vector3().fromBufferAttribute(a,i-1).lerp(new THREE.Vector3().fromBufferAttribute(a,i),t);
              const xy=[p.x,-p.z];if(!slab.some(poly=>inside(xy,poly.outer)&&!poly.holes.some(h=>inside(xy,h))))continue;
              p.project(camera);const x=(p.x+1)*host.clientWidth/2,y=(1-p.y)*host.clientHeight/2;
              if(x>1&&y>1&&x<host.clientWidth-2&&y<host.clientHeight-2)samples.push([x,y]);
            }
          }
          return samples;
        };
        api.__labelMaskCheck=(occluded=false,angle=0)=>{
          if(api.__labelOccluder)structureScene.remove(api.__labelOccluder);
          controls.target.set(90,10,-24);camera.position.copy(controls.target).add(angle?new THREE.Vector3(-100,100,80):new THREE.Vector3(100,100,80));
          camera.zoom=.8;camera.updateProjectionMatrix();controls.update();
          if(occluded){
            const screen=new THREE.Mesh(new THREE.PlaneGeometry(1000,1000),new THREE.MeshBasicMaterial({color:0x000000,side:THREE.DoubleSide}));
            screen.quaternion.copy(camera.quaternion);screen.position.copy(camera.position).lerp(controls.target,.1);
            structureScene.add(screen);api.__labelOccluder=screen;
          }
          render();
          return labelRoot.children.map(n=>{const p=n.position.clone().project(camera);return {text:n.userData.labelText,
            point:[(p.x+1)*host.clientWidth/2,(1-p.y)*host.clientHeight/2]};});
        };
        api.__upperFloorMask=(enabled=true)=>{
          for(const child of root.children)child.visible=enabled||child.userData.role!=='floor'||child.userData.floorId!=='F1';
          structureRoot.visible=true;for(const child of structureRoot.children)child.visible=true;
          floorDepthRoot.visible=true;for(const child of floorDepthRoot.children)child.visible=enabled||child.userData.floorId!=='F1';
          const room=structureRoot.children.find(n=>n.userData.floorId==='B1'&&n.userData.label==='북측·동측·남측 통합 시설 구역');
          const bounds=new THREE.Box3().setFromObject(room);
          // A concave building's bounding-box center is in the open concourse.
          const point=new THREE.Vector3((860-382)*4/15.12,bounds.max.y+.01,-((347-215)*4/15.12));
          controls.target.set(point.x,10,point.z);camera.position.set(point.x,110,point.z+.01);camera.zoom=4;
          const angle=controls.minPolarAngle;controls.minPolarAngle=0;camera.updateProjectionMatrix();controls.update();controls.minPolarAngle=angle;render();
          point.project(camera);return [(point.x+1)*host.clientWidth/2,(1-point.y)*host.clientHeight/2];
        };
        return counts;
      };` }));
    const documentResponse=await page.goto('http://localhost:8010/',{waitUntil:'domcontentloaded'});
    assert.match(documentResponse.headers()['cache-control'],/no-store/);
    await page.waitForFunction(()=>typeof state!=='undefined'&&state.data?.scenario&&window.station3D?.ready,{timeout:30000});
    const report=await page.evaluate(()=>{
      const c=state.data.scenario,s=c.stairs.find(s=>s.id==='H_E');
      const platform=c.stairs.filter(s=>c.nodes[s.from_node].floor==='B2');
      return {has_results:state.hasResults,platform_lengths:platform.map(s=>s.length_m),
        b1_connections:c.stairs.filter(s=>c.nodes[s.from_node].floor==='B1').map(s=>s.id),
        flights:s.flights.map(f=>({role:f.role,width_m:f.width_m,path:f.path_xyz})),
        landings:s.landings,center_handrail:s.center_handrail_xyz,
        b1_voids:c.stairs.filter(s=>c.nodes[s.from_node].floor==='B2'&&s.floor_opening_polygon).length,
        gates:c.fare_gates.length,gate_body_heights:c.floors.find(f=>f.id==='B1').obstacles.filter(o=>o.label==='개찰구 본체').map(o=>o.height_m),
        connections:c.jupedsim_connections,calculation_enabled:c.simulation_enabled,
        spawn_counts:c.spawn_points.reduce((counts,p)=>(counts[p.area]=(counts[p.area]||0)+1,counts),{}),
        exits:c.stairs.filter(s=>s.kind==='exit').map(s=>({id:s.id,form:s.exit_form,render_mode:s.render_mode,path:s.path_xyz})),
        vestibules:c.vestibules.map(v=>v.exit_id),
        play_disabled:document.getElementById('play').disabled,calculation_disabled:document.getElementById('behaviorRun').disabled,
        results_hidden:['metrics','liveStats'].every(id=>getComputedStyle(document.getElementById(id)).display==='none'),
        notice:document.getElementById('message').textContent};
    });
    assert.equal(report.has_results,false);assert.equal(new Set(report.platform_lengths).size,1);
    const scripts=await page.evaluate(()=>Array.from(document.scripts,s=>s.src).filter(Boolean));
    assert(scripts.every(src=>src.includes('v=20261029-layout-14')));
    const asset=await page.request.get('http://localhost:8010/station_viewer.js?v=20261029-layout-14');
    assert.match(asset.headers()['cache-control'],/no-store/);
    assert.deepEqual(report.b1_connections,['H_E']);
    assert.equal(report.flights.filter(f=>f.role==='lower').length,2);
    assert.equal(report.flights.filter(f=>f.role==='upper').length,2);
    assert(report.flights.every(f=>f.width_m===report.flights[0].width_m));
    assert.equal(report.b1_voids,4);assert(report.center_handrail.length>1&&report.landings.length===1);
    assert(report.gates>0&&report.gate_body_heights.every(h=>h===1));
    assert.equal(report.calculation_enabled,true);assert(report.connections.enabled);
    assert.deepEqual(report.spawn_counts,{A:80,B:20});
    assert.deepEqual(report.vestibules,['E_1','E_2']);
    assert.deepEqual(report.exits.map(s=>s.form),['bent_stair','vestibule_passage']);
    assert(report.exits.every(s=>s.render_mode==='floor_only'));
    assert(report.play_disabled&&!report.calculation_disabled&&!report.results_hidden);assert.deepEqual(errors,[]);
    await page.screenshot({path:`${output}/overview.png`,fullPage:true});
    const rendered=await page.evaluate(()=>station3D.__inspectStair());
    assert(rendered['H_E:lower']>0&&rendered['H_E:upper']>0&&rendered['H_E:landing']===1);
    assert(!Object.keys(rendered).some(key=>key.startsWith('E_1:')||key.startsWith('E_2:')));
    await page.screenshot({path:`${output}/b1_stair_detail.png`,fullPage:true});
    await page.evaluate(()=>station3D.__inspectStair(true));
    fs.writeFileSync(`${output}/render_bounds.json`,JSON.stringify(await page.evaluate(()=>station3D.__stairBounds),null,2));
    await page.locator('#map3d').screenshot({path:`${output}/b1_stair_connection.png`});
    await page.evaluate(()=>station3D.__inspectB1());
    await page.locator('#map3d').screenshot({path:`${output}/b1_floor_plan.png`});
    const f1=await page.evaluate(()=>station3D.__inspectF1());
    assert.equal(f1.buildings,3);assert.equal(f1.holes,1);
    await page.locator('#map3d').screenshot({path:`${output}/f1_floor_plan.png`});
    await page.evaluate(()=>station3D.__inspectF1(true));
    await page.locator('#map3d').screenshot({path:`${output}/f1_detail.png`});
    const overlap=await page.evaluate(()=>station3D.__overlapCheck(false));
    assert.equal(overlap.gates,report.gates+1);assert(overlap.opaque&&overlap.walls>0);
    await page.locator('#map3d').screenshot({path:`${output}/overlap_all_floors.png`});
    await page.evaluate(()=>station3D.__overlapCheck(true));
    await page.locator('#map3d').screenshot({path:`${output}/overlap_without_b2_slab.png`});
    const withFloor=await sharp(`${output}/overlap_all_floors.png`).removeAlpha().raw().toBuffer({resolveWithObject:true});
    const withoutFloor=await sharp(`${output}/overlap_without_b2_slab.png`).removeAlpha().raw().toBuffer({resolveWithObject:true});
    const samples=overlap.samples.map(([x,y])=>{
      x=Math.round(x);y=Math.round(y);assert(x>=0&&y>=0&&x<withFloor.info.width&&y<withFloor.info.height);
      const index=(y*withFloor.info.width+x)*3,a=Array.from(withFloor.data.subarray(index,index+3)),b=Array.from(withoutFloor.data.subarray(index,index+3));
      return {pixel:[x,y],color:a,delta:Math.max(...a.map((v,i)=>Math.abs(v-b[i]))),gate_visible:a[1]>a[0]+8&&a[2]>a[0]+8};
    });
    assert(samples.every(s=>s.delta<=3&&s.gate_visible),JSON.stringify(samples));
    const hiddenPoint=await page.evaluate(()=>station3D.__upperFloorMask(true));
    await page.locator('#map3d').screenshot({path:`${output}/upper_floor_masks_lower_room.png`});
    await page.evaluate(()=>station3D.__upperFloorMask(false));
    await page.locator('#map3d').screenshot({path:`${output}/without_upper_floor_mask.png`});
    const masked=await sharp(`${output}/upper_floor_masks_lower_room.png`).removeAlpha().raw().toBuffer({resolveWithObject:true});
    const unmasked=await sharp(`${output}/without_upper_floor_mask.png`).removeAlpha().raw().toBuffer({resolveWithObject:true});
    const at=(Math.round(hiddenPoint[1])*masked.info.width+Math.round(hiddenPoint[0]))*3;
    const maskDelta=Math.max(...Array.from(masked.data.subarray(at,at+3),(v,i)=>Math.abs(v-unmasked.data[at+i])));
    assert(maskDelta>5,`상층 바닥의 가림이 적용되지 않았습니다: ${maskDelta}`);
    const edgeSamples=await page.evaluate(()=>station3D.__edgeMaskCheck(true));
    assert(edgeSamples.length>5);
    await page.locator('#map3d').screenshot({path:`${output}/b2_edges_masked_by_b1.png`});
    await page.evaluate(()=>station3D.__edgeMaskCheck(false));
    await page.locator('#map3d').screenshot({path:`${output}/without_b2_edges.png`});
    const edgeA=await sharp(`${output}/b2_edges_masked_by_b1.png`).removeAlpha().raw().toBuffer({resolveWithObject:true});
    const edgeB=await sharp(`${output}/without_b2_edges.png`).removeAlpha().raw().toBuffer({resolveWithObject:true});
    const edgeDeltas=edgeSamples.map(([x,y])=>{const index=(Math.round(y)*edgeA.info.width+Math.round(x))*3;return Math.max(...Array.from(edgeA.data.subarray(index,index+3),(v,k)=>Math.abs(v-edgeB.data[index+k])));});
    assert(edgeDeltas.every(delta=>delta<=3),JSON.stringify(edgeDeltas));
    const labelChecks=[];
    for(const angle of [0,1]){
      const labels=await page.evaluate(angle=>station3D.__labelMaskCheck(false,angle),angle);
      assert.equal(labels.length,6);assert(labels.every(l=>!/[①②⑤]/.test(l.text)));
      assert(labels.some(l=>l.text==='1번 출구')&&labels.some(l=>l.text==='2번 출구')&&labels.some(l=>l.text==='5호선 정차 열차'));
      await page.locator('#map3d').screenshot({path:`${output}/labels_clear_${angle}.png`});
      await page.evaluate(angle=>station3D.__labelMaskCheck(true,angle),angle);
      await page.locator('#map3d').screenshot({path:`${output}/labels_over_structures_${angle}.png`});
      const clear=await sharp(`${output}/labels_clear_${angle}.png`).removeAlpha().raw().toBuffer({resolveWithObject:true});
      const blocked=await sharp(`${output}/labels_over_structures_${angle}.png`).removeAlpha().raw().toBuffer({resolveWithObject:true});
      for(const label of labels){
        const [x,y]=label.point.map(Math.round);assert(x>3&&y>3&&x<clear.info.width-3&&y<clear.info.height-3);
        let delta=0;for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){
          const index=((y+dy)*clear.info.width+x+dx)*3;
          delta=Math.max(delta,...Array.from(clear.data.subarray(index,index+3),(v,k)=>Math.abs(v-blocked.data[index+k])));
        }
        assert(delta<=3,JSON.stringify({angle,text:label.text,delta}));labelChecks.push({angle,text:label.text,delta});
      }
    }
    assert.equal(hash(),before);
    fs.writeFileSync(`${output}/verification.json`,JSON.stringify({...report,rendered,overlap:{...overlap,samples},upper_floor_mask_delta:maskDelta,lower_edge_mask:{samples:edgeSamples.length,max_delta:Math.max(...edgeDeltas)},labels_over_structures:labelChecks,page_errors:errors,stored_result_unchanged:true},null,2));
    console.log(JSON.stringify({calculation_ready:true,b1_voids:report.b1_voids,rendered,gate_bodies:overlap.gates,walls:overlap.walls,gate_pixels_unchanged:samples.every(s=>s.delta<=3),upper_floor_mask_delta:maskDelta,page_errors:errors,stored_result_unchanged:true},null,2));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
