import * as THREE from 'three';
import {OrbitControls} from '/vendor/three/OrbitControls.js';

// Only the view is three-dimensional. Trajectories, stair travel times and FDS
// samples come from the existing simulation without changing their coordinates.
const host = document.getElementById('map3d');
const api = {ready:false};
window.station3D = api;
let renderer, scene, floorDepthScene, floorDepthRoot, structureScene, structureRoot, camera, controls, root, data, people=new Map(), paths=new Map();
const separation=2.5;
let selectionScene,selectionBody,labelScene,labelRoot;
let volumeOpacity=1;
// 승강장을 수평으로 두고 계단·출구가 위에 보이는 공통 시작 시점.
const HOME_VIEW={target:[107.4,19,-21],offset:[-10,24,-145],zoom:1.94};
let floors=[], fireMesh, fireColors, volumeMesh, volumeAlpha, fireKey='', selectedLabel, selectedRoute, lastSelected, overview=true, flames=[];
const v = p => new THREE.Vector3(p.x, elevation(p.floor), -p.y);
const elevation = id => (data.scenario.floors.find(f=>f.id===id)?.elevation_m ?? 8)*separation;
const palette={pre_moving:0xcaa342,walking:0x168b74,waiting:0xb33e88,queueing:0xdd8620,on_stairs:0x3166c5,incapacitated:0xb91c1c};
const ray = new THREE.Raycaster(), pointer = new THREE.Vector2();
const boxGeometry = new THREE.BoxGeometry(1,1,1);
const material = color => new THREE.MeshStandardMaterial({color,roughness:.85,side:THREE.DoubleSide});
// Slabs occlude every lower-floor object, including rails in the main scene.
const matFloor=()=>new THREE.MeshStandardMaterial({color:0xd7d9d4,roughness:1,side:THREE.DoubleSide,depthWrite:true});
const stairMaterial=material(0x839bb5), railMaterial=new THREE.LineBasicMaterial({color:0x355779});
const floorDepthMaterial=new THREE.MeshBasicMaterial({colorWrite:false,depthWrite:true,depthTest:true,side:THREE.DoubleSide});

function label(text, color='#243746', bg='#ffffff') {
  const c=document.createElement('canvas'),ctx=c.getContext('2d');
  ctx.font='600 30px sans-serif';c.width=Math.ceil(ctx.measureText(text).width)+30;c.height=54;
  ctx.fillStyle=bg;ctx.fillRect(0,0,c.width,c.height);ctx.font='600 30px sans-serif';ctx.fillStyle=color;ctx.fillText(text,15,37);
  const tex=new THREE.CanvasTexture(c), sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,depthTest:false,depthWrite:false,transparent:true}));
  sprite.userData.labelWidth=c.width/c.height;sprite.userData.label=true;sprite.userData.labelText=text;return sprite;
}
function shape(points,holes=[]) {
  const s=new THREE.Shape(points.map(([x,y])=>new THREE.Vector2(x,y)));
  for(const h of holes)s.holes.push(new THREE.Path(h.map(([x,y])=>new THREE.Vector2(x,y))));
  return s;
}
function slab(s,height,y,mat,parent=root) {
  const geo=new THREE.ExtrudeGeometry(s,{depth:height,bevelEnabled:false,steps:1});
  geo.rotateX(-Math.PI/2);const m=new THREE.Mesh(geo,mat);m.position.y=y-height;parent.add(m);return m;
}
function line(points,mat,parent=root) {
  const mesh=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),mat);parent.add(mesh);return mesh;
}
function stairPath(s) {
  const display=data.scenario.display_stair_paths?.[s.id];
  if(display){const points=display.map(([x,y,z])=>new THREE.Vector3(x,z*separation,-y));points.nativeLengths=display.slice(1).map((p,i)=>Math.hypot(...p.map((v,k)=>v-display[i][k])));return points;}
  if(s.path_xyz){const points=s.path_xyz.map(([x,y,z])=>new THREE.Vector3(x,z*separation,-y));points.nativeLengths=s.path_xyz.slice(1).map((p,i)=>Math.hypot(...p.map((v,k)=>v-s.path_xyz[i][k])));return points;}
  const a=v(data.scenario.nodes[s.from_node]),b=v(data.scenario.nodes[s.to_node]);
  const flat=Math.hypot(b.x-a.x,b.z-a.z);
  if(s.kind==='exit'||flat>4) return [a,b];
  // A landing and two flights clarify nearly coincident portal coordinates.
  // Their detailed plan, risers and slope have not been surveyed.
  const direction=s.id==='H_W'?-1:1,run=Math.max(3,Math.sqrt(Math.max(0,s.length_m**2-16))/2);
  return [a,new THREE.Vector3(a.x+direction*run,(a.y+b.y)/2,a.z),
    new THREE.Vector3(b.x+direction*run,(a.y+b.y)/2,b.z),b];
}
function sample(points,t) {
  const lengths=points.nativeLengths||points.slice(1).map((p,i)=>p.distanceTo(points[i])),total=lengths.reduce((a,b)=>a+b,0);
  let d=Math.max(0,Math.min(1,t))*total;
  for(let i=0;i<lengths.length;i++) {
    if(d<=lengths[i]||i===lengths.length-1)return points[i].clone().lerp(points[i+1],lengths[i]?d/lengths[i]:0);
    d-=lengths[i];
  }
  return points[0].clone();
}
function buildStairs(s) {
  const points=stairPath(s);paths.set(s.id,points);
  if(s.branch_paths_xyz)for(const [index,path] of s.branch_paths_xyz.entries()){
    const branch=path.map(([x,y,z])=>new THREE.Vector3(x,z*separation,-y));
    branch.nativeLengths=path.slice(1).map((p,i)=>Math.hypot(...p.map((n,k)=>n-path[i][k])));
    paths.set(`${s.id}:${index}`,branch);
  }
  for(const landing of s.landings||[]){
    const mesh=slab(shape(landing.polygon),landing.thickness_m,landing.elevation_m*separation,stairMaterial);
    mesh.userData={stairId:s.id,role:'landing'};
  }
  const flights=s.render_mode==='floor_only'?[]:s.flights?.map(f=>({role:f.role,points:f.path_xyz.map(([x,y,z])=>new THREE.Vector3(x,z*separation,-y)),width:f.width_m}))||[{points,width:s.width_m}];
  for(const flight of flights)for(let k=1;k<flight.points.length;k++) {
    const a=flight.points[k-1],b=flight.points[k],width=flight.width,dx=b.x-a.x,dz=b.z-a.z,flat=Math.hypot(dx,dz);
    if(flat<.04)continue;
    const steps=Math.max(2,Math.ceil(flat/.35)),normal=new THREE.Vector3(-dz/flat,0,dx/flat),yaw=-Math.atan2(dz,dx);
    if(Math.abs(b.y-a.y)<.01) {
      const landing=new THREE.Mesh(boxGeometry,stairMaterial);landing.scale.set(flat+(s.kind==='exit'?0:width),.22,width);landing.rotation.y=yaw;landing.position.copy(a.clone().lerp(b,.5));landing.position.y-=.11;root.add(landing);
      if(s.exit_form==='bent_stair'&&k>=3)for(let i=1;i<steps;i++){
        const p=a.clone().lerp(b,i/steps).add(new THREE.Vector3(0,.015,0));
        line([p.clone().addScaledVector(normal,-width/2),p.clone().addScaledVector(normal,width/2)],railMaterial);
      }
    } else {
    for(let i=0;i<steps;i++) {
      const p=a.clone().lerp(b,(i+.5)/steps),step=new THREE.Mesh(boxGeometry,stairMaterial);
      step.userData={stairId:s.id,role:flight.role||'stair'};
      const rise=(b.y-a.y)/steps,thickness=Math.abs(rise)+.18;
      p.y=a.y+rise*(i+1);
      step.scale.set(flat/steps,thickness,width);step.rotation.y=yaw;step.position.copy(p);step.position.y-=thickness/2;root.add(step);
      if(s.flights){
        const edge=a.clone().lerp(b,(i+1)/steps);edge.y+=.012;
        const tread=line([edge.clone().addScaledVector(normal,-width/2),edge.clone().addScaledVector(normal,width/2)],railMaterial);
        tread.userData={stairId:s.id,role:'tread'};
      }
    }
    }
    for(const sign of s.rail_paths_xyz?[]:[-1,1]) {
      // Leave the inner side of a return landing open at both flight junctions.
      const flatLanding=Math.abs(b.y-a.y)<.01 && s.kind!=='exit';
      const start=a.clone(),end=b.clone();
      if(flatLanding&&sign<0){const inset=Math.min(.49,width/2/flat);start.lerp(b,inset);end.lerp(a,inset);}
      if(!s.flights||!flatLanding)line([start.addScaledVector(normal,sign*width/2).add(new THREE.Vector3(0,.9,0)),
      end.addScaledVector(normal,sign*width/2).add(new THREE.Vector3(0,.9,0))],railMaterial);
    }
  }
  if(s.rail_paths_xyz){
    for(const rail of [...s.rail_paths_xyz,s.center_handrail_xyz]){
      const points=rail.map(([x,y,z])=>new THREE.Vector3(x,z*separation+.9,-y));
      const curve=new THREE.CurvePath();
      for(let i=1;i<points.length;i++)curve.add(new THREE.LineCurve3(points[i-1],points[i]));
      root.add(new THREE.Mesh(new THREE.TubeGeometry(curve,Math.max(8,points.length*8),.045,6,false),stairMaterial));
      for(const [x,y,z] of rail){
        const post=new THREE.Mesh(new THREE.CylinderGeometry(.035,.035,.9,6),stairMaterial);
        post.position.set(x,z*separation+.45,-y);root.add(post);
      }
    }
  }
  if(s.kind==='exit') {
    const badge=label(s.id==='E_1'?'1번 출구':'2번 출구','#4b3e00','#ffdd54');
    badge.position.copy(points.at(-1)).add(new THREE.Vector3(0,3,0));labelRoot.add(badge);
  }
}
function makePerson(a) {
  const g=new THREE.Group(),mat=material(palette.walking);
  const body=new THREE.Mesh(new THREE.CylinderGeometry(.22,.25,.85,7),mat);body.position.y=.8;
  const head=new THREE.Mesh(new THREE.SphereGeometry(.24,8,6),mat);head.position.y=1.52;
  const legs=new THREE.Mesh(new THREE.CylinderGeometry(.16,.18,.45,6),mat);legs.position.y=.25;
  g.add(body,head,legs);g.userData.agentId=a.agent_id;g.userData.mat=mat;
  g.traverse(n=>n.userData.agentId=a.agent_id);root.add(g);people.set(a.agent_id,g);
}
function routeFor(a) {
  if(selectedRoute){root.remove(selectedRoute);selectedRoute.traverse(n=>{n.geometry?.dispose();n.material?.dispose();});}
  selectedRoute=new THREE.Group();root.add(selectedRoute);if(!a)return;
  const mat=new THREE.LineDashedMaterial({color:0x235bc2,transparent:true,opacity:.65,dashSize:1,gapSize:.6,depthWrite:false});
  let current=[],times=[],floor=null;
  const flush=()=>{
    if(current.length>1){
      const l=line(current,mat,selectedRoute);l.computeLineDistances();l.geometry.setDrawRange(0,0);
      l.userData.trail={times,positions:l.geometry.attributes.position.array.slice(),
        distances:l.geometry.attributes.lineDistance.array.slice(),partial:null};
    }
    current=[];times=[];
  };
  for(const s of a.segments) {
    if(s.status==='on_stairs'){flush();floor=null;continue;}
    if(s.from.x===s.to.x&&s.from.y===s.to.y)continue;
    if(s.from.floor!==floor){flush();floor=s.from.floor;}
    if(!times.length||times.at(-1)!==s.start){
      current.push(v(s.from).add(new THREE.Vector3(0,.08,0)));times.push(s.start);
    }
    current.push(v(s.to).add(new THREE.Vector3(0,.08,0)));times.push(s.end);
  }
  flush();
}
function updateRouteTime(time) {
  if(!selectedRoute||selectedRoute.userData.time===time)return;
  selectedRoute.userData.time=time;
  for(const l of selectedRoute.children){
    const trail=l.userData.trail;if(!trail)continue;
    const {times,positions,distances}=trail,position=l.geometry.attributes.position,distance=l.geometry.attributes.lineDistance;
    position.clearUpdateRanges();distance.clearUpdateRanges();
    const setPoint=(index,x,y,z,d)=>{
      position.setXYZ(index,x,y,z);position.addUpdateRange(index*3,3);
      distance.setX(index,d);distance.addUpdateRange(index,1);
    };
    if(trail.partial!==null){
      const i=trail.partial;setPoint(i,positions[i*3],positions[i*3+1],positions[i*3+2],distances[i]);trail.partial=null;
    }
    let lo=0,hi=times.length;
    while(lo<hi){const mid=(lo+hi)>>1;if(times[mid]<=time)lo=mid+1;else hi=mid;}
    let count=lo;
    if(lo>0&&lo<times.length&&time>times[lo-1]){
      const f=(time-times[lo-1])/(times[lo]-times[lo-1]),start=(lo-1)*3,end=lo*3;
      setPoint(lo,...[0,1,2].map(k=>positions[start+k]+f*(positions[end+k]-positions[start+k])),
        distances[lo-1]+f*(distances[lo]-distances[lo-1]));
      trail.partial=lo;count++;
    }
    if(position.updateRanges.length){position.needsUpdate=true;distance.needsUpdate=true;}
    l.geometry.setDrawRange(0,count>1?count:0);
  }
}
function pointFor(a,time) {
  const p=position(a,time),s=p.segment;
  if(p.status==='evacuated')return {p,status:p.status,point:null};
  if(!s)return {p,status:p.status,point:v(a.spawn)};
  if(s.native_motion?.length){
    const motion=s.native_motion;let lo=0,hi=motion.length-1;
    while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(motion[mid][0]<=time)lo=mid;else hi=mid-1;}
    const a=motion[lo],b=motion[Math.min(lo+1,motion.length-1)],t=b[0]>a[0]?Math.max(0,Math.min(1,(time-a[0])/(b[0]-a[0]))):0;
    return {p,status:p.status,point:new THREE.Vector3(a[1]+t*(b[1]-a[1]),(a[3]+t*(b[3]-a[3]))*separation,-a[2]-t*(b[2]-a[2]))};
  }
  if(s.status==='on_stairs') {
    const stair=data.scenario.stairs.find(x=>x.id===s.stair_id);
    if(stair.branch_paths_xyz){
      const branch=Math.min(stair.branch_paths_xyz.length-1,Math.floor((s.stair_lane||0)/(stair.parallel_lanes/stair.branch_paths_xyz.length)));
      const source=paths.get(`${s.stair_id}:${branch}`),pts=source.map(p=>p.clone());pts.nativeLengths=source.nativeLengths;
      const start=v(s.from).sub(source[0]),end=v(s.to).sub(source.at(-1));
      for(let i=0;i<pts.length;i++)pts[i].add(i<2?start:end);
      return {p,status:p.status,point:sample(pts,p.progress)};
    }
    const pts=paths.get(s.stair_id),q=sample(pts,p.progress);
    // Distribute visible users across the entrance width. This is an animation
    // of the capacity model, not a claim to simulate overtaking on stairs.
    q.addScaledVector(v(s.from).sub(pts[0]),1-p.progress);
    q.addScaledVector(v(s.to).sub(pts.at(-1)),p.progress);
    return {p,status:p.status,point:q};
  }
  return {p,status:p.status,point:v(s.from).lerp(v(s.to),p.progress)};
}
function updateLabels() {
  const units=(camera.top-camera.bottom)/camera.zoom/Math.max(1,host.clientHeight);
  root.traverse(n=>{if(n.userData.label)n.scale.set(n.userData.labelWidth*25*units,25*units,1);});
  labelRoot?.traverse(n=>{if(n.userData.label)n.scale.set(n.userData.labelWidth*25*units,25*units,1);});
  selectionScene?.traverse(n=>{if(n.userData.label)n.scale.set(n.userData.labelWidth*25*units,25*units,1);});
}
function render() {
  if(!api.ready||!root)return;updateLabels();renderer.clear();renderer.render(scene,camera);
  // Keep the separate structure pass behind upper slabs and in front of lower
  // ones, using the same floor geometry (including actual stair openings).
  renderer.render(floorDepthScene,camera);
  renderer.render(structureScene,camera);
  // All station names render after geometry; upper floors cannot repaint them.
  renderer.clearDepth();renderer.render(labelScene,camera);
  // A separate final pass keeps the selected person visible above floors/smoke.
  renderer.clearDepth();renderer.render(selectionScene,camera);
}
function buildSelection() {
  for(const child of [...selectionScene.children]){selectionScene.remove(child);child.traverse(n=>{n.geometry?.dispose();n.material?.map?.dispose();n.material?.dispose();});}
  selectionBody=new THREE.Group();selectionScene.add(selectionBody);
  const template=people.values().next().value;
  for(const source of template.children){
    const outline=new THREE.Mesh(source.geometry.clone(),new THREE.MeshBasicMaterial({color:0xffd56b,side:THREE.BackSide}));
    outline.position.copy(source.position);outline.scale.setScalar(1.16);outline.renderOrder=1;selectionBody.add(outline);
    const body=new THREE.Mesh(source.geometry.clone(),new THREE.MeshBasicMaterial({color:0x138eff}));
    body.position.copy(source.position);body.renderOrder=2;body.userData.selectionFill=true;selectionBody.add(body);
  }
  const ring=new THREE.Mesh(new THREE.RingGeometry(.38,.58,32),new THREE.MeshBasicMaterial({color:0xffd56b,side:THREE.DoubleSide}));
  ring.rotation.x=-Math.PI/2;ring.position.y=.04;selectionBody.add(ring);
  selectedLabel=label(`#${state.selected||1}`,'#21252c','#ffd56b');selectionScene.add(selectedLabel);
}
function fit() {
  overview=true;stationView.follow=false;
  controls.target.fromArray(HOME_VIEW.target);
  camera.position.copy(controls.target).add(new THREE.Vector3(...HOME_VIEW.offset));
  camera.zoom=HOME_VIEW.zoom;camera.lookAt(controls.target);
  camera.updateProjectionMatrix();controls.update();render();
}
function resize() {
  if(!renderer)return;
  const w=host.clientWidth,h=host.clientHeight,aspect=w/Math.max(1,h),span=Math.max(115,200/aspect);
  renderer.setSize(w,h,false);camera.left=-span*aspect/2;camera.right=span*aspect/2;camera.top=span/2;camera.bottom=-span/2;
  camera.updateProjectionMatrix();render();
}
function alphaMaterial(soft=false) {
  const mat=new THREE.MeshBasicMaterial({transparent:true,depthWrite:false,side:THREE.DoubleSide});
  mat.customProgramCacheKey=()=>`fds-alpha-${soft}`;
  mat.onBeforeCompile=shader=>{
    shader.vertexShader=shader.vertexShader.replace('#include <common>','#include <common>\nattribute float instanceAlpha; varying float vAlpha;').replace('#include <begin_vertex>','#include <begin_vertex>\nvAlpha=instanceAlpha;');
    shader.fragmentShader=shader.fragmentShader.replace('#include <common>','#include <common>\nvarying float vAlpha;').replace('#include <dithering_fragment>','gl_FragColor.a *= vAlpha;\n#include <dithering_fragment>');
    if(soft) {
      shader.vertexShader=shader.vertexShader.replace('#include <common>','#include <common>\nvarying vec2 smokeUv;').replace('#include <begin_vertex>','#include <begin_vertex>\nsmokeUv=vec2(transformed.x/5.+.5,transformed.y/2.+.5);')
        .replace('#include <project_vertex>','vec4 mvPosition=modelViewMatrix*instanceMatrix*vec4(0.,0.,0.,1.); mvPosition.xy+=vec2(transformed.x,transformed.y*length(instanceMatrix[1].xyz)); gl_Position=projectionMatrix*mvPosition;');
      shader.fragmentShader=shader.fragmentShader.replace('#include <common>','#include <common>\nvarying vec2 smokeUv;').replace('#include <dithering_fragment>','float smokeRadius=length((smokeUv-.5)*2.); gl_FragColor.a*=exp(-2.*smokeRadius*smokeRadius)*(1.-smoothstep(.65,1.,smokeRadius));\n#include <dithering_fragment>');
    }
  };return mat;
}
function smokeHeight(z) {
  const i=Math.floor(Math.max(0,z)/4),local=z-i*4;
  return i*4*separation+Math.min(local,3)+Math.max(0,local-3)*(4*separation-3);
}
function buildTrain() {
  const train=data.scenario.train;if(!train)return;
  const bodyMat=material(0xdbe2e5),purple=material(train.color),glass=material(0x344656),door=material(0xb9c5c9);
  for(const car of train.cars) {
    const g=new THREE.Group(),dx=car.b[0]-car.a[0],dy=car.b[1]-car.a[1],len=Math.hypot(dx,dy),w=car.width_m,h=car.height_m;
    g.position.set((car.a[0]+car.b[0])/2,h/2,-(car.a[1]+car.b[1])/2);g.rotation.y=Math.atan2(dy,dx);root.add(g);
    const box=(x,y,z,sx,sy,sz,mat)=>{const m=new THREE.Mesh(boxGeometry,mat);m.position.set(x,y,z);m.scale.set(sx,sy,sz);g.add(m);};
    box(0,0,0,len,h,w,bodyMat);
    for(const sign of [-1,1]) {
      box(0,-.15,sign*(w/2+.015),len,.22,.04,purple);
      for(let k=0;k<9;k++)box(-len*.43+k*len*.106,.45,sign*(w/2+.04),.95,.65,.04,glass);
      for(const t of [-.35,-.12,.12,.35])box(len*t,-.2,sign*(w/2+.07),1.1,1.75,.05,door);
    }
  }
  const badge=label('5호선 정차 열차','#ffffff',train.color);badge.position.set(35,1,-4);labelRoot.add(badge);
}
function fireUpdate(time) {
  if(!fireMesh)return;
  const mode=document.getElementById('fieldMode').value,key=`${mode}-${Math.floor(time*5)}`;
  if(key===fireKey)return;fireKey=key;fireMesh.visible=!['none','volume'].includes(mode);
  const frames=data.fire.frames;let hi=frames.findIndex(f=>f.time_s>=time);if(hi<0)hi=frames.length-1;
  const a=frames[Math.max(0,hi-1)],b=frames[hi],w=a===b?0:THREE.MathUtils.clamp((time-a.time_s)/(b.time_s-a.time_s),0,1);
  const color=new THREE.Color();
  for(let i=0;i<data.fire.cells.length;i++) {
    const k=a.k[i]+w*(b.k[i]-a.k[i]),t=a.temperature_c[i]+w*(b.temperature_c[i]-a.temperature_c[i]);
    const alpha=mode==='temperature'?THREE.MathUtils.clamp((t-20)/180,0,.9):Math.min(.88,1-Math.exp(-2*k));
    if(mode==='temperature')color.setHSL(Math.max(0,50-(t-20)/180*50)/360,.95,.55);else color.set('#4a5360');
    fireMesh.setColorAt(i,color);fireColors.setX(i,alpha);
  }
  fireMesh.instanceColor.needsUpdate=true;fireColors.needsUpdate=true;
  if(volumeMesh) {
    volumeMesh.visible=mode==='volume';
    if(volumeMesh.visible) {
      const fs=data.fire.volume.frames;let j=fs.findIndex(f=>f.time_s>=time);if(j<0)j=fs.length-1;
      const lo=fs[Math.max(0,j-1)],hi=fs[j],u=lo===hi?0:THREE.MathUtils.clamp((time-lo.time_s)/(hi.time_s-lo.time_s),0,1);
      for(let i=0;i<lo.k.length;i++)volumeAlpha.setX(i,volumeOpacity*Math.min(.50,1-Math.exp(-1.2*(lo.k[i]+u*(hi.k[i]-lo.k[i])))));
      volumeAlpha.needsUpdate=true;
    }
  }
  const ft=time+data.fire.metadata.fire.evacuation_start_s;
  for(const flame of flames) {
    const s=flame.userData.stage,f=THREE.MathUtils.clamp((ft-s.start_s)/(s.full_s-s.start_s),0,1);
    flame.visible=f>0;flame.scale.setScalar(Math.cbrt(f)*(1+.06*Math.sin(time*12+s.car_id)));
  }
}
function rebuild() {
  if(labelRoot){labelScene.remove(labelRoot);labelRoot.traverse(n=>{n.material?.map?.dispose();n.material?.dispose();});}
  labelRoot=new THREE.Group();labelScene.add(labelRoot);
  if(root) {scene.remove(root);root.traverse(n=>{if(n.geometry&&n.geometry!==boxGeometry)n.geometry.dispose();if(n.material?.map)n.material.map.dispose();if(n.material&&n.material!==stairMaterial&&n.material!==railMaterial)n.material.dispose();});}
  if(structureRoot){structureScene.remove(structureRoot);structureRoot.traverse(n=>{if(n.geometry&&n.geometry!==boxGeometry)n.geometry.dispose();n.material?.dispose();});}
  if(floorDepthRoot)floorDepthScene.remove(floorDepthRoot);
  floorDepthRoot=new THREE.Group();floorDepthScene.add(floorDepthRoot);
  structureRoot=new THREE.Group();structureScene.add(structureRoot);
  root=new THREE.Group();scene.add(root);people.clear();paths.clear();floors=[];fireMesh=null;volumeMesh=null;flames=[];fireKey='';
  for(const f of data.scenario.floors) {
    const mats=[];
    const polys=data.scenario.display3d_openings?.[separation.toFixed(1)]?.[f.id]||data.scenario.display3d?.[f.id]||f.polygons.map(outer=>({outer,holes:[]}));
    for(const p of polys){const mat=matFloor();const body=slab(shape(p.outer,p.holes),.3,elevation(f.id),mat);body.userData={floorId:f.id,role:'floor'};mats.push(mat);
      const depthBody=new THREE.Mesh(body.geometry,floorDepthMaterial);depthBody.position.copy(body.position);depthBody.userData={floorId:f.id,role:'floor_depth'};floorDepthRoot.add(depthBody);
      for(const hole of p.holes){const edge=line(hole.map(([x,y])=>new THREE.Vector3(x,elevation(f.id)+.03,-y)),railMaterial.clone(),structureRoot);edge.userData={floorId:f.id,kind:'floor_edge'};}
    }
    for(const p of polys){const edge=line(p.outer.map(([x,y])=>v({floor:f.id,x,y}).add(new THREE.Vector3(0,.02,0))),new THREE.LineBasicMaterial({color:0x83918f,transparent:true,opacity:.6,depthTest:true}),structureRoot);edge.userData={floorId:f.id,kind:'floor_edge'};}
    for(const o of f.obstacles) {
      if(o.display===false||o.render_as==='boundary_walls')continue;
      const mat=material(o.kind==='fare_gate'?0x3d6f79:o.label==='기둥'?0x82918f:0xb7c1bf);
      const height=o.height_m??(o.label==='기둥'?1.8:.7);
      const body=slab(shape(o.polygon),height,elevation(f.id)+height,mat,structureRoot);
      body.userData={floorId:f.id,role:'structure',kind:o.kind==='fare_gate'?'gate':o.label==='기둥'?'column':'building',label:o.label};
      if(o.kind==='fare_gate'){body.userData.fareGate=true;body.userData.height_m=height;}
      const edge=new THREE.LineSegments(new THREE.EdgesGeometry(body.geometry,25),new THREE.LineBasicMaterial({color:o.kind==='fare_gate'?0x214a55:0x64716e}));body.add(edge);
    }
    for(const wall of f.walls||[]){
      const body=slab(shape(wall.polygon),wall.height_m,elevation(f.id)+wall.height_m,material(0xa5afab),structureRoot);
      body.userData={floorId:f.id,role:'structure',kind:'wall',wallId:wall.id,label:wall.label};
      body.add(new THREE.LineSegments(new THREE.EdgesGeometry(body.geometry,25),new THREE.LineBasicMaterial({color:0x818d87})));
    }
    const coordinates=f.id==='F1'?[86,47]:f.id==='B1'?[83,41]:[33,38];
    const title=label(f.id==='F1'?'1F · 지상 대합실':f.id==='B1'?'B1 · 원형 대합실':'B2 · 곡선 승강장');
    title.position.copy(v({floor:f.id,x:coordinates[0],y:coordinates[1]})).add(new THREE.Vector3(0,2,0));labelRoot.add(title);
    floors.push({id:f.id,mats});
  }
  for(const s of data.scenario.stairs)buildStairs(s);
  for(const vestibule of data.scenario.vestibules||[]){
    if(vestibule.render_mode==='floor_only')continue;
    for(const [a,b] of vestibule.door_lines){
      const p=v({floor:vestibule.floor,x:a[0],y:a[1]}),q=v({floor:vestibule.floor,x:b[0],y:b[1]});
      line([p,p.clone().add(new THREE.Vector3(0,2.1,0)),q.clone().add(new THREE.Vector3(0,2.1,0)),q],railMaterial);
    }
  }
  buildTrain();
  for(const a of data.agents)makePerson(a);
  buildSelection();
  if(data.fire?.enabled) {
    const cells=data.fire.cells,geo=new THREE.PlaneGeometry(1,1);geo.rotateX(-Math.PI/2);
    fireColors=new THREE.InstancedBufferAttribute(new Float32Array(cells.length),1);geo.setAttribute('instanceAlpha',fireColors);
    const mat=alphaMaterial();
    fireMesh=new THREE.InstancedMesh(geo,mat,cells.length);const dummy=new THREE.Object3D();
    cells.forEach((c,i)=>{dummy.position.copy(v({floor:c.floor,x:c.x+c.w/2,y:c.y+c.h/2}));dummy.position.y+=.035;dummy.scale.set(c.w,1,c.h);dummy.updateMatrix();fireMesh.setMatrixAt(i,dummy.matrix);});root.add(fireMesh);
    const f=data.fire.metadata.fire;
    for(const stage of f.stages||[{rect:f.rect,start_s:0,full_s:f.ramp_full_s,car_id:0}]) {
      const [x,y,w,h]=stage.rect,group=new THREE.Group();group.position.copy(v({floor:f.floor,x:x+w/2,y:y+h/2}));group.position.y+=data.scenario.train?2.4:0;
      for(let k=0;k<5;k++) {const flame=new THREE.Mesh(new THREE.ConeGeometry(.55,2.5,7),new THREE.MeshBasicMaterial({color:k%2?0xffb32c:0xff5427}));flame.position.set((k%3-.9)*.7,1.2+Math.floor(k/3)*.3,(k%2-.5)*.8);group.add(flame);}
      group.userData.stage=stage;flames.push(group);root.add(group);
    }
    if(data.fire.volume) {
      const volume=data.fire.volume,geo=new THREE.PlaneGeometry(5,2);volumeAlpha=new THREE.InstancedBufferAttribute(new Float32Array(volume.voxels.length),1);geo.setAttribute('instanceAlpha',volumeAlpha);
      volumeMesh=new THREE.InstancedMesh(geo,alphaMaterial(true),volume.voxels.length);const voxel=new THREE.Object3D();
      volume.voxels.forEach((p,i)=>{const lo=smokeHeight(Math.max(0,p.z-.5)),hi=smokeHeight(p.z+.5);voxel.position.set(p.x,(lo+hi)/2,-p.y);voxel.scale.set(1,hi-lo,1);voxel.updateMatrix();volumeMesh.setMatrixAt(i,voxel.matrix);volumeMesh.setColorAt(i,new THREE.Color('#52565e'));});root.add(volumeMesh);
    }
  }
  lastSelected=null;fit();api.update(data,state.time,state.selected);
}
api.init = result => {
  if(!renderer) {
    try {
      renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});renderer.autoClear=false;renderer.setPixelRatio(Math.min(window.devicePixelRatio,1.5));renderer.setClearColor(0xeef1f1);
      renderer.domElement.setAttribute('aria-label','개화산역 입체 안내도. 마우스로 회전하고 확대할 수 있습니다.');host.append(renderer.domElement);
      scene=new THREE.Scene();floorDepthScene=new THREE.Scene();structureScene=new THREE.Scene();selectionScene=new THREE.Scene();labelScene=new THREE.Scene();const ambient=new THREE.HemisphereLight(0xffffff,0x98a6b3,1.3);scene.add(ambient);const light=new THREE.DirectionalLight(0xffffff,1.5);light.position.set(-30,100,80);scene.add(light);structureScene.add(ambient.clone(),light.clone());
      camera=new THREE.OrthographicCamera(-100,100,60,-60,.1,1200);controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=false;controls.minPolarAngle=.15;controls.maxPolarAngle=Math.PI*.48;controls.minZoom=.5;controls.maxZoom=12;
      controls.addEventListener('change',render);controls.addEventListener('start',()=>{stationView.follow=false;api.update(data,state.time,state.selected);});
      let down;
      renderer.domElement.addEventListener('pointerdown',e=>down=[e.clientX,e.clientY]);
      renderer.domElement.addEventListener('pointerup',e=>{
        if(!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>5)return;
        const r=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);ray.setFromCamera(pointer,camera);
        const visible=[...people.values()].filter(p=>p.visible);
        const hit=ray.intersectObjects(visible,true)[0];let id=hit?.object.userData.agentId;
        if(id===undefined){let nearest=10;for(const person of visible){const p=person.position.clone().add(new THREE.Vector3(0,.9,0)).project(camera);const d=Math.hypot((p.x+1)*r.width/2-(e.clientX-r.left),(1-p.y)*r.height/2-(e.clientY-r.top));if(d<nearest){nearest=d;id=person.userData.agentId;}}}
        if(id!==undefined){select(id);api.follow();}
      });
      new ResizeObserver(resize).observe(host);api.ready=true;document.querySelector('.map-viewport').classList.add('has-3d');
      document.getElementById('smokeOpacity').oninput=e=>{volumeOpacity=+e.target.value;document.getElementById('smokeOpacityValue').textContent=`${Math.round(volumeOpacity*100)}%`;fireKey='';api.update(data,state.time,state.selected);};
    } catch(error) {host.textContent='입체 화면을 열 수 없어 평면 지도를 표시합니다.';console.warn(error);return;}
  }
  data=result;resize();rebuild();
};
api.fit=()=>{fit();api.update(data,state.time,state.selected);};
api.zoom=factor=>{camera.zoom=THREE.MathUtils.clamp(camera.zoom*factor,.5,12);camera.updateProjectionMatrix();render();};
api.follow=()=>{stationView.follow=true;overview=false;camera.zoom=Math.max(4,camera.zoom);camera.updateProjectionMatrix();api.update(data,state.time,state.selected);};
api.update=(result,time,selected)=>{
  if(!api.ready)return;
  if(data!==result){api.init(result);return;}
  const active=result.agents.find(a=>a.agent_id===selected);let q=null;
  for(const a of result.agents) {
    const info=pointFor(a,time),g=people.get(a.agent_id);g.visible=!!info.point;
    if(info.point)g.position.copy(info.point);
    const isSelected=a.agent_id===selected;
    if(isSelected)q=info;
    const visual=agentVisual(a,time);
    g.userData.mat.color.set(visual.color);
    g.userData.mat.depthTest=true;g.children.forEach(n=>n.renderOrder=0);
    if(isSelected){selectionBody.visible=!!info.point;if(info.point)selectionBody.position.copy(info.point);selectionBody.traverse(n=>{if(n.userData.selectionFill)n.material.color.set(visual.color);});}
    // A modest size boost in the guide view keeps individual users visible.
    g.scale.setScalar(1);
  }
  selectedLabel.visible=!!q?.point;
  if(q?.point)selectedLabel.position.copy(q.point).add(new THREE.Vector3(0,2.5,0));
  if(lastSelected!==selected){const old=selectedLabel;selectionScene.remove(old);old.material.map.dispose();old.material.dispose();selectedLabel=label(`#${selected}`,'#21252c','#ffd56b');selectionScene.add(selectedLabel);lastSelected=selected;selectedLabel.visible=!!q?.point;if(q?.point)selectedLabel.position.copy(q.point).add(new THREE.Vector3(0,2.5,0));routeFor(active);}
  updateRouteTime(time);
  if(stationView.follow&&q?.point) {
    const delta=q.point.clone().sub(controls.target);controls.target.add(delta);camera.position.add(delta);controls.update();
  } else if(stationView.follow&&q?.status==='evacuated')fit();
  fireUpdate(time);
  render();
};
if(typeof state!=='undefined'&&state.data?.scenario.drawing)api.init(state.data);
