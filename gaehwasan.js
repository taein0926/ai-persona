'use strict';
// One scale for x and y, and for every floor. Panel offsets only separate levels.
const stationScale = 5.6;
const stationPanels = {F1: 330, B1: 660, B2: 1010, OUTSIDE: 330};
const stationView = {zoom:1, center:null, follow:false, initialSelection:true,lastTime:null};
function stationXY(p) {
  return {x: 45+p.x*stationScale, y: stationPanels[p.floor]-p.y*stationScale};
}
function stationPolygon(points, floor, attributes, parent) {
  return svg('polygon', {'data-map-floor':floor,...attributes, points: points.map(([x,y]) => {
    const p=stationXY({floor,x,y}); return `${p.x},${p.y}`;
  }).join(' ')}, '', parent);
}
function stationExitLabel(id) {return state.data?.scenario.exit_labels?.[id] || id;}
function buildStationMap() {
  const map=$('map'), c=state.data.scenario;
  map.replaceChildren();
  map.setAttribute('viewBox','0 0 1100 1050');
  const defs=svg('defs');
  const marker=svg('marker',{id:'upArrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:5,markerHeight:5,orient:'auto'},'',defs);
  svg('path',{d:'M0 0 L10 5 L0 10 Z',fill:'#91baff'},'',marker);
  for(const f of c.floors) {
    for(const points of f.polygons) stationPolygon(points,f.id,{class:'floor-shape'});
    for(const opening of f.stair_voids||[])stationPolygon(opening,f.id,{fill:'#102127',stroke:'#83918f','stroke-width':.7});
    for(const o of f.obstacles) {
      if(o.display===false||o.render_as==='boundary_walls')continue;
      stationPolygon(o.polygon,f.id,{fill:'#102127',stroke:'#526c72','stroke-width':.7});
      if(o.label==='기둥'||o.kind==='fare_gate') continue;
      const center=o.polygon.reduce((p,v)=>[p[0]+v[0]/o.polygon.length,p[1]+v[1]/o.polygon.length],[0,0]);
      const p=xy({floor:f.id,x:center[0],y:center[1]});
      svg('text',{'data-map-floor':f.id,x:p.x,y:p.y,fill:'#8fa8af','font-size':9,'text-anchor':'middle'},o.label);
    }
  }
  for(const floor of c.floors)for(const wall of floor.walls||[]){
    stationPolygon(wall.polygon,floor.id,{fill:'#1d2628',stroke:'#647674','stroke-width':.8});
  }
  for(const gate of c.fare_gates||[]){
    stationPolygon(gate.polygon,gate.floor,{fill:'#5ce6ad33',stroke:'#5ce6ad','stroke-width':.5,'data-fare-gate':gate.id});
  }
  for(const vestibule of c.vestibules||[]){
    if(vestibule.render_mode==='floor_only')continue;
    stationPolygon(vestibule.polygon,vestibule.floor,{fill:'#85d9e419',stroke:'#85d9e4','stroke-width':.7});
    for(const [a,b] of vestibule.door_lines){
      const p=stationXY({floor:vestibule.floor,x:a[0],y:a[1]}),q=stationXY({floor:vestibule.floor,x:b[0],y:b[1]});
      svg('line',{'data-map-floor':vestibule.floor,x1:p.x,y1:p.y,x2:q.x,y2:q.y,stroke:'#85d9e4','stroke-dasharray':'3 3'});
    }
  }
  const stairLayer=svg('g',{id:'stationStairs'});
  for(const s of c.stairs) if(c.nodes[s.to_node].floor!=='OUTSIDE') drawStairRibbon(s,stairLayer);
  const fireLayer=svg('g',{id:'fireLayer','pointer-events':'none'});
  for(const c of state.data.fire?.cells || []) {
    const p=xy({floor:c.floor,x:c.x,y:c.y+c.h});
    svg('rect',{'data-map-floor':c.floor,x:p.x,y:p.y,width:c.w*stationScale,height:c.h*stationScale,opacity:0},'',fireLayer);
  }
  state.fireCells=[...fireLayer.children]; state.lastFireDraw=null;
  for(const [floor,title,subtitle] of [
    ['F1','1F · 대합실과 지상 출구','1번 / 2번 출구로 대피'],
    ['B1','B1 · 원형 대합실','개찰구 비상 개방 · 기능실과 기둥을 피해 이동'],
    ['B2','B2 · 곡선 상대식 승강장',`서로 다른 페르소나 ${state.data.agents.length}명 · 선로 횡단 불가`]]) {
    const y=stationPanels[floor]-270;
    svg('text',{'data-map-floor':floor,x:35,y,fill:'#ecf3f1','font-size':20,'font-weight':600},title);
    svg('text',{'data-map-floor':floor,x:35,y:y+23,fill:'#a1b7bd','font-size':12},subtitle);
    if(floor!=='F1') svg('line',{'data-map-floor':'ALL',x1:30,y1:y-25,x2:1070,y2:y-25,stroke:'#30454c'});
  }
  for(const s of c.stairs) {
    const n=c.nodes[s.from_node], p=xy(n), t=s.portal_tangent;
    if(s.render_mode==='floor_only'){
      const dest=xy(c.nodes[s.to_node]);
      svg('text',{'data-map-floor':n.floor,x:dest.x,y:dest.y-12,fill:'#74e0bb','font-size':15,'font-weight':700},stationExitLabel(s.to_node));
      continue;
    }
    const width=s.width_m*stationScale, normal=[-t[1],t[0]], depth=12;
    // A tread symbol and a width-sized entry line; the physical stair length is timed.
    const corners=[[-1,-1],[1,-1],[1,1],[-1,1]].map(([a,b])=>[
      p.x+a*t[0]*width/2+b*normal[0]*depth/2,
      p.y-a*t[1]*width/2-b*normal[1]*depth/2]);
    svg('polygon',{'data-map-floor':n.floor,points:corners.map(q=>q.join(',')).join(' '),fill:'#28445a',stroke:'#91baff','stroke-width':1});
    for(let k=-4;k<=4;k+=4) svg('line',{
      'data-map-floor':n.floor,x1:p.x-t[0]*width/2+normal[0]*k, y1:p.y+t[1]*width/2-normal[1]*k,
      x2:p.x+t[0]*width/2+normal[0]*k, y2:p.y-t[1]*width/2-normal[1]*k,stroke:'#91baff'});
    const sign=['B_W','B_E','E_2'].includes(s.id)?1:-1;
    svg('text',{'data-map-floor':n.floor,x:p.x+8,y:p.y+sign*21,fill:'#cee2ff','font-size':11},`${s.id} · ${s.width_m.toFixed(2)}m`);
    svg('text',{'data-map-floor':n.floor,id:`live-${s.id}`,x:p.x+8,y:p.y+sign*21+12,fill:'#a1b7bd','font-size':9},'대기 0 · 이동 0');
    if(n.floor==='F1' && s.kind==='exit') {
      const dest=xy(c.nodes[s.to_node]);
      svg('text',{'data-map-floor':n.floor,x:dest.x,y:dest.y+sign*22,fill:'#74e0bb','font-size':15,'font-weight':700},stationExitLabel(s.to_node));
    }
  }
  if(state.data.fire?.enabled) {
    const f=state.data.fire.metadata.fire, [x,y,w,h]=f.rect;
    const p=xy({floor:f.floor,x,y:y+h});
    svg('rect',{'data-map-floor':f.floor,x:p.x,y:p.y,width:w*stationScale,height:h*stationScale,fill:'#ff582e',stroke:'#ffd398'});
    svg('text',{'data-map-floor':f.floor,x:p.x,y:p.y-8,'font-size':22},'🔥');
    svg('text',{'data-map-floor':f.floor,x:p.x,y:p.y-30,fill:'#ffb08a','font-size':13},f.label||'화재 위치');
  }
  // The bar is in the same scale as all floor polygons, not a decorative distance label.
  svg('g',{id:'viewportScale'});
  svg('g',{id:'selectedRoute'});
  const layer=svg('g',{id:'agentsLayer'});
  for(const a of state.data.agents) {
    const g=svg('g',{id:`agent-${a.agent_id}`,class:'agent',role:'button',tabindex:0},'',layer);
    svg('circle',{r:2.8,fill:colors.walking,stroke:'#102127','stroke-width':.7},'',g);
    svg('text',{'text-anchor':'middle',y:-5,fill:'#ecf3f1','font-size':8},a.agent_id,g);
    g.addEventListener('click',()=>select(a.agent_id));
    g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(a.agent_id);}});
  }
  setupStationView();
  if(window.station3D)window.station3D.init(state.data);
}

function stationSelectAgent(agent) {
  if(stationView.initialSelection) {stationView.initialSelection=false;return;}
  if(window.station3D?.ready)window.station3D.follow();
  if(stationView.follow) stationView.center=position(agent,state.time).pos;
}
function stationPlaybackStarted() {
  if(window.station3D?.ready){window.station3D.follow();return;}
  stationView.follow=true;stationView.zoom=Math.max(3,stationView.zoom);
  const agent=state.data.agents.find(a=>a.agent_id===state.selected);
  if(agent) stationView.center=position(agent,state.time).pos;
  $('map').scrollIntoView({behavior:'smooth',block:'center'});
}
function drawStairRibbon(s,parent) {
  const c=state.data.scenario,a=stationXY(c.nodes[s.from_node]),b=stationXY(c.nodes[s.to_node]);
  const length=Math.hypot(b.x-a.x,b.y-a.y),dx=(b.x-a.x)/length,dy=(b.y-a.y)/length;
  const nx=-dy,ny=dx,w=s.width_m*stationScale;
  const corners=[[a.x+nx*w/2,a.y+ny*w/2],[b.x+nx*w/2,b.y+ny*w/2],[b.x-nx*w/2,b.y-ny*w/2],[a.x-nx*w/2,a.y-ny*w/2]];
  svg('polygon',{points:corners.map(p=>p.join(',')).join(' '),fill:'#29475d',stroke:'#91baff','stroke-width':1.2},'',parent);
  for(let t=8;t<length;t+=10) {
    const x=a.x+dx*t,y=a.y+dy*t;
    svg('line',{x1:x-nx*w/2,y1:y-ny*w/2,x2:x+nx*w/2,y2:y+ny*w/2,stroke:'#91baff88','stroke-width':1},'',parent);
  }
}
function setupStationView() {
  Object.assign(stationView,{zoom:1,center:null,follow:false,initialSelection:true,lastTime:null});
  $('zoomFit').onclick=()=>{if(window.station3D?.ready){window.station3D.fit();return;}stationView.zoom=1;stationView.center=null;stationView.follow=false;drawStationView();};
  const mini=$('stationMiniMap');mini.replaceChildren();
  for(const f of state.data.scenario.floors) {
    for(const points of f.polygons) stationPolygon(points,f.id,{fill:'#2e4a53',stroke:'#839ba3','stroke-width':3},mini);
    for(const o of f.obstacles)if(o.display!==false&&o.render_as!=='boundary_walls')stationPolygon(o.polygon,f.id,{fill:'#102127'},mini);
    for(const wall of f.walls||[])stationPolygon(wall.polygon,f.id,{fill:'#1d2628'},mini);
    svg('text',{x:20,y:stationPanels[f.id]-250,fill:'#cce0db','font-size':52},f.id==='F1'?'1F':f.id,mini);
  }
  for(const s of state.data.scenario.stairs) {
    if(state.data.scenario.nodes[s.to_node].floor==='OUTSIDE')continue;
    const a=stationXY(state.data.scenario.nodes[s.from_node]),b=stationXY(state.data.scenario.nodes[s.to_node]);
    svg('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:'#91baff88','stroke-width':8},'',mini);
  }
  svg('rect',{id:'cameraMiniFrame',fill:'#74e0bb18',stroke:'#74e0bb','stroke-width':8},'',mini);
  svg('circle',{id:'selectedMiniAgent',r:20,fill:'#fff'},'',mini);
  mini.onclick=()=>{$('zoomFit').click();};
  const map=$('map');let drag=null;
  map.onwheel=e=>{e.preventDefault();zoomStation(e.deltaY<0?1.2:1/1.2);};
  map.onpointerdown=e=>{
    if(e.target.closest('.agent')||e.button!==0)return;
    const v=map.viewBox.baseVal;drag={x:e.clientX,y:e.clientY,cx:v.x+v.width/2,cy:v.y+v.height/2};
    stationView.follow=false;map.setPointerCapture(e.pointerId);map.classList.add('panning');
  };
  map.onpointermove=e=>{
    if(!drag)return;const box=map.getBoundingClientRect(),v=map.viewBox.baseVal,scale=Math.max(v.width/box.width,v.height/box.height);
    stationView.center={x:drag.cx-(e.clientX-drag.x)*scale,y:drag.cy-(e.clientY-drag.y)*scale};drawStationView();
  };
  map.onpointerup=map.onpointercancel=()=>{drag=null;map.classList.remove('panning');};
  if(!map.stationResizeObserver) {
    map.stationResizeObserver=new ResizeObserver(()=>{if(state.data?.scenario.drawing)drawStationView();});
    map.stationResizeObserver.observe(map);
  }
}
function zoomStation(factor) {
  if(window.station3D?.ready){window.station3D.zoom(factor);return;}
  stationView.zoom=Math.max(1,Math.min(10,stationView.zoom*factor));drawStationView();
}
function drawStationView() {
  if(!state.data)return;
  if(window.station3D?.ready){window.station3D.update(state.data,state.time,state.selected);return;}
  const map=$('map'),view=stationView,selected=state.data.agents.find(a=>a.agent_id===state.selected);
  const target=selected?position(selected,state.time):null;
  if(view.follow && target) {
    if(target.status==='evacuated') {view.follow=false;view.zoom=1;view.center=null;}
    else {
      if(!view.center||view.lastTime===null||Math.abs(state.time-view.lastTime)>2)view.center={...target.pos};
      else view.center={x:view.center.x+(target.pos.x-view.center.x)*.25,y:view.center.y+(target.pos.y-view.center.y)*.25};
    }
  }
  view.lastTime=state.time;
  const rect=map.getBoundingClientRect(),aspect=rect.width/Math.max(1,rect.height);
  const w=Math.max(1100,1050*aspect)/view.zoom,h=w/aspect,center=view.center||{x:550,y:525};
  const vx=center.x-w/2,vy=center.y-h/2,units=w/Math.max(1,rect.width);
  map.setAttribute('viewBox',`${vx} ${vy} ${w} ${h}`);
  const scale=$('viewportScale');scale.replaceChildren();
  const meters=[5,10,20,50,100].find(m=>m*stationScale/units>=60)||100;
  const sx=vx+w-20*units-meters*stationScale,sy=vy+h-42*units;
  svg('line',{x1:sx,y1:sy,x2:sx+meters*stationScale,y2:sy,stroke:'#cce0db','stroke-width':2*units},'',scale);
  svg('text',{x:sx,y:sy-7*units,fill:'#cce0db','font-size':11*units},`${meters}m`,scale);
  for(const agent of state.data.agents) {
    const p=position(agent,state.time),g=$(`agent-${agent.agent_id}`);
    g.style.display=p.status==='evacuated'?'none':'';
    g.setAttribute('transform',`translate(${p.pos.x},${p.pos.y})`);
    const onStairs=p.status==='on_stairs',circle=g.querySelector('circle');
    circle.setAttribute('r',(onStairs?3.3:agent.agent_id===state.selected?7:5)*units);
    circle.setAttribute('cy',onStairs?-6*units:0);
    circle.setAttribute('stroke-width',(agent.agent_id===state.selected?2:1)*units);
    let body=g.querySelector('.person-body');
    if(!body)body=svg('path',{class:'person-body',fill:'none','stroke-linecap':'round'},'',g);
    body.style.display=onStairs?'':'none';body.setAttribute('stroke',colors.on_stairs);body.setAttribute('stroke-width',2*units);
    const step=Math.sin(state.time*8+agent.agent_id)*3;
    body.setAttribute('d',`M0 ${-2*units}V${4*units}M${-4*units} ${units}L0 ${-units}L${4*units} ${units}M0 ${4*units}L${(-2+step)*units} ${10*units}M0 ${4*units}L${(2-step)*units} ${10*units}`);
    const text=g.querySelector('text');text.setAttribute('font-size',11*units);text.setAttribute('y',-13*units);
    text.style.display=agent.agent_id===state.selected||view.zoom>=3?'':'none';
  }
  if(target) {$('selectedMiniAgent').setAttribute('cx',target.pos.x);$('selectedMiniAgent').setAttribute('cy',target.pos.y);$('selectedMiniAgent').style.display=target.status==='evacuated'?'none':'';}
  const frame=$('cameraMiniFrame');
  for(const [key,value] of Object.entries({x:vx,y:vy,width:w,height:h}))frame.setAttribute(key,value);
}
