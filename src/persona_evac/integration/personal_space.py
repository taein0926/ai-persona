"""Private observations and remembered space. No crowd plans or global route oracle."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import heapq
import json
import math
import re

from shapely import prepare
from shapely.geometry import LineString,Point,Polygon,box
from shapely.ops import unary_union
from persona_evac.integration.station_geometry import floor_geometry
from persona_evac.agent_policy import ROOT,POLICY,VERSIONS
from persona_evac.hashing import canonical_json

SENSOR=POLICY['sensor'];GRID=SENSOR['grid_m']
FRONTIER_REVISIT_DELAY_S=10.
PRIOR=json.loads((ROOT/'configs/agents/gaehwasan_prior.json').read_text(encoding='utf-8'))
PRIOR_HASH=hashlib.sha256(canonical_json(PRIOR).encode()).hexdigest()

def residence_basis(persona):
    province=str(persona.get('province',''));district=str(persona.get('district',''))
    structured=province.startswith('서울') and bool(re.fullmatch(r'(?:서울(?:특별시)?[-\s]*)?강서구',district.strip()))
    basis=[];explicit_locations=[]
    if structured:basis.append({'field':'province/district','text':province+' / '+district,'kind':'structured_residence'})
    # Only explicit current living clauses. Hometown, work and travel do not qualify.
    for field in ('family_persona','persona','current_residence','residence'):
        text=persona.get(field,'')
        if not isinstance(text,str):continue
        for sentence in re.findall(r'[^.!?\n]+[.!?]?',text):
            for match in re.finditer(r'(서울(?:특별시)?|부산(?:광역시)?|인천(?:광역시)?|대구(?:광역시)?)\s*([가-힣]+[구시군]).{0,35}?(?:거주|살고|살며|살아가|생활하고)',sentence):
                if re.search(r'고향|태어나|출신|직장|근무|출퇴근|방문',match.group()):continue
                location=(match.group(1),match.group(2));explicit_locations.append(location)
                if location[0].startswith('서울') and location[1]=='강서구':
                    basis.append({'field':field,'text':match.group(),'kind':'explicit_current_residence'})
    if explicit_locations and not any(city.startswith('서울') and district=='강서구' for city,district in explicit_locations):basis=[]
    return {'level':'limited' if basis else 'first_visit','evidence':basis,'prior_version':PRIOR['version'],'prior_hash':PRIOR_HASH,
            'explicit_current_locations':explicit_locations,
            'assumption':PRIOR['basis'] if basis else '주거 근거가 확인되지 않아 초행으로 시작합니다.',
            'distance_to_station_m':None,'current_conditions_known':False}

def cell_key(p):return (round(p[0]/GRID),round(p[1]/GRID))
def cell_pos(k):return (k[0]*GRID,k[1]*GRID)
def key_string(k):return f'{k[0]},{k[1]}'
def smoke_label(k):return 'opaque' if k>=1 else 'dense' if k>=.3 else 'light' if k>=.05 else 'clear'
def heat_label(t):return 'hot' if t>=60 else 'warm' if t>=40 else 'normal'
def hazard_level(h):return max({'clear':0,'light':1,'dense':2,'opaque':3}[h['smoke']],{'normal':0,'warm':1,'hot':3}[h['heat']])


def familiarity_basis(persona, mapping=None):
    estimate=(mapping or {}).get('familiarity')
    if estimate is None:return residence_basis(persona)
    from persona_evac.schemas import StationFamiliarity
    estimate=StationFamiliarity.model_validate(estimate)
    return {'level':estimate.level, 'assumption':estimate.assumption,
            'classification_source':'llm', 'prior_version':PRIOR['llm_familiarity']['version'],
            'prior_hash':PRIOR_HASH, 'distance_to_station_m':None, 'current_conditions_known':False}


def station_stair_memories(nodes, stairs):
    """Only static stair entrances and floor links, without hazards or route maps."""
    result=[]
    for sid,stair in sorted(stairs.items()):
        origin=nodes[stair['from_node']];destination=nodes[stair['to_node']]
        result.append({'id':sid, 'area':origin.get('area','hall'), 'floor':origin['floor'],
            'position':[origin['x'],origin['y']], 'to_floor':destination['floor'],
            'to_area':destination.get('area'), 'kind':'fare_gate' if stair.get('kind')=='fare_gate' else 'exit_stair' if destination['floor']=='OUTSIDE' else 'up_stair'})
    return result


def station_route_memories(areas,landmarks):
    """Static structure only; no hazard, crowd, agent or live passage state."""
    maps={}
    for area,record in areas.items():
        geometry=record['geometry'];boundary=geometry.boundary
        x0,y0,x1,y1=geometry.bounds
        keys={(x,y) for x in range(math.floor(x0/GRID)-1,math.ceil(x1/GRID)+2)
              for y in range(math.floor(y0/GRID)-1,math.ceil(y1/GRID)+2)}
        cells={k for k in keys if geometry.covers(Point(cell_pos(k))) and boundary.distance(Point(cell_pos(k)))>=.22}
        edges={}
        for k in sorted(cells):
            check=observed_connector_checker(geometry,cell_pos(k),boundary)
            for dx,dy in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                n=(k[0]+dx,k[1]+dy)
                if n in cells and check(cell_pos(n)):edges.setdefault(k,{})[n]=math.dist(cell_pos(k),cell_pos(n))
        links={}
        for landmark in landmarks:
            if landmark['area']!=area:continue
            links[landmark['id']]=[k for k in sorted(cells) if math.dist(cell_pos(k),landmark['position'])<=GRID*1.5
                and observed_connector_checker(geometry,cell_pos(k),boundary)(landmark['position'])]
        maps[area]={'cells':cells,'blocked':keys-cells,'blocked_frontier_edges':set(),'edges':edges,'landmark_links':links,
                    'cell_first_seen':{k:0. for k in sorted(cells)},
                    'edge_first_seen':{(k,n):0. for k,neighbors in edges.items() for n in neighbors},
                    'initial_source':'remembered_station_structure'}
    return maps

def observed_connector(geometry,start,end):
    """Allow recovery from the unchanged engine's soft wall contact, never deepen it.

    This changes neither body radius nor engine collision rules. Navigation cannot
    freeze just because an AVM center is already slightly inside its comfort margin.
    """
    return observed_connector_checker(geometry,start)(end)


def observed_connector_checker(geometry,start,boundary=None):
    """한 위치의 후보 검사에서만 공통 벽 거리를 재사용한다."""
    start=tuple(start);clearance=None
    def check(end):
        nonlocal clearance
        line=LineString([start,end])
        if not geometry.covers(line):return False
        if clearance is None:
            edge=geometry.boundary if boundary is None else boundary
            clearance=max(0,min(.22,edge.distance(Point(start))-.00001))
        return geometry.covers(line.buffer(clearance)) if clearance else True
    return check


class PersonalSpace:
    def __init__(self,persona,mapping=None,prior_landmarks=()):
        self.uuid=persona['uuid'];self.familiarity=familiarity_basis(persona,mapping)
        self.landmarks={};self.maps={};self.prior_maps={};self.observations=[];self.last_visible_people=[]
        self._observation_index={};self._indexed_observations=self.observations;self._indexed_observation_count=0
        self.last_risk={'smoke':'clear','heat':'normal'};self.last_observation_ids=[];self.current_origin_links=[]
        self.last_sense=-1e9;self.revision=0;self.visits={};self.traversals=[]
        self.local_ids={}
        self.people_track={}
        self.people_observations=[]
        self.visible_landmark_ids=set();self.people_view_active=False
        self._shortest_cache=None
        self.direct_landmark_links={}
        self.reached_frontiers={}
        if self.familiarity['level'] in {'limited','familiar'}:
            prior=prior_landmarks if self.familiarity['level']=='familiar' else PRIOR['landmarks']
            for item in prior:
                self.landmarks[item['id']]={**deepcopy(item),'source':'familiarity_prior' if self.familiarity['level']=='familiar' else 'residential_prior','observed_at_s':None,
                    'current_state':None,'observation_id':None,'anchors':[]}
                if self.familiarity['level']=='limited':self.landmarks[item['id']]['position']=list(cell_pos(cell_key(item['position'])))
                self.local_ids[item['id']]=f'L{len(self.local_ids)+1}'
        self.initial_knowledge=deepcopy(self.landmarks)

    def record(self,now,kind,data,source='vision'):
        item={'id':f'O{len(self.observations)+1}','time_s':round(now,3),'source':source,'kind':kind,**data}
        self._ensure_observation_index()
        position=len(self.observations)
        self.observations.append(item)
        self._observation_index.setdefault(item['id'],[]).append((position,item))
        self._indexed_observation_count=position+1
        return item['id']

    def _ensure_observation_index(self):
        # Histories restored/replaced outside record() are rebuilt once. Normal
        # appends update the private index directly without scanning old records.
        if self._indexed_observations is not self.observations or self._indexed_observation_count!=len(self.observations):
            self._observation_index={}
            for position,item in enumerate(self.observations):
                self._observation_index.setdefault(item['id'],[]).append((position,item))
            self._indexed_observations=self.observations
            self._indexed_observation_count=len(self.observations)

    def observations_for_ids(self,ids):
        """Return the same records/order as filtering history by an ID set."""
        self._ensure_observation_index()
        matches=[entry for oid in set(ids) for entry in self._observation_index.get(oid,())]
        matches.sort(key=lambda entry:entry[0])
        return [item for _,item in matches]

    def traversed(self,sid,area,floor,pos,now):
        self._shortest_cache=None
        old=self.landmarks.get(sid)
        if old:
            old.update(to_floor=floor,to_area=area,connection_source='own_traversal')
            if floor=='OUTSIDE':old['kind']='exit_stair'
            oid=self.record(now,'connection_learned',{'landmark_id':sid,'to_floor':floor,'to_area':area},'own_traversal')
            old['connection_observation_id']=oid
        self.traversals.append({'stair_id':sid,'area':area,'floor':floor,'time_s':now,'position':list(pos)})

    def map_for(self,area):
        if area not in self.maps:
            self.maps[area]=deepcopy(self.prior_maps[area]) if area in self.prior_maps else {
                'cells':set(),'blocked':set(),'blocked_frontier_edges':set(),'edges':{},'landmark_links':{},'cell_first_seen':{},'edge_first_seen':{}}
        return self.maps[area]

    def shortest(self,area,pos):
        m=self.map_for(area);dist={};previous={};heap=[]
        for k in self.current_origin_links:
            if k in m['cells']:
                d=math.dist(pos,cell_pos(k));dist[k]=d;heapq.heappush(heap,(d,k))
        while heap:
            d,k=heapq.heappop(heap)
            if d!=dist.get(k):continue
            for n,length in m['edges'].get(k,{}).items():
                if d+length<dist.get(n,math.inf):dist[n]=d+length;previous[n]=k;heapq.heappush(heap,(d+length,n))
        return dist,previous

    def mark_frontier_reached(self,area,position,now):
        """현재 보이는 탐색 공간에 도달한 경험만 개인별로 기록한다."""
        key=tuple(value/GRID for value in position)
        self.reached_frontiers[area,key]=now
        self.visits[area,key]=self.visits.get((area,key),0)+1

    def options(self,area,pos,speed,now,current_target=None):
        """Only this private map and its observations are read here."""
        m=self.map_for(area);out=[]
        # Reuse only the graph search, never time-dependent risks or candidate costs.
        # Every sensor refresh invalidates this private cache, including refreshes
        # that only change edges/blocked frontiers without discovering new cells.
        search_key=(area,tuple(pos),tuple(self.current_origin_links))
        if self._shortest_cache is None or self._shortest_cache[0]!=search_key:
            dist,previous=self.shortest(area,pos)
            self._shortest_cache=(search_key,dist,previous)
        else:
            _,dist,previous=self._shortest_cache
        def path(k):
            result=[k]
            while k in previous:k=previous[k];result.append(k)
            return [list(cell_pos(p)) for p in reversed(result)]
        for sid,l in sorted(self.landmarks.items()):
            if l['area']!=area:continue
            anchors=[k for k in m['landmark_links'].get(sid,[]) if k in dist]
            direct=self.direct_landmark_links.get(sid)
            direct=direct and direct['area']==area and now-direct['time_s']<=SENSOR['current_observation_ttl_s']
            if not anchors and not direct:continue
            anchor=min(anchors,key=lambda k:dist[k]+math.dist(cell_pos(k),l['position'])) if anchors else None
            length=dist[anchor]+math.dist(cell_pos(anchor),l['position']) if anchors else math.inf
            direct=direct and not anchors
            if direct:length=math.dist(pos,l['position'])
            age=None if l['observed_at_s'] is None else round(now-l['observed_at_s'],1)
            current=sid in self.visible_landmark_ids and age is not None and age<=SENSOR['current_observation_ttl_s']
            state=l.get('current_state') or {}
            out.append({'id':sid,'kind':l['kind'],'position':l['position'],'known_to_floor':l.get('to_floor'),
                'distance_m':round(length,2),'walking_s':round(length/max(.1,speed),2),
                'visible_people':state.get('people') if current else None,
                'hazard':state.get('hazard') if current else None,
                'past_state':None if current else state or None,'state_is_current':current,'age_s':age,
                'source':l['source'],'observation_ids':[x for x in (l.get('observation_id'),l.get('connection_observation_id')) if x],
                'path':([l['position']] if direct else path(anchor)+[l['position']])})
        # Use a known, privately reachable stair, exit or fare gate before
        # searching for an unknown alternative. This also applies to first visitors.
        if out:
            for option in out:option['observed_flow']=self.observed_flow(area,pos,now,option)
            return out
        # Frontiers are boundaries of the observed map, never hidden destination coordinates.
        frontiers=[];adaptive=m.get('adaptive',False);step=.25 if adaptive else 1
        known_cells=m['cells']|m['blocked']
        known_targets=[l['position'] for l in self.landmarks.values() if l['area']==area]
        for k,d in dist.items():
            if not adaptive and any(value!=round(value) for value in k):continue
            if now-self.reached_frontiers.get((area,k),-1e9)<FRONTIER_REVISIT_DELAY_S:continue
            unseen=sum((k[0]+dx*step,k[1]+dy*step) not in known_cells and (k,(k[0]+dx*step,k[1]+dy*step)) not in m['blocked_frontier_edges'] for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)))
            if not unseen or d<(.2 if adaptive else 2.5):continue
            visited=self.visits.get((area,k),0)
            prior_bearing=min((math.dist(cell_pos(k),p) for p in known_targets),default=0)
            # Locally known novelty and remembered bearing only; no global exit direction.
            score=visited*30+d*.12+prior_bearing*.15-unseen*2
            frontiers.append((score,k,d,unseen,visited))
        selected=[]
        for _,k,d,unseen,visited in sorted(frontiers):
            if any(math.dist(cell_pos(k),cell_pos(other))<(1. if adaptive else 8) for other in selected):continue
            selected.append(k)
            progress=min((math.dist(pos,p) for p in known_targets),default=0)-min((math.dist(cell_pos(k),p) for p in known_targets),default=0)
            out.append({'id':'explore:'+key_string(k),'kind':'frontier','position':list(cell_pos(k)),
                'arrival_radius_m':.4 if adaptive else 1.,
                'known_to_floor':None,'distance_m':round(d,2),'walking_s':round(d/max(.1,speed),2),
                'visible_people':None,'hazard':None,'past_state':None,'state_is_current':False,'age_s':None,
                'source':'personal_observed_map_frontier','observation_ids':self.last_observation_ids[-1:],
                'novelty':unseen,'visited':visited,'remembered_bearing_progress_m':round(progress,2),'path':path(k)})
            if len(selected)==2:break
        if current_target and current_target.startswith('explore:') and current_target!='explore:none' and not any(p['id']==current_target for p in out):
            k=tuple(map(float,current_target.split(':')[1].split(',')))
            if k in dist and now-self.reached_frontiers.get((area,k),-1e9)>=FRONTIER_REVISIT_DELAY_S:
                out.append({'id':current_target,'kind':'frontier','position':list(cell_pos(k)),
                    'arrival_radius_m':.4 if adaptive else 1.,
                    'known_to_floor':None,'distance_m':round(dist[k],2),'walking_s':round(dist[k]/max(.1,speed),2),
                    'visible_people':None,'hazard':None,'past_state':None,'state_is_current':False,'age_s':None,
                    'source':'remembered_exploration_goal','observation_ids':self.last_observation_ids[-1:],
                    'novelty':0,'visited':self.visits.get((area,k),0),'path':path(k)})
        for option in out:
            option['observed_flow']=self.observed_flow(area,pos,now,option)
        return out

    def observed_flow(self,area,pos,now,option):
        """Compare a known local path with motion measured by this observer only."""
        if not self.people_view_active or not self.people_observations:return None
        view=self.people_observations[-1]
        if not 0<=now-view['time_s']<=SENSOR['current_observation_ttl_s']:return None
        heading=next((p for p in option.get('path',[]) if math.dist(pos,p)>.75),option['position'])
        dx,dy=heading[0]-pos[0],heading[1]-pos[1];distance=math.hypot(dx,dy)
        if distance<.01:return None
        alignments=[];observed=0
        for person in self.last_visible_people:
            velocity=person.get('observed_velocity_mps')
            if person.get('area')!=area or velocity is None:continue
            speed=math.hypot(*velocity)
            if speed<.1:continue
            observed+=1;alignment=(velocity[0]*dx+velocity[1]*dy)/(speed*distance)
            if alignment>.5:alignments.append(alignment)
        if not observed:return None
        strength=sum(alignments)/observed*min(1,len(alignments)/3)
        return {'people':len(alignments),'strength':round(strength,4),'observed_at_s':view['time_s'],
                'observation_id':view['id'],'source':'own_observed_movement'}

    def snapshot(self,area,now):
        landmarks=[]
        for sid,l in sorted(self.landmarks.items()):
            age=None if l['observed_at_s'] is None else round(now-l['observed_at_s'],1)
            landmarks.append({k:deepcopy(l.get(k)) for k in ('id','area','floor','position','to_floor','to_area','kind','source','observed_at_s','current_state','observation_id')})
            landmarks[-1]['state_is_current']=sid in self.visible_landmark_ids and age is not None and age<=SENSOR['current_observation_ttl_s']
        return {'version':VERSIONS['knowledge'],'revision':self.revision,'area':area,
                'local_ids':dict(self.local_ids),
                'known_landmarks':landmarks,'known_cells':len(self.map_for(area)['cells']),
                'visible_people':deepcopy(self.last_visible_people) if self.people_view_active else None,'perceived_hazard':self.last_risk,
                'people_view_active':self.people_view_active,
                'people_observed_at_s':self.people_observations[-1]['time_s'] if self.people_observations else None,
                'people_observation_id':self.people_observations[-1]['id'] if self.people_observations else None,
                'observation_ids':list(self.last_observation_ids)}


class OccludedSensor:
    """Full geometry/FDS is read only by the sensor, not by private route costs."""
    def __init__(self,areas,nodes,stairs,fire=None,signs=(),physical_config=None):
        self.areas,self.nodes,self.stairs,self.fire=areas,nodes,stairs,fire
        self.signs=signs;self.static_cache={};self.hazard_cache={};self.frame=None
        self.edge_lengths={};self.grid_coverage={}
        # Wall predicates are static. Dynamic samples use exact coordinates and
        # the current frame so one ray cannot replace a nearby ray's FDS sample.
        self._static_ray_covered=lru_cache(maxsize=65536)(self._ray_covered)
        self._static_wall_face=lru_cache(maxsize=65536)(self._wall_face)
        self._static_landmark_link=lru_cache(maxsize=16384)(self._landmark_link_covered)
        self._observer_point=lru_cache(maxsize=1024)(Point)
        self.boundaries={area:a['geometry'].boundary for area,a in areas.items()}
        self.visibility={};self.config=physical_config or {};self.geometry_by_floor={}
        for floor in self.config.get('floors',[]):
            geom=floor_geometry(floor)
            if floor['id']=='B2' and self.config.get('train'):
                train=self.config['train']
                geom=unary_union([geom,Polygon(train['track_polygon'])])
                solids=[]
                # Same continuous car-shell and open-door construction as the
                # existing FDS input generator. This is a perception projection.
                for car in train['cars']:
                    a,b=car['a'],car['b'];axis=LineString([a,b]);shell=axis.buffer(car['width_m']/2,cap_style=2).difference(axis.buffer(.75,cap_style=2))
                    dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy);nx,ny=-dy/length,dx/length
                    for fraction in (.2,.5,.8):
                        x=a[0]+fraction*dx+nx*car['width_m']/2;y=a[1]+fraction*dy+ny*car['width_m']/2
                        shell=shell.difference(box(x-1.1,y-.9,x+1.1,y+.9))
                    solids.append(shell)
                for obstacle in floor.get('obstacles',[]):
                    if 'polygon' in obstacle:solids.append(Polygon(obstacle['polygon']))
                    elif 'rect' in obstacle:
                        x,y,w,h=obstacle['rect'];solids.append(box(x,y,x+w,y+h))
                geom=geom.difference(unary_union(solids))
            self.geometry_by_floor[floor['id']]=geom
        for geom in [a['geometry'] for a in areas.values()]+list(self.geometry_by_floor.values()):
            prepare(geom)

    def vision_geometry(self,area,floor):return self.geometry_by_floor.get(floor,self.areas[area]['geometry'])

    def _ray_covered(self,area,floor,p,q):
        return self.vision_geometry(area,floor).covers(LineString([p,q]))

    def _landmark_link_covered(self,area,k,position):
        return self.areas[area]['geometry'].covers(LineString([cell_pos(k),position]).buffer(.24))

    def landmark_links(self,area,seen,position):
        """Reuse only static coverage; rebuild from the current seen cells in order."""
        position=tuple(position)
        return [k for k in seen if math.dist(cell_pos(k),position)<3.5
                and self._static_landmark_link(area,k,position)]

    def _wall_face(self,area,p,k):
        """Static intersection only; visibility/smoke is checked by the caller."""
        boundary=self.boundaries[area];line=LineString([p,cell_pos(k)])
        qpoint=self.grid_coverage[area,k][0]
        intersections=line.intersection(boundary)
        if not intersections.is_empty and qpoint.distance(intersections)<GRID*.7:
            # Preserve the original reversed intersection and nearest-point tie order.
            first=boundary.intersection(line)
            points=[g for g in getattr(first,'geoms',[first]) if g.geom_type=='Point']
            if points:
                edge=min(points,key=self._observer_point(p).distance)
                if qpoint.distance(edge)<GRID*.7:return edge.x,edge.y
        return None

    def physical(self,floor,pos,now):
        key=(floor,*pos)
        if self.frame!=now:self.hazard_cache={};self.frame=now
        if key not in self.hazard_cache:
            self.hazard_cache[key]=self.fire.sample(floor,pos,now) if self.fire else {'extinction_per_m':0,'temperature_c':20}
        return self.hazard_cache[key]

    def visible(self,area,floor,p,q,now):
        distance=math.dist(p,q)
        if distance>SENSOR['range_m'] or not self._static_ray_covered(area,floor,tuple(p),tuple(q)):return False
        if self.fire and distance>.1:
            n=max(1,math.ceil(distance/SENSOR['ray_sample_spacing_m']));contrast=0
            for i in range(n):
                xy=tuple(p[j]+(q[j]-p[j])*(i+.5)/n for j in (0,1))
                contrast+=self.physical(floor,xy,now)['extinction_per_m']*distance/n
                if contrast>SENSOR['smoke_contrast_budget']:return False
        return True

    def sense(self,space,r,runtime,now):
        area,floor,p=r['area'],r['floor'],r['pos'];geom=self.areas[area]['geometry'];m=space.map_for(area)
        space._shortest_cache=None
        boundary=self.boundaries[area]
        raw=self.physical(floor,p,now);hazard={'smoke':smoke_label(raw['extinction_per_m']),'heat':heat_label(raw['temperature_c'])}
        old_risk=space.last_risk;space.last_risk=hazard
        center=cell_key(p);radius=math.ceil(SENSOR['range_m']/GRID);seen=set();blocked=set()
        for dx in range(-radius,radius+1):
            for dy in range(-radius,radius+1):
                k=(center[0]+dx,center[1]+dy);q=cell_pos(k);distance=math.dist(p,q)
                if distance>SENSOR['range_m']:continue
                grid_key=(area,k)
                if grid_key not in self.grid_coverage:
                    qpoint=Point(q);self.grid_coverage[grid_key]=(qpoint,geom.covers(qpoint))
                qpoint,covered=self.grid_coverage[grid_key]
                if covered and self.visible(area,floor,p,q,now):seen.add(k)
                elif distance>0:
                    edge=self._static_wall_face(area,tuple(p),k)
                    if edge is not None and self.visible(area,floor,p,edge,now):blocked.add(k)
        new=seen-m['cells'];m['cells']|=seen;m['blocked']|=blocked
        for k in new:m['cell_first_seen'][k]=now
        for k in seen:
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                n=(k[0]+dx,k[1]+dy)
                edge_key=('edge',area,k,n)
                if n in seen and edge_key not in self.static_cache:
                    line=LineString([cell_pos(k),cell_pos(n)])
                    self.static_cache[edge_key]=geom.covers(line.buffer(.26))
                    self.edge_lengths[edge_key]=line.length
                if (n in seen and self.static_cache[edge_key] and
                        all(self.visible(area,floor,p,tuple(cell_pos(k)[i]+fraction*(cell_pos(n)[i]-cell_pos(k)[i]) for i in (0,1)),now)
                            for fraction in (.25,.5,.75))):
                    m['edges'].setdefault(k,{})[n]=self.edge_lengths[edge_key]
                    m['edge_first_seen'].setdefault((k,n),now)
                    m['blocked_frontier_edges'].discard((k,n))
                elif n not in seen:
                    boundary_key=('boundary',area,k,n)
                    if boundary_key not in self.static_cache:
                        line=LineString([cell_pos(k),cell_pos(n)])
                        intersection=line.intersection(boundary)
                        self.static_cache[boundary_key]=[x for x in getattr(intersection,'geoms',[intersection]) if x.geom_type=='Point']
                    points=self.static_cache[boundary_key]
                    if any(self.visible(area,floor,p,(x.x,x.y),now) for x in points):
                        m['blocked_frontier_edges'].add((k,n));m['edges'].get(k,{}).pop(n,None);m['edges'].get(n,{}).pop(k,None)
        connector=observed_connector_checker(geom,p,boundary)
        narrow=max((math.dist(p,cell_pos(k)) for k in seen),default=0)<2.5
        if m.get('adaptive') or space.familiarity['level']=='first_visit':
            m['adaptive']=m.get('adaptive',False) or narrow;fine_seen=set()
            cx,cy=round(p[0]*2),round(p[1]*2)
            for ix in range(cx-5,cx+6):
                for iy in range(cy-5,cy+6):
                    k=(ix/4,iy/4);q=cell_pos(k)
                    if math.dist(p,q)>2.5:continue
                    visible=self.visible(area,floor,p,q,now)
                    if visible and boundary.distance(Point(q))>=.22:fine_seen.add(k)
                    elif visible:m['blocked'].add(k)
                    else:
                        intersections=LineString([p,q]).intersection(boundary)
                        if any(self.visible(area,floor,p,(x.x,x.y),now) for x in getattr(intersections,'geoms',[intersections]) if x.geom_type=='Point'):
                            m['blocked'].add(k)
            fine_new=fine_seen-m['cells'];new|=fine_new;m['cells']|=fine_seen
            for k in fine_new:m['cell_first_seen'][k]=now
            for k in fine_seen:
                check=observed_connector_checker(geom,cell_pos(k),boundary)
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                    n=(k[0]+dx*.25,k[1]+dy*.25)
                    midpoint=tuple((cell_pos(k)[i]+cell_pos(n)[i])/2 for i in (0,1))
                    if n in fine_seen and check(cell_pos(n)) and self.visible(area,floor,p,midpoint,now):
                        m['edges'].setdefault(k,{})[n]=math.dist(cell_pos(k),cell_pos(n))
                        m['edge_first_seen'].setdefault((k,n),now)
            seen|=fine_seen
        # A loss of sight does not erase a previously observed route. Familiar
        # people use their static map; first visitors use only their own cells.
        # Actual movement still checks the visible prefix on every step.
        origin_candidates=m['cells']
        space.current_origin_links=[k for k in sorted(origin_candidates) if math.dist(p,cell_pos(k))<3.1 and connector(cell_pos(k))]
        visible_people=[]
        space.direct_landmark_links={}
        for aid,other in runtime.items():
            if other is r or other.get('floor')!=floor or other['phase'] in {'on_stairs','evacuated'}:continue
            if self.visible(area,floor,p,other['pos'],now):
                previous=space.people_track.get(aid);estimate=None;velocity=None
                if (previous and previous.get('floor',floor)==floor and previous.get('area',other['area'])==other['area']
                    and 0<now-previous['time_s']<=SENSOR['current_observation_ttl_s']):
                    elapsed=now-previous['time_s']
                    velocity=[(other['pos'][i]-previous['position'][i])/elapsed for i in (0,1)]
                    estimate=math.hypot(*velocity)
                space.people_track[aid]={'time_s':now,'position':list(other['pos']),'estimated_speed_mps':estimate,
                                        'floor':floor,'area':other['area']}
                visible_people.append({'id':aid,'position':[round(x,2) for x in other['pos']],
                    'observed_speed_mps':None if estimate is None else round(estimate,2),
                    'observed_velocity_mps':None if velocity is None else [round(v,3) for v in velocity],
                    'area':other['area'],'observed_at_s':now,'source':'vision',
                    'speed_source':'own_observed_displacement' if estimate is not None else 'not_yet_observed'})
        space.people_observations.append({'id':f'V{len(space.people_observations)+1}','time_s':now,'source':'vision',
                                          'floor':floor,'observer_position':list(p),'people':deepcopy(visible_people)})
        space.last_visible_people=visible_people;space.people_view_active=True
        space.visible_landmark_ids=set();discovered=[];current_ids=[]
        landmark_risk_changed=False;landmark_critical=False
        smoke_order={'clear':0,'light':1,'dense':2,'opaque':3}
        heat_order={'normal':0,'warm':1,'hot':3,'unknown':0}
        for sid,s in self.stairs.items():
            node=self.nodes[s['from_node']]
            if node['floor']!=floor or not self.visible(area,floor,p,(node['x'],node['y']),now):continue
            position=[node['x'],node['y']];old=space.landmarks.get(sid);new_landmark=old is None
            if sid not in space.local_ids:space.local_ids[sid]=f'L{len(space.local_ids)+1}'
            space.visible_landmark_ids.add(sid)
            hraw=self.physical(floor,position,now)
            h={'smoke':smoke_label(hraw['extinction_per_m']),'heat':hazard['heat'] if math.dist(p,position)<=2 else 'unknown'}
            state={'people':sum(o['area']==node.get('area') and math.dist(o['position'],position)<5 for o in visible_people),'hazard':h}
            previous=(old or {}).get('current_state')
            previous_hazard=(previous or {}).get('hazard')
            if previous_hazard is not None and previous_hazard!=h:landmark_risk_changed=True
            old_h=previous_hazard or {'smoke':'clear','heat':'unknown'}
            if (smoke_order[h['smoke']]>smoke_order[old_h['smoke']] and smoke_order[h['smoke']]>=2 or
                heat_order[h['heat']]>heat_order[old_h['heat']] and heat_order[h['heat']]>=3):landmark_critical=True
            oid=(old or {}).get('observation_id')
            oid=space.record(now,'landmark_seen',{'landmark_id':sid,'area':node.get('area'),'floor':floor,'position':position,
                'observed_same_walkway':node.get('area')==area,'perceived_state':state})
            # Seeing an ascending stair does not reveal its destination floor/exit.
            kind=(old or {}).get('kind','fare_gate' if s.get('kind')=='fare_gate' else 'up_stair')
            space.landmarks[sid]={**(old or {}),'id':sid,'area':node.get('area'),'floor':floor,'position':position,'kind':kind,
                'source':'vision','observed_at_s':now,'current_state':state,'observation_id':oid}
            if node.get('area')==area:
                links=self.landmark_links(area,seen,position)
                m['landmark_links'][sid]=links
                if connector(position):space.direct_landmark_links[sid]={'area':area,'time_s':now}
            if oid:current_ids.append(oid)
            if new_landmark:discovered.append(sid)
        for sign in self.signs:
            if sign.get('area')!=area or sign.get('floor',floor)!=floor or not self.visible(area,floor,p,sign['position'],now):continue
            oid=space.record(now,'implemented_sign_seen',{'sign_id':sign['id'],'content':sign['content']})
            current_ids.append(oid)
        current_ids.append(space.record(now,'local_hazard',{'area':area,'floor':floor,'position':list(p),'perceived':hazard},'local_sensation'))
        if new:current_ids.append(space.record(now,'space_seen',{'area':area,'floor':floor,'new_cells':len(new)},'vision'))
        space.last_observation_ids=current_ids;space.last_sense=now;space.revision+=bool(new or discovered or hazard!=old_risk)
        if m.get('adaptive'):center=(round(p[0]*2)/4,round(p[1]*2)/4)
        space.visits[area,center]=space.visits.get((area,center),0)+1
        smoke_order={'clear':0,'light':1,'dense':2,'opaque':3};heat_order={'normal':0,'warm':1,'hot':3}
        smoke_up=smoke_order[hazard['smoke']]>smoke_order[old_risk['smoke']]
        heat_up=heat_order[hazard['heat']]>heat_order[old_risk['heat']]
        return {'discovered':discovered,'map_expanded':bool(new),'risk_increased':smoke_up or heat_up,
                'risk_changed':hazard!=old_risk or landmark_risk_changed,
                'critical':landmark_critical or smoke_up and smoke_order[hazard['smoke']]>=2 or heat_up and heat_order[hazard['heat']]>=3}
