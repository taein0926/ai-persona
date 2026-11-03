"""Layered JuPedSim connectors. Building a network never advances its clock.

JuPedSim is planar: each stair branch is an unfolded (distance, lateral) walking
domain. Its native positions are mapped onto the surveyed display path in XYZ.
Gate channels have independent capacity-one ownership, retained during spillback.
"""
from copy import deepcopy
import math

from shapely.geometry import LineString,Point,Polygon,box
from shapely.ops import unary_union

from persona_evac.integration.station_geometry import floor_geometry
from persona_evac.integration.station_layout import view_geometry,stair_path


def path_length(path):
    return sum(math.dist(a,b) for a,b in zip(path,path[1:]))


def path_position(path, distance, lateral=0):
    """Map an unfolded coordinate to a branch, preserving its vertical profile."""
    return PreparedPath(path).position(distance,lateral)


class PreparedPath:
    """Fixed branch geometry; preserve the original arithmetic and segment order."""
    def __init__(self,path):
        self.path=tuple(tuple(p) for p in path)
        self.lengths=tuple(math.dist(a,b) for a,b in zip(self.path,self.path[1:]))
        self.length=sum(self.lengths)
        normals=[]
        for a,b in zip(self.path,self.path[1:]):
            dx,dy=b[0]-a[0],b[1]-a[1];flat=math.hypot(dx,dy)
            normals.append((-dy/flat,dx/flat) if flat else (0,1))
        vertex_normals=[normals[0]]
        for previous,following in zip(normals,normals[1:]):
            denominator=1+sum(a*b for a,b in zip(previous,following))
            vertex_normals.append(tuple((a+b)/denominator for a,b in zip(previous,following)) if denominator>.05 else following)
        vertex_normals.append(normals[-1])
        self.vertex_normals=tuple(vertex_normals)

    def position(self,distance,lateral=0):
        distance=max(0,min(distance,self.length))
        for index,(a,b) in enumerate(zip(self.path,self.path[1:])):
            length=self.lengths[index]
            if distance<=length+1e-9:
                t=distance/length if length else 0
                dx,dy=b[0]-a[0],b[1]-a[1]
                nx,ny=[(1-t)*self.vertex_normals[index][k]+t*self.vertex_normals[index+1][k] for k in (0,1)]
                return [a[0]+t*dx+lateral*nx,a[1]+t*dy+lateral*ny,a[2]+t*(b[2]-a[2])]
            distance-=length
        return list(self.path[-1])


class BentPath:
    """수평 회전은 엔진에서도 걷고 높이만 경사 구간에 따라 복원한다."""
    def __init__(self,path):
        self.path=tuple(tuple(p) for p in path)
        self.lengths=tuple(math.dist(a,b) for a,b in zip(self.path,self.path[1:]))
        self.flat_lengths=tuple(math.dist(a[:2],b[:2]) for a,b in zip(self.path,self.path[1:]))
        self.length=sum(self.lengths)
        self.line=LineString([p[:2] for p in self.path])

    def sample(self,xy):
        flat_distance=self.line.project(Point(xy));remaining=flat_distance;distance=0
        for index,(a,b) in enumerate(zip(self.path,self.path[1:])):
            flat=self.flat_lengths[index]
            if remaining<=flat+1e-9 or index==len(self.flat_lengths)-1:
                fraction=max(0,min(1,remaining/flat)) if flat else 0
                progress=(distance+fraction*self.lengths[index])/self.length
                z=a[2]+fraction*(b[2]-a[2])
                return progress,[xy[0],xy[1],z],flat/self.lengths[index]
            remaining-=flat;distance+=self.lengths[index]

    def geometry(self,width):
        start,next_point=self.path[:2];previous,end=self.path[-2:]
        first=self.flat_lengths[0];last=self.flat_lengths[-1]
        a=[start[k]-.6*(next_point[k]-start[k])/first for k in (0,1)]
        b=[end[k]+.6*(end[k]-previous[k])/last for k in (0,1)]
        # 높이가 다른 평행 계단을 2D에 놓을 때 경계가 붙어 지름길이
        # 생기지 않도록 0.1mm 수치 여유를 둔다. 실측 폭 변경이 아니다.
        return LineString([a,*[p[:2] for p in self.path],b]).buffer(width/2-1e-4,cap_style=2,join_style=2)


def has_horizontal_turn(path):
    directions=[]
    for a,b in zip(path,path[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
        if length:directions.append((dx/length,dy/length))
    return any(sum(x*y for x,y in zip(a,b))<1-1e-8 for a,b in zip(directions,directions[1:]))


def compile_network(config):
    """Compile structural openings, solid walls and gate-separated floor areas."""
    nodes=deepcopy(config['nodes']);connectors=deepcopy(config['stairs'])
    from persona_evac.integration.station import resolve_stair
    for s in connectors:resolve_stair(s)
    views=view_geometry(config,1)
    floors={f['id']:f for f in config['floors']}
    geometry={}
    for spec in config['areas']:
        floor=floors[spec['floor']]
        slab=unary_union([Polygon(p['outer'],p['holes']) for p in views[spec['floor']]])
        solid_walls=unary_union([Polygon(w['polygon']) for w in floor.get('walls',[])])
        ramps=[]
        for s in connectors:
            if s.get('kind')!='stairs' or nodes[s['from_node']]['floor']!=spec['floor']:continue
            flights=[f for f in s.get('flights',[]) if f.get('role')=='lower']
            if not flights:flights=[{'path_xyz':stair_path(config,s),'width_m':s['width_m']}]
            ramps.extend(LineString([p[:2] for p in f['path_xyz']]).buffer(f['width_m']/2,cap_style=2,join_style=2) for f in flights)
        geometry[spec['id']]=floor_geometry(floor,spec.get('region')).intersection(slab).difference(unary_union([solid_walls,*ramps]))
    specs=deepcopy(config['areas'])
    gates=config.get('fare_gates',[])
    if gates:
        center=gates[0]['center'];normal=gates[0]['normal'];tangent=gates[0]['tangent']
        half=gates[0]['depth_m']/2
        def half_plane(side):
            return Polygon([[center[k]+a*tangent[k]+side*b*normal[k] for k in (0,1)]
                            for a,b in [(-1000,half),(1000,half),(1000,1000),(-1000,1000)]])
        hall=geometry['hall']
        geometry['hall_paid']=hall.intersection(half_plane(-1))
        geometry['hall']=hall.intersection(half_plane(1))
        specs.append({'id':'hall_paid','floor':'B1'})
        for s in connectors:
            dest=nodes[s['to_node']]
            if nodes[s['from_node']]['floor']=='B2' and dest['floor']=='B1':dest['area']='hall_paid'
        for g in gates:
            offset=g['depth_m']/2+.36
            start=[g['center'][k]-offset*g['normal'][k] for k in (0,1)]
            end=[g['center'][k]+offset*g['normal'][k] for k in (0,1)]
            a,b=g['id']+'_in',g['id']+'_out'
            nodes[a]={'floor':'B1','area':'hall_paid','x':start[0],'y':start[1]}
            nodes[b]={'floor':'B1','area':'hall','x':end[0],'y':end[1]}
            connectors.append({'id':g['id'],'kind':'fare_gate','label':'개찰구 '+g['id'],
                'from_node':a,'to_node':b,'width_m':g['width_m'],'speed_mps':1.,
                'length_m':math.dist(start,end),'capacity':1,'parallel_lanes':1,
                'entry_rate_per_s':10.,'travel_time_s':math.dist(start,end),
                'portal_tangent':g['tangent'],'path_xyz':[[*start,4.],[*end,4.]]})
    for s in connectors:
        paths=s.get('branch_paths_xyz') or [stair_path(config,s)]
        if s.get('kind')=='stairs':
            # Transfer whole bodies onto the floor apron, not with their
            # centers only 15 cm from the upper shaft opening.
            extended=[]
            for path in paths:
                path=deepcopy(path);a,b=path[:2];c,d=path[-2:]
                first=math.hypot(b[0]-a[0],b[1]-a[1]);last=math.hypot(d[0]-c[0],d[1]-c[1])
                extended.append([[a[0]-.4*(b[0]-a[0])/first,a[1]-.4*(b[1]-a[1])/first,a[2]],
                                 *path,[d[0]+.4*(d[0]-c[0])/last,d[1]+.4*(d[1]-c[1])/last,d[2]]])
            paths=extended
        width=s['width_m']/len(paths)
        branch_lanes=max(1,int(width/.60)) if s['kind']!='fare_gate' else 1
        entries=[]
        for branch,path in enumerate(paths):
            prepared=PreparedPath(path)
            for i in range(branch_lanes):
                offset=(i-(branch_lanes-1)/2)*width/branch_lanes
                entries.append({'branch':branch,'offset':offset,
                    'entry':prepared.position(0,offset),
                    'destination':prepared.position(prepared.length,offset)})
        s['parallel_lanes']=len(entries);s['native_lane_specs']=entries
        s['native_paths']=paths;s['native_branch_width_m']=width
        if s.get('kind')=='stairs':
            nodes[s['from_node']].update(x=sum(p['entry'][0] for p in entries)/len(entries),
                                         y=sum(p['entry'][1] for p in entries)/len(entries))
    excluded={}
    from persona_evac.integration.station_dynamics import pedestrian_geometry
    for area,geom in geometry.items():
        geom=pedestrian_geometry(geom);geometry[area]=geom
        parts=[p for p in getattr(geom,'geoms',[geom]) if p.geom_type=='Polygon' and not p.is_empty]
        if len(parts)>1:
            main=max(parts,key=lambda p:p.area)
            excluded[area]=sum(p.area for p in parts if p is not main)
            geometry[area]=main
    return {'nodes':nodes,'connectors':{s['id']:s for s in connectors},
            'area_specs':specs,'geometry':geometry,'excluded_island_areas_m2':excluded}


def connector_landing(connector,nodes,lane,lateral=None):
    spec=connector['native_lane_specs'][lane];path=connector['native_paths'][spec['branch']]
    # Defaults are fixed during network compilation. Return fresh node metadata.
    xyz=spec['destination'] if lateral is None else path_position(path,path_length(path),lateral)
    return {**nodes[connector['to_node']],'x':xyz[0],'y':xyz[1]}


class NativePassage:
    """A real JuPedSim simulation per branch; no timed/FIFO position updates."""
    def __init__(self,connector,simulation_factory,parameter_factory,radius=.22):
        self.connector=connector;self.parameters=parameter_factory;self.radius=radius
        self.branches=[];self.occupants={};self.incapacitated=set()
        for path in connector['native_paths']:
            prepared=PreparedPath(path)
            length=prepared.length;width=connector['native_branch_width_m']
            bent=BentPath(path) if has_horizontal_turn(path) else None
            sim=simulation_factory(bent.geometry(width) if bent else box(-.6,-width/2,length+.6,width/2))
            self.branches.append({'sim':sim,'path':path,'prepared_path':prepared,'length':length,'stages':{},'bent_path':bent})
        # Register all exit targets without creating passengers or advancing time.
        import jupedsim as jps
        for lane,spec in enumerate(connector['native_lane_specs']):
            branch=self.branches[spec['branch']]
            final_xy=spec['destination'][:2] if branch.get('bent_path') else (branch['length'],spec['offset'])
            final=branch['sim'].add_waypoint_stage(final_xy,.07)
            hold_xy=branch['prepared_path'].position(max(.25,branch['length']-1.),spec['offset'])[:2] if branch.get('bent_path') else (max(.25,branch['length']-1.),spec['offset'])
            hold=branch['sim'].add_waypoint_stage(hold_xy,.07)
            turns=[]
            if branch.get('bent_path'):
                distance=0
                for i in range(1,len(branch['path'])-1):
                    distance+=branch['prepared_path'].lengths[i-1]
                    a,b,c=branch['path'][i-1:i+2]
                    before=(b[0]-a[0],b[1]-a[1]);after=(c[0]-b[0],c[1]-b[1])
                    if sum(x*y for x,y in zip(before,after))<math.hypot(*before)*math.hypot(*after)*(1-1e-8):
                        xy=branch['prepared_path'].position(distance,spec['offset'])[:2]
                        turns.append(branch['sim'].add_waypoint_stage(xy,.15))
            def journey_to(target):
                stages=[*turns,target];journey=jps.JourneyDescription(stages)
                for a,b in zip(stages,stages[1:]):journey.set_transition_for_stage(a,jps.Transition.create_fixed_transition(b))
                return branch['sim'].add_journey(journey)
            final_journey=journey_to(final);hold_journey=journey_to(hold)
            branch['stages'][lane]=(hold_journey,turns[0] if turns else hold)
            branch.setdefault('turn_stages',{})[lane]=set(turns)
            branch.setdefault('final_stages',{})[lane]=(final_journey,final)

    def can_enter(self,lane):
        if len(self.occupants)>=self.connector['capacity']:return False
        spec=self.connector['native_lane_specs'][lane];branch=self.branches[spec['branch']]
        entry=spec['entry'][:2] if branch.get('bent_path') else (0,spec['offset'])
        return all(math.dist(entry,a.position)>=2*self.radius+.08 for a in branch['sim'].agents())

    def enter(self,agent_id,lane,speed):
        if agent_id in self.occupants or not self.can_enter(lane):return False
        spec=self.connector['native_lane_specs'][lane];branch=self.branches[spec['branch']]
        journey,stage=branch['stages'][lane]
        entry=spec['entry'][:2] if branch.get('bent_path') else (0,spec['offset'])
        jid=branch['sim'].add_agent(self.parameters(entry,journey,stage,speed))
        self.occupants[agent_id]=(lane,jid)
        return True

    def release_landing(self,agent_id):
        """Allow the final metre only after the receiving floor is reserved."""
        if agent_id in self.incapacitated:return
        lane,jid=self.occupants[agent_id];spec=self.connector['native_lane_specs'][lane]
        branch=self.branches[spec['branch']];journey,stage=branch['final_stages'][lane]
        current=branch['sim'].agent(jid).stage_id
        if current in branch.get('turn_stages',{}).get(lane,set()):stage=current
        branch['sim'].switch_agent_journey(jid,journey,stage)

    def sample(self,agent_id):
        lane,jid=self.occupants[agent_id];spec=self.connector['native_lane_specs'][lane]
        branch=self.branches[spec['branch']]
        if branch.get('bent_path'):
            xy=branch['sim'].agent(jid).position
            progress,xyz,_=branch['bent_path'].sample(xy)
            center=branch['prepared_path'].position(progress*branch['length'],0)
            normal=branch['prepared_path'].position(progress*branch['length'],1)
            axis=[normal[i]-center[i] for i in (0,1)]
            lateral=sum((xy[i]-center[i])*axis[i] for i in (0,1))/sum(v*v for v in axis)
            return progress,lateral,xyz
        s,lateral=branch['sim'].agent(jid).position
        progress=max(0,min(1,s/branch['length']))
        xyz=branch['prepared_path'].position(progress*branch['length'],lateral)
        return progress,lateral,xyz

    def engine_position(self,agent_id):
        lane,jid=self.occupants[agent_id]
        branch=self.branches[self.connector['native_lane_specs'][lane]['branch']]
        return list(branch['sim'].agent(jid).position)

    def ready(self,agent_id):
        if agent_id in self.incapacitated:return False
        lane,jid=self.occupants[agent_id];spec=self.connector['native_lane_specs'][lane]
        branch=self.branches[spec['branch']];length=branch['length']
        if branch.get('bent_path'):
            return math.dist(branch['sim'].agent(jid).position,spec['destination'][:2])<=.09
        s,lateral=branch['sim'].agent(jid).position
        progress=max(0,min(1,s/length))
        return math.hypot((1-progress)*length,lateral-spec['offset'])<=.09

    def release(self,agent_id):
        self.incapacitated.discard(agent_id)
        lane,jid=self.occupants.pop(agent_id)
        branch=self.branches[self.connector['native_lane_specs'][lane]['branch']]
        branch['sim'].mark_agent_for_removal(jid)

    def incapacitate(self,agent_id):
        self.incapacitated.add(agent_id)
        lane,jid=self.occupants[agent_id]
        branch=self.branches[self.connector['native_lane_specs'][lane]['branch']]
        branch['sim'].agent(jid).model.desired_speed=0.

    def advance(self,speed_factor=1):
        stopped={(self.connector['native_lane_specs'][self.occupants[aid][0]]['branch'],self.occupants[aid][1])
                 for aid in self.incapacitated}
        for index,branch in enumerate(self.branches):
            for actor in branch['sim'].agents():
                if (index,actor.id) in stopped:
                    actor.model.desired_speed=0.
                    continue
                slope=branch['bent_path'].sample(actor.position)[2] if branch.get('bent_path') else 1
                actor.model.desired_speed=self.connector['speed_mps']*speed_factor*slope
            branch['sim'].iterate()
