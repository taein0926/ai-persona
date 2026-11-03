"""Precomputed FDS -> pedestrian and replay coupling. No synthetic smoke fallback."""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
ROOT = PACKAGE_ROOT if (PACKAGE_ROOT / "configs/scenarios/station_view.json").exists() else Path.cwd()
GAS_SPECIES = {'co':'CARBON MONOXIDE','co2':'CARBON DIOXIDE','o2':'OXYGEN'}


def cache_obstruction_masks(sim):
    """높이·가스마다 반복하는 동일 메시·시각의 장애물 검사를 재사용한다."""
    for mesh in sim.meshes:
        original=mesh.get_obstruction_mask
        cache={}
        def cached(times,cell_centered=False,_original=original,_cache=cache):
            values=np.asarray(times)
            key=(values.dtype.str,tuple(values),cell_centered)
            if key not in _cache:
                result=_original(times,cell_centered=cell_centered)
                result.flags.writeable=False
                _cache[key]=result
            return _cache[key]
        mesh.get_obstruction_mask=cached


def _gas_slice(sim, species, height, settings, times, xs, ys):
    quantity = species+' VOLUME FRACTION'
    matches = [s for s in sim.slices if s.quantity.name.upper() == quantity
               and s.orientation == 3 and abs(s.extent.z_start-height) <= settings.get('mesh_cell_m',.5)/2+.01]
    matches.sort(key=lambda s:(abs(s.extent.z_start-height),-s.extent.z_start))
    if not matches:
        raise ValueError(f'FDS 가스 단면이 없습니다: {species}, {height}m. 가스 출력을 추가해 재계산하세요.')
    field=matches[0];values,coords=field.to_global(return_coordinates=True)
    values=np.asarray(values,dtype=np.float32)
    if (values.shape != (len(times),len(xs),len(ys)) or not np.allclose(field.times,times)
            or not np.allclose(coords['x'],xs) or not np.allclose(coords['y'],ys)):
        raise ValueError('FDS 가스 단면의 시간·좌표가 연기장과 일치하지 않습니다.')
    if not np.isfinite(values).all() or np.any(values < -1e-6) or np.any(values > 1+1e-6):
        raise ValueError('FDS 가스 체적분율은 유한한 0~1 값이어야 합니다.')
    masked=field.to_global(masked=True,fill=np.nan)
    valid=np.isfinite(masked[0])
    if not np.array_equal(np.isfinite(masked),np.broadcast_to(valid,masked.shape)):
        raise ValueError('시간에 따라 바뀌는 FDS 장애물은 가스 표본 추출에서 지원하지 않습니다.')
    return np.clip(values,0,1),float(field.extent.z_start),valid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_hash(config):
    # Pedestrian flow parameters may vary; the fire geometry stays fixed.
    geometry = {k: config[k] for k in ("floors", "nodes")}
    if config.get('train'):
        geometry.update(train=config['train'],fire=config['fire'],
                        stair_paths=[{key:s.get(key) for key in ('id','kind','path_xyz','branch_paths_xyz','flights',
                            'landings','floor_opening_polygon','exit_boundary_xy','width_m')} for s in config['stairs']])
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def export_fds(fds_dir: Path, output_dir: Path, config_path: Path, deck_path: Path):
    """Run in the pyfds image, which contains fdsreader. Require a complete FDS run."""
    from fdsreader import Simulation
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = deepcopy(config['fire'])
    requested_heights = dict(settings["slice_height_m"])
    chid = deck_path.stem
    log = (fds_dir / f"{chid}.out").read_text(errors="replace")
    if "FDS completed successfully" not in log:
        raise ValueError("FDS 계산이 정상 완료되지 않았습니다.")
    if sha(deck_path) != sha(fds_dir / f"{chid}.fds"):
        raise ValueError("Completed FDS deck does not match the current input")
    # A setup-check subdirectory may contain another SMV. Select the completed
    # run explicitly instead of letting the reader recursively choose one.
    sim = Simulation(str(fds_dir / f"{chid}.smv"))
    cache_obstruction_masks(sim)
    arrays, times, xs, ys = {}, None, None, None
    for floor, height in settings["slice_height_m"].items():
        for key, quantity in (("k", "EXTINCTION COEFFICIENT"), ("temperature", "TEMPERATURE")):
            matches = [s for s in sim.slices if quantity in s.quantity.name.upper()
                       and abs(s.extent.z_start-height) <= settings.get("mesh_cell_m",.5)/2+.01 and s.orientation == 3]
            matches=sorted(matches,key=lambda s:(abs(s.extent.z_start-height),-s.extent.z_start))
            if not matches:
                raise ValueError(f"Expected one {quantity} horizontal slice at {height}m, got {len(matches)}")
            field = matches[0]
            settings["slice_height_m"][floor] = float(field.extent.z_start)
            values, coords = field.to_global(return_coordinates=True)
            values = np.asarray(values, dtype=np.float32)
            current_times = np.asarray(field.times)
            if values.shape != (len(current_times), len(coords["x"]), len(coords["y"])):
                raise ValueError("Unexpected FDS slice axis order")
            if not np.isfinite(values).all():
                raise ValueError("Incomplete/nonfinite FDS slice")
            if times is None:
                times, xs, ys = current_times, coords["x"], coords["y"]
            elif not (np.allclose(times, current_times) and np.allclose(xs, coords["x"])
                      and np.allclose(ys, coords["y"])):
                raise ValueError("FDS fields are not on matching time and spatial grids")
            arrays[f"{floor}_{key}"] = np.maximum(values, 0) if key == "k" else values
    if settings.get('volume_slice_heights_m'):
        layers=[];zs=[]
        for height in settings['volume_slice_heights_m']:
            matches=[s for s in sim.slices if 'EXTINCTION COEFFICIENT' in s.quantity.name.upper()
                     and s.orientation==3 and abs(s.extent.z_start-height)<=settings['mesh_cell_m']/2+.01]
            matches=sorted(matches,key=lambda s:(abs(s.extent.z_start-height),-s.extent.z_start))
            if not matches:raise ValueError('입체 연기 단면이 누락되었습니다.')
            field=matches[0];values,coords=field.to_global(return_coordinates=True)
            if not np.allclose(field.times,times):raise ValueError('입체 연기 시각이 맞지 않습니다.')
            layers.append(np.maximum(np.asarray(values,dtype=np.float32)[:,::3,::2],0));zs.append(float(field.extent.z_start))
        arrays.update(volume_k=np.stack(layers,axis=-1),volume_x=xs[::3],volume_y=ys[::2],volume_z=np.array(zs))
    # Keep gases at the original FDS x/y resolution, unlike display smoke.
    gas_heights=sorted(set(settings.get('volume_slice_heights_m',[])) | set(requested_heights.values()))
    gas_z=None;gas_valid=None
    for key,species in GAS_SPECIES.items():
        layers=[];zs=[];masks=[]
        for height in gas_heights:
            values,z,valid=_gas_slice(sim,species,height,settings,times,xs,ys)
            layers.append(values);zs.append(z);masks.append(valid)
        if gas_z is not None and zs != gas_z:
            raise ValueError('FDS 가스별 표본 높이가 일치하지 않습니다.')
        gas_z=zs
        masks=np.stack(masks,axis=-1)
        if gas_valid is not None and not np.array_equal(gas_valid,masks):
            raise ValueError('FDS 가스별 장애물 마스크가 일치하지 않습니다.')
        gas_valid=masks
        arrays['gas_'+key]=np.stack(layers,axis=-1)
        for floor,height in settings['slice_height_m'].items():
            arrays[f'{floor}_{key}']=arrays['gas_'+key][...,zs.index(height)]
    arrays['gas_z']=np.array(gas_z)
    arrays['gas_valid']=gas_valid
    if times[-1] < settings.get("time_end_s", 240)-.1 or max(arrays[f+"_k"].max() for f in settings["slice_height_m"]) <= 0:
        raise ValueError("Missing full time coverage or smoke production")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"engine": "FDS 6.11.1", "deck_sha256": sha(deck_path),
                "geometry_sha256": geometry_hash(config), "fire": settings,
                "requested_slice_height_m": requested_heights,
                "mesh_cell_m": settings.get("mesh_cell_m", .5), "time_start_s": float(times[0]), "time_end_s": float(times[-1]),
                "quantities": {"k": "extinction coefficient [1/m]", "temperature": "temperature [C]"},
                "sampling": "nearest FDS spatial node, linear time interpolation",
                "raw_files_sha256": {p.name: sha(p) for p in sorted(fds_dir.glob(f"{chid}*.sf"))},
                "out_sha256": sha(fds_dir / f"{chid}.out")}
    metadata['quantities'].update({key:species+' volume fraction [mol/mol]' for key,species in GAS_SPECIES.items()})
    metadata['gas_sampling']={'xy':'nearest unobstructed FDS node within 1.5 mesh cells',
                              'z':'linear between air levels; air-side sample at solid boundaries',
                              'breathing_height_m':1.5,'species':GAS_SPECIES}
    if (fds_dir/'geometry_audit.json').exists():
        metadata['geometry_audit']=json.loads((fds_dir/'geometry_audit.json').read_text(encoding='utf8'))
    np.savez_compressed(output_dir / "fields.npz", times=times, x=xs, y=ys, **arrays)
    metadata["fields_sha256"] = sha(output_dir / "fields.npz")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(times), "time_end_s": float(times[-1]),
                      "max_k": {f: float(arrays[f+"_k"].max()) for f in settings["slice_height_m"]}}))


@lru_cache(maxsize=256)
def _stair_path_metrics(path):
    """경로 내용이 같은 동안만 원래 NumPy 연산 결과를 재사용한다."""
    points=np.asarray(path,dtype=float)
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    total=sum(lengths)
    points.flags.writeable=False;lengths.flags.writeable=False
    return points,lengths,total


class FireField:
    def __init__(self, config, directory=None, deck_path=None):
        slug = config.get("station_slug", "gaehwasan")
        directory = Path(directory or Path(os.environ.get("PERSONA_EVAC_OUTPUT", ROOT / "output")) / f"fds/{slug}/derived")
        deck_path = Path(deck_path or ROOT / f"configs/fds/{slug}.fds")
        if not (directory / "metadata.json").exists():
            raise ValueError(f"{config['name']} FDS 계산·변환 파일이 없습니다. 해당 역의 FDS 계산과 변환이 필요합니다.")
        self.meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if self.meta["deck_sha256"] != sha(deck_path) or self.meta["geometry_sha256"] != geometry_hash(config):
            raise ValueError("FDS 결과와 현재 공간/화재 설정이 다릅니다. FDS를 다시 계산하세요.")
        if self.meta["fields_sha256"] != sha(directory / "fields.npz"):
            raise ValueError("FDS 변환 파일의 해시가 일치하지 않습니다.")
        with np.load(directory / "fields.npz", allow_pickle=False) as data:
            self.data = {key: data[key] for key in data.files}
        self.times = self.data["times"]
        self.offset = self.meta["fire"]["evacuation_start_s"]
        # Loaded FDS fields are fixed for this replay. Cache exact node lookups,
        # never rounded pedestrian positions or a previous frame's smoke.
        self._sample_frame = None
        self._sample_grid = None

    def indices(self, evacuation_time):
        t = evacuation_time+self.offset
        if t < self.times[0] or t > self.times[-1]+1e-6:
            raise ValueError(f"대피 시각 {evacuation_time:.1f}초가 FDS 계산 범위를 넘었습니다. FDS 계산 시간을 늘려야 합니다.")
        high = min(int(np.searchsorted(self.times, t, side="right")), len(self.times)-1)
        low = max(0, high-1)
        weight = 0 if high == low else (t-self.times[low])/(self.times[high]-self.times[low])
        return low, high, weight

    def require_gases(self):
        """가스가 없는 과거 파일을 독성 영향이 없는 파일로 오인하지 않는다."""
        required=['gas_z','gas_valid',*(f'gas_{key}' for key in GAS_SPECIES),
                  *(f'{floor}_{key}' for floor in self.meta['fire']['slice_height_m'] for key in GAS_SPECIES)]
        if any(key not in self.data for key in required):
            raise ValueError('FED 계산에 필요한 CO·CO₂·산소 농도장이 없습니다. 가스 출력을 포함해 FDS를 재계산·변환하세요.')
        expected=(len(self.times),len(self.data['x']),len(self.data['y']),len(self.data['gas_z']))
        if (not len(self.data['gas_z']) or not np.isfinite(self.data['gas_z']).all() or np.any(np.diff(self.data['gas_z'])<=0)
                or self.data['gas_valid'].shape!=expected[1:]
                or any(self.data['gas_'+key].shape!=expected for key in GAS_SPECIES)):
            raise ValueError('FDS 가스 농도장의 격자·높이가 올바르지 않습니다.')
        for key in GAS_SPECIES:
            values=self.data['gas_'+key]
            if not np.isfinite(values).all() or np.any(values<0) or np.any(values>1):
                raise ValueError('FDS 가스 농도장은 유한한 0~1 체적분율이어야 합니다.')
        self._gas_air_nodes={}

    def _gas_node(self,x,y,iz):
        xs,ys=self.data['x'],self.data['y']
        ix,iy=int(np.argmin(abs(xs-x))),int(np.argmin(abs(ys-y)))
        valid=self.data['gas_valid'][...,iz]
        if valid[ix,iy]:return ix,iy
        # Slice nodes inside solids often contain zero O2. Never interpret that
        # as an asphyxiating atmosphere; use a nearby actual air sample or fail.
        if not hasattr(self,'_gas_air_nodes'):self._gas_air_nodes={}
        if iz not in self._gas_air_nodes:
            nodes=np.argwhere(valid)
            self._gas_air_nodes[iz]=(nodes,xs[nodes[:,0]],ys[nodes[:,1]])
        nodes,nx,ny=self._gas_air_nodes[iz]
        if not len(nodes):raise ValueError('해당 높이에 유효한 FDS 공기 표본이 없습니다.')
        distances=(nx-x)**2+(ny-y)**2;i=int(np.argmin(distances))
        if distances[i]>(1.5*self.meta.get('mesh_cell_m',self.meta['fire'].get('mesh_cell_m',1.)))**2:
            raise ValueError('승객 근처에 유효한 FDS 공기 표본이 없습니다. 가스 격자와 형상을 확인하세요.')
        return tuple(nodes[i])

    def sample_gases_xyz(self,xyz,evacuation_time):
        """실제 바닥 XYZ + 호흡 높이 1.5m. 계단에서도 층 단면으로 대체하지 않는다."""
        lo,hi,w=self.indices(evacuation_time)
        xs,ys,zs=self.data['x'],self.data['y'],self.data['gas_z']
        x,y,z=xyz;z+=1.5
        if not (xs[0]<=x<=xs[-1] and ys[0]<=y<=ys[-1] and zs[0]<=z<=zs[-1]):
            raise ValueError('승객의 호흡 위치가 FDS 가스 농도장 범위를 벗어났습니다.')
        high=min(int(np.searchsorted(zs,z,side='right')),len(zs)-1);low=max(0,high-1)
        weight=0 if high==low else (z-zs[low])/(zs[high]-zs[low])
        # Do not query a masked adjacent height when its interpolation weight is zero.
        levels=[(low,1-weight)] if weight==0 else [(high,1.)] if weight==1 else [(low,1-weight),(high,weight)]
        ix,iy=int(np.argmin(abs(xs-x))),int(np.argmin(abs(ys-y)))
        air_levels=[(iz,zw) for iz,zw in levels if self.data['gas_valid'][ix,iy,iz]]
        if air_levels:
            # A ceiling/slab can occupy one endpoint even when the breathing
            # position is in air. Never blend across that solid boundary.
            total=sum(zw for _,zw in air_levels)
            levels=[(iz,zw/total) for iz,zw in air_levels]
        nodes=[(iz,zw,*self._gas_node(x,y,iz)) for iz,zw in levels]
        result={}
        for key,label,scale in [('co','co_ppm',1e6),('co2','co2_percent',100),('o2','o2_percent',100)]:
            field=self.data['gas_'+key]
            value=sum(zw*((1-w)*float(field[lo,ix,iy,iz])+w*float(field[hi,ix,iy,iz])) for iz,zw,ix,iy in nodes)
            result[label]=value*scale
        return result

    def sample(self, floor, pos, evacuation_time):
        frame = (type(evacuation_time), evacuation_time, type(self.offset), self.offset)
        if self._sample_frame != frame or self._sample_times is not self.times:
            self._sample_indices = self.indices(evacuation_time)
            self._sample_frame = frame
            self._sample_times = self.times
            self._sample_floors = {}
        low, high, w = self._sample_indices
        xs, ys = self.data["x"], self.data["y"]
        if self._sample_grid is None or self._sample_grid[0] is not xs or self._sample_grid[1] is not ys:
            self._sample_grid = (xs, ys)
            # Keep NumPy's original argmin, including dtype and midpoint ties.
            self._nearest_x = lru_cache(maxsize=8192, typed=True)(lambda x: int(np.argmin(abs(xs-x))))
            self._nearest_y = lru_cache(maxsize=8192, typed=True)(lambda y: int(np.argmin(abs(ys-y))))
            self._sample_floors = {}
        x, y = pos
        if not (xs[0] <= x <= xs[-1] and ys[0] <= y <= ys[-1]):
            raise ValueError("Position lies outside FDS field")
        ix, iy = self._nearest_x(x), self._nearest_y(y)
        k_field, temperature_field = self.data[f"{floor}_k"], self.data[f"{floor}_temperature"]
        cached = self._sample_floors.get(floor)
        if cached is None or cached[0] is not k_field or cached[1] is not temperature_field:
            cached = self._sample_floors[floor] = (k_field, temperature_field, {})
        cells = cached[2]
        node = (ix, iy)
        if node in cells:
            return cells[node].copy()
        values = {key: float((1-w)*self.data[f"{floor}_{key}"][low, ix, iy]
                            + w*self.data[f"{floor}_{key}"][high, ix, iy]) for key in ("k", "temperature")}
        k = values["k"]
        reading = {"source": "fds", "fire_time_s": round(evacuation_time+self.offset, 3),
                "extinction_per_m": round(k, 5), "temperature_c": round(values["temperature"], 2),
                "visibility_m_capped_100": round(min(100, 3/max(k, 1e-9)), 2),
                "speed_factor": max(0.1, min(1.0, 1-0.057*k/0.706))}
        if all(f'{floor}_{key}' in self.data for key in GAS_SPECIES):
            for key,label,scale in [('co','co_ppm',1e6),('co2','co2_percent',100),('o2','o2_percent',100)]:
                field=self.data[f'{floor}_{key}']
                reading[label]=float((1-w)*field[low,ix,iy]+w*field[high,ix,iy])*scale
        cells[node] = reading
        return reading.copy()

    def route_observation(self, floor, path, time):
        samples = []
        for a, b in zip(path, path[1:]):
            for f in np.linspace(0, 1, max(2, int(np.linalg.norm(np.array(b)-a))+1)):
                samples.append(self.sample(floor, (a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1])), time))
        return {"max_extinction_per_m": max((s["extinction_per_m"] for s in samples), default=0),
                "max_temperature_c": max((s["temperature_c"] for s in samples), default=20),
                "min_visibility_m_capped_100": min((s["visibility_m_capped_100"] for s in samples), default=100)}

    def sample_stair(self,config,stair,segment,progress,time):
        """Smoke at physical stair height; temperature from connected floor slices.

        The existing FDS run has no temperature slices at every stair height.
        Keep this interpolation explicit rather than reporting it as a 3D observation.
        """
        from persona_evac.integration.station_layout import stair_path
        if any('elevation_m' not in f for f in config['floors']):
            config={**config,'floors':[{**f,'elevation_m':self.meta['fire']['slice_height_m'][f['id']]-1.5} for f in config['floors']]}
        path=tuple(tuple(p) for p in stair_path(config,stair))
        points,lengths,total=_stair_path_metrics(path);distance=np.clip(progress,0,1)*total
        xyz=points[-1].copy()
        for i,length in enumerate(lengths):
            if distance<=length or i==len(lengths)-1:
                xyz=points[i]+(points[i+1]-points[i])*min(1,distance/max(length,1e-9));break
            distance-=length
        # Same lane offsets as the display, without the illustrative 2.5 height spread.
        for key,endpoint,weight in [('from',points[0],1-progress),('to',points[-1],progress)]:
            xyz[:2]+=weight*(np.array([segment[key]['x'],segment[key]['y']])-endpoint[:2])
        a,b=segment['from'],segment['to'];target=b['floor'] if b['floor']!='OUTSIDE' else a['floor']
        lower=self.sample(a['floor'],(a['x'],a['y']),time)
        upper=self.sample(target,(b['x'],b['y']),time)
        k=(1-progress)*lower['extinction_per_m']+progress*upper['extinction_per_m']
        mode='connected_floor_interpolation'
        if 'volume_k' in self.data:
            lo,hi,w=self.indices(time);ix=int(np.argmin(abs(self.data['volume_x']-xyz[0])));iy=int(np.argmin(abs(self.data['volume_y']-xyz[1])))
            column=(1-w)*self.data['volume_k'][lo,ix,iy]+w*self.data['volume_k'][hi,ix,iy]
            k=float(np.interp(xyz[2]+1.5,self.data['volume_z'],column));mode='height_slices'
        return {'source':'fds','fire_time_s':round(time+self.offset,3),'extinction_per_m':round(k,5),
                'temperature_c':round((1-progress)*lower['temperature_c']+progress*upper['temperature_c'],2),
                'visibility_m_capped_100':round(min(100,3/max(k,1e-9)),2),
                'speed_factor':max(.1,min(1,1-.057*k/.706)),'speed_factor_applied':False,
                'smoke_sampling':mode,'temperature_sampling':'connected_floor_interpolation',
                'physical_position_m':xyz.tolist(),'breathing_height_m':1.5}

    def stair_observations(self,config,agents):
        stairs={s['id']:s for s in config['stairs']};observations=[]
        for a in agents:
            for segment in a['segments']:
                if segment['status']!='on_stairs':continue
                motion=np.asarray(segment['stair_motion'])
                times=sorted(set([segment['start']]+list(np.arange(np.ceil(segment['start']),segment['end'],1))))
                for time in times:
                    progress=float(np.interp(time,motion[:,0],motion[:,1]))
                    observations.append({'agent_id':a['agent_id'],'time_s':float(time),'floor':'층간 연결',
                        'stair_id':segment['stair_id'],**self.sample_stair(config,stairs[segment['stair_id']],segment,progress,time)})
        return observations

    def replay(self, config, until):
        """2m cells clipped to walkable polygons; values come from the FDS nodes."""
        from shapely.geometry import box
        from persona_evac.integration.station_geometry import floor_geometry
        cells = []
        for floor in config["floors"]:
            geom = floor_geometry(floor)
            xmin,ymin,xmax,ymax=geom.bounds
            for x in np.arange(np.floor(xmin),xmax,2):
                for y in np.arange(np.floor(ymin),ymax,1):
                    if geom.covers(box(x,y,x+2,y+1)):
                        cells.append({"floor": floor["id"], "x": float(x), "y": float(y), "w": 2, "h": 1})
        times = sorted(set([0.0, float(until)]+[float(t-self.offset) for t in self.times if 0 < t-self.offset < until]))
        frames = []
        for t in times:
            readings = [self.sample(c["floor"], (c["x"]+1,c["y"]+0.5), t) for c in cells]
            frames.append({"time_s": t, "k": [s["extinction_per_m"] for s in readings],
                           "temperature_c": [s["temperature_c"] for s in readings]})
        result={"enabled": True, "metadata": self.meta, "cells": cells, "frames": frames,
                "notice": f"FDS 수평 단면 높이(z,m): {self.meta['fire']['slice_height_m']} · 대피 시작은 발화 {self.offset:g}초 후. {self.meta.get('mesh_cell_m', .5):g}m 계산 격자를 2×1m 표시 셀로 표본화."}
        if 'volume_k' in self.data:
            raw=self.data['volume_k'];indices=np.argwhere(raw.max(axis=0)>.003)
            voxels=[{'x':float(self.data['volume_x'][ix]),'y':float(self.data['volume_y'][iy]),
                     'z':float(self.data['volume_z'][iz])} for ix,iy,iz in indices]
            volume_frames=[]
            selected=tuple(indices.T)
            for t in sorted(set([float(until)]+[float(t) for t in np.arange(0,float(until),4)])):
                lo,hi,w=self.indices(t)
                values=(1-w)*raw[lo][selected]+w*raw[hi][selected]
                volume_frames.append({'time_s':t,'k':[round(float(value),4) for value in values]})
            result['volume']={'voxels':voxels,'frames':volume_frames,'cell_m':[3,2,1],
                              'source':'FDS extinction coefficient; actual horizontal slices stacked in z',
                              'notice':'FDS 높이별 소광계수를 3×2×1m 표시 셀로 표본화한 입체 연기. 계단을 통한 상승도 계산값을 사용합니다.'}
        return result
