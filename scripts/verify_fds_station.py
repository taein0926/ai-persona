"""Verify preserved FDS artifacts independently of evacuation success."""
import json,re
from pathlib import Path
import numpy as np
from shapely.geometry import Point
from persona_evac.integration.fds_environment import FireField,sha
from persona_evac.integration.station_geometry import floor_geometry
from persona_evac.llm.ollama import atomic_json

c=json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf-8'));field=FireField(c)
assert field.times[-1]>=c['fire']['time_end_s']-.1
assert np.isfinite(field.data['volume_k']).all() and np.all(np.diff(field.data['volume_z'])>0)
assert len(field.data['volume_z'])==len(c['fire']['volume_slice_heights_m'])
log=Path('output/fds/gaehwasan/gaehwasan.out');data=log.read_bytes()
assert sha(log)==field.meta['out_sha256']
assert b'\x00' not in data and data.count(b'FDS completed successfully')==1
onsets={};maxima={}
for f in c['floors']:
    geom=floor_geometry(f)
    mask=np.array([[geom.covers(Point(x,y)) for y in field.data['y']] for x in field.data['x']])
    curve=field.data[f['id']+'_k'][:,mask].max(axis=1)
    maxima[f['id']]=float(curve.max());assert maxima[f['id']]>.01
    onsets[f['id']]=float(field.times[np.flatnonzero(curve>.01)[0]])
elapsed=re.search(rb'Total Elapsed Wall Clock Time \(s\):\s*([\d.]+)',data)
report={'status':'통과','completed_time_s':float(field.times[-1]),'fields_sha256':field.meta['fields_sha256'],
    'deck_sha256':field.meta['deck_sha256'],'geometry_sha256':field.meta['geometry_sha256'],
    'out_sha256':field.meta['out_sha256'],'original_fds_wall_s':float(elapsed.group(1)) if elapsed else None,
    'floor_max_extinction_per_m':maxima,'first_floor_smoke_above_0_01_per_m_s':onsets,
    'volume_heights_m':field.data['volume_z'].tolist(),'source_files_recomputed':False}
atomic_json(Path('output/gaehwasan/private_verification/fds_preservation.json'),report)
print(json.dumps(report,ensure_ascii=False,indent=2))
