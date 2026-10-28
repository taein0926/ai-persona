"""Researcher-only diagnostics, never imported by agent decision code."""
import json,sys
from pathlib import Path
from shapely.geometry import Point
from persona_evac.integration.station_geometry import floor_geometry
from persona_evac.integration.station_dynamics import pedestrian_geometry
r=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
c=r['scenario'];floors={f['id']:f for f in c['floors']};areas={a['id']:a for a in c['areas']}
print(json.dumps({'summary':r['summary'],'performance':r['performance']},ensure_ascii=False))
for a in r['agents']:
    if a['evacuation_time_s'] is not None:continue
    s=a['segments'][-1];area=a['personal_knowledge']['area'];spec=areas[area]
    geom=pedestrian_geometry(floor_geometry(floors[spec['floor']],spec.get('region')))
    p=Point(s['to']['x'],s['to']['y'])
    print(json.dumps({'id':a['agent_id'],'last':s['to'],'cause':a['noncompletion_cause'],
        'boundary_distance_m':geom.boundary.distance(p),'known':a['personal_knowledge'],
        'map':a['known_map_summary'],'last_navigation':[n for n in r['navigation_updates'] if n['agent_id']==a['agent_id']][-3:]},ensure_ascii=False))
