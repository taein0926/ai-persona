"""Audit local classifications against original quotes and empirical record values."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from persona_evac.empirical import load_empirical_records
from persona_evac.llm.mapping_evidence import validate_mapping_evidence
from persona_evac.persona.station_population import MAPPING_FIELDS
from persona_evac.persona.sampling import load_personas
from persona_evac.schemas import PersonaProfile

p=argparse.ArgumentParser();p.add_argument('source',type=Path);args=p.parse_args()
source=json.loads(args.source.read_text(encoding='utf-8'));agents=source['variants']['C']
raw=load_personas(Path('data/raw/nemotron_personas_100.jsonl'))
assert len(agents)==100 and [a['persona']['uuid'] for a in agents]==[a.uuid for a in raw]
records={r.record_id:r for r in load_empirical_records(Path('data/processed/julich_features.csv'))}
assert source['provenance']['provider']=='ollama' and source['provenance']['model']=='qwen3:4b'
for a in agents:
    profile=PersonaProfile.model_validate({k:a['persona'][k] for k in MAPPING_FIELDS if k in a['persona']})
    validate_mapping_evidence(a['mapping'],profile)
    assert a['desired_speed_mps']==records[a['empirical']['speed_record_id']].observed_speed_proxy_mps
    assert a['pre_movement_s']==records[a['empirical']['response_record_id']].response_time_s
timings=source['provenance']['mapping_timings'];live=[m['wall_s'] for m in timings if m['source']=='ollama_live']
report={'count':len(agents),'unique_original_personas':True,'literal_evidence_valid':True,'empirical_values_valid':True,
        'response':dict(Counter(a['mapping']['response_tendency'] for a in agents)),
        'speed':dict(Counter(a['mapping']['speed_tendency'] for a in agents)),
        'response_evidence':dict(Counter(a['mapping']['evidence']['response_tendency']['status'] for a in agents)),
        'speed_evidence':dict(Counter(a['mapping']['evidence']['speed_tendency']['status'] for a in agents)),
        'overtaking_disposition':dict(Counter(a['mapping']['overtaking']['disposition'] for a in agents)),
        'urgent_people':[{'id':a['agent_id'],'speed_mps':a['desired_speed_mps'],'mapping':a['mapping']['overtaking']} for a in agents if a['mapping']['overtaking']['disposition']=='urgent'],
        'fast_threshold_mps':source['provenance']['speed_fast_threshold_mps'],
        'preparation_wall_s':source['provenance']['preparation_wall_s'],'live_mapping_count':len(live),
        'median_live_mapping_wall_s':statistics.median(live) if live else None,'maximum_live_mapping_wall_s':max(live) if live else None}
target=args.source.with_suffix('.audit.json');target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
