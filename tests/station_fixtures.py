import json
from pathlib import Path

def source(count=10):
    return {'experiment_id': 'test', 'backend': 'fixture', 'assignment_manifest_hash': 'test', 'provenance': {}, 'variants': {'C': [{'agent_slot': i, 'agent_id': 100 + i, 'persona': {'uuid': str(i)}, 'desired_speed_mps': 1.2, 'pre_movement_s': 0.5} for i in range(count)]}}

def config():
    return json.loads(Path('configs/scenarios/gaehwasan.json').read_text(encoding='utf-8'))

