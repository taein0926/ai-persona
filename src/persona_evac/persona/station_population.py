"""고유 Nemotron 인물과 모델별 사전 분류·실측 기반 초기값을 준비한다."""
from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from persona_evac.empirical import EmpiricalResolver, load_empirical_records
from persona_evac.empirical.resolver import _quantile
from persona_evac.hashing import sha256_file, sha256_value
from persona_evac.llm import OllamaProvider
from persona_evac.llm.baseline_profile import POLICY as BASELINE_POLICY
from persona_evac.persona.sampling import load_personas
from persona_evac.schemas import PersonaProfile, SocialMode

CONTEXT = {"scenario": "지하철역 화재 대피", "multiple_exits": True, "incident": "fire",
           "station": "개화산역", "line": "5호선", "district": "서울 강서구",
           "mapping_policy": BASELINE_POLICY, "role": "visitor", "social_mode": "individual",
           "familiarity": "discovery", "input_policy": "all_persona_fields_except_uuid"}


def _mapping_model(provider):
    return {'model':provider.model,
            'model_digest':getattr(getattr(provider,'local',None),'digest',None)
                           or getattr(provider,'settings',{}).get('model_digest')}


def refresh_population(source, count, provider=None, empirical_path=None):
    """Refresh old baselines in memory on an explicit calculation, retaining raw input."""
    selected = sorted(source['variants']['C'], key=lambda a: a['agent_slot'])[:count]
    if len(selected) != count:
        raise ValueError('요청한 인원만큼 초기 페르소나가 준비되지 않았습니다.')
    provider = provider or OllamaProvider(cache_dir=Path(os.environ.get('PERSONA_EVAC_OUTPUT', 'output'))/'gaehwasan/mapping_cache/ollama')
    identity = _mapping_model(provider)
    def current(agent):
        mapping = agent.get('provenance',{}).get('mapping',{})
        model = mapping.get('source_ref',source.get('provenance',{}).get('model'))
        digest = mapping.get('model_digest')
        if digest is None and model==source.get('provenance',{}).get('model'):
            digest = source.get('provenance',{}).get('mapping_settings',{}).get('model_digest')
        return mapping.get('policy')==BASELINE_POLICY and {'model':model,'model_digest':digest}==identity
    pending = [a for a in selected if not current(a)]
    if not pending:
        return source
    empirical_path = Path(empirical_path or pending[0]['provenance']['empirical']['source_ref'])
    resolver = EmpiricalResolver(load_empirical_records(empirical_path), source['provenance']['snapshot']['selection_seed'])
    updated = deepcopy(source)
    by_slot = {a['agent_slot']: a for a in updated['variants']['C']}
    for original in pending:
        agent = by_slot[original['agent_slot']]
        profile = PersonaProfile.model_validate(agent['persona'])
        mapping = provider.map(profile, CONTEXT)
        empirical = resolver.resolve(persona_uuid=profile.uuid, variant='C', treatment=SocialMode.INDIVIDUAL,
                                     response=mapping.response_tendency, speed=mapping.speed_tendency)
        fields = profile.model_dump(mode='json', exclude={'uuid'})
        agent.update(mapping=mapping.model_dump(mode='json'), mapping_source=provider.source_name,
                     empirical=empirical.model_dump(mode='json'), pre_movement_s=empirical.response_time_s,
                     desired_speed_mps=empirical.desired_speed_mps)
        agent.pop('classification_pending',None)
        agent['provenance']['mapping'] = {'source_kind':'llm', 'source_ref':provider.model,
            'model_digest':identity['model_digest'],
            'record_id':profile.uuid, 'policy':BASELINE_POLICY, 'input_fields':list(fields),
            'input_hash':sha256_value(fields), 'artifact_hash':sha256_value(agent['mapping'])}
        agent['provenance']['empirical'] = {'source_kind':'empirical', 'source_ref':str(empirical_path),
            'response_record_id':empirical.response_record_id, 'speed_record_id':empirical.speed_record_id}
    updated['provenance']['baseline_refresh'] = {'policy':BASELINE_POLICY, 'count':len(pending),
        **identity, 'empirical_source_hash':sha256_file(empirical_path),
        'mapping_settings':getattr(provider, 'settings', {}), 'mapping_timings':getattr(provider, 'records', [])}
    if all(current(agent) for agent in updated['variants']['C']):
        updated['provenance'].update(model=provider.model, mapping_settings=getattr(provider,'settings',{}),
                                     mapping_context=CONTEXT, mapping_timings=getattr(provider,'records',[]))
        updated['provenance']['classification_status']='ready'
    return updated


def prepare_population(snapshot: Path, empirical_path: Path, output: Path, provider=None) -> Path:
    profiles = load_personas(snapshot)
    metadata = json.loads(snapshot.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    records = [p.model_dump(mode="json") for p in profiles]
    if metadata["count"] != len(profiles) or metadata["records_hash"] != sha256_value(records):
        raise ValueError("Nemotron snapshot metadata does not match its records")
    started=time.perf_counter()
    if provider is None:
        if os.environ.get('LLM_PROVIDER', 'ollama') != 'ollama':
            raise ValueError('페르소나 준비는 로컬 Ollama만 지원합니다. LLM_PROVIDER=ollama를 사용하세요.')
        provider = OllamaProvider(cache_dir=output.parent/'mapping_cache'/'ollama')
    empirical_records=load_empirical_records(empirical_path)
    resolver = EmpiricalResolver(empirical_records, int(metadata["selection_seed"]))
    agents = []
    for slot, profile in enumerate(profiles):
        mapping_input = profile
        input_fields = [k for k in records[slot] if k != 'uuid']
        mapping = provider.map(mapping_input, CONTEXT)
        empirical = resolver.resolve(persona_uuid=profile.uuid, variant="C", treatment=SocialMode.INDIVIDUAL,
                                     response=mapping.response_tendency, speed=mapping.speed_tendency)
        provenance = {
            "persona": {"source_kind": "nemotron", "source_ref": metadata["dataset"],
                        "record_id": profile.uuid, "artifact_hash": sha256_value(records[slot]),
                        "method": metadata.get("sampling", "snapshot without replacement"), "version": metadata["revision"]},
            "mapping": {"source_kind": "llm", "source_ref": provider.model, "record_id": profile.uuid,
                        "model_digest": _mapping_model(provider)['model_digest'],
                        "policy": BASELINE_POLICY, "input_fields": input_fields,
                        "input_hash": sha256_value(mapping_input.model_dump(mode="json", exclude={'uuid'})),
                        "artifact_hash": sha256_value(mapping.model_dump(mode="json"))},
            "empirical": {"source_kind": "empirical", "source_ref": str(empirical_path),
                          "response_record_id": empirical.response_record_id, "speed_record_id": empirical.speed_record_id},
        }
        agents.append({"agent_slot": slot, "agent_id": slot+1, "persona": records[slot],
                       "mapping": mapping.model_dump(mode="json"), "mapping_source": provider.source_name,
                       "scenario_context": {"role": "visitor", "social_mode": "individual", "familiarity": "discovery"},
                       "empirical": empirical.model_dump(mode="json"), "provenance": provenance,
                       "pre_movement_s": empirical.response_time_s, "desired_speed_mps": empirical.desired_speed_mps})
        print(json.dumps({"prepared_personas": slot+1, "total": len(profiles)}, ensure_ascii=False), flush=True)
    payload = {"schema_version": "station-personas-2", "experiment_id": "station-unique-nemotron",
               "backend": "nemotron_qwen_empirical", "variants": {"C": agents},
               "assignment_manifest_hash": sha256_value([p.uuid for p in profiles]),
               "provenance": {"snapshot": metadata, "snapshot_path": str(snapshot),
                              "snapshot_sha256": sha256_file(snapshot), "empirical_source_hash": sha256_file(empirical_path),
                              "provider": provider.source_name, "model": provider.model, "mapping_context": CONTEXT,
                              "speed_fast_threshold_mps":_quantile([r.observed_speed_proxy_mps for r in empirical_records],.67),
                              "speed_fast_threshold_basis":"pooled Jülich speed proxy 67th percentile",
                              "mapping_settings": getattr(provider, "settings", {}),
                              "preparation_wall_s":time.perf_counter()-started,
                              "mapping_timings":getattr(provider,'records',[])}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(f".{uuid4().hex}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp.replace(output)
    return output
