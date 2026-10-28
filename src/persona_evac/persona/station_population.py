"""Prepare unique Nemotron people and their individual, cached Qwen mappings."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from uuid import uuid4

from persona_evac.empirical import EmpiricalResolver, load_empirical_records
from persona_evac.empirical.resolver import _quantile
from persona_evac.hashing import sha256_file, sha256_value
from persona_evac.llm import GroqProvider, OllamaProvider
from persona_evac.persona.sampling import load_personas
from persona_evac.schemas import PersonaProfile, SocialMode

MAPPING_FIELDS = ("uuid", "age", "sex", "occupation", "persona", "professional_persona",
                  "sports_persona", "hobbies_and_interests", "skills_and_expertise")
CONTEXT = {"scenario": "지하철역 화재 대피", "multiple_exits": True, "incident": "fire",
           "mapping_policy": "sports_age_sex_urgency_evidence_v2", "role": "visitor", "social_mode": "individual",
           "familiarity": "discovery", "input_fields": list(MAPPING_FIELDS)}


def prepare_population(snapshot: Path, empirical_path: Path, output: Path, provider=None) -> Path:
    profiles = load_personas(snapshot)
    metadata = json.loads(snapshot.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    records = [p.model_dump(mode="json") for p in profiles]
    if metadata["count"] != len(profiles) or metadata["records_hash"] != sha256_value(records):
        raise ValueError("Nemotron snapshot metadata does not match its records")
    started=time.perf_counter()
    provider = provider or (OllamaProvider(cache_dir=output.parent/'mapping_cache'/'ollama')
                           if os.environ.get('LLM_PROVIDER','ollama')=='ollama'
                           else GroqProvider(cache_dir=output.parent/'mapping_cache',max_retries=10))
    empirical_records=load_empirical_records(empirical_path)
    resolver = EmpiricalResolver(empirical_records, int(metadata["selection_seed"]))
    agents = []
    for slot, profile in enumerate(profiles):
        # Preserve the full original record; send explicitly recorded behavior-relevant fields.
        mapping_input = PersonaProfile.model_validate({k: records[slot][k] for k in MAPPING_FIELDS if k in records[slot]})
        mapping = provider.map(mapping_input, CONTEXT)
        empirical = resolver.resolve(persona_uuid=profile.uuid, variant="C", treatment=SocialMode.INDIVIDUAL,
                                     response=mapping.response_tendency, speed=mapping.speed_tendency)
        provenance = {
            "persona": {"source_kind": "nemotron", "source_ref": metadata["dataset"],
                        "record_id": profile.uuid, "artifact_hash": sha256_value(records[slot]),
                        "method": metadata.get("sampling", "snapshot without replacement"), "version": metadata["revision"]},
            "mapping": {"source_kind": "llm", "source_ref": provider.model, "record_id": profile.uuid,
                        "input_fields": list(MAPPING_FIELDS), "input_hash": sha256_value(mapping_input.model_dump(mode="json")),
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
