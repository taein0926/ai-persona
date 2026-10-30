import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from persona_evac.hashing import sha256_value
from persona_evac.integration.station import StationRequest, initialize_station
from persona_evac.llm import MockProvider
from persona_evac.persona.station_population import prepare_population, refresh_population
from persona_evac.llm.baseline_profile import POLICY
from persona_evac.persona.sampling import fetch_nemotron_snapshot, load_personas
from station_fixtures import source, config


def test_count_uses_distinct_original_people_and_stable_prefix():
    original = source(100)
    for i, person in enumerate(original["variants"]["C"]):
        person["desired_speed_mps"] = 0.8 + i/100
    for count in (10, 30, 60, 100):
        result = initialize_station(original, config(), StationRequest(agent_count=count))
        agents = result["agents"]
        assert len(agents) == len({a["persona"]["uuid"] for a in agents}) == count
        assert [a["persona"]["uuid"] for a in agents] == [str(i) for i in range(count)]
        assert [a["desired_speed_mps"] for a in agents] == [0.8+i/100 for i in range(count)]
    with pytest.raises(ValueError, match="60명"):
        initialize_station(source(), config(), StationRequest(agent_count=60))
    original["variants"]["C"][99]["persona"]["uuid"] = "0"
    with pytest.raises(ValueError, match="중복"):
        initialize_station(original, config(), StationRequest(agent_count=10))


def test_fetch_pages_past_truncated_preview_and_is_reproducible(tmp_path, monkeypatch):
    calls = []
    def response(request, timeout):
        url = request.full_url
        calls.append(url)
        if "/api/datasets/" in url:
            data = {"sha": "test-revision"}
        else:
            query = parse_qs(urlparse(url).query)
            start = int(query["offset"][0])
            end = min(100, start+42, start+int(query["length"][0]))
            data = {"rows": [{"row": {"uuid": str(i), "age": 30, "persona": f"person-{i}"}}
                             for i in range(start, end)]}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr("persona_evac.persona.sampling.urlopen", response)
    a, b = tmp_path/"a.jsonl", tmp_path/"b.jsonl"
    fetch_nemotron_snapshot(a, 100)
    fetch_nemotron_snapshot(b, 100)
    assert a.read_bytes() == b.read_bytes()
    assert len({p.uuid for p in load_personas(a)}) == 100
    assert any("offset=84" in url for url in calls)
    meta = json.loads(a.with_suffix(".metadata.json").read_text())
    assert meta["candidate_count"] == 100 and meta["count"] == 100


def test_prepare_maps_each_person_preserves_raw_profile_and_provenance(tmp_path):
    snapshot = tmp_path/"people.jsonl"
    rows = [{"uuid": str(i), "age": 25+i, "persona": f"person-{i}", "culinary_persona": "original"} for i in range(12)]
    # Metadata hash is over validated model records, including default fields.
    snapshot.write_text("\n".join(json.dumps(r) for r in rows))
    profiles = load_personas(snapshot)
    metadata = {"count": 12, "records_hash": sha256_value([p.model_dump(mode="json") for p in profiles]),
                "selection_seed": 123, "dataset": "test-nemotron", "revision": "test-revision"}
    snapshot.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    class RecordingProvider(MockProvider):
        seen = []
        def map(self, profile, context):
            self.seen.append(profile.uuid)
            assert profile.model_dump()['culinary_persona'] == 'original'
            return super().map(profile, context)
    provider = RecordingProvider()
    path = prepare_population(snapshot, Path("data/fixtures/julich_features.csv"), tmp_path/"source.json", provider)
    agents = json.loads(path.read_text())["variants"]["C"]
    assert provider.seen == [str(i) for i in range(12)]
    for agent in agents:
        assert agent["persona"]["culinary_persona"] == "original"
        assert agent["provenance"]["persona"]["record_id"] == agent["persona"]["uuid"]
        assert set(agent["provenance"]["mapping"]["input_fields"]) == set(agent['persona']) - {'uuid'}
        assert agent['provenance']['mapping']['policy'] == POLICY
        assert agent["pre_movement_s"] == agent["empirical"]["response_time_s"]
        assert agent["desired_speed_mps"] == agent["empirical"]["desired_speed_mps"]


def test_refresh_reclassifies_only_requested_old_baselines_without_changing_originals():
    from copy import deepcopy
    class Provider(MockProvider):
        def __init__(self): self.seen=[]
        def map(self,profile,context):
            self.seen.append(profile.uuid)
            assert profile.model_dump()['culinary_persona']=='원문 전체 유지'
            return super().map(profile,context)
    original=source(12)
    original.update(backend='nemotron_qwen_empirical',provenance={'snapshot':{'selection_seed':123}})
    for agent in original['variants']['C']:
        agent['persona'].update(age=30,culinary_persona='원문 전체 유지')
        agent['provenance']={'empirical':{'source_ref':'data/fixtures/julich_features.csv'}}
    before=deepcopy(original);provider=Provider()
    updated=refresh_population(original,10,provider)
    assert original==before
    assert provider.seen==[str(i) for i in range(10)]
    for agent in updated['variants']['C'][:10]:
        assert agent['provenance']['mapping']['policy']==POLICY
        assert agent['pre_movement_s']==agent['empirical']['response_time_s']
        assert agent['desired_speed_mps']==agent['empirical']['desired_speed_mps']
    assert updated['variants']['C'][10:]==original['variants']['C'][10:]
    assert refresh_population(updated,10,provider) is updated
    assert len(provider.seen)==10
