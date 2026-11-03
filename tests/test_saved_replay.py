"""현재 계산과 과거 저장 결과의 재생을 분리하고 파일 보존을 검사한다."""
from copy import deepcopy
import hashlib
import json

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def saved_replay(tmp_path, monkeypatch):
    from persona_evac.visualization import app as module
    from persona_evac.llm.ollama import LocalOllama

    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'OUTPUT', tmp_path/'output')
    monkeypatch.setattr(module, '_station_signature', lambda: 'current-code')
    monkeypatch.setattr(module, 'LOADED_SIGNATURE', 'loaded-before-change')
    monkeypatch.setattr(module, 'policy_matches', lambda _: False)
    module._JSON_FILES.clear()

    def forbidden(*args, **kwargs):
        pytest.fail('저장 결과 조회 중 계산·원본 갱신·모델 연결 금지')

    monkeypatch.setattr(module, 'simulate_station', forbidden)
    monkeypatch.setattr(module, '_station_source_path', forbidden)
    monkeypatch.setattr(LocalOllama, '__init__', forbidden)

    def write(relative, payload):
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding='utf-8')
        return path

    config = write('configs/scenarios/gaehwasan.json', {'id': 'current-map'})
    result = {
        'implementation_sha256': 'stored-code', 'input_config_sha256': 'stored-map',
        'scenario': {'id': 'stored-map', 'floors': [{'id': 'B2'}]},
        'summary': {'count': 1, 'completed_count': 0, 'total_time_s': 10},
        'agents': [{'agent_id': 1, 'segments': [
            {'start': 0, 'end': 10, 'status': 'walking', 'from': {'x': 0}, 'to': {'x': 1}}
        ]}],
        'fire': {'enabled': True, 'metadata': {'engine': 'stored-FDS'}},
        'decisions': [{'agent_id': 1, 'time_s': 1, 'priority': 'experience'}],
    }
    artifact = write('output/gaehwasan/results.json', result)

    def activate():
        return write('output/gaehwasan/private_verification/activation.json', {
            'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
            'implementation_sha256': result['implementation_sha256'],
        })

    activation = activate()
    return module, artifact, activation, result, config, activate


def test_verified_history_replays_exact_snapshot_without_mutating_cache_or_files(saved_replay):
    module, artifact, activation, result, _, _ = saved_replay
    before = {path: path.read_bytes() for path in (artifact, activation)}
    with TestClient(module.app) as client:
        assert client.get('/api/gaehwasan').status_code == 409
        response = client.get('/api/gaehwasan/replay')
        assert response.status_code == 200
        payload = response.json()
        assert {key: value for key, value in payload.items() if key != 'artifact_status'} == result
        assert payload['artifact_status'] == {
            'mode': 'stored_replay', 'snapshot_verified': True,
            'artifact_sha256': hashlib.sha256(before[artifact]).hexdigest(),
            'implementation_matches': False, 'config_matches': False,
            'policy_matches': False, 'automatic_recomputation': False,
        }
        # 저장 결과를 현재 코드로 계산한 결과로 취급하지 않는다.
        assert client.post('/api/gaehwasan', json={}).status_code == 409
    assert {path: path.read_bytes() for path in before} == before
    assert 'artifact_status' not in module._read_json(artifact)


@pytest.mark.parametrize('change', ['missing', 'invalid_json', 'digest', 'code'])
def test_missing_or_changed_activation_never_bypasses_verification(saved_replay, change):
    module, artifact, activation, _, _, _ = saved_replay
    before = artifact.read_bytes()
    if change == 'missing':
        activation.unlink()
    elif change == 'invalid_json':
        activation.write_text('{', encoding='utf-8')
    else:
        record = json.loads(activation.read_bytes())
        record['artifact_sha256' if change == 'digest' else 'implementation_sha256'] = 'changed'
        activation.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(HTTPException) as error:
        module.gaehwasan_saved_replay()
    assert error.value.status_code == 409
    assert error.value.detail['code'] == 'station_replay_unverified'
    assert artifact.read_bytes() == before


def test_changed_result_bytes_are_rejected_even_when_json_meaning_is_unchanged(saved_replay):
    module, artifact, _, _, _, _ = saved_replay
    module.gaehwasan_saved_replay()
    artifact.write_bytes(artifact.read_bytes()+b'\n')
    before = artifact.read_bytes()
    with pytest.raises(HTTPException) as error:
        module.gaehwasan_saved_replay()
    assert error.value.status_code == 409
    assert artifact.read_bytes() == before


def test_incomplete_snapshot_is_not_replaced_with_current_setup(saved_replay):
    module, artifact, _, result, _, activate = saved_replay
    incomplete = deepcopy(result)
    incomplete['scenario'] = {}
    artifact.write_text(json.dumps(incomplete), encoding='utf-8')
    activate()
    with pytest.raises(HTTPException) as error:
        module.gaehwasan_saved_replay()
    assert error.value.status_code == 422
    assert error.value.detail['code'] == 'station_replay_incomplete'


def test_missing_saved_result_remains_missing_without_computation(saved_replay):
    module, artifact, _, _, _, _ = saved_replay
    artifact.unlink()
    with pytest.raises(HTTPException) as error:
        module.gaehwasan_saved_replay()
    assert error.value.status_code == 404


def test_concurrent_result_update_returns_retry_instead_of_mixed_snapshot(saved_replay, monkeypatch):
    module, artifact, activation, _, _, _ = saved_replay
    original = module._read_json

    def changing(path):
        value = original(path)
        if path == activation:
            artifact.write_bytes(artifact.read_bytes()+b'\n')
        return value

    monkeypatch.setattr(module, '_read_json', changing)
    with pytest.raises(HTTPException) as error:
        module.gaehwasan_saved_replay()
    assert error.value.status_code == 503
