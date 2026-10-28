from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from persona_evac.agent_policy import POLICY_HASH, VERSIONS
from persona_evac.persona.experimental_profile import ExperimentalProfiles, candidates, validate_profile
from persona_evac.persona.full_profile import (
    CONFIG, POLICY_HASH_FULL, ProfileContextError, ProfileFormatError,
    build_full_profile, persona_fields, validate_estimates,
    expand_inference,
)


def estimates():
    return {t: {'level': 'low', 'evidence_level': 'hypothesis', 'source_fields': [],
                'quotes': [], 'assumption': '관련 정보가 부족하여 낮은 수준을 가정합니다.',
                'information_insufficient': True}
            for t in ('crowd_avoidance', 'waiting_patience', 'risk_avoidance', 'exploration')}


class FakeLocal:
    def __init__(self, payload=None, error=None, digest='digest'):
        self.model = 'test-model'
        self.digest = digest
        self.settings = {'num_ctx': 4096}
        self.payload = payload if payload is not None else estimates()
        self.error = error
        self.inputs = []
        self._full_profile_context_limit = 262144

    def generate(self, system, body, schema, max_tokens, **kwargs):
        self.inputs.append(deepcopy(body))
        assert self.settings['num_ctx'] == CONFIG['num_ctx']
        if self.error:
            raise self.error
        payload = deepcopy(self.payload)
        for value in payload.values():
            value['quote_fields'] = [q['field'] for q in value.pop('quotes')]
        return payload, {'source': 'ollama_live', 'prompt_eval_count': 100, 'eval_count': 200}


def test_original_policy_hash_and_existing_seeded_cache_remain_compatible(tmp_path, monkeypatch):
    monkeypatch.delenv('PERSONA_EVAC_PROFILE_METHOD', raising=False)
    assert POLICY_HASH == '9c4c60fcf8ad41c3492c007be32056a028ada3f9ceda592485f4b421ae75712b'
    agents = [{'agent_id': 1, 'persona': {'uuid': 'legacy', 'sports_persona': '수영을 즐깁니다.'}}]
    first = ExperimentalProfiles(tmp_path, seed=7)
    original = first.prepare(agents)
    path = next(tmp_path.rglob('*.json'))
    before = path.read_bytes()
    second = ExperimentalProfiles(tmp_path, seed=7)
    assert second.prepare(agents, cache_only=True) == original
    assert path.read_bytes() == before
    assert second.method == 'keyword_candidates'
    assert second.version == VERSIONS['profile']
    assert second.records == [{'source': 'profile_cache', 'uuid': 'legacy'}]
    with pytest.raises(ValueError, match='캐시가 없습니다'):
        second.prepare([{'agent_id': 2, 'persona': {'uuid': 'missing'}}], cache_only=True)
    assert len(list(tmp_path.rglob('*.json'))) == 1


def test_complete_fields_without_keywords_and_more_than_two_sentences_reach_llm(tmp_path):
    p = {'uuid': 'complete', 'sports_persona': '수영을 즐깁니다.',
         'arts_persona': '피아노를 연주합니다.', 'culinary_persona': '빵을 굽습니다.',
         'professional_persona': '악기를 조율합니다.', 'career_goals_and_ambitions': '연주자가 되고 싶습니다.',
         'skills_and_expertise_list': ['음감', '독보'], 'age': 25,
         'persona': '매일 기록합니다. 악보를 읽습니다. 라디오를 듣습니다. 연습을 마칩니다.'}
    assert candidates(p)[0] == []
    before = deepcopy(p)
    local = FakeLocal()
    provider = ExperimentalProfiles(tmp_path, local, method='full_persona')
    profile = provider.prepare([{'agent_id': 1, 'persona': p}])[1]
    assert p == before
    assert local.inputs == [{'fields': persona_fields(p)}]
    assert profile['input_fields'] == list(persona_fields(p))
    assert profile['input_truncated'] is False
    assert local.settings['num_ctx'] == 4096
    assert validate_profile(profile, p, provider.seed) == profile
    reordered = dict(reversed(list(p.items())))
    assert provider.prepare([{'agent_id':1,'persona':reordered}], cache_only=True)[1] == profile
    assert len(local.inputs) == 1
    assert len(profile['traits']) == 4
    for trait in profile['traits'].values():
        assert trait['level'] == 'low' and trait['evidence_level'] == 'hypothesis'
        assert trait['information_insufficient'] and trait['distribution'] is None
        assert trait['quotes'] == [] and trait['evidence_ids'] == []


def test_direct_contextual_hypothesis_are_separate_and_citations_are_literal():
    p = {'uuid': 'quote', 'travel_persona': '행렬에 서서 두 시간을 기다렸습니다.',
         'arts_persona': '즉흥 연주를 즐깁니다.'}
    payload = estimates()
    payload['waiting_patience'].update(level='high', evidence_level='direct',
        source_fields=['travel_persona'], information_insufficient=False,
        quotes=[{'field': 'travel_persona', 'text': '두 시간을 기다렸습니다.'}])
    payload['exploration'].update(evidence_level='contextual', source_fields=['arts_persona'], information_insufficient=False)
    result = build_full_profile(p, payload, 1, {'name': 'test', 'digest': 'digest'})
    assert validate_profile(result, p, 1) == result
    assert result['traits']['waiting_patience']['evidence_ids'] == ['E1']
    assert result['traits']['exploration']['evidence_ids'] == []
    bad = deepcopy(payload)
    bad['waiting_patience']['quotes'][0]['text'] = '혼잡을 싫어합니다.'
    with pytest.raises(ProfileFormatError, match='인용'):
        validate_estimates(bad, p)
    bad = deepcopy(payload)
    bad['waiting_patience']['quotes'][0]['field'] = 'arts_persona'
    with pytest.raises(ProfileFormatError):
        validate_estimates(bad, p)
    bad = deepcopy(result)
    bad['excerpts'][0]['text'] = '가짜 인용'
    with pytest.raises(ProfileFormatError):
        validate_profile(bad, p, 1)
    with pytest.raises(ProfileFormatError):
        validate_profile(result, {**p, 'age': 1}, 1)


def test_inference_selects_unrestricted_source_fields_and_never_writes_quote_text():
    p = {'uuid': 'refs', 'arts_persona': '즉흥 연주를 즐깁니다.',
         'travel_persona': '낮선 마을도 매번 방문합니다.', 'culinary_persona': '향신료를 직접 조합합니다.'}
    payload = {t: {**v, 'quote_fields': []} for t, v in estimates().items()}
    for value in payload.values():
        value.pop('quotes')
    payload['exploration'].update(evidence_level='direct', information_insufficient=False,
        source_fields=list(persona_fields(p)), quote_fields=list(persona_fields(p)))
    expanded = expand_inference(payload, p)
    assert len(expanded['exploration']['quotes']) == 3
    assert expanded['exploration']['quotes'] == [{'field': k, 'text': v} for k,v in persona_fields(p).items()]
    payload['exploration']['quote_fields'] = ['가짜필드']
    with pytest.raises(ProfileFormatError, match='인용할 수 없는'):
        expand_inference(payload, p)
    payload['exploration']['quote_fields'] = []
    with pytest.raises(ProfileFormatError, match='실제 인용'):
        expand_inference(payload, p)


@pytest.mark.parametrize('change', [
    {'evidence_level': 'direct'}, {'evidence_level': 'contextual', 'information_insufficient': False},
    {'source_fields': ['nonexistent']}, {'information_insufficient': False},
    {'level': 'unknown'}, {'assumption': ' '}, {'extra': True},
])
def test_invalid_estimate_is_rejected(change):
    payload = estimates()
    payload['crowd_avoidance'].update(change)
    with pytest.raises(ProfileFormatError):
        validate_estimates(payload, {'uuid': 'invalid'})


def test_cache_separates_methods_models_sources_and_experiment_seeds(tmp_path):
    agents = [{'agent_id': 1, 'persona': {'uuid': 'separate'}}]
    legacy = ExperimentalProfiles(tmp_path, seed=3, method='keyword_candidates').prepare(agents)
    local = FakeLocal()
    full = ExperimentalProfiles(tmp_path, local, seed=3, method='full_persona')
    prepared = full.prepare(agents)
    assert full.prepare(agents) == prepared and len(local.inputs) == 1
    assert full.records[-1]['source'] == 'profile_cache'
    assert legacy[1]['version'] != prepared[1]['version']
    assert prepared[1]['policy_hash'] == POLICY_HASH_FULL != POLICY_HASH
    assert len(list(tmp_path.rglob('*.json'))) == 2
    ExperimentalProfiles(tmp_path, FakeLocal(digest='new'), seed=3, method='full_persona').prepare(agents)
    ExperimentalProfiles(tmp_path, local, seed=4, method='full_persona').prepare(agents)
    changed = [{'agent_id': 1, 'persona': {'uuid': 'separate', 'arts_persona': '노래합니다.'}}]
    ExperimentalProfiles(tmp_path, local, seed=3, method='full_persona').prepare(changed)
    assert len(list(tmp_path.rglob('*.json'))) == 5


@pytest.mark.parametrize('failure,kind', [
    (httpx.ConnectError('연결 실패'), 'communication_error'),
    (ValueError('Ollama 입력이 문맥 길이 제한에 도달했습니다.'), 'context_overflow'),
    (ValueError('출력 형식 오류'), 'invalid_output'),
])
def test_technical_failure_does_not_create_seeded_or_normal_cache(tmp_path, failure, kind):
    local = FakeLocal(error=failure)
    provider = ExperimentalProfiles(tmp_path, local, method='full_persona')
    with pytest.raises(Exception):
        provider.prepare([{'agent_id': 1, 'persona': {'uuid': 'failure'}}])
    assert not list(tmp_path.rglob('*.json'))
    assert provider.records[-1]['failure_kind'] == kind
    assert local.settings['num_ctx'] == 4096


def test_context_overflow_fails_before_generation_without_truncation(tmp_path):
    local = FakeLocal()
    provider = ExperimentalProfiles(tmp_path, local, method='full_persona')
    with pytest.raises(ProfileContextError, match='원문을 자르지 않았습니다'):
        provider.prepare([{'agent_id': 1, 'persona': {'uuid': 'long', 'persona': '가'*30000}}])
    assert not local.inputs and not list(tmp_path.rglob('*.json'))


@pytest.mark.parametrize('metadata', [{}, {'model.context_length': 2048}])
def test_unknown_or_small_model_context_is_rejected_before_generation(tmp_path, metadata):
    local = FakeLocal()
    del local._full_profile_context_limit
    local.client = httpx.Client(base_url='http://localhost', transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={'model_info': metadata})))
    provider = ExperimentalProfiles(tmp_path, local, method='full_persona')
    with pytest.raises(ProfileContextError):
        provider.prepare([{'agent_id': 1, 'persona': {'uuid': 'model-limit'}}])
    assert not local.inputs and not list(tmp_path.rglob('*.json'))
    assert local.settings['num_ctx'] == 4096


def test_invalid_citations_fail_after_bounded_repairs_without_saving(tmp_path):
    payload = estimates()
    payload['exploration'].update(evidence_level='direct', information_insufficient=False,
        source_fields=['persona'], quotes=[{'field': '없는필드', 'text': '없는 인용'}])
    local = FakeLocal(payload)
    provider = ExperimentalProfiles(tmp_path, local, method='full_persona')
    with pytest.raises(ProfileFormatError):
        provider.prepare([{'agent_id': 1, 'persona': {'uuid': 'badquote', 'persona': '노래합니다.'}}])
    assert len(local.inputs) == CONFIG['max_validation_attempts']
    assert all(body['fields'] == {'persona': '노래합니다.'} for body in local.inputs)
    assert not list(tmp_path.rglob('*.json'))


def test_settings_selection_and_unknown_methods(tmp_path, monkeypatch):
    monkeypatch.setenv('PERSONA_EVAC_PROFILE_METHOD', 'full_persona')
    assert ExperimentalProfiles(tmp_path).method == 'full_persona'
    assert ExperimentalProfiles(tmp_path, method='keyword_candidates').method == 'keyword_candidates'
    with pytest.raises(ValueError):
        ExperimentalProfiles(tmp_path, method='unknown')


def test_reading_active_result_never_recomputes_or_retags_provenance(tmp_path, monkeypatch):
    from persona_evac.visualization import app
    monkeypatch.setattr(app, 'OUTPUT', tmp_path)
    monkeypatch.setattr(app, 'gaehwasan_run', lambda *args: pytest.fail('조회 중 재계산'))
    path = tmp_path/'gaehwasan/results.json'
    path.parent.mkdir()
    original = {'agents': [], 'implementation_sha256': 'old', 'agent_policy': {'versions': {'profile': 'original'}}}
    path.write_text(json.dumps(original), encoding='utf-8')
    before = path.read_bytes()
    result = app.gaehwasan_results()
    assert path.read_bytes() == before
    assert result['agent_policy'] == original['agent_policy']
    assert result['artifact_status']['automatic_recomputation'] is False


def test_full_profile_projection_preserves_evidence_and_separates_decision_cache():
    from persona_evac.llm.private_decisions import project_private
    from test_private_agents import sensor_fixture
    _, _, _, observation, _ = sensor_fixture()
    original = project_private(observation)
    assert 'classification_method' not in original['profile']
    p = {'uuid': 'fullprojection'}
    observation['profile'] = build_full_profile(p, estimates(), 1, {'name': 'fake', 'digest': 'fake'})
    full = project_private(observation)
    assert full['profile']['classification_method'] == 'full_persona'
    assert all(v['evidence_level'] == 'hypothesis' and v['information_insufficient']
               for v in full['profile']['traits'].values())


def test_private_decision_prompts_cache_paths_and_metadata_are_separate(tmp_path):
    from persona_evac.llm.private_decisions import PrivateDecisions, PROMPT, FULL_PROFILE_PROMPT
    from test_private_agents import sensor_fixture, persona
    _, _, _, observation, _ = sensor_fixture()
    prompts = []
    def generate(prompt, *args, **kwargs):
        prompts.append(prompt)
        return {str(observation['agent_id']): {'priority':'environment', 'memory':[],
                 'observations':[], 'movement':'continue'}}, {'source':'ollama_live'}
    local = SimpleNamespace(model='fake', digest='digest', settings={'num_ctx':4096}, generate=generate)
    decisions = PrivateDecisions(tmp_path, local)
    _, legacy_meta = decisions.decide([observation])
    observation['profile'] = build_full_profile(persona(), estimates(), 1, {'name':'fake','digest':'digest'})
    first, full_meta = decisions.decide([observation])
    cached, cached_meta = decisions.decide([observation])
    assert prompts == [PROMPT, FULL_PROFILE_PROMPT]
    assert cached == first and cached_meta['source'] == 'ollama_cache'
    assert full_meta['profile_version'] == CONFIG['version']
    assert full_meta['classification_method'] == 'full_persona'
    assert full_meta['policy_versions']['profile'] == CONFIG['version']
    assert legacy_meta['cache_key'] != full_meta['cache_key']
    assert (tmp_path/VERSIONS['decision']/f'{legacy_meta["cache_key"]}.json').exists()
    assert (tmp_path/VERSIONS['decision']/CONFIG['version']/f'{full_meta["cache_key"]}.json').exists()
