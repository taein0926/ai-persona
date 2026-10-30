"""Read-only audit of staged classification results; never runs evacuation or inference."""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

from persona_evac.agent_policy import POLICY_HASH, VERSIONS
from persona_evac.llm.ollama import atomic_json
from persona_evac.persona.experimental_profile import ExperimentalProfiles, FULL_TRAITS as TRAITS, validate_profile
from persona_evac.persona.full_profile import CONFIG, POLICY_HASH_FULL, ProfileFormatError, persona_fields


def main():
    parser = argparse.ArgumentParser(description='별도 준비 결과·기존 캐시·활성 결과의 보존을 검사합니다.')
    parser.add_argument('--root', type=Path, default=Path('output/gaehwasan'))
    args = parser.parse_args()
    root = args.root
    destination = root/'profile_preparations/full_persona'/CONFIG['version']
    preparation = json.loads((destination/'report.json').read_text(encoding='utf-8'))
    assert preparation['status'] == 'completed'
    source = json.loads((root/'persona_source.json').read_text(encoding='utf-8'))
    agents = source['variants']['C'][:preparation['count']]
    saved = json.loads((destination/'profiles.json').read_text(encoding='utf-8'))
    profiles = {p['agent_id']:p['profile'] for p in saved['profiles']}
    assert len(profiles) == len(agents) == preparation['count']
    assert saved['source_sha256'] == hashlib.sha256((root/'persona_source.json').read_bytes()).hexdigest()
    manifest = json.loads((root/'profile_baseline_manifest.json').read_text(encoding='utf-8'))
    changed = [name for name,digest in manifest.items() if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
    assert not changed, changed
    legacy = ExperimentalProfiles(root/'private_profile_cache', method='keyword_candidates')
    baseline = legacy.prepare(agents, cache_only=True)
    assert all(record['source'] == 'profile_cache' for record in legacy.records)
    active = json.loads((root/'results.json').read_text(encoding='utf-8'))
    # 과거 응답은 당시 캐시 키와 원본 응답으로 확인한다. 새 입력·프롬프트로
    # 과거 판단을 다시 실행하거나 이전 응답을 새 판단의 캐시로 사용하지 않는다.
    recorded_cache_checks = 0
    for decision in active['decisions']:
        paths = list(root.rglob(f'{decision["cache_key"]}.json'))
        expected = {str(decision['agent_id']): decision['raw_model_choice']}
        assert any(json.loads(path.read_text(encoding='utf-8')) == expected for path in paths)
        recorded_cache_checks += 1
    active_profiles = {a['persona']['uuid']:a['experimental_profile'] for a in active['agents']}
    for agent in agents:
        p = profiles[agent['agent_id']]
        validate_profile(p, agent['persona'], legacy.seed)
        assert p['classification_method'] == 'full_persona' and p['policy_hash'] == POLICY_HASH_FULL
        assert p['version'] != VERSIONS['profile']
        assert p['input_fields'] == list(persona_fields(agent['persona'])) and not p['input_truncated']
        assert all(t['origin'] == 'llm_full_persona' and t['distribution'] is None for t in p['traits'].values())
        if active['agent_policy'].get('classification_method', 'keyword_candidates') == 'keyword_candidates':
            assert baseline[agent['agent_id']] == active_profiles[agent['persona']['uuid']]
    model = next(iter(profiles.values()))['model']
    cache_provider = ExperimentalProfiles(root/'private_profile_cache', local=SimpleNamespace(
        model=model['name'], digest=model['digest'], settings={}), seed=legacy.seed, method='full_persona')
    with redirect_stdout(io.StringIO()):
        reused = cache_provider.prepare(list(reversed(agents)), cache_only=True)
    assert reused == profiles
    assert all(record['source'] == 'profile_cache' for record in cache_provider.records)
    fabricated = deepcopy(profiles[agents[0]['agent_id']])
    first = next(iter(fabricated['traits'].values()))
    first['quotes'] = [{'field': 'persona', 'text': '원문에 존재하지 않는 검증용 인용 7f0142'}]
    rejected = False
    try:
        validate_profile(fabricated, agents[0]['persona'], legacy.seed)
    except ProfileFormatError:
        rejected = True
    assert rejected
    sparse = json.loads(Path('output/qa/full_profile_sparse/result.json').read_text(encoding='utf-8'))
    sparse_profile = sparse['profiles']['1']
    validate_profile(sparse_profile, sparse['persona'], legacy.seed)
    assert all(t['assumption'].strip() and not t['evidence_ids'] and 'evidence_level' not in t
               for t in sparse_profile['traits'].values())
    repeat = preparation['repeatability']
    tests=ET.parse('output/qa/full_profile_tests.xml').getroot().find('testsuite').attrib
    assert int(tests['failures'])==0 and int(tests['errors'])==0
    pairs = []
    for left,right in ((0,1),(0,2),(1,2)):
        pairs.append({'runs': [left+1,right+1], 'changed_by_trait': {
            t:sum(values[t][left] != values[t][right] for values in repeat['levels'].values()) for t in TRAITS}})
    report = {'status': 'passed', 'verified_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'count': len(profiles), 'classification_method': 'full_persona', 'version': CONFIG['version'],
        'policy_hash': POLICY_HASH_FULL, 'legacy_policy_hash': POLICY_HASH, 'model': model,
        'tests': tests,
        'protected_files_checked': len(manifest), 'protected_files_changed': changed,
        'source_and_active_result_sha256_unchanged': True,
        'legacy_profiles_match_active_result': all(baseline[a['agent_id']] == active_profiles[a['persona']['uuid']] for a in agents),
        'legacy_cache_hits': len(legacy.records),
        'recorded_decision_cache_payloads_verified': recorded_cache_checks,
        'full_profile_cache_hits_with_reversed_order': len(cache_provider.records),
        'all_original_fields_included': True, 'truncated_inputs': 0,
        'full_profile_seeded_assignments': 0, 'fabricated_citation_rejected': rejected,
        'real_sparse_information_sample': {t: {k:v[k] for k in ('level','assumption')}
                                           for t,v in sparse_profile['traits'].items()},
        'comparison': preparation['comparison'], 'repeatability_pairs': pairs,
        'actual_successful_repeat_llm_calls': sum(record.get('source')=='ollama_live'
            for run in repeat['records'] for record in run),
        'repeatability': {k:v for k,v in repeat.items() if k != 'records'},
        'full_evacuation_simulation_reexecuted': False,
        'changed_source_files': [
            'configs/agents/private_agent_policy.json', 'src/persona_evac/agent_policy.py',
            'src/persona_evac/persona/experimental_profile.py', 'src/persona_evac/persona/full_profile.py',
            'src/persona_evac/integration/station.py', 'src/persona_evac/integration/private_agents.py',
            'src/persona_evac/integration/station_dynamics.py', 'src/persona_evac/cli.py',
            'src/persona_evac/station_artifacts.py', 'src/persona_evac/visualization/app.py',
            'src/persona_evac/llm/private_decisions.py', 'scripts/prepare_private_profiles.py',
            'scripts/audit_private_result.py', 'scripts/audit_profile_preparation.py',
            'tests/test_full_profile.py', 'private_inspector.js', 'gaehwasan.html', 'README.md'],
        'limitations': ['등급과 근거 요약의 의미 타당성은 LLM 판단이며 사람의 행동으로 검증하지 않았습니다.',
            '반복 검증은 동일 모델·다이제스트·설정과 5명 표본의 3회 추론입니다. 다른 모델·설정·장치에서의 안정성을 보장하지 않습니다.',
            'UTF-8 바이트 기반 문맥 예산은 보수적이므로 실제 토큰 수가 더 적어도 긴 입력을 명시적으로 거부할 수 있습니다.',
            '분류 출력은 등급과 근거 요약으로 제한하며 요약의 사실성을 자동 판정하지 않습니다.',
            '준비 결과를 활성 대피 결과에 적용하지 않았으므로 새 분류의 이동·완료율 효과는 계산하지 않았습니다.']}
    atomic_json(destination/'verification.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'comparison','repeatability','limitations'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
