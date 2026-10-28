"""Prepare profiles and comparisons without replacing active simulation results."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from persona_evac.persona.experimental_profile import ExperimentalProfiles, TRAITS, LEVELS
from persona_evac.llm.ollama import atomic_json


def summary(profiles):
    values = list(profiles.values())
    def counts(items, categories):
        counted = Counter(items)
        total = sum(counted.values())
        return {'counts': {k: counted[k] for k in categories},
                'ratios': {k: counted[k]/total if total else 0 for k in categories}}
    evidence = lambda d: d.get('evidence_level', 'legacy_literal_interpretation'
                               if d['origin'] == 'qwen_interpretation' else 'seeded_assignment')
    categories = ['direct', 'contextual', 'hypothesis', 'legacy_literal_interpretation', 'seeded_assignment']
    return {'count': len(values),
            'levels': {t: counts((p['traits'][t]['level'] for p in values), LEVELS) for t in TRAITS},
            'evidence_levels': counts((evidence(d) for p in values for d in p['traits'].values()), categories),
            'evidence_by_trait': {t: counts((evidence(p['traits'][t]) for p in values), categories) for t in TRAITS}}


def main():
    parser = argparse.ArgumentParser(description='활성 대피 결과를 변경하지 않고 성향을 별도로 준비합니다.')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--method', choices=['keyword_candidates', 'full_persona'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path(os.environ.get('PERSONA_EVAC_OUTPUT', 'output'))/'gaehwasan')
    parser.add_argument('--compare', action='store_true', help='기존 방식 캐시와 동일 인물을 비교합니다.')
    parser.add_argument('--repeat-count', type=int, default=0, help='캐시를 분리해 재추론할 표본 수')
    parser.add_argument('--repeat-runs', type=int, default=2, help='최초 추정 이후 추가 추론 횟수')
    args = parser.parse_args()
    if args.count < 1 or args.repeat_count < 0 or args.repeat_runs < 1:
        parser.error('인원·추가 추론 횟수는 양수이며 반복 표본 수는 0 이상이어야 합니다.')
    root = args.output_root
    source_path = args.source or root/'persona_source.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    agents = source['variants']['C'][:args.count]
    if len(agents) != args.count:
        parser.error('요청 인원보다 원본 인원 수가 적습니다.')
    provider = ExperimentalProfiles(root/'private_profile_cache', method=args.method)
    destination = root/'profile_preparations'/provider.method/provider.version
    report_path = (root/'private_profile_preparation.json' if provider.method == 'keyword_candidates'
                   else destination/'report.json')
    report = {'status': 'running', 'classification_method': provider.method, 'version': provider.version,
              'requested_count': args.count, 'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest()}
    try:
        profiles = provider.prepare(agents)
        report.update(count=len(profiles), wall_s=provider.wall_s, records=provider.records,
            origins=dict(Counter(t['origin'] for p in profiles.values() for t in p['traits'].values())),
            levels={t:dict(Counter(p['traits'][t]['level'] for p in profiles.values())) for t in TRAITS},
            summary=summary(profiles))
        atomic_json(destination/'profiles.json', {'classification_method': provider.method, 'version': provider.version,
            'source_sha256': report['source_sha256'],
            'profiles': [{'agent_id': aid, 'profile': p} for aid, p in profiles.items()]})
        if args.compare:
            legacy = ExperimentalProfiles(root/'private_profile_cache', method='keyword_candidates', seed=provider.seed)
            baseline = legacy.prepare(agents, cache_only=True)
            if any(record['source'] != 'profile_cache' for record in legacy.records):
                raise RuntimeError('기존 방식의 저장 결과가 없어 재사용 비교를 완료하지 못했습니다.')
            report['comparison'] = {'keyword_candidates': summary(baseline), provider.method: summary(profiles),
                'legacy_cache_hits': len(legacy.records),
                'changed_levels': {t: sum(profiles[a['agent_id']]['traits'][t]['level'] !=
                    baseline[a['agent_id']]['traits'][t]['level'] for a in agents) for t in TRAITS},
                'evidence_note': '기존 원문 해석과 새 직접 근거는 판정 기준이 달라 동일 범주로 합치지 않습니다.'}
        if args.repeat_count:
            if provider.method != 'full_persona':
                raise ValueError('반복 추론 검증은 full_persona에서 사용합니다.')
            sample = agents[:args.repeat_count]
            runs = [profiles]
            repeat_records = []
            for index in range(args.repeat_runs):
                repeated = ExperimentalProfiles(destination/'repeats'/str(index+1), provider.local,
                                                provider.seed, method='full_persona')
                runs.append(repeated.prepare(sample))
                repeat_records.append(repeated.records)
            report['repeatability'] = {'sample_count': len(sample), 'total_runs': len(runs),
                'settings': profiles[sample[0]['agent_id']]['inference_settings'], 'records': repeat_records,
                'levels': {str(a['agent_id']): {t: [r[a['agent_id']]['traits'][t]['level'] for r in runs]
                    for t in TRAITS} for a in sample},
                'changed_people_by_trait': {t: sum(len({r[a['agent_id']]['traits'][t]['level'] for r in runs}) > 1
                    for a in sample) for t in TRAITS}}
        report['status'] = 'completed'
    except Exception as exc:
        report.update(status='technical_failure', error_type=type(exc).__name__, error=str(exc),
                      wall_s=getattr(provider, 'wall_s', None), records=provider.records)
        atomic_json(report_path, report)
        raise
    atomic_json(report_path, report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'records', 'repeatability'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
