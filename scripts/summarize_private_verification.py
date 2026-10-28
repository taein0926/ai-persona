"""Gather measured evidence without turning incomplete evacuation into success."""
from collections import Counter
import hashlib,json,platform,sys
from pathlib import Path
import xml.etree.ElementTree as ET
import jupedsim,numpy,shapely
from persona_evac.llm.ollama import atomic_json

root=Path('output/gaehwasan/private_verification')
def read(name):return json.loads((root/name).read_text(encoding='utf-8'))
audit=read('live.audit.json');replay=read('recompute_comparison.json');controlled=read('controlled_choices.json')
status=read('status.json');activation=read('activation.json');protected=read('protected_inputs.json')
for name,expected in protected.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
assert hashlib.sha256(Path('output/gaehwasan/results.json').read_bytes()).hexdigest()==activation['artifact_sha256']
tests=ET.parse(root/'final_tests.xml').getroot().find('testsuite')
assert int(tests.attrib['failures'])==0 and int(tests.attrib['errors'])==0
ui=json.loads(Path('output/qa/private_editor_verification.json').read_text(encoding='utf-8'))
views=json.loads(Path('output/qa/simulation_views.json').read_text(encoding='utf-8'))
assert not ui['page_errors'] and not views['page_errors'] and replay['status']=='통과'
assert status['code_frozen_during_run'] and status['protected_inputs_unchanged']
preparation=json.loads(Path('output/gaehwasan/private_profile_preparation.json').read_text(encoding='utf-8'))
trait_pairs=controlled['pairs'][:4]
report={'status':'구현·실제 검증·정리·웹 반영 완료, 모형의 미완료·한계를 그대로 기록',
    'date':'2026-10-27','policy_versions':status['policy_versions'],'policy_hash':status['policy_hash'],
    'preserved_input_hashes':protected,'all_protected_inputs_unchanged':True,
    'original_count':100,'original_personas_uuid_and_empirical_values_preserved':True,
    'profile_preparation':{'actual_calls':sum(m.get('source')=='ollama_live' for m in preparation['records']),
        'wall_s':preparation['wall_s'],'origins':audit['profile_origins'],'levels':audit['trait_levels']},
    'initial_familiarity':audit['familiarity'],'original_population_changed_for_familiarity':False,
    'live_run':{'summary':audit['summary'],'decisions':audit['decision_count'],'performance':audit['performance'],
        'overall_wall_s':status['wall_s'],'fallback_count':audit['fallback_count'],'noncompletion_causes':audit['noncompletion_causes'],
        'critical_calls_after_routine_limit':audit['critical_calls_after_routine_limit'],
        'source_note':'보완 중 중단한 계산에서 같은 입력의 응답 184건을 캐시로 사용. 완전한 무캐시 첫 실행 측정이 아님.'},
    'collision_and_stairs':{k:audit[k] for k in ('body_overlap_pair_steps','min_surface_gap_m','minimum_same_lane_gap_m','stair_following_steps','stair_used_lanes')},
    'overtaking':{k:audit[k] for k in ('overtake_intents','overtake_plans','completed_overtakes')},
    'controlled_synthetic_tests':{'raw_model_choice_changed_trait_pairs':sum(p['model_choice_changed'] for p in trait_pairs),
        'trait_pairs':4,'executed_goal_changed_trait_pairs':sum(p['executed_goal_changed'] for p in trait_pairs),
        'all_pairs':controlled['pairs'],'actual_calls':len(controlled['attempts']),
        'cache_cases':sum(c['inference']['source']=='ollama_cache' for c in controlled['cases']),
        'settings':controlled['inference_settings'],'population':'원본 100명과 분리한 합성 통제 자료'},
    'runtime_priorities':dict(Counter(k['priority'] for k in [])),
    'recomputed_after_cleanup':replay,'tests':dict(tests.attrib),'ui':ui,'views':views,
    'fds':read('fds_preservation.json'),'active_artifact':activation,
    'runtime':{'python':sys.version,'platform':platform.platform(),'jupedsim':jupedsim.__version__,'numpy':numpy.__version__,'shapely':shapely.__version__},
    'cleanup_manifest':'output/gaehwasan/private_verification/cleanup_manifest.json',
    'limitations':[
        '실험 가중치는 사람에게서 검증한 성격값이 아님. 원문 인용 검증과 해석의 타당성은 구분함.',
        '성향 4개 통제 쌍 중 Qwen 원래 선택 변화는 위험 회피 1쌍. 나머지 목표 차이의 주된 원인은 실행 비용 코드임.',
        '원본 100명 모두 초행이므로 실제 명단에서 친숙도 효과의 집단 비교는 불가. 기능은 별도 자료로 확인함.',
        '전방향 18m·2m 지도·정확한 좌표 인식·연기 단계의 단순화. 촉각·벽을 짚는 탐색과 안내 표지 지식은 구현되지 않음.',
        '9번과 13번은 짙은 연기에서 같은 근거리 격자 목표 주변에 정체. 미관측 우회로나 완화한 충돌 조건을 주입하지 않음.',
        'walked_m는 누적 이동거리여서 접촉 중 미세 움직임을 포함할 수 있음. 진전은 stalled_s 및 위치 기록과 함께 해석해야 함.',
        '계단 내부 역방향 이동·추월·개별 속도 제어, 독성·열 상해·사망, 불꽃의 시각적 행동 인식은 지원하지 않음.',
        '100명 성향별 통계에는 위치·관측·경험 차이가 섞여 있어 인과 효과나 인간 행동 재현성을 단정할 수 없음.'
    ]}
report['runtime_priorities']={}
for row in audit['decision_histogram']:report['runtime_priorities'][row['priority']]=report['runtime_priorities'].get(row['priority'],0)+row['count']
atomic_json(root/'verification_report.json',report)
print(json.dumps({'status':report['status'],'tests':report['tests']['tests'],'summary':audit['summary'],
    'protected_inputs_unchanged':True,'controlled_model_changes':f"{sum(p['model_choice_changed'] for p in trait_pairs)}/4",'ui_errors':ui['page_errors']},ensure_ascii=False))
