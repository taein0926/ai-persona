"""성능 비교가 실행 메타데이터만 분리하고 판단 차이를 감지하는지 검사한다."""
from copy import deepcopy
import importlib.util
from pathlib import Path


spec=importlib.util.spec_from_file_location('optimization_comparison',
    Path(__file__).resolve().parents[1]/'scripts/compare_observation_optimization.py')
comparison=importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def decision_result():
    return {'decisions':[{'agent_id':1,'time_s':10.,'wall_s':1.,
        'model_choice':{'movement':'continue'},'observation':{'time_s':10.,'options':[]}}]}


def test_execution_metadata_does_not_change_judgment_comparison():
    before=decision_result();after=deepcopy(before)
    after['decisions'][0].update(wall_s=2.,prompt_batch_size=128,runtime_profile=None)
    assert comparison.semantic_hashes(before)==comparison.semantic_hashes(after)


def test_actual_choice_and_observation_time_changes_are_detected():
    before=decision_result();changed=deepcopy(before)
    changed['decisions'][0]['model_choice']['movement']='hurry'
    assert comparison.semantic_hashes(before)!=comparison.semantic_hashes(changed)
    changed=deepcopy(before)
    changed['decisions'][0]['observation']['time_s']=11.
    assert comparison.semantic_hashes(before)!=comparison.semantic_hashes(changed)


def test_a_nonempty_runtime_profile_is_not_silently_discarded():
    before=decision_result();changed=deepcopy(before)
    changed['decisions'][0]['runtime_profile']={'backend':'different'}
    assert comparison.semantic_hashes(before)!=comparison.semantic_hashes(changed)
